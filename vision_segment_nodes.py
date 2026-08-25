"""Vision segmentation nodes: garment parts and portrait regions.

Both nodes run a SegFormer label map (see segformer_backend.py), union the
class ids for the requested region, refine the mask, and return
(mask, cutout_rgba, mask_image) at the input resolution.
"""

import cv2
import numpy as np
import torch

from .segformer_backend import load_segformer, offload, predict_labels

try:
    from comfy.model_management import throw_exception_if_processing_interrupted
except Exception:  # outside ComfyUI (tests)
    def throw_exception_if_processing_interrupted():
        return None

DEVICES = ["cuda", "cpu"]

# ---------------------------------------------------------------------------
# Fashionpedia (47 classes) ids used by 1038lab/segformer_fashion
# ---------------------------------------------------------------------------
GARMENT_GROUPS = {
    # main garments + parts that only ever belong to an upper-body garment
    "upper_garment": {1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 17, 26, 28, 29, 30, 31, 32, 34},
    # pants, shorts, skirt, belt, buckle, leg warmer, tights/stockings (shoes excluded)
    "lower_garment": {7, 8, 9, 20, 21, 22, 35},
    # shoe, sock
    "footwear": {24, 23},
    # hat, headband / head covering / hair accessory
    "headwear": {15, 16},
}
# Decorations/parts that can sit on any garment. They are merged into the
# selected group only when their connected component touches that group.
GARMENT_ATTACHED_PARTS = {33, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46}
# Deliberately unassigned: 14 glasses, 18 glove, 19 watch, 25 bag/wallet, 27 umbrella.
GARMENT_PART_CHOICES = list(GARMENT_GROUPS.keys())

# ---------------------------------------------------------------------------
# CelebAMask-HQ (19 classes) ids used by jonathandinu/face-parsing
# ---------------------------------------------------------------------------
PORTRAIT_REGIONS = {
    # skin + nose + eyeglasses + eyes + brows + lips + mouth -> solid face region without holes
    "face": {1, 2, 3, 4, 5, 6, 7, 10, 11, 12},
    "neck": {17},
    "hair": {13},
    "eyes": {4, 5},
    "lips": {11, 12},
    "nose": {2},
    "eyebrows": {6, 7},
    "ears": {8, 9},
    "mouth_interior": {10},
}
PORTRAIT_DEFAULT_ON = {"face", "neck", "hair", "eyes", "lips", "nose", "eyebrows", "ears"}


# ---------------------------------------------------------------------------
# Mask helpers (pure numpy/cv2, unit-testable without a model)
# ---------------------------------------------------------------------------
def labels_to_mask(labels, class_ids):
    """HxW int label map + iterable of ids -> HxW uint8 {0,1}."""
    if not class_ids:
        return np.zeros(labels.shape, dtype=np.uint8)
    return np.isin(labels, list(class_ids)).astype(np.uint8)


def absorb_touching_parts(main_mask, parts_mask, reach=3):
    """Add connected components of `parts_mask` that touch `main_mask` (within `reach` px)."""
    if main_mask.max() == 0 or parts_mask.max() == 0:
        return main_mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
    reach_zone = cv2.dilate(main_mask, kernel) > 0
    _, components = cv2.connectedComponents(parts_mask.astype(np.uint8), connectivity=8)
    touching = np.unique(components[reach_zone])
    touching = touching[touching > 0]
    return (main_mask.astype(bool) | np.isin(components, touching)).astype(np.uint8)


def fill_mask_holes(mask):
    """Fill enclosed holes in a HxW uint8 {0,1} mask."""
    padded = np.pad(mask, 1, mode="constant", constant_values=0).astype(np.uint8)
    flood = padded.copy()
    flood_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = (flood == 0) & (padded == 0)
    filled = padded | holes.astype(np.uint8)
    return filled[1:-1, 1:-1]


def refine_mask(mask, expand=0, blur=0, holes=False):
    """uint8 {0,1} -> float32 [0,1] after optional hole fill, dilate/erode and blur."""
    work = mask.astype(np.uint8)
    if holes:
        work = fill_mask_holes(work)
    if expand != 0:
        size = 2 * abs(int(expand)) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        work = cv2.dilate(work, kernel) if expand > 0 else cv2.erode(work, kernel)
    result = work.astype(np.float32)
    if blur > 0:
        size = 2 * int(blur) + 1
        result = cv2.GaussianBlur(result, (size, size), 0)
    return np.clip(result, 0.0, 1.0)


def pack_outputs(image_rgb, mask):
    """HxWx3 float image + HxW float mask -> (MASK 1xHxW, RGBA 1xHxWx4, mask IMAGE 1xHxWx3)."""
    mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
    rgba = np.concatenate([image_rgb, mask[:, :, None]], axis=-1).astype(np.float32)
    mask_image = np.repeat(mask[:, :, None], 3, axis=-1).astype(np.float32)
    return mask_t, torch.from_numpy(rgba).unsqueeze(0), torch.from_numpy(mask_image).unsqueeze(0)


def _image_batch_to_numpy(image):
    """ComfyUI IMAGE (B,H,W,3|4) -> list of HxWx3 float32 arrays in [0,1]."""
    array = image.detach().cpu().numpy().astype(np.float32)
    if array.ndim == 3:
        array = array[None]
    channels = array.shape[-1]
    if channels == 1:
        array = np.repeat(array, 3, axis=-1)
    elif channels not in (3, 4):
        raise ValueError(f"[NH-Nodes] Expected IMAGE with 1, 3 or 4 channels, got {channels}")
    return [np.clip(frame[:, :, :3], 0.0, 1.0) for frame in array]


def _run_batch(image, model_name, device, build_mask, expand, blur, holes):
    """Shared per-image loop: predict labels, build mask via callback, refine, pack, stack."""
    bundle = load_segformer(model_name, device)
    masks, cutouts, mask_images = [], [], []
    frames = _image_batch_to_numpy(image)
    try:
        for frame in frames:
            throw_exception_if_processing_interrupted()
            labels = predict_labels(bundle, (frame * 255.0).round().astype(np.uint8))
            mask = refine_mask(build_mask(labels), expand, blur, holes)
            mask_t, rgba_t, mask_img_t = pack_outputs(frame, mask)
            masks.append(mask_t)
            cutouts.append(rgba_t)
            mask_images.append(mask_img_t)
    finally:
        offload(bundle)
    return torch.cat(masks, 0), torch.cat(cutouts, 0), torch.cat(mask_images, 0)


def _refine_inputs(fill_holes_default):
    return {
        "mask_expand": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1,
                                "tooltip": "Grow (+) or shrink (-) the mask by N pixels."}),
        "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1,
                              "tooltip": "Feather the mask edge by N pixels."}),
        "fill_holes": ("BOOLEAN", {"default": fill_holes_default,
                                   "tooltip": "Fill enclosed holes. Off for garments by default: an arm across the torso would be filled in."}),
        "device": (DEVICES, {"default": "cuda"}),
    }


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
class NH_GarmentSegment:
    """Segment upper/lower garment, footwear or headwear from a person image."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image": ("IMAGE",),
            "part": (GARMENT_PART_CHOICES, {"default": "upper_garment"}),
            "attach_parts": ("BOOLEAN", {"default": True,
                                         "tooltip": "Merge pockets, zippers, bows, ruffles... that touch the selected garment."}),
        }
        required.update(_refine_inputs(fill_holes_default=False))
        return {"required": required}

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("mask", "cutout_rgba", "mask_image")
    FUNCTION = "segment"
    CATEGORY = "NH-Nodes/Vision"
    SEARCH_ALIASES = ["garment segment", "clothes mask", "upper garment", "lower garment", "footwear", "headwear"]

    @staticmethod
    def build_mask(labels, part, attach_parts=True):
        mask = labels_to_mask(labels, GARMENT_GROUPS[part])
        if attach_parts:
            mask = absorb_touching_parts(mask, labels_to_mask(labels, GARMENT_ATTACHED_PARTS))
        return mask

    def segment(self, image, part, attach_parts, mask_expand, mask_blur, fill_holes, device):
        if part not in GARMENT_GROUPS:
            part = GARMENT_PART_CHOICES[0]
        return _run_batch(
            image, "segformer_fashion", device,
            lambda labels: self.build_mask(labels, part, attach_parts),
            mask_expand, mask_blur, fill_holes,
        )


class NH_PortraitSegment:
    """Segment selected face/hair/neck regions from a portrait."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {"image": ("IMAGE",)}
        for region in PORTRAIT_REGIONS:
            required[region] = ("BOOLEAN", {"default": region in PORTRAIT_DEFAULT_ON})
        required.update(_refine_inputs(fill_holes_default=True))
        return {"required": required}

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("mask", "cutout_rgba", "mask_image")
    FUNCTION = "segment"
    CATEGORY = "NH-Nodes/Vision"
    SEARCH_ALIASES = ["portrait segment", "face parsing", "face mask", "hair mask", "skin mask"]

    @staticmethod
    def build_mask(labels, selected_regions):
        class_ids = set()
        for region in selected_regions:
            class_ids |= PORTRAIT_REGIONS.get(region, set())
        return labels_to_mask(labels, class_ids)

    def segment(self, image, mask_expand, mask_blur, fill_holes, device, **regions):
        selected = [name for name, enabled in regions.items() if enabled and name in PORTRAIT_REGIONS]
        return _run_batch(
            image, "face_parsing", device,
            lambda labels: self.build_mask(labels, selected),
            mask_expand, mask_blur, fill_holes,
        )


NODE_CLASS_MAPPINGS = {
    "NH_GarmentSegment": NH_GarmentSegment,
    "NH_PortraitSegment": NH_PortraitSegment,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_GarmentSegment": "Garment Segment (NH)",
    "NH_PortraitSegment": "Portrait Segment (NH)",
}
