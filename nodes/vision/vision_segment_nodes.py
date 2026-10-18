"""Vision segmentation nodes: garment parts and portrait regions.

Both nodes build a SegFormer label map of the whole picture (see segformer_backend.py), union the class ids for
the requested region, refine the mask, and return (mask, cutout_rgba, mask_image) at the input resolution. The
parsers see 512 x 512 pictures, so they do not get the whole frame squeezed: Portrait Segment parses a crop around
each face (found with YuNet, see face_detect.py), Garment Segment a crop of the model's shape around the clothes
(see parse_crops.py).
"""

import cv2
import numpy as np
import torch

from ...core.face_detect import detect_faces
from ...core.parse_crops import (
    inner_mask,
    nearest_owner,
    parse_region,
    people_labels,
    person_box,
    square_around,
    square_frame,
    visible,
)
from ...core.segformer_backend import load_segformer, offload, predict_labels
from ...core.segs import SEGS_TYPE, SegsFrame, Segment

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
# Parts that only some outfits have, chosen on their own.
FASHIONPEDIA_EXTRA = {"dress": {11, 12}, "bag": {25}, "belt": {20, 35}, "scarf": {26}, "glasses": {14}, "jewelry": set()}
GARMENT_PART_CHOICES = list(GARMENT_GROUPS.keys()) + ["all_garments", *FASHIONPEDIA_EXTRA]

# ---------------------------------------------------------------------------
# FASHN Human Parser (18 classes) ids used by fashn-ai/fashn-human-parser
# ---------------------------------------------------------------------------
FASHN_PARTS = {
    "upper_garment": {3, 4, 10},  # top, dress, scarf (a dress is an upper garment, as with Fashionpedia)
    "lower_garment": {5, 6, 7},  # skirt, pants, belt
    "footwear": {15},  # feet: FASHN has no shoe class, its "feet" are the feet with whatever they wear
    "headwear": {9},  # hat
    "all_garments": {3, 4, 5, 6, 7, 9, 10},  # every garment, not the feet
    "dress": {4}, "bag": {8}, "belt": {7}, "scarf": {10}, "glasses": {11}, "jewelry": {17},
}
GARMENT_MODELS = {"fashn": "fashn_human_parser", "fashionpedia": "segformer_fashion"}

# One instance per part in the segs outputs, named as people say them.
FASHN_SEGS = {"face": {1}, "hair": {2}, "top": {3}, "dress": {4}, "skirt": {5}, "pants": {6}, "belt": {7},
              "bag": {8}, "hat": {9}, "scarf": {10}, "glasses": {11}, "arms": {12}, "hands": {13}, "legs": {14},
              "feet": {15}, "torso": {16}, "jewelry": {17}}
FASHIONPEDIA_SEGS = {
    "shirt": {1}, "top": {2}, "sweater": {3}, "cardigan": {4}, "jacket": {5}, "vest": {6}, "pants": {7},
    "shorts": {8}, "skirt": {9}, "coat": {10}, "dress": {11}, "jumpsuit": {12}, "cape": {13}, "glasses": {14},
    "hat": {15}, "headband": {16}, "tie": {17}, "glove": {18}, "watch": {19}, "belt": {20}, "leg warmer": {21},
    "tights": {22}, "sock": {23}, "shoe": {24}, "bag": {25}, "scarf": {26}, "umbrella": {27}, "hood": {28},
    "collar": {29}, "lapel": {30}, "epaulette": {31}, "sleeve": {32}, "pocket": {33}, "neckline": {34},
    "buckle": {35}, "zipper": {36}, "applique": {37}, "bead": {38}, "bow": {39}, "flower": {40}, "fringe": {41},
    "ribbon": {42}, "rivet": {43}, "ruffle": {44}, "sequin": {45}, "tassel": {46},
}

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
PORTRAIT_SEGS = {"skin": {1}, "nose": {2}, "glasses": {3}, "eyes": {4, 5}, "eyebrows": {6, 7}, "ears": {8, 9},
                 "mouth": {10}, "lips": {11, 12}, "hair": {13}, "hat": {14}, "earrings": {15}, "necklace": {16},
                 "neck": {17}, "clothes": {18}}


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


# ---------------------------------------------------------------------------
# Label maps of the whole picture, parsed on crops
# ---------------------------------------------------------------------------
# Square crops around each face, in face-box sides: the wide one keeps long hair and the neck, the tight one (moved
# up a little, for the forehead and hair line) is framed like the faces the parser learnt on, for eyes, brows and lips.
FACE_WIDE = 3.6
FACE_TIGHT = 2.2
FACE_LIFT = 0.06
GARMENT_MARGIN = 0.08  # around the clothes found on the first look, as a fraction of their longer side


FACE_CONTEXT = 1 / 3  # grey added on each side for a second look when no face is found (a face cut by the frame)
BIG_FACE = 0.9  # a face whose tight crop would span this share of the picture's long side is parsed whole-frame
FACE_CORE = 1.3  # the tight crop decides only inside an ellipse of the face box grown this much, and only about
# the face itself (see FACE_DETAIL): hair, neck and background come from the wide crop, which sees them whole instead
# of cut by the tight crop's edges.
# CelebAMask-HQ classes of the face parser: what a face is made of, and the features that make it a face
FACE_PARTS = frozenset({1, 2, 3, 4, 5, 6, 7, 10, 11, 12})  # skin, nose, glasses, eyes, brows, mouth, lips
FACE_FEATURES = frozenset({2, 4, 5, 6, 7, 10, 11, 12})  # nose, eyes, brows, mouth, lips
FACE_DETAIL = FACE_PARTS | {8, 9}  # with the ears: what the tight crop may decide

_detection_failed = False


def find_faces(image, second_look=True):
    """Face boxes of the uint8 RGB ``image``. A face cut by the frame is often found only once the picture gets a
    grey margin around it, so that is tried when the first look finds none; boxes mostly outside the frame are left
    out. If the detector cannot be loaded (offline first run, an old OpenCV, a damaged file) the node says so once
    and parses whole frames, as before the detector existed."""
    global _detection_failed
    if _detection_failed:
        return []
    try:
        faces = detect_faces(image)
        if faces or not second_look:
            return faces
        height, width = image.shape[:2]
        top, left = int(height * FACE_CONTEXT), int(width * FACE_CONTEXT)
        padded = cv2.copyMakeBorder(np.ascontiguousarray(image), top, top, left, left, cv2.BORDER_CONSTANT,
                                    value=(114, 114, 114))
        found = []
        for x1, y1, x2, y2, score in detect_faces(padded):
            box = (max(0.0, x1 - left), max(0.0, y1 - top), min(float(width), x2 - left), min(float(height), y2 - top))
            inside = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            if inside >= 0.5 * (x2 - x1) * (y2 - y1):
                found.append((*box, score))
        return found
    except Exception as exc:  # noqa: BLE001 - any failure of the optional detector falls back to whole frames
        _detection_failed = True
        print(f"[NH-Nodes] Portrait Segment: the face detector is not available ({exc}); whole frames are parsed "
              "instead. Restart ComfyUI to try again.")
        return []


def is_face(labels, seen, face):
    """True when ``labels`` (the parse of the part ``seen`` of a tight crop) show a face inside the detected box:
    mostly face parts, with some nose, eye, brow or lip. A hand or a pattern the detector took for a face is turned
    down, so its crops change nothing."""
    sx1, sy1 = seen[:2]
    fx1, fy1, fx2, fy2 = (int(round(value)) for value in face[:4])
    inside = labels[max(0, fy1 - sy1):max(0, fy2 - sy1), max(0, fx1 - sx1):max(0, fx2 - sx1)]
    if inside.size == 0:
        return False
    return (np.isin(inside, list(FACE_PARTS)).mean() >= 0.3
            and np.isin(inside, list(FACE_FEATURES)).mean() >= 0.005)


def portrait_labels(bundle, image):
    """Face-parsing label map of the uint8 RGB ``image``.

    The whole frame is parsed first, as a padded square (the parser learnt on square face pictures), and that answer
    is kept wherever the crops below do not reach. Each face found, unless it already fills the frame, is parsed again
    on a tight square crop (eyes, brows, lips) and, once that parse confirms it is a face, on a wide one (hair, neck).
    The wide crop is pasted inside its region away from its cut edges, the tight one only around the face itself, and
    each only where its face is the nearest one, so neighbours in a group photo do not overwrite each other and no
    straight crop edge shows in long hair.
    """
    def predict(picture):
        throw_exception_if_processing_interrupted()
        return predict_labels(bundle, picture)

    height, width = image.shape[:2]
    canvas = parse_region(predict, image, square_frame((height, width)))[0]
    long_side = max(height, width)
    candidates = [face for face in find_faces(image)
                  if FACE_TIGHT * max(face[2] - face[0], face[3] - face[1]) < BIG_FACE * long_side]
    tight = []
    for face in candidates:
        region = square_around(face, FACE_TIGHT, FACE_LIFT)
        labels, seen = parse_region(predict, image, region)
        if labels.size and is_face(labels, seen, face):
            tight.append((face, region, labels, seen))
    if not tight:
        return canvas
    faces = [face for face, _, _, _ in tight]
    wide_regions = [square_around(face, FACE_WIDE) for face in faces]
    # with the faces known, face parts the whole-frame parse saw far from all of them are mistakes (a reflection,
    # a pattern): only hair, neck and clothes are kept there
    near_faces = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in (visible(region, (height, width)) for region in wide_regions):
        near_faces[y1:y2, x1:x2] = True
    canvas[~near_faces & np.isin(canvas, list(FACE_DETAIL))] = 0
    tight_regions = [region for _, region, _, _ in tight]
    for index, region in enumerate(wide_regions):
        labels, seen = parse_region(predict, image, region)
        _paste(canvas, labels, seen,
               nearest_owner(seen, faces, wide_regions, index) & inner_mask(region, (height, width)))
    detail = list(FACE_DETAIL)
    for index, (face, region, labels, seen) in enumerate(tight):
        x1, y1, x2, y2 = seen
        about_face = np.isin(labels, detail) | np.isin(canvas[y1:y2, x1:x2], detail)
        _paste(canvas, labels, seen, nearest_owner(seen, faces, tight_regions, index) & _core(face, seen) & about_face)
    return canvas


def _core(face, seen):
    """Bool mask over ``seen`` (a region inside the picture): the ellipse of the face box grown FACE_CORE times."""
    x1, y1, x2, y2 = face[:4]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w, half_h = max(1.0, FACE_CORE * (x2 - x1) / 2.0), max(1.0, FACE_CORE * (y2 - y1) / 2.0)
    sx1, sy1, sx2, sy2 = seen
    ys = (np.arange(sy1, sy2, dtype=np.float32)[:, None] + 0.5 - cy) / half_h
    xs = (np.arange(sx1, sx2, dtype=np.float32)[None, :] + 0.5 - cx) / half_w
    return xs * xs + ys * ys <= 1.0


def _paste(canvas, labels, seen, keep):
    if labels.size:
        x1, y1, x2, y2 = seen
        canvas[y1:y2, x1:x2][keep] = labels[keep]


def garment_labels(bundle, image):
    """Fashion-parsing label map of the uint8 RGB ``image``: a first look at the whole frame finds the clothes, then
    crops of the model's shape around the people (found from the clothes and from their faces; a row of crops for a
    group; not squeezed, padded where they leave the picture) are parsed again and merged by confidence (see
    parse_crops.people_labels)."""
    size = getattr(bundle.get("processor"), "size", None) or {"height": 1, "width": 1}

    def predict(picture):
        throw_exception_if_processing_interrupted()
        return predict_labels(bundle, picture, confidence=True)

    people = [person_box(face, image.shape[:2]) for face in find_faces(image, second_look=False)]
    return people_labels(predict, image, size["width"] / size["height"], GARMENT_MARGIN, people)


WORK_SIDE = 4096  # label maps are worked out on a copy at most this long, then enlarged to the picture


def labels_at_work_size(labeler, bundle, frame, full=True):
    """(label map of the float HxWx3 ``frame`` at its size, the map it was enlarged from), worked out by
    ``labeler(bundle, uint8 RGB)`` on a copy at most WORK_SIDE pixels long. The parsers look at 512 to 1024 px at a
    time, so their labels hold no finer detail than that on an 8K photo, while merging crops at full size would take
    seconds. Both maps are the same when the frame is small enough; with ``full`` off the first one is None (a caller
    that builds everything with parts_frame does not need it)."""
    height, width = frame.shape[:2]
    pixels = cv2.convertScaleAbs(frame, alpha=255.0)  # rounds and saturates in one pass
    scale = min(1.0, WORK_SIDE / max(height, width))
    if scale >= 1.0:
        labels = labeler(bundle, pixels)
        return labels, labels
    small = cv2.resize(pixels, (max(1, round(width * scale)), max(1, round(height * scale))),
                       interpolation=cv2.INTER_AREA)
    work = labeler(bundle, small)
    if not full:
        return None, work
    rows, cols = _nearest_index(height, work.shape[0]), _nearest_index(width, work.shape[1])
    return work[np.ix_(rows, cols)], work


def _nearest_index(size, work_size, start=0, stop=None):
    """Index into a ``work_size`` map of each of the ``size`` pixels in [start, stop) it is enlarged to."""
    stop = size if stop is None else stop
    return np.minimum((np.arange(start, stop) * work_size) // size, work_size - 1)


def parts_frame(work, parts, size):
    """SegsFrame of a picture of ``size`` (height, width) with one instance per part of ``parts`` (name -> class
    ids) present in the label map ``work``, enlarged to the picture the same way as the mask: each instance is cut
    from the small map and only its own box is enlarged, so this costs little even on an 8K photo."""
    height, width = size
    small_h, small_w = work.shape
    segments = []
    for name, ids in parts.items():
        if not ids:
            continue
        found = np.isin(work, list(ids))
        rows, cols = np.nonzero(found.any(axis=1))[0], np.nonzero(found.any(axis=0))[0]
        if rows.size == 0:
            continue
        # the picture's pixels whose nearest map pixel lies in the part's box, and the map pixels they take
        y1, y2 = -(-int(rows[0]) * height // small_h), -(-(int(rows[-1]) + 1) * height // small_h)
        x1, x2 = -(-int(cols[0]) * width // small_w), -(-(int(cols[-1]) + 1) * width // small_w)
        crop = found[np.ix_(_nearest_index(height, small_h, y1, min(height, y2)),
                            _nearest_index(width, small_w, x1, min(width, x2)))]
        if crop.any():
            segments.append(Segment(name, 1.0, (x1, y1, x1 + crop.shape[1], y1 + crop.shape[0]),
                                    torch.from_numpy(np.ascontiguousarray(crop)), source="parser"))
    return SegsFrame(height, width, tuple(segments))


def _run_batch(image, model_name, device, build_mask, expand, blur, holes, labeler, parts):
    """Shared per-image loop: label map via ``labeler(bundle, uint8 RGB)``, mask via ``build_mask``, refine, pack;
    plus NH_SEGS with one instance per part of ``parts`` found."""
    bundle = load_segformer(model_name, device)
    masks, cutouts, mask_images, frames_out = [], [], [], []
    frames = _image_batch_to_numpy(image)
    try:
        for frame in frames:
            throw_exception_if_processing_interrupted()
            labels, work = labels_at_work_size(labeler, bundle, frame)
            mask = refine_mask(build_mask(labels), expand, blur, holes)
            mask_t, rgba_t, mask_img_t = pack_outputs(frame, mask)
            masks.append(mask_t)
            cutouts.append(rgba_t)
            mask_images.append(mask_img_t)
            frames_out.append(parts_frame(work, parts, labels.shape))
    finally:
        offload(bundle)
    return torch.cat(masks, 0), torch.cat(cutouts, 0), torch.cat(mask_images, 0), tuple(frames_out)


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
    """Segment a garment group or all garment groups from a person image."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image": ("IMAGE",),
            "part": (GARMENT_PART_CHOICES, {
                "default": "upper_garment",
                "tooltip": "all_garments combines upper/lower garments, footwear and headwear. With fashn, footwear "
                           "is the feet with whatever they wear, and jewelry is found; fashionpedia has shoes and "
                           "socks but no jewelry.",
            }),
            "attach_parts": ("BOOLEAN", {
                "default": True,
                "tooltip": "fashionpedia only: merge pockets, zippers, bows, ruffles... that touch the selected "
                           "garment (fashn counts them as part of the garment).",
            }),
        }
        required.update(_refine_inputs(fill_holes_default=False))
        return {
            "required": required,
            # optional, so prompts saved before it existed still validate (and get the better model)
            "optional": {
                "model": (list(GARMENT_MODELS), {
                    "default": "fashn",
                    "tooltip": "fashn: FASHN Human Parser (2025), clean tops, dresses, skirts, pants, bags, hats, "
                               "scarves, glasses and jewelry, 256 MB. fashionpedia: the older model, with shoes, "
                               "socks and small parts (sleeves, collars, pockets) in segs. Both are non-commercial "
                               "(NVIDIA SegFormer licence).",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE", SEGS_TYPE)
    RETURN_NAMES = ("mask", "cutout_rgba", "mask_image", "segs")
    FUNCTION = "segment"
    CATEGORY = "NH-Nodes/Vision"
    DESCRIPTION = (
        "Mask a garment group (upper, lower, footwear, headwear, all) or a single part (dress, bag, belt, scarf, "
        "glasses, jewelry) of the people in a photo. Each person is parsed close up at the model's full resolution. "
        "segs holds every part found, for Segs Compose."
    )
    SEARCH_ALIASES = ["garment segment", "clothes mask", "upper garment", "lower garment", "footwear", "headwear",
                      "all garments", "dress mask", "bag mask", "fashn"]

    @staticmethod
    def build_mask(labels, part, attach_parts=True, model="fashionpedia"):
        if model == "fashn":
            return labels_to_mask(labels, FASHN_PARTS.get(part, set()))
        if part in FASHIONPEDIA_EXTRA:
            return labels_to_mask(labels, FASHIONPEDIA_EXTRA[part])
        class_ids = set().union(*GARMENT_GROUPS.values()) if part == "all_garments" else GARMENT_GROUPS[part]
        mask = labels_to_mask(labels, class_ids)
        if attach_parts:
            mask = absorb_touching_parts(mask, labels_to_mask(labels, GARMENT_ATTACHED_PARTS))
        return mask

    def segment(self, image, part, attach_parts, mask_expand, mask_blur, fill_holes, device, model="fashn"):
        if part not in GARMENT_PART_CHOICES:
            part = GARMENT_PART_CHOICES[0]
        model = model if model in GARMENT_MODELS else "fashn"
        if model == "fashionpedia" and part == "jewelry":
            print("[NH-Nodes] Garment Segment: fashionpedia has no jewelry class; choose model fashn for jewelry")
        return _run_batch(
            image, GARMENT_MODELS[model], device,
            lambda labels: self.build_mask(labels, part, attach_parts, model),
            mask_expand, mask_blur, fill_holes, garment_labels,
            FASHN_SEGS if model == "fashn" else FASHIONPEDIA_SEGS,
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

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE", SEGS_TYPE)
    RETURN_NAMES = ("mask", "cutout_rgba", "mask_image", "segs")
    FUNCTION = "segment"
    CATEGORY = "NH-Nodes/Vision"
    DESCRIPTION = (
        "Mask face, hair, neck and facial features (eyes, eyebrows, nose, lips, ears) of every face in a photo, each "
        "face parsed close up. segs holds every part found (skin, eyes, lips, hair, earrings...), for Segs Compose."
    )
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
            mask_expand, mask_blur, fill_holes, portrait_labels, PORTRAIT_SEGS,
        )


NODE_CLASS_MAPPINGS = {
    "NH_GarmentSegment": NH_GarmentSegment,
    "NH_PortraitSegment": NH_PortraitSegment,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_GarmentSegment": "Garment Segment (NH)",
    "NH_PortraitSegment": "Portrait Segment (NH)",
}
