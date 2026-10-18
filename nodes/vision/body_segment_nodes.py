"""Body Segment (NH): skin, hair, arms, lips, teeth, clothes... of the people in a picture, from a few words.

Sapiens2 (core/sapiens2_backend.py) labels every pixel of each person with one of 28 body parts and clothes, on crops
of its 3:4 shape around the people (core/parse_crops.people_labels), so a person gets about 1024 px of the model
whatever the size of the photo. The words you type pick the regions, with the formula of SAM3 Segment (core/segs.py):
``skin - lips``, ``hair + face``, ``~person``.
"""

import numpy as np
import torch

from ...core import sapiens2_backend as sapiens2
from ...core.parse_crops import people_labels, person_box
from ...core.segs import (
    SEGS_TYPE,
    EmptyExpressionError,
    ExpressionError,
    evaluate,
    expression_labels,
    expression_terms,
    normalize_label,
    parse_concepts,
    preview_picture,
)
from ...core.ui_preview import MAX_PREVIEWS, save_temp_previews
from .vision_segment_nodes import (
    DEVICES,
    _image_batch_to_numpy,
    find_faces,
    labels_at_work_size,
    pack_outputs,
    parts_frame,
    refine_mask,
)

try:
    from comfy.model_management import throw_exception_if_processing_interrupted
except Exception:  # outside ComfyUI (tests)
    def throw_exception_if_processing_interrupted():
        return None

_C = {name: index for index, name in enumerate(sapiens2.CLASSES)}


def _ids(*names):
    return frozenset(_C[name] for name in names)


ARMS = _ids("left_lower_arm", "left_upper_arm", "right_lower_arm", "right_upper_arm")
LEGS = _ids("left_lower_leg", "left_upper_leg", "right_lower_leg", "right_upper_leg")
HANDS = _ids("left_hand", "right_hand")
FEET = _ids("left_foot", "right_foot")
LIPS = _ids("lower_lip", "upper_lip")
TEETH = _ids("lower_teeth", "upper_teeth")
SKIN = _ids("face_neck", "torso") | ARMS | LEGS | HANDS | FEET
CLOTHES = _ids("apparel", "upper_clothing", "lower_clothing", "left_shoe", "right_shoe", "left_sock", "right_sock")

# The words Body Segment understands, and the classes each one covers. Plurals and the common synonyms are listed so
# that what people naturally type works; Vietnamese words reach here already translated by vi_words.
REGIONS = {
    "skin": SKIN, "body skin": SKIN,
    "face": _ids("face_neck"), "neck": _ids("face_neck"), "face neck": _ids("face_neck"),
    "hair": _ids("hair"),
    "arm": ARMS, "arms": ARMS, "upper arm": _ids("left_upper_arm", "right_upper_arm"),
    "upper arms": _ids("left_upper_arm", "right_upper_arm"),
    "lower arm": _ids("left_lower_arm", "right_lower_arm"), "lower arms": _ids("left_lower_arm", "right_lower_arm"),
    "forearm": _ids("left_lower_arm", "right_lower_arm"), "forearms": _ids("left_lower_arm", "right_lower_arm"),
    "hand": HANDS, "hands": HANDS,
    "leg": LEGS, "legs": LEGS, "thigh": _ids("left_upper_leg", "right_upper_leg"),
    "thighs": _ids("left_upper_leg", "right_upper_leg"), "upper legs": _ids("left_upper_leg", "right_upper_leg"),
    "calf": _ids("left_lower_leg", "right_lower_leg"), "calves": _ids("left_lower_leg", "right_lower_leg"),
    "lower legs": _ids("left_lower_leg", "right_lower_leg"),
    "foot": FEET, "feet": FEET,
    "torso": _ids("torso"), "chest": _ids("torso"), "belly": _ids("torso"),
    "lips": LIPS, "lip": LIPS, "upper lip": _ids("upper_lip"), "lower lip": _ids("lower_lip"),
    "teeth": TEETH, "tooth": TEETH, "upper teeth": _ids("upper_teeth"), "lower teeth": _ids("lower_teeth"),
    "tongue": _ids("tongue"), "mouth": LIPS | TEETH | _ids("tongue"),
    "glasses": _ids("eyeglass"), "eyeglasses": _ids("eyeglass"),
    "clothes": CLOTHES, "clothing": CLOTHES, "outfit": CLOTHES,
    "top": _ids("upper_clothing"), "upper clothes": _ids("upper_clothing"), "shirt": _ids("upper_clothing"),
    "dress": _ids("upper_clothing"),
    "bottom": _ids("lower_clothing"), "lower clothes": _ids("lower_clothing"), "pants": _ids("lower_clothing"),
    "trousers": _ids("lower_clothing"), "jeans": _ids("lower_clothing"), "shorts": _ids("lower_clothing"),
    "skirt": _ids("lower_clothing"),
    "shoe": _ids("left_shoe", "right_shoe"), "shoes": _ids("left_shoe", "right_shoe"),
    "sock": _ids("left_sock", "right_sock"), "socks": _ids("left_sock", "right_sock"),
    "accessories": _ids("apparel"), "apparel": _ids("apparel"),
    "person": frozenset(range(1, len(sapiens2.CLASSES))), "body": frozenset(range(1, len(sapiens2.CLASSES))),
    "left hand": _ids("left_hand"), "right hand": _ids("right_hand"),
    "left arm": _ids("left_lower_arm", "left_upper_arm"), "right arm": _ids("right_lower_arm", "right_upper_arm"),
    "left leg": _ids("left_lower_leg", "left_upper_leg"), "right leg": _ids("right_lower_leg", "right_upper_leg"),
    "left foot": _ids("left_foot"), "right foot": _ids("right_foot"),
    "left shoe": _ids("left_shoe"), "right shoe": _ids("right_shoe"),
}
# One instance per part in the segs output, named as people say them.
PARTS = {
    "hair": _ids("hair"), "face": _ids("face_neck"), "torso": _ids("torso"), "arms": ARMS, "hands": HANDS,
    "legs": LEGS, "feet": FEET, "lips": LIPS, "teeth": TEETH, "tongue": _ids("tongue"), "glasses": _ids("eyeglass"),
    "top": _ids("upper_clothing"), "bottom": _ids("lower_clothing"), "shoes": _ids("left_shoe", "right_shoe"),
    "socks": _ids("left_sock", "right_sock"), "accessories": _ids("apparel"),
}
MODEL_SIZES = tuple(sapiens2.MODELS)
_EXAMPLES = "skin, hair, face, arms, hands, legs, feet, torso, lips, teeth, glasses, clothes, top, bottom, shoes"


PERSON = frozenset(range(1, len(sapiens2.CLASSES)))


def region_mask(labels, word):
    """Bool mask of the classes ``word`` covers in a Sapiens2 label map, or None for a word Body Segment lacks."""
    ids = REGIONS.get(normalize_label(word))
    return None if ids is None else np.isin(labels, list(ids))


def body_labels(bundle, picture):
    """Sapiens2 label map of the people in the uint8 RGB ``picture``, parsed close up (see parse_crops.people_labels),
    with the people its first look misses found from their faces."""
    people = [person_box(face, picture.shape[:2]) for face in find_faces(picture, second_look=False)]
    return people_labels(lambda crop: sapiens2.predict_labels(bundle, crop, confidence=True), picture,
                         sapiens2.INPUT_SIZE[1] / sapiens2.INPUT_SIZE[0], people=people)


def describe(tree, labels_list):
    """Report lines for a request over the label maps of a batch: share of the pictures each word covers."""
    pixels = sum(labels.size for labels in labels_list)
    lines, seen = [], set()
    for written, english, status in expression_terms(tree):
        key = (status == "keyword", normalize_label(english))
        if key in seen:
            continue
        seen.add(key)
        if status == "keyword":
            if english == "background":
                lines.append("background = everything outside the people")
            continue
        name = f"{written} -> {english}" if status == "translated" else written
        if normalize_label(english) not in REGIONS:
            lines.append(f"{name}: not a body region; use {_EXAMPLES}")
            continue
        covered = sum(int(region_mask(labels, english).sum()) for labels in labels_list)
        lines.append(f"{name}: {100.0 * covered / max(1, pixels):.1f}% of the picture" if covered
                     else f"{name}: not in the picture")
    return lines


class NH_BodySegment:
    """Mask body parts, skin and clothes of the people in a picture by name, with Sapiens2 (Meta)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "regions": ("STRING", {
                    "default": "skin", "multiline": True,
                    "tooltip": f"Which parts, by name: {_EXAMPLES}, upper arm, thighs, left hand, mouth, person... "
                               "Vietnamese works too (da, tóc, môi, răng). Combine like SAM3 Segment: skin - lips, "
                               "hair + face, ~person, background. Commas list several.",
                }),
                "model_size": (list(MODEL_SIZES), {
                    "default": "0.4b",
                    "tooltip": "Sapiens2 size: 0.4b (1.6 GB download, about 1.5 GB VRAM) is fast and fine; 0.8b and "
                               "1b (5.9 GB) are a little more precise and slower.",
                }),
                "mask_expand": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1,
                                        "tooltip": "Grow (+) or shrink (-) the mask by N pixels."}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1,
                                      "tooltip": "Feather the mask edge by N pixels."}),
                "fill_holes": ("BOOLEAN", {"default": False, "tooltip": "Fill enclosed holes."}),
                "device": (DEVICES, {"default": "cuda"}),
                "show_preview": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show the regions and the report on the node itself.",
                }),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE", SEGS_TYPE, "STRING")
    RETURN_NAMES = ("mask", "cutout_rgba", "mask_image", "segs", "report")
    OUTPUT_NODE = True  # it shows its own preview, so it runs even with nothing connected
    FUNCTION = "segment"
    CATEGORY = "NH-Nodes/Vision"
    DESCRIPTION = (
        "Type the body parts you want ('skin', 'hair, face', 'skin - lips', 'teeth') and get their mask: Sapiens2 "
        "(Meta) labels skin, hair, face, arms, hands, legs, feet, torso, lips, teeth, glasses and clothes of every "
        "person at about 1024 px per person. Shows what it found on the node. segs feed Segs Compose."
    )
    SEARCH_ALIASES = ["body segment", "skin mask", "body parts", "sapiens", "teeth mask", "lips mask", "arms mask"]

    def segment(self, image, regions, model_size, mask_expand, mask_blur, fill_holes, device, show_preview=True):
        try:
            tree = parse_concepts(regions)
        except EmptyExpressionError:
            raise ValueError("[NH-Nodes] Body Segment: write at least one region, e.g. 'skin' or 'hair'.") from None
        except ExpressionError as exc:
            raise ValueError(f"[NH-Nodes] Body Segment: {exc}") from None
        words = list(dict.fromkeys(normalize_label(word) for word in expression_labels(tree)))
        bundle = sapiens2.load(model_size, sapiens2.torch_device(device))
        masks, cutouts, mask_images, frames, label_maps = [], [], [], [], []
        try:
            for picture in _image_batch_to_numpy(image):
                throw_exception_if_processing_interrupted()
                size = picture.shape[:2]
                _, work = labels_at_work_size(body_labels, bundle, picture, full=False)
                label_maps.append(work)
                # the formula sees one instance per word it names, plus the person (for ~person and background)
                named = {word: REGIONS[word] for word in words if word in REGIONS}
                formula_frame = parts_frame(work, {**named, "person": PERSON}, size)
                mask = refine_mask(evaluate(tree, formula_frame).numpy().astype(np.uint8), mask_expand, mask_blur,
                                   fill_holes)
                mask_t, rgba_t, mask_img_t = pack_outputs(picture, mask)
                masks.append(mask_t)
                cutouts.append(rgba_t)
                mask_images.append(mask_img_t)
                frames.append(parts_frame(work, PARTS, size))
        finally:
            sapiens2.offload(bundle)
        lines = describe(tree, label_maps)
        report = "\n".join([f"Body Segment (Sapiens2 {model_size}): {len(frames)} image(s)", *lines])
        mask_batch = torch.cat(masks, 0)
        result = (mask_batch, torch.cat(cutouts, 0), torch.cat(mask_images, 0), tuple(frames), report)
        if not show_preview:
            return result
        pictures = []
        for picture, frame, held in zip(image[:MAX_PREVIEWS], frames[:MAX_PREVIEWS], mask_batch[:MAX_PREVIEWS]):
            shown = frame.with_segments(seg for seg in frame.segments
                                        if any(REGIONS.get(word, frozenset()) & PARTS[seg.label] for word in words))
            pictures.append(preview_picture(picture, shown, lines, mask=held))
        return {"ui": {"images": save_temp_previews(pictures, "NH_body")}, "result": result}


NODE_CLASS_MAPPINGS = {
    "NH_BodySegment": NH_BodySegment,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_BodySegment": "Body Segment (NH)",
}
