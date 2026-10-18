"""Body Segment (NH): skin, hair, arms, lips, teeth, clothes... of the people in a picture, from a few words.

Sapiens2 (core/sapiens2_backend.py) labels every pixel of each person with one of 28 body parts and clothes, on crops
of its 3:4 shape around the people (core/parse_crops.people_labels), so a person gets about 1024 px of the model
whatever the size of the photo. The words you type pick the regions, with the formula of SAM3 Segment (core/segs.py):
``clothes - shoes``, ``hair + face``, ``~person``. Sapiens2 has fewer classes than the words people use, so some words
share one (the face and the neck, a top and a dress); the report says so when it matters.
"""

import numpy as np
import torch

from ...core import sapiens2_backend as sapiens2
from ...core.parse_crops import people_labels, person_box
from ...core.segs import (
    SEGS_TYPE,
    EmptyExpressionError,
    ExpressionError,
    add_aliases,
    evaluate,
    expression_terms,
    fold_label,
    normalize_label,
    parse_concepts,
    preview_picture,
    stamped,
)
from ...core.ui_preview import MAX_PREVIEWS, save_temp_previews
from ...core.vi_words import normalize
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
FACE = _ids("face_neck")
HAIR = _ids("hair")
TOP = _ids("upper_clothing")
BOTTOM = _ids("lower_clothing")
SHOES = _ids("left_shoe", "right_shoe")
GLASSES = _ids("eyeglass")
SKIN = FACE | _ids("torso") | ARMS | LEGS | HANDS | FEET
CLOTHES = _ids("apparel", "upper_clothing", "lower_clothing", "left_shoe", "right_shoe", "left_sock", "right_sock")
PERSON = frozenset(range(1, len(sapiens2.CLASSES)))

# The words Body Segment understands, and the classes each one covers. Plurals and the common synonyms are listed so
# that what people naturally type works; Vietnamese words reach here already translated by vi_words.
REGIONS = {
    "skin": SKIN, "body skin": SKIN,
    "face": FACE, "neck": FACE, "face neck": FACE, "face skin": FACE, "head": FACE | HAIR,
    "hair": HAIR,
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
    "glasses": GLASSES, "eyeglasses": GLASSES, "sunglasses": GLASSES,
    "clothes": CLOTHES, "clothing": CLOTHES, "outfit": CLOTHES,
    # Sapiens2 has one class for everything worn on the upper body, dresses included, and one for the lower body
    "top": TOP, "upper clothes": TOP, "shirt": TOP, "t-shirt": TOP, "blouse": TOP, "jacket": TOP, "suit jacket": TOP,
    "sweater": TOP, "hoodie": TOP, "coat": TOP, "vest": TOP, "bra": TOP, "dress": TOP, "wedding dress": TOP,
    "ao dai": TOP,
    "bottom": BOTTOM, "lower clothes": BOTTOM, "pants": BOTTOM, "trousers": BOTTOM, "jeans": BOTTOM,
    "shorts": BOTTOM, "skirt": BOTTOM,
    "shoe": SHOES, "shoes": SHOES, "high heels": SHOES, "sandal": SHOES, "sandals": SHOES, "sneakers": SHOES,
    "boots": SHOES,
    "sock": _ids("left_sock", "right_sock"), "socks": _ids("left_sock", "right_sock"),
    "accessories": _ids("apparel"), "apparel": _ids("apparel"),
    "person": PERSON, "body": PERSON,
    "left hand": _ids("left_hand"), "right hand": _ids("right_hand"),
    "left arm": _ids("left_lower_arm", "left_upper_arm"), "right arm": _ids("right_lower_arm", "right_upper_arm"),
    "left leg": _ids("left_lower_leg", "left_upper_leg"), "right leg": _ids("right_lower_leg", "right_upper_leg"),
    "left foot": _ids("left_foot"), "right foot": _ids("right_foot"),
    "left shoe": _ids("left_shoe"), "right shoe": _ids("right_shoe"),
}
# Vietnamese "tay" and "chân" name the whole limb, hand or foot included ("cánh tay" is the arm alone, "bàn tay" the
# hand); vi_words turns them into "arm" and "leg", the limb alone in English, so the request keeps them apart under
# their own name (see _whole_limbs).
WHOLE_LIMBS = {
    "tay": ARMS | HANDS, "đôi tay": ARMS | HANDS, "hai tay": ARMS | HANDS,
    "chân": LEGS | FEET, "đôi chân": LEGS | FEET, "hai chân": LEGS | FEET,
    "tay trái": _ids("left_lower_arm", "left_upper_arm", "left_hand"),
    "tay phải": _ids("right_lower_arm", "right_upper_arm", "right_hand"),
    "chân trái": _ids("left_lower_leg", "left_upper_leg", "left_foot"),
    "chân phải": _ids("right_lower_leg", "right_upper_leg", "right_foot"),
}
# Words people try that Sapiens2 has no class for, and what to do instead.
_IN_FACE = "eyes, eyebrows, nose and ears are part of 'face' here; Portrait Segment masks them one by one"
_ACCESSORY = ("Sapiens2 puts hats, bags and other accessories in one class, 'accessories'; for one of them use "
              "Garment Segment or SAM3 Segment")
NOT_REGIONS = {
    **dict.fromkeys(("eye", "eyes", "eyebrow", "eyebrows", "eyelash", "nose", "ear", "ears", "beard"), _IN_FACE),
    **dict.fromkeys(("hat", "cap", "bag", "handbag", "backpack", "scarf", "belt", "necklace", "jewelry", "watch",
                     "glove", "gloves", "tie"), _ACCESSORY),
    **dict.fromkeys(("shoulder", "shoulders"), "shoulders are part of 'torso' and 'upper arms'"),
    **dict.fromkeys(("finger", "fingers", "fingernail"), "fingers are part of 'hands'"),
}
# One instance per part in the segs output, named as people say them.
PARTS = {
    "hair": HAIR, "face": FACE, "torso": _ids("torso"), "arms": ARMS, "hands": HANDS,
    "legs": LEGS, "feet": FEET, "lips": LIPS, "teeth": TEETH, "tongue": _ids("tongue"), "glasses": GLASSES,
    "top": TOP, "bottom": BOTTOM, "shoes": SHOES, "socks": _ids("left_sock", "right_sock"), "accessories": _ids("apparel"),
}
MODEL_SIZES = tuple(sapiens2.MODELS)
_EXAMPLES = "skin, hair, face, arms, hands, legs, feet, torso, lips, teeth, glasses, clothes, top, bottom, shoes"
_VIETNAMESE = "da, tóc, mặt, môi, răng, tay, chân, áo, quần, giày"
# With no face found, Sapiens2's first look must be sure (0.9 or more) of this share of what it labels as people: it
# was 0.61 to 0.98 on 32 faceless photos of people (backs, bodies without heads, a mannequin) and 0.00 to 0.41 on 17
# pictures of no one (a facade, a cafe, an elephant, products, masks), with PIL's and with ComfyUI's JPEG decoder.
FIRST_LOOK_FLOOR = 0.5

_SIDES = ("left and right are the person's own sides: on the right of the picture when they face the camera")
_FACE_CLASS = ("face is one Sapiens2 class with the neck, eyes, brows and beard (skin includes it); Portrait Segment "
               "masks the parts of a face")
_NOBODY = "no person found (none Sapiens2 was sure of, and no face)"
# the words above in singular (fold_label), so "lower leg", "accessory" or "dresses" are read like the listed form
_FOLDED_REGIONS = {fold_label(word): ids for word, ids in REGIONS.items()}
_FOLDED_NOT_REGIONS = {fold_label(word): advice for word, advice in NOT_REGIONS.items()}
# Segs Compose and Segs Select read Body Segment's words on its segs: "skin" is the face, torso, arms... parts.
add_aliases("sapiens2", REGIONS, PARTS)


def region_ids(written, english, status):
    """Class ids a word of the request covers, or None when Body Segment has no such region."""
    if status == "translated" and normalize(written) in WHOLE_LIMBS:
        return WHOLE_LIMBS[normalize(written)]
    word = normalize_label(english)
    return REGIONS.get(word, _FOLDED_REGIONS.get(fold_label(word)))


def _whole_limbs(tree):
    """The parsed request with "tay", "chân"... named as written, so "tay - cánh tay" (whole arms minus arms) leaves the
    hands instead of nothing: both would be "arm" in English."""
    if tree[0] == "label":
        word = normalize(tree[2])
        return ("label", word, tree[2], tree[3]) if tree[3] == "translated" and word in WHOLE_LIMBS else tree
    if tree[0] == "keyword":
        return tree
    return (tree[0], *(_whole_limbs(child) for child in tree[1:]))


def check_regions(tree):
    """{english label: class ids} of the regions a parsed request names; a word Body Segment does not know raises a
    ValueError saying what to write instead, before any model is loaded."""
    named, problems = {}, []
    for written, english, status in expression_terms(tree):
        if status == "keyword":
            continue
        word = normalize_label(english)
        shown = f"'{written}' ({english})" if status == "translated" else f"'{written}'"
        if any(ch in word for ch in "*?["):
            problems.append(f"{shown} has a wildcard; Body Segment takes region names ({_EXAMPLES})")
            continue
        ids = region_ids(written, english, status)
        if ids is not None:
            named[word] = named.get(word, frozenset()) | ids
        elif status == "unknown":
            problems.append(f"{shown} is not in the Vietnamese word list; write it in English or use {_VIETNAMESE}")
        elif fold_label(word) in _FOLDED_NOT_REGIONS:
            problems.append(f"{shown} is not a body region: {_FOLDED_NOT_REGIONS[fold_label(word)]}")
        elif {"and", "or", "not"} & set(word.split()):
            problems.append(f"{shown} is not a body region; combine regions with + or a comma (skin + hair), remove "
                            "with - and invert with ~")
        else:
            problems.append(f"{shown} is not a body region; use {_EXAMPLES} (Vietnamese too: {_VIETNAMESE})")
    if problems:
        raise ValueError("[NH-Nodes] Body Segment: " + "; ".join(problems) + ".")
    return named


def _share(covered, pixels, where):
    if not covered:
        return f"not in the {where}"
    share = 100.0 * covered / max(1, pixels)
    return f"{share:.1f}% of the {where}" if share >= 0.1 else f"under 0.1% of the {where}"


def describe(tree, counts, pixels, where="picture"):
    """Report lines for a request: the share of the picture each word covers, from the per-class pixel ``counts``
    of a label map of ``pixels`` pixels; then what the words mean where Sapiens2 differs from everyday use."""
    lines, notes, seen = [], [], set()
    for written, english, status in expression_terms(tree):
        word = normalize_label(english)
        if (status == "keyword", word) in seen:
            continue
        seen.add((status == "keyword", word))
        if status == "keyword":
            if english == "background":
                lines.append("background = everything outside the people")
            continue
        ids = region_ids(written, english, status)
        name = f"{written} -> {english}" if status == "translated" and word != normalize(written) else written
        lines.append(f"{name}: {_share(int(sum(counts[i] for i in ids)), pixels, where)}")
        folded = fold_label(word)
        if word.startswith(("left ", "right ")):
            notes.append(_SIDES)
        if folded in ("skin", "body skin", "face", "neck", "face neck", "face skin", "head"):
            notes.append(_FACE_CLASS)
        if folded in ("dress", "wedding dress", "ao dai"):
            notes.append(f"{name}: Sapiens2 labels a dress as a top (upper clothes)")
        if folded == "skirt":
            notes.append("skirt: the skirt of a dress counts as a top in Sapiens2")
        if status == "translated" and normalize(written) in WHOLE_LIMBS:
            notes.append(f"{written}: the whole limb, {'hands' if ids & HANDS else 'feet'} included")
    return lines + list(dict.fromkeys(notes))


def body_labels(bundle, picture):
    """Sapiens2 label map of the people in the uint8 RGB ``picture``, parsed close up (see parse_crops.people_labels),
    with the people its first look misses found from their faces; nobody when no face is found and it is unsure of
    everyone it sees (on a picture without people it would otherwise see faint people in walls and shelves)."""

    def predict(crop):
        throw_exception_if_processing_interrupted()
        return sapiens2.predict_labels(bundle, crop, confidence=True)

    people = [person_box(face, picture.shape[:2]) for face in find_faces(picture, second_look=False)]
    return people_labels(predict, picture, sapiens2.INPUT_SIZE[1] / sapiens2.INPUT_SIZE[0], people=people,
                         floor=FIRST_LOOK_FLOOR)


def _mostly_inside(seg, held):
    """True when at least half of the instance ``seg`` lies in the bool mask ``held``."""
    x1, y1, x2, y2 = seg.box
    pixels = seg.mask.numpy()
    return 2 * int((held[y1:y2, x1:x2] & pixels).sum()) >= int(pixels.sum())


class NH_BodySegment:
    """Mask body parts, skin and clothes of the people in a picture by name, with Sapiens2 (Meta)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "regions": ("STRING", {
                    "default": "skin", "multiline": True,
                    "tooltip": f"Which parts, by name: {_EXAMPLES}, upper arms, thighs, left hand, mouth, person... "
                               f"Vietnamese works too ({_VIETNAMESE}). Combine like SAM3 Segment: clothes - shoes, "
                               "hair + face, ~person, background. Commas list several. Left and right are the "
                               "person's own sides; face includes the neck.",
                }),
                "model_size": (list(MODEL_SIZES), {
                    "default": "0.4b",
                    "tooltip": "Sapiens2 size: 0.4b (1.6 GB download, about 1.5 GB of VRAM) is fast and good; 0.8b "
                               "(3.3 GB) and 1b (5.9 GB) are a little more precise, slower and need more VRAM.",
                }),
                "mask_expand": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1,
                                        "tooltip": "Grow (+) or shrink (-) the mask by N pixels."}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1,
                                      "tooltip": "Feather the mask edge by N pixels."}),
                "fill_holes": ("BOOLEAN", {"default": False, "tooltip": "Fill enclosed holes."}),
                "device": (DEVICES, {"default": "cuda"}),
            },
            # optional, like SAM3 Segment's, so API prompts without it still validate
            "optional": {
                "show_preview": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show the regions and the report on the node itself.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE", SEGS_TYPE, "STRING")
    RETURN_NAMES = ("mask", "cutout_rgba", "mask_image", "segs", "report")
    OUTPUT_NODE = True  # it shows its own preview, so it runs even with nothing connected
    FUNCTION = "segment"
    CATEGORY = "NH-Nodes/Vision"
    DESCRIPTION = (
        "Type the body parts you want ('skin', 'hair, face', 'clothes - shoes', 'teeth') and get their mask: Sapiens2 "
        "(Meta) labels skin, hair, face, arms, hands, legs, feet, torso, lips, teeth, glasses and clothes of every "
        "person at about 1024 px per person. Shows what it found on the node. segs feed Segs Compose."
    )
    SEARCH_ALIASES = ["body segment", "skin mask", "body parts", "sapiens", "teeth mask", "lips mask", "arms mask"]

    def segment(self, image, regions, model_size, mask_expand, mask_blur, fill_holes, device, show_preview=True):
        try:
            tree = _whole_limbs(parse_concepts(regions))
        except EmptyExpressionError:
            raise ValueError("[NH-Nodes] Body Segment: write at least one region, e.g. 'skin' or 'hair'.") from None
        except ExpressionError as exc:
            raise ValueError(f"[NH-Nodes] Body Segment: {exc}") from None
        named = check_regions(tree)
        bundle = sapiens2.load(model_size, sapiens2.torch_device(device))
        masks, cutouts, mask_images, frames, counts, pixels = [], [], [], [], [], []
        try:
            for picture in _image_batch_to_numpy(image):
                throw_exception_if_processing_interrupted()
                size = picture.shape[:2]
                _, work = labels_at_work_size(body_labels, bundle, picture, full=False)
                counts.append(np.bincount(work.ravel(), minlength=len(sapiens2.CLASSES)))
                pixels.append(work.size)
                # the formula sees one instance per word it names, plus the person (for ~person and background)
                formula_frame = parts_frame(work, {**named, "person": PERSON}, size)
                mask = refine_mask(evaluate(tree, formula_frame).numpy().astype(np.uint8), mask_expand, mask_blur,
                                   fill_holes)
                mask_t, rgba_t, mask_img_t = pack_outputs(picture, mask)
                masks.append(mask_t)
                cutouts.append(rgba_t)
                mask_images.append(mask_img_t)
                frames.append(parts_frame(work, PARTS, size, "sapiens2"))
        finally:
            sapiens2.offload(bundle)
        frames = stamped(frames, image)
        where = "picture" if len(frames) == 1 else "pictures"
        lines = describe(tree, np.sum(counts, axis=0), sum(pixels), where)
        empty = [str(index + 1) for index, found in enumerate(counts) if not found[1:].any()]
        if empty:
            lines.append(_NOBODY if len(frames) == 1 else f"{_NOBODY} in image(s) {', '.join(empty)}")
        report = "\n".join([f"Body Segment (Sapiens2 {model_size}): {len(frames)} image(s)", *lines])
        mask_batch = torch.cat(masks, 0)
        result = (mask_batch, torch.cat(cutouts, 0), torch.cat(mask_images, 0), frames, report)
        if not show_preview:
            return result
        wanted = frozenset().union(*named.values())
        try:
            pictures = []
            for index, (picture, frame, held) in enumerate(zip(image[:MAX_PREVIEWS], frames, mask_batch)):
                inside = held.numpy() >= 0.5
                # the parts the request names and keeps: "skin - lips" shows the skin parts, not the lips
                shown = frame.with_segments(seg for seg in frame.segments
                                            if PARTS[seg.label] & wanted and _mostly_inside(seg, inside))
                notes = lines if len(frames) == 1 else describe(tree, counts[index], pixels[index]) + (
                    [] if counts[index][1:].any() else [_NOBODY])
                pictures.append(preview_picture(picture, shown, notes, mask=held))
            images = save_temp_previews(pictures, "NH_body")
        except Exception as exc:  # noqa: BLE001 - a preview that cannot be drawn or saved must not lose the run
            print(f"[NH-Nodes] Body Segment: could not show the preview ({exc}); the outputs are fine")
            return result
        return {"ui": {"images": images}, "result": result}


NODE_CLASS_MAPPINGS = {
    "NH_BodySegment": NH_BodySegment,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_BodySegment": "Body Segment (NH)",
}
