"""SAM3 Segment (NH): find what you name with SAM 3 and get its mask, straight from a short formula.

The model work lives in core/sam3_backend.py and the formula (labels, + - & ~, background, Vietnamese words) in
core/segs.py; this module maps the node's inputs and outputs and draws its preview.
"""

from ...core import sam3_backend as sam3
from ...core.segs import (
    SEGS_TYPE,
    EmptyExpressionError,
    ExpressionError,
    count_segments,
    describe_request,
    expression_labels,
    expression_masks,
    normalize_label,
    parse_concepts,
    preview_picture,
    stamped,
    uses_keyword,
)
from ...core.ui_preview import MAX_PREVIEWS, save_temp_previews

try:
    from comfy.model_management import throw_exception_if_processing_interrupted
except Exception:  # outside ComfyUI (tests)
    def throw_exception_if_processing_interrupted():
        return None

try:
    from comfy.utils import ProgressBar
except Exception:  # outside ComfyUI (tests)
    ProgressBar = None


def concepts_to_find(tree):
    """(concepts SAM 3 must look for, the ones the node added) for a parsed request, without repeats.

    ``background`` needs to know where the subject is, so "person" is added when the request does not name it.
    """
    names, seen = [], set()
    for label in expression_labels(tree):
        if any(ch in label for ch in "*?["):
            raise ValueError(f"[NH-Nodes] SAM3 Segment: '{label}' has a wildcard; SAM 3 needs plain words (wildcards "
                             "work in Segs Select and Segs Compose).")
        if normalize_label(label) not in seen:
            seen.add(normalize_label(label))
            names.append(label)
    added = []
    if uses_keyword(tree, "background") and "person" not in seen:
        names.append("person")
        added.append("person")
    return names, added


class NH_SAM3Segment:
    """Find every instance of the concepts you name with SAM 3 (Meta), at the full resolution of the image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "concepts": ("STRING", {
                    "default": "person", "multiline": True,
                    "tooltip": "What to find, in short English words (person, red car, window, sky); common "
                               "Vietnamese words work too (tóc, váy, túi, nền). Commas or new lines list several. "
                               "The mask follows the formula: person - bag (minus), hair + face (plus), ~person "
                               "(everything else), background (everything outside what was found; people are "
                               "searched for it), shirt & jacket (overlap), parentheses.",
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.05, "max": 0.95, "step": 0.01,
                    "tooltip": "Minimum confidence. Lower finds more (and more mistakes), higher keeps only sure ones. "
                               "The report says how close a concept that was not found came.",
                }),
                "detail_level": (list(sam3.DETAIL_LEVELS), {
                    "default": "balanced",
                    "tooltip": "balanced: each instance is segmented again on a close-up cut from the full-resolution "
                               "image, so small and medium objects get crisp edges even on 4K-8K pictures (about "
                               "0.1 s per instance on a fast GPU). A subject filling most of the frame keeps edges "
                               "at about 1/300 of its size. fast: one look at the whole image only.",
                }),
                "max_per_concept": ("INT", {
                    "default": 0, "min": 0, "max": 200, "step": 1,
                    "tooltip": "Keep at most this many instances of each concept, best first. 0 keeps all.",
                }),
                "checkpoint": (sam3.checkpoint_choices(), {
                    "tooltip": "SAM 3 checkpoint in models/sam3 (shared with other SAM 3 packs). Download sam3.pt from "
                               "huggingface.co/facebook/sam3 after accepting Meta's license.",
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep SAM 3 on the GPU after the run (about 3.6 GB): faster when it runs again, but "
                               "ComfyUI cannot use that VRAM meanwhile, and its Unload Models button does not "
                               "release it (turn this off and run the node once, or restart ComfyUI).",
                }),
            },
            # optional, so prompts saved before it existed still validate
            "optional": {
                "show_preview": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show the mask (bright), the instances found (outlined and labelled) and the report on "
                               "the node itself. The node runs even with nothing connected.",
                }),
            },
        }

    RETURN_TYPES = (SEGS_TYPE, "MASK", "INT", "STRING")
    RETURN_NAMES = ("segs", "mask", "count", "report")
    OUTPUT_NODE = True  # it shows its own preview, so it runs even with nothing connected
    FUNCTION = "segment"
    CATEGORY = "NH-Nodes/Vision"
    DESCRIPTION = (
        "Type what you want and get its mask: 'hair', 'person - bag', 'background', 'tóc, váy'. SAM 3 (Meta) finds "
        "every instance at full resolution; the node shows them on itself with a report of what was and was not "
        "found. segs feed Segs Select / Compose / To Masks for finer picking. Needs the sam3 package, "
        "models/sam3/sam3.pt and a CUDA GPU."
    )
    SEARCH_ALIASES = ["sam3", "sam 3", "segment anything", "text segment", "concept segment", "object mask",
                      "detect objects", "instance segmentation", "background mask", "mask formula"]

    def segment(self, image, concepts, threshold, detail_level, max_per_concept, checkpoint, keep_model_loaded,
                show_preview=True):
        try:
            tree = parse_concepts(concepts)
        except EmptyExpressionError:
            raise ValueError("[NH-Nodes] SAM3 Segment: write at least one concept, e.g. 'person' or 'window'.") \
                from None
        except ExpressionError as exc:
            raise ValueError(f"[NH-Nodes] SAM3 Segment: {exc}") from None
        names, added = concepts_to_find(tree)
        if not names:
            raise ValueError("[NH-Nodes] SAM3 Segment: write at least one concept, e.g. 'person' or 'window'.")
        bar = ProgressBar(image.shape[0] * len(names)) if ProgressBar is not None else None
        misses = {}
        segs = sam3.segment(
            image, names, threshold, detail_level, max_per_concept, checkpoint, keep_model_loaded,
            interrupt=throw_exception_if_processing_interrupted,
            progress=(lambda done, total: bar.update_absolute(done, total)) if bar is not None else None,
            misses=misses,
        )
        segs = stamped(segs, image)
        count = count_segments(segs)
        lines = describe_request(tree, segs, threshold, misses, added)
        report = "\n".join([f"SAM3 Segment: {count} instance(s) in {len(segs)} image(s)", *lines])
        mask = expression_masks(tree, segs)
        result = (segs, mask, count, report)
        if not show_preview:
            return result
        try:
            pictures = [preview_picture(picture, frame, lines, mask=held) for picture, frame, held in
                        zip(image[:MAX_PREVIEWS], segs[:MAX_PREVIEWS], mask[:MAX_PREVIEWS])]
            images = save_temp_previews(pictures, "NH_sam3")
        except Exception as exc:  # noqa: BLE001 - a preview that cannot be drawn or saved must not lose the run
            print(f"[NH-Nodes] SAM3 Segment: could not show the preview ({exc}); the outputs are fine")
            return result
        return {"ui": {"images": images}, "result": result}


NODE_CLASS_MAPPINGS = {
    "NH_SAM3Segment": NH_SAM3Segment,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_SAM3Segment": "SAM3 Segment (NH)",
}
