"""NH_SEGS nodes: pick instances, combine them into masks, preview them and hand them to Impact Pack.

All of them work on the instances SAM3 Segment (NH) finds; the logic lives in core/segs.py.
"""

from ...core.segs import (
    SEGS_TYPE,
    SORT_ORDERS,
    merge_segs,
    count_segments,
    expression_labels,
    expression_masks,
    instance_masks,
    label_matches,
    parse_expression,
    parse_label_list,
    preview_batch,
    segs_union,
    select_segments,
    to_impact_segs,
)
from ...core.vi_words import translate

CATEGORY = "NH-Nodes/Vision"


class NH_SegsSelect:
    """Keep the instances that match labels, score and size, in a chosen order."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": (SEGS_TYPE,),
                "labels": ("STRING", {
                    "default": "",
                    "tooltip": "Labels to keep, separated by commas; * and ? are wildcards (\"*shirt\"). Common "
                               "Vietnamese words work too (tóc, túi). Empty keeps every label.",
                }),
                "min_score": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Drop instances with a lower confidence."}),
                "min_area": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.0001,
                                       "tooltip": "Drop instances smaller than this fraction of the image (0.01 = 1%)."}),
                "max_area": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.0001,
                                       "tooltip": "Drop instances larger than this fraction of the image."}),
                "sort_by": (list(SORT_ORDERS), {
                    "default": "score",
                    "tooltip": "Order of the kept instances: highest score, largest area, by position (box centre), "
                               "nearest the image centre first, or as found.",
                }),
                "start": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1,
                                  "tooltip": "Skip this many instances of the sorted list (0 starts at the first)."}),
                "count": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1,
                                  "tooltip": "Keep this many instances from start on; 0 keeps all of them."}),
            }
        }

    RETURN_TYPES = (SEGS_TYPE, "MASK", "INT")
    RETURN_NAMES = ("segs", "mask", "count")
    FUNCTION = "select"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Filter and order instances, per image: by label (wildcards allowed), score and size, then sort and take a "
        "slice. Example: the second person from the left = labels 'person', sort_by left_to_right, start 1, count 1."
    )
    SEARCH_ALIASES = ["segs select", "filter segs", "pick instance", "largest object", "leftmost person"]

    def select(self, segs, labels, min_score, min_area, max_area, sort_by, start, count):
        patterns = []
        for pattern in parse_label_list(labels):
            english, status = translate(pattern)
            # "nền" / "tất cả" are formula words, not labels: Select matches labels only
            patterns.append(english if status == "translated" and english not in ("background", "all", "image")
                            else pattern)
        picked = tuple(
            select_segments(frame, patterns, min_score, min_area, max_area, sort_by, start, count) for frame in segs
        )
        return picked, segs_union(picked), count_segments(picked)


class NH_SegsCompose:
    """Build a mask from instance labels with + - & ~ and parentheses."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "segs_2": (SEGS_TYPE, {"tooltip": "More instances of the same pictures (from another SAM3 Segment, "
                                                  "Garment, Portrait or Body Segment), mixed into the formula."}),
                "segs_3": (SEGS_TYPE, {"tooltip": "And more."}),
            },
            "required": {
                "segs": (SEGS_TYPE,),
                "expression": ("STRING", {
                    "default": "all", "multiline": True,
                    "tooltip": "Labels combined with | or + (union), - (minus), & (overlap), ~ (everything except) "
                               "and parentheses; commas or new lines list several. all = every instance, background = "
                               "everything outside them, image = whole frame, none = nothing. Words in a row are one "
                               "label (red car); t-shirt needs no quotes; Vietnamese works (tóc, nền). Put spaces "
                               "around a minus to subtract. Examples: person - hair · background · "
                               "(wall | ceiling) - window.",
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "compose"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Turn instances into one mask with a small formula over their labels: | or + (or a comma) joins, - removes, "
        "& keeps the overlap, ~ inverts, background is everything outside them. (building | roof) - window keeps the "
        "facade. SAM3 Segment takes the same formula in its concepts, so this node is for mixing or re-using segs."
    )
    SEARCH_ALIASES = ["segs compose", "mask formula", "combine masks", "subtract mask", "background mask"]

    def compose(self, segs, expression, segs_2=None, segs_3=None):
        segs = merge_segs(segs, segs_2, segs_3)
        tree = parse_expression(expression)
        labels = [seg.label for frame in segs for seg in frame.segments]

        def found(pattern):
            return any(label_matches(label, pattern) for label in labels)

        unmatched = [pattern for pattern in expression_labels(tree) if not found(pattern)]
        for pattern in unmatched:
            words = pattern.split()
            if len(words) > 1 and all(found(word) for word in words):
                # words in a row make one label; with each of them found, an operator was left out
                raise ValueError(f"[NH-Nodes] Segs Compose: no instance is labelled '{pattern}', but "
                                 f"{' and '.join(repr(word) for word in words)} are. Join them with + or a comma: "
                                 f"{' + '.join(words)}")
            parts = pattern.split("-")
            if len(parts) > 1 and all(part and found(part) for part in parts):
                # "person-hair" is one label; with person and hair both found, a subtraction was meant
                raise ValueError(f"[NH-Nodes] Segs Compose: no instance is labelled '{pattern}', but "
                                 f"{' and '.join(repr(part) for part in parts)} are. To subtract, put spaces around "
                                 f"the minus: {' - '.join(parts)}")
        if unmatched:
            print(f"[NH-Nodes] Segs Compose: no instance is labelled {', '.join(repr(p) for p in unmatched)}; "
                  "those parts are empty")
        return (expression_masks(tree, segs),)


class NH_SegsToMasks:
    """One full-size mask per instance, plus the union and the labels."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"segs": (SEGS_TYPE,)}}

    RETURN_TYPES = ("MASK", "MASK", "STRING", "INT")
    RETURN_NAMES = ("masks", "union", "labels", "count")
    FUNCTION = "convert"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "masks: one mask per instance, image after image (a single empty mask when there is none). union: one mask "
        "per image. labels: the instance labels, one per line, in the order of masks."
    )
    SEARCH_ALIASES = ["segs to masks", "instance masks", "segs to mask batch"]

    def convert(self, segs):
        masks, labels = instance_masks(segs)
        return masks, segs_union(segs), "\n".join(labels), len(labels)


class NH_SegsToImpactSEGS:
    """Hand the instances of one image to Impact Pack detailers as its SEGS type."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": (SEGS_TYPE,),
                "crop_factor": ("FLOAT", {
                    "default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Size of the region a detailer crops around each instance, as a multiple of its box "
                               "(Impact Pack's crop_factor).",
                }),
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 4095, "step": 1,
                                        "tooltip": "Impact SEGS hold one image: which image of the batch to convert."}),
            }
        }

    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("segs",)
    FUNCTION = "convert"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Convert the instances of one image to Impact Pack's SEGS, for its detailers and SEGS tools. Impact Pack is "
        "not needed for anything else in NH-Nodes."
    )
    SEARCH_ALIASES = ["segs to impact", "impact segs", "detailer segs"]

    def convert(self, segs, crop_factor, batch_index):
        if batch_index >= len(segs):
            raise ValueError(f"[NH-Nodes] Segs To Impact SEGS: batch_index {batch_index} but the segs hold "
                             f"{len(segs)} image(s)")
        return (to_impact_segs(segs[batch_index], crop_factor),)


class NH_SegsPreview:
    """Draw the instances on their image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "segs": (SEGS_TYPE,)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "preview"
    CATEGORY = CATEGORY
    DESCRIPTION = "Each instance tinted in its own colour, outlined and labelled with its label and score."
    SEARCH_ALIASES = ["segs preview", "show segs", "draw instances"]

    def preview(self, image, segs):
        return (preview_batch(image, segs),)


NODE_CLASS_MAPPINGS = {
    "NH_SegsSelect": NH_SegsSelect,
    "NH_SegsCompose": NH_SegsCompose,
    "NH_SegsToMasks": NH_SegsToMasks,
    "NH_SegsToImpactSEGS": NH_SegsToImpactSEGS,
    "NH_SegsPreview": NH_SegsPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_SegsSelect": "Segs Select (NH)",
    "NH_SegsCompose": "Segs Compose (NH)",
    "NH_SegsToMasks": "Segs To Masks (NH)",
    "NH_SegsToImpactSEGS": "Segs To Impact SEGS (NH)",
    "NH_SegsPreview": "Segs Preview (NH)",
}
