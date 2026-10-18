"""SAM3 Segment (NH): every instance of the concepts you name, as NH_SEGS plus their union mask.

The model work lives in core/sam3_backend.py; this module only maps the node's inputs and outputs.
"""

from ...core import sam3_backend as sam3
from ...core.segs import SEGS_TYPE, count_segments, parse_label_list, segs_union

try:
    from comfy.model_management import throw_exception_if_processing_interrupted
except Exception:  # outside ComfyUI (tests)
    def throw_exception_if_processing_interrupted():
        return None

try:
    from comfy.utils import ProgressBar
except Exception:  # outside ComfyUI (tests)
    ProgressBar = None


class NH_SAM3Segment:
    """Find every instance of each concept with SAM 3 (Meta), at the full resolution of the image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "concepts": ("STRING", {
                    "default": "person", "multiline": True,
                    "tooltip": "What to find: short English noun phrases, one per line or separated by commas "
                               "(person, red car, window, sky). Each phrase becomes the label of its instances.",
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.05, "max": 0.95, "step": 0.01,
                    "tooltip": "Minimum confidence. Lower finds more (and more mistakes), higher keeps only sure ones.",
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
                               "ComfyUI cannot use that VRAM meanwhile.",
                }),
            }
        }

    RETURN_TYPES = (SEGS_TYPE, "MASK", "INT")
    RETURN_NAMES = ("segs", "mask", "count")
    FUNCTION = "segment"
    CATEGORY = "NH-Nodes/Vision"
    DESCRIPTION = (
        "Find every instance of the concepts you name (people, garments, objects, sky, walls, windows...) with "
        "Meta's SAM 3. Outputs the instances (segs) for Segs Select / Compose / To Masks / Preview, their union mask "
        "and their number. Needs the sam3 package, models/sam3/sam3.pt and a CUDA GPU."
    )
    SEARCH_ALIASES = ["sam3", "sam 3", "segment anything", "text segment", "concept segment", "object mask",
                      "detect objects", "instance segmentation"]

    def segment(self, image, concepts, threshold, detail_level, max_per_concept, checkpoint, keep_model_loaded):
        names = parse_label_list(concepts)
        if not names:
            raise ValueError("[NH-Nodes] SAM3 Segment: write at least one concept, e.g. 'person' or 'window'.")
        bar = ProgressBar(image.shape[0] * len(names)) if ProgressBar is not None else None
        segs = sam3.segment(
            image, names, threshold, detail_level, max_per_concept, checkpoint, keep_model_loaded,
            interrupt=throw_exception_if_processing_interrupted,
            progress=(lambda done, total: bar.update_absolute(done, total)) if bar is not None else None,
        )
        return segs, segs_union(segs), count_segments(segs)


NODE_CLASS_MAPPINGS = {
    "NH_SAM3Segment": NH_SAM3Segment,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_SAM3Segment": "SAM3 Segment (NH)",
}
