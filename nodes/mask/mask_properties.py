from ...core.mask_ops import mask_bbox

# This file holds the logic of the Mask Properties node (original version).

class NH_MaskProperties:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT", "BBOX")
    RETURN_NAMES = ("width", "height", "x1", "y1", "x2", "y2", "bbox")
    FUNCTION = "get_properties"
    CATEGORY = "NH-Nodes/Mask"

    def get_properties(self, mask):
        # Only the first mask of the batch is processed; the box encloses the pixels > 0.5
        bbox = mask_bbox(mask[0].cpu().numpy())

        # Handle an empty mask (all black)
        if bbox is None:
            return (0, 0, 0, 0, 0, 0, (0, 0, 0, 0))

        x1, y1, x2, y2 = bbox

        # Compute the width and height
        width = x2 - x1 + 1
        height = y2 - y1 + 1

        # Return all values in the declared order
        return (width, height, x1, y1, x2, y2, bbox)

# Mapping that lets ComfyUI find the class
NODE_CLASS_MAPPINGS = {
    "NH_MaskProperties": NH_MaskProperties
}

# Name shown in the ComfyUI menu
NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_MaskProperties": "Mask Properties (NH)"
}
