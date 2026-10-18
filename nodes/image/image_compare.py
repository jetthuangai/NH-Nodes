"""Image Compare (NH): two images side by side, split, blended or as a difference."""

import torch
from PIL import Image, ImageChops

from ...core.utils import tensor_to_pil
from ...core.image_ops import pil_to_image_tensor, resize_pil


def _match_batch_sizes(batch_a, batch_b):
    count_a = batch_a.shape[0]
    count_b = batch_b.shape[0]

    if count_a == count_b:
        return batch_a, batch_b
    if count_a == 1:
        return batch_a.repeat(count_b, 1, 1, 1), batch_b
    if count_b == 1:
        return batch_a, batch_b.repeat(count_a, 1, 1, 1)

    shared = min(count_a, count_b)
    print(f"[NH-Nodes] Image Compare: batch mismatch ({count_a} vs {count_b}), truncating to {shared}")
    return batch_a[:shared], batch_b[:shared]


class NH_ImageCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "compare_mode": (
                    [
                        "side_by_side",
                        "top_bottom",
                        "split_vertical",
                        "split_horizontal",
                        "difference",
                        "overlay",
                    ],
                ),
                "split_position": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "overlay_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "resize_method": (["lanczos", "bicubic", "bilinear", "nearest"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "compare"
    CATEGORY = "NH-Nodes/Image"

    def compare(self, image_a, image_b, compare_mode, split_position, overlay_opacity, resize_method):
        batch_a, batch_b = _match_batch_sizes(image_a, image_b)
        compared = []

        for tensor_a, tensor_b in zip(batch_a, batch_b):
            pil_a = tensor_to_pil(tensor_a, rgb=True)
            pil_b = resize_pil(tensor_to_pil(tensor_b, rgb=True), pil_a.size, resize_method)
            width, height = pil_a.size

            if compare_mode == "side_by_side":
                canvas = Image.new("RGB", (width * 2, height))
                canvas.paste(pil_a, (0, 0))
                canvas.paste(pil_b, (width, 0))
            elif compare_mode == "top_bottom":
                canvas = Image.new("RGB", (width, height * 2))
                canvas.paste(pil_a, (0, 0))
                canvas.paste(pil_b, (0, height))
            elif compare_mode == "split_vertical":
                split_x = max(0, min(width, int(round(width * split_position))))
                canvas = pil_b.copy()
                canvas.paste(pil_a.crop((0, 0, split_x, height)), (0, 0))
            elif compare_mode == "split_horizontal":
                split_y = max(0, min(height, int(round(height * split_position))))
                canvas = pil_b.copy()
                canvas.paste(pil_a.crop((0, 0, width, split_y)), (0, 0))
            elif compare_mode == "difference":
                canvas = ImageChops.difference(pil_a, pil_b)
            else:
                canvas = Image.blend(pil_a, pil_b, overlay_opacity)

            compared.append(pil_to_image_tensor(canvas))

        return (torch.cat(compared, dim=0),)


NODE_CLASS_MAPPINGS = {
    "NH_ImageCompare": NH_ImageCompare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_ImageCompare": "Image Compare (NH)",
}
