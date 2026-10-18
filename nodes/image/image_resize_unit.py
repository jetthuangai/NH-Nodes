"""Image Resize Unit (NH): resize to a size given in pixels, millimetres or centimetres at a DPI."""

import torch

import comfy.utils

from ...core.image_ops import parse_color


def _image_resize_methods():
    return ["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"]


def _resize_value_to_pixels(value, unit, dpi):
    if unit == "pixel":
        return max(0, int(round(value)))
    divisor = 25.4 if unit == "mm" else 2.54
    return max(0, int(round((value / divisor) * dpi)))


def _resolve_target_size(orig_width, orig_height, width_value, height_value, unit, dpi):
    width_px = _resize_value_to_pixels(width_value, unit, dpi)
    height_px = _resize_value_to_pixels(height_value, unit, dpi)

    if width_px == 0 and height_px == 0:
        return orig_width, orig_height
    if width_px == 0:
        width_px = max(1, round(orig_width * height_px / orig_height))
    elif height_px == 0:
        height_px = max(1, round(orig_height * width_px / orig_width))
    return width_px, height_px


def _common_upscale_method(resize_method):
    return "nearest-exact" if resize_method == "nearest" else resize_method


def _pad_samples(samples, target_width, target_height, fill_rgb):
    """Centre ``samples`` ([B, C, H, W]) on a canvas of the padding colour.

    ``fill_rgb`` is (r, g, b) or (r, g, b, a) in 0-255. The fourth channel of RGBA images takes the alpha value, which
    is opaque when the colour has none.
    """
    batch, channels, resized_h, resized_w = samples.shape
    canvas = torch.empty((batch, channels, target_height, target_width), dtype=samples.dtype, device=samples.device)
    fill_values = [channel / 255.0 for channel in fill_rgb[:3]] + [fill_rgb[3] / 255.0 if len(fill_rgb) > 3 else 1.0]
    for channel in range(channels):
        canvas[:, channel, :, :] = fill_values[min(channel, 3)]

    offset_x = max(0, (target_width - resized_w) // 2)
    offset_y = max(0, (target_height - resized_h) // 2)
    canvas[:, :, offset_y:offset_y + resized_h, offset_x:offset_x + resized_w] = samples
    return canvas


class NH_ImageResizeByUnit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "unit": (["pixel", "mm", "cm"],),
                "width_value": ("FLOAT", {"default": 1024.0, "min": 0.0, "max": 100000.0, "step": 0.1}),
                "height_value": ("FLOAT", {"default": 1024.0, "min": 0.0, "max": 100000.0, "step": 0.1}),
                "dpi": ("INT", {"default": 300, "min": 1, "max": 2400, "step": 1}),
                "resize_mode": (["stretch", "pad", "crop", "lock_ratio"],),
                "resize_method": (_image_resize_methods(),),
                "pad_color_hex": ("STRING", {"default": "#000000", "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width_px", "height_px")
    FUNCTION = "resize"
    CATEGORY = "NH-Nodes/Image"

    def resize(self, image, unit, width_value, height_value, dpi, resize_mode, resize_method, pad_color_hex):
        samples = image.movedim(-1, 1)
        orig_height = samples.shape[-2]
        orig_width = samples.shape[-1]

        target_width, target_height = _resolve_target_size(
            orig_width, orig_height, width_value, height_value, unit, dpi
        )
        upscale_method = _common_upscale_method(resize_method)

        if resize_mode == "lock_ratio":
            if width_value <= 0 and height_value <= 0:
                result = samples
                actual_width, actual_height = orig_width, orig_height
            elif width_value <= 0 or height_value <= 0:
                actual_width, actual_height = target_width, target_height
                result = comfy.utils.common_upscale(samples, actual_width, actual_height, upscale_method, "disabled")
            else:
                scale = min(target_width / orig_width, target_height / orig_height)
                actual_width = max(1, round(orig_width * scale))
                actual_height = max(1, round(orig_height * scale))
                result = comfy.utils.common_upscale(samples, actual_width, actual_height, upscale_method, "disabled")
        elif resize_mode == "crop":
            actual_width, actual_height = target_width, target_height
            result = comfy.utils.common_upscale(samples, actual_width, actual_height, upscale_method, "center")
        elif resize_mode == "pad":
            actual_width, actual_height = target_width, target_height
            scale = min(target_width / orig_width, target_height / orig_height)
            scaled_width = max(1, round(orig_width * scale))
            scaled_height = max(1, round(orig_height * scale))
            resized = comfy.utils.common_upscale(samples, scaled_width, scaled_height, upscale_method, "disabled")
            pad_color = parse_color(pad_color_hex, (0, 0, 0))
            result = _pad_samples(resized, target_width, target_height, pad_color)
        else:
            actual_width, actual_height = target_width, target_height
            result = comfy.utils.common_upscale(samples, actual_width, actual_height, upscale_method, "disabled")

        return (result.movedim(1, -1), actual_width, actual_height)


NODE_CLASS_MAPPINGS = {
    "NH_ImageResizeByUnit": NH_ImageResizeByUnit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_ImageResizeByUnit": "Image Resize Unit (NH)",
}
