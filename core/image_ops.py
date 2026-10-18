"""Pillow / tensor helpers shared by the image nodes (save, load, compare, label, resize, layer composite).

IMAGE tensors become PIL images with core.utils.tensor_to_pil (rgb=True, plus keep_alpha=True for layers);
``pil_to_image_tensor`` converts back in ComfyUI's intermediate dtype.
"""

from PIL import Image, ImageColor

import comfy.model_management

from .utils import parse_hex_color, pil_to_tensor

RESAMPLING = getattr(Image, "Resampling", Image)


def parse_color(value, default):
    """Parse a colour into an (r, g, b, a) tuple of 0-255 ints; ``default`` when the text is not a colour.

    '#RGB', '#RRGGBB' and '#RRGGBBAA' go through core.utils.parse_hex_color (opaque unless AA is given). Other CSS
    colours Pillow understands ('red', 'rgb(1, 2, 3)') keep working, as they did before.
    """
    text = str(value or "").strip()
    color = parse_hex_color(text, ())
    if color:
        return color
    try:
        return ImageColor.getcolor(text, "RGBA")
    except ValueError:
        return tuple(default) + (255,) * (4 - len(default))




def pil_to_image_tensor(image):
    """A PIL image as a [1, H, W, C] float IMAGE in ComfyUI's intermediate dtype."""
    return pil_to_tensor(image).to(dtype=comfy.model_management.intermediate_dtype())


def resize_pil(image, size, method):
    """``image`` resized to ``size`` (width, height) with the 'lanczos', 'bicubic', 'bilinear' or 'nearest' filter."""
    resample = {
        "lanczos": RESAMPLING.LANCZOS,
        "bicubic": RESAMPLING.BICUBIC,
        "bilinear": RESAMPLING.BILINEAR,
        "nearest": RESAMPLING.NEAREST,
    }[method]
    return image.resize(size, resample)
