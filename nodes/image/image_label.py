"""Image Label (NH): a text band above, below or over an image."""

import functools
import os
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from ...core.utils import tensor_to_pil
from ...core.image_ops import RESAMPLING, parse_color, pil_to_image_tensor


_FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}


def _font_choices():
    fonts = ["default"]
    seen = set(fonts)
    font_dir = Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts"
    if font_dir.exists():
        for path in sorted(font_dir.iterdir(), key=lambda item: item.name.lower()):
            if path.is_file() and path.suffix.lower() in _FONT_EXTENSIONS and path.name not in seen:
                fonts.append(path.name)
                seen.add(path.name)
    return fonts


_FONT_CHOICES = _font_choices()


@functools.lru_cache(maxsize=64)
def _cached_truetype(path, size):
    """Open a TrueType font once per (path, size), as the Label node asks for many sizes. None if it cannot be opened."""
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _default_font():
    return ImageFont.load_default()


def _load_font(font_name, font_size):
    if font_name and font_name != "default":
        font_path = Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts" / font_name
        if font_path.exists():
            font = _cached_truetype(str(font_path), font_size)
            if font is not None:
                return font
    font = _cached_truetype("arial.ttf", font_size)
    return font if font is not None else _default_font()


_MIN_FIT_FONT_SIZE = 8


def _fit_font_size(font_size, fits):
    """Largest size in [8, font_size] for which ``fits(size)`` is true; 8 when none does.

    A label gets wider as the font grows, so a binary search lands on the same size as shrinking one point at a
    time, after about log2(font_size) measurements instead of up to font_size. Hinting can make a text a pixel or
    two narrower at a few isolated sizes; then the size returned still fits but may sit a point or two below what
    the one-point-at-a-time loop would have found.
    """
    if font_size <= _MIN_FIT_FONT_SIZE or fits(font_size):
        return font_size
    low, high = _MIN_FIT_FONT_SIZE, font_size  # `low` fits (or is the floor), `high` does not
    while high - low > 1:
        middle = (low + high) // 2
        if fits(middle):
            low = middle
        else:
            high = middle
    return low


def _label_text_width(draw, text, font_name, font_size, stroke_width):
    left, _top, right, _bottom = draw.multiline_textbbox(
        (0, 0), text, font=_load_font(font_name, font_size), stroke_width=stroke_width
    )
    return right - left


class NH_ImageLabel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "label_text": ("STRING", {"default": "NH Label", "multiline": True}),
                "position": (["header", "bottom"],),
                "frame_height": ("INT", {"default": 80, "min": 1, "max": 4096, "step": 1}),
                "fill_enabled": ("BOOLEAN", {"default": True}),
                "frame_color_hex": ("STRING", {"default": "#111111", "multiline": False}),
                "text_color_hex": ("STRING", {"default": "#FFFFFF", "multiline": False}),
                "font_name": (_FONT_CHOICES,),
                "font_size": ("INT", {"default": 32, "min": 6, "max": 512, "step": 1}),
                "fit_text": ("BOOLEAN", {"default": True}),
                "overlay_on_image": ("BOOLEAN", {"default": False}),
                "align": (["left", "center", "right"],),
                "padding_x": ("INT", {"default": 20, "min": 0, "max": 2048, "step": 1}),
                "outline_width": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1}),
                "outline_color_hex": ("STRING", {"default": "#000000", "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "add_label"
    CATEGORY = "NH-Nodes/Image"

    def _band_image(self, image, position, frame_height, fill_enabled, frame_color):
        width, height = image.size
        if fill_enabled:
            return Image.new("RGB", (width, frame_height), frame_color)

        sample_box = (0, 0, width, 1) if position == "header" else (0, max(0, height - 1), width, height)
        return image.crop(sample_box).resize((width, frame_height), RESAMPLING.BILINEAR).convert("RGB")

    def add_label(
        self,
        image,
        label_text,
        position,
        frame_height,
        fill_enabled,
        frame_color_hex,
        text_color_hex,
        font_name,
        font_size,
        fit_text,
        overlay_on_image,
        align,
        padding_x,
        outline_width,
        outline_color_hex,
    ):
        frame_color = parse_color(frame_color_hex, (17, 17, 17))
        text_color = parse_color(text_color_hex, (255, 255, 255))
        outline_color = parse_color(outline_color_hex, (0, 0, 0))

        labeled = []
        for item in image:
            pil_image = tensor_to_pil(item, rgb=True)
            width, height = pil_image.size
            band = self._band_image(pil_image, position, frame_height, fill_enabled, frame_color)

            if overlay_on_image:
                canvas = pil_image.copy()
                max_band_height = min(frame_height, height)
                if band.height != max_band_height:
                    band = band.resize((width, max_band_height), RESAMPLING.BILINEAR)
                if position == "header":
                    band_top = 0
                else:
                    band_top = height - max_band_height
                canvas.paste(band, (0, band_top))
            else:
                canvas = Image.new("RGB", (width, height + frame_height))
                if position == "header":
                    canvas.paste(band, (0, 0))
                    canvas.paste(pil_image, (0, frame_height))
                    band_top = 0
                else:
                    canvas.paste(pil_image, (0, 0))
                    canvas.paste(band, (0, height))
                    band_top = height

            draw = ImageDraw.Draw(canvas)
            current_size = font_size

            if fit_text:
                max_width = max(1, width - (padding_x * 2))
                current_size = _fit_font_size(
                    font_size,
                    lambda size: _label_text_width(draw, label_text, font_name, size, outline_width) <= max_width,
                )

            font = _load_font(font_name, current_size)
            bbox = draw.multiline_textbbox((0, 0), label_text, font=font, stroke_width=outline_width)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            if align == "left":
                text_x = padding_x
            elif align == "right":
                text_x = max(padding_x, width - padding_x - text_width)
            else:
                text_x = max(0, (width - text_width) // 2)

            text_y = band_top + max(0, (frame_height - text_height) // 2) - bbox[1]
            draw.multiline_text(
                (text_x, text_y),
                label_text,
                font=font,
                fill=text_color,
                align=align,
                stroke_width=outline_width,
                stroke_fill=outline_color,
            )
            labeled.append(pil_to_image_tensor(canvas))

        return (torch.cat(labeled, dim=0),)


NODE_CLASS_MAPPINGS = {
    "NH_ImageLabel": NH_ImageLabel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_ImageLabel": "Image Label (NH)",
}
