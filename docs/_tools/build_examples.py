"""Build the example pictures of the NH-Nodes guides by running the real node code on open-licensed sample photos.

Run with ComfyUI's Python from the ComfyUI root:
    python custom_nodes/NH-Nodes/docs/_tools/build_examples.py [assets_dir] [--only a.jpg,b.jpg] [--gpu]
It writes <assets_dir>/examples/*.jpg and <assets_dir>/banner.jpg (default assets_dir: docs/assets of the pack).

Every picture is made from the live node classes, called the way ComfyUI calls them, so the examples cannot drift
from the code: rerun the script after a node changes. The photos are the public-domain / CC0 sample images that ship
with scikit-image (astronaut, coffee, chelsea, rocket). The script is deterministic and runs on the CPU only: the GPU
is hidden before torch loads, so a ComfyUI server that is running at the same time is never disturbed. The one
exception is the SAM 3 example, which needs a CUDA GPU: it is built only with --gpu, which leaves the GPU visible (SAM 3
goes back to RAM after its run). An example whose models are missing is skipped (and reported); nothing is downloaded.
"""

import argparse
import functools
import importlib
import importlib.util
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

GPU = "--gpu" in sys.argv[1:]  # read before torch loads: only the SAM 3 example needs the GPU
if not GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only. (An empty value would be dropped by Windows, so use -1.)
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _comfy_root():
    for base in (Path.cwd(), *Path(__file__).resolve().parents):
        if (base / "comfy").is_dir() and (base / "custom_nodes").is_dir():
            return base
    raise SystemExit("Run this from the ComfyUI root: python custom_nodes/NH-Nodes/docs/_tools/build_examples.py")


COMFY_ROOT = _comfy_root()
sys.path[:0] = [str(COMFY_ROOT), str(COMFY_ROOT / "custom_nodes")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont  # noqa: E402

from comfy.cli_args import args  # noqa: E402

args.cpu = not GPU  # before the pack imports comfy.model_management
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))  # leave the rest of the machine to its other users

PACK_DIR = COMFY_ROOT / "custom_nodes" / "NH-Nodes"
LANCZOS = Image.Resampling.LANCZOS

# ---------------------------------------------------------------------------------------------------------------
# Look
# ---------------------------------------------------------------------------------------------------------------
PANEL_W = 1200
BG = (29, 31, 35)  # #1d1f23, reads well on GitHub's light and dark themes
TEXT = (234, 236, 239)
TEXT_DIM = (139, 146, 156)
BORDER = (63, 67, 76)
ARROW_COLOR = (112, 119, 131)
TEAL = (45, 212, 191)
PINK = (244, 114, 182)
AMBER = (251, 191, 36)
BLUE = (96, 165, 250)
GREEN = (74, 222, 128)
VIOLET = (167, 139, 250)
CORAL = (251, 113, 133)
SKY = (56, 189, 248)
ACCENT = TEAL
RADIUS = 10

MARGIN = 28
ROW_GAP = 30
TILE_GAP = 20
ARROW_W = 40
JPEG_QUALITY = 85
JPEG_MAX_KB = 200

_FONT_FILES = {
    "semibold": ("Inter-SemiBold.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"),
    "medium": ("Inter-Medium.ttf", "segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"),
    "regular": ("Inter-Regular.ttf", "segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"),
}


class Skip(Exception):
    """An example that cannot be built on this machine (missing models or libraries)."""


# ---------------------------------------------------------------------------------------------------------------
# Fonts and drawing helpers
# ---------------------------------------------------------------------------------------------------------------
def _font_dirs():
    dirs = [Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    return [d for d in dirs if d.is_dir()]


@functools.lru_cache(maxsize=None)
def font(weight, size):
    """Inter first (the look of the guides), then Segoe UI / Arial / DejaVu, then Pillow's built-in font."""
    for name in _FONT_FILES[weight]:
        for folder in _font_dirs():
            path = folder / name
            if path.is_file():
                try:
                    return ImageFont.truetype(str(path), size)
                except OSError:
                    pass
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def text_width(text, weight, size):
    return font(weight, size).getlength(text)


def fit_font(text, weight, size, max_width, min_size=10):
    """The largest size <= `size` at which `text` fits in `max_width` (the minimum size otherwise)."""
    while size > min_size and text_width(text, weight, size) > max_width:
        size -= 0.5
    return size


def draw_text(draw, xy, text, weight, size, fill, anchor="la"):
    draw.text(xy, text, font=font(weight, size), fill=fill, anchor=anchor)


@functools.lru_cache(maxsize=None)
def _rounded_mask(width, height, radius, ring):
    """Anti-aliased rounded-rectangle mask (ring > 0: only a ring of that many pixels along the edge)."""
    ss = 4
    mask = Image.new("L", (width * ss, height * ss), 0)
    box = (0, 0, width * ss - 1, height * ss - 1)
    if ring:
        ImageDraw.Draw(mask).rounded_rectangle(box, radius=radius * ss, outline=255, width=ring * ss)
    else:
        ImageDraw.Draw(mask).rounded_rectangle(box, radius=radius * ss, fill=255)
    return mask.resize((width, height), LANCZOS)


def paste_rounded(canvas, image, xy, radius=RADIUS, border=BORDER):
    """Paste `image` (RGB or RGBA) with rounded corners and a hairline border."""
    width, height = image.size
    mask = _rounded_mask(width, height, radius, 0)
    if image.mode == "RGBA":
        mask = ImageChops.multiply(mask, image.getchannel("A"))
        image = image.convert("RGB")
    canvas.paste(image, xy, mask)
    if border:
        canvas.paste(Image.new("RGB", (width, height), border), xy, _rounded_mask(width, height, radius, 1))


@functools.lru_cache(maxsize=None)
def _arrow_mask(width, height):
    ss = 8
    big = Image.new("L", (width * ss, height * ss), 0)
    draw = ImageDraw.Draw(big)
    thick = int(2.6 * ss)
    x0, x1, cy = 0.16 * width * ss, 0.84 * width * ss, 0.5 * height * ss
    head = 0.17 * width * ss
    draw.line([(x0, cy), (x1, cy)], fill=255, width=thick)
    draw.line([(x1 - head, cy - head), (x1, cy), (x1 - head, cy + head)], fill=255, width=thick, joint="curve")
    for px, py in ((x0, cy), (x1, cy), (x1 - head, cy - head), (x1 - head, cy + head)):
        draw.ellipse((px - thick / 2, py - thick / 2, px + thick / 2, py + thick / 2), fill=255)
    return big.resize((width, height), LANCZOS)


def paste_arrow(canvas, center, color=ARROW_COLOR, size=(ARROW_W - 8, 26)):
    width, height = size
    canvas.paste(Image.new("RGB", size, color), (int(center[0] - width / 2), int(center[1] - height / 2)),
                 _arrow_mask(width, height))


@functools.lru_cache(maxsize=None)
def _checker(width, height, cell):
    ys, xs = np.mgrid[0:height, 0:width]
    light = ((xs // cell + ys // cell) % 2).astype(bool)
    arr = np.where(light[..., None], np.array((84, 89, 98), np.uint8), np.array((62, 66, 74), np.uint8))
    return Image.fromarray(arr.astype(np.uint8))


def over_checker(image, cell=12):
    """Composite an RGBA image on a dark checkerboard so that its transparency is visible."""
    base = _checker(image.width, image.height, cell).copy()
    base.paste(image.convert("RGB"), (0, 0), image.getchannel("A"))
    return base


# ---------------------------------------------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------------------------------------------
def to_pil(tensor, index=0):
    """One item of a ComfyUI IMAGE ([B, H, W, C], 0..1; 3 or 4 channels) as an 8-bit PIL image."""
    array = tensor.detach().cpu().float().numpy()
    if array.ndim == 4:
        array = array[index]
    array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(array)


def to_tensor(pil):
    """A PIL image as a ComfyUI IMAGE tensor [1, H, W, C] (float32, 0..1)."""
    return torch.from_numpy(np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0).unsqueeze(0)


def mask_array(tensor, index=0):
    """A ComfyUI MASK ([B, H, W] or [H, W]) as a float32 numpy array [H, W]."""
    array = tensor.detach().cpu().float().numpy()
    return array[index] if array.ndim == 3 else array


def center_crop_ratio(image, ratio, anchor=(0.5, 0.5)):
    """Crop of a PIL image to width / height == ratio, placed by `anchor` (0..1 per axis): input preparation only."""
    width, height = image.size
    if width / height > ratio:
        new_w = round(height * ratio)
        left = round((width - new_w) * anchor[0])
        return image.crop((left, 0, left + new_w, height))
    new_h = round(width / ratio)
    top = round((height - new_h) * anchor[1])
    return image.crop((0, top, width, top + new_h))


_COMMON_RATIOS = ((1, 1), (4, 5), (3, 4), (2, 3), (9, 16), (9, 21), (5, 4), (4, 3), (3, 2), (16, 9), (21, 9))


def ratio_text(width, height):
    """'16:9' for a size close to a common ratio (within 1.5 %), otherwise the reduced ratio or a decimal."""
    value = width / height
    for a, b in _COMMON_RATIOS:
        if abs(value / (a / b) - 1.0) <= 0.015:
            return f"{a}:{b}"
    divisor = math.gcd(int(width), int(height))
    a, b = int(width) // divisor, int(height) // divisor
    return f"{a}:{b}" if max(a, b) <= 32 else f"{value:.2f}:1"


@dataclass
class Layer:
    """One mask drawn over a photo: a translucent colour fill (alpha) and an outline of `outline` pixels."""

    mask: np.ndarray  # float [H, W], 0..1, any size
    color: tuple = TEAL
    alpha: float = 0.42
    outline: float = 2.0


def overlay_masks(photo, layers, size=None, ss=2):
    """Show masks on a photo, drawn at the final size (x ss, then reduced) so lines stay crisp at any tile size."""
    width, height = size if size else photo.size
    base = np.asarray(photo.convert("RGB").resize((width * ss, height * ss), LANCZOS), dtype=np.float32)
    outlines = []
    for layer in layers:
        mask = np.asarray(layer.mask, dtype=np.float32)
        soft = cv2.resize(mask, (width * ss, height * ss),
                          interpolation=cv2.INTER_AREA if mask.shape[1] > width * ss else cv2.INTER_LINEAR)
        soft = np.clip(soft, 0.0, 1.0)
        if layer.alpha > 0:
            weight = (soft * layer.alpha)[..., None]
            base = base * (1.0 - weight) + np.asarray(layer.color, dtype=np.float32) * weight
        if layer.outline > 0:
            outlines.append((soft, layer))
    out = np.clip(base, 0, 255).astype(np.uint8)
    for soft, layer in outlines:
        contours, _ = cv2.findContours((soft >= 0.5).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, tuple(int(c) for c in layer.color), max(1, round(layer.outline * ss)),
                         cv2.LINE_AA)
    result = Image.fromarray(out)
    return result.resize((width, height), LANCZOS) if ss != 1 else result


def masked(photo, *layers):
    """A tile image that draws `layers` over `photo` at whatever size the row gives it (see Tile.image)."""
    return lambda size: overlay_masks(photo, layers, size=size)


# ---------------------------------------------------------------------------------------------------------------
# Panels: rows of captioned tiles on a dark background
# ---------------------------------------------------------------------------------------------------------------
ARROW = "arrow"


@dataclass
class Tile:
    """One picture with a caption. `image` is a PIL image, or a function (width, height) -> PIL image that draws
    itself at the size the row gives it (then `aspect` = width / height is required)."""

    image: object
    caption: str = ""
    sub: str = ""
    frame: bool = True
    aspect: float = 0.0
    legend: tuple = ()  # ((label, colour), ...) drawn as dots under the caption

    @property
    def ratio(self):
        return self.aspect or self.image.width / self.image.height

    def picture(self, width, height):
        if callable(self.image):
            return self.image((width, height))
        return self.image if self.image.size == (width, height) else self.image.resize((width, height), LANCZOS)


@dataclass
class Row:
    items: list
    title: str = ""
    sub: str = ""
    max_h: int = 420


LEGEND_SIZE = 12.5
LEGEND_LINE = 22


def legend_lines(items, max_width):
    """Split legend entries into lines that fit `max_width`: [[(label, colour, width), ...], ...]."""
    lines, current, used = [], [], 0.0
    for label, color in items:
        entry_w = 15 + text_width(label, "regular", LEGEND_SIZE) + 14
        if current and used + entry_w - 14 > max_width:
            lines.append(current)
            current, used = [], 0.0
        current.append((label, color, entry_w))
        used += entry_w
    if current:
        lines.append(current)
    return lines


def render_row(row, width):
    """Lay the row out at one common tile height (tiles keep their aspect ratio) and draw captions under the tiles."""
    seq, previous_tile = [], False
    for item in row.items:
        if isinstance(item, Tile):
            if previous_tile:
                seq.append(("gap", TILE_GAP))
            seq.append(("tile", item))
            previous_tile = True
        else:
            seq.append(("arrow", ARROW_W))
            previous_tile = False
    fixed = sum(v for kind, v in seq if kind in ("gap", "arrow"))
    aspects = [t.ratio for kind, t in seq if kind == "tile"]
    height = min(row.max_h, int((width - fixed) / sum(aspects)))
    while True:
        widths = [max(1, round(a * height)) for a in aspects]
        if sum(widths) + fixed <= width or height <= 1:
            break
        height -= 1
    total = sum(widths) + fixed

    title_h = 40 if row.title else 0
    cap_h, tile_index = 0, 0
    for kind, t in seq:
        if kind != "tile":
            continue
        block = 12 + (21 if t.caption else 0) + (18 if t.sub else 0)
        if t.legend:
            block += 8 + LEGEND_LINE * len(legend_lines(t.legend, widths[tile_index] + TILE_GAP - 4))
        cap_h = max(cap_h, block)
        tile_index += 1
    canvas = Image.new("RGB", (width, title_h + height + cap_h), BG)
    draw = ImageDraw.Draw(canvas)

    if row.title:
        draw.rounded_rectangle((0, 3, 4, 21), radius=2, fill=ACCENT)
        draw_text(draw, (14, 0), row.title, "semibold", 17.5, TEXT)
        if row.sub:
            title_w = text_width(row.title, "semibold", 17.5)
            draw_text(draw, (14 + title_w + 14, 2.5), row.sub, "regular", 14, TEXT_DIM)

    x, tile_index = (width - total) // 2, 0
    for kind, value in seq:
        if kind == "gap":
            x += value
        elif kind == "arrow":
            paste_arrow(canvas, (x + value / 2, title_h + height / 2))
            x += value
        else:
            tile_w = widths[tile_index]
            tile_index += 1
            image = value.picture(tile_w, height)
            if value.frame:
                paste_rounded(canvas, image, (x, title_h))
            elif image.mode == "RGBA":
                canvas.paste(image.convert("RGB"), (x, title_h), image.getchannel("A"))
            else:
                canvas.paste(image, (x, title_h))
            room = tile_w + TILE_GAP - 4
            cy = title_h + height + 12
            if value.caption:
                size = fit_font(value.caption, "semibold", 15.5, room)
                draw_text(draw, (x + tile_w / 2, cy), value.caption, "semibold", size, TEXT, anchor="ma")
                cy += 21
            if value.sub:
                size = fit_font(value.sub, "regular", 12.5, room, min_size=10.5)
                draw_text(draw, (x + tile_w / 2, cy), value.sub, "regular", size, TEXT_DIM, anchor="ma")
                cy += 18
            if value.legend:
                cy += 8
                for line in legend_lines(value.legend, room):
                    cursor = x + tile_w / 2 - (sum(w for _l, _c, w in line) - 14) / 2
                    for label, color, entry_w in line:
                        draw.ellipse((cursor, cy + 4, cursor + 9, cy + 13), fill=color)
                        draw_text(draw, (cursor + 15, cy), label, "regular", LEGEND_SIZE, TEXT_DIM)
                        cursor += entry_w
                    cy += LEGEND_LINE
            x += tile_w
    return canvas


def render_panel(rows, title="", sub="", width=PANEL_W):
    content_w = width - 2 * MARGIN
    parts = [render_row(row, content_w) for row in rows]
    header_h = (30 + (24 if sub else 0) + 12) if title else 0
    height = MARGIN + header_h + sum(p.height for p in parts) + ROW_GAP * (len(parts) - 1) + MARGIN - 4
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    y = MARGIN
    if title:
        draw.rounded_rectangle((MARGIN, y + 4, MARGIN + 5, y + 26), radius=2.5, fill=ACCENT)
        draw_text(draw, (MARGIN + 16, y), title, "semibold", 22, TEXT)
        if sub:
            draw_text(draw, (MARGIN + 16, y + 30), sub, "regular", 14.5, TEXT_DIM)
        y += header_h
    for index, part in enumerate(parts):
        canvas.paste(part, (MARGIN, y))
        y += part.height + (ROW_GAP if index < len(parts) - 1 else 0)
    return canvas


def save_jpeg(image, path, quality=JPEG_QUALITY, max_kb=JPEG_MAX_KB):
    """Save at `quality`; step the quality (then the chroma resolution) down only when the file would exceed max_kb."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB")
    attempts = [(quality, 0), (quality, 2)] + [(q, 2) for q in range(quality - 3, 59, -3)]
    for q, subsampling in attempts:
        image.save(path, "JPEG", quality=q, subsampling=subsampling, optimize=True, progressive=True)
        if path.stat().st_size <= max_kb * 1024:
            return q, subsampling
    return attempts[-1]


# ---------------------------------------------------------------------------------------------------------------
# The pack and its nodes
# ---------------------------------------------------------------------------------------------------------------
_PACK = None


def pack():
    global _PACK
    if _PACK is None:
        _PACK = importlib.import_module("NH-Nodes")
    return _PACK


def run(node_id, **given):
    """Call a node the way ComfyUI does: its FUNCTION, every input filled with its declared default unless given."""
    cls = pack().NODE_CLASS_MAPPINGS[node_id]
    spec = cls.INPUT_TYPES()
    kwargs = {}
    for section in ("required", "optional"):
        for name, value in (spec.get(section) or {}).items():
            if name in given or not isinstance(value, (list, tuple)):
                continue
            kind, opts = value[0], (value[1] if len(value) > 1 and isinstance(value[1], dict) else {})
            if "default" in opts:
                kwargs[name] = opts["default"]
            elif isinstance(kind, (list, tuple)) and kind and section == "required":
                kwargs[name] = kind[0]
    kwargs.update(given)
    if getattr(cls, "INPUT_IS_LIST", False):  # ComfyUI hands such nodes every input as a list
        kwargs = {k: v if isinstance(v, list) else [v] for k, v in kwargs.items()}
    return getattr(cls(), cls.FUNCTION)(**kwargs)


@functools.lru_cache(maxsize=None)
def photo(name):
    """A scikit-image sample photo (public domain / CC0) as an RGB PIL image."""
    from skimage import data

    return Image.fromarray(getattr(data, name)()).convert("RGB")


# ---------------------------------------------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------------------------------------------
EXAMPLES = {}  # file name -> function returning the finished PIL image
HERO = {}  # name -> tile image reused by the banner


def example(filename):
    def register(function):
        EXAMPLES[filename] = function
        return function

    return register


PAD_HEX = "#E5E7EB"


@example("resize-modes.jpg")
def ex_resize_modes():
    modes = (
        ("cover_crop", "fills the frame, trims the edges"),
        ("contain_pad", "fits the whole image, adds padding"),
        ("stretch", "fills the frame, distorts"),
    )
    # Row A: Smart Ratio Image Resize picks the preset closest to the source ratio (a 21:9 panorama lands on 16:9).
    panorama = center_crop_ratio(photo("rocket"), 21 / 9)
    tiles, info = [Tile(panorama, "source", f"{panorama.width}×{panorama.height} · 21:9")], ""
    for mode, note in modes:
        image, width, height, _info, _src, _dst, preset, _data = run(
            "NH_SmartRatioImageResize", image=to_tensor(panorama), model_family="FLUX.2 Klein 9B",
            resolution_level="2 MP", resize_mode=mode, resize_method="lanczos", pad_color_hex=PAD_HEX,
        )
        label = preset.split(" | ")  # "2 MP | 16:9 | 1888×1056"
        info = f"closest preset {label[1] if len(label) == 3 else ratio_text(width, height)} ({width}×{height})"
        tiles.append(Tile(to_pil(image), mode, note))
        if mode == "cover_crop":
            HERO["resize-cover"] = tiles[-1].image
    row_a = Row([tiles[0], ARROW, *tiles[1:]], "NH Smart Ratio Image Resize", f"FLUX.2 Klein 9B · 2 MP · {info}")

    # Row B: Ratio Preset Image Resize goes to the ratio you ask for.
    landscape = photo("coffee")
    tiles = [Tile(landscape, "source", f"{landscape.width}×{landscape.height} · {ratio_text(*landscape.size)}")]
    for (mode, note), alias in zip(modes, ("crop", "pad", "fill")):
        image, width, height, *_ = run(
            "NH_RatioPresetImageResize", image=to_tensor(landscape), model_family="FLUX.2 Klein 9B",
            resolution_level="1 MP", target_ratio="1:1", resize_mode=alias, resize_method="lanczos",
            pad_color_hex=PAD_HEX,
        )
        info = f"target_ratio 1:1 ({width}×{height})"
        tiles.append(Tile(to_pil(image), mode, note))
        if mode == "contain_pad":
            HERO["resize-pad"] = tiles[-1].image
    row_b = Row([tiles[0], ARROW, *tiles[1:]], "NH Ratio Preset Image Resize", f"FLUX.2 Klein 9B · 1 MP · {info}")
    return render_panel([row_a, row_b])


def duotone(image, dark=(7, 34, 49), light=(205, 247, 240)):
    """A stand-in for 'your model': a plain edit that keeps the framing, so the restored area is easy to see."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)[..., None] / 255.0
    mix = np.asarray(dark, np.float32) * (1.0 - gray) + np.asarray(light, np.float32) * gray
    return Image.fromarray(np.clip(np.rint(mix), 0, 255).astype(np.uint8))


@example("restore-geometry.jpg")
def ex_restore_geometry():
    original = photo("chelsea")
    resized_t, width, height, *_, resize_data = run(
        "NH_RatioPresetImageResize", image=to_tensor(original), model_family="FLUX.2 Klein 9B",
        resolution_level="1 MP", target_ratio="1:1", resize_mode="crop", resize_method="lanczos",
        pad_color_hex=PAD_HEX,
    )
    resized = to_pil(resized_t)
    processed = duotone(resized)
    restored_t, mask_t = run(
        "NH_RestoreImageGeometry", image=to_tensor(processed), original_image=to_tensor(original),
        resize_data=resize_data, resize_method="bicubic",
    )
    restored, mask = to_pil(restored_t), mask_array(mask_t)
    size = f"{original.width}×{original.height}"
    tiles = [
        Tile(original, "original", f"{size} · {ratio_text(*original.size)}"),
        Tile(resized, "resized", f"cover_crop · {width}×{height}"),
        Tile(processed, "processed", "your model, any edit"),
        Tile(masked(restored, Layer(mask, TEAL, 0.0, 2.2)), "restored image + mask",
             f"back to {size} · teal outline = mask", aspect=original.width / original.height),
    ]
    return render_panel(
        [Row([tiles[0], ARROW, tiles[1], ARROW, tiles[2], ARROW, tiles[3]])],
        "NH Restore Image Geometry", "put an edited image back into the frame of the original",
    )


def badge(draw, center, number, color, radius):
    """A small round number badge (anti-aliasing comes from the final down-scaling)."""
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    draw_text(draw, (x, y + 1), str(number), "semibold", radius * 1.25, (22, 24, 28), anchor="mm")


TILE_COLORS = (TEAL, PINK, AMBER, BLUE, GREEN, VIOLET, CORAL, SKY)


def tint_tile(tile, index, columns):
    """A mild brighter / darker change, alternating like a chessboard: 'each tile edited on its own'."""
    brighter = ((index // columns) + (index % columns)) % 2 == 0
    return np.clip(tile * (1.08 if brighter else 0.92), 0.0, 1.0)


@example("tile-untile.jpg")
def ex_tile_untile():
    source = photo("rocket")
    tiles_t, tile_data, count, overlap_x, overlap_y, _info = run(
        "NH_ImageTile", image=to_tensor(source), tile_mode="square", tile_size=256, tile_width=256, tile_height=256,
    )
    entries = tile_data["images"][0]["tiles"]
    columns = len({entry["x"] for entry in entries})
    rows = len({entry["y"] for entry in entries})
    tile_w, tile_h = tile_data["tile_width"], tile_data["tile_height"]

    # 1. the source with the footprint of every tile; the strips where tiles overlap are lightened
    coverage = np.zeros((source.height, source.width), np.int32)
    for entry in entries:
        coverage[entry["y"]:entry["y"] + entry["crop_height"], entry["x"]:entry["x"] + entry["crop_width"]] += 1
    shaded = np.asarray(source, dtype=np.float32) * 0.62
    shaded = np.where((coverage >= 2)[..., None], shaded * 0.55 + 255.0 * 0.45, shaded)
    marked = Image.fromarray(np.clip(shaded, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(marked)
    for index, entry in enumerate(entries):
        color = TILE_COLORS[index % len(TILE_COLORS)]
        draw.rectangle((entry["x"], entry["y"], entry["x"] + entry["crop_width"] - 1,
                        entry["y"] + entry["crop_height"] - 1), outline=color, width=3)
    for index, entry in enumerate(entries):
        badge(draw, (entry["x"] + 24, entry["y"] + 24), index + 1, TILE_COLORS[index % len(TILE_COLORS)], 14)

    # 2. the tiles after a per-tile edit, laid out like the grid
    edited = np.stack([tint_tile(tile.numpy(), index, columns) for index, tile in enumerate(tiles_t)])
    gap = 14
    sheet = Image.new("RGBA", (columns * tile_w + (columns - 1) * gap, rows * tile_h + (rows - 1) * gap), (0, 0, 0, 0))
    for index, tile in enumerate(edited):
        x, y = (index % columns) * (tile_w + gap), (index // columns) * (tile_h + gap)
        picture = Image.fromarray(np.rint(tile * 255).astype(np.uint8))
        paste_rounded(sheet, picture, (x, y), radius=16, border=TILE_COLORS[index % len(TILE_COLORS)])
        badge(ImageDraw.Draw(sheet), (x + 26, y + 26), index + 1, TILE_COLORS[index % len(TILE_COLORS)], 15)

    # 3. Image Untile puts them back and blends the overlaps
    merged_t, _info = run("NH_ImageUntile", tiles=torch.from_numpy(edited), tile_data=tile_data)
    merged = to_pil(merged_t)
    HERO["tile-grid"] = marked
    return render_panel(
        [Row([
            Tile(marked, "Image Tile", f"{columns}×{rows} grid · light strips = overlap (at least {overlap_x} px)"), ARROW,
            Tile(sheet, f"{count} tiles of {tile_w}×{tile_h}", "each one edited on its own", frame=False), ARROW,
            Tile(merged, "Image Untile", "overlaps blended, no seams"),
        ])],
        "NH Image Tile  →  NH Image Untile", "split a large image into overlapping tiles, process them, merge them back",
    )


def label_font():
    """A font file name the Image Label node can load (it looks in the Windows font folder), else 'default'."""
    folder = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    for name in ("Inter-SemiBold.ttf", "segoeuib.ttf", "arialbd.ttf"):
        if (folder / name).is_file():
            return name
    return "default"


@example("image-label.jpg")
def ex_image_label():
    base = to_tensor(photo("coffee"))
    common = dict(font_name=label_font(), fit_text=True, outline_width=0, outline_color_hex="#000000")
    variants = (
        ("position = header", "default band above the image", dict(
            label_text="Espresso · 40 ml", position="header", frame_height=64, fill_enabled=True,
            frame_color_hex="#111111", text_color_hex="#FFFFFF", font_size=34, overlay_on_image=False,
            align="left", padding_x=22)),
        ("position = bottom", "centred text, own colours", dict(
            label_text="Espresso · 40 ml", position="bottom", frame_height=64, fill_enabled=True,
            frame_color_hex="#0F766E", text_color_hex="#F0FDFA", font_size=34, overlay_on_image=False,
            align="center", padding_x=22)),
        ("overlay_on_image", "the band covers the image edge", dict(
            label_text="Espresso · 40 ml", position="bottom", frame_height=64, fill_enabled=True,
            frame_color_hex="#111111", text_color_hex="#FFFFFF", font_size=34, overlay_on_image=True,
            align="right", padding_x=22)),
        ("fit_text = on", "a long label shrinks to fit the width", dict(
            label_text="Espresso · 40 ml · 9 bar · 25 s", position="header", frame_height=64, fill_enabled=True,
            frame_color_hex="#111111", text_color_hex="#FFFFFF", font_size=64, overlay_on_image=False,
            align="left", padding_x=22)),
    )
    tiles = []
    for caption, note, options in variants:
        (labelled,) = run("NH_ImageLabel", image=base, **{**common, **options})
        tiles.append(Tile(to_pil(labelled), caption, note))
    HERO["label"] = tiles[1].image
    return render_panel([Row(tiles)], "NH Image Label", "add a caption band to an image: position, colours, overlay and fit")


@functools.lru_cache(maxsize=None)
def retouched_coffee():
    """image_b for the compare example: the same photo with a softer background and a livelier cup (an 'edit')."""
    original = photo("coffee")
    array = np.asarray(original, dtype=np.float32)
    height, width = array.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width]
    focus = np.exp(-(((xs - 0.43 * width) / (0.34 * width)) ** 2 + ((ys - 0.45 * height) / (0.42 * height)) ** 2) * 1.6)
    focus = np.clip(focus, 0.0, 1.0)[..., None]
    soft = cv2.GaussianBlur(array, (0, 0), 9.0)
    mixed = array * focus + soft * (1.0 - focus)
    mixed = np.clip((mixed - 128.0) * 1.2 + 128.0 + 10.0 * focus, 0, 255)
    gray = mixed.mean(axis=2, keepdims=True)
    mixed = np.clip(gray + (mixed - gray) * (1.0 + 0.45 * focus), 0, 255)
    return Image.fromarray(mixed.astype(np.uint8))


@example("image-compare.jpg")
def ex_image_compare():
    before, after = photo("coffee"), retouched_coffee()
    modes = {}
    for mode in ("side_by_side", "top_bottom", "split_vertical", "split_horizontal", "difference", "overlay"):
        (result,) = run("NH_ImageCompare", image_a=to_tensor(before), image_b=to_tensor(after), compare_mode=mode,
                        split_position=0.5, overlay_opacity=0.5, resize_method="lanczos")
        modes[mode] = to_pil(result)
    notes = {
        "side_by_side": "image_a | image_b", "top_bottom": "image_a over image_b",
        "split_vertical": "A left, B right · split_position 0.5", "split_horizontal": "A top, B bottom",
        "difference": "|A − B|: bright where they differ", "overlay": "50 % blend · overlay_opacity 0.5",
    }

    def tile(mode):
        return Tile(modes[mode], mode, notes[mode])

    HERO["compare"] = modes["split_vertical"]

    return render_panel(
        [Row([tile("side_by_side"), tile("split_vertical")], max_h=250),
         Row([tile("top_bottom"), tile("split_horizontal"), tile("difference"), tile("overlay")], max_h=250)],
        "NH Image Compare", "image_a = original photo · image_b = retouched copy (softer background, livelier cup)",
    )


def fit_box(image, box, radius=14):
    """`image` scaled to fit inside a box x box square (aspect kept) and centred on a transparent square."""
    scale = min(box / image.width, box / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    cell = Image.new("RGBA", (box, box), (0, 0, 0, 0))
    paste_rounded(cell, image.resize(size, LANCZOS), ((box - size[0]) // 2, (box - size[1]) // 2), radius=radius)
    return cell


@example("grid-composite.jpg")
def ex_grid_composite():
    names = ("astronaut", "coffee", "chelsea", "rocket")
    labelled = []
    for name in names:
        # the usual chain: crop every photo to the same square, name it, then lay them out
        square, _w, _h = run("NH_ImageResizeByUnit", image=to_tensor(photo(name)), unit="pixel", width_value=512.0,
                             height_value=512.0, dpi=300, resize_mode="crop", resize_method="lanczos",
                             pad_color_hex="#000000")
        (named,) = run("NH_ImageLabel", image=square, label_text=name, position="bottom", frame_height=60,
                       fill_enabled=True, frame_color_hex="#111111", text_color_hex="#FFFFFF", font_name=label_font(),
                       font_size=30, fit_text=True, overlay_on_image=True, align="left", padding_x=22,
                       outline_width=0, outline_color_hex="#000000")
        labelled.append(named)
    sheet, slots, used, _info = run(
        "NH_ImageGridComposite", images=labelled, columns=2, rows=2, cell_size_mode="first_image", cell_width=512,
        cell_height=512, resize_mode="fill", gap_x=16, gap_y=16, background_color_hex="#F3F4F6",
    )
    inputs = Image.new("RGBA", (2 * 300 + 12, 2 * 300 + 12), (0, 0, 0, 0))
    for index, name in enumerate(names):
        inputs.alpha_composite(fit_box(photo(name), 300), ((index % 2) * 312, (index // 2) * 312))
    cell = to_pil(sheet)
    HERO["sheet"] = cell
    return render_panel(
        [Row([Tile(inputs, f"{len(names)} photos", "different sizes and shapes", frame=False), ARROW,
              Tile(cell, "Image Grid Composite", f"2×2 · 512×512 cells · gap 16 px · {used} of {slots} slots used")],
             max_h=470)],
        "NH Image Grid Composite", "a labelled contact sheet: Image Resize Unit → Image Label → Image Grid Composite",
    )


@example("layer-layout.jpg")
def ex_layer_layout():
    paper, size = "#F4F1EA", (1000, 1250)
    coffee, chelsea, rocket, astronaut = (photo(n) for n in ("coffee", "chelsea", "rocket", "astronaut"))
    # The node takes one batch, so every photo is brought to the same 3:2 shape first.
    shots = [
        center_crop_ratio(astronaut, 1.5, (0.5, 0.0)), coffee, rocket, chelsea,
        coffee.crop((130, 10, 430, 210)), chelsea.crop((200, 70, 380, 190)),
    ]
    shots = [shot.resize((900, 600), LANCZOS) for shot in shots]

    def page(images, **options):
        stacked = torch.cat([to_tensor(image) for image in images], dim=0)
        canvas, _slots, used, _info = run(
            "NH_LayerLayoutComposite", images=stacked, canvas_width=size[0], canvas_height=size[1], **options)
        return to_pil(canvas), used

    page_a, used_a = page(shots[:3], background_color=paper, layout_mode="preset", layout_preset="2_top_1_bottom",
                          resize_mode="cover", margin_x=60, margin_y=60, gap_x=40, gap_y=40)
    page_b, used_b = page(shots, background_color=paper, layout_mode="preset", layout_preset="2x3_mixed",
                          resize_mode="cover", margin_x=60, margin_y=60, gap_x=40, gap_y=40)
    blurred = cv2.GaussianBlur(np.asarray(coffee), (0, 0), 14)
    backdrop = Image.fromarray(np.clip(blurred.astype(np.float32) * 0.55, 0, 255).astype(np.uint8))
    page_c, used_c = page(
        [shots[2], shots[0], shots[3]], background_color="#1D1F23", layout_mode="custom", custom_units="relative",
        resize_mode="cover", background_fit="cover",
        custom_layout="0.07,0.06,0.86,0.46\n0.07,0.56,0.41,0.38\n0.52,0.56,0.41,0.38",
        background_image=to_tensor(backdrop),
    )
    HERO["layout"] = page_b
    return render_panel(
        [Row([Tile(page_a, "preset: 2_top_1_bottom", f"{used_a} images · cover"),
              Tile(page_b, "preset: 2x3_mixed", f"{used_b} images · cover"),
              Tile(page_c, "custom layout + background_image", f"{used_c} images · slots as fractions of the canvas")])],
        "NH Layer Layout Composite", "place images into slots of a canvas: presets, a grid, or your own slot list",
    )


# ---------------------------------------------------------------------------------------------------------------
# Masks and models
# ---------------------------------------------------------------------------------------------------------------
def require_segformer(name):
    """The local SegFormer folder for `name`, or Skip. Nothing is downloaded by this script."""
    try:
        backend = importlib.import_module("NH-Nodes.core.segformer_backend")
        found = backend.resolve_model_dir(name, download=False)
    except Exception as error:
        raise Skip(f"cannot look for the SegFormer model '{name}': {error}") from error
    if not found:
        raise Skip(f"SegFormer model '{name}' is not installed (this script never downloads models)")
    return found


def require_vton():
    """The local folder of the four VTON ONNX models, or Skip."""
    if importlib.util.find_spec("onnxruntime") is None:
        raise Skip("onnxruntime is not installed")
    try:
        module = importlib.import_module("NH-Nodes.nodes.vton.vton_preprocessor_nodes")
        found = module.resolve_vton_models_dir(download=False)
    except Exception as error:
        raise Skip(f"cannot look for the VTON models: {error}") from error
    if not found:
        raise Skip("the VTON ONNX models (dwpose, humanparsing) are not installed (this script never downloads them)")
    return found


def mask_box(mask):
    """(x0, y0, x1, y1), inclusive, of the pixels above 0.5; None for an empty mask."""
    ys, xs = np.where(np.asarray(mask) > 0.5)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def as_mask(array):
    """A float [H, W] numpy mask as a ComfyUI MASK tensor [1, H, W]."""
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).unsqueeze(0)


@functools.lru_cache(maxsize=None)
def garment_run(part="upper_garment", attach_parts=True):
    """The real Garment Segment node on the astronaut: (mask [H, W] float, cutout_rgba PIL RGBA)."""
    require_segformer("segformer_fashion")
    mask, cutout, _mask_image = run(
        "NH_GarmentSegment", image=to_tensor(photo("astronaut")), part=part, attach_parts=attach_parts,
        mask_expand=0, mask_blur=0, fill_holes=False, device="cpu",
    )
    return mask_array(mask), to_pil(cutout)


def suit_mask():
    """The suit of the astronaut as a mask: Garment Segment output (a colour key only if the model is missing)."""
    try:
        return garment_run()[0]
    except Skip as reason:
        print(f"  note: {reason}; using a colour-key mask of the suit instead", flush=True)
        hsv = cv2.cvtColor(np.asarray(photo("astronaut")), cv2.COLOR_RGB2HSV)
        key = ((hsv[..., 0] >= 3) & (hsv[..., 0] <= 14) & (hsv[..., 1] > 120) & (hsv[..., 2] > 90)).astype(np.uint8)
        return cv2.morphologyEx(key, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(np.float32)


@example("mask-morphology.jpg")
def ex_mask_morphology():
    image = photo("coffee")
    # A typical rough input: a colour key of the red cup and saucer, with specks on the table and holes inside.
    hsv = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2HSV)
    rough = (((hsv[..., 0] <= 8) | (hsv[..., 0] >= 170)) & (hsv[..., 1] > 150)).astype(np.float32)
    source, grow = as_mask(rough), 15

    def morph(mask, amount, fill_holes=False, blur=0.0):
        (result,) = run("NH_MaskAdvancedMorphology", mask=mask, horizontal_expand=amount, vertical_expand=amount,
                        fill_holes=fill_holes, blur_radius=blur)
        return result

    steps = (
        ("source mask", "colour key of the red cup and saucer", source),
        ("dilate", f"horizontal / vertical_expand = +{grow}", morph(source, grow)),
        ("erode", f"horizontal / vertical_expand = -{grow}", morph(source, -grow)),
        ("open", "erode, then dilate: drops the specks", morph(morph(source, -grow), grow)),
        ("close", "dilate, then erode: closes small holes", morph(morph(source, grow), -grow)),
        ("fill_holes + blur_radius", "fill_holes on, blur_radius 4", morph(source, 0, True, 4.0)),
    )
    tiles = [Tile(masked(image, Layer(mask_array(mask))), caption, sub, aspect=image.width / image.height)
             for caption, sub, mask in steps]
    HERO["morph"] = overlay_masks(image, [Layer(mask_array(steps[4][2]))])
    return render_panel([Row(tiles[:3]), Row(tiles[3:])], "NH Mask Morphology",
                        "grow, shrink, clean up and soften a mask (open and close chain two nodes)")


@example("mask-aware-resize.jpg")
def ex_mask_aware_resize():
    image, garment = photo("astronaut"), suit_mask()
    target = (384, 512)
    tiles = [Tile(masked(image, Layer(garment, TEAL, 0.38)), "image + mask",
                  f"{image.width}×{image.height} · garment mask", aspect=1.0)]
    notes = {"crop": "window slides inside the image", "pad": "mask centred, border padded"}
    for mode in ("crop", "pad"):
        resized, mask = run("NH_MaskResizeImage", image=to_tensor(image), mask=as_mask(garment), width=target[0],
                            height=target[1], mode=mode, pad_color_hex=PAD_HEX)
        tiles.append(Tile(masked(to_pil(resized), Layer(mask_array(mask), TEAL, 0.38)), f"mode = {mode}",
                          f"{target[0]}×{target[1]} · {notes[mode]}", aspect=target[0] / target[1]))
        if mode == "crop":
            HERO["mask-resize"] = overlay_masks(to_pil(resized), [Layer(mask_array(mask), TEAL, 0.38)])
    return render_panel([Row([tiles[0], ARROW, tiles[1], tiles[2]], max_h=400)], "NH Mask-Aware Resize",
                        "resize to a target size without losing the masked area")


@example("agnostic.jpg")
def ex_agnostic():
    image, garment = photo("astronaut"), suit_mask()
    outputs = {}
    for fill in ("gray", "noise", "blur"):
        outputs[fill] = run("NH_AgnosticImageGenerator", image=to_tensor(image), mask=as_mask(garment), fill_mode=fill,
                            blur_radius=65, noise_strength=1.0, gray_value=0.5, feathering=6)
    agnostic = {fill: to_pil(result[0]) for fill, result in outputs.items()}
    HERO["agnostic"] = agnostic["noise"]
    tiles = [
        Tile(masked(image, Layer(garment)), "image + mask", "garment mask", aspect=1.0),
        Tile(agnostic["gray"], "agnostic_img", "fill_mode = gray"),
        Tile(agnostic["noise"], "agnostic_img", "fill_mode = noise"),
        Tile(agnostic["blur"], "agnostic_img", "fill_mode = blur"),
        Tile(to_pil(outputs["gray"][1]), "masked_img", "the mask area painted white"),
        Tile(to_pil(outputs["gray"][2]), "composite", "left: outline of the mask · right: agnostic_img"),
    ]
    return render_panel([Row([tiles[0], ARROW, *tiles[1:4]], max_h=260), Row(tiles[4:], max_h=260)],
                        "NH Agnostic Image Generator",
                        "blank out the masked area (feathering 6) to get the input for inpainting or try-on")


@example("mask-aspect.jpg")
def ex_mask_aspect():
    image, garment = photo("astronaut"), suit_mask()
    x0, y0, x1, y1 = mask_box(garment)
    width_a, height_a = x1 - x0 + 1, y1 - y0 + 1
    # B: a tall ellipse (ratio about 0.6) on the middle of A, so the three modes visibly differ.
    ys, xs = np.mgrid[0:image.height, 0:image.width]
    ellipse = ((((xs - (x0 + x1) / 2) / (0.26 * width_a)) ** 2 + ((ys - (y0 + y1) / 2) / (0.46 * height_a)) ** 2) <= 1.0
               ).astype(np.float32)
    bx0, by0, bx1, by1 = mask_box(ellipse)
    guide = Layer(garment, TEAL, 0.0, 1.6)
    tiles = [Tile(masked(image, Layer(garment, TEAL, 0.28, 1.8), Layer(ellipse, AMBER, 0.5, 2.0)),
                  "A (teal) and B (amber)",
                  f"ratio of A {width_a / height_a:.2f} · of B {(bx1 - bx0 + 1) / (by1 - by0 + 1):.2f}", aspect=1.0)]
    notes = {"stretch": "B scaled to the ratio of A", "pad": "B kept, mask pixels added", "crop": "B trimmed to the ratio"}
    for mode in ("stretch", "pad", "crop"):
        (adjusted,) = run("NH_MaskAspectRatioMatch", target_ratio_mask=as_mask(garment), mask_to_adjust=as_mask(ellipse),
                          mode=mode)
        result = mask_array(adjusted)
        bx0, by0, bx1, by1 = mask_box(result)
        tiles.append(Tile(masked(image, guide, Layer(result, AMBER, 0.5, 2.0)), f"mode = {mode}",
                          f"{notes[mode]} · box {bx1 - bx0 + 1}×{by1 - by0 + 1}", aspect=1.0))
    return render_panel([Row([tiles[0], ARROW, *tiles[1:]])], "NH Mask Aspect Ratio Match",
                        "reshape the box of mask B (amber) to the aspect ratio of mask A (teal outline)")


@example("garment-segment.jpg")
def ex_garment_segment():
    image = photo("astronaut")
    plain, _cutout = garment_run("upper_garment", False)
    attached, cutout = garment_run("upper_garment", True)
    merged = np.clip(attached - plain, 0.0, 1.0)
    on_layers = (Layer(attached, TEAL, 0.45, 0.0), Layer(plain, TEAL, 0.0, 2.0), Layer(merged, AMBER, 0.75, 0.0))
    HERO["garment"] = overlay_masks(image, on_layers)
    tiles = [
        Tile(image, "image", "input photo"),
        Tile(masked(image, Layer(plain, TEAL, 0.45)), "part = upper_garment", "attach_parts = off", aspect=1.0),
        Tile(masked(image, *on_layers), "part = upper_garment", "attach_parts = on · amber = merged parts",
             aspect=1.0),
        Tile(over_checker(cutout), "cutout_rgba", "the mask as the alpha channel"),
    ]
    return render_panel([Row([tiles[0], ARROW, *tiles[1:]])], "NH Garment Segment",
                        "mask of the upper garment, lower garment, footwear or headwear of a person")


SLATE = (148, 163, 184)
MAGENTA = (236, 72, 153)
# The face is drawn first in a neutral tone, so the small features on top of it stand out.
PORTRAIT_COLORS = {
    "face": SLATE, "hair": AMBER, "neck": SKY, "ears": BLUE, "nose": GREEN, "lips": MAGENTA,
    "eyebrows": VIOLET, "eyes": TEAL,
}


@example("portrait-segment.jpg")
def ex_portrait_segment():
    require_segformer("face_parsing")
    head = photo("astronaut").crop((95, 0, 330, 190))
    head = head.resize((head.width * 2, head.height * 2), LANCZOS)
    head_t = to_tensor(head)
    required = pack().NODE_CLASS_MAPPINGS["NH_PortraitSegment"].INPUT_TYPES()["required"]
    toggles = [name for name, value in required.items()
               if isinstance(value, tuple) and value[0] == "BOOLEAN" and name != "fill_holes"]

    def segment(**flags):
        mask, cutout, _mask_image = run("NH_PortraitSegment", image=head_t, mask_expand=0, mask_blur=0,
                                        fill_holes=True, device="cpu", **flags)
        return mask_array(mask), to_pil(cutout)

    default_mask, cutout = segment()  # the default toggles
    layers = []
    for region, color in PORTRAIT_COLORS.items():
        if region in toggles:
            mask, _ = segment(**{name: name == region for name in toggles})
            layers.append(Layer(mask, color, 0.55, 1.4))
    legend = tuple((region, color) for region, color in PORTRAIT_COLORS.items() if region in toggles)
    aspect = head.width / head.height
    HERO["portrait"] = overlay_masks(head, layers)
    tiles = [
        Tile(head, "image", "head and shoulders crop, upscaled 2x"),
        Tile(masked(head, *layers), "one run per region", "face_parsing model, regions on their own", aspect=aspect,
             legend=legend),
        Tile(masked(head, Layer(default_mask, TEAL, 0.45)), "default toggles", "mask output", aspect=aspect),
        Tile(over_checker(cutout), "cutout_rgba", "the mask as the alpha channel"),
    ]
    return render_panel([Row([tiles[0], ARROW, *tiles[1:]], max_h=330)], "NH Portrait Segment",
                        "face, hair, neck and facial features of a portrait as masks")


@example("vton-masks.jpg")
def ex_vton_masks():
    require_vton()
    image = photo("astronaut")
    final_mask, _agnostic_mask, pose, parsing, *_others = run(
        "NH_VTonUltimateProcessor", human_image=to_tensor(image), category="Upper-body", mask_feathering=4,
        cover_shoes=False, refine_hands=True, refine_hair=True, device="cpu",
    )
    HERO["parsing"], HERO["pose"] = to_pil(parsing), to_pil(pose)
    tiles = [
        Tile(image, "human_image", "input photo"),
        Tile(to_pil(pose), "densepose_image", "DWPose keypoints"),
        Tile(to_pil(parsing), "parsing_image", "human parsing, one colour per part"),
        Tile(masked(image, Layer(mask_array(final_mask), TEAL, 0.45)), "final_mask", "category = Upper-body",
             aspect=1.0),
    ]
    return render_panel([Row([tiles[0], ARROW, *tiles[1:]])], "NH VTON Ultimate Processor",
                        "pose, body parsing and the try-on mask of a person photo")


SAM3_CONCEPTS = "person, helmet, face"


@example("sam3-segment.jpg")
def ex_sam3_segment():
    if not GPU or not torch.cuda.is_available():
        raise Skip("SAM 3 needs a CUDA GPU: run with --gpu")
    if importlib.util.find_spec("sam3") is None:
        raise Skip("the sam3 package is not installed")
    try:
        importlib.import_module("NH-Nodes.core.sam3_backend").resolve_checkpoint("sam3.pt")
    except RuntimeError:
        raise Skip("models/sam3/sam3.pt is missing") from None
    image = photo("astronaut")
    segs, _mask, count = run("NH_SAM3Segment", image=to_tensor(image), concepts=SAM3_CONCEPTS, threshold=0.5,
                             detail_level="balanced", max_per_concept=0, checkpoint="sam3.pt",
                             keep_model_loaded=False)
    (preview,) = run("NH_SegsPreview", image=to_tensor(image), segs=segs)

    def compose(expression):
        (mask,) = run("NH_SegsCompose", segs=segs, expression=expression)
        return mask_array(mask)

    _biggest, biggest_mask, biggest_count = run("NH_SegsSelect", segs=segs, labels="helmet", min_score=0.0,
                                                min_area=0.0, max_area=1.0, sort_by="area", start=0, count=1)
    if biggest_count != 1:
        raise RuntimeError("SAM 3 found no helmet in the astronaut photo")
    top = Row(
        [Tile(image, "image", "input photo"), ARROW,
         Tile(to_pil(preview), "Segs Preview", f"concepts: {SAM3_CONCEPTS}")],
        "SAM3 Segment (NH)  →  Segs Preview (NH)", f"{count} instances · threshold 0.5 · detail_level balanced",
        max_h=470,
    )
    bottom = Row(
        [Tile(masked(image, Layer(compose("person - face"), TEAL, 0.45, 2.0)), "person - face",
              "Segs Compose: the person without the face", aspect=1.0),
         Tile(masked(image, Layer(compose("~person"), TEAL, 0.45, 2.0)), "~person",
              "Segs Compose: everything except the person", aspect=1.0),
         Tile(masked(image, Layer(compose("helmet"), TEAL, 0.0, 1.6), Layer(mask_array(biggest_mask), AMBER, 0.5, 2.2)),
              "Segs Select", "labels helmet · sort_by area · count 1: the biggest helmet", aspect=1.0)],
        "Segs Compose and Segs Select (NH)", "masks from the labels", max_h=330,
    )
    return render_panel([top, bottom])


# ---------------------------------------------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------------------------------------------
BANNER_SIZE = (1600, 560)
BANNER_TITLE = "NH-Nodes"
BANNER_SUBTITLE = "Image, mask, resolution and workflow nodes for ComfyUI"
# The collage: three columns of node outputs made by the examples above, top to bottom (missing ones are skipped,
# tiles that no column names are added to the shortest column).
BANNER_COLUMNS = (
    ("resize-cover", "parsing", "sheet", "layout"),
    ("garment", "tile-grid", "morph", "label"),
    ("portrait", "compare", "mask-resize", "agnostic"),
)


def banner_background(width, height):
    """Dark diagonal gradient with a faint teal glow top right and a violet one bottom left."""
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    mix = (xs / width * 0.6 + ys / height * 0.4)[..., None]
    base = np.asarray((25, 27, 31), np.float32) * (1.0 - mix) + np.asarray((34, 37, 44), np.float32) * mix

    def glow(cx, cy, radius, color, strength):
        distance = ((xs - cx) ** 2 + (ys - cy) ** 2) / radius ** 2
        return (np.exp(-distance * 2.2) * strength)[..., None] * np.asarray(color, np.float32)

    base = base + glow(1280, 40, 560, TEAL, 0.11) + glow(60, 560, 500, VIOLET, 0.09)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def wrap_lines(text, weight, size, max_width):
    """Greedy word wrap."""
    lines, current = [], ""
    for word in text.split(" "):
        candidate = (current + " " + word).strip()
        if current and text_width(candidate, weight, size) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    return lines + [current] if current else lines


def draw_tracked(draw, xy, text, weight, size, fill, tracking):
    """Text with extra letter spacing."""
    x, y = xy
    for char in text:
        draw.text((x, y), char, font=font(weight, size), fill=fill)
        x += text_width(char, weight, size) + tracking
    return x


def cover_to(image, size):
    """`image` scaled to cover `size` and cropped around its centre."""
    width, height = size
    scale = max(width / image.width, height / image.height)
    resized = image.convert("RGB").resize((max(width, round(image.width * scale)), max(height, round(image.height * scale))),
                                          LANCZOS)
    left, top = (resized.width - width) // 2, (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def banner_collage(canvas, columns):
    """Columns of node outputs on the right, bleeding off the top and bottom, fading into the background on the left."""
    width, height = canvas.size
    column_w, gap, x_start = 236, 20, 826
    tops = (26, 4, 62)
    placed = []
    for index, images in enumerate(columns):
        y = tops[index % len(tops)]
        for image in images:
            if y > height:
                break
            aspect = min(1.8, max(0.75, image.width / image.height))
            tile_h = round(column_w / aspect)
            placed.append((image, x_start + index * (column_w + gap), y, tile_h))
            y += tile_h + gap
    layer = canvas.copy()
    shadow = Image.new("L", canvas.size, 0)
    shadow_draw = ImageDraw.Draw(shadow)
    for _image, x, y, tile_h in placed:
        shadow_draw.rounded_rectangle((x, y + 12, x + column_w, y + 12 + tile_h), radius=RADIUS + 4, fill=255)
    shadow = shadow.filter(ImageFilter.GaussianBlur(16)).point(lambda value: int(value * 0.62))
    layer.paste(Image.new("RGB", canvas.size, (8, 9, 11)), (0, 0), shadow)
    for image, x, y, tile_h in placed:
        paste_rounded(layer, cover_to(image, (column_w, tile_h)), (x, y), radius=RADIUS + 4, border=(74, 79, 90))
    fade = np.clip((np.arange(width, dtype=np.float32) - (x_start - 36)) / 96.0, 0.0, 1.0)
    fade_mask = Image.fromarray(np.tile((fade * 255).astype(np.uint8), (height, 1)))
    return Image.composite(layer, canvas, fade_mask)


def banner_chips():
    """The groups of the pack (NH-Nodes/<Group>/...), the biggest first."""
    counts = {}
    for cls in pack().NODE_CLASS_MAPPINGS.values():
        parts = str(getattr(cls, "CATEGORY", "")).split("/")
        if len(parts) > 1 and parts[1]:
            counts[parts[1]] = counts.get(parts[1], 0) + 1
    return [name for name, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def build_banner():
    """The hero image of the README: title, subtitle, groups of the pack and a collage of real node outputs."""
    width, height = BANNER_SIZE
    if not HERO:
        raise Skip("no example tiles to put in the collage (every example was skipped or failed)")
    columns = [[HERO[key] for key in column if key in HERO] for column in BANNER_COLUMNS]
    named = {key for column in BANNER_COLUMNS for key in column}
    for key in HERO:
        if key not in named:
            min(columns, key=len).append(HERO[key])
    canvas = banner_collage(banner_background(width, height), columns)
    draw = ImageDraw.Draw(canvas)
    left, top = 84, 86

    draw.rounded_rectangle((left, top + 2, left + 5, top + 22), radius=2.5, fill=ACCENT)
    draw_tracked(draw, (left + 18, top), "COMFYUI CUSTOM NODES", "semibold", 15, ACCENT, 3.2)

    baseline = top + 168  # title: white fading to a pale teal towards the bottom
    title_font = font("semibold", 120)
    mask = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(mask).text((left - 4, baseline), BANNER_TITLE, font=title_font, fill=255, anchor="ls")
    fade = np.clip((np.arange(height, dtype=np.float32) - (baseline - 90)) / 100.0, 0.0, 1.0)[:, None, None]
    colour = np.asarray((255, 255, 255), np.float32) * (1.0 - fade) + np.asarray((176, 240, 229), np.float32) * fade
    canvas.paste(Image.fromarray(np.broadcast_to(colour, (height, width, 3)).astype(np.uint8)), (0, 0), mask)

    y = baseline + 58
    for line in wrap_lines(BANNER_SUBTITLE, "regular", 33, 640):
        draw_text(draw, (left, y), line, "regular", 33, (186, 192, 202), anchor="ls")
        y += 44

    x, y = left, y + 18
    for name in banner_chips():
        chip_w = round(text_width(name, "medium", 15.5)) + 34
        if x + chip_w > left + 640:
            x, y = left, y + 46
        draw.rounded_rectangle((x, y, x + chip_w, y + 34), radius=17, fill=(38, 41, 48), outline=(70, 75, 86))
        draw_text(draw, (x + chip_w / 2, y + 17.5), name, "medium", 15.5, (176, 183, 194), anchor="mm")
        x += chip_w + 10
    return canvas


# ---------------------------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("assets_dir", nargs="?", default=str(PACK_DIR / "docs" / "assets"))
    parser.add_argument("--only", default="", help="comma separated file names to build (default: all)")
    parser.add_argument("--gpu", action="store_true", help="leave the GPU visible so the SAM 3 example is built too")
    options = parser.parse_args()
    assets = Path(options.assets_dir)
    wanted = {name.strip() for name in options.only.split(",") if name.strip()}
    unknown = wanted - set(EXAMPLES) - {"banner.jpg"}
    if unknown:
        raise SystemExit(f"unknown example(s): {', '.join(sorted(unknown))}")

    started = time.time()
    pack()
    print(f"pack loaded: {len(pack().NODE_CLASS_MAPPINGS)} nodes, {time.time() - started:.1f}s", flush=True)
    jobs = [(name, function, assets / "examples" / name) for name, function in EXAMPLES.items()]
    jobs.append(("banner.jpg", build_banner, assets / "banner.jpg"))  # last: it uses the tiles of the examples
    written, skipped, failed = [], [], []
    for filename, function, target in jobs:
        write = not wanted or filename in wanted
        feeds_banner = filename != "banner.jpg" and (not wanted or "banner.jpg" in wanted)
        if not write and not feeds_banner:
            continue  # --only: the banner needs the examples to have run, but they are not written again
        begun = time.time()
        try:
            image = function()
        except Skip as reason:
            skipped.append((filename, str(reason)))
            print(f"SKIPPED {filename}: {reason}", flush=True)
            continue
        except Exception as error:  # report, keep building the others, fail at the end
            import traceback

            traceback.print_exc()
            failed.append((filename, f"{type(error).__name__}: {error}"))
            print(f"FAILED  {filename}: {error}", flush=True)
            continue
        if not write:
            continue
        quality, subsampling = save_jpeg(image, target)
        size_kb = target.stat().st_size / 1024
        written.append((filename, image.size, size_kb))
        print(f"wrote   {filename:<22} {image.width}x{image.height}  {size_kb:6.1f} KB  "
              f"q{quality}/{'444' if subsampling == 0 else '420'}  {time.time() - begun:5.1f}s", flush=True)
    print(f"\n{len(written)} written, {len(skipped)} skipped, {len(failed)} failed in {time.time() - started:.1f}s")
    for filename, reason in skipped:
        print(f"  skipped {filename}: {reason}")
    for filename, reason in failed:
        print(f"  FAILED  {filename}: {reason}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
