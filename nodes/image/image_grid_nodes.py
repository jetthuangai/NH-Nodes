import numpy as np
import torch
from PIL import Image, ImageColor

try:
    import psutil
except ImportError:  # ComfyUI installs psutil; without it the memory guard below is simply skipped
    psutil = None


_RESAMPLING = getattr(Image, "Resampling", Image)

# Size guard for the output canvas. It is about RAM, not about Pillow's decompression-bomb limit
# (Image.MAX_IMAGE_PIXELS): that limit is process-wide, other packs raise or disable it, and it only applies to
# Image.open, never to an IMAGE that stays in memory.
#
# Peak RAM per canvas pixel, reached inside _pil_to_tensor (measured 28.1-28.2 B/px on Windows, Pillow 11, NumPy 2):
#   4 B   the PIL canvas (Pillow stores an RGB image as 4 bytes per pixel)
#  12 B   the float32 array from astype(np.float32)
#  12 B   the float32 result of "/ 255.0" (NumPy only reuses that temporary on Linux)
# By then the fitted cells and the uint8 array from np.array() are freed, and torch.from_numpy() shares the float32
# result, so the IMAGE tensor adds nothing (its 12 B/px are what stays alive after the node returns). The input
# images were converted before the check, so psutil's "available" already leaves their memory out.
_PEAK_BYTES_PER_PIXEL = 4 + 12 + 12
# Share of the available RAM the node may plan to use; the rest is left to the OS and the other programs.
_USABLE_MEMORY_FRACTION = 0.85
# Image.new raises a bare MemoryError for an image wider than INT_MAX // 4 - 1 px, whatever the memory; the node's
# own limits never get near it. The same ceiling is applied to the height to keep one rule.
_PIL_MAX_SIDE = (2**31 - 1) // 4 - 1


def _estimate_canvas_bytes(canvas_width, canvas_height):
    """Peak RAM in bytes the node needs to turn a canvas of this size into an IMAGE tensor."""
    return _PEAK_BYTES_PER_PIXEL * canvas_width * canvas_height


def _available_memory_bytes():
    """Free physical RAM in bytes, or None when it cannot be read (the guard is advisory, so it then stays quiet)."""
    if psutil is None:
        return None
    try:
        return int(psutil.virtual_memory().available)
    except Exception:  # whatever the platform backend raises must not stop a canvas that may well fit
        return None


def _format_size(nbytes):
    return f"{nbytes / 2**30:.1f} GiB" if nbytes >= 2**30 else f"{nbytes / 2**20:.0f} MiB"


def _check_canvas_size(canvas_width, canvas_height, columns, rows, cell_width, cell_height):
    """Raise ValueError, before anything is allocated, for a canvas the node cannot build in the free memory."""
    layout = f"{canvas_width}x{canvas_height} px canvas ({columns}x{rows} cells of {cell_width}x{cell_height} px)"
    hint = "Lower columns/rows, the cell size or the gaps."

    if max(canvas_width, canvas_height) > _PIL_MAX_SIDE:
        raise ValueError(
            f"Image Grid Composite: the {layout} is too large: Pillow cannot create an image wider or taller "
            f"than {_PIL_MAX_SIDE:,} px. {hint}"
        )

    available = _available_memory_bytes()
    if available is None:
        return
    needed = _estimate_canvas_bytes(canvas_width, canvas_height)
    if needed > available * _USABLE_MEMORY_FRACTION:
        raise ValueError(
            f"Image Grid Composite: the {layout} is too large for the free memory: it needs about "
            f"{_format_size(needed)} of RAM and {_format_size(available)} is available. {hint}"
        )


def _parse_hex_color(value, default):
    try:
        return ImageColor.getrgb((value or "").strip())
    except Exception:
        return default


def _tensor_to_pil(image_tensor):
    array = np.clip(255.0 * image_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return Image.fromarray(array).convert("RGB")
    if array.ndim == 3 and array.shape[-1] == 1:
        return Image.fromarray(array[:, :, 0]).convert("RGB")
    return Image.fromarray(array).convert("RGB")


def _pil_to_tensor(image):
    array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _first_value(value):
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return value[0]
    return value


def _flatten_images(images):
    image_list = images if isinstance(images, (list, tuple)) else [images]
    pil_images = []

    for image_item in image_list:
        if image_item is None:
            continue
        if not isinstance(image_item, torch.Tensor):
            raise TypeError(f"Expected IMAGE tensor input, got {type(image_item).__name__}")

        if image_item.ndim == 3:
            pil_images.append(_tensor_to_pil(image_item))
        elif image_item.ndim == 4:
            for batch_index in range(int(image_item.shape[0])):
                pil_images.append(_tensor_to_pil(image_item[batch_index]))
        else:
            raise ValueError(f"Unsupported IMAGE tensor shape: {tuple(image_item.shape)}")

    if not pil_images:
        raise ValueError("Image Grid Composite requires at least one image")

    return pil_images


def _resize_stretch(image, target_width, target_height):
    return image.resize((target_width, target_height), _RESAMPLING.LANCZOS)


def _resize_fill(image, target_width, target_height):
    width, height = image.size
    scale = max(target_width / max(width, 1), target_height / max(height, 1))
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = image.resize((resized_width, resized_height), _RESAMPLING.LANCZOS)
    left = max(0, (resized_width - target_width) // 2)
    top = max(0, (resized_height - target_height) // 2)
    return resized.crop((left, top, left + target_width, top + target_height))


def _resize_pad(image, target_width, target_height, fill_rgb):
    width, height = image.size
    scale = min(target_width / max(width, 1), target_height / max(height, 1))
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = image.resize((resized_width, resized_height), _RESAMPLING.LANCZOS)
    canvas = Image.new("RGB", (target_width, target_height), fill_rgb)
    canvas.paste(resized, ((target_width - resized_width) // 2, (target_height - resized_height) // 2))
    return canvas


def _fit_image_to_cell(image, target_width, target_height, resize_mode, fill_rgb):
    if resize_mode == "stretch":
        return _resize_stretch(image, target_width, target_height)
    if resize_mode == "fill":
        return _resize_fill(image, target_width, target_height)
    return _resize_pad(image, target_width, target_height, fill_rgb)


class NH_ImageGridComposite:
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "columns": ("INT", {"default": 2, "min": 1, "max": 64, "step": 1}),
                "rows": ("INT", {"default": 2, "min": 1, "max": 64, "step": 1}),
                "cell_size_mode": (["manual", "first_image"],),
                "cell_width": ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
                "cell_height": ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
                "resize_mode": (["stretch", "pad", "fill"],),
                "gap_x": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "gap_y": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "background_color_hex": ("STRING", {"default": "#000000", "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "slot_count", "used_count", "grid_info")
    FUNCTION = "composite"
    CATEGORY = "NH-Nodes/Image"
    SEARCH_ALIASES = ["image grid", "grid composite", "contact sheet", "image collage"]

    def composite(
        self,
        images,
        columns,
        rows,
        cell_size_mode,
        cell_width,
        cell_height,
        resize_mode,
        gap_x,
        gap_y,
        background_color_hex,
    ):
        columns = _first_value(columns)
        rows = _first_value(rows)
        cell_size_mode = _first_value(cell_size_mode)
        cell_width = _first_value(cell_width)
        cell_height = _first_value(cell_height)
        resize_mode = _first_value(resize_mode)
        gap_x = _first_value(gap_x)
        gap_y = _first_value(gap_y)
        background_color_hex = _first_value(background_color_hex)

        columns = max(1, int(columns))
        rows = max(1, int(rows))
        gap_x = max(0, int(gap_x))
        gap_y = max(0, int(gap_y))
        source_images = _flatten_images(images)

        first_image = source_images[0]
        if cell_size_mode == "first_image":
            cell_width, cell_height = first_image.size
        else:
            cell_width = max(1, int(cell_width))
            cell_height = max(1, int(cell_height))

        canvas_width = (columns * cell_width) + ((columns - 1) * gap_x)
        canvas_height = (rows * cell_height) + ((rows - 1) * gap_y)
        _check_canvas_size(canvas_width, canvas_height, columns, rows, cell_width, cell_height)
        fill_rgb = _parse_hex_color(background_color_hex, (0, 0, 0))
        canvas = Image.new("RGB", (canvas_width, canvas_height), fill_rgb)

        slot_count = columns * rows
        used_count = min(len(source_images), slot_count)
        info_lines = [
            f"grid: columns={columns}, rows={rows}, cell={cell_width}x{cell_height}, gap={gap_x}x{gap_y}",
            f"resize_mode={resize_mode}, used={used_count}/{slot_count}",
        ]

        for image_index in range(used_count):
            row = image_index // columns
            column = image_index % columns
            x = column * (cell_width + gap_x)
            y = row * (cell_height + gap_y)
            source = source_images[image_index]
            # No local name for the fitted cell: it must be freed here, not kept until the tensor conversion.
            canvas.paste(_fit_image_to_cell(source, cell_width, cell_height, resize_mode, fill_rgb), (x, y))
            info_lines.append(
                f"{image_index}: x={x}, y={y}, w={cell_width}, h={cell_height}, source={source.width}x{source.height}"
            )

        return (_pil_to_tensor(canvas), slot_count, used_count, "\n".join(info_lines))


NODE_CLASS_MAPPINGS = {
    "NH_ImageGridComposite": NH_ImageGridComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_ImageGridComposite": "Image Grid Composite (NH)",
}
