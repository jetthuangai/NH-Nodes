import numpy as np
import torch
from PIL import Image, ImageColor


_RESAMPLING = getattr(Image, "Resampling", Image)


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
            fitted = _fit_image_to_cell(source, cell_width, cell_height, resize_mode, fill_rgb)
            canvas.paste(fitted, (x, y))
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
