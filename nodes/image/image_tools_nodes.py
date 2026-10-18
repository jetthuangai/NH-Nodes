import functools
import os
import re
import unicodedata
import uuid
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFont, ImageOps

import comfy.model_management
import comfy.utils
import folder_paths
import node_helpers
from comfy_execution.graph_utils import ExecutionBlocker

from ...core.safe_eval import evaluate
from ...core.utils import parse_hex_color, tensor_to_pil


_RESAMPLING = getattr(Image, "Resampling", Image)
_WINDOWS = os.name == "nt"
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".bmp", ".tif", ".tiff"}
_FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}
_MAX_LAYER_STACK_INPUTS = 64


def _natural_key(value):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def _normalize_match_value(value):
    value = unicodedata.normalize("NFD", str(value or ""))
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    value = re.sub(r"[_\-]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip().casefold()


def _input_image_choices():
    input_dir = folder_paths.get_input_directory()
    if not os.path.isdir(input_dir):
        return [""]

    files = []
    for entry in os.listdir(input_dir):
        full_path = os.path.join(input_dir, entry)
        if os.path.isfile(full_path) and Path(entry).suffix.lower() in _IMAGE_EXTENSIONS:
            files.append(entry)

    return [""] + sorted(files, key=_natural_key)


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


def _parse_hex_color(value, default):
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


def _tensor_to_pil(image_tensor):
    array = np.clip(255.0 * image_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _layer_to_pil(image_tensor):
    """Like _tensor_to_pil, but a 4-channel image stays RGBA so its transparency is not lost."""
    pil_image = tensor_to_pil(image_tensor)
    return pil_image if pil_image.mode == "RGBA" else pil_image.convert("RGB")


def _pil_to_tensor(image):
    array = np.array(image).astype(np.float32) / 255.0
    dtype = comfy.model_management.intermediate_dtype()
    return torch.from_numpy(array).unsqueeze(0).to(dtype=dtype)


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


def _resize_image(image, size, method):
    resample = {
        "lanczos": _RESAMPLING.LANCZOS,
        "bicubic": _RESAMPLING.BICUBIC,
        "bilinear": _RESAMPLING.BILINEAR,
        "nearest": _RESAMPLING.NEAREST,
    }[method]
    return image.resize(size, resample)


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


def _resize_cover(image, target_size):
    target_width, target_height = target_size
    width, height = image.size
    scale = max(target_width / max(width, 1), target_height / max(height, 1))
    resized = _resize_image(image, (max(1, round(width * scale)), max(1, round(height * scale))), "lanczos")
    left = max(0, (resized.width - target_width) // 2)
    top = max(0, (resized.height - target_height) // 2)
    return resized.crop((left, top, left + target_width, top + target_height))


def _resize_contain(image, target_size):
    target_width, target_height = target_size
    width, height = image.size
    scale = min(target_width / max(width, 1), target_height / max(height, 1))
    return _resize_image(image, (max(1, round(width * scale)), max(1, round(height * scale))), "lanczos")


def _paste_layer(canvas, layer, position):
    """Paste ``layer`` onto ``canvas`` through its own alpha, so transparent areas leave the canvas untouched."""
    if layer.mode != "RGBA":
        canvas.paste(layer, position)  # opaque, whatever the canvas mode
    elif canvas.mode == "RGBA":
        # Both have alpha: a straight paste would blend the alpha channels as if they were colours.
        box = (position[0], position[1], position[0] + layer.width, position[1] + layer.height)
        canvas.paste(Image.alpha_composite(canvas.crop(box), layer), position)
    else:
        canvas.paste(layer, position, layer)


def _fit_layer_image(image, target_size, resize_mode):
    target_width, target_height = target_size
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Layer slot width and height must be greater than 0")
    if resize_mode == "stretch":
        return _resize_image(image, (target_width, target_height), "lanczos"), 0, 0
    if resize_mode == "contain":
        fitted = _resize_contain(image, (target_width, target_height))
        return fitted, (target_width - fitted.width) // 2, (target_height - fitted.height) // 2
    fitted = _resize_cover(image, (target_width, target_height))
    return fitted, 0, 0


def _grid_slots(canvas_width, canvas_height, columns, rows, margin_x, margin_y, gap_x, gap_y):
    columns = max(1, int(columns))
    rows = max(1, int(rows))
    content_width = max(1, canvas_width - (2 * margin_x) - (gap_x * (columns - 1)))
    content_height = max(1, canvas_height - (2 * margin_y) - (gap_y * (rows - 1)))
    cell_width = max(1, content_width // columns)
    cell_height = max(1, content_height // rows)
    slots = []
    right_edge = canvas_width - margin_x
    bottom_edge = canvas_height - margin_y

    for row in range(rows):
        for column in range(columns):
            x = margin_x + column * (cell_width + gap_x)
            y = margin_y + row * (cell_height + gap_y)
            width = max(1, (right_edge - x) if column == columns - 1 else cell_width)
            height = max(1, (bottom_edge - y) if row == rows - 1 else cell_height)
            slots.append((x, y, width, height))
    return slots


def _relative_slot(canvas_width, canvas_height, x, y, width, height):
    return (
        int(round(canvas_width * x)),
        int(round(canvas_height * y)),
        int(round(canvas_width * width)),
        int(round(canvas_height * height)),
    )


def _preset_layout_slots(preset, canvas_width, canvas_height, margin_x, margin_y, gap_x, gap_y):
    content_x = margin_x
    content_y = margin_y
    content_width = max(1, canvas_width - (2 * margin_x))
    content_height = max(1, canvas_height - (2 * margin_y))

    if preset == "1_full":
        return [(content_x, content_y, content_width, content_height)]
    if preset == "2_vertical":
        return _grid_slots(canvas_width, canvas_height, 1, 2, margin_x, margin_y, gap_x, gap_y)
    if preset == "2_horizontal":
        return _grid_slots(canvas_width, canvas_height, 2, 1, margin_x, margin_y, gap_x, gap_y)
    if preset == "3_vertical":
        return _grid_slots(canvas_width, canvas_height, 1, 3, margin_x, margin_y, gap_x, gap_y)
    if preset == "4_grid":
        return _grid_slots(canvas_width, canvas_height, 2, 2, margin_x, margin_y, gap_x, gap_y)
    if preset == "2_top_1_bottom":
        top_height = max(1, round((content_height - gap_y) * 0.42))
        bottom_y = content_y + top_height + gap_y
        bottom_height = max(1, content_y + content_height - bottom_y)
        half_width = max(1, (content_width - gap_x) // 2)
        return [
            (content_x, content_y, half_width, top_height),
            (content_x + half_width + gap_x, content_y, max(1, content_width - half_width - gap_x), top_height),
            (content_x, bottom_y, content_width, bottom_height),
        ]
    if preset == "2x3_mixed":
        row_heights = [0.20, 0.20, 0.44]
        usable_height = max(1, content_height - (2 * gap_y))
        heights = [max(1, round(usable_height * ratio / sum(row_heights))) for ratio in row_heights]
        heights[-1] = max(1, content_height - (2 * gap_y) - heights[0] - heights[1])
        half_width = max(1, (content_width - gap_x) // 2)
        slots = []
        y = content_y
        for height in heights:
            slots.append((content_x, y, half_width, height))
            slots.append((content_x + half_width + gap_x, y, max(1, content_width - half_width - gap_x), height))
            y += height + gap_y
        return slots
    return _grid_slots(canvas_width, canvas_height, 2, 3, margin_x, margin_y, gap_x, gap_y)


def _parse_custom_layout(custom_layout, canvas_width, canvas_height, custom_units):
    slots = []
    for raw_line in str(custom_layout or "").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        values = [float(value) for value in re.split(r"[\s,;]+", line) if value.strip()]
        if len(values) < 4:
            raise ValueError(f"Invalid custom layout line: {raw_line}")
        if len(values) >= 5:
            values = values[-4:]
        x, y, width, height = values[:4]

        units = custom_units
        if units == "auto":
            units = "relative" if max(abs(x), abs(y), abs(width), abs(height)) <= 1.0 else "pixels"
        if units == "relative":
            slot = _relative_slot(canvas_width, canvas_height, x, y, width, height)
        elif units == "percent":
            slot = _relative_slot(canvas_width, canvas_height, x / 100.0, y / 100.0, width / 100.0, height / 100.0)
        else:
            slot = (int(round(x)), int(round(y)), int(round(width)), int(round(height)))

        if slot[2] <= 0 or slot[3] <= 0:
            raise ValueError(f"Custom layout slot size must be greater than 0: {raw_line}")
        slots.append(slot)
    return slots


def _layout_slots(layout_mode, layout_preset, canvas_width, canvas_height, columns, rows, margin_x, margin_y, gap_x, gap_y, custom_layout, custom_units):
    if layout_mode == "custom":
        slots = _parse_custom_layout(custom_layout, canvas_width, canvas_height, custom_units)
        if not slots:
            raise ValueError("Custom layout is empty")
        return slots
    if layout_mode == "grid":
        return _grid_slots(canvas_width, canvas_height, columns, rows, margin_x, margin_y, gap_x, gap_y)
    return _preset_layout_slots(layout_preset, canvas_width, canvas_height, margin_x, margin_y, gap_x, gap_y)


def _eval_position_expr(expression, variables):
    # Variable keys are already casefolded. core.safe_eval works in floating point, so 9**9**9 overflows into an
    # error instead of building a huge integer and freezing the worker.
    try:
        return evaluate(expression, variables, zero_division="error", case_insensitive=True)
    except ValueError as exc:  # core.safe_eval.ExpressionError is a ValueError
        raise ValueError(f"Invalid position expression '{expression}': {exc}") from exc


def _split_position_line(line):
    parts = [part.strip() for part in re.split(r"[,;]+", line, maxsplit=1)]
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid position line: {line}")
    return parts[0], parts[1]


def _parse_layer_positions(positions, count, background_width=0, background_height=0, layer_sizes=None):
    variables = {
        "a": int(background_width),
        "b": int(background_height),
        "bg_w": int(background_width),
        "bg_h": int(background_height),
        "width": int(background_width),
        "height": int(background_height),
    }
    for index, (layer_width, layer_height) in enumerate(layer_sizes or [], start=1):
        variables[f"w{index}"] = int(layer_width)
        variables[f"h{index}"] = int(layer_height)

    parsed = []
    for raw_line in str(positions or "").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        x_expr, y_expr = _split_position_line(line)
        x = int(round(_eval_position_expr(x_expr, variables)))
        y = int(round(_eval_position_expr(y_expr, variables)))
        parsed.append((x, y))

        index = len(parsed)
        variables[f"x{index}"] = x
        variables[f"y{index}"] = y

    while len(parsed) < count:
        parsed.append((0, 0))
    return parsed[:count]


def _save_dir_from_text(folder_path):
    folder_path = (folder_path or "").strip()
    if not folder_path:
        return folder_paths.get_output_directory()
    if os.path.isfile(folder_path):
        return os.path.dirname(folder_path)
    if os.path.isabs(folder_path):
        return folder_path
    return os.path.join(folder_paths.get_output_directory(), folder_path)


def _contained_path(base_dir, relative, what):
    """Absolute path of ``relative`` below ``base_dir``; ValueError (naming ``what``) if it would leave ``base_dir``.

    Both slash styles count as separators on every OS, so a workflow written on Windows is judged the same elsewhere.
    """
    base_dir = os.path.abspath(base_dir)
    path = os.path.abspath(os.path.join(base_dir, str(relative).replace("\\", "/")))
    try:
        inside = os.path.commonpath([base_dir, path]) == base_dir
    except ValueError:  # another drive or a UNC share
        inside = False
    if not inside:
        raise ValueError(f"{what} must stay inside the target folder: {relative!r}")
    return path


def _save_child_dir(save_dir, folder_name):
    folder_name = str(folder_name or "").strip()
    if not folder_name:
        return save_dir
    return _contained_path(save_dir, folder_name, "new_folder_name")


def _split_filename_prefix(save_dir, prefix):
    """('sub/dir/', 'name', directory) for a filename_prefix such as 'sub/dir/name'.

    Subfolders are allowed, but the folder part must stay inside ``save_dir`` (no '..', no absolute path) and the
    file name must be a plain name. Without a separator the name is not looked at by the folder check, and on
    Windows a colon turns it into a path: os.path.join(save_dir, 'D:x0001.png') is a file on drive D:, 'C:x' is
    relative to the current folder of drive C: (join drops the 'C:', so the reported name is wrong and the
    numbering never finds the file again) and 'name:stream' writes an NTFS alternate data stream. A Windows file
    name cannot contain a colon anyway, so such a prefix is refused whole, before anything is created.
    """
    if _WINDOWS and ":" in prefix:
        raise ValueError(
            "filename_prefix cannot contain ':' on Windows (it selects a drive or an NTFS stream, "
            f"not a file name): {prefix!r}"
        )
    folder_part, name = re.match(r"(.*[\\/])?([^\\/]*)\Z", prefix, re.DOTALL).groups()
    if not folder_part:
        return "", name, save_dir
    return folder_part, name, _contained_path(save_dir, folder_part, "filename_prefix")


def _resolve_load_dir(folder_path, image="", upload_probe=""):
    folder_path = (folder_path or "").strip()
    if folder_path:
        resolved = folder_path
        if not os.path.isabs(resolved):
            resolved = os.path.join(folder_paths.get_input_directory(), resolved)
        if os.path.isfile(resolved):
            resolved = os.path.dirname(resolved)
        return resolved

    upload_name = image or upload_probe
    if upload_name:
        return os.path.dirname(folder_paths.get_annotated_filepath(upload_name))

    return folder_paths.get_input_directory()


def _collect_image_files(folder_path, recursive=False):
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder does not exist: {folder_path}")

    files = []
    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        if os.path.isfile(full_path) and Path(entry).suffix.lower() in _IMAGE_EXTENSIONS:
            files.append(entry)

    if recursive or not files:
        models_dir = os.path.normcase(os.path.abspath(folder_paths.models_dir))
        for current_root, dir_names, file_names in os.walk(folder_path):
            # Natural order, and never into the ComfyUI models folder (a walk from one of its parents would crawl it).
            dir_names[:] = sorted(
                (name for name in dir_names if os.path.normcase(os.path.abspath(os.path.join(current_root, name))) != models_dir),
                key=_natural_key,
            )
            for file_name in sorted(file_names, key=_natural_key):
                full_path = os.path.join(current_root, file_name)
                if os.path.isfile(full_path) and Path(file_name).suffix.lower() in _IMAGE_EXTENSIONS:
                    rel_path = os.path.relpath(full_path, folder_path)
                    if rel_path not in files:
                        files.append(rel_path)

    files.sort(key=_natural_key)
    if not files:
        raise ValueError(f"No images found in folder: {folder_path}")
    return files


def _path_parts(value):
    return [part.strip() for part in re.split(r"[\\/]+", str(value or "")) if part.strip()]


def _normalized_path_parts(value):
    return [part for part in (_normalize_match_value(item) for item in _path_parts(value)) if part]


def _longest_common_path_run(candidate_parts, source_parts):
    if not candidate_parts or not source_parts:
        return 0

    previous = [0] * (len(source_parts) + 1)
    best = 0
    for candidate_part in candidate_parts:
        current = [0] * (len(source_parts) + 1)
        for index, source_part in enumerate(source_parts, start=1):
            if candidate_part == source_part:
                current[index] = previous[index - 1] + 1
                best = max(best, current[index])
        previous = current
    return best


def _ordered_path_overlap(candidate_parts, source_parts):
    if not candidate_parts or not source_parts:
        return 0

    count = 0
    cursor = 0
    for candidate_part in candidate_parts:
        while cursor < len(source_parts) and source_parts[cursor] != candidate_part:
            cursor += 1
        if cursor >= len(source_parts):
            break
        count += 1
        cursor += 1
    return count


def _candidate_context_score(candidate_path, source_context_paths):
    if not source_context_paths:
        return 0

    candidate_parts = _normalized_path_parts(candidate_path)
    best_score = 0
    for source_path in source_context_paths:
        source_parts = _normalized_path_parts(source_path)
        run = _longest_common_path_run(candidate_parts, source_parts)
        ordered = _ordered_path_overlap(candidate_parts, source_parts)
        score = run * 100 + ordered
        if candidate_parts and len(source_parts) >= 2 and candidate_parts[-1] == source_parts[-2]:
            score += 10
        elif candidate_parts and source_parts and candidate_parts[-1] == source_parts[-1]:
            score += 5
        best_score = max(best_score, score)
    return best_score


def _source_segment_set(source_context_paths):
    segments = set()
    for path_value in source_context_paths or []:
        segments.update(_normalized_path_parts(path_value))
    return segments


def _add_unique_path(paths, path_value):
    if not path_value:
        return
    key = os.path.normcase(os.path.normpath(path_value))
    if key not in {os.path.normcase(os.path.normpath(path)) for path in paths}:
        paths.append(path_value)


_SIGNATURE_ENTRY_LIMIT = 20000  # most folders + files Load Images Matching hashes for IS_CHANGED before it gives up


class _NoMatchError(ValueError):
    """The search itself came up empty: no matching folder, or no images in it. A bad setting is a plain ValueError."""

    def __init__(self, message, match_text=""):
        super().__init__(message)
        self.match_text = match_text


def _hash_parts(hasher, *parts):
    for part in parts:
        hasher.update(str(part).encode("utf-8", errors="replace"))
        hasher.update(b"\0")


def _same_path(first, second):
    return os.path.normcase(os.path.abspath(first)) == os.path.normcase(os.path.abspath(second))


def _is_inside(path, folder):
    """True when ``path`` lies below ``folder`` (``folder`` itself does not count)."""
    path = os.path.normcase(os.path.abspath(path))
    folder = os.path.normcase(os.path.abspath(folder))
    return path != folder and path.startswith(folder.rstrip(os.sep) + os.sep)


def _is_drive_root(path):
    path = os.path.abspath(path)
    return os.path.dirname(path) == path


def _entry_is_dir(entry, follow_symlinks=True):
    try:
        return entry.is_dir(follow_symlinks=follow_symlinks)
    except OSError:
        return False


def _absolute_root(root_path):
    """The matching root as an absolute path; a relative one is taken relative to ComfyUI's input folder."""
    root_path = (root_path or "").strip()
    if root_path and not os.path.isabs(root_path):
        root_path = os.path.join(folder_paths.get_input_directory(), root_path)
    return root_path


def _check_search_root(root_path, recursive):
    """Refuse the roots a search must not scan: a whole drive (when it would be walked) and the models folder."""
    if recursive and _is_drive_root(root_path):
        raise ValueError(
            f"Matching root is a drive root ({root_path}); refusing to scan a whole drive. "
            "Point matching_root_path at the folder that holds the images, or turn recursive off."
        )
    if _same_path(root_path, folder_paths.models_dir):
        raise ValueError(f"Matching root is the ComfyUI models folder ({root_path}), which holds no images to match.")


def _list_folder(folder_path):
    """The os.DirEntry objects in ``folder_path`` in natural order, without the ComfyUI models folder; [] if unreadable."""
    try:
        with os.scandir(folder_path) as scan:
            entries = sorted(scan, key=lambda entry: _natural_key(entry.name))
    except OSError:
        return []
    models_dir = os.path.normcase(os.path.abspath(folder_paths.models_dir))
    return [
        entry for entry in entries
        if not (_entry_is_dir(entry) and os.path.normcase(os.path.normpath(entry.path)) == models_dir)
    ]


def _child_folders(folder_path):
    """Paths of the sub-folders directly inside ``folder_path``, in natural order."""
    return [entry.path for entry in _list_folder(folder_path) if _entry_is_dir(entry)]


def _walk_folders(root_path, recursive=True):
    """Yield ``(folder, entries)`` for ``root_path`` and, when ``recursive``, for every folder below it.

    The matching search's stand-in for os.walk, with limits: it refuses the roots ``_check_search_root`` names,
    never enters the ComfyUI models folder and, like os.walk, does not follow symlinks into other trees.
    """
    _check_search_root(root_path, recursive)
    pending = [root_path]
    while pending:
        folder = pending.pop()
        entries = _list_folder(folder)
        yield folder, entries
        if recursive:
            pending.extend(entry.path for entry in reversed(entries) if _entry_is_dir(entry, follow_symlinks=False))


def _context_twins(root_path, context_names, wanted=None):
    """Folders that stand in for ``root_path`` under another context folder.

    When the root sits inside a context folder (``.../Nam/...``), the same place under a sibling context folder
    (``.../Nu/...``) is a twin, and so is the sibling context folder as a whole. ``wanted`` limits the siblings to
    those names (normalized); all other configured context names count when it is None.
    """
    context = {_normalize_match_value(name) for name in context_names} - {""}
    if not context:
        return []

    parts = list(Path(os.path.normpath(root_path)).parts)
    twins = []
    for position, part in enumerate(parts):
        own_name = _normalize_match_value(part)
        if position == 0 or own_name not in context:
            continue
        rest = parts[position + 1:]
        for sibling_path in _child_folders(str(Path(*parts[:position]))):
            sibling_name = _normalize_match_value(os.path.basename(sibling_path))
            if sibling_name == own_name or sibling_name not in context:
                continue
            if wanted is not None and sibling_name not in wanted:
                continue
            twins.append(sibling_path)
            if rest and os.path.isdir(os.path.join(sibling_path, *rest)):
                twins.append(os.path.join(sibling_path, *rest))
    return twins


def _context_root_candidates(root_path, source_context_paths, context_names=()):
    """Folders to search for the match, best context first.

    That is the matching root, the sub-folders of it that the source path names (``Nu`` when the source lives in a
    ``Nu`` folder) and the context twins of the root. Nothing else above the root is listed, let alone walked.
    """
    root_path = _absolute_root(root_path)
    if not root_path:
        return []

    candidates = []
    _add_unique_path(candidates, root_path)

    source_segments = _source_segment_set(source_context_paths)
    if not source_segments:
        return candidates

    for child_path in _child_folders(root_path):
        if _normalize_match_value(os.path.basename(child_path)) in source_segments:
            _add_unique_path(candidates, child_path)
    for twin_path in _context_twins(root_path, context_names, source_segments):
        _add_unique_path(candidates, twin_path)

    candidates.sort(key=lambda item: (-_candidate_context_score(item, source_context_paths), _natural_key(item)))
    return candidates


def _resolve_matching_folder(root_path, match_text, match_mode, recursive, source_context_paths=None):
    root_path = _absolute_root(root_path)
    if not root_path:
        raise ValueError("Matching root path is empty")
    if not os.path.isdir(root_path):
        raise ValueError(f"Matching root folder does not exist: {root_path}")

    normalized_match = _normalize_match_value(match_text)
    if not normalized_match:
        raise ValueError("Match text is empty")

    candidates = [root_path]
    for _folder, entries in _walk_folders(root_path, recursive):
        candidates.extend(entry.path for entry in entries if _entry_is_dir(entry))

    exact_matches = []
    contains_matches = []
    for candidate in candidates:
        normalized_name = _normalize_match_value(os.path.basename(candidate))
        if normalized_name == normalized_match:
            exact_matches.append(candidate)
        if normalized_match in normalized_name:
            contains_matches.append(candidate)

    matches = exact_matches if match_mode == "exact" else contains_matches or exact_matches
    if not matches:
        raise _NoMatchError(f"No matching folder found for '{match_text}' under: {root_path}", match_text)

    if source_context_paths:
        matches.sort(key=lambda item: (
            -_candidate_context_score(item, source_context_paths),
            _natural_key(os.path.relpath(item, root_path)),
        ))
    else:
        matches.sort(key=lambda item: _natural_key(os.path.relpath(item, root_path)))
    return matches[0]


def _resolve_best_matching_folder(root_path, match_text, match_mode, recursive, source_context_paths=None, context_names=()):
    root_path = _absolute_root(root_path)
    if not root_path:
        raise ValueError("Matching root path is empty")
    _check_search_root(root_path, recursive)

    search_roots = _context_root_candidates(root_path, source_context_paths, context_names)
    if recursive:
        # Walking a folder covers everything below it, so a search root inside another one is not walked twice.
        search_roots = [path for path in search_roots if not any(_is_inside(path, other) for other in search_roots)]

    matched_folders = []
    errors = []
    for search_root in search_roots:
        try:
            matched_folder = _resolve_matching_folder(
                search_root,
                match_text,
                match_mode,
                recursive,
                source_context_paths=source_context_paths,
            )
            _add_unique_path(matched_folders, matched_folder)
        except ValueError as exc:
            errors.append(exc)

    if not matched_folders:
        # A setting that cannot work (missing root, empty text) is reported as such; otherwise nothing matched.
        for exc in errors:
            if not isinstance(exc, _NoMatchError):
                raise exc
        raise _NoMatchError(f"No matching folder found for '{match_text}' under: {root_path}", match_text)

    matched_folders.sort(key=lambda item: (-_candidate_context_score(item, source_context_paths), _natural_key(item)))
    return matched_folders[0]


def _parse_context_folder_names(value):
    names = [item.strip() for item in re.split(r"[\n,;|]+", str(value or "")) if item.strip()]
    return names


def _detect_context_folder(paths, context_folder_names):
    if not context_folder_names:
        return ""

    normalized_context = {_normalize_match_value(name): name for name in context_folder_names}
    for path_value in paths:
        for part in re.split(r"[\\/]+", str(path_value or "")):
            match = normalized_context.get(_normalize_match_value(part))
            if match:
                return match
    return ""


def _matching_selection(
    source_folder_path,
    matching_root_path,
    match_text,
    match_mode,
    recursive,
    index,
    step,
    context_folder_names,
    source_file_names,
    person_folder_path=None,
):
    """Work out what Load images matching reads: one ``(match_text, folder, [file names])`` per match item.

    Raises _NoMatchError when an item finds no folder, or a folder without images, and ValueError for a setting
    that cannot work (missing root, empty match text, a root that must not be scanned).
    """
    if not source_folder_path and person_folder_path:
        source_folder_path = person_folder_path

    source_paths = [line.strip() for line in str(source_folder_path or "").splitlines() if line.strip()]
    first_source_path = source_paths[0] if source_paths else ""
    resolved_source_folder = _resolve_load_dir(first_source_path)
    selected_match_text = (match_text or "").strip()
    source_names = [line.strip() for line in str(source_file_names or "").splitlines() if line.strip()]
    context_names = _parse_context_folder_names(context_folder_names)
    match_items = []

    if selected_match_text:
        item_count = max(len(source_paths), len(source_names), 1)
        for item_index in range(item_count):
            match_items.append((
                selected_match_text,
                source_paths[item_index] if item_index < len(source_paths) else "",
                source_names[item_index] if item_index < len(source_names) else "",
            ))
    elif source_names:
        for item_index, source_name in enumerate(source_names):
            source_dir = os.path.dirname(source_name)
            source_match_text = os.path.basename(os.path.normpath(source_dir)) if source_dir else os.path.basename(os.path.normpath(resolved_source_folder))
            match_items.append((
                source_match_text,
                source_paths[item_index] if item_index < len(source_paths) else "",
                source_name,
            ))
    else:
        selected_match_text = os.path.basename(os.path.normpath(resolved_source_folder))
        match_items.append((selected_match_text, first_source_path, ""))

    plan = []
    for current_match_text, current_source_path, current_source_name in match_items:
        context_paths = [current_source_path, current_source_name, resolved_source_folder]
        context_folder = _detect_context_folder(context_paths, context_names)
        if context_folder:
            context_paths.append(context_folder)
        matched_folder = _resolve_best_matching_folder(
            matching_root_path,
            current_match_text,
            match_mode,
            recursive,
            source_context_paths=context_paths,
            context_names=context_names,
        )
        try:
            image_files = _collect_image_files(matched_folder)
        except ValueError as exc:  # no images in the folder that matched
            raise _NoMatchError(str(exc), current_match_text) from exc
        start_index = index % len(image_files)
        selected_files = [image_files[(start_index + offset) % len(image_files)] for offset in range(step)]
        plan.append((current_match_text, matched_folder, selected_files))
    return plan


def _selection_signature(plan):
    """Digest of what a matching plan reads: the folders chosen, and the selected files with size and mtime."""
    hasher = hashlib.sha256()
    for current_match_text, matched_folder, selected_files in plan:
        _hash_parts(hasher, current_match_text, os.path.abspath(matched_folder))
        for file_name in selected_files:
            stat = os.stat(os.path.join(matched_folder, file_name))
            _hash_parts(hasher, file_name, stat.st_size, stat.st_mtime_ns)
    return hasher.hexdigest()


def _tree_signature(roots, limit=None):
    """Digest of every folder and image below ``roots`` (names, sizes, mtimes); None beyond ``limit`` entries."""
    limit = _SIGNATURE_ENTRY_LIMIT if limit is None else limit
    hasher = hashlib.sha256()
    seen = 0
    for root in roots:
        _hash_parts(hasher, "root", os.path.abspath(root))
        for folder, entries in _walk_folders(root, recursive=True):
            _hash_parts(hasher, "folder", os.path.relpath(folder, root))
            for entry in entries:
                seen += 1
                if seen > limit:
                    return None
                if _entry_is_dir(entry):
                    _hash_parts(hasher, "dir", entry.name)
                elif Path(entry.name).suffix.lower() in _IMAGE_EXTENSIONS:
                    try:
                        stat = entry.stat()
                    except OSError:
                        _hash_parts(hasher, "image", entry.name)
                    else:
                        _hash_parts(hasher, "image", entry.name, stat.st_size, stat.st_mtime_ns)
    return hasher.hexdigest()


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


def _save_pil_image(image, save_path, file_format, dpi):
    image = image.convert("RGB")
    save_kwargs = {"dpi": (dpi, dpi)}

    if file_format == "png":
        pil_format = "PNG"
        save_kwargs["compress_level"] = 4
    elif file_format == "jpg":
        pil_format = "JPEG"
        save_kwargs["quality"] = 95
        save_kwargs["subsampling"] = 0
    elif file_format == "webp":
        pil_format = "WEBP"
        save_kwargs["quality"] = 95
        save_kwargs["method"] = 6
    elif file_format == "avif":
        pil_format = "AVIF"
        save_kwargs["quality"] = 90
    else:
        raise ValueError(f"Unsupported format: {file_format}")

    try:
        image.save(save_path, format=pil_format, **save_kwargs)
    except TypeError:
        if file_format == "avif" and "dpi" in save_kwargs:
            save_kwargs.pop("dpi", None)
            image.save(save_path, format=pil_format, **save_kwargs)
            print("[NH-Nodes] Save Image: AVIF DPI metadata is not supported by the current Pillow build.")
        else:
            raise


class NH_SaveImagePath:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "folder_path": ("STRING", {"default": "", "multiline": False}),
                "create_new_folder": ("BOOLEAN", {"default": False}),
                "new_folder_name": ("STRING", {"default": "", "multiline": False}),
                "filename_prefix": ("STRING", {"default": "NH", "multiline": False}),
                "number_suffix": ("BOOLEAN", {"default": True}),
                "preview_image": ("BOOLEAN", {"default": False}),
                "file_format": (["png", "jpg", "webp", "avif"],),
                "dpi": ("INT", {"default": 300, "min": 1, "max": 2400, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("filenames", "path", "saved_count")
    FUNCTION = "save_images"
    CATEGORY = "NH-Nodes/Image"

    def save_images(self, images, folder_path, filename_prefix, number_suffix=True, preview_image=False, file_format="png", dpi=300, create_new_folder=False, new_folder_name=""):
        save_dir = _save_dir_from_text(folder_path)
        if create_new_folder:
            save_dir = _save_child_dir(save_dir, new_folder_name)

        # A prefix may name subfolders ("renders/img"); they live below save_dir and are where files are counted.
        prefix = (filename_prefix or "NH").strip() or "NH"
        prefix_folder, prefix_name, target_dir = _split_filename_prefix(save_dir, prefix)
        os.makedirs(target_dir, exist_ok=True)

        numbered_pattern = re.compile(rf"^{re.escape(prefix_name)}(\d+)\.(?:png|jpe?g|webp|avif)$", re.IGNORECASE)
        legacy_numbered_pattern = re.compile(rf"^(?:{re.escape(prefix_name)})?(\d+)\.(?:png|jpe?g|webp|avif)$", re.IGNORECASE)
        bare_pattern = re.compile(rf"^{re.escape(prefix_name)}\.(?:png|jpe?g|webp|avif)$", re.IGNORECASE)

        used_numbers = set()
        bare_exists = False
        for name in os.listdir(target_dir):
            if bare_pattern.match(name):
                bare_exists = True
                continue
            match = numbered_pattern.match(name)
            if match is None and number_suffix:
                match = legacy_numbered_pattern.match(name)
            if match:
                used_numbers.add(int(match.group(1)))

        saved_files = []
        preview_results = []
        preview_dir = folder_paths.get_temp_directory()
        next_number = 1
        for batch_index, image in enumerate(images):
            if number_suffix:
                while next_number in used_numbers:
                    next_number += 1
                filename = f"{prefix_name}{next_number:04d}.{file_format}"
                used_numbers.add(next_number)
                next_number += 1
            elif not bare_exists and not used_numbers:
                filename = f"{prefix_name}.{file_format}"
                bare_exists = True
            else:
                while next_number in used_numbers:
                    next_number += 1
                filename = f"{prefix_name}{next_number}.{file_format}"
                used_numbers.add(next_number)
                next_number += 1

            pil_image = _tensor_to_pil(image)
            _save_pil_image(pil_image, os.path.join(target_dir, filename), file_format, int(dpi))
            saved_files.append(prefix_folder + filename)

            if preview_image:
                preview_filename = f"NH_preview_{uuid.uuid4().hex}_{batch_index}.png"
                _save_pil_image(pil_image, os.path.join(preview_dir, preview_filename), "png", int(dpi))
                preview_results.append({
                    "filename": preview_filename,
                    "subfolder": "",
                    "type": "temp",
                })

        result = ("\n".join(saved_files), save_dir, len(saved_files))
        if preview_image:
            return {"ui": {"images": preview_results}, "result": result}
        return result


class NH_LoadImagesFromFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "multiline": False}),
                "index": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "step": ("INT", {"default": 1, "min": 1, "max": 9999}),
                "seed_mode": (["fixed", "increment", "decrement", "random"],),
                "recursive": ("BOOLEAN", {"default": False}),
                "filename_extension": ("BOOLEAN", {"default": True}),
                "path_output": (["file_path", "folder_path"], {"default": "file_path"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "filename", "path")
    FUNCTION = "load_images"
    CATEGORY = "NH-Nodes/Image"

    @classmethod
    def IS_CHANGED(cls, folder_path, index, step, seed_mode, recursive=False, filename_extension=True, path_output="file_path", **kwargs):
        try:
            resolved_folder = _resolve_load_dir(folder_path)
            image_files = _collect_image_files(resolved_folder, recursive=recursive)
            step = max(1, int(step))
            start_index = int(index) % len(image_files)

            if seed_mode == "decrement":
                indices = [((start_index - offset) % len(image_files)) for offset in range(step)]
            else:
                indices = [((start_index + offset) % len(image_files)) for offset in range(step)]

            hasher = hashlib.sha256()
            signature_parts = (
                os.path.abspath(resolved_folder),
                str(start_index),
                str(step),
                str(seed_mode),
                str(bool(recursive)),
                str(bool(filename_extension)),
                str(path_output),
            )
            for part in signature_parts:
                hasher.update(part.encode("utf-8", errors="replace"))
                hasher.update(b"\0")

            for selected_index in indices:
                rel_path = image_files[selected_index]
                full_path = os.path.join(resolved_folder, rel_path)
                stat = os.stat(full_path)
                file_parts = (rel_path, str(stat.st_size), str(stat.st_mtime_ns))
                for part in file_parts:
                    hasher.update(str(part).encode("utf-8", errors="replace"))
                    hasher.update(b"\0")

            return hasher.hexdigest()
        except Exception as exc:
            return f"error:{type(exc).__name__}:{exc}"

    def load_images(self, folder_path, index, step, seed_mode, recursive=False, filename_extension=True, path_output="file_path", **kwargs):
        resolved_folder = _resolve_load_dir(folder_path)
        image_files = _collect_image_files(resolved_folder, recursive=recursive)
        start_index = index % len(image_files)

        if seed_mode == "decrement":
            indices = [((start_index - offset) % len(image_files)) for offset in range(step)]
        else:
            indices = [((start_index + offset) % len(image_files)) for offset in range(step)]

        selected_files = [image_files[i] for i in indices]
        output_images = []
        output_paths = []
        target_size = None

        for file_name in selected_files:
            full_path = os.path.join(resolved_folder, file_name)
            img = node_helpers.pillow(Image.open, full_path)
            img = node_helpers.pillow(ImageOps.exif_transpose, img).convert("RGB")
            if target_size is None:
                target_size = img.size
            elif img.size != target_size:
                img = _resize_image(img, target_size, "lanczos")
            output_images.append(_pil_to_tensor(img))
            if path_output == "folder_path":
                output_paths.append(os.path.dirname(full_path))
            else:
                output_paths.append(full_path)

        display_filenames = []
        for file_name in selected_files:
            base_name = os.path.basename(file_name)
            if not filename_extension:
                base_name = os.path.splitext(base_name)[0]
            display_filenames.append(base_name)

        return (torch.cat(output_images, dim=0), "\n".join(display_filenames), "\n".join(output_paths))


class NH_LoadImagesMatching:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_folder_path": ("STRING", {"default": "", "multiline": False}),
                "matching_root_path": ("STRING", {"default": "", "multiline": False}),
                "match_text": ("STRING", {"default": "", "multiline": False}),
                "match_mode": (["contains", "exact"],),
                "recursive": ("BOOLEAN", {"default": True}),
                "index": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "step": ("INT", {"default": 1, "min": 1, "max": 9999}),
                "repeat_count": ("INT", {"default": 1, "min": 1, "max": 9999}),
                "context_folder_names": ("STRING", {"default": "Nam\nNu", "multiline": True}),
            },
            "optional": {
                "source_file_names": ("STRING", {"default": "", "multiline": True}),
                "fallback_images": ("IMAGE", {
                    "lazy": True,
                    "tooltip": (
                        "Comes out as images when nothing matches, and is only evaluated then. Connect it here "
                        "instead of routing images through a switch or If/Else: a node that receives the held-back "
                        "images never runs."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("images", "filename", "path", "matched_text", "found")
    FUNCTION = "load_matching"
    CATEGORY = "NH-Nodes/Image"
    DESCRIPTION = (
        "Load the images of the folder under matching_root_path whose name matches the source folder (or match_text). "
        "When nothing matches, found is False and images are held back: every node that receives them is skipped, "
        "a switch or If/Else included. To carry on with another image, connect it to fallback_images instead of "
        "routing images through a switch or If/Else: it is only evaluated when nothing matches, and then comes out "
        "as images (found stays False)."
    )

    @classmethod
    def IS_CHANGED(cls, source_folder_path, matching_root_path, match_text, match_mode, recursive, index, step, repeat_count, context_folder_names="Nam\nNu", source_file_names="", **kwargs):
        # ComfyUI's cache key already holds every setting and every upstream result; this adds what is on disk:
        # the folder(s) the search picks and the selected files with their size and time, so an edit re-runs the node.
        # An input wired from another node arrives here as None, which leaves the folder unknown: the whole matching
        # tree is fingerprinted instead. A search that finds nothing, or a tree too big to hash cheaply, gives NaN:
        # that result is recomputed every time (so a "not found" is never remembered).
        unknown = float("nan")
        try:
            settings = (source_folder_path, matching_root_path, match_text, match_mode, recursive, index, step, context_folder_names, source_file_names)
            if None not in settings:
                plan = _matching_selection(
                    source_folder_path, matching_root_path, match_text, match_mode, recursive, index, step,
                    context_folder_names, source_file_names, kwargs.get("person_folder_path"),
                )
                return _selection_signature(plan)
            root_path = _absolute_root(matching_root_path)
            if not root_path or context_folder_names is None:
                return unknown
            roots = [root_path] + _context_twins(root_path, _parse_context_folder_names(context_folder_names))
            signature = _tree_signature(roots)
            return unknown if signature is None else signature
        except (ValueError, OSError):
            return unknown

    def check_lazy_status(self, source_folder_path, matching_root_path, match_text, match_mode, recursive, index, step, repeat_count, context_folder_names="Nam\nNu", source_file_names="", **kwargs):
        # fallback_images is asked for only when it is wired (ComfyUI sends its key, with None until the input has
        # run) and the search finds nothing; on a match its upstream nodes never run. The search is the one
        # load_matching makes. A setting that cannot work asks for nothing: load_matching raises it right after.
        if "fallback_images" not in kwargs or kwargs["fallback_images"] is not None:
            return []
        try:
            _matching_selection(
                source_folder_path, matching_root_path, match_text, match_mode, recursive, index, step,
                context_folder_names, source_file_names, kwargs.get("person_folder_path"),
            )
        except _NoMatchError:
            return ["fallback_images"]
        except (ValueError, OSError):
            pass
        return []

    def load_matching(self, source_folder_path, matching_root_path, match_text, match_mode, recursive, index, step, repeat_count, context_folder_names="Nam\nNu", source_file_names="", fallback_images=None, **kwargs):
        try:
            plan = _matching_selection(
                source_folder_path, matching_root_path, match_text, match_mode, recursive, index, step,
                context_folder_names, source_file_names, kwargs.get("person_folder_path"),
            )
        except _NoMatchError as exc:
            # Nothing to load for this request. `found` lets a workflow carry on with something else, and so does
            # fallback_images, which comes out as the images. Without it the images are held back (whatever needs
            # them is told why). A setting that cannot work still raises.
            message = str(exc)
            if fallback_images is not None:
                print(f"[NH-Nodes] Load images matching: {message} (using fallback_images)")
                return (fallback_images, "", "", exc.match_text, False)
            print(f"[NH-Nodes] Load images matching: {message}")
            return (ExecutionBlocker(message), "", "", exc.match_text, False)

        selected_files = []
        matched_folders = []
        matched_texts = []
        output_images = []
        target_size = None

        for current_match_text, matched_folder, file_names in plan:
            for file_name in file_names:
                full_path = os.path.join(matched_folder, file_name)
                img = node_helpers.pillow(Image.open, full_path)
                img = node_helpers.pillow(ImageOps.exif_transpose, img).convert("RGB")
                if target_size is None:
                    target_size = img.size
                elif img.size != target_size:
                    img = _resize_image(img, target_size, "lanczos")
                output_images.append(_pil_to_tensor(img))
                selected_files.append(file_name)
                matched_folders.append(matched_folder)
                matched_texts.append(current_match_text)

        images = torch.cat(output_images, dim=0)
        if repeat_count > 1:
            images = images.repeat(repeat_count, 1, 1, 1)

        return (images, "\n".join(selected_files), "\n".join(matched_folders), "\n".join(matched_texts), True)


class NH_LayerLayoutComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "canvas_width": ("INT", {"default": 2480, "min": 1, "max": 20000, "step": 1}),
                "canvas_height": ("INT", {"default": 3508, "min": 1, "max": 20000, "step": 1}),
                "background_color": ("STRING", {"default": "#FFFFFF", "multiline": False}),
                "layout_mode": (["preset", "grid", "custom"],),
                "layout_preset": (["2x3", "2x3_mixed", "2_top_1_bottom", "1_full", "2_vertical", "2_horizontal", "3_vertical", "4_grid"],),
                "resize_mode": (["cover", "contain", "stretch"],),
                "columns": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                "rows": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
                "margin_x": ("INT", {"default": 150, "min": 0, "max": 10000, "step": 1}),
                "margin_y": ("INT", {"default": 150, "min": 0, "max": 10000, "step": 1}),
                "gap_x": ("INT", {"default": 150, "min": 0, "max": 10000, "step": 1}),
                "gap_y": ("INT", {"default": 150, "min": 0, "max": 10000, "step": 1}),
                "custom_units": (["auto", "relative", "percent", "pixels"],),
                "custom_layout": ("STRING", {
                    "default": "0.06,0.04,0.39,0.20\n0.55,0.04,0.39,0.20\n0.06,0.28,0.39,0.20\n0.55,0.28,0.39,0.20\n0.06,0.58,0.39,0.36\n0.55,0.58,0.39,0.36",
                    "multiline": True,
                }),
                "background_fit": (["stretch", "cover", "contain"],),
            },
            "optional": {
                "background_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "slot_count", "used_count", "layout_info")
    FUNCTION = "composite"
    CATEGORY = "NH-Nodes/Image"

    @staticmethod
    def _paste_clipped(canvas, layer, x, y):
        left = max(0, int(x))
        top = max(0, int(y))
        right = min(canvas.width, int(x) + layer.width)
        bottom = min(canvas.height, int(y) + layer.height)
        if right <= left or bottom <= top:
            return
        crop_box = (left - int(x), top - int(y), right - int(x), bottom - int(y))
        _paste_layer(canvas, layer.crop(crop_box), (left, top))

    def _create_canvas(self, canvas_width, canvas_height, background_color, background_fit, background_image=None):
        fill = _parse_hex_color(background_color, (255, 255, 255))
        # An opaque colour gives the RGB canvas (and RGB output) this node always had; a colour with transparency
        # ('#RRGGBBAA' with AA below FF) gives an RGBA canvas so that the transparency survives.
        canvas = Image.new("RGBA" if fill[3] < 255 else "RGB", (canvas_width, canvas_height), fill)

        if background_image is None:
            return canvas

        background = _layer_to_pil(background_image[0])
        fitted, offset_x, offset_y = _fit_layer_image(background, (canvas_width, canvas_height), background_fit)
        self._paste_clipped(canvas, fitted, offset_x, offset_y)
        return canvas

    def composite(
        self,
        images,
        canvas_width,
        canvas_height,
        background_color,
        layout_mode,
        layout_preset,
        resize_mode,
        columns,
        rows,
        margin_x,
        margin_y,
        gap_x,
        gap_y,
        custom_units,
        custom_layout,
        background_fit,
        background_image=None,
    ):
        canvas_width = int(canvas_width)
        canvas_height = int(canvas_height)
        slots = _layout_slots(
            layout_mode,
            layout_preset,
            canvas_width,
            canvas_height,
            int(columns),
            int(rows),
            int(margin_x),
            int(margin_y),
            int(gap_x),
            int(gap_y),
            custom_layout,
            custom_units,
        )

        canvas = self._create_canvas(canvas_width, canvas_height, background_color, background_fit, background_image)
        used_count = min(int(images.shape[0]), len(slots))
        info_lines = []

        for image_index in range(used_count):
            x, y, width, height = slots[image_index]
            layer = _layer_to_pil(images[image_index])
            fitted, offset_x, offset_y = _fit_layer_image(layer, (int(width), int(height)), resize_mode)
            paste_x = int(x) + offset_x
            paste_y = int(y) + offset_y
            self._paste_clipped(canvas, fitted, paste_x, paste_y)
            info_lines.append(
                f"{image_index}: x={int(x)}, y={int(y)}, w={int(width)}, h={int(height)}, mode={resize_mode}"
            )

        return (_pil_to_tensor(canvas), len(slots), used_count, "\n".join(info_lines))


class NH_LayerStackComposite:
    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = {
            f"layer_{index}": ("IMAGE",)
            for index in range(1, _MAX_LAYER_STACK_INPUTS + 1)
        }
        return {
            "required": {
                "background_image": ("IMAGE",),
                "layer_count": ("INT", {"default": 1, "min": 0, "max": _MAX_LAYER_STACK_INPUTS, "step": 1}),
                "positions": ("STRING", {
                    "default": "0,0",
                    "multiline": True,
                }),
            },
            "optional": optional_inputs,
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "used_count", "layer_info")
    FUNCTION = "composite"
    CATEGORY = "NH-Nodes/Image"

    @staticmethod
    def _paste_clipped(canvas, layer, x, y):
        x = int(x)
        y = int(y)
        left = max(0, x)
        top = max(0, y)
        right = min(canvas.width, x + layer.width)
        bottom = min(canvas.height, y + layer.height)
        if right <= left or bottom <= top:
            return False

        crop_box = (left - x, top - y, right - x, bottom - y)
        _paste_layer(canvas, layer.crop(crop_box), (left, top))
        return True

    def composite(self, background_image, layer_count, positions, **kwargs):
        # The output has the channels of the background: RGB stays RGB, an RGBA background gives RGBA.
        background = _layer_to_pil(background_image[0])
        canvas = background.copy()
        requested_count = max(0, min(int(layer_count), _MAX_LAYER_STACK_INPUTS))
        layer_images = {}
        layer_sizes = []
        for index in range(1, requested_count + 1):
            layer = kwargs.get(f"layer_{index}")
            if layer is None:
                layer_sizes.append((0, 0))
                continue
            layer_image = _layer_to_pil(layer[0])
            layer_images[index] = layer_image
            layer_sizes.append((layer_image.width, layer_image.height))

        layer_positions = _parse_layer_positions(
            positions,
            requested_count,
            background.width,
            background.height,
            layer_sizes=layer_sizes,
        )
        used_count = 0
        info_lines = []

        for index in range(1, requested_count + 1):
            layer_image = layer_images.get(index)
            if layer_image is None:
                info_lines.append(f"layer_{index}: missing")
                continue

            x, y = layer_positions[index - 1]
            pasted = self._paste_clipped(canvas, layer_image, x, y)
            if pasted:
                used_count += 1
                info_lines.append(
                    f"layer_{index}: x={x}, y={y}, w={layer_image.width}, h={layer_image.height}"
                )
            else:
                info_lines.append(
                    f"layer_{index}: outside canvas x={x}, y={y}, w={layer_image.width}, h={layer_image.height}"
                )

        return (_pil_to_tensor(canvas), used_count, "\n".join(info_lines))


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
            pad_color = _parse_hex_color(pad_color_hex, (0, 0, 0))
            result = _pad_samples(resized, target_width, target_height, pad_color)
        else:
            actual_width, actual_height = target_width, target_height
            result = comfy.utils.common_upscale(samples, actual_width, actual_height, upscale_method, "disabled")

        return (result.movedim(1, -1), actual_width, actual_height)


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
            pil_a = _tensor_to_pil(tensor_a)
            pil_b = _resize_image(_tensor_to_pil(tensor_b), pil_a.size, resize_method)
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

            compared.append(_pil_to_tensor(canvas))

        return (torch.cat(compared, dim=0),)


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
        return image.crop(sample_box).resize((width, frame_height), _RESAMPLING.BILINEAR).convert("RGB")

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
        frame_color = _parse_hex_color(frame_color_hex, (17, 17, 17))
        text_color = _parse_hex_color(text_color_hex, (255, 255, 255))
        outline_color = _parse_hex_color(outline_color_hex, (0, 0, 0))

        labeled = []
        for item in image:
            pil_image = _tensor_to_pil(item)
            width, height = pil_image.size
            band = self._band_image(pil_image, position, frame_height, fill_enabled, frame_color)

            if overlay_on_image:
                canvas = pil_image.copy()
                max_band_height = min(frame_height, height)
                if band.height != max_band_height:
                    band = band.resize((width, max_band_height), _RESAMPLING.BILINEAR)
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
            labeled.append(_pil_to_tensor(canvas))

        return (torch.cat(labeled, dim=0),)


NODE_CLASS_MAPPINGS = {
    "NH_SaveImagePath": NH_SaveImagePath,
    "NH_LoadImagesFromFolder": NH_LoadImagesFromFolder,
    "NH_LoadImagesMatching": NH_LoadImagesMatching,
    "NH_LayerLayoutComposite": NH_LayerLayoutComposite,
    "NH_LayerStackComposite": NH_LayerStackComposite,
    "NH_ImageResizeByUnit": NH_ImageResizeByUnit,
    "NH_ImageCompare": NH_ImageCompare,
    "NH_ImageLabel": NH_ImageLabel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_SaveImagePath": "Save Image Path (NH)",
    "NH_LoadImagesFromFolder": "Load Images Folder (NH)",
    "NH_LoadImagesMatching": "Load images matching",
    "NH_LayerLayoutComposite": "Layer Layout Composite (NH)",
    "NH_LayerStackComposite": "Layer Stack Composite (NH)",
    "NH_ImageResizeByUnit": "Image Resize Unit (NH)",
    "NH_ImageCompare": "Image Compare (NH)",
    "NH_ImageLabel": "Image Label (NH)",
}
