import os
import re
import unicodedata
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFont, ImageOps

import comfy.model_management
import comfy.utils
import folder_paths
import node_helpers


_RESAMPLING = getattr(Image, "Resampling", Image)
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".bmp", ".tif", ".tiff"}
_FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}


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
    try:
        return ImageColor.getrgb((value or "").strip())
    except Exception:
        return default


def _tensor_to_pil(image_tensor):
    array = np.clip(255.0 * image_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _pil_to_tensor(image):
    array = np.array(image).astype(np.float32) / 255.0
    dtype = comfy.model_management.intermediate_dtype()
    return torch.from_numpy(array).unsqueeze(0).to(dtype=dtype)


def _load_font(font_name, font_size):
    if font_name and font_name != "default":
        font_path = Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts" / font_name
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), font_size)
            except Exception:
                pass
    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        return ImageFont.load_default()


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
    batch, channels, resized_h, resized_w = samples.shape
    canvas = torch.empty((batch, channels, target_height, target_width), dtype=samples.dtype, device=samples.device)
    fill_values = [channel / 255.0 for channel in fill_rgb]
    for channel in range(channels):
        canvas[:, channel, :, :] = fill_values[min(channel, len(fill_values) - 1)]

    offset_x = max(0, (target_width - resized_w) // 2)
    offset_y = max(0, (target_height - resized_h) // 2)
    canvas[:, :, offset_y:offset_y + resized_h, offset_x:offset_x + resized_w] = samples
    return canvas


def _save_dir_from_text(folder_path):
    folder_path = (folder_path or "").strip()
    if not folder_path:
        return folder_paths.get_output_directory()
    if os.path.isfile(folder_path):
        return os.path.dirname(folder_path)
    if os.path.isabs(folder_path):
        return folder_path
    return os.path.join(folder_paths.get_output_directory(), folder_path)


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
        for current_root, dir_names, file_names in os.walk(folder_path):
            dir_names.sort(key=_natural_key)
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


def _resolve_matching_folder(root_path, match_text, match_mode, recursive):
    root_path = (root_path or "").strip()
    if not root_path:
        raise ValueError("Matching root path is empty")
    if not os.path.isabs(root_path):
        root_path = os.path.join(folder_paths.get_input_directory(), root_path)
    if not os.path.isdir(root_path):
        raise ValueError(f"Matching root folder does not exist: {root_path}")

    normalized_match = _normalize_match_value(match_text)
    if not normalized_match:
        raise ValueError("Match text is empty")

    candidates = [root_path]
    if recursive:
        for current_root, dir_names, _file_names in os.walk(root_path):
            dir_names.sort(key=_natural_key)
            for dir_name in dir_names:
                candidates.append(os.path.join(current_root, dir_name))
    else:
        candidates.extend(
            os.path.join(root_path, entry)
            for entry in sorted(os.listdir(root_path), key=_natural_key)
            if os.path.isdir(os.path.join(root_path, entry))
        )

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
        raise ValueError(f"No matching folder found for '{match_text}' under: {root_path}")

    matches.sort(key=lambda item: _natural_key(os.path.relpath(item, root_path)))
    return matches[0]


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

    def save_images(self, images, folder_path, filename_prefix, number_suffix=True, preview_image=False, file_format="png", dpi=300):
        save_dir = _save_dir_from_text(folder_path)
        os.makedirs(save_dir, exist_ok=True)

        prefix = (filename_prefix or "NH").strip() or "NH"
        numbered_pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.(?:png|jpe?g|webp|avif)$", re.IGNORECASE)
        legacy_numbered_pattern = re.compile(rf"^(?:{re.escape(prefix)})?(\d+)\.(?:png|jpe?g|webp|avif)$", re.IGNORECASE)
        bare_pattern = re.compile(rf"^{re.escape(prefix)}\.(?:png|jpe?g|webp|avif)$", re.IGNORECASE)

        used_numbers = set()
        bare_exists = False
        for name in os.listdir(save_dir):
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
                filename = f"{prefix}{next_number:04d}.{file_format}"
                used_numbers.add(next_number)
                next_number += 1
            elif not bare_exists and not used_numbers:
                filename = f"{prefix}.{file_format}"
                bare_exists = True
            else:
                while next_number in used_numbers:
                    next_number += 1
                filename = f"{prefix}{next_number}.{file_format}"
                used_numbers.add(next_number)
                next_number += 1

            pil_image = _tensor_to_pil(image)
            _save_pil_image(pil_image, os.path.join(save_dir, filename), file_format, int(dpi))
            saved_files.append(filename)

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
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "filename", "path")
    FUNCTION = "load_images"
    CATEGORY = "NH-Nodes/Image"

    @classmethod
    def IS_CHANGED(cls, folder_path, index, step, seed_mode, recursive=False, **kwargs):
        return float("nan")

    def load_images(self, folder_path, index, step, seed_mode, recursive=False, **kwargs):
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
            output_paths.append(full_path)

        return (torch.cat(output_images, dim=0), "\n".join(selected_files), "\n".join(output_paths))


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
            },
            "optional": {
                "source_file_names": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("images", "filename", "path", "matched_text", "found")
    FUNCTION = "load_matching"
    CATEGORY = "NH-Nodes/Image"

    @classmethod
    def IS_CHANGED(cls, source_folder_path, matching_root_path, match_text, match_mode, recursive, index, step, repeat_count, source_file_names="", **kwargs):
        return float("nan")

    def load_matching(self, source_folder_path, matching_root_path, match_text, match_mode, recursive, index, step, repeat_count, source_file_names="", **kwargs):
        if not source_folder_path and kwargs.get("person_folder_path"):
            source_folder_path = kwargs["person_folder_path"]

        first_source_path = str(source_folder_path or "").splitlines()[0].strip()
        resolved_source_folder = _resolve_load_dir(first_source_path)
        selected_match_text = (match_text or "").strip()
        source_names = [line.strip() for line in str(source_file_names or "").splitlines() if line.strip()]
        if selected_match_text:
            match_values = [selected_match_text]
        elif source_names:
            match_values = []
            for source_name in source_names:
                source_dir = os.path.dirname(source_name)
                match_values.append(os.path.basename(os.path.normpath(source_dir)) if source_dir else os.path.basename(os.path.normpath(resolved_source_folder)))
        else:
            selected_match_text = os.path.basename(os.path.normpath(resolved_source_folder))
            match_values = [selected_match_text]

        selected_files = []
        matched_folders = []
        output_images = []
        target_size = None

        for current_match_text in match_values:
            matched_folder = _resolve_matching_folder(matching_root_path, current_match_text, match_mode, recursive)
            image_files = _collect_image_files(matched_folder)
            start_index = index % len(image_files)
            indices = [((start_index + offset) % len(image_files)) for offset in range(step)]

            for image_index in indices:
                file_name = image_files[image_index]
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

        images = torch.cat(output_images, dim=0)
        if repeat_count > 1:
            images = images.repeat(repeat_count, 1, 1, 1)

        return (images, "\n".join(selected_files), "\n".join(matched_folders), "\n".join(match_values), True)


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
            font = _load_font(font_name, font_size)
            bbox = draw.multiline_textbbox((0, 0), label_text, font=font, stroke_width=outline_width)
            current_size = font_size

            if fit_text:
                max_width = max(1, width - (padding_x * 2))
                while (bbox[2] - bbox[0]) > max_width and current_size > 8:
                    current_size -= 1
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
    "NH_ImageResizeByUnit": NH_ImageResizeByUnit,
    "NH_ImageCompare": NH_ImageCompare,
    "NH_ImageLabel": NH_ImageLabel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_SaveImagePath": "Save Image Path (NH)",
    "NH_LoadImagesFromFolder": "Load Images Folder (NH)",
    "NH_LoadImagesMatching": "Load images matching",
    "NH_ImageResizeByUnit": "Image Resize Unit (NH)",
    "NH_ImageCompare": "Image Compare (NH)",
    "NH_ImageLabel": "Image Label (NH)",
}
