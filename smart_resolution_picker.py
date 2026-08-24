"""NH Smart Resolution Picker node."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .resolution_data import (
    ASPECT_SORT_ORDER,
    DEFAULT_MODEL_LABEL,
    DEFAULT_PRESET,
    MODEL_ID_BY_LABEL,
    MODEL_LABELS,
    MODEL_SPECS,
    RESOLUTION_PRESETS,
    PRESET_LABELS,
    PRESET_LABELS_BY_MODEL,
    PRESET_LOOKUP_BY_MODEL,
)


TARGET_RESOLUTION_LEVELS = [f"{mp} MP" for mp in range(1, 13)]
RESIZE_METHODS = ["lanczos", "bicubic", "bilinear", "nearest", "nearest-exact", "area", "box", "hamming"]
RESIZE_DATA_TYPE = "NH_RESIZE_DATA"
TARGET_RATIO_LABELS = sorted(
    {entry["aspect"] for entry in RESOLUTION_PRESETS},
    key=lambda aspect: ASPECT_SORT_ORDER.get(aspect, 99),
)
TARGET_MP_VALUES = {
    f"{mp} MP": float(mp)
    for mp in range(1, 13)
}

Z_IMAGE_TIER_MAP = {
    "1 MP": ("Tier 1024",),
    "2 MP": ("Tier 1280",),
    "3 MP": ("Tier 1536",),
    "4 MP": ("Tier 2K",),
}

_RESAMPLING = getattr(Image, "Resampling", Image)
_PIL_RESAMPLE_METHODS = {
    "lanczos": _RESAMPLING.LANCZOS,
    "box": _RESAMPLING.BOX,
    "hamming": _RESAMPLING.HAMMING,
}


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


def _snap_half_up(value, multiple_of):
    multiple_of = max(1, int(multiple_of))
    return int(math.floor((float(value) / multiple_of) + 0.5) * multiple_of)


def _ratio_label(width, height):
    width = max(1, int(width))
    height = max(1, int(height))
    divisor = math.gcd(width, height)
    ratio_w = width // divisor
    ratio_h = height // divisor
    if ratio_w > 100 or ratio_h > 100:
        return f"{width / height:.3f}:1"
    return f"{ratio_w}:{ratio_h}"


def _parse_hex_color(value, default=(255, 255, 255)):
    text = str(value or "").strip().lstrip("#")
    try:
        if len(text) == 3:
            text = "".join(char * 2 for char in text)
        if len(text) != 6:
            raise ValueError
        return tuple(int(text[index:index + 2], 16) for index in (0, 2, 4))
    except Exception:
        return default


def _needs_warning(entry, extra_warnings):
    if extra_warnings:
        return True
    if entry is None:
        return False
    text = f"{entry.get('tier', '')} {entry.get('reliability', '')} {entry.get('warning', '')}".casefold()
    return any(marker in text for marker in ("community", "extrapolation", "rủi ro", "risk"))


def _build_info(model_spec, tier, ratio, width, height, entry=None, extra_warnings=None):
    extra_warnings = extra_warnings or []
    mp = (int(width) * int(height)) / 1_000_000
    prefix = "⚠️ " if _needs_warning(entry, extra_warnings) else ""
    info = f"{prefix}{model_spec['name']} | {tier} | {ratio} | {int(width)}×{int(height)} | {mp:.2f}MP"
    warning = entry.get("warning") if entry else ""
    if warning:
        info += f"\nWarning: {warning}"
    note = model_spec.get("note")
    if note:
        info += f"\nNote: {note}"
    if extra_warnings:
        info += "\nAdjustment: " + "; ".join(extra_warnings)
    return info


def _custom_resolution(custom_width, custom_height, model_spec):
    original = (int(custom_width), int(custom_height))
    multiple_of = int(model_spec["multiple_of"])
    min_width, min_height = model_spec["min_res"]
    max_width, max_height = model_spec["max_res"]

    width = _snap_half_up(original[0], multiple_of)
    height = _snap_half_up(original[1], multiple_of)
    warnings = []
    if (width, height) != original:
        warnings.append(f"snapped to multiple of {multiple_of}: {original[0]}×{original[1]} -> {width}×{height}")

    clamped_width = _clamp(width, min_width, max_width)
    clamped_height = _clamp(height, min_height, max_height)
    if (clamped_width, clamped_height) != (width, height):
        warnings.append(f"clamped to range {min_width}×{min_height}..{max_width}×{max_height}")
    return clamped_width, clamped_height, warnings


def _entry_reliability_rank(entry):
    if entry.get("sweet"):
        return 0
    reliability = entry.get("reliability", "")
    if reliability == "verified":
        return 1
    if reliability == "community":
        return 2
    if reliability == "extrapolation":
        return 3
    return 4


def _entry_matches_resolution_level(entry, resolution_level):
    tier = str(entry.get("tier", "")).replace("⭐", "").strip()
    if entry.get("model_id") == "z_image":
        return any(tier.startswith(prefix) for prefix in Z_IMAGE_TIER_MAP.get(resolution_level, ()))
    return tier.startswith(resolution_level)


def _extrapolated_ratio_entry(model_family, resolution_level, aspect):
    target_mp = TARGET_MP_VALUES.get(resolution_level, 1.0)
    target_pixels = max(1.0, target_mp * 1_000_000)
    ratio = _ratio_value(aspect)
    model_id = MODEL_ID_BY_LABEL.get(model_family, MODEL_ID_BY_LABEL[DEFAULT_MODEL_LABEL])
    model_spec = MODEL_SPECS[model_id]
    multiple_of = int(model_spec.get("multiple_of", 16))

    width = max(multiple_of, _snap_half_up(math.sqrt(target_pixels * ratio), multiple_of))
    height = max(multiple_of, _snap_half_up(math.sqrt(target_pixels / ratio), multiple_of))
    actual_mp = (width * height) / 1_000_000
    warning = (
        f"Extrapolated {resolution_level} preset generated from ratio {aspect}; "
        f"actual size is {actual_mp:.2f}MP and may exceed the model's usual tested range."
    )

    return {
        "model_id": model_id,
        "model": model_family,
        "tier": f"{resolution_level} extrapolated",
        "aspect": aspect,
        "width": int(width),
        "height": int(height),
        "reliability": "extrapolation",
        "warning": warning,
        "sweet": False,
        "label": f"{resolution_level} extrapolated | {aspect} | {int(width)}×{int(height)}",
    }


def _extrapolated_resolution_candidates(model_family, resolution_level):
    model_entries = [entry for entry in RESOLUTION_PRESETS if entry["model"] == model_family]
    aspects = sorted(
        {entry["aspect"] for entry in model_entries},
        key=lambda aspect: ASPECT_SORT_ORDER.get(aspect, 99),
    )
    return [_extrapolated_ratio_entry(model_family, resolution_level, aspect) for aspect in aspects]


def _resolution_candidates(model_family, resolution_level):
    model_entries = [entry for entry in RESOLUTION_PRESETS if entry["model"] == model_family]
    candidates = [entry for entry in model_entries if _entry_matches_resolution_level(entry, resolution_level)]
    if candidates:
        return candidates

    target_mp = TARGET_MP_VALUES.get(resolution_level, 1.0)
    if target_mp > 4.0:
        return _extrapolated_resolution_candidates(model_family, resolution_level)

    return sorted(
        model_entries,
        key=lambda entry: (abs(((entry["width"] * entry["height"]) / 1_000_000) - target_mp), _entry_reliability_rank(entry)),
    )[:12]


def _pick_closest_ratio_entry(model_family, resolution_level, source_width, source_height):
    source_ratio = max(float(source_width), 1.0) / max(float(source_height), 1.0)
    target_mp = TARGET_MP_VALUES.get(resolution_level, 1.0)
    candidates = _resolution_candidates(model_family, resolution_level)
    if not candidates:
        raise ValueError(f"No resolution candidates for {model_family} / {resolution_level}")

    def score(entry):
        entry_ratio = max(float(entry["width"]), 1.0) / max(float(entry["height"]), 1.0)
        ratio_distance = abs(math.log(entry_ratio / source_ratio))
        mp_distance = abs(((entry["width"] * entry["height"]) / 1_000_000) - target_mp)
        return (ratio_distance, _entry_reliability_rank(entry), mp_distance, entry["width"] * entry["height"])

    return min(candidates, key=score)


def _ratio_value(ratio_label):
    try:
        width_text, height_text = str(ratio_label).split(":", 1)
        return max(float(width_text), 1.0) / max(float(height_text), 1.0)
    except Exception:
        return 1.0


def _pick_requested_ratio_entry(model_family, resolution_level, target_ratio):
    requested_ratio = _ratio_value(target_ratio)
    target_mp = TARGET_MP_VALUES.get(resolution_level, 1.0)
    candidates = _resolution_candidates(model_family, resolution_level)
    if not candidates:
        raise ValueError(f"No resolution candidates for {model_family} / {resolution_level}")

    exact_candidates = [entry for entry in candidates if entry["aspect"] == target_ratio]
    selectable = exact_candidates or candidates

    def score(entry):
        entry_ratio = _ratio_value(entry["aspect"])
        ratio_distance = abs(math.log(entry_ratio / requested_ratio))
        mp_distance = abs(((entry["width"] * entry["height"]) / 1_000_000) - target_mp)
        return (ratio_distance, _entry_reliability_rank(entry), mp_distance, entry["width"] * entry["height"])

    entry = min(selectable, key=score)
    return entry, bool(exact_candidates)


def _normalize_resize_mode(resize_mode):
    aliases = {
        "crop": "cover_crop",
        "cover": "cover_crop",
        "cover_crop": "cover_crop",
        "pad": "contain_pad",
        "contain": "contain_pad",
        "contain_pad": "contain_pad",
        "fill": "stretch",
        "stretch": "stretch",
    }
    return aliases.get(str(resize_mode), "cover_crop")


def _build_resize_data(source_width, source_height, target_width, target_height, resize_mode, resize_method, pad_color_hex):
    source_width = max(1, int(source_width))
    source_height = max(1, int(source_height))
    target_width = max(1, int(target_width))
    target_height = max(1, int(target_height))
    normalized_mode = _normalize_resize_mode(resize_mode)

    data = {
        "version": 1,
        "source_width": source_width,
        "source_height": source_height,
        "target_width": target_width,
        "target_height": target_height,
        "resize_mode": normalized_mode,
        "requested_resize_mode": str(resize_mode),
        "resize_method": str(resize_method),
        "pad_color_hex": str(pad_color_hex),
        "scale_x": float(target_width) / float(source_width),
        "scale_y": float(target_height) / float(source_height),
        "scale": None,
        "resized_width": target_width,
        "resized_height": target_height,
        "content_left": 0,
        "content_top": 0,
        "content_right": target_width,
        "content_bottom": target_height,
        "pad_left": 0,
        "pad_top": 0,
        "pad_right": 0,
        "pad_bottom": 0,
        "crop_left": 0,
        "crop_top": 0,
        "crop_right": 0,
        "crop_bottom": 0,
    }

    if normalized_mode == "stretch":
        return data

    scale_x = float(target_width) / float(source_width)
    scale_y = float(target_height) / float(source_height)
    scale = max(scale_x, scale_y) if normalized_mode == "cover_crop" else min(scale_x, scale_y)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    data.update({
        "scale": float(scale),
        "resized_width": resized_width,
        "resized_height": resized_height,
    })

    if normalized_mode == "cover_crop":
        left = max(0, (resized_width - target_width) // 2)
        top = max(0, (resized_height - target_height) // 2)
        data.update({
            "crop_left": left,
            "crop_top": top,
            "crop_right": max(0, resized_width - target_width - left),
            "crop_bottom": max(0, resized_height - target_height - top),
        })
        return data

    left = max(0, (target_width - resized_width) // 2)
    top = max(0, (target_height - resized_height) // 2)
    right = target_width - left - resized_width
    bottom = target_height - top - resized_height
    data.update({
        "content_left": left,
        "content_top": top,
        "content_right": left + resized_width,
        "content_bottom": top + resized_height,
        "pad_left": left,
        "pad_top": top,
        "pad_right": max(0, right),
        "pad_bottom": max(0, bottom),
    })
    return data


def _resize_bchw_pil(samples, height, width, resize_method):
    original_dtype = samples.dtype
    original_device = samples.device
    resample = _PIL_RESAMPLE_METHODS[resize_method]
    resized_items = []

    for sample in samples:
        array = sample.movedim(0, -1).detach().cpu().numpy()
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        if array.shape[-1] == 1:
            pil_image = Image.fromarray(array[:, :, 0], mode="L")
        else:
            pil_image = Image.fromarray(array)

        pil_image = pil_image.resize((int(width), int(height)), resample)
        resized_array = np.array(pil_image).astype(np.float32) / 255.0
        if resized_array.ndim == 2:
            resized_array = resized_array[:, :, None]
        resized_items.append(torch.from_numpy(resized_array).movedim(-1, 0))

    return torch.stack(resized_items, dim=0).to(device=original_device, dtype=original_dtype)


def _resize_bchw(samples, height, width, resize_method):
    resize_method = resize_method if resize_method in RESIZE_METHODS else "lanczos"
    if resize_method in _PIL_RESAMPLE_METHODS:
        return _resize_bchw_pil(samples, height, width, resize_method)

    if resize_method in {"bilinear", "bicubic"}:
        resized = F.interpolate(samples, size=(int(height), int(width)), mode=resize_method, align_corners=False)
    else:
        resized = F.interpolate(samples, size=(int(height), int(width)), mode=resize_method)
    return resized.clamp(0.0, 1.0)


def _resize_image_batch_with_data(images, target_width, target_height, resize_mode, resize_method, pad_color_hex):
    if int(target_width) <= 0 or int(target_height) <= 0:
        raise ValueError("Target width and height must be greater than 0")

    batch, source_height, source_width, channels = images.shape
    resize_data = _build_resize_data(
        source_width,
        source_height,
        target_width,
        target_height,
        resize_mode,
        resize_method,
        pad_color_hex,
    )
    resize_mode = resize_data["resize_mode"]
    samples = images.movedim(-1, 1)

    if resize_mode == "stretch":
        resized = _resize_bchw(samples, target_height, target_width, resize_method).movedim(1, -1)
        return resized, resize_data

    resized_width = int(resize_data["resized_width"])
    resized_height = int(resize_data["resized_height"])
    resized = _resize_bchw(samples, resized_height, resized_width, resize_method)

    if resize_mode == "cover_crop":
        left = int(resize_data["crop_left"])
        top = int(resize_data["crop_top"])
        cropped = resized[:, :, top:top + int(target_height), left:left + int(target_width)]
        return cropped.movedim(1, -1), resize_data

    fill_rgb = _parse_hex_color(pad_color_hex, (255, 255, 255))
    canvas = torch.empty(
        (batch, channels, int(target_height), int(target_width)),
        dtype=images.dtype,
        device=images.device,
    )
    for channel in range(channels):
        canvas[:, channel, :, :] = fill_rgb[min(channel, 2)] / 255.0

    left = int(resize_data["content_left"])
    top = int(resize_data["content_top"])
    canvas[:, :, top:top + resized_height, left:left + resized_width] = resized
    return canvas.movedim(1, -1), resize_data


def _resize_image_batch(images, target_width, target_height, resize_mode, resize_method, pad_color_hex):
    resized, _resize_data = _resize_image_batch_with_data(
        images,
        target_width,
        target_height,
        resize_mode,
        resize_method,
        pad_color_hex,
    )
    return resized


def _scaled_content_bounds(resize_data, image_width, image_height):
    target_width = max(1, int(resize_data.get("target_width", image_width)))
    target_height = max(1, int(resize_data.get("target_height", image_height)))
    scale_x = float(image_width) / float(target_width)
    scale_y = float(image_height) / float(target_height)

    left = int(round(float(resize_data.get("content_left", 0)) * scale_x))
    top = int(round(float(resize_data.get("content_top", 0)) * scale_y))
    right = int(round(float(resize_data.get("content_right", target_width)) * scale_x))
    bottom = int(round(float(resize_data.get("content_bottom", target_height)) * scale_y))

    left = _clamp(left, 0, max(0, int(image_width) - 1))
    top = _clamp(top, 0, max(0, int(image_height) - 1))
    right = _clamp(right, left + 1, int(image_width))
    bottom = _clamp(bottom, top + 1, int(image_height))
    return left, top, right, bottom


def _has_padding(resize_data):
    return any(int(resize_data.get(key, 0)) > 0 for key in ("pad_left", "pad_top", "pad_right", "pad_bottom"))


class NH_SmartResolutionPicker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_family": (MODEL_LABELS, {"default": DEFAULT_MODEL_LABEL}),
                "preset": (PRESET_LABELS, {"default": DEFAULT_PRESET}),
            },
            "optional": {
                "use_custom": ("BOOLEAN", {"default": False}),
                "custom_width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "custom_height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "swap_orientation": ("BOOLEAN", {"default": False}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "LATENT", "STRING", "FLOAT")
    RETURN_NAMES = ("width", "height", "latent", "info", "aspect_ratio")
    FUNCTION = "pick"
    CATEGORY = "NH-Nodes/Resolution"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def pick(
        self,
        model_family=DEFAULT_MODEL_LABEL,
        preset=DEFAULT_PRESET,
        use_custom=False,
        custom_width=1024,
        custom_height=1024,
        swap_orientation=False,
        batch_size=1,
    ):
        if model_family not in MODEL_ID_BY_LABEL:
            model_family = DEFAULT_MODEL_LABEL

        preset_lookup = PRESET_LOOKUP_BY_MODEL.get(model_family, {})
        model_presets = PRESET_LABELS_BY_MODEL.get(model_family, [])
        default_preset = DEFAULT_PRESET if model_family == DEFAULT_MODEL_LABEL else model_presets[0]
        entry = preset_lookup.get(preset) or preset_lookup[default_preset]
        model_spec = MODEL_SPECS[entry["model_id"]]
        extra_warnings = []

        if use_custom:
            width, height, extra_warnings = _custom_resolution(custom_width, custom_height, model_spec)
            tier = "Custom"
            ratio = _ratio_label(width, height)
            entry_for_warning = None
        else:
            width = int(entry["width"])
            height = int(entry["height"])
            tier = entry["tier"]
            ratio = entry["aspect"]
            entry_for_warning = entry

        if swap_orientation:
            width, height = height, width
            ratio = _ratio_label(width, height)
            extra_warnings.append("orientation swapped")

        requested_batch = int(batch_size)
        batch_size = _clamp(requested_batch, 1, 64)
        if batch_size != requested_batch:
            extra_warnings.append(f"batch size clamped: {requested_batch} -> {batch_size}")

        channels = int(model_spec.get("latent_channels", 16))
        latent_height = max(1, int(height) // 8)
        latent_width = max(1, int(width) // 8)
        latent = {"samples": torch.zeros((batch_size, channels, latent_height, latent_width), dtype=torch.float32)}
        aspect_ratio = float(width) / max(float(height), 1.0)
        info = _build_info(model_spec, tier, ratio, width, height, entry_for_warning, extra_warnings)

        return (int(width), int(height), latent, info, aspect_ratio)


class NH_SmartRatioImageResize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_family": (MODEL_LABELS, {"default": DEFAULT_MODEL_LABEL}),
                "resolution_level": (TARGET_RESOLUTION_LEVELS, {"default": "1 MP"}),
                "resize_mode": (["cover_crop", "contain_pad", "stretch"], {"default": "cover_crop"}),
                "resize_method": (RESIZE_METHODS, {"default": "lanczos"}),
                "pad_color_hex": ("STRING", {"default": "#FFFFFF", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING", "FLOAT", "FLOAT", "STRING", RESIZE_DATA_TYPE)
    RETURN_NAMES = (
        "image",
        "width",
        "height",
        "info",
        "source_aspect_ratio",
        "target_aspect_ratio",
        "selected_preset",
        "resize_data",
    )
    FUNCTION = "resize"
    CATEGORY = "NH-Nodes/Resolution"

    def resize(
        self,
        image,
        model_family=DEFAULT_MODEL_LABEL,
        resolution_level="1 MP",
        resize_mode="cover_crop",
        resize_method="lanczos",
        pad_color_hex="#FFFFFF",
    ):
        if model_family not in MODEL_ID_BY_LABEL:
            model_family = DEFAULT_MODEL_LABEL
        if resolution_level not in TARGET_RESOLUTION_LEVELS:
            resolution_level = "1 MP"
        if resize_method not in RESIZE_METHODS:
            resize_method = "lanczos"

        source_height = int(image.shape[1])
        source_width = int(image.shape[2])
        entry = _pick_closest_ratio_entry(model_family, resolution_level, source_width, source_height)
        target_width = int(entry["width"])
        target_height = int(entry["height"])

        resized, resize_data = _resize_image_batch_with_data(
            image,
            target_width,
            target_height,
            resize_mode,
            resize_method,
            pad_color_hex,
        )
        model_spec = MODEL_SPECS[entry["model_id"]]
        source_aspect = float(source_width) / max(float(source_height), 1.0)
        target_aspect = float(target_width) / max(float(target_height), 1.0)
        ratio_delta = abs(math.log(target_aspect / source_aspect))
        info = _build_info(
            model_spec,
            entry["tier"],
            entry["aspect"],
            target_width,
            target_height,
            entry,
            [
                f"source {source_width}×{source_height} aspect={source_aspect:.4f}",
                f"selected closest ratio delta={ratio_delta:.4f}",
                f"resize_mode={resize_mode}",
                f"resize_method={resize_method}",
            ],
        )

        return (
            resized,
            target_width,
            target_height,
            info,
            source_aspect,
            target_aspect,
            entry["label"],
            resize_data,
        )


class NH_RatioPresetImageResize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_family": (MODEL_LABELS, {"default": DEFAULT_MODEL_LABEL}),
                "resolution_level": (TARGET_RESOLUTION_LEVELS, {"default": "1 MP"}),
                "target_ratio": (TARGET_RATIO_LABELS, {"default": "1:1"}),
                "resize_mode": (["crop", "pad", "fill"], {"default": "crop"}),
                "resize_method": (RESIZE_METHODS, {"default": "lanczos"}),
                "pad_color_hex": ("STRING", {"default": "#FFFFFF", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING", "FLOAT", "FLOAT", "STRING", RESIZE_DATA_TYPE)
    RETURN_NAMES = (
        "image",
        "width",
        "height",
        "info",
        "source_aspect_ratio",
        "target_aspect_ratio",
        "selected_preset",
        "resize_data",
    )
    FUNCTION = "resize"
    CATEGORY = "NH-Nodes/Resolution"

    def resize(
        self,
        image,
        model_family=DEFAULT_MODEL_LABEL,
        resolution_level="1 MP",
        target_ratio="1:1",
        resize_mode="crop",
        resize_method="lanczos",
        pad_color_hex="#FFFFFF",
    ):
        if model_family not in MODEL_ID_BY_LABEL:
            model_family = DEFAULT_MODEL_LABEL
        if resolution_level not in TARGET_RESOLUTION_LEVELS:
            resolution_level = "1 MP"
        if target_ratio not in TARGET_RATIO_LABELS:
            target_ratio = "1:1"
        if resize_method not in RESIZE_METHODS:
            resize_method = "lanczos"

        source_height = int(image.shape[1])
        source_width = int(image.shape[2])
        entry, exact_match = _pick_requested_ratio_entry(model_family, resolution_level, target_ratio)
        target_width = int(entry["width"])
        target_height = int(entry["height"])

        resized, resize_data = _resize_image_batch_with_data(
            image,
            target_width,
            target_height,
            resize_mode,
            resize_method,
            pad_color_hex,
        )
        model_spec = MODEL_SPECS[entry["model_id"]]
        source_aspect = float(source_width) / max(float(source_height), 1.0)
        target_aspect = float(target_width) / max(float(target_height), 1.0)
        requested_aspect = _ratio_value(target_ratio)
        selected_delta = abs(math.log(target_aspect / requested_aspect))
        notes = [
            f"source {source_width}×{source_height} aspect={source_aspect:.4f}",
            f"requested_ratio={target_ratio}",
            f"selected_ratio={entry['aspect']}",
            f"resize_mode={resize_mode}",
            f"resize_method={resize_method}",
        ]
        if not exact_match:
            notes.insert(2, f"requested ratio unavailable in this model/tier; closest delta={selected_delta:.4f}")

        info = _build_info(
            model_spec,
            entry["tier"],
            entry["aspect"],
            target_width,
            target_height,
            entry,
            notes,
        )

        return (
            resized,
            target_width,
            target_height,
            info,
            source_aspect,
            target_aspect,
            entry["label"],
            resize_data,
        )


class NH_RemoveResizePadding:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resize_data": (RESIZE_DATA_TYPE,),
                "output_size": (["content_region", "original_size"], {"default": "content_region"}),
                "resize_method": (RESIZE_METHODS, {"default": "lanczos"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "info")
    FUNCTION = "remove_padding"
    CATEGORY = "NH-Nodes/Resolution"

    def remove_padding(self, image, resize_data, output_size="content_region", resize_method="lanczos"):
        if resize_method not in RESIZE_METHODS:
            resize_method = "lanczos"

        image_height = int(image.shape[1])
        image_width = int(image.shape[2])
        if not isinstance(resize_data, dict):
            return (
                image,
                image_width,
                image_height,
                f"No resize_data metadata found; passed through {image_width}×{image_height}.",
            )

        left, top, right, bottom = _scaled_content_bounds(resize_data, image_width, image_height)
        cropped = image[:, top:bottom, left:right, :]

        source_width = int(resize_data.get("source_width", cropped.shape[2]))
        source_height = int(resize_data.get("source_height", cropped.shape[1]))
        if output_size == "original_size":
            cropped = _resize_image_batch(cropped, source_width, source_height, "stretch", resize_method, "#000000")

        output_height = int(cropped.shape[1])
        output_width = int(cropped.shape[2])
        target_width = int(resize_data.get("target_width", image_width))
        target_height = int(resize_data.get("target_height", image_height))
        pad_text = (
            f"pad L{int(resize_data.get('pad_left', 0))} "
            f"T{int(resize_data.get('pad_top', 0))} "
            f"R{int(resize_data.get('pad_right', 0))} "
            f"B{int(resize_data.get('pad_bottom', 0))}"
        )
        if _has_padding(resize_data):
            action = "removed padding"
        else:
            action = "no padding detected"

        info = (
            f"{action}; source {source_width}×{source_height}; "
            f"resize target {target_width}×{target_height}; "
            f"input {image_width}×{image_height}; crop box ({left},{top})-({right},{bottom}); "
            f"output {output_width}×{output_height}; {pad_text}"
        )
        return (cropped, output_width, output_height, info)


NODE_CLASS_MAPPINGS = {
    "NH_SmartResolutionPicker": NH_SmartResolutionPicker,
    "NH_SmartRatioImageResize": NH_SmartRatioImageResize,
    "NH_RatioPresetImageResize": NH_RatioPresetImageResize,
    "NH_RemoveResizePadding": NH_RemoveResizePadding,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_SmartResolutionPicker": "NH Smart Resolution Picker",
    "NH_SmartRatioImageResize": "NH Smart Ratio Image Resize",
    "NH_RatioPresetImageResize": "NH Ratio Preset Image Resize",
    "NH_RemoveResizePadding": "NH Remove Resize Padding",
}
