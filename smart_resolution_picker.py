"""NH Smart Resolution Picker node."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .resolution_data import (
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


TARGET_RESOLUTION_LEVELS = ["1 MP", "2 MP", "3 MP", "4 MP"]
RESIZE_METHODS = ["lanczos", "bicubic", "bilinear", "nearest", "nearest-exact", "area", "box", "hamming"]
TARGET_MP_VALUES = {
    "1 MP": 1.0,
    "2 MP": 2.0,
    "3 MP": 3.0,
    "4 MP": 4.0,
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


def _resolution_candidates(model_family, resolution_level):
    model_entries = [entry for entry in RESOLUTION_PRESETS if entry["model"] == model_family]
    candidates = [entry for entry in model_entries if _entry_matches_resolution_level(entry, resolution_level)]
    if candidates:
        return candidates

    target_mp = TARGET_MP_VALUES.get(resolution_level, 1.0)
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


def _resize_image_batch(images, target_width, target_height, resize_mode, resize_method, pad_color_hex):
    if int(target_width) <= 0 or int(target_height) <= 0:
        raise ValueError("Target width and height must be greater than 0")

    batch, source_height, source_width, channels = images.shape
    samples = images.movedim(-1, 1)

    if resize_mode == "stretch":
        return _resize_bchw(samples, target_height, target_width, resize_method).movedim(1, -1)

    scale_x = float(target_width) / max(float(source_width), 1.0)
    scale_y = float(target_height) / max(float(source_height), 1.0)
    scale = max(scale_x, scale_y) if resize_mode == "cover_crop" else min(scale_x, scale_y)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = _resize_bchw(samples, resized_height, resized_width, resize_method)

    if resize_mode == "cover_crop":
        left = max(0, (resized_width - int(target_width)) // 2)
        top = max(0, (resized_height - int(target_height)) // 2)
        cropped = resized[:, :, top:top + int(target_height), left:left + int(target_width)]
        return cropped.movedim(1, -1)

    fill_rgb = _parse_hex_color(pad_color_hex, (255, 255, 255))
    canvas = torch.empty(
        (batch, channels, int(target_height), int(target_width)),
        dtype=images.dtype,
        device=images.device,
    )
    for channel in range(channels):
        canvas[:, channel, :, :] = fill_rgb[min(channel, 2)] / 255.0

    left = max(0, (int(target_width) - resized_width) // 2)
    top = max(0, (int(target_height) - resized_height) // 2)
    canvas[:, :, top:top + resized_height, left:left + resized_width] = resized
    return canvas.movedim(1, -1)


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

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "image",
        "width",
        "height",
        "info",
        "source_aspect_ratio",
        "target_aspect_ratio",
        "selected_preset",
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

        resized = _resize_image_batch(image, target_width, target_height, resize_mode, resize_method, pad_color_hex)
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
        )


NODE_CLASS_MAPPINGS = {
    "NH_SmartResolutionPicker": NH_SmartResolutionPicker,
    "NH_SmartRatioImageResize": NH_SmartRatioImageResize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_SmartResolutionPicker": "NH Smart Resolution Picker",
    "NH_SmartRatioImageResize": "NH Smart Ratio Image Resize",
}
