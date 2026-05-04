"""NH Smart Resolution Picker node."""

import math

import torch

from .resolution_data import (
    DEFAULT_MODEL_LABEL,
    DEFAULT_PRESET,
    MODEL_ID_BY_LABEL,
    MODEL_LABELS,
    MODEL_SPECS,
    PRESET_LABELS,
    PRESET_LABELS_BY_MODEL,
    PRESET_LOOKUP_BY_MODEL,
)


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


NODE_CLASS_MAPPINGS = {
    "NH_SmartResolutionPicker": NH_SmartResolutionPicker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_SmartResolutionPicker": "NH Smart Resolution Picker",
}
