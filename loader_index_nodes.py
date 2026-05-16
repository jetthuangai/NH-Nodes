"""Indexed loader nodes for LoRA and diffusion models."""

import torch

import folder_paths
from comfy_execution.graph_utils import ExecutionBlocker


_MAX_INDEXED_LOADERS = 64


def _choices(folder_name):
    return [""] + folder_paths.get_filename_list(folder_name)


def _block():
    return ExecutionBlocker(None)


def _selected_name(kwargs, prefix, count, index):
    try:
        index = int(index)
        count = max(1, min(int(count), _MAX_INDEXED_LOADERS))
    except (TypeError, ValueError):
        return None

    if index < 1 or index > count:
        return None

    selected = kwargs.get(f"{prefix}_{index}", "")
    if selected is None:
        return None

    selected = str(selected)
    return selected or None


class _IndexedLoraBase:
    def __init__(self):
        self.loaded_lora = None

    def _load_lora(self, lora_name):
        import comfy.utils

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                return self.loaded_lora[1]
            self.loaded_lora = None

        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        self.loaded_lora = (lora_path, lora)
        return lora


class NH_LoraModelIndex(_IndexedLoraBase):
    """Apply one selected LoRA to MODEL by 1-based index."""

    @classmethod
    def INPUT_TYPES(cls):
        lora_choices = _choices("loras")
        loras = {
            f"lora_{index}": (lora_choices,)
            for index in range(1, _MAX_INDEXED_LOADERS + 1)
        }
        return {
            "required": {
                "model": ("MODEL",),
                "index": ("INT", {"default": 1, "min": 1, "max": _MAX_INDEXED_LOADERS}),
                "lora_count": ("INT", {"default": 5, "min": 1, "max": _MAX_INDEXED_LOADERS}),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
            },
            "optional": loras,
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora_model_index"
    CATEGORY = "NH-Nodes/Loaders"

    def load_lora_model_index(self, model, index, lora_count, strength_model, **kwargs):
        lora_name = _selected_name(kwargs, "lora", lora_count, index)
        if lora_name is None:
            return (_block(),)

        if strength_model == 0:
            return (model,)

        import comfy.sd

        lora = self._load_lora(lora_name)
        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength_model, 0)
        return (model_lora,)


class NH_LoraClipIndex(_IndexedLoraBase):
    """Apply one selected LoRA to CLIP by 1-based index."""

    @classmethod
    def INPUT_TYPES(cls):
        lora_choices = _choices("loras")
        loras = {
            f"lora_{index}": (lora_choices,)
            for index in range(1, _MAX_INDEXED_LOADERS + 1)
        }
        return {
            "required": {
                "clip": ("CLIP",),
                "index": ("INT", {"default": 1, "min": 1, "max": _MAX_INDEXED_LOADERS}),
                "lora_count": ("INT", {"default": 5, "min": 1, "max": _MAX_INDEXED_LOADERS}),
                "strength_clip": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
            },
            "optional": loras,
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_lora_clip_index"
    CATEGORY = "NH-Nodes/Loaders"

    def load_lora_clip_index(self, clip, index, lora_count, strength_clip, **kwargs):
        lora_name = _selected_name(kwargs, "lora", lora_count, index)
        if lora_name is None:
            return (_block(),)

        if strength_clip == 0:
            return (clip,)

        import comfy.sd

        lora = self._load_lora(lora_name)
        _, clip_lora = comfy.sd.load_lora_for_models(None, clip, lora, 0, strength_clip)
        return (clip_lora,)


class NH_DiffusionModelIndex:
    """Load one diffusion model by 1-based index."""

    @classmethod
    def INPUT_TYPES(cls):
        model_choices = _choices("diffusion_models")
        models = {
            f"diffusion_model_{index}": (model_choices,)
            for index in range(1, _MAX_INDEXED_LOADERS + 1)
        }
        return {
            "required": {
                "index": ("INT", {"default": 1, "min": 1, "max": _MAX_INDEXED_LOADERS}),
                "model_count": ("INT", {"default": 5, "min": 1, "max": _MAX_INDEXED_LOADERS}),
                "weight_dtype": (
                    ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                    {"advanced": True},
                ),
            },
            "optional": models,
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_diffusion_model_index"
    CATEGORY = "NH-Nodes/Loaders"

    def load_diffusion_model_index(self, index, model_count, weight_dtype, **kwargs):
        model_name = _selected_name(kwargs, "diffusion_model", model_count, index)
        if model_name is None:
            return (_block(),)

        import comfy.sd

        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        model = comfy.sd.load_diffusion_model(model_path, model_options=model_options)
        return (model,)


NODE_CLASS_MAPPINGS = {
    "NH_LoraModelIndex": NH_LoraModelIndex,
    "NH_LoraClipIndex": NH_LoraClipIndex,
    "NH_DiffusionModelIndex": NH_DiffusionModelIndex,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_LoraModelIndex": "Load LoRA Model Index (NH)",
    "NH_LoraClipIndex": "Load LoRA Clip Index (NH)",
    "NH_DiffusionModelIndex": "Load Diffusion Model Index (NH)",
}
