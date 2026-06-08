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
        loras = {}
        for index in range(1, _MAX_INDEXED_LOADERS + 1):
            loras[f"lora_{index}"] = (lora_choices,)
            loras[f"enable_{index}"] = ("BOOLEAN", {"default": False})
        return {
            "required": {
                "select_mode": (["index", "boolean"], {"default": "index"}),
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

    RETURN_TYPES = ("MODEL", "STRING", "INT")
    RETURN_NAMES = ("model", "filename", "index")
    FUNCTION = "load_lora_model_index"
    CATEGORY = "NH-Nodes/Loaders"

    def load_lora_model_index(self, select_mode, model, index, lora_count, strength_model, **kwargs):
        if select_mode == "boolean":
            selected_index = -1
            count = max(1, min(int(lora_count), _MAX_INDEXED_LOADERS))
            for i in range(1, count + 1):
                if kwargs.get(f"enable_{i}", False):
                    selected_index = i
                    break
            if selected_index == -1:
                return (_block(), "", -1)
            lora_name = _selected_name(kwargs, "lora", lora_count, selected_index)
            final_index = selected_index
        else:
            lora_name = _selected_name(kwargs, "lora", lora_count, index)
            final_index = index

        if lora_name is None:
            return (_block(), "", -1)

        if strength_model == 0:
            return (model, "", final_index)

        import comfy.sd

        lora = self._load_lora(lora_name)
        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength_model, 0)
        return (model_lora, lora_name, final_index)


class NH_LoraClipIndex(_IndexedLoraBase):
    """Apply one selected LoRA to CLIP by 1-based index."""

    @classmethod
    def INPUT_TYPES(cls):
        lora_choices = _choices("loras")
        loras = {}
        for index in range(1, _MAX_INDEXED_LOADERS + 1):
            loras[f"lora_{index}"] = (lora_choices,)
            loras[f"enable_{index}"] = ("BOOLEAN", {"default": False})
        return {
            "required": {
                "select_mode": (["index", "boolean"], {"default": "index"}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "index": ("INT", {"default": 1, "min": 1, "max": _MAX_INDEXED_LOADERS}),
                "lora_count": ("INT", {"default": 5, "min": 1, "max": _MAX_INDEXED_LOADERS}),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
                "strength_clip": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
            },
            "optional": loras,
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "INT")
    RETURN_NAMES = ("model", "clip", "filename", "index")
    FUNCTION = "load_lora_clip_index"
    CATEGORY = "NH-Nodes/Loaders"

    def load_lora_clip_index(self, select_mode, model, clip, index, lora_count, strength_model, strength_clip, **kwargs):
        if select_mode == "boolean":
            selected_index = -1
            count = max(1, min(int(lora_count), _MAX_INDEXED_LOADERS))
            for i in range(1, count + 1):
                if kwargs.get(f"enable_{i}", False):
                    selected_index = i
                    break
            if selected_index == -1:
                return (_block(), _block(), "", -1)
            lora_name = _selected_name(kwargs, "lora", lora_count, selected_index)
            final_index = selected_index
        else:
            lora_name = _selected_name(kwargs, "lora", lora_count, index)
            final_index = index

        if lora_name is None:
            return (_block(), _block(), "", -1)

        if strength_model == 0 and strength_clip == 0:
            return (model, clip, "", final_index)

        import comfy.sd

        lora = self._load_lora(lora_name)
        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora, lora_name, final_index)


class NH_DiffusionModelIndex:
    """Load one diffusion model by 1-based index."""

    @classmethod
    def INPUT_TYPES(cls):
        model_choices = _choices("diffusion_models")
        models = {}
        for index in range(1, _MAX_INDEXED_LOADERS + 1):
            models[f"diffusion_model_{index}"] = (model_choices,)
            models[f"enable_{index}"] = ("BOOLEAN", {"default": False})

        return {
            "required": {
                "select_mode": (["index", "boolean"], {"default": "index"}),
                "index": ("INT", {"default": 1, "min": 1, "max": _MAX_INDEXED_LOADERS}),
                "model_count": ("INT", {"default": 5, "min": 1, "max": _MAX_INDEXED_LOADERS}),
                "weight_dtype": (
                    ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                    {"advanced": True},
                ),
            },
            "optional": models,
        }

    RETURN_TYPES = ("MODEL", "STRING", "INT")
    RETURN_NAMES = ("model", "filename", "index")
    FUNCTION = "load_diffusion_model_index"
    CATEGORY = "NH-Nodes/Loaders"

    def load_diffusion_model_index(self, select_mode, index, model_count, weight_dtype, **kwargs):
        if select_mode == "boolean":
            selected_index = -1
            count = max(1, min(int(model_count), _MAX_INDEXED_LOADERS))
            for i in range(1, count + 1):
                if kwargs.get(f"enable_{i}", False):
                    selected_index = i
                    break
            if selected_index == -1:
                return (_block(), "", -1)
            model_name = _selected_name(kwargs, "diffusion_model", model_count, selected_index)
            final_index = selected_index
        else:
            model_name = _selected_name(kwargs, "diffusion_model", model_count, index)
            final_index = index

        if model_name is None:
            return (_block(), "", -1)

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
        return (model, model_name, final_index)


class NH_VAEIndex:
    """Load one VAE by 1-based index."""

    @classmethod
    def INPUT_TYPES(cls):
        vae_choices = _choices("vae")
        vaes = {}
        for index in range(1, _MAX_INDEXED_LOADERS + 1):
            vaes[f"vae_{index}"] = (vae_choices,)
            vaes[f"enable_{index}"] = ("BOOLEAN", {"default": False})

        return {
            "required": {
                "select_mode": (["index", "boolean"], {"default": "index"}),
                "index": ("INT", {"default": 1, "min": 1, "max": _MAX_INDEXED_LOADERS}),
                "vae_count": ("INT", {"default": 5, "min": 1, "max": _MAX_INDEXED_LOADERS}),
            },
            "optional": vaes,
        }

    RETURN_TYPES = ("VAE", "STRING", "INT")
    RETURN_NAMES = ("vae", "filename", "index")
    FUNCTION = "load_vae_index"
    CATEGORY = "NH-Nodes/Loaders"

    def load_vae_index(self, select_mode, index, vae_count, **kwargs):
        if select_mode == "boolean":
            selected_index = -1
            count = max(1, min(int(vae_count), _MAX_INDEXED_LOADERS))
            for i in range(1, count + 1):
                if kwargs.get(f"enable_{i}", False):
                    selected_index = i
                    break
            if selected_index == -1:
                return (_block(), "", -1)
            vae_name = _selected_name(kwargs, "vae", vae_count, selected_index)
            final_index = selected_index
        else:
            vae_name = _selected_name(kwargs, "vae", vae_count, index)
            final_index = index

        if vae_name is None:
            return (_block(), "", -1)

        import folder_paths
        import comfy.sd
        import comfy.utils

        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
        return (vae, vae_name, final_index)


class NH_ClipIndex:
    """Load one CLIP by 1-based index."""

    @classmethod
    def INPUT_TYPES(cls):
        clip_choices = _choices("clip")
        clips = {}
        for index in range(1, _MAX_INDEXED_LOADERS + 1):
            clips[f"clip_{index}"] = (clip_choices,)
            clips[f"enable_{index}"] = ("BOOLEAN", {"default": False})

        return {
            "required": {
                "select_mode": (["index", "boolean"], {"default": "index"}),
                "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "hunyuan_video"], ),
                "index": ("INT", {"default": 1, "min": 1, "max": _MAX_INDEXED_LOADERS}),
                "clip_count": ("INT", {"default": 5, "min": 1, "max": _MAX_INDEXED_LOADERS}),
            },
            "optional": clips,
        }

    RETURN_TYPES = ("CLIP", "STRING", "INT")
    RETURN_NAMES = ("clip", "filename", "index")
    FUNCTION = "load_clip_index"
    CATEGORY = "NH-Nodes/Loaders"

    def load_clip_index(self, select_mode, type, index, clip_count, **kwargs):
        if select_mode == "boolean":
            selected_index = -1
            count = max(1, min(int(clip_count), _MAX_INDEXED_LOADERS))
            for i in range(1, count + 1):
                if kwargs.get(f"enable_{i}", False):
                    selected_index = i
                    break
            if selected_index == -1:
                return (_block(), "", -1)
            clip_name = _selected_name(kwargs, "clip", clip_count, selected_index)
            final_index = selected_index
        else:
            clip_name = _selected_name(kwargs, "clip", clip_count, index)
            final_index = index

        if clip_name is None:
            return (_block(), "", -1)

        import folder_paths
        import comfy.sd
        clip_path = folder_paths.get_full_path_or_raise("clip", clip_name)
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=comfy.sd.CLIP_ALIASES[type])
        return (clip, clip_name, final_index)


NODE_CLASS_MAPPINGS = {
    "NH_LoraModelIndex": NH_LoraModelIndex,
    "NH_LoraClipIndex": NH_LoraClipIndex,
    "NH_DiffusionModelIndex": NH_DiffusionModelIndex,
    "NH_VAEIndex": NH_VAEIndex,
    "NH_ClipIndex": NH_ClipIndex,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_LoraModelIndex": "Load LoRA Model Index (NH)",
    "NH_LoraClipIndex": "Load LoRA Clip Index (NH)",
    "NH_DiffusionModelIndex": "Load Diffusion Model Index (NH)",
    "NH_VAEIndex": "Load VAE Index (NH)",
    "NH_ClipIndex": "Load Clip Index (NH)",
}
