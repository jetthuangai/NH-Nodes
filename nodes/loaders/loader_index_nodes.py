"""Indexed loader nodes for LoRA and diffusion models."""

import os
import weakref

import torch

import folder_paths
from comfy.cli_args import args
from comfy_execution.graph_utils import ExecutionBlocker


_MAX_INDEXED_LOADERS = 64


def _choices(folder_name):
    return [""] + folder_paths.get_filename_list(folder_name)


def _block():
    return ExecutionBlocker(None)


def _clip_type_choices():
    # Follow the installed ComfyUI's CLIPType enum so new families appear without code changes.
    import comfy.sd
    return [member.name.lower() for member in comfy.sd.CLIPType]


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


def _file_stamp(path):
    """(size, mtime) of a model file, so a file replaced under the same name is not taken for the one already loaded."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (stat.st_size, stat.st_mtime_ns)


def _keeps_strong_reference():
    """True when ComfyUI runs with --cache-classic (--high-ram implies it)."""
    return bool(getattr(args, "cache_classic", False))


class _LoadedObjectCache:
    """Remembers the last object a loader node built, under a key of everything that decides what gets loaded.

    ComfyUI keeps one instance per graph node, but the node's cache key holds every one of its 64 slot widgets, so
    editing a slot nobody selected runs the node again. Handing back the object that is already in memory then saves
    reading it from disk again.

    The object is held by a weak reference. While ComfyUI's own output cache holds it, the next run gets it back; when
    that cache evicts it (the default RAM-pressure cache does, under memory pressure) nothing here keeps it alive, so
    it can be freed. A strong reference would defeat that eviction. Only with --cache-classic, which drops an output as
    soon as the prompt stops using it, the node holds the object itself: otherwise there would be nothing to reuse.
    An object that cannot be weakly referenced is not remembered at all and is loaded again each time.

    One object only: choosing another file drops the old one before the new one loads, so two big models never sit in
    memory together, and "Free model and node cache" (which throws the instance away) frees it as well.
    """

    def __init__(self):
        self._loaded = None  # None, or (key, a callable that returns the object - None once it has been freed)

    def _load_cached(self, key, load):
        remembered, self._loaded = self._loaded, None  # whatever happens below, the node lets go of the old object
        if remembered is not None and remembered[0] == key:
            obj = remembered[1]()
            if obj is not None:
                self._loaded = remembered
                return obj
        del remembered  # the old object must be gone before the next one loads
        obj = load()
        self._loaded = self._remember(key, obj)
        return obj

    @staticmethod
    def _remember(key, obj):
        if _keeps_strong_reference():
            return (key, lambda: obj)
        try:
            return (key, weakref.ref(obj))
        except TypeError:  # e.g. a __slots__ class without __weakref__: loading it again beats keeping it for good
            return None


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


class NH_DiffusionModelIndex(_LoadedObjectCache):
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
        model = self._load_cached(
            (model_path, weight_dtype, _file_stamp(model_path)),
            lambda: comfy.sd.load_diffusion_model(model_path, model_options=model_options),
        )
        return (model, model_name, final_index)


class NH_VAEIndex(_LoadedObjectCache):
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

        import comfy.sd
        import comfy.utils

        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)

        def load():
            # Like the core VAE loader: the file's metadata goes to the VAE as well (it can hold the VAE's config).
            sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
            vae = comfy.sd.VAE(sd=sd, metadata=metadata)
            # The core VAE loader does this too: a file with no usable VAE fails here, not in the sampler's decode.
            check = getattr(vae, "throw_exception_if_invalid", None)
            if check is not None:
                check()
            return vae

        vae = self._load_cached((vae_path, _file_stamp(vae_path)), load)
        return (vae, vae_name, final_index)


class NH_ClipIndex(_LoadedObjectCache):
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
                "type": (_clip_type_choices(), ),
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

        import comfy.sd
        clip_path = folder_paths.get_full_path_or_raise("clip", clip_name)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        clip = self._load_cached(
            (clip_path, clip_type, _file_stamp(clip_path)),
            lambda: comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type),
        )
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
