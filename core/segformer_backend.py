"""Shared SegFormer loader/inference for NH-Nodes vision nodes.

Handles model discovery (reuse dirs already on disk before downloading),
per-model caching with CPU offload between runs, and full-resolution label
map prediction.

It also holds the model-folder, download and VRAM helpers that the VTON node
reuses, so both model families follow the same rules:

* Model folders are registered with ComfyUI's ``folder_paths`` under the names
  ``nh_nodes`` (SegFormer) and ``vton_mask`` (VTON). Users can add more
  locations with those names in ``extra_model_paths.yaml``; every registered
  location is searched before anything is downloaded. Each name is registered
  with the extension of its model files (``.safetensors``, ``.onnx``), so
  ComfyUI's model lists show those files and not the Hugging Face cache
  metadata that downloads leave in the same folder.
* Downloads always go to the default folder under the models dir, pinned to a
  fixed Hugging Face revision.
"""

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    import folder_paths
except Exception:  # running outside ComfyUI (tests)
    folder_paths = None

try:
    import comfy.model_management as _mm
except Exception:
    _mm = None

_REQUIRED_FILES = ("config.json", "model.safetensors", "preprocessor_config.json")
# Logits are interpolated at most to this size before argmax; the label map is
# then nearest-resized to the input. Keeps VRAM bounded on very large inputs.
_MAX_LOGIT_SIDE = 2048

# extra_model_paths.yaml key for the SegFormer models. Each registered location holds
# one sub-folder per model (see MODEL_SPECS), e.g. <location>/segformer_fashion/config.json.
FOLDER_NAME = "nh_nodes"
FOLDER_EXTENSIONS = (".safetensors",)  # what ComfyUI lists for FOLDER_NAME: the weights, not config.json or cache metadata
_DEFAULT_FOLDER = "NH-Nodes"  # under the models dir: the download target, searched first by default

# name -> HF repo + pinned revision, legacy local dirs to probe after the registered
# folders (relative to the models dir; older versions of this pack and other packs may
# already hold the same weights) and a label signature that guards against a different
# checkpoint sitting in a probed dir.
MODEL_SPECS = {
    "segformer_fashion": {
        "repo_id": "1038lab/segformer_fashion",
        "revision": "e5bae2d8e3333be9f12eb6aae454c0542c1371d3",
        "legacy_dirs": ("RMBG/segformer_fashion",),
        "signature": {1: "shirt, blouse", 24: "shoe", 46: "tassel"},
    },
    "face_parsing": {
        "repo_id": "jonathandinu/face-parsing",
        "revision": "758b82e15a0178c9db39c1ff666a8b56e3a550c8",
        "legacy_dirs": ("face_parsing", "RMBG/segformer_face"),
        "signature": {1: "skin", 13: "hair", 17: "neck"},
    },
}

_LOADED = {}
_warned_no_cuda = False


# ---------------------------------------------------------------------------
# Model folders and downloads (shared with the VTON node)
# ---------------------------------------------------------------------------
def models_dir():
    """ComfyUI's models directory, read on every call (--models-directory, tests)."""
    if folder_paths is not None:
        return folder_paths.models_dir
    return os.path.join(os.path.dirname(__file__), "models")


def default_folder():
    """Where SegFormer models are downloaded and searched first: <models dir>/NH-Nodes."""
    return os.path.join(models_dir(), _DEFAULT_FOLDER)


def _unique_dirs(paths):
    """Drop repeats (same folder spelled differently) but keep the first spelling and the order."""
    seen, unique = set(), []
    for path in paths:
        key = os.path.normcase(os.path.abspath(path))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _list_only(folder_name, extensions):
    """Add `extensions` to the file types ComfyUI lists for the registered `folder_name`.

    add_model_folder_path registers a name with an empty extension set, which ComfyUI reads as "list
    every file": /models and the Model Library then show the .cache/huggingface metadata of our
    downloads and a repo's .gitattributes. The set is replaced, never changed in place (ComfyUI shares
    one set between many folder names), and the locations list stays the very same object, so every
    extra_model_paths.yaml entry survives.
    """
    if not extensions:
        return
    registry = folder_paths.folder_names_and_paths
    paths, current = registry[folder_name]
    merged = {*current, *extensions}
    if merged != set(current):
        registry[folder_name] = (paths, merged)


def register_model_folder(folder_name, default_dir, extensions=()):
    """Register `folder_name` with ComfyUI and return every folder searched for it, best first.

    ComfyUI applies extra_model_paths.yaml before custom nodes load, so locations the user
    configured under `folder_name` are already registered; they come first. `default_dir`
    is added too (a no-op when already present) and is always part of the result, so models
    that live there are found whether or not the user configured anything. Safe to call repeatedly.

    `extensions` (".onnx") are the file types of the models in these folders: ComfyUI's lists for
    `folder_name` then show only those, also for the locations the user added.
    """
    registered = []
    if folder_paths is not None:
        try:
            folder_paths.add_model_folder_path(folder_name, default_dir)
            registered = folder_paths.get_folder_paths(folder_name)
            _list_only(folder_name, extensions)  # last: if it fails, the locations above are still returned
        except Exception as exc:  # an unusual ComfyUI build: fall back to the default location alone
            print(f"[NH-Nodes] Could not register model folder '{folder_name}' with ComfyUI: {exc}")
    return _unique_dirs([*registered, default_dir])


def download_pinned(repo_id, revision, target, allow_patterns):
    """snapshot_download of `allow_patterns` from `repo_id` at `revision` into `target`.

    Finished files are kept and partial ones resume, so calling it again after an interrupted
    download completes the folder. Failures are re-raised with where to get the files by hand.
    """
    from huggingface_hub import snapshot_download

    os.makedirs(target, exist_ok=True)
    print(f"[NH-Nodes] Downloading {repo_id}@{revision[:8]} -> {target}")
    try:
        snapshot_download(repo_id=repo_id, revision=revision, local_dir=target, allow_patterns=list(allow_patterns))
    except Exception as exc:
        raise RuntimeError(
            f"[NH-Nodes] Downloading {repo_id} (revision {revision[:8]}) into {target} failed: {exc}\n"
            f"Run again once the connection works (finished files are kept), or download "
            f"{', '.join(allow_patterns)} from https://huggingface.co/{repo_id}/tree/{revision} into that folder."
        ) from exc


def make_room_on_gpu(nbytes):
    """Ask ComfyUI to unload its own models until about `nbytes` of VRAM are free.

    ComfyUI's own loaders ask for their size plus the VRAM it keeps in reserve (extra_reserved_memory(),
    set by --reserve-vram or its built-in default), so that reserve is added here as well: asking for
    `nbytes` alone would leave us that much short. Also hands PyTorch's cached blocks back to the
    driver: libraries that allocate outside PyTorch (ONNX Runtime) cannot use them. Nothing happens
    without a CUDA device, when ComfyUI itself runs on the CPU, or outside ComfyUI. A failure is
    reported, not raised: the caller still tries its allocation.
    """
    if _mm is None or not torch.cuda.is_available():
        return
    try:
        # ComfyUI's loaded models carry an indexed device (cuda:0); an index-less
        # torch.device("cuda") compares unequal to it and would match none of them.
        device = _mm.get_torch_device()
        if device.type != "cuda":
            return
        reserved = getattr(_mm, "extra_reserved_memory", None)  # absent in older ComfyUI builds
        _mm.free_memory(nbytes + (reserved() if reserved else 0), device)
        _mm.soft_empty_cache()
    except Exception as exc:
        print(f"[NH-Nodes] Could not ask ComfyUI to free VRAM ({exc}); continuing")


def _has_required_files(path):
    return all(os.path.isfile(os.path.join(path, name)) for name in _REQUIRED_FILES)


def model_search_dirs(model_name):
    """Folders that may hold `model_name`, best match first.

    The registered ``nh_nodes`` locations come first (models/NH-Nodes by default, plus whatever
    extra_model_paths.yaml adds), then the legacy dirs under the models dir.
    """
    spec = MODEL_SPECS[model_name]
    dirs = [os.path.join(base, model_name)
            for base in register_model_folder(FOLDER_NAME, default_folder(), FOLDER_EXTENSIONS)]
    dirs += [os.path.normpath(os.path.join(models_dir(), *rel.split("/"))) for rel in spec["legacy_dirs"]]
    return _unique_dirs(dirs)


def resolve_model_dir(model_name, download=True):
    """Return a local dir holding the model, downloading into models/NH-Nodes if absent."""
    spec = MODEL_SPECS[model_name]
    for candidate in model_search_dirs(model_name):
        if _has_required_files(candidate):
            return candidate

    if not download:
        return None
    target = os.path.join(default_folder(), model_name)
    download_pinned(spec["repo_id"], spec["revision"], target, _REQUIRED_FILES)
    if not _has_required_files(target):
        raise RuntimeError(f"[NH-Nodes] Model files missing after download: {target}")
    return target


# Make the folder name known to ComfyUI as soon as the pack loads.
register_model_folder(FOLDER_NAME, default_folder(), FOLDER_EXTENSIONS)


# ---------------------------------------------------------------------------
# SegFormer inference
# ---------------------------------------------------------------------------
def _torch_device(device):
    global _warned_no_cuda
    if device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if not _warned_no_cuda:
            print("[NH-Nodes] CUDA not available, vision nodes run on CPU")
            _warned_no_cuda = True
    return torch.device("cpu")


def _check_signature(model_name, id2label):
    expected = MODEL_SPECS[model_name]["signature"]
    for class_id, label in expected.items():
        if id2label.get(class_id) != label:
            raise RuntimeError(
                f"[NH-Nodes] Model in use for '{model_name}' has unexpected labels "
                f"(id {class_id} = {id2label.get(class_id)!r}, expected {label!r}). "
                f"Remove the wrong copy so the correct weights are downloaded."
            )


def load_segformer(model_name, device="cuda"):
    """Load (or fetch from cache) processor + model for `model_name`; weights stay on CPU until used."""
    target_device = _torch_device(device)
    bundle = _LOADED.get(model_name)
    if bundle is None:
        from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

        model_dir = resolve_model_dir(model_name)
        processor = SegformerImageProcessor.from_pretrained(model_dir)
        model = AutoModelForSemanticSegmentation.from_pretrained(model_dir)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        _check_signature(model_name, id2label)
        bundle = {"processor": processor, "model": model, "id2label": id2label}
        _LOADED[model_name] = bundle
    bundle["device"] = target_device
    return bundle


def _is_on(model_device, wanted):
    """True when `model_device` satisfies `wanted`; an index-less "cuda" accepts any cuda:N."""
    return model_device.type == wanted.type and (wanted.index is None or model_device.index == wanted.index)


def _ensure_on_device(bundle):
    model, device = bundle["model"], bundle["device"]
    if _is_on(model.device, device):
        return
    if device.type == "cuda":
        # Let ComfyUI evict its own models first instead of OOM-ing on ours.
        size = sum(p.numel() * p.element_size() for p in model.parameters())
        make_room_on_gpu(size * 2)
    model.to(device)


def offload(bundle):
    """Move weights back to CPU so they never pin VRAM between workflow runs."""
    if bundle["model"].device.type != "cpu":
        bundle["model"].cpu()
        torch.cuda.empty_cache()


def predict_labels(bundle, image_uint8):
    """Run SegFormer on one HxWx3 uint8 RGB array -> HxW int64 label map at input size."""
    _ensure_on_device(bundle)
    processor, model = bundle["processor"], bundle["model"]
    height, width = image_uint8.shape[:2]
    scale = min(1.0, _MAX_LOGIT_SIDE / max(height, width))
    logit_size = (max(1, round(height * scale)), max(1, round(width * scale)))

    inputs = processor(images=image_uint8, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        logits = F.interpolate(logits, size=logit_size, mode="bilinear", align_corners=False)
        labels = logits.argmax(dim=1)[0].to(torch.int32).cpu().numpy()
    if labels.shape != (height, width):
        labels = cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)
    return labels.astype(np.int64)
