"""Shared SegFormer loader/inference for NH-Nodes vision nodes.

Handles model discovery (reuse dirs already on disk before downloading),
per-model caching with CPU offload between runs, and full-resolution label
map prediction.

The model-folder, download and VRAM helpers it shares with the VTON node live
in model_files.py, so both model families follow the same rules. SegFormer
models are registered with ComfyUI's ``folder_paths`` under the name
``nh_nodes``; users can add more locations with that name in
``extra_model_paths.yaml`` and every registered location is searched before
anything is downloaded. Downloads always go to the default folder under the
models dir, pinned to a fixed Hugging Face revision.
"""

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .model_files import download_pinned, make_room_on_gpu, models_dir, register_model_folder, unique_dirs

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
# Model folders and downloads (the shared helpers are in model_files.py)
# ---------------------------------------------------------------------------
def default_folder():
    """Where SegFormer models are downloaded and searched first: <models dir>/NH-Nodes."""
    return os.path.join(models_dir(), _DEFAULT_FOLDER)


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
    return unique_dirs(dirs)


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


def predict_labels(bundle, image_uint8, confidence=False):
    """Run SegFormer on one HxWx3 uint8 RGB array -> HxW int64 label map at input size.

    With ``confidence`` it returns (labels, HxW float16 probability of each pixel's label), which lets crops of the
    same picture be merged by how sure each one is (see parse_crops.people_labels).
    """
    _ensure_on_device(bundle)
    processor, model = bundle["processor"], bundle["model"]
    height, width = image_uint8.shape[:2]
    scale = min(1.0, _MAX_LOGIT_SIDE / max(height, width))
    logit_size = (max(1, round(height * scale)), max(1, round(width * scale)))

    # The processor squeezes the picture to its fixed size anyway; doing that here first, with area averaging, is
    # many times faster on a big photo than its own resize and keeps fine detail from aliasing.
    size = getattr(processor, "size", None)
    if isinstance(size, dict) and "height" in size and "width" in size and (
            height > 2 * size["height"] or width > 2 * size["width"]):
        image_uint8 = cv2.resize(image_uint8, (size["width"], size["height"]), interpolation=cv2.INTER_AREA)
    inputs = processor(images=image_uint8, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        logits = F.interpolate(logits, size=logit_size, mode="bilinear", align_corners=False)
        best, labels = logits.float().softmax(dim=1)[0].max(dim=0)
        labels = labels.to(torch.int32).cpu().numpy()
        best = best.cpu().numpy() if confidence else None
    if labels.shape != (height, width):
        labels = cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)
        if best is not None:
            best = cv2.resize(best, (width, height), interpolation=cv2.INTER_LINEAR)
    if best is not None:
        return labels.astype(np.int64), best.astype(np.float16)
    return labels.astype(np.int64)
