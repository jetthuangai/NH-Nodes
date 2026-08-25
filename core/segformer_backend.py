"""Shared SegFormer loader/inference for NH-Nodes vision nodes.

Handles model discovery (reuse dirs already on disk before downloading),
per-model caching with CPU offload between runs, and full-resolution label
map prediction.
"""

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    import folder_paths

    _MODELS_DIR = folder_paths.models_dir
except Exception:  # running outside ComfyUI (tests)
    _MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

try:
    import comfy.model_management as _mm
except Exception:
    _mm = None

_REQUIRED_FILES = ("config.json", "model.safetensors", "preprocessor_config.json")
# Logits are interpolated at most to this size before argmax; the label map is
# then nearest-resized to the input. Keeps VRAM bounded on very large inputs.
_MAX_LOGIT_SIDE = 2048

# name -> HF repo, local dirs to probe before downloading (relative to models
# dir; other packs may already hold the same weights) and a label signature
# that guards against a different checkpoint sitting in a probed dir.
MODEL_SPECS = {
    "segformer_fashion": {
        "repo_id": "1038lab/segformer_fashion",
        "probe_dirs": ("NH-Nodes/segformer_fashion", "RMBG/segformer_fashion"),
        "signature": {1: "shirt, blouse", 24: "shoe", 46: "tassel"},
    },
    "face_parsing": {
        "repo_id": "jonathandinu/face-parsing",
        "probe_dirs": ("NH-Nodes/face_parsing", "face_parsing", "RMBG/segformer_face"),
        "signature": {1: "skin", 13: "hair", 17: "neck"},
    },
}

_LOADED = {}
_warned_no_cuda = False


def _has_required_files(path):
    return all(os.path.isfile(os.path.join(path, name)) for name in _REQUIRED_FILES)


def resolve_model_dir(model_name, download=True):
    """Return a local dir holding the model, downloading into models/NH-Nodes if absent."""
    spec = MODEL_SPECS[model_name]
    for rel in spec["probe_dirs"]:
        candidate = os.path.join(_MODELS_DIR, rel)
        if _has_required_files(candidate):
            return candidate

    target = os.path.join(_MODELS_DIR, spec["probe_dirs"][0])
    if not download:
        return None
    from huggingface_hub import snapshot_download

    os.makedirs(target, exist_ok=True)
    print(f"[NH-Nodes] Downloading {spec['repo_id']} -> {target}")
    snapshot_download(
        repo_id=spec["repo_id"],
        local_dir=target,
        allow_patterns=list(_REQUIRED_FILES),
    )
    if not _has_required_files(target):
        raise RuntimeError(f"[NH-Nodes] Model files missing after download: {target}")
    return target


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


def _ensure_on_device(bundle):
    model, device = bundle["model"], bundle["device"]
    if model.device == device:
        return
    if device.type == "cuda" and _mm is not None:
        # Let ComfyUI evict its own models first instead of OOM-ing on ours.
        size = sum(p.numel() * p.element_size() for p in model.parameters())
        _mm.free_memory(size * 2, device)
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
