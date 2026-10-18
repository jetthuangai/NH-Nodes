"""Sapiens2 body-part segmentation (Meta, April 2026) for Body Segment (NH).

The model labels every pixel of a person with one of 28 body parts and clothes (plus background), predicted at
1024 x 768. The model code is Meta's standalone file, vendored in core/sapiens2/ with its licence (the Sapiens2
License: royalty-free and commercial use allowed, with prohibited uses that include surveillance, biometric processing,
identifying people, deepfakes and inferring sensitive data; see LICENSE.md there and THIRD_PARTY_NOTICES.md). The
weights download once from Hugging Face, pinned to a fixed revision, into the ``nh_nodes`` model folder
(models/NH-Nodes/sapiens2-seg-<size>/ by default), like the SegFormer parsers. It needs PyTorch 2.4 or newer.
"""

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_files import download_pinned, make_room_on_gpu, register_model_folder
from .segformer_backend import FOLDER_EXTENSIONS, FOLDER_NAME, default_folder

CLASSES = (
    "background", "apparel", "eyeglass", "face_neck", "hair", "left_foot", "left_hand", "left_lower_arm",
    "left_lower_leg", "left_shoe", "left_sock", "left_upper_arm", "left_upper_leg", "lower_clothing", "right_foot",
    "right_hand", "right_lower_arm", "right_lower_leg", "right_shoe", "right_sock", "right_upper_arm",
    "right_upper_leg", "torso", "upper_clothing", "lower_lip", "upper_lip", "lower_teeth", "upper_teeth", "tongue",
)
INPUT_SIZE = (1024, 768)  # height, width the model was trained on (3:4)
MODELS = {
    "0.4b": {"repo_id": "facebook/sapiens2-seg-0.4b", "revision": "449b3c5335e6722bb94990abdd1aa6e612432f22",
             "file": "sapiens2_0.4b_seg.safetensors", "arch": "sapiens2_0.4b", "dims": 1024},
    "0.8b": {"repo_id": "facebook/sapiens2-seg-0.8b", "revision": "196a627b928676c4429b738ed76f78a21d96c4eb",
             "file": "sapiens2_0.8b_seg.safetensors", "arch": "sapiens2_0.8b", "dims": 1280},
    "1b": {"repo_id": "facebook/sapiens2-seg-1b", "revision": "60cf0b3584d68dccca9adb3b59677b72a5611378",
           "file": "sapiens2_1b_seg.safetensors", "arch": "sapiens2_1b", "dims": 1536},
}
_MEAN = (123.675, 116.28, 103.53)  # ImageNet, RGB on 0..255
_STD = (58.395, 57.12, 57.375)
_MAX_LOGIT_SIDE = 2048  # logits are resized at most to this size before argmax, like the SegFormer parsers

_LOADED = {}
_said_cpu = False


class SegHead(nn.Module):
    """Meta's SegHead at inference: four 2x deconvolutions (16x, so 64 x 48 patches give 1024 x 768) and two 1x1
    convolutions, then one score per class. Layer names match the released checkpoints."""

    def __init__(self, in_channels, deconv_channels=(512, 256, 128, 64), conv_channels=(64, 64), classes=29):
        super().__init__()
        layers = []
        for channels in deconv_channels:
            layers += [nn.ConvTranspose2d(in_channels, channels, 4, stride=2, padding=1, bias=False),
                       nn.InstanceNorm2d(channels), nn.SiLU(inplace=True)]
            in_channels = channels
        self.deconv_layers = nn.Sequential(*layers)
        layers = []
        for channels in conv_channels:
            layers += [nn.Conv2d(in_channels, channels, 1), nn.InstanceNorm2d(channels), nn.SiLU(inplace=True)]
            in_channels = channels
        self.conv_layers = nn.Sequential(*layers)
        self.conv_seg = nn.Conv2d(in_channels, classes, 1)

    def forward(self, features):
        return self.conv_seg(self.conv_layers(self.deconv_layers(features)))


class Sapiens2Seg(nn.Module):
    def __init__(self, arch, dims):
        super().__init__()
        from .sapiens2.backbone import Sapiens2

        self.backbone = Sapiens2(arch=arch, img_size=INPUT_SIZE, patch_size=16, final_norm=True, use_tokenizer=False,
                                 with_cls_token=True, out_type="featmap")
        self.decode_head = SegHead(dims, classes=len(CLASSES))

    def forward(self, pixels):
        return self.decode_head(self.backbone(pixels)[0])


def model_path(size, download=True):
    """Path of the weights for ``size`` from any registered ``nh_nodes`` folder; downloaded if absent."""
    spec = MODELS[size]
    folder = f"sapiens2-seg-{size}"
    for base in register_model_folder(FOLDER_NAME, default_folder(), FOLDER_EXTENSIONS):
        candidate = os.path.join(base, folder, spec["file"])
        if os.path.isfile(candidate):
            return candidate
    if not download:
        return None
    target = os.path.join(default_folder(), folder)
    download_pinned(spec["repo_id"], spec["revision"], target, (spec["file"],))
    path = os.path.join(target, spec["file"])
    if not os.path.isfile(path):
        raise RuntimeError(f"[NH-Nodes] Sapiens2 weights missing after download: {path}")
    return path


def torch_device(name):
    """The torch device for the node's ``device`` choice; "cuda" without a CUDA GPU falls back to the CPU (said once)."""
    global _said_cpu
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "cuda" and not _said_cpu:
        _said_cpu = True
        print("[NH-Nodes] CUDA not available, Body Segment runs on the CPU (slow)")
    return torch.device("cpu")


def _native_bf16(device):
    if device.type != "cuda":
        return False
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # PyTorch before 2.3 has no including_emulation
        return torch.cuda.get_device_capability(device)[0] >= 8


def load(size, device):
    """The model for ``size`` ("0.4b", "0.8b" or "1b") with the device it runs on; read from disk once and kept in
    system memory, one size at a time. Built without random initialisation: the layers are created empty and take the
    checkpoint's tensors, which a mismatching file fails loudly."""
    from safetensors.torch import load_file

    if not hasattr(nn, "RMSNorm"):
        raise RuntimeError(f"[NH-Nodes] Body Segment needs PyTorch 2.4 or newer (this is {torch.__version__}).")
    bundle = _LOADED.get(size)
    if bundle is None:
        _LOADED.clear()  # the 1b alone keeps 3 to 6 GB of system memory
        spec = MODELS[size]
        path = model_path(size)
        print(f"[NH-Nodes] Loading Sapiens2 {size} from {path}")
        try:
            state = load_file(path)
        except Exception as exc:  # noqa: BLE001 - a damaged or cut-short download
            raise RuntimeError(f"[NH-Nodes] {path} cannot be read ({exc}); delete it so it is downloaded again.") \
                from None
        # own copies, in bfloat16 on GPUs that compute it natively (what it runs in there; half the system memory):
        # a tensor left on the file's memory map keeps the file open, and Windows then cannot replace or delete it
        dtype = torch.bfloat16 if _native_bf16(device) else None
        state = {name: tensor.to(dtype) if dtype is not None and tensor.is_floating_point() and tensor.dtype != dtype
                 else tensor.clone() for name, tensor in state.items()}
        with torch.device("meta"):
            model = Sapiens2Seg(spec["arch"], spec["dims"])
        try:
            model.load_state_dict(state, strict=True, assign=True)
        except RuntimeError as exc:
            raise RuntimeError(f"[NH-Nodes] {path} is not the Sapiens2 {size} segmentation checkpoint; delete it so "
                               f"the right one is downloaded.\n{exc}") from None
        model.eval().requires_grad_(False)
        bundle = {"model": model, "size": size}
        _LOADED[size] = bundle
    bundle["device"] = device
    return bundle


def _ensure_on_device(bundle):
    """Move the model to its device, in bfloat16 on GPUs that compute it natively and in float32 elsewhere."""
    model, device = bundle["model"], bundle["device"]
    dtype = torch.bfloat16 if _native_bf16(device) else torch.float32
    current = next(model.parameters())
    if (current.device.type == device.type and (device.index is None or current.device.index == device.index)
            and current.dtype == dtype):
        return
    if device.type == "cuda":
        make_room_on_gpu(sum(p.numel() for p in model.parameters()) * dtype.itemsize * 2)
    model.to(device=device, dtype=dtype)


def offload(bundle):
    """Weights back to system memory, so they never hold VRAM between runs."""
    model = bundle["model"]
    if next(model.parameters()).device.type != "cpu":
        model.cpu()
        torch.cuda.empty_cache()


def predict_labels(bundle, image, confidence=False):
    """Class index (into CLASSES) of every pixel of the uint8 RGB ``image``, at its size; with ``confidence`` also the
    float16 probability of each pixel's class, to merge crops by how sure each one is.

    The picture is resized to 768 x 1024 as the model was trained (callers hand it crops of that 3:4 shape, see
    parse_crops.people_labels), so a crop gets the model's full resolution.
    """
    _ensure_on_device(bundle)
    model = bundle["model"]
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    height, width = image.shape[:2]
    target_h, target_w = INPUT_SIZE
    shrink = height > target_h or width > target_w
    small = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    pixels = torch.from_numpy(np.ascontiguousarray(small)).to(device).permute(2, 0, 1).float()
    mean = torch.tensor(_MEAN, device=device)[:, None, None]
    std = torch.tensor(_STD, device=device)[:, None, None]
    pixels = ((pixels - mean) / std)[None].to(dtype)
    scale = min(1.0, _MAX_LOGIT_SIDE / max(height, width))
    size = (max(1, round(height * scale)), max(1, round(width * scale)))
    with torch.inference_mode():
        logits = model(pixels).float()
        logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
        best, labels = logits.softmax(dim=1)[0].max(dim=0)
        labels = labels.to(torch.int32).cpu().numpy()
        best = best.cpu().numpy() if confidence else None
    if labels.shape != (height, width):
        labels = cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)
        if best is not None:
            best = cv2.resize(best, (width, height), interpolation=cv2.INTER_LINEAR)
    if best is not None:
        return labels.astype(np.int64), best.astype(np.float16)
    return labels.astype(np.int64)
