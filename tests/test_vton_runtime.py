"""VTON ONNX Runtime handling: optional dependency, install hint, CPU fallback."""

import importlib
import os
import sys
import types

import pytest

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PACK_DIR))
for path in (COMFY_ROOT, os.path.dirname(PACK_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

ort_session = importlib.import_module("NH-Nodes.nodes.vton.preprocess.ort_session")
vton = importlib.import_module("NH-Nodes.nodes.vton.vton_preprocessor_nodes")


def _hide_onnxruntime(monkeypatch):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        ort_session.importlib.util, "find_spec",
        lambda name, *args: None if name == "onnxruntime" else real_find_spec(name, *args),
    )


def test_cuda_request_keeps_cpu_fallback():
    assert ort_session.providers_for("cuda") == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert ort_session.providers_for("cpu") == ["CPUExecutionProvider"]


def test_missing_onnxruntime_raises_install_hint(monkeypatch):
    _hide_onnxruntime(monkeypatch)
    with pytest.raises(RuntimeError, match="onnxruntime-gpu"):
        ort_session.ensure_onnxruntime()


def test_vton_reports_missing_onnxruntime_before_downloading(monkeypatch):
    _hide_onnxruntime(monkeypatch)
    monkeypatch.setattr(vton, "LOADED_VTON_MODELS", None)
    downloads = []
    monkeypatch.setitem(
        sys.modules, "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **kwargs: downloads.append(kwargs)),
    )
    with pytest.raises(RuntimeError, match="ONNX Runtime"):
        vton.NH_VTonUltimateProcessor().load_model("cpu")
    assert downloads == []


def test_broken_onnxruntime_install_is_reported(monkeypatch):
    # find_spec succeeds but the import fails, e.g. a CUDA DLL mismatch in onnxruntime-gpu.
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        ort_session.importlib.util, "find_spec",
        lambda name, *args: object() if name == "onnxruntime" else real_find_spec(name, *args),
    )
    with pytest.raises(RuntimeError, match="failed to load"):
        ort_session.ensure_onnxruntime()


def test_cpu_fallback_is_reported_once(monkeypatch, capsys):
    class FakeSession:
        def __init__(self, path, sess_options=None, providers=None):
            self.requested = providers

        def get_providers(self):
            return ["CPUExecutionProvider"]

    fake_ort = types.SimpleNamespace(InferenceSession=FakeSession, get_available_providers=lambda: ["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(ort_session, "_warned_cpu_fallback", False)

    first = ort_session.create_session("models/dwpose/yolox_l.onnx", "cuda")
    ort_session.create_session("models/dwpose/dw-ll_ucoco_384.onnx", "cuda")

    assert first.requested == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    out = capsys.readouterr().out
    assert out.count("Install onnxruntime-gpu") == 1

    # GPU build present but CUDA unusable: different advice, no reinstall suggestion.
    fake_ort.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    monkeypatch.setattr(ort_session, "_warned_cpu_fallback", False)
    ort_session.create_session("models/dwpose/yolox_l.onnx", "cuda")
    out = capsys.readouterr().out
    assert "could not start" in out and "Install onnxruntime-gpu" not in out


_MODEL_FILES = (
    "dwpose/yolox_l.onnx", "dwpose/dw-ll_ucoco_384.onnx",
    "humanparsing/parsing_atr.onnx", "humanparsing/parsing_lip.onnx",
)


def _vton_ready():
    if importlib.util.find_spec("onnxruntime") is None or importlib.util.find_spec("skimage") is None:
        return False
    return all(os.path.isfile(os.path.join(vton.MODELS_DIR, name)) for name in _MODEL_FILES)


@pytest.mark.skipif(not _vton_ready(), reason="VTON models or onnxruntime not available")
def test_vton_process_smoke_on_cpu(monkeypatch):
    import numpy as np
    import torch
    from skimage import data

    monkeypatch.setattr(vton, "LOADED_VTON_MODELS", None)
    image = torch.from_numpy(data.astronaut().astype(np.float32) / 255.0).unsqueeze(0)
    outputs = vton.NH_VTonUltimateProcessor().process(image, "Upper-body", 4, False, True, True, "cpu")

    assert len(outputs) == 8
    masks = [outputs[i] for i in (0, 1, 5, 6, 7)]
    images = [outputs[i] for i in (2, 3, 4)]
    assert all(m.shape == (1, 512, 512) for m in masks)
    assert all(i.shape == (1, 512, 512, 3) for i in images)
    assert all(float(t.min()) >= 0.0 and float(t.max()) <= 1.0 for t in outputs)
    assert 0.0 < float(outputs[0].mean()) < 1.0  # the garment mask is neither empty nor full
