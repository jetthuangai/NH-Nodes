"""ONNX Runtime session factory for the vendored VTON preprocessors (NH-Nodes addition).

onnxruntime is not a hard dependency of the pack: installing the CPU wheel next to
onnxruntime-gpu overwrites the GPU build. VTON therefore checks for it on first
use and explains how to install the right wheel.
"""

import importlib.util
import os

INSTALL_HINT = (
    "VTON Ultimate Processor (NH) needs ONNX Runtime, which is not installed in ComfyUI's Python.\n"
    "Install exactly one of these, then restart ComfyUI:\n"
    "  NVIDIA GPU:  python -m pip install onnxruntime-gpu\n"
    "  CPU only:    python -m pip install onnxruntime"
)

_warned_cpu_fallback = False


def ensure_onnxruntime():
    if importlib.util.find_spec("onnxruntime") is None:
        raise RuntimeError(INSTALL_HINT)
    try:
        import onnxruntime  # noqa: F401
    except (ImportError, OSError) as exc:
        # Present but unusable, typically a CUDA/cuDNN DLL mismatch of onnxruntime-gpu.
        raise RuntimeError(f"{INSTALL_HINT}\nThe installed onnxruntime failed to load: {exc}") from exc


def providers_for(device):
    # CPU stays in the list so a CUDA request degrades instead of failing outright.
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def create_session(model_path, device, sess_options=None):
    global _warned_cpu_fallback
    import onnxruntime as ort

    session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers_for(device))
    active = session.get_providers()
    if device == "cuda" and (not active or active[0] != "CUDAExecutionProvider") and not _warned_cpu_fallback:
        _warned_cpu_fallback = True
        if "CUDAExecutionProvider" in ort.get_available_providers():
            advice = "The CUDA provider is installed but could not start: check that a GPU is visible and that its CUDA/cuDNN versions match onnxruntime-gpu."
        else:
            advice = "Install onnxruntime-gpu built for your CUDA version to use the GPU."
        print(
            f"[NH-Nodes] VTON: device=cuda was requested but ONNX Runtime runs "
            f"{os.path.basename(model_path)} on {active[0] if active else 'an unknown provider'}. {advice}"
        )
    return session
