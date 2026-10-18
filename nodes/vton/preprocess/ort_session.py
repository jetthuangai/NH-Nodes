"""ONNX Runtime session factory for the vendored VTON preprocessors (NH-Nodes addition).

onnxruntime is not a hard dependency of the pack: installing the CPU wheel next to
onnxruntime-gpu overwrites the GPU build. VTON therefore checks for it on first
use and explains how to install the right wheel.
"""

import importlib.util
import os
import weakref

INSTALL_HINT = (
    "VTON Ultimate Processor (NH) needs ONNX Runtime, which is not installed in ComfyUI's Python.\n"
    "Install exactly one of these, then restart ComfyUI:\n"
    "  NVIDIA GPU:  python -m pip install onnxruntime-gpu\n"
    "  CPU only:    python -m pip install onnxruntime"
)

_warned_cpu_fallback = False
# Where the sessions really run, as ONNX Runtime reports it (session.get_providers()[0]). The device widget only
# says what was asked for: a CUDA/cuDNN mismatch, or the CPU-only wheel, silently serves a "cuda" request from the CPU.
_cpu_fallback_seen = False  # a cuda request was served from the CPU at least once in this process (never reset)
_cuda_sessions = weakref.WeakSet()  # sessions served by the GPU; weak, so a session still dies with its last reference


def ensure_onnxruntime():
    if importlib.util.find_spec("onnxruntime") is None:
        raise RuntimeError(INSTALL_HINT)
    try:
        import onnxruntime  # noqa: F401
    except (ImportError, OSError) as exc:
        # Present but unusable, typically a CUDA/cuDNN DLL mismatch of onnxruntime-gpu.
        raise RuntimeError(f"{INSTALL_HINT}\nThe installed onnxruntime failed to load: {exc}") from exc


def cuda_provider_available():
    """True when the installed onnxruntime can use the GPU (the onnxruntime-gpu wheel)."""
    import onnxruntime as ort

    return "CUDAExecutionProvider" in ort.get_available_providers()


def cpu_fallback_seen():
    """True once a session requested on cuda was served from the CPU in this process.

    cuda_provider_available() only says the GPU wheel is installed; a CUDA/cuDNN mismatch still makes
    ONNX Runtime fall back to the CPU. From then on, asking ComfyUI to free VRAM cannot help these sessions.
    """
    return _cpu_fallback_seen


def cuda_sessions_alive():
    """True while a session made by create_session is served by the GPU, i.e. holds VRAM that ComfyUI's
    model management cannot see or free. Sessions that run on the CPU never count."""
    return len(_cuda_sessions) > 0


def providers_for(device):
    # CPU stays in the list so a CUDA request degrades instead of failing outright.
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def create_session(model_path, device, sess_options=None):
    global _warned_cpu_fallback, _cpu_fallback_seen
    import onnxruntime as ort

    session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers_for(device))
    active = session.get_providers()
    if active and active[0] == "CUDAExecutionProvider":
        _cuda_sessions.add(session)
    elif device == "cuda":
        _cpu_fallback_seen = True
        if not _warned_cpu_fallback:
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
