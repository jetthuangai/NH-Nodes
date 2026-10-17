"""Shared test setup: make ComfyUI importable and keep it on the CPU when no GPU is visible.

comfy.model_management queries the CUDA device at import time unless ``args.cpu`` is set,
so on a GPU-less machine (GitHub Actions) every module that imports it would fail.
"""

import os
import sys

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PACK_DIR))
for path in (COMFY_ROOT, os.path.dirname(PACK_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402

if not torch.cuda.is_available():
    from comfy.cli_args import args  # noqa: E402

    args.cpu = True
