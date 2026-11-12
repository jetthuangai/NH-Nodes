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

import importlib  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _pytest_folders_are_declared(tmp_path_factory, monkeypatch):
    """Every test runs as if pytest's temp folder were listed under nh_image_folders.

    The image nodes only read and write inside ComfyUI's folders and the declared ones. A test of that limit
    replaces ``_declared`` itself.
    """
    allowed_folders = importlib.import_module("NH-Nodes.core.allowed_folders")
    monkeypatch.setattr(allowed_folders, "_declared", [str(tmp_path_factory.getbasetemp())])
