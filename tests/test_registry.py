"""Registry smoke test: auto-discovery must register every node exactly once."""

import importlib
import os
import subprocess
import sys

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PACK_DIR))
for path in (COMFY_ROOT, os.path.dirname(PACK_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

pack = importlib.import_module("NH-Nodes")

EXPECTED_NODE_COUNT = 76
# Heavy libraries only some nodes need (VTON: onnxruntime, skimage; SAM 3: sam3, whose import also switches TF32
# on process-wide); importing the pack must not load them.
LAZY_ONLY = ("onnxruntime", "skimage", "sam3")


def test_registry_is_complete_and_consistent():
    assert len(pack.NODE_CLASS_MAPPINGS) == EXPECTED_NODE_COUNT
    assert set(pack.NODE_DISPLAY_NAME_MAPPINGS) == set(pack.NODE_CLASS_MAPPINGS)
    assert pack.WEB_DIRECTORY == "./web"


def test_pack_import_does_not_load_heavy_optional_libraries():
    # Fresh interpreter: other tests in this session may import these libraries legitimately.
    code = (
        "import importlib, sys\n"
        f"sys.path[:0] = [{COMFY_ROOT!r}, {os.path.dirname(PACK_DIR)!r}]\n"
        "import torch\n"
        "if not torch.cuda.is_available():\n"
        "    from comfy.cli_args import args\n"
        "    args.cpu = True\n"
        "pack = importlib.import_module('NH-Nodes')\n"
        "print('NODE_COUNT=%d' % len(pack.NODE_CLASS_MAPPINGS))\n"
        f"print('LAZY_CHECK=' + ','.join(n for n in {LAZY_ONLY!r} if n in sys.modules))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=COMFY_ROOT, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]

    def marker(name):
        lines = [line for line in result.stdout.splitlines() if line.startswith(name + "=")]
        assert lines, result.stdout[-2000:]
        return lines[-1].split("=", 1)[1]

    # A module that fails to import is only logged by the pack loader, so check the count here too.
    assert int(marker("NODE_COUNT")) == EXPECTED_NODE_COUNT, result.stdout[-2000:]
    loaded = marker("LAZY_CHECK")
    assert not loaded, f"imported at pack load: {loaded}"


def test_every_node_id_is_defined_by_exactly_one_module():
    import pkgutil

    nodes_pkg = importlib.import_module("NH-Nodes.nodes")
    owners = {}
    for category in pkgutil.iter_modules(nodes_pkg.__path__):
        category_pkg = importlib.import_module(f"{nodes_pkg.__name__}.{category.name}")
        for entry in pkgutil.iter_modules(category_pkg.__path__):
            if entry.ispkg:
                continue
            module = importlib.import_module(f"{category_pkg.__name__}.{entry.name}")
            for node_id in getattr(module, "NODE_CLASS_MAPPINGS", {}):
                assert node_id not in owners, f"{node_id} defined in {owners[node_id]} and {module.__name__}"
                owners[node_id] = module.__name__
    assert set(owners) == set(pack.NODE_CLASS_MAPPINGS)
