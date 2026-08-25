"""Registry smoke test: auto-discovery must register every node exactly once."""

import importlib
import os
import sys

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PACK_DIR))
for path in (COMFY_ROOT, os.path.dirname(PACK_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

pack = importlib.import_module("NH-Nodes")

EXPECTED_NODE_COUNT = 68


def test_registry_is_complete_and_consistent():
    assert len(pack.NODE_CLASS_MAPPINGS) == EXPECTED_NODE_COUNT
    assert set(pack.NODE_DISPLAY_NAME_MAPPINGS) == set(pack.NODE_CLASS_MAPPINGS)
    assert pack.WEB_DIRECTORY == "./web"


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
