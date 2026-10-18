"""NH-Nodes entry point.

Layout:
  core/    shared helpers and backends (no nodes)
  nodes/   one folder per ComfyUI menu category; every module that defines
           NODE_CLASS_MAPPINGS is discovered and registered automatically
  web/     frontend extensions
"""

import importlib
import pkgutil

from . import nodes as _nodes_pkg

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

# Layout is deliberately two levels deep: nodes/<category>/<module>.py. Sub-packages
# inside a category (vendored code such as nodes/vton/preprocess) define no nodes
# and are never imported here, so a broken one cannot take the whole pack down.
_module_count = 0
for _category in pkgutil.iter_modules(_nodes_pkg.__path__):
    if not _category.ispkg:
        continue
    _category_name = f"{_nodes_pkg.__name__}.{_category.name}"
    try:
        _category_pkg = importlib.import_module(_category_name)
    except Exception as e:
        print(f"[NH-Nodes] Failed to load category '{_category.name}': {e}")
        continue
    for _entry in pkgutil.iter_modules(_category_pkg.__path__):
        if _entry.ispkg:
            continue
        _module_name = f"{_category_name}.{_entry.name}"
        try:
            _module = importlib.import_module(_module_name)
        except Exception as e:
            print(f"[NH-Nodes] Failed to load module '{_module_name}': {e}")
            continue
        _new_nodes = getattr(_module, "NODE_CLASS_MAPPINGS", {})
        for _dup in _new_nodes.keys() & NODE_CLASS_MAPPINGS.keys():
            print(f"[NH-Nodes] WARNING: node id '{_dup}' redefined by {_module_name}")
        NODE_CLASS_MAPPINGS.update(_new_nodes)
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_module, "NODE_DISPLAY_NAME_MAPPINGS", {}))
        _module_count += 1

# Route-only modules: they register the /nh-nodes/* HTTP routes on import.
try:
    from .core import large_preview_routes  # noqa: F401
except Exception as e:
    print(f"[NH-Nodes] Failed to import core.large_preview_routes: {e}")

try:
    from .core import folder_routes  # noqa: F401
except Exception as e:
    print(f"[NH-Nodes] Failed to import core.folder_routes: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print(f"NH-Nodes: Loaded {len(NODE_CLASS_MAPPINGS)} nodes from {_module_count} modules")
