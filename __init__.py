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

# Route-only module: registers /nh-nodes/large-preview/* on import.
try:
    from .core import large_preview_routes  # noqa: F401
except Exception as e:
    print(f"[NH-Nodes] Failed to import core.large_preview_routes: {e}")

try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/nh-nodes/load-images-folder/count")
    async def nh_load_images_folder_count(request):
        folder_path = request.query.get("folder_path", "")
        recursive = request.query.get("recursive", "").lower() in {"1", "true", "yes", "on"}
        try:
            from .nodes.image.image_tools_nodes import _collect_image_files, _resolve_load_dir

            resolved_folder = _resolve_load_dir(folder_path)
            image_count = len(_collect_image_files(resolved_folder, recursive=recursive))
            return web.json_response({"count": image_count, "path": resolved_folder})
        except Exception as exc:
            return web.json_response({"count": 0, "error": str(exc)})

    @PromptServer.instance.routes.get("/nh-nodes/smart-resolution-picker/presets")
    async def nh_smart_resolution_picker_presets(request):
        try:
            from .core.resolution_data import DEFAULT_MODEL_LABEL, DEFAULT_PRESET, MODEL_LABELS, PRESET_LABELS_BY_MODEL
            from .nodes.resolution.smart_resolution_picker import TARGET_RESOLUTION_LEVELS, _resolution_candidates

            ratios_by_model_level = {}
            for model_label in MODEL_LABELS:
                ratios_by_model_level[model_label] = {}
                for resolution_level in TARGET_RESOLUTION_LEVELS:
                    candidates = _resolution_candidates(model_label, resolution_level)
                    ratios = []
                    for entry in candidates:
                        if entry["aspect"] not in ratios:
                            ratios.append(entry["aspect"])
                    ratios_by_model_level[model_label][resolution_level] = ratios

            return web.json_response({
                "models": MODEL_LABELS,
                "default_model": DEFAULT_MODEL_LABEL,
                "default_preset": DEFAULT_PRESET,
                "presets_by_model": PRESET_LABELS_BY_MODEL,
                "resolution_levels": TARGET_RESOLUTION_LEVELS,
                "ratios_by_model_level": ratios_by_model_level,
            })
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)
except Exception as e:
    print(f"[NH-Nodes] Failed to register web routes: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print(f"NH-Nodes: Loaded {len(NODE_CLASS_MAPPINGS)} nodes from {_module_count} modules")
