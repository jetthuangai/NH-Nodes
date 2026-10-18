"""HTTP routes behind the NH-Nodes frontend extensions.

GET /nh-nodes/load-images-folder/count          how many images a folder loader sees (auto increment / wrap)
GET /nh-nodes/smart-resolution-picker/presets   the preset lists for the resolution pickers' dropdowns

Importing this module registers both on ComfyUI's PromptServer, like core.large_preview_routes. The handlers are plain
module-level functions, so a test can serve them from any aiohttp app through ``register_routes``.
"""

import asyncio
import os

from aiohttp import web

from .image_folders import collect_image_files, resolve_load_dir

COUNT_PATH = "/nh-nodes/load-images-folder/count"
PRESETS_PATH = "/nh-nodes/smart-resolution-picker/presets"

_MAX_FOLDER_PATH_CHARS = 4096
# Spellings of the ``recursive`` query parameter. Anything else is a bad request, not a silent "false".
_FLAGS = {"1": True, "true": True, "yes": True, "on": True, "": False, "0": False, "false": False, "no": False, "off": False}


def _error(status, message):
    # Fixed short texts only: ComfyUI can be reachable from the network (--listen), so an answer never carries a
    # resolved path or the text of an exception.
    return web.json_response({"error": message}, status=status)


def _count_images(folder_path, recursive):
    """How many images a loader with these settings sees. Blocking (a recursive walk): run it in a worker thread."""
    folder = resolve_load_dir(folder_path)
    if not os.path.isdir(folder):
        raise FileNotFoundError("folder not found")
    try:
        return len(collect_image_files(folder, recursive=recursive))
    except ValueError:
        return 0  # the folder exists but holds no image


async def nh_load_images_folder_count(request):
    folder_path = request.query.get("folder_path", "")
    recursive = _FLAGS.get(request.query.get("recursive", "").strip().lower())
    if recursive is None:
        return _error(400, "recursive must be true or false")
    if len(folder_path) > _MAX_FOLDER_PATH_CHARS or "\0" in folder_path:
        return _error(400, "invalid folder_path")

    try:
        count = await asyncio.to_thread(_count_images, folder_path, recursive)
    except FileNotFoundError:
        return _error(404, "folder not found")
    except PermissionError:
        return _error(403, "folder is not readable")
    except Exception as exc:
        print(f"[NH-Nodes] load-images-folder/count failed: {type(exc).__name__}: {exc}")
        return _error(500, "could not read the folder")
    return web.json_response({"count": count})


async def nh_smart_resolution_picker_presets(request):
    try:
        # Imported here: core must not depend on the node modules at import time.
        from ..nodes.resolution.smart_resolution_picker import TARGET_RESOLUTION_LEVELS, _resolution_candidates
        from .resolution_data import DEFAULT_MODEL_LABEL, DEFAULT_PRESET, MODEL_LABELS, PRESET_LABELS_BY_MODEL

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
        print(f"[NH-Nodes] smart-resolution-picker/presets failed: {type(exc).__name__}: {exc}")
        return _error(500, "could not build the preset lists")


def register_routes(routes):
    """Attach the handlers to an aiohttp RouteTableDef (ComfyUI's ``PromptServer.instance.routes``)."""
    routes.get(COUNT_PATH)(nh_load_images_folder_count)
    routes.get(PRESETS_PATH)(nh_smart_resolution_picker_presets)


try:
    from server import PromptServer

    register_routes(PromptServer.instance.routes)
except Exception as exc:
    print(f"[NH-Nodes] Failed to register folder routes: {exc}")
