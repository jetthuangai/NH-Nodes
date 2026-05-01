import importlib

_MODULE_NAMES = [
    # Existing — Mask & Image
    "mask_morphology",
    "mask_properties",
    "face_paste",
    "mask_aspect_match",
    "mask_to_bbox",
    "mask_resize_image",
    "agnostic_image",
    "image_tools_nodes",
    # Existing — VTON
    "vton_preprocessor_nodes",
    # Existing — Utils
    "slider_nodes",
    "universal_pipe_nodes",
    "utils_nodes",
    # NEW — Phase 1: Logic Core
    "logic_nodes",
    # NEW — Phase 2: Math & Random
    "math_nodes",
    # NEW — Phase 3: Text Processing
    "text_nodes",
    "text_split_lines",
    # NEW — Phase 4: Prompt Building
    "prompt_nodes",
    # NEW — Phase 5: List Management
    "list_nodes",
    # NEW — Phase 6: Batch & Counter
    "batch_nodes",
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

for _module_name in _MODULE_NAMES:
    try:
        _module = importlib.import_module(f".{_module_name}", package=__name__)
        _cls = getattr(_module, "NODE_CLASS_MAPPINGS", {})
        _disp = getattr(_module, "NODE_DISPLAY_NAME_MAPPINGS", {})
        NODE_CLASS_MAPPINGS.update(_cls)
        NODE_DISPLAY_NAME_MAPPINGS.update(_disp)
    except Exception as e:
        print(f"[NH-Nodes] Failed to load module '{_module_name}': {e}")

try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/nh-nodes/load-images-folder/count")
    async def nh_load_images_folder_count(request):
        folder_path = request.query.get("folder_path", "")
        recursive = request.query.get("recursive", "").lower() in {"1", "true", "yes", "on"}
        try:
            from .image_tools_nodes import _collect_image_files, _resolve_load_dir

            resolved_folder = _resolve_load_dir(folder_path)
            image_count = len(_collect_image_files(resolved_folder, recursive=recursive))
            return web.json_response({"count": image_count, "path": resolved_folder})
        except Exception as exc:
            return web.json_response({"count": 0, "error": str(exc)})
except Exception as e:
    print(f"[NH-Nodes] Failed to register web routes: {e}")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print(f"NH-Nodes: Loaded {len(NODE_CLASS_MAPPINGS)} nodes from {len(_MODULE_NAMES)} modules")
