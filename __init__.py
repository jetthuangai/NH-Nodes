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

for _module_name in _MODULE_NAMES:
    try:
        _module = importlib.import_module(f".{_module_name}", package=__name__)
        _cls = getattr(_module, "NODE_CLASS_MAPPINGS", {})
        _disp = getattr(_module, "NODE_DISPLAY_NAME_MAPPINGS", {})
        NODE_CLASS_MAPPINGS.update(_cls)
        NODE_DISPLAY_NAME_MAPPINGS.update(_disp)
    except Exception as e:
        print(f"[NH-Nodes] Failed to load module '{_module_name}': {e}")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print(f"NH-Nodes: Loaded {len(NODE_CLASS_MAPPINGS)} nodes from {len(_MODULE_NAMES)} modules")
