"""Dump every NH node's live definition (inputs with defaults/ranges/options/tooltips, outputs) to JSON.
Run with ComfyUI's Python from the ComfyUI root: python dump_node_spec.py <out.json>
"""
import importlib
import json
import os
import sys

sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "custom_nodes")]
import torch  # noqa: E402

if not torch.cuda.is_available():
    from comfy.cli_args import args  # noqa: E402

    args.cpu = True

pack = importlib.import_module("NH-Nodes")


FILE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".safetensors", ".ckpt",
                   ".pt", ".pth", ".bin", ".gguf", ".onnx", ".sft")


def is_file_combo(options, opts):
    """Combos listing files from the user's folders (inputs, models): never publish those names."""
    if opts.get("image_upload") or opts.get("video_upload"):
        return True
    return any(str(option).lower().endswith(FILE_EXTENSIONS) for option in options)


def jsonable(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


spec = []
for node_id, cls in sorted(pack.NODE_CLASS_MAPPINGS.items()):
    inputs = []
    for section, entries in (cls.INPUT_TYPES() or {}).items():
        if section == "hidden":
            continue
        for name, value in (entries or {}).items():
            spec_type = value[0] if isinstance(value, (list, tuple)) and value else value
            opts = value[1] if isinstance(value, (list, tuple)) and len(value) > 1 and isinstance(value[1], dict) else {}
            entry = {"name": name, "section": section}
            if isinstance(spec_type, (list, tuple)):
                entry["type"] = "COMBO"
                options = [str(o) for o in spec_type]
                if is_file_combo(options, opts):
                    entry["options"] = ["(files from the user's folder)"]
                    opts = {k: v for k, v in opts.items() if k != "default"}
                else:
                    entry["options"] = options if len(options) <= 40 else options[:40] + [f"... ({len(options)} total)"]
            else:
                entry["type"] = str(spec_type)
            for key in ("default", "min", "max", "step", "multiline", "tooltip", "forceInput", "lazy"):
                if key in opts:
                    entry[key] = jsonable(opts[key])
            inputs.append(entry)
    names = list(getattr(cls, "RETURN_NAMES", None) or getattr(cls, "RETURN_TYPES", ()))
    types = [str(t) for t in getattr(cls, "RETURN_TYPES", ())]
    is_list = list(getattr(cls, "OUTPUT_IS_LIST", ()) or [])
    outputs = [{"name": n, "type": t, "is_list": bool(is_list[i]) if i < len(is_list) else False}
               for i, (n, t) in enumerate(zip(names, types))]
    spec.append({
        "id": node_id,
        "name": pack.NODE_DISPLAY_NAME_MAPPINGS.get(node_id, node_id),
        "category": getattr(cls, "CATEGORY", ""),
        "module": cls.__module__.replace("NH-Nodes.", ""),
        "description": getattr(cls, "DESCRIPTION", "") or (cls.__doc__ or "").strip(),
        "output_node": bool(getattr(cls, "OUTPUT_NODE", False)),
        "deprecated": bool(getattr(cls, "DEPRECATED", False)),
        "inputs": inputs,
        "outputs": outputs,
    })

with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(spec, handle, ensure_ascii=False, indent=1)
print(f"{len(spec)} nodes -> {sys.argv[1]}")
