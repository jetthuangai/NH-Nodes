"""Render a ComfyUI-style SVG card for every NH-Nodes node from its INPUT_TYPES / RETURN_TYPES.

Run with ComfyUI's Python from the ComfyUI root:
    python custom_nodes/NH-Nodes/docs/_tools/build_node_cards.py [output_dir]
Cards are generated from the live node definitions, so they never drift from the code.
"""

import importlib
import json
import os
import re
import sys
from html import escape

COMFY_ROOT = os.getcwd()
sys.path[:0] = [COMFY_ROOT, os.path.join(COMFY_ROOT, "custom_nodes")]

import torch  # noqa: E402

if not torch.cuda.is_available():
    from comfy.cli_args import args  # noqa: E402

    args.cpu = True

PACK_DIR = os.path.join(COMFY_ROOT, "custom_nodes", "NH-Nodes")
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PACK_DIR, "docs", "assets", "nodes")

# ComfyUI default dark palette (frontend 1.55) for link types; NH custom types get amber.
TYPE_COLORS = {
    "IMAGE": "#64B5F6", "MASK": "#81C784", "LATENT": "#FF9CF9", "MODEL": "#B39DDB",
    "CLIP": "#FFD500", "VAE": "#FF6E6E", "CONDITIONING": "#FFA931", "CONTROL_NET": "#6EE7B7",
    "INT": "#9CC2E5", "FLOAT": "#9CC2E5", "STRING": "#C5E1A5", "BOOLEAN": "#F48FB1", "*": "#CFD8DC",
}
CUSTOM_TYPE_COLOR = "#E0B272"
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}

W = 360
PAD = 12
TITLE_H = 46
ROW_H = 22
WIDGET_H = 24
WIDGET_GAP = 5
FONT = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', Consolas, 'Courier New', monospace"


FILE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".safetensors", ".ckpt",
                   ".pt", ".pth", ".bin", ".gguf", ".onnx", ".sft")


def is_file_combo(options, opts):
    """Combos listing files from the user's folders (inputs, models): never publish those names."""
    if opts.get("image_upload") or opts.get("video_upload"):
        return True
    return any(str(option).lower().endswith(FILE_EXTENSIONS) for option in options)


def type_name(spec_type):
    if isinstance(spec_type, (list, tuple)):
        return "COMBO"
    return str(spec_type)


def type_color(name):
    return TYPE_COLORS.get(name, CUSTOM_TYPE_COLOR)


def short(value, limit=24):
    text = str(value).replace("\n", " ⏎ ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def fmt_number(value, opts):
    if isinstance(value, float):
        step = opts.get("step", 0.01) or 0.01
        decimals = max(0, min(4, len(str(step).split(".")[1]) if "." in str(step) else 0))
        return f"{value:.{decimals}f}"
    return str(value)


def collapse_numbered(entries):
    """Fold long numbered families (lora_1..lora_64, enable_1..) into first two + a summary row."""
    families = {}
    for name, *_ in entries:
        match = re.match(r"^(.*?)(\d+)$", name)
        if match:
            families.setdefault(match.group(1), []).append(int(match.group(2)))
    folded = {prefix: max(nums) for prefix, nums in families.items() if len(nums) > 4}
    result, emitted = [], set()
    for entry in entries:
        match = re.match(r"^(.*?)(\d+)$", entry[0])
        if match and match.group(1) in folded:
            prefix, num = match.group(1), int(match.group(2))
            if num <= 2:
                result.append(entry)
            elif prefix not in emitted:
                emitted.add(prefix)
                result.append((f"{prefix}3 … {prefix}{folded[prefix]}", "FOLD", {}, entry[3]))
            continue
        result.append(entry)
    return result


def node_entries(cls):
    spec = cls.INPUT_TYPES()
    entries = []
    for section in ("required", "optional"):
        for name, value in (spec.get(section) or {}).items():
            spec_type = value[0] if isinstance(value, (list, tuple)) and value else value
            opts = value[1] if isinstance(value, (list, tuple)) and len(value) > 1 and isinstance(value[1], dict) else {}
            entries.append((name, spec_type, opts, section == "optional"))
    return collapse_numbered(entries)


def split_entries(entries):
    sockets, widgets = [], []
    for name, spec_type, opts, optional in entries:
        tname = type_name(spec_type)
        is_widget = tname == "COMBO" or tname == "FOLD" or (tname in WIDGET_TYPES and not opts.get("forceInput"))
        (widgets if is_widget else sockets).append((name, spec_type, opts, optional))
    return sockets, widgets


def widget_value(spec_type, opts):
    tname = type_name(spec_type)
    if tname == "COMBO":
        options = list(spec_type)
        if is_file_combo(options, opts):
            return "(choose a file)", ""
        default = opts.get("default", options[0] if options else "")
        return (short(default) if default != "" else "—"), f"{len(options)} options" if len(options) > 1 else ""
    default = opts.get("default", "")
    if tname == "BOOLEAN":
        return ("true" if default else "false"), ""
    if tname in ("INT", "FLOAT"):
        return fmt_number(default, opts), ""
    if tname == "STRING":
        return short(default, 30) if default != "" else "", "multiline" if opts.get("multiline") else ""
    return short(default), ""


def svg_card(node_id, display, category, cls):
    entries = node_entries(cls)
    sockets, widgets = split_entries(entries)
    return_types = [str(t) for t in getattr(cls, "RETURN_TYPES", ())]
    return_names = list(getattr(cls, "RETURN_NAMES", None) or return_types)
    is_list = list(getattr(cls, "OUTPUT_IS_LIST", ()) or [])
    output_node = getattr(cls, "OUTPUT_NODE", False)

    io_rows = max(len(sockets), len(return_types))
    multiline_rows = sum(1 for _, t, o, _ in widgets if type_name(t) == "STRING" and o.get("multiline"))
    widgets_h = len(widgets) * (WIDGET_H + WIDGET_GAP) + multiline_rows * 22
    height = TITLE_H + 10 + io_rows * ROW_H + (8 if io_rows and widgets else 0) + widgets_h + PAD

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}" '
        f'font-family="{FONT}" role="img" aria-label="{escape(display)} node">',
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{height - 1}" rx="10" fill="#26282c" stroke="#45484f"/>',
        f'<path d="M0.5 10.5 a10 10 0 0 1 10 -10 h{W - 21} a10 10 0 0 1 10 10 v{TITLE_H - 10} h-{W - 1} z" fill="#34373d"/>',
        f'<circle cx="{PAD + 5}" cy="17" r="4.5" fill="{"#F6AD55" if output_node else "#8FD18F"}"/>',
        f'<text x="{PAD + 16}" y="22" font-size="14" font-weight="600" fill="#EEF0F3">{escape(short(display, 40))}</text>',
        f'<text x="{PAD + 16}" y="38" font-size="10.5" fill="#9AA0A8" font-family="{MONO}">'
        f'{escape(node_id)} · {escape(category.replace("NH-Nodes/", ""))}</text>',
    ]

    y = TITLE_H + 10
    for index in range(io_rows):
        cy = y + index * ROW_H + ROW_H / 2
        if index < len(sockets):
            name, spec_type, opts, optional = sockets[index]
            tname = type_name(spec_type)
            color = type_color(tname)
            parts.append(f'<circle cx="{PAD}" cy="{cy}" r="5" fill="{color}" stroke="#1b1c1f" stroke-width="1.5"/>')
            label = escape(name) + (' <tspan fill="#7d838c" font-style="italic">(optional)</tspan>' if optional else "")
            parts.append(f'<text x="{PAD + 12}" y="{cy + 4}" font-size="12" fill="#D7DADF">{label}</text>')
        if index < len(return_types):
            tname = return_types[index]
            color = type_color(tname)
            name = return_names[index] if index < len(return_names) else tname
            list_mark = " ≡" if index < len(is_list) and is_list[index] else ""
            parts.append(f'<text x="{W - PAD - 12}" y="{cy + 4}" font-size="12" fill="#D7DADF" text-anchor="end">'
                         f'{escape(name)}{list_mark}</text>')
            parts.append(f'<circle cx="{W - PAD}" cy="{cy}" r="5" fill="{color}" stroke="#1b1c1f" stroke-width="1.5"/>')
    y += io_rows * ROW_H + (8 if io_rows and widgets else 0)

    for name, spec_type, opts, optional in widgets:
        tname = type_name(spec_type)
        multiline = tname == "STRING" and opts.get("multiline")
        h = WIDGET_H + (22 if multiline else 0)
        parts.append(f'<rect x="{PAD}" y="{y}" width="{W - 2 * PAD}" height="{h}" rx="7" fill="#1d1f22" stroke="#3a3d43"/>')
        if tname == "FOLD":
            parts.append(f'<text x="{W / 2}" y="{y + 16}" font-size="11" fill="#9AA0A8" text-anchor="middle" '
                         f'font-style="italic">{escape(name)}</text>')
            y += h + WIDGET_GAP
            continue
        value, hint = widget_value(spec_type, opts)
        label = escape(name) + (' <tspan fill="#6f747c" font-style="italic">opt</tspan>' if optional else "")
        if multiline:
            parts.append(f'<text x="{PAD + 10}" y="{y + 16}" font-size="11.5" fill="#9AA0A8">{label}</text>')
            parts.append(f'<text x="{PAD + 10}" y="{y + 36}" font-size="11.5" fill="#E6E8EB" font-family="{MONO}">'
                         f'{escape(value) or "…"}</text>')
        elif tname == "BOOLEAN":
            on = value == "true"
            parts.append(f'<text x="{PAD + 10}" y="{y + 16}" font-size="11.5" fill="#9AA0A8">{label}</text>')
            tx = W - PAD - 40
            parts.append(f'<rect x="{tx}" y="{y + 6}" width="30" height="13" rx="6.5" fill="{"#4C8BF5" if on else "#4a4d53"}"/>')
            parts.append(f'<circle cx="{tx + (23 if on else 7)}" cy="{y + 12.5}" r="5" fill="#F5F6F7"/>')
        else:
            arrows = tname in ("INT", "FLOAT", "COMBO")
            if arrows:
                parts.append(f'<text x="{PAD + 8}" y="{y + 16}" font-size="10" fill="#6f747c">◀</text>')
                parts.append(f'<text x="{W - PAD - 16}" y="{y + 16}" font-size="10" fill="#6f747c">▶</text>')
            lx = PAD + (22 if arrows else 10)
            rx = W - PAD - (22 if arrows else 10)
            parts.append(f'<text x="{lx}" y="{y + 16}" font-size="11.5" fill="#9AA0A8">{label}</text>')
            parts.append(f'<text x="{rx}" y="{y + 16}" font-size="11.5" fill="#E6E8EB" text-anchor="end">{escape(value)}</text>')
        y += h + WIDGET_GAP

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main():
    pack = importlib.import_module("NH-Nodes")
    os.makedirs(OUT_DIR, exist_ok=True)
    index = []
    for node_id, cls in sorted(pack.NODE_CLASS_MAPPINGS.items()):
        display = pack.NODE_DISPLAY_NAME_MAPPINGS.get(node_id, node_id)
        category = getattr(cls, "CATEGORY", "")
        svg = svg_card(node_id, display, category, cls)
        with open(os.path.join(OUT_DIR, f"{node_id}.svg"), "w", encoding="utf-8", newline="\n") as handle:
            handle.write(svg)
        index.append({"id": node_id, "name": display, "category": category,
                      "deprecated": bool(getattr(cls, "DEPRECATED", False))})
    with open(os.path.join(OUT_DIR, "index.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(index, handle, ensure_ascii=False, indent=1)
    print(f"wrote {len(index)} cards to {OUT_DIR}")


if __name__ == "__main__":
    main()
