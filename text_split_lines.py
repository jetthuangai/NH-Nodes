"""NH Text Split Lines — Split multiline text into NH_LIST + pick by index/random.

Two nodes:
- Text Split Lines: text -> NH_LIST, pick 1 line by index
- Text Random Line: text -> pick 1 random line with seed
"""

import random


class NH_TextSplitLines:
    """Split multiline text into a list. Pick one line by index or random.

    Flow: multiline text -> split -> clean -> reorder -> output list + pick one.
    Connect `lines` output to List Index / List Filter for further processing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "first prompt\nsecond prompt\nthird prompt",
                    "multiline": True,
                }),
                "index": ("INT", {
                    "default": 0, "min": -1, "max": 9999,
                    "tooltip": "Which line to pick. -1 = random (uses seed).",
                }),
                "mode": (["sequential", "shuffle", "reverse"],),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2**32 - 1,
                    "tooltip": "Seed for shuffle mode and index=-1 random pick.",
                }),
            },
            "optional": {
                "delimiter": ("STRING", {"default": "\\n"}),
                "skip_empty": ("BOOLEAN", {"default": True}),
                "strip_whitespace": ("BOOLEAN", {"default": True}),
                "wrap_index": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Wrap index when out of range (cycle through list).",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "NH_LIST", "INT", "STRING", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("selected", "lines", "count", "preview", "is_first", "is_last")
    FUNCTION = "split_lines"
    CATEGORY = "NH-Nodes/Text"

    def split_lines(self, text, index, mode, seed,
                    delimiter="\\n", skip_empty=True,
                    strip_whitespace=True, wrap_index=True):

        # Parse delimiter
        actual_delim = delimiter.replace("\\n", "\n").replace("\\t", "\t")

        # Split + clean
        lines = text.split(actual_delim)
        if strip_whitespace:
            lines = [line.strip() for line in lines]
        if skip_empty:
            lines = [line for line in lines if line]

        count = len(lines)

        if count == 0:
            return ("", [], 0, "(empty)", True, True)

        # Reorder by mode
        rng = random.Random(seed if seed > 0 else None)

        if mode == "shuffle":
            lines = lines[:]
            rng.shuffle(lines)
        elif mode == "reverse":
            lines = lines[::-1]
        # sequential: keep as-is

        # Pick one line
        if index == -1:
            # Random pick
            pick_idx = rng.randint(0, count - 1)
        elif wrap_index:
            pick_idx = index % count
        else:
            pick_idx = max(0, min(index, count - 1))

        selected = lines[pick_idx]
        is_first = pick_idx == 0
        is_last = pick_idx == count - 1

        # Preview: show all lines with marker on selected
        preview_parts = []
        for i, line in enumerate(lines):
            marker = " >>>" if i == pick_idx else ""
            preview_parts.append(f"[{i}] {line}{marker}")
        preview = "\n".join(preview_parts)

        return (selected, lines, count, preview, is_first, is_last)


class NH_TextRandomLine:
    """Pick one random line from multiline text. Minimal node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "option A\noption B\noption C",
                    "multiline": True,
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2**32 - 1,
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("text", "picked_index", "total_lines")
    FUNCTION = "pick"
    CATEGORY = "NH-Nodes/Text"

    def pick(self, text, seed):
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        total = len(lines)

        if total == 0:
            return ("", -1, 0)

        rng = random.Random(seed if seed > 0 else None)
        idx = rng.randint(0, total - 1)

        return (lines[idx], idx, total)


# --- Dang ky ---
NODE_CLASS_MAPPINGS = {
    "NH_TextSplitLines": NH_TextSplitLines,
    "NH_TextRandomLine": NH_TextRandomLine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_TextSplitLines": "Text Split Lines (NH)",
    "NH_TextRandomLine": "Text Random Line (NH)",
}
