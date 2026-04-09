"""NH_TextSplitLines — Split multiline text into individual STRING outputs.

Improvements over basic "Text Random Multiline":
- Up to 10 individual line outputs (not just joined text)
- 4 modes: sequential, random, shuffle, reverse
- Seed control for reproducible random/shuffle
- Skip empty lines, strip whitespace
- Wrap/cycle when lines < outputs
- Outputs NH_LIST for chaining with List nodes
- Preview: shows which line went to which output
"""

import random


class NH_TextSplitLines:
    """Split multiline text into individual STRING outputs with multiple modes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "first line\nsecond line\nthird line",
                    "multiline": True,
                }),
                "amount": ("INT", {
                    "default": 3, "min": 1, "max": 10, "step": 1,
                }),
                "mode": (["sequential", "random", "shuffle", "reverse"],),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2**32 - 1,
                }),
            },
            "optional": {
                "delimiter": ("STRING", {"default": "\\n"}),
                "skip_empty": ("BOOLEAN", {"default": True}),
                "strip_whitespace": ("BOOLEAN", {"default": True}),
                "wrap_lines": ("BOOLEAN", {"default": True}),
            }
        }

    # 10 individual outputs + list + count + preview
    RETURN_TYPES = ("STRING",) * 10 + ("NH_LIST", "INT", "STRING")
    RETURN_NAMES = (
        "line_1", "line_2", "line_3", "line_4", "line_5",
        "line_6", "line_7", "line_8", "line_9", "line_10",
        "all_lines", "count", "preview",
    )
    FUNCTION = "split_lines"
    CATEGORY = "NH-Nodes/Text"

    def split_lines(self, text, amount, mode, seed,
                    delimiter="\\n", skip_empty=True,
                    strip_whitespace=True, wrap_lines=True):

        # Parse delimiter
        actual_delim = delimiter.replace("\\n", "\n").replace("\\t", "\t")

        # Split
        lines = text.split(actual_delim)

        # Clean
        if strip_whitespace:
            lines = [line.strip() for line in lines]
        if skip_empty:
            lines = [line for line in lines if line]

        total = len(lines)

        if total == 0:
            empty = ("",) * 10 + ([], 0, "(empty input)")
            return empty

        # Apply mode
        rng = random.Random(seed if seed > 0 else None)

        if mode == "sequential":
            ordered = lines[:]
        elif mode == "reverse":
            ordered = lines[::-1]
        elif mode == "shuffle":
            ordered = lines[:]
            rng.shuffle(ordered)
        elif mode == "random":
            # Pick `amount` random items independently (can repeat)
            ordered = [rng.choice(lines) for _ in range(max(amount, total))]
        else:
            ordered = lines[:]

        # Assign to outputs
        outputs = []
        preview_parts = []
        for i in range(10):
            if i < amount:
                if wrap_lines:
                    idx = i % len(ordered)
                else:
                    idx = i if i < len(ordered) else -1

                if idx >= 0 and idx < len(ordered):
                    val = ordered[idx]
                else:
                    val = ""

                outputs.append(val)
                if val:
                    preview_parts.append(f"[{i+1}] {val}")
            else:
                outputs.append("")

        preview = "\n".join(preview_parts) if preview_parts else "(no lines)"

        return tuple(outputs) + (lines, total, preview)


class NH_TextRandomLine:
    """Pick one random line from multiline text. Simple and fast."""

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
            "optional": {
                "control_after_generate": (["fixed", "increment", "decrement", "randomize"],
                                           {"default": "fixed"}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("text", "picked_index", "total_lines")
    FUNCTION = "pick"
    CATEGORY = "NH-Nodes/Text"

    def pick(self, text, seed, control_after_generate="fixed"):
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
