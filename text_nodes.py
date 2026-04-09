"""Phase 3: Text Processing — NH_StringOps, NH_PromptJoin, NH_TextSplit, NH_RegexExtract"""

import re


class NH_StringOps:
    """Xu ly chuoi da nang."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "operation": (["upper", "lower", "strip", "title",
                              "replace", "contains", "startswith",
                              "endswith", "length", "slice"],),
            },
            "optional": {
                "param_a": ("STRING", {"default": ""}),
                "param_b": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "INT")
    RETURN_NAMES = ("text_out", "bool_out", "int_out")
    FUNCTION = "process"
    CATEGORY = "NH-Nodes/Text"

    def process(self, text, operation, param_a="", param_b=""):
        try:
            if operation == "upper":
                return (text.upper(), True, len(text))
            elif operation == "lower":
                return (text.lower(), True, len(text))
            elif operation == "strip":
                result = text.strip()
                return (result, True, len(result))
            elif operation == "title":
                return (text.title(), True, len(text))
            elif operation == "replace":
                result = text.replace(param_a, param_b)
                return (result, True, len(result))
            elif operation == "contains":
                found = param_a.lower() in text.lower()
                return (text, found, len(text))
            elif operation == "startswith":
                found = text.startswith(param_a)
                return (text, found, len(text))
            elif operation == "endswith":
                found = text.endswith(param_a)
                return (text, found, len(text))
            elif operation == "length":
                return (text, len(text) > 0, len(text))
            elif operation == "slice":
                result = self._do_slice(text, param_a)
                return (result, True, len(result))
            else:
                return (text, False, len(text))
        except Exception as e:
            print(f"[NH-Nodes] StringOps error: {e}")
            return (text, False, len(text))

    @staticmethod
    def _do_slice(text, param):
        """Parse slice string like '2:10' or ':5' or '3:'."""
        parts = param.split(":")
        try:
            if len(parts) == 1:
                idx = int(parts[0])
                return text[idx] if 0 <= idx < len(text) else ""
            elif len(parts) == 2:
                start = int(parts[0]) if parts[0].strip() else None
                end = int(parts[1]) if parts[1].strip() else None
                return text[start:end]
            else:
                return text
        except (ValueError, IndexError):
            return text


class NH_PromptJoin:
    """Noi nhieu text inputs lai voi nhau."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "separator": ("STRING", {"default": ", "}),
                "skip_empty": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "text_1": ("STRING", {"default": "", "forceInput": True}),
                "text_2": ("STRING", {"default": "", "forceInput": True}),
                "text_3": ("STRING", {"default": "", "forceInput": True}),
                "text_4": ("STRING", {"default": "", "forceInput": True}),
                "text_5": ("STRING", {"default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("result", "count")
    FUNCTION = "join"
    CATEGORY = "NH-Nodes/Text"

    def join(self, separator, skip_empty, **kwargs):
        parts = []
        for i in range(1, 6):
            key = f"text_{i}"
            val = kwargs.get(key)
            if val is None:
                continue
            val = str(val).strip()
            if skip_empty and not val:
                continue
            parts.append(val)

        return (separator.join(parts), len(parts))


class NH_TextSplit:
    """Tach text thanh cac items."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "delimiter": ("STRING", {"default": ", "}),
            },
            "optional": {
                "max_splits": ("INT", {"default": -1, "min": -1, "max": 100}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("items_joined", "count", "first", "last")
    FUNCTION = "split"
    CATEGORY = "NH-Nodes/Text"

    def split(self, text, delimiter, max_splits=-1):
        if not text.strip():
            return ("", 0, "", "")

        # Xu ly delimiter dac biet
        actual_delim = delimiter.replace("\\n", "\n").replace("\\t", "\t")

        if max_splits < 0:
            items = text.split(actual_delim)
        else:
            items = text.split(actual_delim, max_splits)

        items = [item.strip() for item in items if item.strip()]
        count = len(items)

        if count == 0:
            return ("", 0, "", "")

        items_joined = "\n".join(items)
        return (items_joined, count, items[0], items[-1])


class NH_RegexExtract:
    """Regex operations tren text."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "pattern": ("STRING", {"default": "(\\d+)"}),
                "mode": (["match", "findall", "replace", "split"],),
            },
            "optional": {
                "replacement": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING", "INT")
    RETURN_NAMES = ("result", "matched", "groups", "count")
    FUNCTION = "extract"
    CATEGORY = "NH-Nodes/Text"

    def extract(self, text, pattern, mode, replacement=""):
        try:
            if mode == "match":
                m = re.search(pattern, text)
                if m:
                    groups = "\n".join(m.groups()) if m.groups() else m.group(0)
                    return (m.group(0), True, groups, 1)
                return ("", False, "", 0)

            elif mode == "findall":
                matches = re.findall(pattern, text)
                if matches:
                    # findall co the tra ve list of tuples (khi co groups)
                    flat = []
                    for m in matches:
                        if isinstance(m, tuple):
                            flat.append(", ".join(m))
                        else:
                            flat.append(str(m))
                    result = "\n".join(flat)
                    return (result, True, result, len(matches))
                return ("", False, "", 0)

            elif mode == "replace":
                result = re.sub(pattern, replacement, text)
                count = len(re.findall(pattern, text))
                return (result, count > 0, text, count)

            elif mode == "split":
                parts = re.split(pattern, text)
                parts = [p.strip() for p in parts if p and p.strip()]
                result = "\n".join(parts)
                return (result, len(parts) > 1, result, len(parts))

            return (text, False, "", 0)

        except re.error as e:
            print(f"[NH-Nodes] RegexExtract: Invalid pattern '{pattern}': {e}")
            return (text, False, "", 0)
        except Exception as e:
            print(f"[NH-Nodes] RegexExtract error: {e}")
            return (text, False, "", 0)


# --- Dang ky ---
NODE_CLASS_MAPPINGS = {
    "NH_StringOps": NH_StringOps,
    "NH_PromptJoin": NH_PromptJoin,
    "NH_TextSplit": NH_TextSplit,
    "NH_RegexExtract": NH_RegexExtract,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_StringOps": "String Operations (NH)",
    "NH_PromptJoin": "Prompt Join (NH)",
    "NH_TextSplit": "Text Split (NH)",
    "NH_RegexExtract": "Regex Extract (NH)",
}
