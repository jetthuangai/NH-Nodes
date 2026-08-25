"""Phase 5: List Management — NH_ListCreate, NH_ListIndex, NH_ListFilter"""

import re


class NH_ListCreate:
    """Tao NH_LIST tu multiline text."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "item 1\nitem 2\nitem 3",
                    "multiline": True,
                }),
            },
            "optional": {
                "delimiter": ("STRING", {"default": "\\n"}),
            }
        }

    RETURN_TYPES = ("NH_LIST", "INT", "STRING", "STRING")
    RETURN_NAMES = ("items", "count", "first", "last")
    FUNCTION = "create"
    CATEGORY = "NH-Nodes/Batch"

    def create(self, text, delimiter="\\n"):
        actual_delim = delimiter.replace("\\n", "\n").replace("\\t", "\t")
        items = [item.strip() for item in text.split(actual_delim) if item.strip()]
        count = len(items)

        if count == 0:
            return ([], 0, "", "")

        return (items, count, items[0], items[-1])


class NH_ListIndex:
    """Lay 1 item tu NH_LIST theo index."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "items": ("NH_LIST",),
                "index": ("INT", {"default": 0, "min": -100, "max": 1000}),
            },
            "optional": {
                "wrap": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "BOOLEAN", "INT")
    RETURN_NAMES = ("item", "is_first", "is_last", "count")
    FUNCTION = "get"
    CATEGORY = "NH-Nodes/Batch"

    def get(self, items, index, wrap=True):
        count = len(items)

        if count == 0:
            print("[NH-Nodes] Warning: ListIndex received empty list")
            return ("", True, True, 0)

        if wrap:
            idx = index % count
        else:
            # Negative index: Python style
            if index < 0:
                idx = max(0, count + index)
            else:
                idx = min(index, count - 1)

        is_first = idx == 0
        is_last = idx == count - 1

        return (str(items[idx]), is_first, is_last, count)


class NH_ListFilter:
    """Loc NH_LIST theo dieu kien."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "items": ("NH_LIST",),
                "condition": ("STRING", {"default": "contains:dress"}),
                "mode": (["include", "exclude"],),
            }
        }

    RETURN_TYPES = ("NH_LIST", "NH_LIST", "INT", "INT")
    RETURN_NAMES = ("passed", "rejected", "passed_count", "rejected_count")
    FUNCTION = "filter_list"
    CATEGORY = "NH-Nodes/Batch"

    def filter_list(self, items, condition, mode):
        passed = []
        rejected = []

        for item in items:
            item_str = str(item)
            match = self._check_condition(item_str, condition)

            if mode == "include":
                if match:
                    passed.append(item)
                else:
                    rejected.append(item)
            else:  # exclude
                if match:
                    rejected.append(item)
                else:
                    passed.append(item)

        return (passed, rejected, len(passed), len(rejected))

    @staticmethod
    def _check_condition(item, condition):
        """Parse va kiem tra dieu kien."""
        try:
            # Cac condition dang key:value
            if ":" in condition:
                key, value = condition.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                if key == "contains":
                    return value.lower() in item.lower()
                elif key == "startswith":
                    return item.startswith(value)
                elif key == "endswith":
                    return item.endswith(value)
                elif key == "regex":
                    return bool(re.search(value, item))
                elif key == "equals":
                    return item.lower() == value.lower()

            # Cac condition dang len>5, len<=10
            len_match = re.match(r'^len\s*(>|<|>=|<=|==|!=)\s*(\d+)$', condition.strip())
            if len_match:
                op_str = len_match.group(1)
                threshold = int(len_match.group(2))
                item_len = len(item)

                if op_str == ">":
                    return item_len > threshold
                elif op_str == "<":
                    return item_len < threshold
                elif op_str == ">=":
                    return item_len >= threshold
                elif op_str == "<=":
                    return item_len <= threshold
                elif op_str == "==":
                    return item_len == threshold
                elif op_str == "!=":
                    return item_len != threshold

            # Fallback: contains check
            return condition.lower() in item.lower()

        except Exception as e:
            print(f"[NH-Nodes] ListFilter condition error: {e}")
            return False


# --- Dang ky ---
NODE_CLASS_MAPPINGS = {
    "NH_ListCreate": NH_ListCreate,
    "NH_ListIndex": NH_ListIndex,
    "NH_ListFilter": NH_ListFilter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_ListCreate": "List Create (NH)",
    "NH_ListIndex": "List Index (NH)",
    "NH_ListFilter": "List Filter (NH)",
}
