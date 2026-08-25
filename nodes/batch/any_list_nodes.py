"""Generic list split nodes for NH-Nodes."""

try:
    import torch
except Exception:
    torch = None

from comfy_execution.graph_utils import ExecutionBlocker


_MAX_INDEX_OUTPUTS = 10


def _first_value(value, default):
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return default if value is None else value


def _is_image_batch_tensor(value):
    return torch is not None and isinstance(value, torch.Tensor) and value.ndim == 4


def _normalize_items(value):
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        raw_items = list(value)
    else:
        raw_items = [value]

    if len(raw_items) == 1 and isinstance(raw_items[0], (list, tuple)):
        raw_items = list(raw_items[0])

    items = []
    for item in raw_items:
        if item is None:
            continue
        if _is_image_batch_tensor(item):
            for batch_index in range(int(item.shape[0])):
                items.append(item[batch_index:batch_index + 1])
        else:
            items.append(item)
    return items


class NH_AnyListSplit:
    """Split any incoming ComfyUI list into All plus up to 10 indexed outputs."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "items": ("*",),
                "index_count": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": _MAX_INDEX_OUTPUTS,
                        "step": 1,
                        "tooltip": "Number of index outputs to show/use. All is always available.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("*", "*", "*", "*", "*", "*", "*", "*", "*", "*", "*")
    RETURN_NAMES = (
        "All",
        "index 1",
        "index 2",
        "index 3",
        "index 4",
        "index 5",
        "index 6",
        "index 7",
        "index 8",
        "index 9",
        "index 10",
    )
    OUTPUT_IS_LIST = (True, False, False, False, False, False, False, False, False, False, False)
    FUNCTION = "split"
    CATEGORY = "NH-Nodes/Batch"
    SEARCH_ALIASES = ["any list split", "list unpack", "list indexes", "split list"]

    def split(self, items, index_count):
        source_items = _normalize_items(items)
        visible_count = max(0, min(int(_first_value(index_count, 4)), _MAX_INDEX_OUTPUTS))
        blocked = ExecutionBlocker(None)
        all_output = source_items if source_items else [blocked]
        indexed_outputs = []

        for index in range(_MAX_INDEX_OUTPUTS):
            if index < visible_count and index < len(source_items):
                indexed_outputs.append(source_items[index])
            else:
                indexed_outputs.append(blocked)

        return (all_output, *indexed_outputs)


NODE_CLASS_MAPPINGS = {
    "NH_AnyListSplit": NH_AnyListSplit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_AnyListSplit": "Any List Split (NH)",
}
