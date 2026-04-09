"""Phase 6: Batch & Counter — NH_BatchIndex, NH_BatchMerge, NH_Counter"""

import torch
import torch.nn.functional as F


class NH_BatchIndex:
    """Lay 1 hoac nhieu items tu IMAGE batch theo index."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch": ("IMAGE",),
                "index": ("INT", {"default": 0, "min": -100, "max": 1000}),
            },
            "optional": {
                "end_index": ("INT", {"default": -1, "min": -100, "max": 1000}),
                "step": ("INT", {"default": 1, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("result", "original_count")
    FUNCTION = "select"
    CATEGORY = "NH-Nodes/Batch"

    def select(self, batch, index, end_index=-1, step=1):
        count = batch.shape[0]

        # Negative index
        if index < 0:
            index = max(0, count + index)
        index = min(index, count - 1)

        if end_index == -1:
            # Single item
            result = batch[index:index + 1]
        else:
            if end_index < 0:
                end_index = count + end_index + 1
            end_index = min(end_index, count)
            result = batch[index:end_index:step]

        if result.shape[0] == 0:
            print("[NH-Nodes] Warning: BatchIndex resulted in empty batch, returning first item")
            result = batch[0:1]

        return (result, count)


class NH_BatchMerge:
    """Gop nhieu IMAGE batches lai voi nhau."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_a": ("IMAGE",),
            },
            "optional": {
                "batch_b": ("IMAGE",),
                "batch_c": ("IMAGE",),
                "resize_mode": (["stretch", "crop", "pad"],
                                {"default": "stretch"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("result", "count")
    FUNCTION = "merge"
    CATEGORY = "NH-Nodes/Batch"

    def merge(self, batch_a, batch_b=None, batch_c=None, resize_mode="stretch"):
        # Target size tu batch_a
        target_h, target_w = batch_a.shape[1], batch_a.shape[2]
        batches = [batch_a]

        for batch in [batch_b, batch_c]:
            if batch is None:
                continue
            if batch.shape[1] != target_h or batch.shape[2] != target_w:
                batch = self._resize_batch(batch, target_h, target_w, resize_mode)
            batches.append(batch)

        result = torch.cat(batches, dim=0)
        return (result, result.shape[0])

    @staticmethod
    def _resize_batch(batch, th, tw, mode):
        """Resize batch ve target size."""
        # (B, H, W, C) -> (B, C, H, W) cho interpolate
        b = batch.permute(0, 3, 1, 2)

        if mode == "stretch":
            b = F.interpolate(b, size=(th, tw), mode='bilinear', align_corners=False)

        elif mode == "crop":
            # Scale de fill, roi crop
            _, _, sh, sw = b.shape
            scale = max(th / sh, tw / sw)
            new_h, new_w = round(sh * scale), round(sw * scale)
            b = F.interpolate(b, size=(new_h, new_w), mode='bilinear', align_corners=False)
            # Center crop
            y1 = (new_h - th) // 2
            x1 = (new_w - tw) // 2
            b = b[:, :, y1:y1 + th, x1:x1 + tw]

        elif mode == "pad":
            # Scale de fit, roi pad
            _, _, sh, sw = b.shape
            scale = min(th / sh, tw / sw)
            new_h, new_w = round(sh * scale), round(sw * scale)
            b = F.interpolate(b, size=(new_h, new_w), mode='bilinear', align_corners=False)
            # Center pad
            pad_top = (th - new_h) // 2
            pad_bottom = th - new_h - pad_top
            pad_left = (tw - new_w) // 2
            pad_right = tw - new_w - pad_left
            b = F.pad(b, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

        # (B, C, H, W) -> (B, H, W, C)
        return b.permute(0, 2, 3, 1)


class NH_Counter:
    """Counter tang dan qua moi lan Queue."""

    _states = {}  # Class-level state storage

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start": ("INT", {"default": 0}),
                "step": ("INT", {"default": 1, "min": 1}),
                "max_value": ("INT", {"default": 9, "min": 0}),
                "reset": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("INT", "BOOLEAN", "FLOAT", "INT")
    RETURN_NAMES = ("current", "is_done", "progress", "remaining")
    FUNCTION = "count"
    CATEGORY = "NH-Nodes/Batch"

    def count(self, start, step, max_value, reset):
        node_id = id(self)

        if reset or node_id not in NH_Counter._states:
            NH_Counter._states[node_id] = start

        current = NH_Counter._states[node_id]
        is_done = current > max_value
        progress = min(1.0, current / max(max_value, 1))
        remaining = max(0, max_value - current + 1)

        # Auto-increment cho lan Queue tiep theo
        if not is_done:
            NH_Counter._states[node_id] += step

        return (min(current, max_value), is_done, progress, remaining)


# --- Dang ky ---
NODE_CLASS_MAPPINGS = {
    "NH_BatchIndex": NH_BatchIndex,
    "NH_BatchMerge": NH_BatchMerge,
    "NH_Counter": NH_Counter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_BatchIndex": "Batch Index (NH)",
    "NH_BatchMerge": "Batch Merge (NH)",
    "NH_Counter": "Counter (NH)",
}
