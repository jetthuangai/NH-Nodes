import torch
import numpy as np

from ...core.mask_ops import mask_bbox


def _fit_window(start, length, limit):
    """Slide the window [start, start + length) inside [0, limit); one longer than limit is cut at the far edge."""
    start = max(0, min(start, limit - length))
    return start, min(limit, start + length)


def _adjust_mask(mask_np, bbox, target_size, mode):
    """Adjust the box (bbox) of one mask [H, W] to the aspect ratio of target_size = (width, height).
    The new box size is computed in integers, so there is no 1px drift from floats. Outside the box is always 0."""
    height, width = mask_np.shape
    xmin, ymin, xmax, ymax = (int(v) for v in bbox)
    cur_w, cur_h = xmax - xmin + 1, ymax - ymin + 1
    target_w, target_h = target_size
    target_is_wider = target_w * cur_h > cur_w * target_h
    output = np.zeros_like(mask_np)

    if mode in ("stretch", "pad"):
        # Grow one dimension: horizontally when the target is wider, vertically otherwise
        new_w, new_h = cur_w, cur_h
        if target_is_wider:
            new_w = max(1, cur_h * target_w // target_h)
        else:
            new_h = max(1, cur_w * target_h // target_w)

        # The new box grows symmetrically around the old one. At an image edge it slides inside to keep its full size.
        x0, x1 = _fit_window(xmin - (new_w - cur_w) // 2, new_w, width)
        y0, y1 = _fit_window(ymin - (new_h - cur_h) // 2, new_h, height)

        if mode == "stretch":
            # Stretch the content (nearest) to fit the new box. A box larger than the image shows only its middle part.
            off_x = (new_w - (x1 - x0)) // 2
            off_y = (new_h - (y1 - y0)) // 2
            cols = (np.arange(x1 - x0) + off_x) * cur_w // new_w
            rows = (np.arange(y1 - y0) + off_y) * cur_h // new_h
            output[y0:y1, x0:x1] = mask_np[ymin:ymax + 1, xmin:xmax + 1][np.ix_(rows, cols)]
        else:
            # Pad: the padding belongs to the mask (1.0), the original content keeps its position
            output[y0:y1, x0:x1] = 1.0
            output[ymin:ymax + 1, xmin:xmax + 1] = mask_np[ymin:ymax + 1, xmin:xmax + 1]

    elif mode == "crop":
        # Trim one dimension around the centre; the part that is kept does not move
        new_w, new_h = cur_w, cur_h
        if target_is_wider:
            new_h = max(1, cur_w * target_h // target_w)
        else:
            new_w = max(1, cur_h * target_w // target_h)
        x0 = xmin + (cur_w - new_w) // 2
        y0 = ymin + (cur_h - new_h) // 2
        output[y0:y0 + new_h, x0:x0 + new_w] = mask_np[y0:y0 + new_h, x0:x0 + new_w]

    else:
        raise ValueError(f"NH Mask Aspect Ratio Match: unknown mode '{mode}'")

    return output


class NH_MaskAspectRatioMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_ratio_mask": ("MASK",), # Mask A (the garment)
                "mask_to_adjust": ("MASK",),    # Mask B (the square box)
                "mode": (["stretch", "pad", "crop"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "match_aspect_ratio"
    CATEGORY = "NH-Nodes/Mask"
    DESCRIPTION = (
        "Resizes the bounding box of mask_to_adjust to the aspect ratio of target_ratio_mask, around the same centre. "
        "stretch: scale the content to the new box. "
        "pad: keep the content and add mask pixels (1.0) around it until the box has the ratio. "
        "crop: keep the middle part of the content. "
        "A box that would leave the image slides back inside it. Every mask of the batch is processed; "
        "a shorter target batch repeats."
    )

    def match_aspect_ratio(self, target_ratio_mask, mask_to_adjust, mode):
        # Accepts a mask [H, W] or [B, H, W]; every mask of the batch is processed
        targets = target_ratio_mask.reshape(-1, *target_ratio_mask.shape[-2:])
        masks = mask_to_adjust.reshape(-1, *mask_to_adjust.shape[-2:])
        if targets.shape[0] == 0 or masks.shape[0] == 0:
            return (mask_to_adjust,)

        adjusted = []
        for index in range(masks.shape[0]):
            mask_np = masks[index].cpu().numpy()
            bbox_b = mask_bbox(mask_np)
            if bbox_b is None:
                adjusted.append(masks[index]) # Empty mask B: keep it as is
                continue

            # A shorter mask A batch repeats cyclically
            bbox_a = mask_bbox(targets[index % targets.shape[0]].cpu().numpy())
            # An empty mask A counts as a 1:1 ratio
            target_size = (1, 1) if bbox_a is None else (bbox_a[2] - bbox_a[0] + 1, bbox_a[3] - bbox_a[1] + 1)
            output_np = _adjust_mask(mask_np, bbox_b, target_size, mode)
            adjusted.append(torch.from_numpy(output_np).to(masks.device))

        return (torch.stack(adjusted),)


# Mapping that lets ComfyUI find the class
NODE_CLASS_MAPPINGS = {
    "NH_MaskAspectRatioMatch": NH_MaskAspectRatioMatch
}

# Name shown in the ComfyUI menu
NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_MaskAspectRatioMatch": "NH Mask Aspect Ratio Match"
}
