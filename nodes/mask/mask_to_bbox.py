import torch
import numpy as np

from ...core.mask_ops import mask_bbox

class NH_MaskToBBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK",)
    # THAY ĐỔI Ở ĐÂY: Thêm tên cho đầu ra
    RETURN_NAMES = ("box_mask",)
    FUNCTION = "create_bbox_mask"
    CATEGORY = "NH-Nodes/Mask"

    def create_bbox_mask(self, mask):
        batch_size = mask.shape[0]
        output_masks = []

        for i in range(batch_size):
            single_mask_np = mask[i].cpu().numpy()

            bbox_mask_np = np.zeros_like(single_mask_np)

            bbox = mask_bbox(single_mask_np)
            if bbox is not None:
                xmin, ymin, xmax, ymax = bbox

                bbox_mask_np[ymin:ymax+1, xmin:xmax+1] = 1.0

            output_masks.append(torch.from_numpy(bbox_mask_np))

        final_output = torch.stack(output_masks)

        return (final_output,)

# Ánh xạ class không đổi
NODE_CLASS_MAPPINGS = {
    "NH_MaskToBBox": NH_MaskToBBox
}

# THAY ĐỔI Ở ĐÂY: Đổi tên hiển thị của node
NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_MaskToBBox": "NH Create Box Mask"
}
