import torch
import numpy as np

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
            
            rows, cols = np.where(single_mask_np > 0.5)
            
            bbox_mask_np = np.zeros_like(single_mask_np)

            if len(rows) > 0 and len(cols) > 0:
                ymin, ymax = np.min(rows), np.max(rows)
                xmin, xmax = np.min(cols), np.max(cols)
                
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