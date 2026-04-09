import torch
import numpy as np
import cv2

def get_bounding_box_and_ratio(mask_tensor):
    """Tính toán bounding box và tỷ lệ (width / height) từ một mask tensor."""
    m_np = mask_tensor.cpu().numpy()
    rows, cols = np.where(m_np > 0.5)
    
    if len(rows) == 0:
        return None, 1.0 # Trả về None và tỷ lệ 1:1 nếu mask rỗng

    ymin, ymax = np.min(rows), np.max(rows)
    xmin, xmax = np.min(cols), np.max(cols)
    
    bbox = (xmin, ymin, xmax, ymax)
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    
    aspect_ratio = width / height if height > 0 else 1.0
    return bbox, aspect_ratio

class NH_MaskAspectRatioMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_ratio_mask": ("MASK",), # Mask A (Cái áo)
                "mask_to_adjust": ("MASK",),    # Mask B (Hộp vuông)
                # THÊM LỰA CHỌN MỚI Ở ĐÂY
                "mode": (["stretch", "pad", "crop"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "match_aspect_ratio"
    CATEGORY = "NH-Nodes/Mask"

    def match_aspect_ratio(self, target_ratio_mask, mask_to_adjust, mode):
        mask_a = target_ratio_mask[0]
        mask_b = mask_to_adjust[0]

        _ , ratio_a = get_bounding_box_and_ratio(mask_a)
        bbox_b, ratio_b = get_bounding_box_and_ratio(mask_b)

        if bbox_b is None:
            return (mask_to_adjust,)

        xmin, ymin, xmax, ymax = bbox_b
        current_width = xmax - xmin + 1
        current_height = ymax - ymin + 1
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2

        mask_b_np = mask_b.cpu().numpy()
        output_mask_np = np.zeros_like(mask_b_np)

        # Cắt nội dung thực sự của Mask B
        cropped_content_b = mask_b_np[ymin:ymax+1, xmin:xmax+1]

        # ==========================================================
        # LOGIC CHO TỪNG CHẾ ĐỘ
        # ==========================================================

        if mode == "stretch":
            new_height, new_width = current_height, current_width
            if ratio_a > ratio_b: # Target rộng hơn
                new_width = int(current_height * ratio_a)
            else: # Target cao hơn
                new_height = int(current_width / ratio_a)
            
            # Kéo dãn nội dung đã cắt
            stretched_content = cv2.resize(cropped_content_b, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            
            # Dán vào canvas mới tại vị trí trung tâm
            paste_x = center_x - new_width // 2
            paste_y = center_y - new_height // 2
            
            # Giới hạn tọa độ để không bị tràn
            p_x_start, p_y_start = max(0, paste_x), max(0, paste_y)
            p_x_end, p_y_end = min(mask_b_np.shape[1], paste_x + new_width), min(mask_b_np.shape[0], paste_y + new_height)
            
            s_x_start, s_y_start = p_x_start - paste_x, p_y_start - paste_y
            s_x_end, s_y_end = s_x_start + (p_x_end - p_x_start), s_y_start + (p_y_end - p_y_start)
            
            output_mask_np[p_y_start:p_y_end, p_x_start:p_x_end] = stretched_content[s_y_start:s_y_end, s_x_start:s_x_end]

        elif mode == "pad":
            new_height, new_width = current_height, current_width
            if ratio_a > ratio_b: # Target rộng hơn -> Thêm đệm ngang
                new_width = int(current_height * ratio_a)
            else: # Target cao hơn -> Thêm đệm dọc
                new_height = int(current_width / ratio_a)

            # Dán nội dung gốc vào trung tâm của hộp mới lớn hơn
            paste_x = center_x - current_width // 2
            paste_y = center_y - current_height // 2
            output_mask_np[paste_y:paste_y + current_height, paste_x:paste_x + current_width] = cropped_content_b
        
        elif mode == "crop":
            new_height, new_width = current_height, current_width
            if ratio_a > ratio_b: # Target rộng hơn -> Cắt bớt chiều dọc
                new_height = int(current_width / ratio_a)
            else: # Target cao hơn -> Cắt bớt chiều ngang
                new_width = int(current_height * ratio_a)

            # Cắt bớt nội dung
            crop_y = (current_height - new_height) // 2
            crop_x = (current_width - new_width) // 2
            cropped_content = cropped_content_b[crop_y:crop_y + new_height, crop_x:crop_x + new_width]

            # Dán nội dung đã cắt vào vị trí trung tâm
            paste_x = center_x - new_width // 2
            paste_y = center_y - new_height // 2
            output_mask_np[paste_y:paste_y + new_height, paste_x:paste_x + new_width] = cropped_content

        # Chuyển lại sang định dạng tensor
        output_mask = torch.from_numpy(output_mask_np).unsqueeze(0)
        return (output_mask,)


# Ánh xạ để ComfyUI nhận diện class
NODE_CLASS_MAPPINGS = {
    "NH_MaskAspectRatioMatch": NH_MaskAspectRatioMatch
}

# Tên sẽ hiển thị trong menu của ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_MaskAspectRatioMatch": "NH Mask Aspect Ratio Match"
}