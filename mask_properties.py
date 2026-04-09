import torch
import numpy as np

# File này chứa logic cho node Mask Properties (phiên bản gốc).

class NH_MaskProperties:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT", "BBOX")
    RETURN_NAMES = ("width", "height", "x1", "y1", "x2", "y2", "bbox")
    FUNCTION = "get_properties"
    CATEGORY = "NH-Nodes/Mask"

    def get_properties(self, mask):
        # Chỉ xử lý mask đầu tiên trong batch
        m = mask[0]
        
        # Chuyển tensor sang numpy array
        m_np = m.cpu().numpy()
        
        # Lấy tọa độ của tất cả các pixel không phải màu đen (giá trị > 0.5)
        rows, cols = np.where(m_np > 0.5)
        
        # Xử lý trường hợp mask rỗng (toàn màu đen)
        if len(rows) == 0:
            return (0, 0, 0, 0, 0, 0, (0, 0, 0, 0))
            
        # Tìm các giá trị min và max để xác định hộp bao quanh
        x1 = int(np.min(cols))
        y1 = int(np.min(rows))
        x2 = int(np.max(cols))
        y2 = int(np.max(rows))
        
        # Tính toán chiều rộng và chiều cao
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        
        # Tạo bounding box tuple để trả về
        bbox = (x1, y1, x2, y2)
        
        # Trả về tất cả các giá trị theo thứ tự đã định nghĩa
        return (width, height, x1, y1, x2, y2, bbox)

# Ánh xạ để ComfyUI nhận diện class
NODE_CLASS_MAPPINGS = {
    "NH_MaskProperties": NH_MaskProperties
}

# Tên sẽ hiển thị trong menu của ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_MaskProperties": "Mask Properties (NH)"
}