from ...core.mask_ops import mask_bbox

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
        # Chỉ xử lý mask đầu tiên trong batch; hộp bao quanh các pixel > 0.5
        bbox = mask_bbox(mask[0].cpu().numpy())

        # Xử lý trường hợp mask rỗng (toàn màu đen)
        if bbox is None:
            return (0, 0, 0, 0, 0, 0, (0, 0, 0, 0))

        x1, y1, x2, y2 = bbox

        # Tính toán chiều rộng và chiều cao
        width = x2 - x1 + 1
        height = y2 - y1 + 1

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
