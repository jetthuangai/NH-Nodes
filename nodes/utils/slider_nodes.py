#
# --- HƯỚNG DẪN TÙY CHỈNH ---
# Để thay đổi tên của một thanh trượt, hãy chỉnh sửa 2 nơi:
# 1. Trong `INPUT_TYPES`: Thay đổi giá trị của "label". Đây là tên sẽ hiển thị trên node.
# 2. Trong `RETURN_NAMES`: Thay đổi tên của đầu ra (output) tương ứng.
#
# Ví dụ, để đổi "Float 1" thành "Strength":
# - "float_1": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "label": "Strength"}),
# - RETURN_NAMES = ("Strength", "Float 2", ...)
#
# Sau khi chỉnh sửa, hãy lưu file và khởi động lại ComfyUI.
#

class NH_MultiSliderFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float_1": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "label": "Float 1"}),
                "float_2": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "label": "Float 2"}),
                "float_3": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "label": "Float 3"}),
                "float_4": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "label": "Float 4"}),
                "float_5": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "label": "Float 5"}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("Float 1", "Float 2", "Float 3", "Float 4", "Float 5")
    FUNCTION = "get_values"
    CATEGORY = "NH-Nodes/Utils"

    def get_values(self, float_1, float_2, float_3, float_4, float_5):
        return (float_1, float_2, float_3, float_4, float_5)

class NH_MultiSliderInt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int_1": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1, "label": "Int 1"}),
                "int_2": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1, "label": "Int 2"}),
                "int_3": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1, "label": "Int 3"}),
                "int_4": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1, "label": "Int 4"}),
                "int_5": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1, "label": "Int 5"}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("Int 1", "Int 2", "Int 3", "Int 4", "Int 5")
    FUNCTION = "get_values"
    CATEGORY = "NH-Nodes/Utils"

    def get_values(self, int_1, int_2, int_3, int_4, int_5):
        return (int_1, int_2, int_3, int_4, int_5)

NODE_CLASS_MAPPINGS = {
    "NH_MultiSliderFloat": NH_MultiSliderFloat,
    "NH_MultiSliderInt": NH_MultiSliderInt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_MultiSliderFloat": "Multi-Slider (FLOAT)",
    "NH_MultiSliderInt": "Multi-Slider (INT)"
}