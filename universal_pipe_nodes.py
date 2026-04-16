# Kiểu dữ liệu tùy chỉnh cho đường ống đa năng
NH_UNIVERSAL_PIPE = ("NH_UNIVERSAL_PIPE",)

class NH_PackUniversal:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            # Cung cấp sẵn 10 đầu vào tùy chọn
            "optional": {
                "input_0": ("*",), "input_1": ("*",), "input_2": ("*",), "input_3": ("*",), "input_4": ("*",),
                "input_5": ("*",), "input_6": ("*",), "input_7": ("*",), "input_8": ("*",), "input_9": ("*",),
            }
        }

    RETURN_TYPES = NH_UNIVERSAL_PIPE
    RETURN_NAMES = ("package",)
    FUNCTION = "pack"
    CATEGORY = "NH-Nodes/Utils/Pipe"

    def pack(self, **kwargs):
        # Đóng gói tất cả các giá trị đầu vào thành một list
        # kwargs sẽ là một dictionary chứa tất cả các input đã được kết nối
        pipe = list(kwargs.values())
        return (pipe,)

class NH_UnpackUniversal:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 9}),
            },
            "optional": {
                "package": NH_UNIVERSAL_PIPE,
                "pipe": NH_UNIVERSAL_PIPE,
            },
        }
    
    # Đầu ra là kiểu dữ liệu bất kỳ
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("item",)
    FUNCTION = "unpack"
    CATEGORY = "NH-Nodes/Utils/Pipe"

    def unpack(self, index, package=None, pipe=None):
        pipe_data = package if package is not None else pipe
        if pipe_data is None:
            return (None,)

        # Kiểm tra xem index có hợp lệ không
        if index < len(pipe_data):
            return (pipe_data[index],)
        else:
            # Trả về None nếu index nằm ngoài phạm vi
            return (None,)

# --- Đăng ký các node ---
NODE_CLASS_MAPPINGS = {
    "NH_PackUniversal": NH_PackUniversal,
    "NH_UnpackUniversal": NH_UnpackUniversal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_PackUniversal": "Pack Universal (NH)",
    "NH_UnpackUniversal": "Unpack Universal (NH)",
}
