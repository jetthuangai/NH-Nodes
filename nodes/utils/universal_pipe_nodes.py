# Kiểu dữ liệu tùy chỉnh cho đường ống đa năng
NH_UNIVERSAL_PIPE = ("NH_UNIVERSAL_PIPE",)

# Số ô input của Pack (input_0 .. input_9) cũng là độ dài của package
_PIPE_SLOTS = 10

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
        # Đóng gói theo số ô: package[i] là input_i, ô chưa nối là None.
        # (list(kwargs.values()) chỉ giữ ô đã nối nên index bị lệch khi bỏ trống input_0.)
        pipe = [kwargs.get(f"input_{i}") for i in range(_PIPE_SLOTS)]
        return (pipe,)

class NH_UnpackUniversal:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": _PIPE_SLOTS - 1}),
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

        # index = số ô input của Pack; ngoài phạm vi (kể cả âm) thì trả None
        if 0 <= index < len(pipe_data):
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
