# Custom data type for the universal pipe
NH_UNIVERSAL_PIPE = ("NH_UNIVERSAL_PIPE",)

# The number of Pack input slots (input_0 .. input_9) is also the length of the package
_PIPE_SLOTS = 10

class NH_PackUniversal:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            # Provides 10 optional inputs
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
        # Pack by slot number: package[i] is input_i, an unconnected slot is None.
        # (list(kwargs.values()) keeps only connected slots, so the indices shift when input_0 is left empty.)
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

    # The output can be any data type
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("item",)
    FUNCTION = "unpack"
    CATEGORY = "NH-Nodes/Utils/Pipe"

    def unpack(self, index, package=None, pipe=None):
        pipe_data = package if package is not None else pipe
        if pipe_data is None:
            return (None,)

        # index = the Pack input slot number; out of range (negative included) returns None
        if 0 <= index < len(pipe_data):
            return (pipe_data[index],)
        else:
            # Return None when the index is out of range
            return (None,)

# --- Node registration ---
NODE_CLASS_MAPPINGS = {
    "NH_PackUniversal": NH_PackUniversal,
    "NH_UnpackUniversal": NH_UnpackUniversal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_PackUniversal": "Pack Universal (NH)",
    "NH_UnpackUniversal": "Unpack Universal (NH)",
}
