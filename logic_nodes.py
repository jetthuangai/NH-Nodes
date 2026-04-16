"""Phase 1: Logic Core — NH_Compare, NH_LogicGate, NH_IfElse, NH_SwitchN, NH_AnySwitch"""

import torch
from comfy_execution.graph_utils import ExecutionBlocker


class NH_Compare:
    """So sanh 2 gia tri, tra ve BOOL."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("*",),
                "b": ("*",),
                "op": (["==", "!=", ">", "<", ">=", "<="],),
            },
            "optional": {
                "type_cast": (["auto", "int", "float", "string"],
                              {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "*", "*")
    RETURN_NAMES = ("result", "a_passthrough", "b_passthrough")
    FUNCTION = "compare"
    CATEGORY = "NH-Nodes/Logic"

    def compare(self, a, b, op, type_cast="auto"):
        if a is None or b is None:
            return (False, a, b)

        try:
            va, vb = self._cast(a, b, type_cast)
        except Exception:
            return (False, a, b)

        try:
            if op == "==":
                result = va == vb
            elif op == "!=":
                result = va != vb
            elif op == ">":
                result = va > vb
            elif op == "<":
                result = va < vb
            elif op == ">=":
                result = va >= vb
            elif op == "<=":
                result = va <= vb
            else:
                result = False
        except Exception as e:
            print(f"[NH-Nodes] Compare error: {e}")
            result = False

        return (bool(result), a, b)

    @staticmethod
    def _cast(a, b, mode):
        if mode == "int":
            return int(a), int(b)
        elif mode == "float":
            return float(a), float(b)
        elif mode == "string":
            return str(a), str(b)
        else:  # auto
            try:
                return float(a), float(b)
            except (ValueError, TypeError):
                return str(a), str(b)


class NH_LogicGate:
    """Ket hop nhieu dieu kien BOOL."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("BOOLEAN", {"default": False}),
                "op": (["AND", "OR", "NOT", "XOR", "NAND", "NOR"],),
            },
            "optional": {
                "b": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("result",)
    FUNCTION = "gate"
    CATEGORY = "NH-Nodes/Logic"

    def gate(self, a, op, b=False):
        if b is None:
            b = False

        if op == "AND":
            result = a and b
        elif op == "OR":
            result = a or b
        elif op == "NOT":
            result = not a
        elif op == "XOR":
            result = a != b
        elif op == "NAND":
            result = not (a and b)
        elif op == "NOR":
            result = not (a or b)
        else:
            result = False

        return (bool(result),)


class NH_IfElse:
    """Routing data theo dieu kien. SELECT, khong phai BRANCH."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("BOOLEAN", {"default": True}),
                "if_true": ("*",),
                "if_false": ("*",),
            }
        }

    RETURN_TYPES = ("*", "BOOLEAN")
    RETURN_NAMES = ("result", "condition_out")
    FUNCTION = "switch"
    CATEGORY = "NH-Nodes/Logic"

    def switch(self, condition, if_true, if_false):
        if condition:
            return (if_true, condition)
        else:
            return (if_false, condition)


class NH_SwitchN:
    """Chon 1 trong toi da 10 inputs theo index."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 9}),
            },
            "optional": {
                "input_0": ("*",), "input_1": ("*",),
                "input_2": ("*",), "input_3": ("*",),
                "input_4": ("*",), "input_5": ("*",),
                "input_6": ("*",), "input_7": ("*",),
                "input_8": ("*",), "input_9": ("*",),
            }
        }

    RETURN_TYPES = ("*", "INT")
    RETURN_NAMES = ("result", "count")
    FUNCTION = "switch"
    CATEGORY = "NH-Nodes/Logic"

    def switch(self, index, **kwargs):
        connected = {}
        for i in range(10):
            key = f"input_{i}"
            if key in kwargs and kwargs[key] is not None:
                connected[i] = kwargs[key]

        count = len(connected)

        if count == 0:
            print("[NH-Nodes] Warning: SwitchN has no connected inputs")
            return (None, 0)

        if index in connected:
            return (connected[index], count)

        # Fallback: input dau tien co san
        first_key = sorted(connected.keys())[0]
        print(f"[NH-Nodes] SwitchN: index {index} not connected, falling back to input_{first_key}")
        return (connected[first_key], count)


class NH_AnySwitch:
    """Route 1 input to 1 of 5 outputs by index.

    Inactive outputs use ExecutionBlocker — downstream nodes connected
    to them are silently skipped (no error, no execution).
    Only the selected output carries data and triggers its downstream chain.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("*",),
                "index": ("INT", {"default": 0, "min": 0, "max": 4}),
            },
        }

    RETURN_TYPES = ("*", "*", "*", "*", "*", "INT")
    RETURN_NAMES = ("output 0", "output 1", "output 2", "output 3", "output 4", "index")
    FUNCTION = "route"
    CATEGORY = "NH-Nodes/Logic"

    def route(self, input, index):
        index = max(0, min(index, 4))
        blocked = ExecutionBlocker(None)

        outputs = [blocked, blocked, blocked, blocked, blocked]
        outputs[index] = input

        return (*outputs, index)


class NH_GateSwitch:
    """Gate that passes or blocks data AND upstream execution.

    - enabled = True:  upstream runs, data passes through to output.
    - enabled = False: upstream is NEVER executed (lazy skip),
      output is blocked (ExecutionBlocker). Nothing runs.

    Use this to completely disable a branch — e.g. prevent a model
    from loading, skip an entire pipeline, etc.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("*", {"lazy": True}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "gate"
    CATEGORY = "NH-Nodes/Logic"

    def check_lazy_status(self, input, enabled):
        """Only request the input when enabled=True.
        When False, input stays None and upstream never executes."""
        if enabled:
            # Input not yet evaluated -> request it
            if input is None:
                return ["input"]
        # enabled=False or input already evaluated -> don't need anything
        return []

    def gate(self, input, enabled):
        if enabled and input is not None:
            return (input,)
        return (ExecutionBlocker(None),)


# --- Dang ky ---
NODE_CLASS_MAPPINGS = {
    "NH_Compare": NH_Compare,
    "NH_LogicGate": NH_LogicGate,
    "NH_IfElse": NH_IfElse,
    "NH_SwitchN": NH_SwitchN,
    "NH_AnySwitch": NH_AnySwitch,
    "NH_GateSwitch": NH_GateSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_Compare": "Compare (NH)",
    "NH_LogicGate": "Logic Gate (NH)",
    "NH_IfElse": "If/Else (NH)",
    "NH_SwitchN": "Switch N (NH)",
    "NH_AnySwitch": "Any Switch (NH)",
    "NH_GateSwitch": "Gate Switch (NH)",
}
