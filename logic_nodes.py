"""Phase 1: Logic Core — compare, branching, and gating nodes."""

import re

from comfy_execution.graph_utils import ExecutionBlocker


class NH_Compare:
    """Compare two values and return a BOOL result."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("*",),
                "b": ("*",),
                "op": (["is", "not", ">", "<", ">=", "<=", "in"],),
            },
            "optional": {
                "type_cast": (["auto", "int", "float", "string"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "*", "*")
    RETURN_NAMES = ("result", "a_passthrough", "b_passthrough")
    FUNCTION = "compare"
    CATEGORY = "NH-Nodes/Logic"

    def compare(self, a, b, op, type_cast="auto"):
        if a is None or b is None:
            return (False, a, b)

        try:
            if op == "in":
                result = self._contains(a, b)
            else:
                va, vb = self._cast(a, b, type_cast)
                if op == "is":
                    result = va == vb
                elif op == "not":
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
        except Exception as exc:
            print(f"[NH-Nodes] Compare error: {exc}")
            result = False

        return (bool(result), a, b)

    @staticmethod
    def _cast(a, b, mode):
        if mode == "int":
            return int(a), int(b)
        if mode == "float":
            return float(a), float(b)
        if mode == "string":
            return str(a), str(b)
        try:
            return float(a), float(b)
        except (ValueError, TypeError):
            return str(a), str(b)

    @staticmethod
    def _contains(a, b):
        if isinstance(b, dict):
            return a in b or str(a) in [str(x) for x in b.keys()]
        if isinstance(b, (list, tuple, set)):
            return a in b or str(a) in [str(x) for x in b]
        needle = str(a)
        haystack = str(b)
        if needle in haystack:
            return True

        if not NH_Compare._supports_subsequence(a, needle):
            return False

        idx = 0
        for char in haystack:
            if idx < len(needle) and char == needle[idx]:
                idx += 1
        return idx == len(needle)

    @staticmethod
    def _supports_subsequence(a, needle):
        if isinstance(a, (int, float)):
            return True
        return bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", needle.strip()))


class NH_LogicGate:
    """Combine BOOL values using common logic operations."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("BOOLEAN", {"default": False}),
                "op": (["AND", "OR", "NOT", "XOR", "NAND", "NOR"],),
            },
            "optional": {
                "b": ("BOOLEAN", {"default": False}),
            },
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
    """Route data according to a boolean condition."""

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
        return (if_true if condition else if_false, condition)


class NH_SwitchN:
    """Pick one input from up to 10 candidates by index."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 9}),
            },
            "optional": {
                "input_0": ("*",),
                "input_1": ("*",),
                "input_2": ("*",),
                "input_3": ("*",),
                "input_4": ("*",),
                "input_5": ("*",),
                "input_6": ("*",),
                "input_7": ("*",),
                "input_8": ("*",),
                "input_9": ("*",),
            },
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

        first_key = sorted(connected.keys())[0]
        print(f"[NH-Nodes] SwitchN: index {index} not connected, falling back to input_{first_key}")
        return (connected[first_key], count)


class NH_AnySwitchBoolean:
    """Lazy any-type boolean switch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("BOOLEAN", {"default": True}),
                "input_true": ("*", {"lazy": True}),
                "input_false": ("*", {"lazy": True}),
            }
        }

    RETURN_TYPES = ("*", "BOOLEAN")
    RETURN_NAMES = ("result", "condition")
    FUNCTION = "select"
    CATEGORY = "NH-Nodes/Logic"

    def check_lazy_status(self, condition, input_true=None, input_false=None):
        if condition and input_true is None:
            return ["input_true"]
        if (not condition) and input_false is None:
            return ["input_false"]
        return []

    def select(self, condition, input_true=None, input_false=None):
        return (input_true if condition else input_false, condition)


class NH_AnyBranchSwitch:
    """Route one input to one of five downstream branches."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("*",),
                "index": ("INT", {"default": 0, "min": 0, "max": 4}),
            }
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
    """Pass or block data and upstream execution."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("*", {"lazy": True}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "gate"
    CATEGORY = "NH-Nodes/Logic"

    def check_lazy_status(self, input, enabled):
        if enabled and input is None:
            return ["input"]
        return []

    def gate(self, input, enabled):
        if enabled and input is not None:
            return (input,)
        return (ExecutionBlocker(None),)


NODE_CLASS_MAPPINGS = {
    "NH_Compare": NH_Compare,
    "NH_LogicGate": NH_LogicGate,
    "NH_IfElse": NH_IfElse,
    "NH_SwitchN": NH_SwitchN,
    "NH_AnySwitchBoolean": NH_AnySwitchBoolean,
    "NH_AnySwitch": NH_AnyBranchSwitch,
    "NH_GateSwitch": NH_GateSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_Compare": "Compare (NH)",
    "NH_LogicGate": "Logic Gate (NH)",
    "NH_IfElse": "If/Else (NH)",
    "NH_SwitchN": "Switch N (NH)",
    "NH_AnySwitchBoolean": "Any Switch (NH)",
    "NH_AnySwitch": "Any Branch Switch (NH)",
    "NH_GateSwitch": "Gate Switch (NH)",
}
