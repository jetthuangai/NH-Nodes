"""Phase 1: Logic Core — compare, branching, and gating nodes."""

from comfy_execution.graph_utils import ExecutionBlocker


_MAX_VALUE_MATCH_ITEMS = 64
_SWITCH_SLOTS = 10  # input_0 .. input_9 of Switch N


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
    DESCRIPTION = (
        'Compare a with b. "in" is plain membership: when b is a list, tuple, set or dict, a must be one of '
        "its items or keys; otherwise str(a) must be a substring of str(b), case-sensitive "
        '(13 is in "a13b" but not in "1024x768 3").'
    )

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
        """`a in b` as plain membership / substring, nothing fuzzier:

        - b is a dict: a is one of its keys.
        - b is a list, tuple or set: a equals one of its items.
        - anything else: str(a) is a substring of str(b), case-sensitive, so 13 in "a13b" is True
          and 13 in "1024x768 3" is False.
        Values also match by their text (7 matches "7"), because widgets hand over strings.
        """
        if isinstance(b, dict):
            return a in b or str(a) in [str(x) for x in b.keys()]
        if isinstance(b, (list, tuple, set)):
            return a in b or str(a) in [str(x) for x in b]
        return str(a) in str(b)


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
                "if_true": ("*", {"lazy": True}),
                "if_false": ("*", {"lazy": True}),
            }
        }

    RETURN_TYPES = ("*", "BOOLEAN")
    RETURN_NAMES = ("result", "condition_out")
    FUNCTION = "switch"
    CATEGORY = "NH-Nodes/Logic"
    DESCRIPTION = "Pick if_true or if_false by the condition. Only the chosen branch is executed; the other one never runs."

    def check_lazy_status(self, condition, if_true=None, if_false=None):
        # Chi xin nhanh duoc chon; nhanh con lai khong bao gio chay
        if condition and if_true is None:
            return ["if_true"]
        if (not condition) and if_false is None:
            return ["if_false"]
        return []

    def switch(self, condition, if_true=None, if_false=None):
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
                "input_0": ("*", {"lazy": True}),
                "input_1": ("*", {"lazy": True}),
                "input_2": ("*", {"lazy": True}),
                "input_3": ("*", {"lazy": True}),
                "input_4": ("*", {"lazy": True}),
                "input_5": ("*", {"lazy": True}),
                "input_6": ("*", {"lazy": True}),
                "input_7": ("*", {"lazy": True}),
                "input_8": ("*", {"lazy": True}),
                "input_9": ("*", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("*", "INT")
    RETURN_NAMES = ("result", "count")
    FUNCTION = "switch"
    CATEGORY = "NH-Nodes/Logic"
    DESCRIPTION = (
        "Pick input_<index>. Only the chosen input is executed. If that slot is not connected, the lowest connected "
        "slot is used instead. count = number of connected slots. Connected inputs count even if they produce no "
        "value (an upstream node that returns nothing still takes its slot)."
    )

    @staticmethod
    def _connected_slots(kwargs):
        # "Noi" = key co mat trong kwargs (slot chua noi thi ComfyUI khong gui key). Gia tri None chi co nghia
        # la nhanh lazy chua chay, nen khong dung gia tri de biet slot nao dang noi. Tang dan theo slot.
        return [i for i in range(_SWITCH_SLOTS) if f"input_{i}" in kwargs]

    @staticmethod
    def _chosen_slot(index, slots):
        # Slot duoc chon; neu slot do khong noi thi roi ve slot noi thap nhat (nhu truoc day)
        return next((slot for slot in slots if slot == index), slots[0])

    def check_lazy_status(self, index, **kwargs):
        slots = self._connected_slots(kwargs)
        if not slots:
            return []
        key = f"input_{self._chosen_slot(index, slots)}"
        return [key] if kwargs[key] is None else []

    def switch(self, index, **kwargs):
        slots = self._connected_slots(kwargs)
        count = len(slots)
        if count == 0:
            print("[NH-Nodes] Warning: SwitchN has no connected inputs")
            return (None, 0)

        chosen = self._chosen_slot(index, slots)
        if chosen != index:
            print(f"[NH-Nodes] SwitchN: index {index} not connected, falling back to input_{chosen}")
        return (kwargs[f"input_{chosen}"], count)


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


class NH_ValueMatchIndex:
    """Return the 1-based index of the only configured value matching input."""

    @classmethod
    def INPUT_TYPES(cls):
        values = {
            f"value_{index}": ("STRING", {"default": "", "multiline": False})
            for index in range(1, _MAX_VALUE_MATCH_ITEMS + 1)
        }

        return {
            "required": {
                "input": ("*",),
                "compare_mode": (["in", "is"], {"default": "in"}),
                "value_count": (
                    "INT",
                    {"default": 5, "min": 1, "max": _MAX_VALUE_MATCH_ITEMS},
                ),
                "no_match_index": ("INT", {"default": -1, "min": -9999, "max": 9999}),
                "multi_match_index": ("INT", {"default": -1, "min": -9999, "max": 9999}),
            },
            "optional": values,
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("index",)
    FUNCTION = "match_index"
    CATEGORY = "NH-Nodes/Logic"

    def match_index(self, input, compare_mode, value_count, no_match_index, multi_match_index, **kwargs):
        input_text = "" if input is None else str(input)
        count = max(1, min(int(value_count), _MAX_VALUE_MATCH_ITEMS))
        matches = []

        for index in range(1, count + 1):
            raw_value = kwargs.get(f"value_{index}", "")
            value = "" if raw_value is None else str(raw_value)
            if value == "":
                continue

            if compare_mode == "is":
                matched = value == input_text
            else:
                matched = value in input_text

            if matched:
                matches.append(index)
                if len(matches) > 1:
                    return (int(multi_match_index),)

        if len(matches) == 1:
            return (matches[0],)

        return (int(no_match_index),)


NODE_CLASS_MAPPINGS = {
    "NH_Compare": NH_Compare,
    "NH_LogicGate": NH_LogicGate,
    "NH_IfElse": NH_IfElse,
    "NH_SwitchN": NH_SwitchN,
    "NH_AnySwitchBoolean": NH_AnySwitchBoolean,
    "NH_AnySwitch": NH_AnyBranchSwitch,
    "NH_GateSwitch": NH_GateSwitch,
    "NH_ValueMatchIndex": NH_ValueMatchIndex,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_Compare": "Compare (NH)",
    "NH_LogicGate": "Logic Gate (NH)",
    "NH_IfElse": "If/Else (NH)",
    "NH_SwitchN": "Switch N (NH)",
    "NH_AnySwitchBoolean": "Any Switch (NH)",
    "NH_AnySwitch": "Any Branch Switch (NH)",
    "NH_GateSwitch": "Gate Switch (NH)",
    "NH_ValueMatchIndex": "Value Match Index (NH)",
}
