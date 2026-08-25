"""Phase 2: Math & Random — NH_MathEval, NH_RandomChoice"""

import ast
import operator
import math
import random


# --- Safe eval engine ---
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

_SAFE_FUNCS = {
    "min": min, "max": max, "abs": abs,
    "round": round,
    "sqrt": math.sqrt, "floor": math.floor, "ceil": math.ceil,
    "clamp": lambda v, lo, hi: max(lo, min(hi, v)),
}


def _safe_eval(node, variables):
    """De quy eval AST node, chi cho phep operators va functions an toan."""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        if node.id in _SAFE_FUNCS:
            return _SAFE_FUNCS[node.id]
        raise ValueError(f"Unknown variable: {node.id}")
    elif isinstance(node, ast.BinOp):
        left = _safe_eval(node.left, variables)
        right = _safe_eval(node.right, variables)
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        if isinstance(node.op, ast.Div) and right == 0:
            print("[NH-Nodes] Warning: Division by zero, returning 0")
            return 0.0
        if isinstance(node.op, ast.Pow):
            # Gioi han power de tranh treo
            if abs(right) > 100:
                print("[NH-Nodes] Warning: Power too large, clamped to 100")
                right = 100 if right > 0 else -100
        return op(left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand, variables)
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
        return op(operand)
    elif isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name not in _SAFE_FUNCS:
            raise ValueError(f"Unknown function: {func_name}")
        args = [_safe_eval(arg, variables) for arg in node.args]
        return _SAFE_FUNCS[func_name](*args)
    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


class NH_MathEval:
    """Tinh toan bieu thuc toan hoc an toan bang AST parsing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": ("STRING", {
                    "default": "a + b",
                    "multiline": False
                }),
            },
            "optional": {
                "a": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "b": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "c": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "d": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "round_to": ("INT", {"default": -1, "min": -1, "max": 10}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "STRING")
    RETURN_NAMES = ("result_float", "result_int", "result_string")
    FUNCTION = "evaluate"
    CATEGORY = "NH-Nodes/Logic"

    def evaluate(self, expression, a=0.0, b=0.0, c=0.0, d=0.0, round_to=-1):
        variables = {"a": a, "b": b, "c": c, "d": d}
        try:
            tree = ast.parse(expression, mode='eval')
            result = _safe_eval(tree.body, variables)
        except Exception as e:
            print(f"[NH-Nodes] MathEval error: {e}")
            result = 0.0

        result = float(result)
        if round_to >= 0:
            result = round(result, round_to)

        return (result, int(result), str(result))


class NH_RandomChoice:
    """Chon ngau nhien 1 trong N inputs, ho tro trong so xac suat."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "weights": ("STRING", {"default": "1,1,1,1,1,1"}),
            },
            "optional": {
                "input_0": ("*",), "input_1": ("*",),
                "input_2": ("*",), "input_3": ("*",),
                "input_4": ("*",), "input_5": ("*",),
            }
        }

    RETURN_TYPES = ("*", "INT", "STRING")
    RETURN_NAMES = ("result", "picked_index", "probabilities")
    FUNCTION = "choose"
    CATEGORY = "NH-Nodes/Logic"

    def choose(self, seed, weights, **kwargs):
        connected = []
        for i in range(6):
            key = f"input_{i}"
            if key in kwargs and kwargs[key] is not None:
                connected.append((i, kwargs[key]))

        if not connected:
            return (None, -1, "")

        # Parse weights
        w = []
        for s in weights.split(","):
            try:
                w.append(max(0.0, float(s.strip())))
            except (ValueError, TypeError):
                w.append(1.0)
        while len(w) < len(connected):
            w.append(1.0)
        w = w[:len(connected)]

        # Normalize
        total = sum(w) or 1.0
        probs = [x / total for x in w]

        # Seed
        rng = random.Random(seed if seed > 0 else None)

        # Weighted pick
        r = rng.random()
        cumsum = 0.0
        for idx, (orig_i, item) in enumerate(connected):
            cumsum += probs[idx]
            if r <= cumsum:
                prob_str = ", ".join(f"{p:.1%}" for p in probs)
                return (item, orig_i, prob_str)

        # Fallback
        orig_i, item = connected[-1]
        prob_str = ", ".join(f"{p:.1%}" for p in probs)
        return (item, orig_i, prob_str)


# --- Dang ky ---
NODE_CLASS_MAPPINGS = {
    "NH_MathEval": NH_MathEval,
    "NH_RandomChoice": NH_RandomChoice,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_MathEval": "Math Eval (NH)",
    "NH_RandomChoice": "Random Choice (NH)",
}
