"""Phase 2: Math & Random — NH_MathEval, NH_RandomChoice"""

import random

from ...core import safe_eval

# Random Choice co 6 input: input_0 .. input_5
_CHOICE_SLOTS = 6


class NH_MathEval:
    """Tinh toan bieu thuc toan hoc an toan bang AST parsing (core/safe_eval.py)."""

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
        # Bieu thuc sai hoac ket qua khong huu han -> ValueError (ComfyUI hien loi, khong tra 0 im lang).
        # x / 0, x // 0, x % 0 -> 0 kem canh bao.
        try:
            result = safe_eval.evaluate(
                expression,
                {"a": a, "b": b, "c": c, "d": d},
                safe_eval.MATH_FUNCTIONS,
                zero_division="zero",
                on_warning=lambda message: print(f"[NH-Nodes] MathEval warning: {message}"),
            )
        except safe_eval.ExpressionError as exc:
            raise ValueError(f"Math Eval: {exc} (expression: {expression!r})") from exc
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
                "input_0": ("*", {"lazy": True}), "input_1": ("*", {"lazy": True}),
                "input_2": ("*", {"lazy": True}), "input_3": ("*", {"lazy": True}),
                "input_4": ("*", {"lazy": True}), "input_5": ("*", {"lazy": True}),
            }
        }

    RETURN_TYPES = ("*", "INT", "STRING")
    RETURN_NAMES = ("result", "picked_index", "probabilities")
    FUNCTION = "choose"
    CATEGORY = "NH-Nodes/Logic"
    DESCRIPTION = (
        "Pick one connected input at random (weight N belongs to input_N, the seed is used as given). "
        "Only the picked input is executed; the other ones never run. Connected inputs count even if they produce "
        "no value (an upstream node that returns nothing still takes its slot)."
    )

    @staticmethod
    def _connected_slots(kwargs):
        # "Noi" = key co mat trong kwargs (slot chua noi thi ComfyUI khong gui key). Gia tri None chi co nghia
        # la nhanh lazy chua chay, nen khong dung gia tri de biet slot nao dang noi. Tang dan theo slot.
        return [i for i in range(_CHOICE_SLOTS) if f"input_{i}" in kwargs]

    @staticmethod
    def _pick(seed, weights, slots):
        """Tra ve (slot duoc chon, xac suat cua cac slot dang noi). Chi phu thuoc seed, weights va tap slot
        dang noi, khong dung gia tri input: nho vay check_lazy_status chon dung slot ma choose se tra ve."""
        # Weight thu i thuoc input_i (khong phai input thu i dang noi), slot thieu weight = 1
        slot_weights = []
        for s in weights.split(",")[:_CHOICE_SLOTS]:
            try:
                slot_weights.append(max(0.0, float(s.strip())))
            except (ValueError, TypeError):
                slot_weights.append(1.0)
        slot_weights += [1.0] * (_CHOICE_SLOTS - len(slot_weights))
        w = [slot_weights[i] for i in slots]

        # Normalize
        total = sum(w) or 1.0
        probs = [x / total for x in w]

        # Seed dung dung gia tri (0 cung tat dinh): ComfyUI cache output theo input
        rng = random.Random(seed)

        # Weighted pick
        r = rng.random()
        cumsum = 0.0
        for slot, prob in zip(slots, probs):
            cumsum += prob
            if r <= cumsum:
                return slot, probs

        # Fallback
        return slots[-1], probs

    def check_lazy_status(self, seed, weights, **kwargs):
        slots = self._connected_slots(kwargs)
        if not slots:
            return []
        key = f"input_{self._pick(seed, weights, slots)[0]}"
        return [key] if kwargs[key] is None else []

    def choose(self, seed, weights, **kwargs):
        slots = self._connected_slots(kwargs)
        if not slots:
            return (None, -1, "")

        slot, probs = self._pick(seed, weights, slots)
        prob_str = ", ".join(f"{p:.1%}" for p in probs)
        return (kwargs[f"input_{slot}"], slot, prob_str)


# --- Dang ky ---
NODE_CLASS_MAPPINGS = {
    "NH_MathEval": NH_MathEval,
    "NH_RandomChoice": NH_RandomChoice,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_MathEval": "Math Eval (NH)",
    "NH_RandomChoice": "Random Choice (NH)",
}
