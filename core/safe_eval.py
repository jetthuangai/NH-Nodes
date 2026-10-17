"""Arithmetic expression evaluator shared by Math Eval (NH) and layer position expressions.

Accepted: numbers, named variables, + - * / // % **, unary + and -, and an optional
whitelist of functions. Everything is computed in floating point, so no expression can
build a huge integer and freeze the worker (``9**9**9`` simply overflows and is reported).
"""

import ast
import math
import numbers
import operator

# Python's ** returns a complex number for a negative base with a fractional exponent;
# math.pow raises instead, which is what a node input needs.
_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: math.pow,
}
_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_DIVISIONS = (ast.Div, ast.FloorDiv, ast.Mod)

MATH_FUNCTIONS = {
    "min": min,
    "max": max,
    "abs": abs,
    "round": lambda value, digits=0: round(value, int(digits)),
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
    "clamp": lambda value, low, high: max(low, min(high, value)),
}


class ExpressionError(ValueError):
    """Raised for any expression that cannot be evaluated to a finite number."""


def evaluate(expression, variables=None, functions=None, *, zero_division="zero",
             case_insensitive=False, on_warning=None):
    """Evaluate ``expression`` and return a finite float.

    zero_division: "zero" makes x/0, x//0 and x%0 evaluate to 0 (reported through
    ``on_warning``); "error" raises ExpressionError.
    case_insensitive: match variable names with str.casefold (keys must already be folded).
    """
    variables = variables or {}
    functions = functions or {}
    text = str(expression).strip()
    if not text:
        raise ExpressionError("Expression is empty")

    def warn(message):
        if on_warning is not None:
            on_warning(message)

    def number(value, what):
        # numpy scalars and 0-d torch tensors arrive from upstream nodes on FLOAT/INT sockets.
        if getattr(value, "ndim", None) == 0 and callable(getattr(value, "item", None)):
            value = value.item()
        if not isinstance(value, numbers.Real):  # bool is a Real (True + 1 == 2, as before)
            raise ExpressionError(f"{what} must be a number, got {type(value).__name__}")
        try:
            return float(value)
        except OverflowError as exc:
            raise ExpressionError(f"{what} is too large") from exc

    def lookup(name):
        key = name.casefold() if case_insensitive else name
        if key in variables:
            return number(variables[key], f"Variable '{name}'")
        raise ExpressionError(f"Unknown variable '{name}'")

    def visit(node):
        if isinstance(node, ast.Constant):
            return number(node.value, f"Constant {node.value!r}")
        if isinstance(node, ast.Name):
            return lookup(node.id)
        if isinstance(node, ast.BinOp):
            op = _BINARY.get(type(node.op))
            if op is None:
                raise ExpressionError(f"Operator '{type(node.op).__name__}' is not allowed")
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, _DIVISIONS) and right == 0:
                if zero_division == "zero":
                    warn("division by zero, using 0")
                    return 0.0
                raise ExpressionError("Division by zero")
            return float(op(left, right))
        if isinstance(node, ast.UnaryOp):
            op = _UNARY.get(type(node.op))
            if op is None:
                raise ExpressionError(f"Operator '{type(node.op).__name__}' is not allowed")
            return op(visit(node.operand))
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else None
            if name not in functions or node.keywords:
                raise ExpressionError(f"Function '{name}' is not allowed")
            return number(functions[name](*[visit(arg) for arg in node.args]), f"{name}()")
        raise ExpressionError(f"'{type(node).__name__}' is not allowed in an expression")

    try:
        result = visit(ast.parse(text, mode="eval").body)
    except ExpressionError:
        raise
    except SyntaxError as exc:
        raise ExpressionError(f"Invalid expression: {text}") from exc
    except OverflowError as exc:
        raise ExpressionError(f"Result is too large: {text}") from exc
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        # math domain errors, wrong argument counts, 0 ** -1 ...
        raise ExpressionError(f"Cannot evaluate {text}: {exc}") from exc
    except (RecursionError, MemoryError) as exc:
        # The parser raises MemoryError ("parser stack overflowed") for absurdly nested input.
        raise ExpressionError("Expression is nested too deeply") from exc

    if not math.isfinite(result):
        raise ExpressionError(f"Result is not a finite number: {text}")
    return result
