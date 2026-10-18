"""Shared helpers: safe arithmetic evaluation, tensor/PIL conversion, hex colours."""

import importlib
import math

import pytest
import torch

safe_eval = importlib.import_module("NH-Nodes.core.safe_eval")
utils = importlib.import_module("NH-Nodes.core.utils")

evaluate = safe_eval.evaluate
ExpressionError = safe_eval.ExpressionError


# --- safe_eval ---

@pytest.mark.parametrize(
    "expression,expected",
    [
        ("a + b * 2", 7.0),
        ("(a + b) / 2", 2.0),
        ("a ** 2", 1.0),
        ("-a + +b", 2.0),
        ("b // 2", 1.0),
        ("b % 2", 1.0),
        ("clamp(10, 0, 5)", 5.0),
        ("round(3.14159, 2)", 3.14),
        ("sqrt(16) + floor(1.7) + ceil(1.2) + abs(-1) + min(a, b) + max(a, b)", 4 + 1 + 2 + 1 + 1 + 3),
    ],
)
def test_math_expressions(expression, expected):
    assert evaluate(expression, {"a": 1, "b": 3}, safe_eval.MATH_FUNCTIONS) == pytest.approx(expected)


@pytest.mark.parametrize("expression", ["9**9**9", "10**400", "(10**100)**100**100", "1e308 * 10"])
def test_runaway_numbers_fail_fast_instead_of_freezing(expression):
    with pytest.raises(ExpressionError):
        evaluate(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "'a' * 10**10",           # string constants are not numbers
        "__import__('os')",       # calls outside the whitelist
        "a.real",                 # attribute access
        "[1, 2][0]",              # containers / subscripts
        "lambda: 1",
        "(-8) ** 0.5",            # would be a complex number
        "sqrt(-1)",
        "1e308 * 10 - 1e308 * 10",  # inf - inf = nan
        "unknown + 1",
        "",
        "1 +",
    ],
)
def test_rejected_expressions(expression):
    with pytest.raises(ExpressionError):
        evaluate(expression, {"a": 1}, safe_eval.MATH_FUNCTIONS)


def test_division_by_zero_modes():
    warnings = []
    assert evaluate("a / 0 + 1", {"a": 5}, on_warning=warnings.append) == 1.0
    assert evaluate("a // 0", {"a": 5}) == 0.0
    assert evaluate("a % 0", {"a": 5}) == 0.0
    assert warnings == ["division by zero, using 0"]
    with pytest.raises(ExpressionError, match="Division by zero"):
        evaluate("a / 0", {"a": 5}, zero_division="error")


def test_case_insensitive_variables():
    assert evaluate("BG_W / 2 - w1 / 2", {"bg_w": 100, "w1": 20}, case_insensitive=True) == 40.0
    with pytest.raises(ExpressionError):
        evaluate("BG_W", {"bg_w": 100})


def test_non_numeric_variables_are_rejected():
    with pytest.raises(ExpressionError):
        evaluate("a", {"a": "5"})
    with pytest.raises(ExpressionError):
        evaluate("a", {"a": None})
    with pytest.raises(ExpressionError):
        evaluate("a", {"a": torch.ones(2)})  # not a scalar


def test_bools_numpy_and_torch_scalars_count_as_numbers():
    import numpy as np

    assert evaluate("True + 1") == 2.0
    assert evaluate("a + b", {"a": np.float32(2), "b": np.int64(3)}) == 5.0
    assert evaluate("a * 2", {"a": torch.tensor(2.5)}) == 5.0
    assert evaluate("a + 1", {"a": np.bool_(True)}) == 2.0


def test_absurd_nesting_is_an_expression_error_not_a_crash():
    with pytest.raises(ExpressionError):
        evaluate("-" * 100000 + "1")
    with pytest.raises(ExpressionError):
        evaluate("(" * 5000 + "1" + ")" * 5000)


def test_result_is_always_a_finite_float():
    result = evaluate("2 ** 10")
    assert isinstance(result, float) and math.isfinite(result) and result == 1024.0


# --- tensor_to_pil ---

def test_tensor_to_pil_clips_instead_of_wrapping():
    image = torch.tensor([[[1.2, -0.1, 0.5]]])  # 1x1 RGB with out-of-range values
    pixel = utils.tensor_to_pil(image).getpixel((0, 0))
    assert pixel == (255, 0, 127)


@pytest.mark.parametrize("channels,mode", [(1, "L"), (3, "RGB"), (4, "RGBA")])
def test_tensor_to_pil_channel_modes(channels, mode):
    image = torch.rand((1, 8, 6, channels))
    pil = utils.tensor_to_pil(image)
    assert pil.mode == mode and pil.size == (6, 8)


def test_tensor_to_pil_rejects_batches_with_clear_message():
    with pytest.raises(ValueError, match="batch of 2"):
        utils.tensor_to_pil(torch.rand((2, 4, 4, 3)))


def _flat(channels, level=0.5):
    """1x1 image whose channel k holds (10 * (k + 1) + level) / 255, i.e. the byte value 10 * (k + 1)."""
    return torch.tensor([[[(10 * (k + 1) + level) / 255.0 for k in range(channels)]]])


@pytest.mark.parametrize("channels,expected", [(1, (10, 10, 10)), (3, (10, 20, 30)), (4, (10, 20, 30))])
def test_tensor_to_pil_rgb_always_gives_rgb(channels, expected):
    pil = utils.tensor_to_pil(_flat(channels), rgb=True)
    assert pil.mode == "RGB" and pil.getpixel((0, 0)) == expected  # grey is expanded, alpha is dropped


@pytest.mark.parametrize(
    "channels,mode,expected",
    [(1, "RGB", (10, 10, 10)), (3, "RGB", (10, 20, 30)), (4, "RGBA", (10, 20, 30, 40))],
)
def test_tensor_to_pil_keep_alpha_keeps_a_fourth_channel_only(channels, mode, expected):
    pil = utils.tensor_to_pil(_flat(channels), rgb=True, keep_alpha=True)
    assert pil.mode == mode and pil.getpixel((0, 0)) == expected


def test_tensor_to_pil_two_channel_image_becomes_rgb_whatever_the_flags():
    for keep_alpha in (False, True):
        assert utils.tensor_to_pil(_flat(2), rgb=True, keep_alpha=keep_alpha).mode == "RGB"


def test_tensor_to_pil_keep_alpha_alone_changes_nothing():
    for channels, mode in ((1, "L"), (3, "RGB"), (4, "RGBA")):
        assert utils.tensor_to_pil(_flat(channels), keep_alpha=True).mode == mode


def test_tensor_to_pil_flags_are_keyword_only():
    with pytest.raises(TypeError):
        utils.tensor_to_pil(_flat(3), True)


def test_tensor_to_pil_accepts_a_single_image_batch_with_the_flags():
    pil = utils.tensor_to_pil(torch.rand((1, 5, 4, 4)), rgb=True, keep_alpha=True)
    assert pil.mode == "RGBA" and pil.size == (4, 5)


# --- pil_to_tensor ---

def test_pil_to_tensor_gives_a_batch_of_one_float32_image():
    from PIL import Image

    pil = Image.new("RGB", (4, 3), (255, 0, 51))
    tensor = utils.pil_to_tensor(pil)
    assert tensor.shape == (1, 3, 4, 3) and tensor.dtype == torch.float32
    assert torch.allclose(tensor[0, 0, 0], torch.tensor([1.0, 0.0, 51 / 255]))


# --- parse_hex_color ---

@pytest.mark.parametrize(
    "text,expected",
    [
        ("#FF8000", (255, 128, 0, 255)),
        ("ff8000", (255, 128, 0, 255)),
        ("#F80", (255, 136, 0, 255)),
        ("#11223344", (17, 34, 51, 68)),
        ("  #000000  ", (0, 0, 0, 255)),
    ],
)
def test_parse_hex_color(text, expected):
    assert utils.parse_hex_color(text) == expected


@pytest.mark.parametrize("text", ["", "#12", "#GGGGGG", "red", None])
def test_parse_hex_color_invalid_returns_default(text):
    assert utils.parse_hex_color(text, default=(1, 2, 3, 4)) == (1, 2, 3, 4)
