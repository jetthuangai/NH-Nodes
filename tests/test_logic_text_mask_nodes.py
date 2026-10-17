"""Regression tests: logic, text, batch, utils and mask nodes.

Each fix has tests that fail on the code before release 1.5.0; a few cases only pin behaviour that must
keep working (variables, rounding, plain membership, filled template variables). Sections follow the
1.5.0 entries of docs/CHANGELOG.md.
"""

import importlib
import random

import numpy as np
import pytest
import torch
from PIL import Image

math_nodes = importlib.import_module("NH-Nodes.nodes.logic.math_nodes")
logic_nodes = importlib.import_module("NH-Nodes.nodes.logic.logic_nodes")
prompt_nodes = importlib.import_module("NH-Nodes.nodes.text.prompt_nodes")
text_nodes = importlib.import_module("NH-Nodes.nodes.text.text_nodes")
split_lines = importlib.import_module("NH-Nodes.nodes.text.text_split_lines")
batch_nodes = importlib.import_module("NH-Nodes.nodes.batch.batch_nodes")
pipe_nodes = importlib.import_module("NH-Nodes.nodes.utils.universal_pipe_nodes")
utils_nodes = importlib.import_module("NH-Nodes.nodes.utils.utils_nodes")
aspect = importlib.import_module("NH-Nodes.nodes.mask.mask_aspect_match")
utils_mask = importlib.import_module("NH-Nodes.core.utils_mask")


# --- 1. Math Eval: bad expressions raise, x / 0 only zeroes its own term ---

def _evaluate(expression, **kwargs):
    return math_nodes.NH_MathEval().evaluate(expression, **kwargs)


def test_math_eval_keeps_variables_functions_and_rounding():
    assert _evaluate("a + b * 2", a=1.0, b=3.0) == (7.0, 7, "7.0")
    assert _evaluate("clamp(a * 10, b, c) + max(d, 1)", a=0.5, b=1.0, c=3.0, d=0.0) == (4.0, 4, "4.0")
    assert _evaluate("a / 3", a=10.0, round_to=2) == (3.33, 3, "3.33")
    assert _evaluate("a", a=2.7, round_to=0) == (3.0, 3, "3.0")  # result_int is int() of the rounded value
    assert _evaluate("a", a=2.7) == (2.7, 2, "2.7")  # round_to = -1: no rounding


@pytest.mark.parametrize(
    "expression",
    ["", "1 +", "unknown + 1", "__import__('os')", "'a' * 3", "(-8) ** 0.5", "1e308 * 10", "9 ** 9 ** 9", "sqrt(-1)"],
)
def test_math_eval_invalid_expression_raises_value_error(expression):
    with pytest.raises(ValueError):
        _evaluate(expression)


def test_math_eval_division_and_modulo_by_zero_only_zero_their_own_term(capsys):
    node = math_nodes.NH_MathEval()
    assert node.evaluate("a + b / 0", a=1.0, b=5.0)[0] == 1.0
    assert node.evaluate("a + b // 0", a=1.0, b=5.0)[0] == 1.0  # used to zero the whole result
    assert node.evaluate("a + b % 0", a=1.0, b=5.0)[0] == 1.0
    assert capsys.readouterr().out.count("[NH-Nodes] MathEval warning: division by zero") == 3


# --- 2. Random Choice: weight N belongs to input_N, the seed is used as given ---

def test_random_choice_weight_belongs_to_its_input_slot():
    node = math_nodes.NH_RandomChoice()
    # input_0 is not connected: input_1 keeps the weight of slot 1 and input_2 the weight of slot 2.
    assert node.choose(1, "1,1,10", input_1="B", input_2="C")[2] == "9.1%, 90.9%"
    picks = [node.choose(seed, "1,1,10", input_1="B", input_2="C")[1] for seed in range(1, 401)]
    assert picks.count(2) > 300  # about 91 % of 400; matching weights to connected inputs gave 50 %
    # A slot without a weight counts as 1: input_3 is the 4th weight, not the 2nd.
    assert node.choose(1, "1,2", input_0="A", input_3="D")[2] == "50.0%, 50.0%"


def test_random_choice_zero_weight_input_is_never_picked():
    node = math_nodes.NH_RandomChoice()
    picks = {node.choose(seed, "0,1,1", input_0="A", input_1="B", input_2="C")[1] for seed in range(200)}
    assert picks == {1, 2}


def test_random_choice_seed_zero_is_deterministic():
    node = math_nodes.NH_RandomChoice()
    inputs = {f"input_{index}": chr(ord("A") + index) for index in range(6)}
    weights = "1,1,1,1,1,1"
    assert len({node.choose(0, weights, **inputs)[1] for _ in range(30)}) == 1
    assert node.choose(5, weights, **inputs) == node.choose(5, weights, **inputs)
    assert {node.choose(seed, weights, **inputs)[1] for seed in range(60)} == set(range(6))


# --- 3. Prompt Template: an empty variable never glues its neighbours together ---

@pytest.mark.parametrize(
    "template,names,values,expected,missing",
    [
        ("red dress, {style}, 8k photo", "style", {}, "red dress, 8k photo", "style"),
        ("{style}, red dress, 8k photo", "style", {}, "red dress, 8k photo", "style"),  # at the start
        ("red dress, 8k photo, {style}", "style", {}, "red dress, 8k photo", "style"),  # at the end
        ("red dress, {a}, {b}, 8k photo", "a,b", {}, "red dress, 8k photo", "a, b"),  # consecutive
        ("{a}, {b}, red dress", "a,b", {}, "red dress", "a, b"),
        ("red dress, {a}, {b}", "a,b", {}, "red dress", "a, b"),
        ("{a}, {b}", "a,b", {}, "", "a, b"),
        ("{a}", "a", {}, "", "a"),
        ("a {color} dress", "color", {}, "a dress", "color"),  # inside a sentence
        ("a {color} {garment}, {style}, 8k photo", "color,garment,style", {"var2": "dress"}, "a dress, 8k photo", "color, style"),
        ("a red {garment}, {style}, 8k photo", "garment,style", {}, "a red, 8k photo", "garment, style"),
        ("line one, {x}\nline two, {y}, tail\n{x}, last", "x,y", {}, "line one\nline two, tail\nlast", "x, y"),
        ("a, {x}\r\nb", "x", {}, "a\r\nb", "x"),
        ("a, {x}, b", "x", {"var1": None}, "a, b", "x"),  # an unconnected input counts as empty
    ],
)
def test_prompt_template_empty_variable_keeps_neighbours_apart(template, names, values, expected, missing):
    assert prompt_nodes.NH_PromptTemplate().render(template, names, **values) == (expected, missing)


def test_prompt_template_filled_variables_are_substituted():
    node = prompt_nodes.NH_PromptTemplate()
    result = node.render("a {color} {garment}, {style}, 8k photo", "color,garment,style", var1="red", var2="dress", var3="studio")
    assert result == ("a red dress, studio, 8k photo", "")


# --- 4. Seeds are used as given, so seed 0 is reproducible too ---

@pytest.mark.parametrize("seed", [0, 3])
def test_prompt_scheduler_random_mode_uses_the_seed_literally(seed):
    node = prompt_nodes.NH_PromptScheduler()
    prompts = "\n".join("abcdef")
    for step in range(8):
        index = node.schedule(prompts, step, 10, "random", seed=seed)[2]
        assert index == random.Random(seed + step).randint(0, 5)
        assert index == node.schedule(prompts, step, 10, "random", seed=seed)[2]


def test_text_split_lines_seed_zero_is_deterministic():
    node = split_lines.NH_TextSplitLines()
    text = "\n".join("abcdef")
    expected = list("abcdef")
    random.Random(0).shuffle(expected)
    assert node.split_lines(text, 0, "shuffle", 0)[0] == expected
    assert len({tuple(node.split_lines(text, 0, "shuffle", 0)[0]) for _ in range(20)}) == 1
    # index -1 picks a random line with the same seeded generator
    picks = {node.split_lines(text, -1, "sequential", 0)[1] for _ in range(30)}
    assert picks == {text.split("\n")[random.Random(0).randint(0, 5)]}


def test_text_random_line_seed_zero_is_deterministic():
    node = split_lines.NH_TextRandomLine()
    text = "\n".join("abcdef")
    expected = random.Random(0).randint(0, 5)
    assert [node.pick(text, 0)[1] for _ in range(30)] == [expected] * 30


# --- 5. Regex Extract: a group that did not take part is an empty string ---

def test_regex_extract_unmatched_optional_group_is_empty():
    node = text_nodes.NH_RegexExtract()
    assert node.extract("b", r"(a)|(b)", "match") == ("b", True, "\nb", 1)
    assert node.extract("x-y", r"(x)(-)?(z)?", "match") == ("x-", True, "x\n-\n", 1)
    assert node.extract("ab", r"(a)(c)?", "match") == ("a", True, "a\n", 1)
    assert node.extract("ab", r"a", "match") == ("a", True, "a", 1)  # no group: the whole match


# --- 6. Pack Universal: the package is indexed by slot ---

def test_pack_universal_keeps_slot_positions():
    pack = pipe_nodes.NH_PackUniversal().pack(input_1="ONE", input_3="THREE")[0]
    unpack = pipe_nodes.NH_UnpackUniversal()
    assert unpack.unpack(0, package=pack) == (None,)  # input_0 was left empty
    assert unpack.unpack(1, package=pack) == ("ONE",)
    assert unpack.unpack(2, package=pack) == (None,)
    assert unpack.unpack(3, package=pack) == ("THREE",)
    assert unpack.unpack(9, package=pack) == (None,)
    assert unpack.unpack(1, pipe=pack) == ("ONE",)  # the legacy `pipe` input reads the same package


def test_pack_universal_ignores_argument_order_and_keeps_falsy_values():
    pack = pipe_nodes.NH_PackUniversal().pack(input_2="C", input_0=0, input_1="")[0]
    unpack = pipe_nodes.NH_UnpackUniversal()
    assert [unpack.unpack(index, package=pack)[0] for index in range(4)] == [0, "", "C", None]
    assert pipe_nodes.NH_UnpackUniversal().unpack(0, package=pipe_nodes.NH_PackUniversal().pack()[0]) == (None,)


# --- 7. Compare "in": plain membership / substring ---

@pytest.mark.parametrize("needle,haystack", [(13, "1024x768 3"), ("13", "1024x768 3"), (15, "1,05"), (25, "2 and 5")])
def test_compare_in_is_not_a_subsequence_match(needle, haystack):
    assert logic_nodes.NH_Compare().compare(needle, haystack, "in")[0] is False


@pytest.mark.parametrize(
    "needle,haystack",
    [
        ("cat", "concatenate"),
        (13, "size 1024x13"),
        (7, [1, 7]),
        ("7", [1, 7]),
        (7, ["1", "7"]),
        ("k", {"k": 1}),
        (1, {"1": "x"}),
        ("", "anything"),
    ],
)
def test_compare_in_finds_substrings_and_members(needle, haystack):
    assert logic_nodes.NH_Compare().compare(needle, haystack, "in")[0] is True


@pytest.mark.parametrize("needle,haystack", [("dog", "concatenate"), ("CAT", "concatenate"), (4, [1, 7]), ("z", {"k": 1})])
def test_compare_in_rejects_non_members(needle, haystack):
    assert logic_nodes.NH_Compare().compare(needle, haystack, "in")[0] is False


# --- 8. Batch Merge: channel counts follow batch_a ---

def test_batch_merge_aligns_channels_to_batch_a():
    node = batch_nodes.NH_BatchMerge()
    rgb = torch.full((2, 4, 4, 3), 0.25)
    rgba = torch.full((1, 4, 4, 4), 0.75)
    rgba[..., 3] = 0.5

    merged, count = node.merge(rgb, rgba)  # RGB first: the alpha channel is dropped
    assert merged.shape == (3, 4, 4, 3) and count == 3
    assert torch.equal(merged[:2], rgb) and torch.equal(merged[2], rgba[0, ..., :3])

    merged, count = node.merge(rgba, rgb)  # RGBA first: RGB images get a fully opaque alpha
    assert merged.shape == (3, 4, 4, 4) and count == 3
    assert torch.equal(merged[0], rgba[0])
    assert torch.equal(merged[1:, ..., :3], rgb)
    assert torch.all(merged[1:, ..., 3] == 1.0)


@pytest.mark.parametrize("resize_mode", ["stretch", "crop", "pad"])
def test_batch_merge_handles_size_and_channel_mismatch_together(resize_mode):
    node = batch_nodes.NH_BatchMerge()
    rgba = torch.rand((1, 4, 4, 4))
    rgb_big = torch.rand((2, 8, 6, 3))
    rgb_same = torch.rand((1, 4, 4, 3))
    merged, count = node.merge(rgba, rgb_big, rgb_same, resize_mode=resize_mode)
    assert merged.shape == (4, 4, 4, 4) and count == 4
    if resize_mode == "pad":
        # 8x6 fits as 4x3: the picture is opaque, the 1 px letterbox column is transparent (like RGBA input)
        assert torch.all(merged[1:3, :, :3, 3] == 1.0) and torch.all(merged[1:3, :, 3, 3] == 0.0)
        assert torch.all(merged[3, ..., 3] == 1.0)
    else:
        assert torch.all(merged[1:, ..., 3] == 1.0)


def test_batch_merge_rejects_unsupported_channel_counts_clearly():
    node = batch_nodes.NH_BatchMerge()
    with pytest.raises(ValueError, match="channel"):
        node.merge(torch.rand((1, 4, 4, 3)), torch.rand((1, 4, 4, 2)))


# --- 9. NH Mask Aspect Ratio Match ---

def _canvas(*boxes, size=100):
    """MASK batch [B, size, size]; each box is (x1, y1, x2, y2) with an exclusive end."""
    masks = torch.zeros((len(boxes), size, size))
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        masks[index, y1:y2, x1:x2] = 1.0
    return masks


def _bounds(mask):
    """(x1, y1, x2, y2), inclusive, of the pixels above 0.5."""
    rows, cols = torch.nonzero(mask > 0.5, as_tuple=True)
    return int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())


TALL = _canvas((20, 10, 50, 70))  # 30 x 60 -> ratio 0.5
WIDE = _canvas((10, 20, 70, 50))  # 60 x 30 -> ratio 2.0


def _match(target, box, mode):
    return aspect.NH_MaskAspectRatioMatch().match_aspect_ratio(target, box, mode)[0]


def test_aspect_match_pad_grows_the_box_to_the_requested_size():
    box = _canvas((30, 30, 50, 50))  # 20 x 20 in the middle
    out = _match(TALL, box, "pad")
    assert out.shape == box.shape
    assert _bounds(out[0]) == (30, 20, 49, 59)  # 20 x 40, same centre, nothing shifted
    assert float(out.sum()) == 20 * 40


def test_aspect_match_pad_keeps_the_original_pixels_where_they_were():
    box = _canvas((30, 30, 50, 50))
    box[0, 38:42, 38:42] = 0.0  # a hole must stay exactly there, the padding is added around it
    out = _match(TALL, box, "pad")[0]
    assert float(out[38:42, 38:42].sum()) == 0.0
    assert torch.equal(out[30:50, 30:50], box[0, 30:50, 30:50])
    assert float(out[20:30, 30:50].min()) == 1.0 and float(out[50:60, 30:50].min()) == 1.0


@pytest.mark.parametrize("mode,expected", [("stretch", (0, 40, 39, 59)), ("pad", (0, 40, 39, 59)), ("crop", (0, 45, 19, 54))])
def test_aspect_match_box_touching_the_left_edge(mode, expected):
    box = _canvas((0, 40, 20, 60))  # x1 = 0 used to crash pad and crop (negative paste offset)
    out = _match(WIDE, box, mode)
    assert out.shape == box.shape
    # The widened box slides inwards instead of being cut by the edge, so it keeps the requested ratio.
    assert _bounds(out[0]) == expected


def test_aspect_match_stretch_is_centred_and_scaled_not_shifted():
    box = _canvas((30, 30, 50, 50))
    box[0, 38:42, 38:42] = 0.0
    out = _match(TALL, box, "stretch")[0]
    assert _bounds(out) == (30, 20, 49, 59)  # the unchanged width must not move by a pixel
    assert float(out.sum()) == 20 * 40 - 4 * 8  # the 4 x 4 hole is stretched to 4 x 8


def test_aspect_match_stretch_larger_than_the_image_shows_the_middle_of_the_content():
    box = _canvas((40, 40, 60, 60))
    box[0, 45:55, 40:44] = 0.0  # a hole on the far left of the box
    extreme = _canvas((0, 0, 1000, 10), size=1100)  # ratio 100: the stretched box would be 2000 px wide
    out = _match(extreme, box, "stretch")[0]
    assert out.shape == box.shape[1:]
    # Only the middle of the stretched content fits into the 100 px wide image, so the hole is not in view.
    assert _bounds(out) == (0, 40, 99, 59)
    assert float(out.sum()) == 100 * 20


def test_aspect_match_crop_trims_around_the_centre_without_shifting():
    out = _match(TALL, _canvas((30, 30, 50, 50)), "crop")
    assert _bounds(out[0]) == (35, 30, 44, 49)  # 10 x 20 taken from the middle of 20 x 20
    assert float(out.sum()) == 10 * 20


def test_aspect_match_processes_the_whole_batch():
    boxes = _canvas((30, 30, 50, 50), (10, 10, 40, 40), (0, 0, 0, 0))  # the last one is empty
    out = _match(TALL, boxes, "pad")
    assert out.shape == boxes.shape
    assert _bounds(out[0]) == (30, 20, 49, 59)
    assert _bounds(out[1]) == (10, 0, 39, 59)  # 30 x 60; grown upwards by sliding back inside the canvas
    assert float(out[2].sum()) == 0.0  # an empty mask is returned as it is


def test_aspect_match_pairs_target_and_box_by_batch_index():
    targets = torch.cat([TALL, WIDE])
    boxes = _canvas((30, 30, 50, 50), (30, 30, 50, 50))
    out = _match(targets, boxes, "pad")
    assert _bounds(out[0]) == (30, 20, 49, 59)  # tall target -> 20 x 40
    assert _bounds(out[1]) == (20, 30, 59, 49)  # wide target -> 40 x 20
    assert _match(TALL, boxes, "pad").shape == boxes.shape  # a single target serves the whole batch


def test_aspect_match_accepts_masks_without_batch_dimension():
    out = _match(TALL[0], _canvas((30, 30, 50, 50))[0], "pad")
    assert out.shape == (1, 100, 100)
    assert _bounds(out[0]) == (30, 20, 49, 59)


# --- 10. core/utils_mask.py (FitDiT garment masks) ---

def test_remove_small_clears_only_the_pixels_of_small_regions():
    mask = np.zeros((60, 60), np.uint8)
    mask[11:50, 11:50] = 255  # large region, 39 x 39
    mask[8, 8:20] = 255  # thin L whose bounding box (rows/cols 8..19) overlaps the corner of the square
    mask[8:20, 8] = 255
    cleaned = utils_mask.remove_small(mask, min_area=100)
    assert np.array_equal(cleaned[11:50, 11:50], mask[11:50, 11:50])  # nothing cut out of the large region
    assert int((cleaned > 0).sum()) == 39 * 39
    assert not cleaned[8, 8:20].any() and not cleaned[8:20, 8].any()


POSE_W, POSE_H = 200, 300


def _pose(points):
    """pose_data [134, 2] like the VTON node builds it: DWPose marks an undetected keypoint -1 and the node
    scales x by the width and y by the height, so a missing keypoint is (-W, -H).
    Rows: body 0-17, foot1 18-20 (right), foot2 21-23 (left), face 24-91, hands1 92-112, hands2 113-133."""
    pose = np.empty((134, 2), np.float32)
    pose[:] = (-POSE_W, -POSE_H)
    for row, xy in points.items():
        pose[row] = xy
    return pose


def _garment_mask(category, pose):
    parse = Image.new("L", (POSE_W, POSE_H), 0)  # no clothes, no head: only the keypoints decide
    mask, _ = utils_mask.get_mask_location(category, parse, pose, POSE_W, POSE_H, 0, 0, 0, 0)
    return mask


def _rect(mask):
    """(x1, y1, x2, y2), inclusive, of the white pixels."""
    rows, cols = np.nonzero(np.array(mask))
    return int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())


def test_upper_mask_ignores_keypoints_that_were_not_detected():
    # Left shoulder, left hip and every hand/face point are missing (negative).
    pose = _pose({2: (80, 100), 3: (70, 150), 4: (65, 190), 6: (130, 150), 7: (135, 190), 8: (85, 200)})
    # x: 65 - 25 .. 135 + 25, y: 100 - 20 .. 200 + 10. A missing point used to pull the box to the image top/left.
    assert _rect(_garment_mask("Upper-body", pose)) == (40, 80, 160, 210)


def test_upper_mask_uses_only_detected_face_points():
    pose = _pose({2: (80, 100), 5: (120, 100), 8: (85, 200), 11: (115, 200), 29: (100, 60)})  # face point 5 only
    assert _rect(_garment_mask("Upper-body", pose)) == (55, 60, 145, 210)


def test_lower_mask_starts_at_the_detected_hip():
    # Left hip, left knee and the left foot are missing; the right hip is at y = 150.
    pose = _pose({8: (85, 150), 9: (84, 220), 10: (88, 240), 13: (115, 245), 20: (92, 262)})
    # top = 150 - 50 (no garment found), bottom = foot 262 + 10. min(right hip, missing left hip) put it at 0.
    assert _rect(_garment_mask("Lower-body", pose)) == (54, 100, 145, 272)


@pytest.mark.parametrize("left_foot", [(0, 0), (-POSE_W, -POSE_H)])
def test_lower_mask_bottom_uses_the_right_foot_when_the_left_foot_is_missing(left_foot):
    pose = _pose({8: (85, 150), 11: (115, 150), 20: (92, 262), 23: left_foot})  # ankles not detected
    assert _rect(_garment_mask("Lower-body", pose)) == (55, 100, 145, 272)


def test_dresses_mask_ignores_keypoints_that_were_not_detected():
    pose = _pose({
        2: (80, 100), 3: (70, 150), 6: (130, 150),  # right shoulder side; left shoulder and both wrists missing
        8: (85, 170), 9: (84, 220), 10: (88, 250), 11: (115, 170), 12: (116, 220), 13: (118, 252),
        18: (95, 262), 21: (120, 264),
    })
    # x: 70 - 50 .. 130 + 50, y: 100 - 20 .. 264 + 10; the missing left shoulder used to put the top at 0.
    assert _rect(_garment_mask("Dresses", pose)) == (20, 80, 180, 274)


@pytest.mark.parametrize("category", ["Upper-body", "Lower-body", "Dresses"])
def test_mask_without_any_detected_keypoint_falls_back_to_the_whole_image(category):
    mask = _garment_mask(category, _pose({}))
    assert np.array(mask).all()


# --- 11. Universal Slider Builder cannot work, so it is hidden from the search ---

def test_universal_slider_builder_is_deprecated_but_still_loads_old_workflows():
    node_class = utils_nodes.NH_UniversalSliderBuilder
    assert node_class.DEPRECATED is True
    assert utils_nodes.NODE_CLASS_MAPPINGS["NH_UniversalSliderBuilder"] is node_class
    assert utils_nodes.NODE_DISPLAY_NAME_MAPPINGS["NH_UniversalSliderBuilder"] == "Universal Slider Builder (NH)"
    assert list(node_class.INPUT_TYPES()["required"]) == ["config"]
    assert not getattr(utils_nodes.NH_BooleanSwitch, "DEPRECATED", False)
