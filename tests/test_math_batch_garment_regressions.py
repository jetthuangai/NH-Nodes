"""Regression tests for Math Eval scalars, Batch Merge padding and VTON garment boxes."""

import importlib

import numpy as np
import pytest
import torch
from PIL import Image

math_nodes = importlib.import_module("NH-Nodes.nodes.logic.math_nodes")
batch_nodes = importlib.import_module("NH-Nodes.nodes.batch.batch_nodes")
utils_mask = importlib.import_module("NH-Nodes.core.utils_mask")


# --- Math Eval accepts the scalar types upstream nodes actually send ---

@pytest.mark.parametrize("value", [np.float32(2), np.int64(2), torch.tensor(2.0), True])
def test_math_eval_accepts_numpy_torch_and_bool_scalars(value):
    result_float, result_int, result_string = math_nodes.NH_MathEval().evaluate("a + b", a=value, b=1.0)
    assert result_float == float(value) + 1.0 and result_int == int(result_float) and result_string == str(result_float)


@pytest.mark.parametrize("expression", ["foo * 2", "__import__('os')", "a < b", "''"])
def test_math_eval_errors_always_name_the_expression(expression):
    with pytest.raises(ValueError) as error:
        math_nodes.NH_MathEval().evaluate(expression, a=1.0, b=2.0)
    assert repr(expression) in str(error.value)


# --- Batch Merge ---

def test_batch_merge_pad_never_rounds_a_side_to_zero():
    batch_a = torch.zeros((1, 64, 64, 3))
    thin = torch.ones((1, 1, 1000, 3))  # scale 0.064 -> 0.064 px tall before the fix
    result, count = batch_nodes.NH_BatchMerge().merge(batch_a, thin, resize_mode="pad")
    assert result.shape == (2, 64, 64, 3) and count == 2


def test_batch_merge_pad_bars_get_the_same_alpha_for_rgb_and_rgba_inputs():
    batch_a = torch.ones((1, 8, 8, 4))
    rgb = torch.ones((1, 4, 8, 3))    # letterboxed: 2 px bars on top and bottom
    rgba = torch.ones((1, 4, 8, 4))
    merged, _ = batch_nodes.NH_BatchMerge().merge(batch_a, rgb, rgba, resize_mode="pad")
    bars_rgb, bars_rgba = merged[1, :2, :, 3], merged[2, :2, :, 3]
    assert torch.equal(bars_rgb, bars_rgba) and float(bars_rgb.max()) == 0.0
    assert float(merged[1, 2:6, :, 3].min()) == 1.0  # the picture itself is opaque


# --- VTON garment boxes when a whole keypoint group is missing ---

W, H = 200, 300


def _pose(points):
    pose = np.empty((134, 2), np.float32)
    pose[:] = (-W, -H)  # DWPose marks undetected keypoints -1, the node scales them by W/H
    for row, xy in points.items():
        pose[row] = xy
    return pose


def _rect(mask):
    rows, cols = np.nonzero(np.array(mask))
    return int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())


def test_lower_mask_uses_the_trousers_when_both_hips_are_missing():
    parse = np.zeros((H, W), np.uint8)
    parse[150:280, 70:130] = 6  # trousers
    pose = _pose({9: (84, 220), 10: (88, 250), 12: (116, 220), 13: (118, 252)})  # knees + ankles only
    mask, _ = utils_mask.get_mask_location("Lower-body", Image.fromarray(parse), pose, W, H, 0, 0, 0, 0)
    x1, y1, x2, y2 = _rect(mask)
    assert 140 <= y1 <= 150  # trousers top minus the 5 px pad; the missing hips used to put it at 0
    assert x1 < 70 and x2 >= 129 and y2 >= 279


def test_upper_mask_with_only_face_points_stays_around_the_shirt():
    parse = np.zeros((H, W), np.uint8)
    parse[90:200, 60:140] = 4  # upper clothes
    pose = _pose({29: (100, 60)})  # one face point, no body or hand keypoints
    mask, _ = utils_mask.get_mask_location("Upper-body", Image.fromarray(parse), pose, W, H, 0, 0, 0, 0)
    x1, y1, x2, y2 = _rect(mask)
    assert (x1, x2) == (35, 165)  # shirt contour 60..140 (exclusive end) plus 25 px padding, not the whole width
    assert y1 == 60 and y2 < H - 1  # from the face line down to the shirt hem, not the whole height


@pytest.mark.parametrize("category", ["Upper-body", "Lower-body", "Dresses"])
def test_no_keypoints_and_no_garment_still_cover_the_whole_image(category):
    parse = Image.fromarray(np.zeros((H, W), np.uint8))
    mask, _ = utils_mask.get_mask_location(category, parse, _pose({}), W, H, 0, 0, 0, 0)
    assert np.array(mask).all()
