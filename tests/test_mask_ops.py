"""core.mask_ops.mask_bbox and the mask nodes built on it (Mask Properties, Create Box Mask)."""

import importlib

import numpy as np
import pytest
import torch

mask_ops = importlib.import_module("NH-Nodes.core.mask_ops")
properties = importlib.import_module("NH-Nodes.nodes.mask.mask_properties")
box_mask = importlib.import_module("NH-Nodes.nodes.mask.mask_to_bbox")

mask_bbox = mask_ops.mask_bbox


def _mask(h, w, *boxes):
    """float32 [H, W] mask; each box is (x1, y1, x2, y2) with an exclusive end."""
    mask = np.zeros((h, w), np.float32)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = 1.0
    return mask


# --- mask_bbox ---

def test_bbox_is_inclusive_and_ordered_x1_y1_x2_y2():
    assert mask_bbox(_mask(10, 12, (3, 2, 9, 5))) == (3, 2, 8, 4)  # width 6, height 3


def test_bbox_covers_several_blobs():
    assert mask_bbox(_mask(20, 30, (2, 3, 5, 6), (20, 15, 27, 18))) == (2, 3, 26, 17)


def test_bbox_of_an_empty_mask_is_none():
    assert mask_bbox(np.zeros((6, 7), np.float32)) is None


@pytest.mark.parametrize("shape", [(0, 5), (5, 0), (0, 0)])
def test_bbox_of_a_zero_sized_mask_is_none(shape):
    assert mask_bbox(np.zeros(shape, np.float32)) is None


def test_bbox_of_a_single_pixel_and_of_a_full_mask():
    single = np.zeros((9, 11), np.float32)
    single[7, 4] = 1.0
    assert mask_bbox(single) == (4, 7, 4, 7)
    assert mask_bbox(np.ones((9, 11), np.float32)) == (0, 0, 10, 8)


def test_threshold_is_strict_and_defaults_to_half():
    half = np.full((4, 4), 0.5, np.float32)
    assert mask_bbox(half) is None  # exactly 0.5 is outside, as in every NH mask node
    assert mask_bbox(np.full((4, 4), np.nextafter(np.float32(0.5), np.float32(1)), np.float32)) == (0, 0, 3, 3)

    soft = _mask(8, 8, (2, 2, 6, 6)) * 0.4
    assert mask_bbox(soft) is None
    assert mask_bbox(soft, threshold=0.25) == (2, 2, 5, 5)
    assert mask_bbox(soft, threshold=0.0) == (2, 2, 5, 5)  # "any non-zero pixel"


def test_nan_pixels_are_outside():
    mask = _mask(5, 5, (1, 1, 3, 3))
    mask[0, 0] = np.nan
    assert mask_bbox(mask) == (1, 1, 2, 2)


@pytest.mark.parametrize("shape", [(), (7,), (4, 5, 1), (2, 4, 5)])
def test_bbox_rejects_masks_that_are_not_2d(shape):
    with pytest.raises(ValueError, match="2-D"):
        mask_bbox(np.ones(shape, np.float32))


def test_bbox_is_four_python_ints():
    bbox = mask_bbox(_mask(6, 6, (1, 2, 4, 5)))
    assert all(type(value) is int for value in bbox)


@pytest.mark.parametrize("dtype", [np.float16, np.float64, np.uint8, np.bool_])
def test_bbox_accepts_other_array_types(dtype):
    mask = _mask(8, 9, (2, 1, 7, 4)).astype(dtype)
    if dtype is np.uint8:
        mask *= 255
    assert mask_bbox(mask) == (2, 1, 6, 3)


def test_bbox_matches_a_brute_force_search():
    rng = np.random.default_rng(5)
    for _ in range(200):
        h, w = rng.integers(1, 30, 2)
        mask = (rng.random((h, w)) * (rng.random((h, w)) < rng.choice([0.0, 0.03, 0.4]))).astype(np.float32)
        points = [(x, y) for y in range(h) for x in range(w) if mask[y, x] > 0.5]
        expected = None if not points else (
            min(x for x, _ in points), min(y for _, y in points), max(x for x, _ in points), max(y for _, y in points)
        )
        assert mask_bbox(mask) == expected


# --- Mask Properties ---

def _properties(mask):
    return properties.NH_MaskProperties().get_properties(torch.from_numpy(mask))


def test_properties_report_size_corners_and_bbox():
    mask = torch.from_numpy(_mask(10, 12, (3, 2, 9, 5))).unsqueeze(0)

    width, height, x1, y1, x2, y2, bbox = properties.NH_MaskProperties().get_properties(mask)

    assert (width, height, x1, y1, x2, y2) == (6, 3, 3, 2, 8, 4)
    assert bbox == (3, 2, 8, 4)


def test_properties_of_an_empty_mask_are_all_zero():
    assert _properties(np.zeros((1, 6, 7), np.float32)) == (0, 0, 0, 0, 0, 0, (0, 0, 0, 0))


def test_properties_only_look_at_the_first_batch_item():
    batch = torch.from_numpy(np.stack([_mask(8, 8, (1, 1, 3, 3)), _mask(8, 8, (4, 4, 8, 8))]))
    assert _properties(batch.numpy())[:6] == (2, 2, 1, 1, 2, 2)


# --- Create Box Mask ---

def test_box_mask_fills_the_bounding_box_of_every_batch_item():
    first = _mask(10, 10, (2, 2, 4, 3), (6, 5, 8, 8))
    second = np.zeros((10, 10), np.float32)  # empty: stays empty
    third = _mask(10, 10, (0, 0, 1, 1))
    masks = torch.from_numpy(np.stack([first, second, third]))

    (result,) = box_mask.NH_MaskToBBox().create_bbox_mask(masks)

    assert result.shape == masks.shape and result.dtype == masks.dtype
    assert torch.equal(result[0], torch.from_numpy(_mask(10, 10, (2, 2, 8, 8))))
    assert float(result[1].sum()) == 0.0
    assert torch.equal(result[2], torch.from_numpy(_mask(10, 10, (0, 0, 1, 1))))


def test_box_mask_ignores_values_at_or_below_half():
    soft = torch.from_numpy(_mask(6, 6, (1, 1, 5, 5)) * 0.5).unsqueeze(0)
    (result,) = box_mask.NH_MaskToBBox().create_bbox_mask(soft)
    assert float(result.sum()) == 0.0
