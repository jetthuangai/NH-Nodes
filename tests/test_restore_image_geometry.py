"""Coordinate and compositing contracts for NH Restore Image Geometry."""

import importlib
import os
import sys

import pytest
import torch
import torch.nn.functional as F

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PACK_DIR))
for path in (COMFY_ROOT, os.path.dirname(PACK_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

resize = importlib.import_module("NH-Nodes.nodes.resolution.smart_resolution_picker")
restore = importlib.import_module("NH-Nodes.nodes.resolution.restore_image_geometry")


def coordinate_image(width, height):
    x = ((torch.arange(width) + 0.5) / width).expand(height, width)
    y = ((torch.arange(height) + 0.5) / height).unsqueeze(1).expand(height, width)
    return torch.stack((x, y, torch.full_like(x, 0.25)), dim=-1).unsqueeze(0)


@pytest.mark.parametrize("width,height", [(101, 67), (67, 101), (111, 67)])
@pytest.mark.parametrize("processed_size", [(20, 20), (40, 40), (33, 29)])
def test_crop_round_trip_preserves_coordinates_after_upscale(width, height, processed_size):
    original = coordinate_image(width, height)
    processed, data = resize._resize_image_batch_with_data(original, 20, 20, "cover_crop", "bilinear", "#FFFFFF")
    processed = F.interpolate(processed.movedim(-1, 1), size=processed_size, mode="bilinear", align_corners=False).movedim(1, -1)
    result, mask = restore.NH_RestoreImageGeometry().restore(processed, original, data, "bilinear")

    # Known source-space footprints, independent of the implementation's sampling grid.
    expected_mask = torch.zeros((1, height, width))
    if (width, height) == (101, 67):
        expected_mask[:, :, 17:84] = 1
    elif (width, height) == (67, 101):
        expected_mask[:, 17:84, :] = 1
    else:
        expected_mask[:, :, 20:87] = 1  # intermediate 33px; asymmetric 6/7px crop
    assert torch.equal(mask, expected_mask)
    interior = F.avg_pool2d(mask.unsqueeze(1), 11, stride=1, padding=5).squeeze(1) == 1
    assert interior.any()
    torch.testing.assert_close(result[interior], original[interior], atol=2e-6, rtol=0)
    assert torch.equal(result[mask == 0], original[mask == 0])
    assert result.shape == original.shape and mask.dtype == torch.float32


@pytest.mark.parametrize("mode", ["cover_crop", "contain_pad", "stretch"])
def test_processed_edit_is_restored_and_mask_blends_with_original(mode):
    original = torch.full((1, 67, 101, 3), 0.2)
    _, data = resize._resize_image_batch_with_data(original, 20, 20, mode, "bilinear", "#000000")
    processed = torch.full((1, 33, 37, 3), 0.8)
    result, mask = restore.NH_RestoreImageGeometry().restore(processed, original, data, "bilinear")
    torch.testing.assert_close(result[mask == 1], torch.full_like(result[mask == 1], 0.8))
    assert torch.equal(result[mask == 0], original[mask == 0])
    composed = result * mask.unsqueeze(-1) + original * (1 - mask.unsqueeze(-1))
    assert torch.equal(composed, result)
    if mode != "cover_crop":
        assert torch.all(mask == 1)


@pytest.mark.parametrize("mode", ["contain_pad", "stretch"])
def test_non_crop_round_trip_has_no_translation_or_padding_color(mode):
    original = coordinate_image(101, 67)
    processed, data = resize._resize_image_batch_with_data(original, 40, 40, mode, "bilinear", "#FF00FF")
    result, mask = restore.NH_RestoreImageGeometry().restore(processed, original, data, "bilinear")
    assert torch.all(mask == 1)
    torch.testing.assert_close(result[:, 4:-4, 4:-4], original[:, 4:-4, 4:-4], atol=2e-6, rtol=0)
    # The constant third channel must never pick up the magenta padding.
    torch.testing.assert_close(result[..., 2], torch.full_like(result[..., 2], 0.25))


@pytest.mark.parametrize("batches", [(1, 3), (3, 1), (3, 3)])
def test_batches_preserve_each_original_outside_crop(batches):
    original_batch, processed_batch = batches
    original = torch.arange(original_batch, dtype=torch.float32).view(-1, 1, 1, 1).expand(-1, 12, 20, 3) / 5
    data = resize._build_resize_data(20, 12, 8, 8, "cover_crop", "bilinear", "#000000")
    processed = torch.full((processed_batch, 16, 16, 3), 0.9)
    result, mask = restore.NH_RestoreImageGeometry().restore(processed, original, data, "bilinear")
    expected_original = original.expand(3, -1, -1, -1)
    assert result.shape == (3, 12, 20, 3) and mask.shape == (3, 12, 20)
    assert torch.equal(result[mask == 0], expected_original[mask == 0])
    torch.testing.assert_close(result[mask == 1], torch.full_like(result[mask == 1], 0.9))


@pytest.mark.parametrize("method", ["nearest", "bilinear", "bicubic"])
def test_no_crop_identity_and_inputs_are_not_mutated(method):
    original = torch.rand(1, 7, 11, 3)
    processed = torch.rand_like(original)
    before_original, before_processed = original.clone(), processed.clone()
    data = resize._build_resize_data(11, 7, 11, 7, "stretch", method, "#000000")
    result, mask = restore.NH_RestoreImageGeometry().restore(processed, original, data, method)
    torch.testing.assert_close(result, processed, atol=2e-6, rtol=0)
    assert torch.all(mask == 1)
    assert torch.equal(original, before_original) and torch.equal(processed, before_processed)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_cpu_dtype_is_preserved(dtype):
    original = torch.full((1, 7, 11, 3), 0.125, dtype=dtype)
    processed = torch.full((1, 14, 22, 3), 0.75, dtype=dtype)
    data = resize._build_resize_data(11, 7, 11, 7, "stretch", "bilinear", "#000000")
    result, mask = restore.NH_RestoreImageGeometry().restore(processed, original, data, "bilinear")
    assert result.dtype == dtype and result.device == original.device
    assert torch.all(result == 0.75) and torch.all(mask == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_and_mixed_input_devices():
    original = torch.zeros((1, 67, 101, 3), device="cuda", dtype=torch.float16)
    data = resize._build_resize_data(101, 67, 20, 20, "cover_crop", "bilinear", "#000000")
    result, mask = restore.NH_RestoreImageGeometry().restore(torch.ones((1, 40, 40, 3)), original, data)
    assert result.device == original.device and mask.device == original.device
    assert result.dtype == original.dtype
    assert torch.all(result[mask == 0] == 0) and torch.all(result[mask == 1] == 1)


@pytest.mark.parametrize("change,error", [
    (lambda data: None, "version-1"),
    (lambda data: {**data, "version": 2}, "version-1"),
    (lambda data: {key: value for key, value in data.items() if key != "crop_top"}, "crop_top"),
    (lambda data: {**data, "resized_width": 0}, "positive"),
    (lambda data: {**data, "source_width": 99}, "source size"),
    (lambda data: {**data, "resize_mode": "unknown"}, "resize_mode"),
])
def test_invalid_metadata_cannot_silently_misalign_original(change, error):
    data = resize._build_resize_data(11, 7, 8, 8, "cover_crop", "bilinear", "#000000")
    with pytest.raises(ValueError, match=error):
        restore.NH_RestoreImageGeometry().restore(torch.zeros(1, 8, 8, 3), torch.zeros(1, 7, 11, 3), change(data))


def test_incompatible_batches_and_channels_are_rejected():
    data = resize._build_resize_data(11, 7, 8, 8, "cover_crop", "bilinear", "#000000")
    node = restore.NH_RestoreImageGeometry()
    with pytest.raises(ValueError, match="batches"):
        node.restore(torch.zeros(2, 8, 8, 3), torch.zeros(3, 7, 11, 3), data)
    with pytest.raises(ValueError, match="channels"):
        node.restore(torch.zeros(1, 8, 8, 4), torch.zeros(1, 7, 11, 3), data)


@pytest.mark.parametrize("portrait", [False, True])
def test_mask_includes_pixel_center_exactly_on_left_or_top_crop_boundary(portrait):
    width, height, target_width, target_height = (41, 123, 2, 3) if portrait else (123, 41, 3, 2)
    original = torch.zeros((1, height, width, 3))
    data = resize._build_resize_data(width, height, target_width, target_height, "cover_crop", "bilinear", "#000000")
    result, mask = restore.NH_RestoreImageGeometry().restore(torch.ones(1, target_height, target_width, 3), original, data)
    expected = torch.zeros_like(mask)
    if portrait:
        expected[:, 20:82, :] = 1
    else:
        expected[:, :, 20:82] = 1
    assert torch.equal(mask, expected)
    torch.testing.assert_close(result, expected.unsqueeze(-1).expand_as(result))
