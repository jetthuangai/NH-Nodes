"""Regression tests: image core nodes (VTON batches, face paste, agnostic image, mask-aware
resize, large image save, cache manager, grid composite)."""

import importlib
import importlib.util
import os

import comfy.cli_args
import cv2
import numpy as np
import pytest
import torch
from PIL import Image

vton = importlib.import_module("NH-Nodes.nodes.vton.vton_preprocessor_nodes")
face_paste = importlib.import_module("NH-Nodes.nodes.image.face_paste")
agnostic = importlib.import_module("NH-Nodes.nodes.image.agnostic_image")
mask_resize = importlib.import_module("NH-Nodes.nodes.image.mask_resize_image")
large_save = importlib.import_module("NH-Nodes.nodes.image.large_image_save")
cache_manager = importlib.import_module("NH-Nodes.nodes.image.large_preview_cache_manager")
grid = importlib.import_module("NH-Nodes.nodes.image.image_grid_nodes")

RED, GREEN, BLUE = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
YELLOW, WHITE, CYAN = (1.0, 1.0, 0.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)


def _solid(color, h=32, w=32):
    """One image [H,W,C] filled with a colour."""
    return torch.tensor(color, dtype=torch.float32).expand(h, w, len(color)).clone()


def _boxes(boxes, h=32, w=32):
    """MASK batch [B,H,W]; each box is (x1, y1, x2, y2) with an exclusive end."""
    masks = torch.zeros((len(boxes), h, w))
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        masks[index, y1:y2, x1:x2] = 1.0
    return masks


def _bbox(condition):
    """(top, bottom, left, right) of the True pixels of a 2-D bool tensor."""
    rows, cols = torch.nonzero(condition, as_tuple=True)
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


# --- VTON Ultimate Processor: batches, RGB conversion, no-person error ---

_MODEL_FILES = (
    "dwpose/yolox_l.onnx", "dwpose/dw-ll_ucoco_384.onnx",
    "humanparsing/parsing_atr.onnx", "humanparsing/parsing_lip.onnx",
)


def _vton_ready():
    if importlib.util.find_spec("onnxruntime") is None or importlib.util.find_spec("skimage") is None:
        return False
    return all(os.path.isfile(os.path.join(vton.MODELS_DIR, name)) for name in _MODEL_FILES)


class _FakePose:
    """DWposeDetector stand-in. Like the real one it never returns an empty array for a person-less
    image: it marks every keypoint -1. A completely black image counts as "no person"."""

    def __init__(self):
        self.seen = []
        self.empty_when_no_person = False

    def __call__(self, bgr):
        self.seen.append(bgr)
        height, width = bgr.shape[:2]
        pose_image = np.zeros((height, width, 3), np.uint8)
        if bgr.max() == 0:
            candidate = np.zeros((0, 134, 2), np.float32) if self.empty_when_no_person else np.full((1, 134, 2), -1.0, np.float32)
        else:
            candidate = np.full((1, 134, 2), 0.5, np.float32)
        return pose_image, None, None, candidate


class _FakeParsing:
    """Parsing stand-in: the left part of the image becomes upper clothes (label 5), wider when brighter."""

    def __call__(self, image):
        pixels = np.array(image)
        labels = np.zeros(pixels.shape[:2], np.uint8)
        labels[:, : max(1, int(pixels.shape[1] * pixels.mean() / 255.0))] = 5
        return Image.fromarray(labels), Image.fromarray(labels)


def _fake_get_mask_location(category, parsing_map, keypoints, width, height, *offsets):
    return Image.fromarray(((np.array(parsing_map) == 5) * 255).astype(np.uint8)), None


@pytest.fixture
def fake_vton(monkeypatch):
    pytest.importorskip("skimage")  # core.utils_mask imports it at module level
    utils_mask = importlib.import_module("NH-Nodes.core.utils_mask")
    pose = _FakePose()
    monkeypatch.setattr(utils_mask, "get_mask_location", _fake_get_mask_location)
    monkeypatch.setattr(vton, "LOADED_VTON_MODELS", {"dwprocessor": pose, "parsing_model": _FakeParsing(), "device": "cpu"})
    return pose


def _process(images, category="Upper-body"):
    return vton.NH_VTonUltimateProcessor().process(images, category, 4, False, True, True, "cpu")


def _flat_batch(levels, channels=3, h=48, w=64):
    return torch.stack([torch.full((h, w, channels), level) for level in levels])


def test_vton_processes_every_batch_image_independently(fake_vton):
    batch = _flat_batch([0.2, 0.5, 0.8])
    outputs = _process(batch)

    assert len(outputs) == 8
    assert all(output.shape[0] == 3 for output in outputs)
    for index in range(3):
        single = _process(batch[index:index + 1])
        for got, want in zip(outputs, single):
            assert torch.equal(got[index:index + 1], want)
    # The items really differ, so the comparison above is not trivially true.
    assert not torch.equal(outputs[0][0], outputs[0][2])


@pytest.mark.parametrize("channels", [1, 4])
def test_vton_converts_grayscale_and_rgba_input_to_rgb(fake_vton, channels):
    outputs = _process(_flat_batch([0.5], channels=channels))

    assert fake_vton.seen[0].ndim == 3 and fake_vton.seen[0].shape[2] == 3  # what DWPose receives
    assert outputs[4].shape == (1, 48, 64, 3)  # masked_image


@pytest.mark.parametrize("empty_candidate", [False, True])
def test_vton_reports_batch_index_when_no_person_is_found(fake_vton, empty_candidate):
    fake_vton.empty_when_no_person = empty_candidate
    batch = _flat_batch([0.5, 0.0, 0.5])  # the middle image is black: nothing to detect

    with pytest.raises(ValueError, match="batch index 1"):
        _process(batch)


@pytest.mark.skipif(not _vton_ready(), reason="VTON models or onnxruntime not available")
def test_vton_batch_of_two_matches_single_results_on_cpu(monkeypatch):
    from skimage import data

    monkeypatch.setattr(vton, "LOADED_VTON_MODELS", None)
    astronaut = torch.from_numpy(data.astronaut().astype(np.float32) / 255.0)
    batch = torch.stack([astronaut, astronaut.flip(1)])  # the second image is mirrored left-right
    node = vton.NH_VTonUltimateProcessor()

    outputs = node.process(batch, "Upper-body", 4, False, True, True, "cpu")

    assert len(outputs) == 8
    assert all(outputs[i].shape == (2, 512, 512) for i in (0, 1, 5, 6, 7))
    assert all(outputs[i].shape == (2, 512, 512, 3) for i in (2, 3, 4))
    for index in range(2):
        single = node.process(batch[index:index + 1], "Upper-body", 4, False, True, True, "cpu")
        for got, want in zip(outputs, single):
            torch.testing.assert_close(got[index:index + 1], want)


@pytest.mark.skipif(not _vton_ready(), reason="VTON models or onnxruntime not available")
def test_vton_real_models_report_batch_index_for_person_less_image(monkeypatch):
    from skimage import data

    monkeypatch.setattr(vton, "LOADED_VTON_MODELS", None)
    astronaut = torch.from_numpy(data.astronaut().astype(np.float32) / 255.0)
    blank = torch.full_like(astronaut, 0.5)

    with pytest.raises(ValueError, match="batch index 1"):
        vton.NH_VTonUltimateProcessor().process(torch.stack([astronaut, blank]), "Upper-body", 4, False, True, True, "cpu")


# --- NH Simple Face Paste ---

def _paste(dest, dest_mask, source, source_mask, feathering=0):
    return face_paste.NH_SimpleFacePaste().paste_face(dest, dest_mask, source, source_mask, feathering)[0]


def test_face_paste_processes_every_item_and_broadcasts_single_inputs():
    dest = torch.stack([_solid(RED), _solid(GREEN)])

    out = _paste(dest, _boxes([(8, 8, 24, 24)]), torch.stack([_solid(BLUE)]), _boxes([(0, 0, 16, 16)]))

    assert out.shape == (2, 32, 32, 3)
    for index, color in enumerate((RED, GREEN)):
        assert torch.equal(out[index, 16, 16], torch.tensor(BLUE))  # pasted
        assert torch.equal(out[index, 2, 2], torch.tensor(color))  # destination untouched


def test_face_paste_pairs_batch_items_by_index():
    dest = torch.stack([_solid(RED), _solid(GREEN)])
    source = torch.stack([_solid(BLUE), _solid(YELLOW)])

    out = _paste(dest, _boxes([(0, 0, 8, 8), (24, 24, 32, 32)]), source, _boxes([(0, 0, 16, 16)]))

    assert torch.equal(out[0, 4, 4], torch.tensor(BLUE)) and torch.equal(out[0, 28, 28], torch.tensor(RED))
    assert torch.equal(out[1, 28, 28], torch.tensor(YELLOW)) and torch.equal(out[1, 4, 4], torch.tensor(GREEN))


def test_face_paste_clips_overshoot_instead_of_wrapping():
    # Resampled images can leave [0, 1]; uint8 conversion must saturate, not wrap around to black.
    dest = torch.stack([_solid((1.2, 1.2, 1.2))])
    source = torch.stack([_solid((-0.2, 1.2, 1.0))])

    out = _paste(dest, _boxes([(8, 8, 24, 24)]), source, _boxes([(0, 0, 16, 16)]))

    assert torch.equal(out[0, 16, 16], torch.tensor([0.0, 1.0, 1.0]))
    assert torch.equal(out[0, 2, 2], torch.tensor(WHITE))


def test_face_paste_leaves_items_without_a_mask_untouched():
    dest = torch.rand((2, 32, 32, 3))
    dest_mask = _boxes([(8, 8, 24, 24), (0, 0, 0, 0)])  # the second mask is empty

    out = _paste(dest, dest_mask, torch.stack([_solid(BLUE)]), _boxes([(0, 0, 16, 16)]))

    assert out.shape == dest.shape
    assert torch.equal(out[0, 16, 16], torch.tensor(BLUE))
    assert torch.equal(out[1], dest[1])  # exact, not re-quantised through uint8


def test_face_paste_accepts_2d_masks():
    out = _paste(
        torch.stack([_solid(RED)]), _boxes([(8, 8, 24, 24)])[0],
        torch.stack([_solid(BLUE)]), _boxes([(0, 0, 16, 16)])[0],
    )
    assert torch.equal(out[0, 16, 16], torch.tensor(BLUE))


# --- NH Agnostic Image Generator ---

def _generate(image, mask, fill_mode="gray", blur_radius=65, noise_strength=1.0, gray_value=0.5, feathering=0):
    return agnostic.NH_AgnosticImageGenerator().generate(
        image, mask, fill_mode, blur_radius, noise_strength, gray_value, feathering,
    )


def test_agnostic_resizes_a_mask_of_a_different_size():
    image = torch.rand((1, 16, 16, 3))
    mask = torch.zeros((1, 8, 8))
    mask[:, 2:6, 2:6] = 1.0

    agnostic_img, masked_img, composite = _generate(image, mask)

    assert agnostic_img.shape == masked_img.shape == composite.shape == (1, 16, 16, 3)
    assert torch.allclose(agnostic_img[0, 8, 8], torch.full((3,), 0.5))  # inside the (rescaled) mask
    assert torch.equal(agnostic_img[0, 0, 0], image[0, 0, 0])  # outside


def test_agnostic_treats_a_2d_mask_as_a_batch_of_one():
    image = torch.rand((1, 16, 16, 3))
    mask = torch.zeros((16, 16))
    mask[4:12, 4:12] = 1.0

    agnostic_img = _generate(image, mask)[0]

    assert agnostic_img.shape == (1, 16, 16, 3)
    assert torch.allclose(agnostic_img[0, 8, 8], torch.full((3,), 0.5))
    assert torch.equal(agnostic_img[0, 0, 0], image[0, 0, 0])


def test_agnostic_single_mask_broadcasts_over_the_image_batch():
    image = torch.rand((3, 16, 16, 3))
    mask = torch.zeros((1, 16, 16))
    mask[:, 4:12, 4:12] = 1.0

    agnostic_img = _generate(image, mask)[0]

    assert agnostic_img.shape == (3, 16, 16, 3)
    assert torch.allclose(agnostic_img[:, 8, 8], torch.full((3, 3), 0.5))


def test_agnostic_binary_mask_result_is_unchanged():
    image = torch.rand((1, 16, 16, 3))
    mask = torch.zeros((1, 16, 16))
    mask[:, 4:12, 4:12] = 1.0

    agnostic_img, masked_img, _ = _generate(image, mask)

    expected = image.clone()
    expected[:, 4:12, 4:12] = 0.5
    assert torch.equal(agnostic_img, expected)
    expected[:, 4:12, 4:12] = 1.0
    assert torch.equal(masked_img, expected)


def test_agnostic_keeps_soft_mask_values():
    image = torch.rand((1, 16, 16, 3))
    mask = torch.full((1, 16, 16), 0.25)  # a soft mask must blend, not be thresholded away

    agnostic_img = _generate(image, mask)[0]

    assert torch.allclose(agnostic_img, image * 0.75 + 0.5 * 0.25, atol=1e-6)


def test_agnostic_feathering_starts_from_the_soft_mask():
    image = torch.rand((1, 16, 16, 3))
    mask = torch.full((1, 16, 16), 0.5)

    agnostic_img = _generate(image, mask, feathering=3)[0]

    assert torch.allclose(agnostic_img, image * 0.5 + 0.5 * 0.5, atol=1e-5)


def test_agnostic_preview_outline_follows_the_half_level_of_a_soft_mask():
    image = torch.zeros((1, 64, 64, 3))
    box = np.zeros((64, 64), np.float32)
    box[20:44, 20:44] = 1.0
    soft = torch.from_numpy(cv2.GaussianBlur(box, (31, 31), 5)).unsqueeze(0)

    composite = _generate(image, soft)[2]

    assert float(composite[0, 32, 8].sum()) == 0.0  # faint fringe (alpha ~0.01) is not outlined
    assert float(composite[0, 32, 20, 1]) > 0.5  # the box edge (alpha 0.5) is


def test_agnostic_noise_is_deterministic_per_batch_index():
    image = torch.rand((2, 8, 8, 3))
    mask = torch.ones((2, 8, 8))

    first = _generate(image, mask, "noise")[0]
    second = _generate(image, mask, "noise")[0]

    assert torch.equal(first, second)
    assert not torch.equal(first[0], first[1])
    for index in range(2):
        expected = np.random.default_rng(index).random((8, 8, 3), dtype=np.float32)
        assert torch.allclose(first[index], torch.from_numpy(expected))


# --- NH Mask-Aware Resize ---

def _marker_image(h=40, w=80):
    """Blue background, red/green end stripes (12.5% wide each) and a white 8x8 marker in the middle."""
    image = torch.zeros((h, w, 3))
    image[..., 2] = 1.0
    image[:, :10] = torch.tensor(RED)
    image[:, 70:] = torch.tensor(GREEN)
    image[16:24, 36:44] = 1.0
    return image.unsqueeze(0)


def _is_white(image):
    return (image[..., 0] > 0.5) & (image[..., 1] > 0.5) & (image[..., 2] > 0.5)


def test_mask_resize_empty_mask_pad_fits_the_whole_image():
    image, mask = _marker_image(), torch.zeros((1, 40, 80))

    out_img, out_mask = mask_resize.NH_MaskResizeImage().resize_by_mask(image, mask, 64, 64, "pad", "#00FFFF")

    assert out_img.shape == (1, 64, 64, 3) and out_mask.shape == (1, 64, 64)
    top, bottom, left, right = _bbox((out_img[0] != torch.tensor(CYAN)).any(-1))
    assert (bottom - top + 1, right - left + 1) == (32, 64)  # 80x40 contained in 64x64, padded top/bottom
    m_top, m_bottom, m_left, m_right = _bbox(_is_white(out_img[0]))
    assert abs((m_bottom - m_top) - (m_right - m_left)) <= 1  # the square marker stays square


def test_mask_resize_empty_mask_crop_covers_the_target_with_a_centred_crop():
    image, mask = _marker_image(), torch.zeros((1, 40, 80))

    out_img, out_mask = mask_resize.NH_MaskResizeImage().resize_by_mask(image, mask, 64, 64, "crop", "#00FFFF")

    assert out_img.shape == (1, 64, 64, 3) and out_mask.shape == (1, 64, 64)
    red = (out_img[0, ..., 0] > 0.5) & (out_img[0, ..., 1] < 0.5) & (out_img[0, ..., 2] < 0.5)
    green = (out_img[0, ..., 1] > 0.5) & (out_img[0, ..., 0] < 0.5) & (out_img[0, ..., 2] < 0.5)
    assert not red.any() and not green.any()  # the end stripes are cropped away, not squeezed in
    assert not (out_img[0] == torch.tensor(CYAN)).all(-1).any()  # no padding: the image covers the target
    m_top, m_bottom, m_left, m_right = _bbox(_is_white(out_img[0]))
    assert abs((m_bottom - m_top) - (m_right - m_left)) <= 1


def test_mask_resize_output_is_clipped_to_the_unit_range():
    image = torch.zeros((1, 64, 64, 3))
    image[:, :, 32:] = 1.0  # a hard edge makes Lanczos ring below 0 and above 1
    mask = torch.ones((1, 64, 64))

    out_img, _ = mask_resize.NH_MaskResizeImage().resize_by_mask(image, mask, 100, 100, "pad", "#000000")

    assert float(out_img.min()) >= 0.0 and float(out_img.max()) <= 1.0


# --- NH Save Large Image ---

def _save(tmp_path, image, file_format, **kwargs):
    node = large_save.NH_SaveLargeImage()
    node.output_dir = str(tmp_path)
    return node.save(image, "nh_test", file_format, **kwargs)


@pytest.mark.parametrize(
    "file_format,width",
    [("webp", 16384), ("jpg", 65501)],
)
def test_save_large_rejects_oversized_images_before_writing(tmp_path, file_format, width):
    batch = torch.zeros((2, 1, width, 3))  # 1 px tall keeps the test light

    with pytest.raises(ValueError, match=r"(?is)too large.*png.*tiff"):
        _save(tmp_path, batch, file_format)

    assert list(tmp_path.rglob("*")) == []  # not even the output folder was touched


@pytest.mark.parametrize(
    "file_format,width",
    [("webp", 16383), ("jpg", 65500), ("png", 70000), ("tiff", 70000)],
)
def test_save_large_writes_images_at_the_format_limit(tmp_path, file_format, width):
    result = _save(tmp_path, torch.zeros((1, 1, width, 3)), file_format)

    saved_count = result["result"][2]
    assert saved_count == 1
    assert os.path.getsize(result["result"][1]) > 0


@pytest.mark.parametrize("disabled", [False, True])
def test_save_large_png_metadata_follows_disable_metadata(tmp_path, monkeypatch, disabled):
    monkeypatch.setattr(comfy.cli_args.args, "disable_metadata", disabled)

    result = _save(
        tmp_path, torch.zeros((1, 4, 4, 3)), "png",
        prompt={"1": {"class_type": "Example"}}, extra_pnginfo={"workflow": {"nodes": []}},
    )

    with Image.open(result["result"][1]) as saved:
        text = dict(saved.text)
    if disabled:
        assert "prompt" not in text and "workflow" not in text
    else:
        assert "prompt" in text and "workflow" in text


# --- NH Large Preview Cache Manager ---

def test_cache_manager_runs_on_every_queue():
    changed = cache_manager.NH_LargePreviewCacheManager.IS_CHANGED(
        action="info", max_age_hours=24.0, max_total_mb=2048.0, cache_id="",
    )
    assert changed != changed  # NaN never equals itself -> ComfyUI never serves the cached output


# --- Image Grid Composite ---

def _grid(images, **overrides):
    params = dict(
        columns=2, rows=2, cell_size_mode="manual", cell_width=8, cell_height=8,
        resize_mode="stretch", gap_x=0, gap_y=0, background_color_hex="#000000",
    )
    params.update(overrides)
    return grid.NH_ImageGridComposite().composite(images, **params)


def test_grid_builds_a_normal_canvas():
    image, slot_count, used_count, _info = _grid([torch.rand((1, 8, 8, 3))])

    assert image.shape == (1, 16, 16, 3)
    assert (slot_count, used_count) == (4, 1)


# The canvas size guard (free memory, not Pillow's decompression-bomb limit) is tested in
# test_grid_save_path_resize_guards.py.
