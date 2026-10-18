"""Regressions: image tools, Smart Resolution Picker and the Layer Stack extension."""

import contextlib
import faulthandler
import importlib
import inspect
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

image_label = importlib.import_module("NH-Nodes.nodes.image.image_label")
layer_composite = importlib.import_module("NH-Nodes.nodes.image.layer_composite")
resize_unit = importlib.import_module("NH-Nodes.nodes.image.image_resize_unit")
save_image_path = importlib.import_module("NH-Nodes.nodes.image.save_image_path")
picker = importlib.import_module("NH-Nodes.nodes.resolution.smart_resolution_picker")
restore = importlib.import_module("NH-Nodes.nodes.resolution.restore_image_geometry")
resolution_data = importlib.import_module("NH-Nodes.core.resolution_data")

PACK_DIR = Path(__file__).resolve().parent.parent


def _q(value):
    """Float that survives the nodes' uint8 truncation as exactly ``value`` / 255."""
    return (value + 0.5) / 255.0


def _rgba(height, width, rgba):
    return torch.tensor(rgba, dtype=torch.float32).expand(1, height, width, 4).clone()


def _px(image, y, x):
    return [int(round(float(v) * 255)) for v in image[0, y, x]]


def _close(actual, expected, tolerance=1):
    return all(abs(a - b) <= tolerance for a, b in zip(actual, expected, strict=True))


@contextlib.contextmanager
def _abort_if_frozen(seconds):
    """A big-integer power holds the GIL, so a regression would hang the run: exit with a traceback instead."""
    faulthandler.dump_traceback_later(seconds, exit=True, file=sys.__stderr__)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


# --- 1. Padding alpha comes from the colour's alpha, not from its blue value ---

@pytest.mark.parametrize(
    "color,alpha",
    [("#000000", 1.0), ("#FFFFFF", 1.0), ("#336699", 1.0), ("#336699FF", 1.0), ("#33669900", 0.0), ("#33669980", 128 / 255)],
)
def test_resize_unit_pad_alpha_follows_the_hex_alpha(color, alpha):
    out, width, height = resize_unit.NH_ImageResizeByUnit().resize(
        _rgba(4, 4, (1.0, 1.0, 1.0, 1.0)), "pixel", 8, 4, 300, "pad", "nearest-exact", color
    )
    assert (width, height) == (8, 4) and out.shape == (1, 4, 8, 4)
    rgb = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    padding = out[0, :, 0]
    assert torch.allclose(padding[:, :3], torch.tensor(rgb).expand(4, 3), atol=1e-6)
    assert torch.allclose(padding[:, 3], torch.full((4,), alpha), atol=1e-6)
    assert torch.all(out[0, :, 2:6] == 1.0)  # the image itself is untouched


def test_resize_unit_pad_keeps_rgb_results():
    image = torch.full((1, 4, 4, 3), 0.25)
    out, _, _ = resize_unit.NH_ImageResizeByUnit().resize(image, "pixel", 8, 4, 300, "pad", "nearest-exact", "#102030")
    assert out.shape == (1, 4, 8, 3)
    expected = torch.tensor([16 / 255.0, 32 / 255.0, 48 / 255.0]).expand(4, 2, 3)
    assert torch.equal(out[0, :, :2], expected) and torch.equal(out[0, :, 6:], expected)
    assert torch.equal(out[0, :, 2:6], image[0])


def test_pad_samples_alpha_is_opaque_unless_given():
    samples = torch.zeros((1, 4, 2, 2))
    opaque = resize_unit._pad_samples(samples, 4, 4, (0, 0, 0))
    assert opaque[0, :, 0, 0].tolist() == [0.0, 0.0, 0.0, 1.0]
    translucent = resize_unit._pad_samples(samples, 4, 4, (10, 20, 30, 64))
    assert translucent[0, 3, 0, 0].item() == pytest.approx(64 / 255)
    assert translucent[0, 0, 0, 0].item() == pytest.approx(10 / 255)


def test_pad_color_named_css_colours_still_work():
    out, _, _ = resize_unit.NH_ImageResizeByUnit().resize(
        torch.ones((1, 4, 4, 3)), "pixel", 8, 4, 300, "pad", "nearest-exact", "red"
    )
    assert out[0, 0, 0].tolist() == [1.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "color,alpha",
    [("#000000", 1.0), ("#112233", 1.0), ("#11223300", 0.0), ("#11223380", 128 / 255)],
)
def test_picker_contain_pad_alpha_follows_the_hex_alpha(color, alpha):
    out, data = picker._resize_image_batch_with_data(_rgba(4, 8, (1.0, 1.0, 1.0, 1.0)), 8, 8, "contain_pad", "bilinear", color)
    assert data["pad_top"] == 2 and out.shape == (1, 8, 8, 4)
    padding = out[0, 0]
    rgb = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    assert torch.allclose(padding[:, :3], torch.tensor(rgb).expand(8, 3), atol=1e-6)
    assert torch.allclose(padding[:, 3], torch.full((8,), alpha), atol=1e-6)
    assert torch.all(out[0, 2:6] == 1.0)


def test_picker_contain_pad_keeps_rgb_results_and_invalid_hex_fallback():
    image = torch.full((1, 4, 8, 3), 0.5)
    out, _ = picker._resize_image_batch_with_data(image, 8, 8, "contain_pad", "bilinear", "#abcdef")
    assert out[0, 0, 0].tolist() == pytest.approx([0xAB / 255, 0xCD / 255, 0xEF / 255])
    fallback, _ = picker._resize_image_batch_with_data(image, 8, 8, "contain_pad", "bilinear", "not a colour")
    assert fallback[0, 0, 0].tolist() == [1.0, 1.0, 1.0]  # default pad colour stays white


# --- 2. RGBA layers are pasted with their own alpha ---

def _alpha_layer(size=4, opaque_rows=2):
    """Top rows opaque red, the rest fully transparent green."""
    layer = torch.zeros((1, size, size, 4))
    layer[:, :opaque_rows] = torch.tensor([_q(255), _q(0), _q(0), _q(255)])
    layer[:, opaque_rows:] = torch.tensor([_q(0), _q(255), _q(0), _q(0)])
    return layer


def test_stack_composite_rgba_layer_keeps_transparent_areas():
    background = torch.full((1, 8, 8, 3), _q(100))
    out, used, _ = layer_composite.NH_LayerStackComposite().composite(background, 1, "2,2", layer_1=_alpha_layer())
    assert used == 1 and out.shape == (1, 8, 8, 3)  # RGB background -> RGB output, as before
    assert _px(out, 2, 2) == [255, 0, 0]
    assert _px(out, 3, 5) == [255, 0, 0]
    assert _px(out, 4, 2) == [100, 100, 100]  # transparent part shows the background
    assert _px(out, 5, 5) == [100, 100, 100]
    assert _px(out, 0, 0) == [100, 100, 100]


def test_stack_composite_blends_semi_transparent_layers():
    background = torch.full((1, 4, 4, 3), _q(100))
    layer = _rgba(2, 2, (_q(255), _q(0), _q(0), _q(128)))
    out, _, _ = layer_composite.NH_LayerStackComposite().composite(background, 1, "0,0", layer_1=layer)
    assert _close(_px(out, 0, 0), [178, 50, 50])  # 128/255 red over gray 100


def test_stack_composite_rgb_layers_are_unchanged():
    background = torch.full((1, 6, 6, 3), _q(100))
    layer = torch.full((1, 2, 3, 3), _q(200))
    out, used, info = layer_composite.NH_LayerStackComposite().composite(background, 1, "1,2", layer_1=layer)
    assert used == 1 and "layer_1: x=1, y=2, w=3, h=2" in info
    assert _px(out, 2, 1) == [200, 200, 200] and _px(out, 3, 3) == [200, 200, 200]
    assert _px(out, 1, 1) == [100, 100, 100] and _px(out, 2, 4) == [100, 100, 100]


def test_stack_composite_rgba_background_gives_rgba_output():
    background = _rgba(8, 8, (_q(0), _q(0), _q(0), _q(0)))  # fully transparent canvas
    layer = _rgba(2, 2, (_q(255), _q(0), _q(0), _q(128)))
    out, _, _ = layer_composite.NH_LayerStackComposite().composite(background, 1, "3,3", layer_1=layer)
    assert out.shape == (1, 8, 8, 4)
    assert _close(_px(out, 3, 3), [255, 0, 0, 128])  # straight alpha over transparent keeps the colour
    assert _px(out, 0, 0)[3] == 0


LAYOUT_ARGS = dict(
    canvas_width=8, canvas_height=8, background_color="#336699", layout_mode="preset", layout_preset="1_full",
    resize_mode="stretch", columns=1, rows=1, margin_x=0, margin_y=0, gap_x=0, gap_y=0, custom_units="auto",
    custom_layout="", background_fit="stretch", background_image=None,
)


def _layout(images, **overrides):
    return layer_composite.NH_LayerLayoutComposite().composite(images, **{**LAYOUT_ARGS, **overrides})


def test_layout_composite_rgba_layer_keeps_transparent_areas():
    out, slots, used, _ = _layout(_alpha_layer(size=8, opaque_rows=4))
    assert (slots, used) == (1, 1) and out.shape == (1, 8, 8, 3)
    assert _px(out, 0, 0) == [255, 0, 0] and _px(out, 3, 7) == [255, 0, 0]
    assert _px(out, 4, 0) == [0x33, 0x66, 0x99]  # background colour shows through
    assert _px(out, 7, 7) == [0x33, 0x66, 0x99]


def test_layout_composite_transparent_background_colour_gives_rgba_output():
    out, _, _, _ = _layout(_alpha_layer(size=8, opaque_rows=4), background_color="#33669900")
    assert out.shape == (1, 8, 8, 4)
    assert _px(out, 0, 0) == [255, 0, 0, 255]
    assert _px(out, 7, 7)[3] == 0


def test_layout_composite_opaque_background_colour_keeps_rgb_output():
    for color in ("#336699", "#336699FF"):
        out, _, _, _ = _layout(torch.full((1, 8, 8, 3), _q(10)), background_color=color)
        assert out.shape == (1, 8, 8, 3)


def test_layout_composite_background_image_alpha_is_respected():
    background = torch.zeros((1, 8, 8, 4))
    background[:, :, :4] = torch.tensor([_q(255), _q(255), _q(255), _q(255)])  # opaque white left half
    out, _, used, _ = _layout(torch.zeros((0, 8, 8, 3)), background_image=background)
    assert used == 0 and out.shape == (1, 8, 8, 3)
    assert _px(out, 0, 0) == [255, 255, 255]
    assert _px(out, 0, 7) == [0x33, 0x66, 0x99]  # transparent right half shows the colour


def test_layout_composite_rgb_layers_are_unchanged():
    layer = torch.full((1, 4, 4, 3), _q(40))
    out, _, used, _ = _layout(layer, layout_preset="1_full", margin_x=2, margin_y=2)
    assert used == 1 and out.shape == (1, 8, 8, 3)
    assert _px(out, 2, 2) == [40, 40, 40] and _px(out, 5, 5) == [40, 40, 40]
    assert _px(out, 1, 1) == [0x33, 0x66, 0x99] and _px(out, 6, 6) == [0x33, 0x66, 0x99]


# --- 3. Save Image Path stays inside its target folder ---

def _save(tmp_path, prefix, images=None, **kwargs):
    settings = dict(folder_path=str(tmp_path / "out"), create_new_folder=False, new_folder_name="")
    settings.update(kwargs)
    return save_image_path.NH_SaveImagePath().save_images(
        torch.rand((1, 4, 4, 3)) if images is None else images, filename_prefix=prefix, **settings
    )


def _no_png_outside(tmp_path):
    assert [p for p in tmp_path.rglob("*.png") if "out" not in p.relative_to(tmp_path).parts[:1]] == []


ESCAPING_PREFIXES = ["../x", "..\\..\\x", "sub/../../x", "sub\\..\\..\\x", "../", "a/b/../../../x"]


@pytest.mark.parametrize("prefix", ESCAPING_PREFIXES)
def test_filename_prefix_cannot_escape_the_target_folder(tmp_path, prefix):
    with pytest.raises(ValueError, match="filename_prefix"):
        _save(tmp_path, prefix)
    _no_png_outside(tmp_path)


def test_absolute_filename_prefix_is_refused(tmp_path):
    with pytest.raises(ValueError, match="filename_prefix"):
        _save(tmp_path, str(tmp_path / "elsewhere" / "x"))
    _no_png_outside(tmp_path)


@pytest.mark.skipif(os.name != "nt", reason="drive letters and UNC shares only exist on Windows")
@pytest.mark.parametrize("prefix", ["C:\\escape\\x", "\\\\server\\share\\x"])
def test_windows_drive_and_unc_prefixes_are_refused(tmp_path, prefix):
    with pytest.raises(ValueError, match="filename_prefix"):
        _save(tmp_path, prefix)


@pytest.mark.parametrize("name", ["../x", "..\\..\\x", "ok/../../x"])
def test_new_folder_name_cannot_escape_the_target_folder(tmp_path, name):
    with pytest.raises(ValueError, match="new_folder_name"):
        _save(tmp_path, "img", create_new_folder=True, new_folder_name=name)
    assert not (tmp_path / "x").exists()
    _no_png_outside(tmp_path)


def test_absolute_new_folder_name_is_refused(tmp_path):
    with pytest.raises(ValueError, match="new_folder_name"):
        _save(tmp_path, "img", create_new_folder=True, new_folder_name=str(tmp_path / "elsewhere"))
    assert not (tmp_path / "elsewhere").exists()


def test_subfolders_inside_the_target_stay_allowed_and_are_numbered(tmp_path):
    filenames, path, count = _save(tmp_path, "renders/portrait")
    assert (filenames, path, count) == ("renders/portrait0001.png", str(tmp_path / "out"), 1)
    assert (tmp_path / "out" / "renders" / "portrait0001.png").is_file()
    filenames, _, _ = _save(tmp_path, "renders/portrait")
    assert filenames == "renders/portrait0002.png"  # numbering sees the files already in the subfolder
    assert (tmp_path / "out" / "renders" / "portrait0002.png").is_file()


def test_new_subfolder_inside_the_target_stays_allowed(tmp_path):
    _save(tmp_path, "img", create_new_folder=True, new_folder_name="batch/day1")
    assert (tmp_path / "out" / "batch" / "day1" / "img0001.png").is_file()


def test_absolute_folder_path_is_still_allowed(tmp_path):
    target = tmp_path / "somewhere" / "else"
    _, path, _ = _save(tmp_path, "img", folder_path=str(target))
    assert path == str(target) and (target / "img0001.png").is_file()


def test_dots_inside_a_name_are_not_an_escape(tmp_path):
    filenames, _, _ = _save(tmp_path, "my..name")
    assert filenames == "my..name0001.png" and (tmp_path / "out" / "my..name0001.png").is_file()


# --- 4. Position expressions use the shared safe evaluator ---

def test_position_expressions_keep_their_values():
    positions = layer_composite._parse_layer_positions(
        "(bg_w - w1) // 2, BG_H / 2 - h1 / 2   # centred\nx1 + w1 + 10, y1\n\n2 ** 3, -(a - b) % 7",
        3, background_width=100, background_height=60, layer_sizes=[(20, 10), (5, 5), (1, 1)],
    )
    assert positions == [(40, 25), (70, 25), (8, 2)]


def test_position_positions_fill_missing_lines_with_zero():
    assert layer_composite._parse_layer_positions("5,6", 3) == [(5, 6), (0, 0), (0, 0)]


@pytest.mark.parametrize("expression", ["9**9**9", "10**10**6", "2**2**20", "(9**9)**(9**9)"])
def test_position_expression_power_bombs_fail_fast(expression):
    started = time.perf_counter()
    with _abort_if_frozen(60):
        with pytest.raises(ValueError):
            layer_composite._parse_layer_positions(f"{expression}, 0", 1)
    assert time.perf_counter() - started < 5


@pytest.mark.parametrize(
    "expression", ["1 / 0", "5 // (a - a)", "unknown_name", "1 +", "min(1, 2)", "a.real", "'x' * 3", "(-8) ** 0.5", "[1][0]"]
)
def test_bad_position_expressions_raise_value_error(expression):
    with pytest.raises(ValueError):
        layer_composite._parse_layer_positions(f"{expression}, 0", 1)


def test_position_expression_error_names_the_expression():
    with pytest.raises(ValueError, match="1 / 0"):
        layer_composite._eval_position_expr("1 / 0", {})


def test_module_no_longer_uses_the_removed_ast_num():
    assert "ast.Num" not in inspect.getsource(layer_composite)


# --- 5. Image Label: cached fonts and a binary-searched fit ---

def _linear_fit(start, fits, floor=8):
    """The pre-fix loop, verbatim: shrink one point at a time until the text fits."""
    size = start
    while not fits(size) and size > floor:
        size -= 1
    return size


@pytest.mark.parametrize("start", [6, 7, 8, 9, 10, 33, 64, 200, 512])
@pytest.mark.parametrize("per_point", [0.4, 1.0, 3.3, 7.01])
def test_fit_font_size_equals_the_linear_search_for_growing_widths(start, per_point):
    for box in list(range(1, 400, 7)) + [10_000, 100_000]:
        def fits(size):
            return math.ceil(per_point * size) <= box
        assert image_label._fit_font_size(start, fits) == _linear_fit(start, fits), (start, per_point, box)


def test_fit_font_size_needs_only_a_logarithmic_number_of_probes():
    probes = []

    def fits(size):
        probes.append(size)
        return size <= 100

    assert image_label._fit_font_size(512, fits) == 100
    assert len(probes) <= 12  # the linear search needed 412


@pytest.fixture
def scalable_font(monkeypatch):
    """Route every font request to one embedded scalable font, count loads and start with a cold cache."""
    probe = ImageFont.load_default(10)
    font_bytes = getattr(probe, "font_bytes", None)
    if not isinstance(probe, ImageFont.FreeTypeFont) or not font_bytes:
        pytest.skip("Pillow was built without FreeType")
    real_truetype = ImageFont.truetype
    loads = []

    def fake_truetype(path, size=10, *args, **kwargs):
        loads.append((str(path), size))
        return real_truetype(io.BytesIO(font_bytes), size)

    cache = getattr(image_label, "_cached_truetype", None)
    if cache is not None:
        cache.cache_clear()
    monkeypatch.setattr(image_label.ImageFont, "truetype", fake_truetype)
    yield loads
    if cache is not None:
        cache.cache_clear()


LABEL_TEXTS = [
    "NH Label",
    "Hello, World! 0123456789",
    "A long label that cannot fit at the requested size",
    "Dòng tiếng Việt: Cộng hòa xã hội chủ nghĩa",
    "First line\nA much longer second line of text",
    "The quick brown fox jumps over the lazy dog " * 3,
]


@pytest.mark.parametrize("stroke", [0, 3])
def test_fit_font_size_matches_the_linear_search_with_a_real_font(scalable_font, stroke):
    draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    checked = 0
    for text in LABEL_TEXTS:
        widths = {}
        for size in range(6, 301):
            box = draw.multiline_textbbox((0, 0), text, font=image_label._load_font("default", size), stroke_width=stroke)
            widths[size] = box[2] - box[0]
        for start in (6, 8, 9, 24, 96, 300):
            # Text width is not strictly increasing for every font (hinting makes it dip by a pixel or two now and
            # then); the search is exact whenever it is, and otherwise still returns a size that fits.
            increasing = all(widths[s] <= widths[s + 1] for s in range(8, start))
            for box_width in range(1, 2200, 41):
                def fits(size, box_width=box_width):
                    return widths[size] <= box_width
                expected = _linear_fit(start, fits)
                actual = image_label._fit_font_size(start, fits)
                checked += 1
                if increasing:
                    assert actual == expected, (text[:20], start, box_width)
                else:
                    assert actual <= expected and (actual == 8 or actual == start or fits(actual))
    assert checked > 1000


def test_label_node_fits_text_like_the_old_linear_search_and_loads_each_font_once(scalable_font):
    image = torch.zeros((3, 40, 600, 3))
    text = "A long label that cannot fit at the requested size"
    settings = dict(
        position="header", frame_height=60, fill_enabled=True, frame_color_hex="#111111", text_color_hex="#FFFFFF",
        font_name="default", overlay_on_image=False, align="left", padding_x=10, outline_width=1,
        outline_color_hex="#000000",
    )
    node = image_label.NH_ImageLabel()
    fitted, = node.add_label(image, text, font_size=120, fit_text=True, **settings)
    loads_when_fitting = len(scalable_font)

    # What the old loop picks, measured with the very same fonts.
    draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))

    def fits(size):
        box = draw.multiline_textbbox((0, 0), text, font=image_label._load_font("default", size), stroke_width=1)
        return box[2] - box[0] <= 600 - 2 * 10

    expected = _linear_fit(120, fits)
    assert 8 < expected < 120  # the text really had to shrink
    reference, = node.add_label(image, text, font_size=expected, fit_text=False, **settings)
    assert torch.equal(fitted, reference)
    # One TrueType load per distinct size (about a dozen), not one per tried size and batch item (hundreds).
    assert loads_when_fitting <= 16, loads_when_fitting


def test_load_font_returns_the_cached_object(scalable_font):
    assert image_label._load_font("default", 24) is image_label._load_font("default", 24)
    assert image_label._load_font("default", 24) is not image_label._load_font("default", 25)
    assert len(scalable_font) == 2


def test_missing_fonts_fall_back_to_the_default_font_and_the_failure_is_remembered(monkeypatch):
    real_truetype = ImageFont.truetype
    attempts = []

    def truetype_without_system_fonts(font, *args, **kwargs):
        if isinstance(font, str):  # a font name or path; the embedded default font arrives as a file object
            attempts.append(font)
            raise OSError("cannot open resource")
        return real_truetype(font, *args, **kwargs)

    image_label._cached_truetype.cache_clear()
    monkeypatch.setattr(image_label.ImageFont, "truetype", truetype_without_system_fonts)
    try:
        for name in ("no-such-font.ttf", "default", "default"):
            font = image_label._load_font(name, 20)
            assert font.getbbox("Hello")[2] > 0
        assert len(attempts) == 1  # arial.ttf is tried once per size, not once per label
    finally:
        monkeypatch.undo()
        image_label._cached_truetype.cache_clear()


# --- 6. Remove Resize Padding must not stretch cropped results ---

def _resize_data(mode, source=(150, 100), target=(76, 52)):
    image = torch.rand((1, source[1], source[0], 3))
    resized, data = picker._resize_image_batch_with_data(image, target[0], target[1], mode, "bilinear", "#000000")
    return image, resized, data


def test_remove_padding_content_region_leaves_cover_crop_results_unchanged():
    # A cover_crop result has no padding, so its content region is the whole image.
    _, cropped, data = _resize_data("cover_crop")
    assert data["crop_left"] + data["crop_right"] > 0  # 150x100 -> 76x52 really crops
    out, width, height, info = picker.NH_RemoveResizePadding().remove_padding(cropped, data, "content_region", "lanczos")
    assert torch.equal(out, cropped) and out.shape == cropped.shape
    assert (width, height) == (76, 52)
    assert "no padding detected" not in info
    assert "cover_crop" in info and "cannot be recovered" in info and "NH Restore Image Geometry" in info


def _coordinate_image(width, height):
    """x and y of every pixel centre (0-1) in the first two channels, a constant in the third.

    A bilinear resize reproduces a linear ramp exactly, so what is left over after a round trip is misplacement.
    """
    x = ((torch.arange(width) + 0.5) / width).expand(height, width)
    y = ((torch.arange(height) + 0.5) / height).unsqueeze(1).expand(height, width)
    return torch.stack((x, y, torch.full_like(x, 0.25)), dim=-1).unsqueeze(0)


def _restore_footprint(original, processed, data):
    """(left, top, right, bottom) of the source pixels that NH Restore Image Geometry takes from the processed image."""
    _, mask = restore.NH_RestoreImageGeometry().restore(processed, original, data, "bilinear")
    columns = torch.nonzero(mask[0].any(dim=0)).flatten()
    rows = torch.nonzero(mask[0].any(dim=1)).flatten()
    return int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1


def _assert_realigned(out, original, footprint, margin=8):
    """The picture sits where it came from in the original frame; the cropped-away borders repeat its edge pixels."""
    height, width = original.shape[1:3]
    left, top, right, bottom = footprint
    assert left + top + (width - right) + (height - bottom) > 0  # something was cropped away, so something is filled in
    inside = (slice(None), slice(top + margin, bottom - margin), slice(left + margin, right - margin))
    error = (out[inside] - original[inside]).abs()
    # The picture is placed on whole source pixels, so it can be half a pixel off (0.5/width in x, 0.5/height in y);
    # 0.1 more allows for the ripple of the Lanczos kernel on a ramp.
    assert float(error[..., 0].max()) * width <= 0.6, float(error[..., 0].max()) * width
    assert float(error[..., 1].max()) * height <= 0.6, float(error[..., 1].max()) * height
    assert torch.allclose(out[..., 2], torch.full_like(out[..., 2], 0.25), atol=1e-5)  # constant, borders included
    for side, strip, edge in (
        ("left", out[:, :, :left], out[:, :, left:left + 1]),
        ("right", out[:, :, right:], out[:, :, right - 1:right]),
        ("top", out[:, :top], out[:, top:top + 1]),
        ("bottom", out[:, bottom:], out[:, bottom - 1:bottom]),
    ):
        assert torch.equal(strip, edge.expand_as(strip)), side


CROP_CASES = [  # (source, resize target)
    ((3000, 2000), (1216, 832)),  # the review's landscape photo: 16 px cut off each side
    ((1080, 1350), (896, 1152)),  # the review's portrait photo
    ((2000, 3000), (832, 1216)),  # crop at the top and bottom
    ((1024, 683), (1216, 832)),  # the resize enlarged it, so restoring shrinks it
    ((4000, 3000), (1152, 896)),  # odd crop: 21 px left, 22 px right
    ((101, 67), (20, 20)),  # tiny, strongly cropped
]


@pytest.mark.parametrize("source,target", CROP_CASES)
def test_remove_padding_original_size_realigns_cover_crop_results(source, target):
    width, height = source
    original = _coordinate_image(width, height)
    processed, data = picker._resize_image_batch_with_data(original, *target, "cover_crop", "bilinear", "#FFFFFF")
    assert data["crop_left"] + data["crop_top"] + data["crop_right"] + data["crop_bottom"] > 0
    out, out_width, out_height, info = picker.NH_RemoveResizePadding().remove_padding(processed, data, "original_size", "bilinear")
    assert (out_width, out_height) == (width, height) and out.shape == original.shape  # the source frame, not the target
    _assert_realigned(out, original, _restore_footprint(original, processed, data))
    assert "NH Restore Image Geometry" in info and "no padding detected" not in info


@pytest.mark.parametrize("processed_size", [(40, 40), (33, 29)])
def test_remove_padding_original_size_follows_an_enlarged_processed_image(processed_size):
    original = _coordinate_image(101, 67)
    processed, data = picker._resize_image_batch_with_data(original, 20, 20, "cover_crop", "bilinear", "#FFFFFF")
    processed = F.interpolate(processed.movedim(-1, 1), size=processed_size, mode="bilinear", align_corners=False).movedim(1, -1)
    out, out_width, out_height, _ = picker.NH_RemoveResizePadding().remove_padding(processed, data, "original_size", "bilinear")
    assert (out_width, out_height) == (101, 67)
    _assert_realigned(out, original, _restore_footprint(original, processed, data))


@pytest.mark.parametrize("source", [(3000, 2000), (1080, 1350)])
def test_smart_ratio_resize_then_remove_padding_returns_the_original_frame(source):
    """The review's default pipeline: cover_crop is the default mode of NH Smart Ratio Image Resize."""
    width, height = source
    original = _coordinate_image(width, height)
    resized, *_, data = picker.NH_SmartRatioImageResize().resize(original)
    assert data["resize_mode"] == "cover_crop" and data["crop_left"] + data["crop_top"] > 0
    out, out_width, out_height, _ = picker.NH_RemoveResizePadding().remove_padding(resized, data, "original_size")
    assert (out_width, out_height) == (width, height) and out.shape == original.shape
    _assert_realigned(out, original, _restore_footprint(original, resized, data))


@pytest.mark.parametrize("channels,dtype", [(1, torch.float32), (3, torch.float16), (4, torch.float32)])
def test_remove_padding_original_size_keeps_batches_channels_and_dtype(channels, dtype):
    data = picker._build_resize_data(50, 30, 20, 20, "cover_crop", "bilinear", "#000000")
    processed = torch.stack([torch.full((20, 20, channels), 0.25), torch.full((20, 20, channels), 0.75)]).to(dtype)
    out, width, height, _ = picker.NH_RemoveResizePadding().remove_padding(processed, data, "original_size", "bilinear")
    assert out.shape == (2, 30, 50, channels) and out.dtype == dtype and (width, height) == (50, 30)
    for index, value in enumerate((0.25, 0.75)):  # a constant stays constant, borders included
        assert torch.allclose(out[index].float(), torch.full((30, 50, channels), value), atol=2e-3)


@pytest.mark.parametrize("method", picker.RESIZE_METHODS)
def test_remove_padding_original_size_works_with_every_resize_method(method):
    original = _coordinate_image(120, 80)
    processed, data = picker._resize_image_batch_with_data(original, 40, 30, "cover_crop", "bilinear", "#000000")
    assert data["crop_left"] > 0
    out, width, height, _ = picker.NH_RemoveResizePadding().remove_padding(processed, data, "original_size", method)
    assert out.shape == original.shape and (width, height) == (120, 80)
    assert bool(torch.isfinite(out).all()) and float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert torch.allclose(out[..., 2], torch.full_like(out[..., 2], 0.25), atol=1e-4)
    inner = (slice(None), slice(10, 70), slice(25, 95))
    assert float((out[inner][..., 0] - original[inner][..., 0]).abs().max()) <= 4.5 / 120  # within 1.5 processed pixels


@pytest.mark.parametrize("damage", [
    lambda data: {key: value for key, value in data.items() if key != "resized_width"},
    lambda data: {**data, "resized_height": 0},
    lambda data: {**data, "source_width": 0},
])
def test_remove_padding_original_size_with_unusable_metadata_leaves_the_image_alone(damage):
    _, cropped, data = _resize_data("cover_crop")
    out, width, height, info = picker.NH_RemoveResizePadding().remove_padding(cropped, damage(data), "original_size", "lanczos")
    assert out is cropped and (width, height) == (76, 52)
    assert "cover_crop" in info and "NH Restore Image Geometry" in info


def test_remove_padding_original_size_info_says_what_was_filled_in():
    original = _coordinate_image(150, 100)
    processed, data = picker._resize_image_batch_with_data(original, 76, 52, "cover_crop", "bilinear", "#000000")
    *_, info = picker.NH_RemoveResizePadding().remove_padding(processed, data, "original_size", "bilinear")
    assert "cover_crop" in info and "cannot be recovered" in info and "NH Restore Image Geometry" in info
    assert "edge" in info and "150×100" in info and "L1 T0 R1 B0" in info


def test_remove_padding_cover_crop_without_any_crop_still_resizes_back():
    original, resized, data = _resize_data("cover_crop", target=(75, 50))  # same aspect: nothing was cropped
    out, width, height, info = picker.NH_RemoveResizePadding().remove_padding(resized, data, "original_size", "lanczos")
    assert (width, height) == (150, 100) and out.shape == original.shape
    assert "no padding detected" in info


def test_remove_padding_still_removes_contain_pad_padding():
    _, padded, data = _resize_data("contain_pad", target=(100, 100))
    out, width, height, info = picker.NH_RemoveResizePadding().remove_padding(padded, data, "content_region", "lanczos")
    assert (width, height) == (data["resized_width"], data["resized_height"]) and (width, height) == (100, 67)
    assert "removed padding" in info


def test_remove_padding_stretch_is_unchanged():
    original, stretched, data = _resize_data("stretch")
    out, width, height, info = picker.NH_RemoveResizePadding().remove_padding(stretched, data, "original_size", "lanczos")
    assert (width, height) == (150, 100) and out.shape == original.shape
    assert "no padding detected" in info


def test_remove_padding_without_metadata_passes_through():
    image = torch.rand((1, 8, 8, 3))
    out, width, height, info = picker.NH_RemoveResizePadding().remove_padding(image, None)
    assert out is image and (width, height) == (8, 8) and "No resize_data" in info


# --- 7. Smart Resolution Picker ---

def test_stale_preset_label_is_reported_in_info():
    node = picker.NH_SmartResolutionPicker()
    default = node.pick()
    width, height, latent, info, _ = node.pick(preset="1 MP | 1:1 | 1x1  (renamed)")
    assert (width, height) == (default[0], default[1])  # still falls back to the default preset
    assert latent["samples"].shape == default[2]["samples"].shape
    assert "Warning:" in info and "1 MP | 1:1 | 1x1  (renamed)" in info
    assert "⚠️" in info.splitlines()[0]
    assert "renamed" not in default[3]


def test_valid_preset_has_no_stale_preset_warning():
    model = resolution_data.MODEL_LABELS[0]
    for label in resolution_data.PRESET_LABELS_BY_MODEL[model][:5]:
        info = picker.NH_SmartResolutionPicker().pick(model_family=model, preset=label)[3]
        assert "not found" not in info and "unknown" not in info.casefold()


def test_stale_preset_is_not_reported_when_custom_size_is_used():
    info = picker.NH_SmartResolutionPicker().pick(preset="stale", use_custom=True, custom_width=1024, custom_height=768)[3]
    assert "stale" not in info


def test_unknown_model_family_is_reported_in_info():
    default = picker.NH_SmartResolutionPicker().pick()
    width, height, _, info, _ = picker.NH_SmartResolutionPicker().pick(model_family="Model That Was Removed")
    assert (width, height) == (default[0], default[1])
    assert "Model That Was Removed" in info and "Warning:" in info


OVERSIZED_PRESETS = [
    entry for entry in resolution_data.RESOLUTION_PRESETS
    if entry["width"] > resolution_data.MODEL_SPECS[entry["model_id"]]["max_res"][0]
    or entry["height"] > resolution_data.MODEL_SPECS[entry["model_id"]]["max_res"][1]
]


@pytest.mark.parametrize("entry", OVERSIZED_PRESETS, ids=lambda entry: entry["label"])
def test_curated_presets_keep_their_size_and_only_their_own_warning(entry):
    # max_res bounds custom sizes; curated 4 MP presets are deliberate and carry their own reliability note.
    width, height, _, info, _ = picker.NH_SmartResolutionPicker().pick(model_family=entry["model"], preset=entry["label"])
    assert (width, height) == (entry["width"], entry["height"])
    assert "max_res" not in info
    assert (entry.get("warning") or "") in info


def test_custom_sizes_are_still_clamped_to_max_res():
    width, height, _, info, _ = picker.NH_SmartResolutionPicker().pick(
        model_family="FLUX.2 Klein 9B", use_custom=True, custom_width=4096, custom_height=1024,
    )
    assert (width, height) == (2048, 1024)
    assert "clamped to range" in info


# Lanczos keeps float precision --------------------------------------------------------------------------

_PIL_FILTERS = {
    "lanczos": Image.Resampling.LANCZOS,
    "hamming": Image.Resampling.HAMMING,
    "box": Image.Resampling.BOX,
}


def _old_uint8_resize(samples, height, width, method="lanczos"):
    """The pre-fix path: every image goes through uint8, and Pillow rounds and clips after each of its two passes."""
    items = []
    for sample in samples:
        array = np.clip(sample.movedim(0, -1).numpy() * 255.0, 0, 255).astype(np.uint8)
        resized = np.array(Image.fromarray(array).resize((width, height), _PIL_FILTERS[method]))
        items.append(torch.from_numpy(resized.astype(np.float32) / 255.0).movedim(-1, 0))
    return torch.stack(items)


def _natural_like(height=96, width=128, seed=3):
    """Smooth gradients with soft shapes and mild noise, the kind of content Lanczos is used on."""
    generator = torch.Generator().manual_seed(seed)
    y, x = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing="ij")
    image = torch.stack([x, y, (x + y) / 2], dim=-1)
    image = image + 0.2 * torch.sin(9 * x * math.pi)[..., None] * torch.cos(5 * y * math.pi)[..., None]
    image = image + 0.01 * torch.randn((height, width, 3), generator=generator)
    return image.clamp(0, 1).unsqueeze(0).movedim(-1, 1)


@pytest.mark.parametrize("method", ["lanczos", "box"])
def test_lanczos_keeps_values_finer_than_8_bit(method):
    samples = torch.full((1, 3, 32, 32), 0.5013)  # 127.83 / 255: not representable in 8 bits
    out = picker._resize_bchw(samples, 16, 16, method)
    assert out.shape == (1, 3, 16, 16)
    assert torch.allclose(out, torch.full_like(out, 0.5013), atol=1e-5)


def test_lanczos_does_not_collapse_a_smooth_ramp_to_256_levels():
    ramp = torch.linspace(0, 1, 2048).expand(1, 1, 8, 2048).contiguous()
    out = picker._resize_bchw(ramp, 8, 1024, "lanczos")
    assert out.shape == (1, 1, 8, 1024)
    assert torch.unique(out).numel() > 600  # a uint8 round trip leaves at most 256


@pytest.mark.parametrize("size", [(48, 64), (60, 160), (192, 256)])
def test_lanczos_matches_the_old_uint8_result_visually(size):
    samples = _natural_like()
    height, width = size
    new = picker._resize_bchw(samples, height, width, "lanczos")
    old = _old_uint8_resize(samples, height, width)
    assert new.shape == old.shape == (1, 3, height, width)
    difference = (new - old).abs()
    # The uint8 path truncates, so it sits half a level (0.5/255) below the float result on average: no more.
    assert float(difference.mean()) < 0.75 / 255
    assert float(difference.max()) < 4.0 / 255


def _text_font():
    try:
        return ImageFont.load_default(24)  # scalable, needs Pillow >= 10.1 built with FreeType
    except (TypeError, ImportError, OSError):
        return ImageFont.load_default()


def _hard_edged(kind, width=512, height=256):
    """[1, 3, H, W] white canvas with marks drawn without anti-aliasing, so every edge is a full 0 -> 1 step."""
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    if kind == "checkerboard":
        for top in range(0, height, 5):
            for left in range(0, width, 7):
                if (left // 7 + top // 5) % 2 == 0:
                    draw.rectangle((left, top, left + 6, top + 4), fill=(0, 0, 0))
    elif kind == "lines":
        for start in range(0, width, 7):
            draw.line((start, 20, start + 30, height - 16), fill=(200, 30, 30))
        for row in range(10, height, 9):
            draw.line((0, row, width - 1, row), fill=(0, 0, 0))
    else:  # text, monochrome glyphs
        draw.fontmode = "1"
        font = _text_font()
        for row in range(5):
            draw.text((8, 6 + row * 48), "Line art & text 0123 NH-Nodes", fill=(0, 0, 0), font=font)
    return torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0).movedim(-1, 0).unsqueeze(0)


HARD_EDGE_SIZES = [(128, 256), (512, 1024), (100, 200)]  # 512x256 down 2x, up 2x and down 2.56x


@pytest.mark.parametrize("method", ["lanczos", "hamming", "box"])
@pytest.mark.parametrize("size", HARD_EDGE_SIZES, ids=["down-2x", "up-2x", "down-2.56x"])
@pytest.mark.parametrize("kind", ["checkerboard", "lines", "text"])
def test_hard_edges_stay_close_to_the_8_bit_path(kind, size, method):
    samples = _hard_edged(kind)
    height, width = size
    new = picker._resize_bchw(samples, height, width, method)
    old = _old_uint8_resize(samples, height, width, method)
    assert new.shape == old.shape == (1, 3, height, width)
    # Lanczos overshoots beside a 0 -> 1 step. Pillow's 8-bit path clips after the width pass; float planes that are
    # not clipped there carry the overshoot into the height pass and were up to 15-33/255 off on text and line art.
    assert float((new - old).abs().max()) <= 2.0 / 255


@pytest.mark.parametrize("method", ["lanczos", "hamming", "box"])
@pytest.mark.parametrize("kind", ["checkerboard", "lines", "text"])
def test_resize_clips_between_the_width_and_the_height_pass(kind, method):
    samples = _hard_edged(kind)
    both = picker._resize_bchw(samples, 100, 200, method)
    # _resize_bchw clips every result, so doing the axes one after the other is the clipped two-pass reference.
    width_first = picker._resize_bchw(samples, samples.shape[2], 200, method)
    assert torch.equal(both, picker._resize_bchw(width_first, 100, 200, method))
    assert float(both.min()) >= 0.0 and float(both.max()) <= 1.0


@pytest.mark.parametrize("size", HARD_EDGE_SIZES, ids=["down-2x", "up-2x", "down-2.56x"])
def test_hard_edged_rgba_stays_close_to_pillows_premultiplied_8_bit_path(size):
    height, width = size
    colour = _hard_edged("lines")
    half_opaque = torch.zeros((1, 1, 256, 512))
    half_opaque[..., :256] = 1.0
    checkered = (_hard_edged("checkerboard")[:, :1] < 0.5).float()  # opaque dark cells on a transparent canvas

    def pillow(samples):  # Pillow premultiplies RGBA itself and clips after each pass in 8 bits
        as_uint8 = np.clip(samples[0].movedim(0, -1).numpy() * 255.0, 0, 255).astype(np.uint8)
        resized = np.asarray(Image.fromarray(as_uint8).resize((width, height), Image.Resampling.LANCZOS))
        return torch.from_numpy(resized.astype(np.float32) / 255.0).movedim(-1, 0).unsqueeze(0)

    # Hard colour edges inside an opaque area: premultiplying changes nothing there, so it must match the 8-bit path.
    samples = torch.cat((colour, half_opaque), dim=1)
    new, reference = picker._resize_bchw(samples, height, width, "lanczos"), pillow(samples)
    assert new.shape == reference.shape and float(new.min()) >= 0.0 and float(new.max()) <= 1.0
    opaque_side = slice(0, width // 2 - 12)  # clear of the filter footprint of the alpha edge
    assert float((new[..., opaque_side] - reference[..., opaque_side]).abs().max()) <= 2.0 / 255

    # Hard alpha edges in both directions: alpha is clipped between the passes like any other plane.
    samples = torch.cat((colour, checkered), dim=1)
    new, reference = picker._resize_bchw(samples, height, width, "lanczos"), pillow(samples)
    assert float((new[:, 3] - reference[:, 3]).abs().max()) <= 2.0 / 255
    assert float(new.min()) >= 0.0 and float(new.max()) <= 1.0


def test_lanczos_output_stays_within_0_1():
    samples = torch.zeros((1, 3, 16, 16))
    samples[:, :, :, 8:] = 1.0  # hard edge: Lanczos rings above 1 and below 0
    out = picker._resize_bchw(samples, 8, 40, "lanczos")
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert float(out.max()) == 1.0 and float(out.min()) == 0.0


def test_lanczos_clips_out_of_range_input_like_before():
    samples = torch.full((1, 3, 8, 8), 7.0)
    samples[:, 1] = -3.0
    out = picker._resize_bchw(samples, 4, 4, "lanczos")
    assert torch.equal(out[:, 0], torch.ones(1, 4, 4)) and torch.equal(out[:, 1], torch.zeros(1, 4, 4))


def test_lanczos_resizes_rgba_with_premultiplied_alpha():
    samples = torch.zeros((1, 4, 8, 16))
    samples[:, :, :, :8] = 1.0  # opaque white left half, transparent black right half
    out = picker._resize_bchw(samples, 4, 8, "lanczos")
    edge = out[0, :, 2, :]  # row through the middle
    partial = (edge[3] > 0.02) & (edge[3] < 0.98)
    assert bool(partial.any())
    # Colour must not bleed in from the invisible half: white wherever there is any alpha.
    assert torch.allclose(edge[:3, partial], torch.ones(3, int(partial.sum())), atol=0.02)
    assert torch.all(edge[3, :3] > 0.98) and torch.all(edge[3, -3:] < 0.02)


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_lanczos_handles_every_channel_count_batches_and_dtypes(channels):
    samples = torch.rand((2, channels, 20, 30), dtype=torch.float16)
    out = picker._resize_bchw(samples, 10, 15, "lanczos")
    assert out.shape == (2, channels, 10, 15) and out.dtype == torch.float16
    single = picker._resize_bchw(samples[1:], 10, 15, "lanczos")
    assert torch.equal(out[1:], single)  # batch items are independent


def test_resize_image_batch_uses_the_float_lanczos_path():
    image = torch.full((1, 32, 32, 3), 0.5013)
    out, _ = picker._resize_image_batch_with_data(image, 16, 16, "stretch", "lanczos", "#000000")
    assert torch.allclose(out, torch.full_like(out, 0.5013), atol=1e-5)


# --- 8. Layer Stack extension: links must follow reordered inputs ---

JS_IMPORT = 'import { app } from "../../../scripts/app.js";'

JS_HARNESS = r"""
import { pathToFileURL } from "node:url";

const extensionPath = process.argv[2];
const registered = [];
globalThis.__app = { graph: null, registerExtension(extension) { registered.push(extension); } };
await import(pathToFileURL(extensionPath).href);
const NodeType = function () {};
await registered[0].beforeRegisterNodeDef(NodeType, { name: "NH_LayerStackComposite" });

// model "object": links in a plain object, input.link holds the link id (litegraph.js, old frontends)
// model "map":    links in a Map, input.link holds the link id
// model "store":  current frontends: link store is the source of truth; input -> link is derived from the input's
//                 *current index* and a link can only move onto a free slot
function makeGraph(model) {
  const graph = { model, dirty: 0, writes: 0, errors: [], all: new Map(), setDirtyCanvas() { this.dirty++; } };
  graph.links = model === "object" ? {} : graph.all;
  if (model === "store") graph.getLink = (id) => graph.all.get(id) ?? null;
  graph.addLink = (id, originId, targetId, slot) => {
    let target = slot;
    const link = {
      id, origin_id: originId, origin_slot: 0, target_id: targetId,
      get target_slot() { return target; },
      set target_slot(value) {
        graph.writes++;
        if (model === "store") {
          for (const other of graph.all.values()) {
            if (other !== link && other.target_id === targetId && other.target_slot === value) {
              graph.errors.push(`link ${id} -> slot ${value} is occupied by link ${other.id}`);
              return;
            }
          }
        }
        target = value;
      },
    };
    graph.all.set(id, link);
    if (model === "object") graph.links[id] = link;
    return link;
  };
  return graph;
}

function makeNode(model, graph, layerCount) {
  const node = {
    id: 7, graph, inputs: [],
    widgets: [{ name: "layer_count", value: layerCount }],
    addWidget(type, name, value, callback) { const widget = { type, name, value, callback }; this.widgets.push(widget); return widget; },
    addInput(name, type) { const slot = makeSlot(name, type); this.inputs.push(slot); return slot; },
    removeInput(index) {
      this.inputs.splice(index, 1);
      for (const link of graph.all.values()) {
        if (link.target_id === this.id && link.target_slot > index) link.target_slot = link.target_slot - 1;
      }
    },
    computeSize() { return [200, 100]; },
    setSize() {},
    setDirtyCanvas() { graph.dirty++; },
  };
  function makeSlot(name, type, linkId = null) {
    const slot = { name, type };
    if (model === "store") {
      Object.defineProperty(slot, "link", {
        get() {
          const link = node.getInputLink(node.inputs.indexOf(slot));
          return link ? link.id : null;
        },
      });
    } else {
      slot.link = linkId;
    }
    return slot;
  }
  node.makeSlot = makeSlot;
  if (model === "store") {
    node.getInputLink = (index) => {
      for (const link of graph.all.values()) {
        if (link.target_id === node.id && link.target_slot === index) return link;
      }
      return null;
    };
  }
  return node;
}

async function scenario(model, layerCount, layout) {
  // layout: [name, linkId|null] in the order the frontend would have them.
  const graph = makeGraph(model);
  globalThis.__app.graph = graph;
  const node = makeNode(model, graph, layerCount);
  layout.forEach(([name, linkId], index) => {
    node.inputs.push(node.makeSlot(name, name.startsWith("layer_") || name === "background_image" ? "IMAGE" : "INT", linkId));
    if (linkId !== null) graph.addLink(linkId, 100 + linkId, node.id, index);
  });
  graph.writes = 0;
  NodeType.prototype.onConfigure.call(node);
  await new Promise((resolve) => setTimeout(resolve, 40));
  const linkTargets = {};
  for (const link of graph.all.values()) linkTargets[link.id] = node.inputs[link.target_slot]?.name ?? null;
  const slotLinks = node.inputs.map((slot) => (model === "store" ? null : slot.link));
  return { names: node.inputs.map((slot) => slot.name), linkTargets, slotLinks, errors: graph.errors, writes: graph.writes, dirty: graph.dirty };
}

const result = {};
for (const model of ["object", "map", "store"]) {
  result[model] = {
    // user raised layer_count after connecting layers: new inputs are appended, then everything is sorted
    grow: await scenario(model, 3, [["background_image", 10], ["layer_1", 11], ["layer_count", null], ["positions", null], ["layer_2", 12]]),
    // widget inputs that are linked swap places with a linked layer input
    swap: await scenario(model, 1, [["background_image", 10], ["layer_count", 13], ["positions", null], ["layer_1", 11]]),
    sorted: await scenario(model, 1, [["background_image", 10], ["layer_1", 11], ["layer_count", null], ["positions", null]]),
  };
}
console.log("RESULT:" + JSON.stringify(result));
"""


@pytest.fixture(scope="module")
def layer_stack_js_result(tmp_path_factory):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    source = (PACK_DIR / "web" / "nh_layer_stack_composite.js").read_text(encoding="utf-8")
    assert JS_IMPORT in source, "the extension no longer imports app.js the way this test expects"
    work = tmp_path_factory.mktemp("layer_stack_js")
    (work / "extension.mjs").write_text(source.replace(JS_IMPORT, "const app = globalThis.__app;"), encoding="utf-8")
    shutil.copy(PACK_DIR / "web" / "nh_widget_utils.js", work / "nh_widget_utils.js")  # the extension imports it
    (work / "harness.mjs").write_text(JS_HARNESS, encoding="utf-8")
    run = subprocess.run(
        [node, str(work / "harness.mjs"), str(work / "extension.mjs")],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert run.returncode == 0, run.stderr
    line = next(line for line in run.stdout.splitlines() if line.startswith("RESULT:"))
    return json.loads(line[len("RESULT:"):])


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_layer_stack_js_is_valid_syntax():
    check = subprocess.run(
        [shutil.which("node"), "--check", str(PACK_DIR / "web" / "nh_layer_stack_composite.js")],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert check.returncode == 0, check.stderr


@pytest.mark.parametrize("model", ["object", "map", "store"])
def test_layer_stack_js_links_follow_the_sorted_inputs(layer_stack_js_result, model):
    grown = layer_stack_js_result[model]["grow"]
    assert grown["names"] == ["background_image", "layer_1", "layer_2", "layer_3", "layer_count", "positions"]
    assert grown["linkTargets"] == {"10": "background_image", "11": "layer_1", "12": "layer_2"}
    assert grown["errors"] == [] and grown["dirty"] > 0
    if model != "store":
        assert grown["slotLinks"][:3] == [10, 11, 12]  # input.link ids still belong to their inputs


@pytest.mark.parametrize("model", ["object", "map", "store"])
def test_layer_stack_js_linked_inputs_can_swap_places(layer_stack_js_result, model):
    swapped = layer_stack_js_result[model]["swap"]
    assert swapped["names"] == ["background_image", "layer_1", "layer_count", "positions"]
    assert swapped["linkTargets"] == {"10": "background_image", "11": "layer_1", "13": "layer_count"}
    assert swapped["errors"] == []


@pytest.mark.parametrize("model", ["object", "map", "store"])
def test_layer_stack_js_does_not_touch_links_when_already_sorted(layer_stack_js_result, model):
    done = layer_stack_js_result[model]["sorted"]
    assert done["names"] == ["background_image", "layer_1", "layer_count", "positions"]
    assert done["linkTargets"] == {"10": "background_image", "11": "layer_1"}
    assert done["writes"] == 0
