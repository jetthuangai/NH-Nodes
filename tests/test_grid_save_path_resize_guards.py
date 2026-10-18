"""Regression tests (2026-10-18 review): the Image Grid Composite size guard and the
file-name part of Save Image Path."""

import importlib
import os
import types
from pathlib import Path

import psutil
import pytest
import torch
from PIL import Image

grid = importlib.import_module("NH-Nodes.nodes.image.image_grid_nodes")
save_image_path = importlib.import_module("NH-Nodes.nodes.image.save_image_path")
utils = importlib.import_module("NH-Nodes.core.utils")

GIB = 2**30


# --- Image Grid Composite: the size guard follows the free memory, not Pillow's decompression-bomb limit ---

@pytest.fixture
def free_memory(monkeypatch):
    """Make psutil report the given available RAM. The tests only exercise the decision: nothing big is allocated."""

    def report(nbytes):
        monkeypatch.setattr(psutil, "virtual_memory", lambda: types.SimpleNamespace(available=int(nbytes)))

    return report


def _check(width, height, columns=1, rows=1):
    grid._check_canvas_size(width, height, columns, rows, width // columns, height // rows)


def _composite(images, **overrides):
    params = dict(
        columns=2, rows=2, cell_size_mode="manual", cell_width=8, cell_height=8,
        resize_mode="stretch", gap_x=0, gap_y=0, background_color_hex="#000000",
    )
    params.update(overrides)
    return grid.NH_ImageGridComposite().composite(images, **params)


@pytest.mark.parametrize("pillow_limit", [89_478_485, 1000, None])  # Pillow's default, a low value, "disabled"
@pytest.mark.parametrize(
    "columns,rows,cell",
    [(40, 1, 1024), (4, 4, 4096), (1, 1, 16384)],  # a 40960x1024 filmstrip; 16384x16384 as 4x4 cells and as one cell
)
def test_grid_guard_accepts_big_canvases_whatever_pillow_limit_other_packs_set(
    monkeypatch, free_memory, pillow_limit, columns, rows, cell
):
    # Image.MAX_IMAGE_PIXELS is process-wide and only guards Image.open; an in-memory IMAGE never goes through it.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", pillow_limit)
    free_memory(64 * GIB)

    grid._check_canvas_size(columns * cell, rows * cell, columns, rows, cell, cell)


def test_grid_estimate_is_the_peak_of_the_canvas_to_tensor_conversion():
    # PIL keeps an RGB canvas as 4 bytes per pixel; pil_to_tensor then holds two float32 RGB arrays (astype, / 255.0).
    pil_rgb, float32_rgb = 4, 3 * 4
    peak_per_pixel = pil_rgb + 2 * float32_rgb

    assert grid._estimate_canvas_bytes(1, 1) == peak_per_pixel
    assert grid._estimate_canvas_bytes(40960, 1024) == 40960 * 1024 * peak_per_pixel
    assert grid._estimate_canvas_bytes(16384, 16384) == 7 * GIB


def test_grid_estimate_covers_what_the_node_keeps_alive():
    canvas = Image.new("RGB", (64, 48))
    tensor = utils.pil_to_tensor(canvas)

    held = tensor.numel() * tensor.element_size() + 64 * 48 * 4  # the IMAGE tensor and the PIL canvas
    assert grid._estimate_canvas_bytes(64, 48) >= held


def test_grid_guard_needs_headroom_on_top_of_the_estimate(free_memory):
    width = height = 8192
    need = grid._estimate_canvas_bytes(width, height)

    free_memory(4 * need)
    _check(width, height)

    for free in (need * 1.05, need, need // 2, 0):  # barely enough is not enough: some memory stays free
        free_memory(free)
        with pytest.raises(ValueError, match=r"(?i)too large"):
            _check(width, height)


def test_grid_guard_switches_exactly_at_the_usable_share_of_the_free_memory(free_memory):
    need = grid._estimate_canvas_bytes(4096, 4096)
    threshold = need / grid._USABLE_MEMORY_FRACTION

    free_memory(threshold + 2)
    _check(4096, 4096)

    free_memory(threshold - 2)
    with pytest.raises(ValueError, match=r"(?i)too large"):
        _check(4096, 4096)


def test_grid_guard_error_names_the_grid_and_the_memory(free_memory):
    free_memory(2 * GIB)

    with pytest.raises(ValueError) as error:
        grid._check_canvas_size(16384, 16384, 4, 4, 4096, 4096)

    message = str(error.value)
    assert "16384x16384" in message and "4x4 cells of 4096x4096" in message
    assert "7.0 GiB" in message and "2.0 GiB" in message  # what it needs / what is free


def test_grid_node_checks_the_memory_before_it_allocates_the_canvas(monkeypatch, free_memory):
    free_memory(1024)  # 1 KiB free: even this 16x16 canvas (2x2 cells of 8x8) cannot be planned
    created_sizes = []
    real_new = Image.new

    def spy(mode, size, *args, **kwargs):
        created_sizes.append(size)
        return real_new(mode, size, *args, **kwargs)

    monkeypatch.setattr(Image, "new", spy)

    with pytest.raises(ValueError, match=r"(?i)too large"):
        _composite([torch.rand((1, 8, 8, 3))])

    assert (16, 16) not in created_sizes  # Pillow itself makes 8x8 images while converting the input


def test_grid_node_ignores_the_pillow_pixel_limit(monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)  # the first guard refused every canvas over 2 x 1000 px

    image = _composite([torch.rand((1, 8, 8, 3))], cell_width=32, cell_height=32)[0]

    assert image.shape == (1, 64, 64, 3)  # 4096 px


@pytest.mark.parametrize(
    "columns,rows,cell_width,cell_height,shape",
    [
        (3, 1, 16384, 1, (1, 1, 49152, 3)),  # 49152 px wide
        (1, 3, 1, 16384, (1, 49152, 1, 3)),  # 49152 px tall
    ],
)
def test_grid_node_builds_strips_longer_than_the_old_32768_px_cap(columns, rows, cell_width, cell_height, shape):
    image = _composite(
        [torch.rand((1, 2, 2, 3))], columns=columns, rows=rows, cell_width=cell_width, cell_height=cell_height
    )[0]

    assert image.shape == shape


def test_grid_guard_does_nothing_without_psutil(monkeypatch):
    monkeypatch.setattr(grid, "psutil", None)

    _check(16384, 16384)  # no information about the memory: no guard, as before the guard existed


def test_grid_guard_does_nothing_when_psutil_cannot_read_the_memory(monkeypatch):
    def broken():
        raise OSError("memory status is not available")

    monkeypatch.setattr(psutil, "virtual_memory", broken)

    _check(16384, 16384)


def test_grid_guard_refuses_a_side_pillow_cannot_create(free_memory):
    free_memory(1024 * GIB)
    ceiling = grid._PIL_MAX_SIDE  # Image.new raises a bare MemoryError for a width above INT_MAX // 4 - 1

    _check(ceiling, 1)
    for width, height in ((ceiling + 1, 1), (1, ceiling + 1)):
        with pytest.raises(ValueError, match=r"(?i)too large.*Pillow"):
            _check(width, height)


# --- Save Image Path: the file-name part of filename_prefix is checked too ---

@pytest.fixture
def written(monkeypatch):
    """Record the files Save Image Path would write instead of writing them, so that a prefix that escapes the
    target folder cannot leave a stray file on the real disk."""
    paths = []
    monkeypatch.setattr(save_image_path, "_save_pil_image", lambda image, path, file_format, dpi: paths.append(path))
    return paths


@pytest.fixture
def as_windows(monkeypatch):
    """Judge prefixes by the Windows rules even where the tests run on another OS (a no-op on Windows itself)."""
    monkeypatch.setattr(save_image_path, "_WINDOWS", True, raising=False)


def _save_image_path(tmp_path, prefix, **kwargs):
    settings = dict(folder_path=str(tmp_path / "out"), create_new_folder=False, new_folder_name="")
    settings.update(kwargs)
    return save_image_path.NH_SaveImagePath().save_images(torch.rand((1, 4, 4, 3)), filename_prefix=prefix, **settings)


def _png_files_outside_out(tmp_path):
    return [p for p in tmp_path.rglob("*.png") if p.relative_to(tmp_path).parts[0] != "out"]


# "D:x" and "x:" are paths on another drive, "C:x" is relative to the current folder of drive C: (os.path.join keeps
# it below the target but drops the "C:", so the reported name is wrong and every run overwrites x0001.png), and
# "name:stream" writes an NTFS alternate data stream. A colon cannot be part of a Windows file name at all.
COLON_PREFIXES = [
    "D:x", "C:x", "c:x", "a:b", "x:", ":", "12:30", "name:stream",
    "sub/D:x", "sub\\C:x", "C:sub/name", "D:\\x", "D:/x",
]


@pytest.mark.usefixtures("as_windows")
@pytest.mark.parametrize("number_suffix", [True, False])
@pytest.mark.parametrize("prefix", COLON_PREFIXES)
def test_windows_prefix_with_a_colon_is_refused_before_anything_is_written(tmp_path, written, prefix, number_suffix):
    with pytest.raises(ValueError, match="filename_prefix"):
        _save_image_path(tmp_path, prefix, number_suffix=number_suffix)

    assert written == []
    assert not (tmp_path / "out").exists()  # not even the target folder is created


@pytest.mark.parametrize(
    "prefix",
    ["/x", "\\x", "//server/share/x", "\\\\server\\share\\x", "\\\\?\\C:\\x", "../x", "..\\x", "sub/../../x"],
)
def test_rooted_unc_and_parent_prefixes_are_refused_on_every_os(tmp_path, written, prefix):
    with pytest.raises(ValueError, match="filename_prefix"):
        _save_image_path(tmp_path, prefix)

    assert written == []
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("name", "name0001.png"),
        ("sub/name", "sub/name0001.png"),
        ("sub\\name", "sub\\name0001.png"),
        ("a/b/name", "a/b/name0001.png"),
        ("my..name", "my..name0001.png"),
        ("sub/", "sub/0001.png"),
    ],
)
def test_plain_and_subfolder_prefixes_keep_working(tmp_path, prefix, expected):
    filenames, path, count = _save_image_path(tmp_path, prefix)

    assert (filenames, path, count) == (expected, str(tmp_path / "out"), 1)
    assert (tmp_path / "out" / Path(expected.replace("\\", "/"))).is_file()
    assert _png_files_outside_out(tmp_path) == []


def test_plain_prefix_is_numbered_across_runs(tmp_path):
    first = _save_image_path(tmp_path, "sub/name")[0]
    second = _save_image_path(tmp_path, "sub\\name")[0]

    assert (first, second) == ("sub/name0001.png", "sub\\name0002.png")
    assert (tmp_path / "out" / "sub" / "name0002.png").is_file()


@pytest.mark.skipif(os.name == "nt", reason="a colon is not a valid character in a Windows file name")
def test_colons_stay_allowed_outside_windows(tmp_path):
    filenames, _, _ = _save_image_path(tmp_path, "render 12:30")

    assert filenames == "render 12:300001.png"
    assert (tmp_path / "out" / "render 12:300001.png").is_file()


def test_same_size_rgba_resize_keeps_colour_under_transparent_pixels():
    picker = importlib.import_module("NH-Nodes.nodes.resolution.smart_resolution_picker")
    image = torch.zeros((1, 4, 8, 8))
    image[:, :3] = torch.tensor([0.2, 0.6, 0.9]).view(1, 3, 1, 1)  # colour kept under alpha 0
    for method in ("lanczos", "box", "hamming"):
        resized = picker._resize_bchw(image, 8, 8, method)
        assert torch.allclose(resized, image), method
