"""Channel handling of the nodes that turn IMAGE tensors into PIL pictures through core.utils.tensor_to_pil.

NH Save Large Image and the large preview cache keep an alpha channel where the format can hold it and give RGB
otherwise (grey included); Image Grid Composite always builds RGB cells and drops alpha without compositing.
"""

import importlib
import os

import folder_paths
import numpy as np
import pytest
import torch
from PIL import Image

large_save = importlib.import_module("NH-Nodes.nodes.image.large_image_save")
preview_cache = importlib.import_module("NH-Nodes.core.large_preview_cache")
grid = importlib.import_module("NH-Nodes.nodes.image.image_grid_nodes")


def _flat(height, width, channels):
    """[1, H, W, C] image whose channel k holds the byte value 10 * (k + 1); the +0.5 keeps float rounding out of it."""
    values = torch.tensor([(10 * (k + 1) + 0.5) / 255.0 for k in range(channels)])
    return values.expand(1, height, width, channels).clone()


def _save(tmp_path, image, file_format):
    node = large_save.NH_SaveLargeImage()
    node.output_dir = str(tmp_path)
    return node.save(image, "nh_channels", file_format)["result"][1].split("\n")[0]


# --- NH Save Large Image ---

def test_png_keeps_the_alpha_channel(tmp_path):
    with Image.open(_save(tmp_path, _flat(4, 6, 4), "png")) as saved:
        assert saved.mode == "RGBA" and saved.getpixel((2, 1)) == (10, 20, 30, 40)


@pytest.mark.parametrize("file_format", ["jpg", "webp"])
def test_formats_that_need_rgb_drop_the_alpha_channel(tmp_path, file_format):
    with Image.open(_save(tmp_path, _flat(4, 6, 4), file_format)) as saved:
        assert saved.mode == "RGB"


@pytest.mark.parametrize("channels,pixel", [(3, (10, 20, 30)), (1, (10, 10, 10))])
def test_rgb_and_grey_images_are_saved_as_rgb(tmp_path, channels, pixel):
    with Image.open(_save(tmp_path, _flat(4, 6, channels), "png")) as saved:
        assert saved.mode == "RGB" and saved.getpixel((2, 1)) == pixel


# --- large preview cache ---

@pytest.fixture
def temp_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "temp_directory", str(tmp_path))
    return tmp_path


def _thumbnail(manifest):
    thumb = manifest["thumb"]
    return Image.open(os.path.join(folder_paths.get_temp_directory(), thumb["subfolder"], thumb["filename"]))


@pytest.mark.parametrize("preview_format,mode", [("png", "RGBA"), ("jpg", "RGB"), ("webp", "RGB")])
def test_preview_thumbnail_keeps_alpha_only_in_png(temp_dir, preview_format, mode):
    manifest = preview_cache.create_preview_cache(_flat(40, 30, 4)[0], preview_format=preview_format, tile_size=64)

    with _thumbnail(manifest) as thumb:
        assert thumb.mode == mode and thumb.size == (30, 40)


@pytest.mark.parametrize("shape", [(40, 30, 1), (40, 30, 3), (40, 30, 2), (40, 30)])
def test_preview_accepts_grey_rgb_and_two_dimensional_images(temp_dir, shape):
    image = torch.rand(shape)

    manifest = preview_cache.create_preview_cache(image, preview_format="png", tile_size=64)

    assert (manifest["width"], manifest["height"]) == (30, 40)
    with _thumbnail(manifest) as thumb:
        assert thumb.mode == "RGB"


def test_preview_rejects_anything_but_a_tensor(temp_dir):
    with pytest.raises(TypeError, match="Expected IMAGE tensor, got ndarray"):
        preview_cache.create_preview_cache(np.zeros((8, 8, 3), np.float32))


# --- Image Grid Composite ---

def test_grid_cells_are_rgb_whatever_the_input_channels():
    transparent = _flat(4, 4, 4)
    transparent[..., 3] = 0.0  # fully transparent: the colour channels must survive, not be composited away
    grey = _flat(4, 4, 1)

    image, _slots, _used, _info = grid.NH_ImageGridComposite().composite(
        [transparent, grey], columns=2, rows=1, cell_size_mode="manual", cell_width=4, cell_height=4,
        resize_mode="stretch", gap_x=0, gap_y=0, background_color_hex="#000000",
    )

    assert image.shape == (1, 4, 8, 3)
    assert torch.allclose(image[0, 0, 0], torch.tensor([10.0, 20.0, 30.0]) / 255.0)
    assert torch.allclose(image[0, 0, 4], torch.tensor([10.0, 10.0, 10.0]) / 255.0)
