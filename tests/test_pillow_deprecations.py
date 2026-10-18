"""Pillow deprecations on the garment-mask and VTON paths.

Pillow 11.3 deprecated Image.fromarray(array, mode) and warned at every call (12.0 took most of that back and 13.0
only rejects a mode that changes the data type). The code reads uint8 label maps with a plain Image.fromarray(array)
instead, which behaves the same on every Pillow version and stays quiet.

Both tests turn any "will be removed in Pillow N" DeprecationWarning into an error, so a `mode=` that comes back
fails here.
"""

import importlib
import warnings

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("skimage")  # core.utils_mask imports it at module level

utils_mask = importlib.import_module("NH-Nodes.core.utils_mask")
vton = importlib.import_module("NH-Nodes.nodes.vton.vton_preprocessor_nodes")

H, W = 48, 64


@pytest.fixture(autouse=True)
def pillow_deprecations_are_errors():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=r".*removed in Pillow", category=DeprecationWarning)
        yield


def _parsing_labels():
    labels = np.zeros((H, W), np.uint8)
    labels[5:12, 20:40] = 11  # head
    labels[12:30, 15:45] = 4  # upper clothes
    labels[30:44, 18:42] = 5  # trousers
    labels[44:48, 18:42] = 9  # shoes
    return labels


@pytest.mark.parametrize("category", ["Upper-body", "Lower-body", "Dresses"])
def test_garment_masks_do_not_trip_a_pillow_deprecation(category):
    pose = np.full((134, 2), -1.0, np.float32)  # nothing detected: the masks fall back to the garment contours

    mask, mask_gray = utils_mask.get_mask_location(category, Image.fromarray(_parsing_labels()), pose, W, H, 0, 0, 0, 0)

    assert mask.mode == mask_gray.mode == "L" and mask.size == (W, H)


class _Pose:
    def __call__(self, bgr):
        return np.zeros(bgr.shape, np.uint8), None, None, np.full((1, 134, 2), 0.5, np.float32)


class _Parsing:
    def __call__(self, image):
        labels = Image.fromarray(_parsing_labels())
        return labels, labels


@pytest.mark.parametrize("category", ["Upper-body", "Lower-body", "Dresses"])
def test_vton_node_builds_its_coloured_parsing_image_without_a_pillow_deprecation(monkeypatch, category):
    monkeypatch.setattr(vton, "LOADED_VTON_MODELS", {"dwprocessor": _Pose(), "parsing_model": _Parsing(), "device": "cpu"})

    outputs = vton.NH_VTonUltimateProcessor().process(torch.full((1, H, W, 3), 0.5), category, 2, False, True, True, "cpu")

    parsing_image = outputs[3]
    assert parsing_image.shape == (1, H, W, 3)
    # The palette is applied label by label: row 20, column 30 is label 4 (upper clothes), row 46, column 25 label 9 (shoes).
    for (row, column), label in (((20, 30), 4), ((46, 25), 9), ((0, 0), 0)):
        colour = torch.tensor(vton.LIP_PALETTE[3 * label:3 * label + 3], dtype=torch.float32) / 255.0
        assert torch.allclose(parsing_image[0, row, column], colour), label
