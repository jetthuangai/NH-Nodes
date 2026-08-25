"""Tests for vision_segment_nodes.

Unit tests use synthetic label maps (no model). The final smoke test runs the
real SegFormer models and is skipped when weights are not on disk.

Run from the ComfyUI root:
    venv/Scripts/python.exe -m pytest custom_nodes/NH-Nodes/tests -q
"""

import importlib
import os
import sys

import numpy as np
import pytest
import torch

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PACK_DIR))
for path in (COMFY_ROOT, os.path.dirname(PACK_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

vs = importlib.import_module("NH-Nodes.nodes.vision.vision_segment_nodes")
backend = importlib.import_module("NH-Nodes.core.segformer_backend")


# --- label map helpers -------------------------------------------------------
def test_labels_to_mask_unions_ids_and_handles_empty():
    labels = np.array([[0, 1, 2], [3, 4, 5]])
    mask = vs.labels_to_mask(labels, {1, 5})
    assert mask.dtype == np.uint8
    assert mask.tolist() == [[0, 1, 0], [0, 0, 1]]
    assert vs.labels_to_mask(labels, set()).sum() == 0


def test_garment_groups_are_disjoint_and_exclude_shoe_from_lower():
    groups = list(vs.GARMENT_GROUPS.values())
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            assert not (a & b)
        assert not (a & vs.GARMENT_ATTACHED_PARTS)
    assert 24 not in vs.GARMENT_GROUPS["lower_garment"]  # shoe
    assert {17, 26} <= vs.GARMENT_GROUPS["upper_garment"]  # tie, scarf


def test_absorb_only_parts_touching_main_mask():
    main = np.zeros((20, 20), np.uint8)
    main[5:10, 5:10] = 1
    parts = np.zeros((20, 20), np.uint8)
    parts[10:12, 5:10] = 1  # touches bottom edge of main
    parts[16:18, 16:18] = 1  # far away
    merged = vs.absorb_touching_parts(main, parts, reach=1)
    assert merged[10:12, 5:10].all()
    assert merged[16:18, 16:18].sum() == 0
    # empty inputs pass through untouched
    assert vs.absorb_touching_parts(np.zeros((4, 4), np.uint8), parts[:4, :4]).sum() == 0


def test_garment_build_mask_respects_attach_parts_flag():
    labels = np.zeros((20, 20), np.int64)
    labels[2:10, 2:18] = 1  # shirt
    labels[10:12, 4:8] = 33  # pocket touching shirt
    labels[16:19, 16:19] = 36  # zipper elsewhere
    with_parts = vs.NH_GarmentSegment.build_mask(labels, "upper_garment", True)
    without_parts = vs.NH_GarmentSegment.build_mask(labels, "upper_garment", False)
    assert with_parts[10:12, 4:8].all() and with_parts[16:19, 16:19].sum() == 0
    assert without_parts[10:12, 4:8].sum() == 0
    assert vs.NH_GarmentSegment.build_mask(labels, "footwear").sum() == 0


def test_portrait_build_mask_face_is_solid_and_regions_union():
    labels = np.zeros((10, 10), np.int64)
    labels[2:8, 2:8] = 1  # skin
    labels[4, 4] = 4  # eye inside face
    labels[6, 5] = 11  # upper lip
    labels[0, 0] = 13  # hair
    face = vs.NH_PortraitSegment.build_mask(labels, ["face"])
    assert face[2:8, 2:8].all(), "face must not have holes at eyes/lips"
    assert face[0, 0] == 0
    hair_eyes = vs.NH_PortraitSegment.build_mask(labels, ["hair", "eyes"])
    assert hair_eyes.sum() == 2
    assert vs.NH_PortraitSegment.build_mask(labels, []).sum() == 0


# --- refinement ----------------------------------------------------------------
def test_fill_holes_closes_enclosed_gap_only():
    mask = np.ones((9, 9), np.uint8)
    mask[4, 4] = 0  # enclosed hole
    mask[0, :] = 0  # open border row stays open
    filled = vs.fill_mask_holes(mask)
    assert filled[4, 4] == 1
    assert filled[0, :].sum() == 0


def test_refine_mask_expand_shrink_blur():
    mask = np.zeros((21, 21), np.uint8)
    mask[8:13, 8:13] = 1
    grown = vs.refine_mask(mask, expand=2)
    shrunk = vs.refine_mask(mask, expand=-1)
    blurred = vs.refine_mask(mask, blur=3)
    assert grown.sum() > mask.sum() > shrunk.sum() > 0
    assert blurred.dtype == np.float32 and 0.0 < blurred[7, 10] < 1.0
    assert blurred.max() <= 1.0 and blurred.min() >= 0.0


def test_pack_outputs_shapes_and_alpha():
    image = np.random.rand(6, 5, 3).astype(np.float32)
    mask = np.zeros((6, 5), np.float32)
    mask[1, 1] = 1.0
    mask_t, rgba_t, mask_img_t = vs.pack_outputs(image, mask)
    assert mask_t.shape == (1, 6, 5)
    assert rgba_t.shape == (1, 6, 5, 4) and mask_img_t.shape == (1, 6, 5, 3)
    assert torch.equal(rgba_t[0, :, :, :3], torch.from_numpy(image))
    assert rgba_t[0, 1, 1, 3] == 1.0 and rgba_t[0, 0, 0, 3] == 0.0


def test_image_batch_to_numpy_strips_alpha_and_batches():
    rgba = torch.rand(2, 4, 4, 4)
    frames = vs._image_batch_to_numpy(rgba)
    assert len(frames) == 2 and frames[0].shape == (4, 4, 3)
    single = vs._image_batch_to_numpy(torch.rand(4, 4, 3))
    assert len(single) == 1
    grey = vs._image_batch_to_numpy(torch.rand(1, 4, 4, 1))
    assert grey[0].shape == (4, 4, 3)
    with pytest.raises(ValueError):
        vs._image_batch_to_numpy(torch.rand(1, 4, 4, 2))


def test_portrait_face_includes_eyeglasses():
    labels = np.zeros((10, 10), np.int64)
    labels[2:8, 1:9] = 1  # skin
    labels[4, 0:10] = 3  # glasses band reaching the silhouette edge
    face = vs.NH_PortraitSegment.build_mask(labels, ["face"])
    assert face[4, 1:9].all()


# --- nodes end-to-end with a stubbed model -------------------------------------
def test_nodes_end_to_end_with_stub_backend(monkeypatch):
    labels = np.zeros((8, 8), np.int64)
    labels[0:4, :] = 2  # top
    labels[4:8, :] = 7  # pants
    monkeypatch.setattr(vs, "load_segformer", lambda name, device: {"name": name})
    monkeypatch.setattr(vs, "predict_labels", lambda bundle, img: labels)
    monkeypatch.setattr(vs, "offload", lambda bundle: None)
    image = torch.rand(2, 8, 8, 3)

    mask, rgba, mask_img = vs.NH_GarmentSegment().segment(image, "lower_garment", True, 0, 0, True, "cpu")
    assert mask.shape == (2, 8, 8) and rgba.shape == (2, 8, 8, 4) and mask_img.shape == (2, 8, 8, 3)
    assert mask[0, 4:, :].all() and mask[0, :4, :].sum() == 0

    mask, _, _ = vs.NH_PortraitSegment().segment(image, 0, 0, False, "cpu", nose=True, face=False)
    assert mask.shape == (2, 8, 8)
    assert mask[0, :4, :].all() and mask[0, 4:, :].sum() == 0  # label 2 is nose in the face model


def test_input_types_expose_expected_widgets():
    garment = vs.NH_GarmentSegment.INPUT_TYPES()["required"]
    assert garment["part"][0] == ["upper_garment", "lower_garment", "footwear", "headwear"]
    portrait = vs.NH_PortraitSegment.INPUT_TYPES()["required"]
    for region in ("face", "neck", "hair", "eyes", "lips", "nose", "eyebrows", "ears"):
        assert portrait[region][1]["default"] is True
    assert portrait["mouth_interior"][1]["default"] is False


# --- real model smoke test (skipped without weights) -------------------------
@pytest.mark.parametrize("model_name", ["segformer_fashion", "face_parsing"])
def test_real_model_smoke(model_name):
    if backend.resolve_model_dir(model_name, download=False) is None:
        pytest.skip(f"{model_name} weights not on disk")
    bundle = backend.load_segformer(model_name, "cpu")
    labels = backend.predict_labels(bundle, np.zeros((64, 48, 3), np.uint8))
    assert labels.shape == (64, 48) and labels.dtype == np.int64
    for class_id, label in backend.MODEL_SPECS[model_name]["signature"].items():
        assert bundle["id2label"][class_id] == label
    backend.offload(bundle)
    assert bundle["model"].device.type == "cpu"


def test_predict_labels_caps_logit_size_and_restores_input_size(monkeypatch):
    class _Logits:
        def __init__(self, t):
            self.logits = t

    class _Model:
        device = torch.device("cpu")

        def __call__(self, **kw):
            return _Logits(torch.zeros(1, 3, 8, 8))

    bundle = {"processor": lambda images, return_tensors: {"pixel_values": torch.zeros(1)},
              "model": _Model(), "device": torch.device("cpu")}
    monkeypatch.setattr(backend, "_ensure_on_device", lambda b: None)
    monkeypatch.setattr(backend, "_MAX_LOGIT_SIDE", 16)
    labels = backend.predict_labels(bundle, np.zeros((40, 30, 3), np.uint8))
    assert labels.shape == (40, 30) and labels.dtype == np.int64
