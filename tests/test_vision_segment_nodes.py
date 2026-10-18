"""Tests for vision_segment_nodes.

Unit tests use synthetic label maps (no model). The final smoke test runs the
real SegFormer models and is skipped when weights are not on disk.

Run from the ComfyUI root:
    venv/Scripts/python.exe -m pytest custom_nodes/NH-Nodes/tests -q
"""

import importlib
import os
import sys
import types

import cv2
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
    with_parts = vs.NH_GarmentSegment.build_mask(labels, "upper_garment", True, model="fashionpedia")
    without_parts = vs.NH_GarmentSegment.build_mask(labels, "upper_garment", False, model="fashionpedia")
    assert with_parts[10:12, 4:8].all() and with_parts[16:19, 16:19].sum() == 0
    assert without_parts[10:12, 4:8].sum() == 0
    assert vs.NH_GarmentSegment.build_mask(labels, "footwear", model="fashionpedia").sum() == 0


def test_all_garments_includes_clothing_groups_and_excludes_unassigned_accessories():
    labels = np.arange(47).reshape(1, 47)
    mask = vs.NH_GarmentSegment.build_mask(labels, "all_garments", False, model="fashionpedia")
    expected_ids = set(range(1, 14)) | {15, 16, 17, 20, 21, 22, 23, 24, 26, 28, 29, 30, 31, 32, 34, 35}
    assert set(np.flatnonzero(mask)) == expected_ids


def test_all_garments_combines_shirt_and_pants_in_mask_and_cutout():
    labels = np.zeros((8, 8), np.int64)
    labels[1:3, 2:6] = 1  # shirt
    labels[4:7, 2:6] = 7  # pants
    mask = vs.NH_GarmentSegment.build_mask(labels, "all_garments", False, model="fashionpedia")
    assert np.array_equal(mask, (labels != 0).astype(np.uint8))
    image = np.full((8, 8, 3), 0.5, np.float32)
    mask_t, rgba, mask_image = vs.pack_outputs(image, mask)
    assert torch.equal(mask_t[0], torch.from_numpy(mask))
    assert torch.equal(rgba[0, :, :, 3], mask_t[0])
    assert torch.equal(mask_image[0, :, :, 0], mask_t[0])


@pytest.mark.parametrize("attach_parts", [False, True])
def test_all_garments_attaches_parts_to_upper_and_lower_only_when_enabled(attach_parts):
    labels = np.zeros((24, 30), np.int64)
    labels[2:6, 2:6] = 1  # shirt
    labels[12:16, 2:6] = 7  # pants
    labels[6:8, 2:4] = 33  # pocket touching shirt
    labels[16:18, 2:4] = 36  # zipper touching pants
    labels[10:12, 24:26] = 38  # detached decoration
    mask = vs.NH_GarmentSegment.build_mask(labels, "all_garments", attach_parts, model="fashionpedia")
    assert mask[2:6, 2:6].all() and mask[12:16, 2:6].all()
    assert bool(mask[6:8, 2:4].all()) == attach_parts
    assert bool(mask[16:18, 2:4].all()) == attach_parts
    assert mask[10:12, 24:26].sum() == 0


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
    # the label map scaled to whatever picture (whole frame or crop) the parser is shown

    def parser(bundle, img, confidence=False):
        scaled = cv2.resize(labels.astype(np.int32), (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(np.int64)
        return (scaled, np.full(img.shape[:2], 0.9, np.float16)) if confidence else scaled

    monkeypatch.setattr(vs, "predict_labels", parser)
    monkeypatch.setattr(vs, "offload", lambda bundle: None)
    monkeypatch.setattr(vs, "detect_faces", lambda img: [])  # no face: Portrait Segment parses the whole frame
    image = torch.rand(2, 8, 8, 3)

    mask, rgba, mask_img, segs = vs.NH_GarmentSegment().segment(image, "lower_garment", True, 0, 0, True, "cpu",
                                                                 "fashionpedia")
    assert mask.shape == (2, 8, 8) and rgba.shape == (2, 8, 8, 4) and mask_img.shape == (2, 8, 8, 3)
    assert mask[0, 4:, :].all() and mask[0, :4, :].sum() == 0
    assert [seg.label for seg in segs[0].segments] == ["top", "pants"] and len(segs) == 2  # Fashionpedia names
    # each frame keeps a print of its own picture, so Segs Compose can refuse segs of another photo
    assert all(frame.picture.shape == (32, 32) for frame in segs)
    assert not importlib.import_module("NH-Nodes.core.segs").same_picture(segs[0].picture, segs[1].picture)

    mask, rgba, mask_img, _ = vs.NH_GarmentSegment().segment(image, "all_garments", True, 0, 0, False, "cpu",
                                                             "fashionpedia")
    assert mask.all() and rgba[:, :, :, 3].all() and mask_img.all()

    # FASHN (the default): 2 is hair and 7 a belt there, so only the belt is a garment
    mask, _, _, segs = vs.NH_GarmentSegment().segment(image, "all_garments", True, 0, 0, False, "cpu")
    assert mask[0, 4:, :].all() and mask[0, :4, :].sum() == 0
    assert [seg.label for seg in segs[0].segments] == ["hair", "belt"]

    mask, _, _, segs = vs.NH_PortraitSegment().segment(image, 0, 0, False, "cpu", nose=True, face=False)
    assert [seg.label for seg in segs[0].segments] == ["nose", "eyebrows"]  # CelebAMask-HQ: 2 nose, 7 right brow
    assert mask.shape == (2, 8, 8)
    assert mask[0, :4, :].all() and mask[0, 4:, :].sum() == 0  # label 2 is nose in the face model


def _centre_marker(bundle, img):
    """Stub parser: skin (1) everywhere and a 4 x 4 nose (2) at the centre of the picture it is shown."""
    labels = np.ones(img.shape[:2], np.int64)
    h, w = labels.shape
    labels[h // 2 - 2:h // 2 + 2, w // 2 - 2:w // 2 + 2] = 2
    return labels


def _face_parser(shown):
    """Stub face parser: hair (13) everywhere on the whole frame (the first call); on a crop, a face in its middle
    (skin 1 with a nose 2 and lips 11 at the centre) and hair around it."""
    def parser(bundle, img):
        shown.append(img.shape[:2])
        labels = np.full(img.shape[:2], 13, np.int64)
        if len(shown) > 1:
            h, w = labels.shape
            labels[h // 4:3 * h // 4, w // 4:3 * w // 4] = 1
            labels[h // 2 - 2:h // 2 + 2, w // 2 - 2:w // 2 + 2] = 2
            labels[h // 2 + 3:h // 2 + 5, w // 2 - 2:w // 2 + 2] = 11
        return labels
    return parser


def test_portrait_labels_refine_the_whole_frame_around_each_face(monkeypatch):
    shown = []
    monkeypatch.setattr(vs, "predict_labels", _face_parser(shown))
    monkeypatch.setattr(vs, "detect_faces", lambda img: [(40, 40, 60, 60, 0.9), (140, 40, 160, 60, 0.9)])
    canvas = vs.portrait_labels({}, np.zeros((100, 200, 3), np.uint8))
    # the whole frame first (a padded square), then a tight crop per face to confirm it, then a wide one per face
    assert shown == [(200, 200), (44, 44), (44, 44), (72, 72), (72, 72)]
    assert canvas[47, 50] == 2 and canvas[47, 150] == 2  # each face has its nose (tight crops, moved up a little)
    assert canvas[95, 100] == 13 and canvas[50, 100] == 13  # outside the crops the whole-frame parse stays


def test_portrait_labels_ignore_boxes_that_are_not_faces(monkeypatch):
    shown = []
    monkeypatch.setattr(vs, "predict_labels", lambda bundle, img: shown.append(img.shape[:2]) or np.full(
        img.shape[:2], 13, np.int64))  # the parser sees only hair in the "face" box (a hand, a pattern)
    monkeypatch.setattr(vs, "detect_faces", lambda img: [(40, 40, 60, 60, 0.61)])
    canvas = vs.portrait_labels({}, np.zeros((100, 200, 3), np.uint8))
    assert shown == [(200, 200), (44, 44)] and (canvas == 13).all()  # turned down: no wide crop, nothing pasted


def test_portrait_labels_parse_a_face_filling_the_frame_whole(monkeypatch):
    shown = []
    monkeypatch.setattr(vs, "predict_labels", _face_parser(shown))
    monkeypatch.setattr(vs, "detect_faces", lambda img: [(20, 10, 180, 190, 0.95)])  # a close-up
    vs.portrait_labels({}, np.zeros((200, 200, 3), np.uint8))
    assert shown == [(200, 200)]  # the same as the whole-frame parse


def test_face_detector_failure_falls_back_to_whole_frames(monkeypatch, capsys):
    monkeypatch.setattr(vs, "_detection_failed", False)

    def broken(img):
        raise RuntimeError("Downloading opencv/face_detection_yunet failed")

    monkeypatch.setattr(vs, "detect_faces", broken)
    assert vs.find_faces(np.zeros((50, 50, 3), np.uint8)) == []
    assert "face detector is not available" in capsys.readouterr().out
    assert vs.find_faces(np.zeros((50, 50, 3), np.uint8)) == []  # not tried again, said once
    assert capsys.readouterr().out == ""


def test_portrait_labels_without_a_face_parse_the_whole_frame(monkeypatch):
    shown = []
    monkeypatch.setattr(vs, "predict_labels", lambda bundle, img: shown.append(img.shape[:2]) or _centre_marker(
        bundle, img))
    looked = []
    monkeypatch.setattr(vs, "detect_faces", lambda img: looked.append(img.shape[:2]) or [])
    canvas = vs.portrait_labels({}, np.zeros((100, 200, 3), np.uint8))
    assert looked == [(100, 200), (166, 332)]  # a second look with a grey margin, for a face cut by the frame
    # the whole frame, as a padded square (the parser learnt on square faces), nose at the centre of the frame
    assert shown == [(200, 200)] and canvas.shape == (100, 200) and canvas[50, 100] == 2


def test_find_faces_takes_a_second_look_with_a_margin_for_faces_cut_by_the_frame(monkeypatch):
    calls = []

    def detector(img):
        calls.append(img.shape[:2])
        return [] if len(calls) == 1 else [(30.0, 40.0, 120.0, 150.0, 0.7)]  # found on the padded copy

    monkeypatch.setattr(vs, "detect_faces", detector)
    faces = vs.find_faces(np.zeros((90, 120, 3), np.uint8))
    assert calls == [(90, 120), (150, 200)] and faces == [(0.0, 10.0, 80.0, 90.0, 0.7)]  # mapped back, clipped
    monkeypatch.setattr(vs, "detect_faces", lambda img: [(1.0, 2.0, 3.0, 4.0, 0.9)])
    assert vs.find_faces(np.zeros((90, 120, 3), np.uint8)) == [(1.0, 2.0, 3.0, 4.0, 0.9)]


def test_garment_labels_refine_a_crop_of_the_model_shape_around_the_clothes(monkeypatch):
    shown = []

    def parser(bundle, img, confidence=False):
        shown.append(img.shape[:2])
        labels = np.zeros(img.shape[:2], np.int64)
        if len(shown) == 1:  # the first look at the whole picture: the clothes, under another label, unsure
            labels[100:300, 240:360] = 3
            return labels, np.full(img.shape[:2], 0.5, np.float16)
        labels[np.asarray(img)[..., 0] == 9] = 5  # the close look
        return labels, np.full(img.shape[:2], 0.9, np.float16)

    monkeypatch.setattr(vs, "predict_labels", parser)
    monkeypatch.setattr(vs, "detect_faces", lambda img: [])
    bundle = {"processor": types.SimpleNamespace(size={"height": 576, "width": 384})}
    image = np.zeros((400, 600, 3), np.uint8)
    image[100:300, 240:360, 0] = 9
    canvas = vs.garment_labels(bundle, image)
    crop_h, crop_w = shown[1]
    assert shown[0] == (400, 600) and abs(crop_w / crop_h - 384 / 576) < 0.02  # the model's 2:3, not squeezed
    assert (canvas[100:300, 240:360] == 5).all() and set(np.unique(canvas)) == {0, 5}

    shown.clear()
    monkeypatch.setattr(vs, "predict_labels", lambda bundle, img, confidence=False: shown.append(1) or (
        np.zeros(img.shape[:2], np.int64), np.ones(img.shape[:2], np.float16)))
    assert not vs.garment_labels(bundle, np.zeros((200, 300, 3), np.uint8)).any() and shown == [1]  # no clothes


def test_input_types_expose_expected_widgets():
    spec = vs.NH_GarmentSegment.INPUT_TYPES()
    garment = spec["required"]
    assert garment["part"][0] == ["upper_garment", "lower_garment", "footwear", "headwear", "all_garments", "dress",
                                  "bag", "belt", "scarf", "glasses", "jewelry"]
    assert garment["part"][1]["default"] == "upper_garment"
    # the model choice is optional, so prompts saved before it existed still validate
    assert spec["optional"]["model"][0] == ["fashn", "fashionpedia"] and spec["optional"]["model"][1]["default"] == "fashn"
    assert vs.NH_GarmentSegment.RETURN_NAMES[-1] == "segs" and vs.NH_PortraitSegment.RETURN_NAMES[-1] == "segs"
    portrait = vs.NH_PortraitSegment.INPUT_TYPES()["required"]
    for region in ("face", "neck", "hair", "eyes", "lips", "nose", "eyebrows", "ears"):
        assert portrait[region][1]["default"] is True
    assert portrait["mouth_interior"][1]["default"] is False


# --- real model smoke test (skipped without weights) -------------------------
@pytest.mark.parametrize("model_name", ["segformer_fashion", "face_parsing", "fashn_human_parser"])
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


def test_fashn_parts_map_to_its_classes():
    labels = np.array([[3, 4, 5, 6], [7, 8, 9, 10], [11, 15, 17, 2]])
    build = vs.NH_GarmentSegment.build_mask
    assert build(labels, "upper_garment", model="fashn").tolist() == [[1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
    assert build(labels, "lower_garment", model="fashn").tolist() == [[0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 0]]
    # footwear is FASHN's feet, and all_garments has it, as with Fashionpedia's shoes
    assert build(labels, "footwear", model="fashn")[2, 1] == 1 and build(labels, "all_garments", model="fashn")[2, 1] == 1
    assert build(labels, "all_garments").tolist() == build(labels, "all_garments", model="fashn").tolist()  # default
    for part, value in (("dress", 4), ("bag", 8), ("headwear", 9), ("glasses", 11), ("jewelry", 17)):
        assert build(labels, part, model="fashn").sum() == 1 and build(labels, part, model="fashn")[labels == value].all()
    # the same parts on Fashionpedia ids; it has no jewelry class
    fashionpedia = np.array([[11, 25, 20, 26], [14, 12, 35, 1]])
    assert build(fashionpedia, "dress", model="fashionpedia").tolist() == [[1, 0, 0, 0], [0, 1, 0, 0]]
    assert build(fashionpedia, "belt", model="fashionpedia").tolist() == [[0, 0, 1, 0], [0, 0, 1, 0]]
    assert not build(fashionpedia, "jewelry", model="fashionpedia").any()


def test_parts_frame_matches_the_enlarged_label_map_pixel_for_pixel():
    rng = np.random.default_rng(0)
    work = rng.integers(0, 4, (13, 17))
    rows, cols = vs._nearest_index(50, 13), vs._nearest_index(61, 17)
    full = work[np.ix_(rows, cols)]
    frame = vs.parts_frame(work, {"a": {1}, "bc": {2, 3}, "none": {9}}, (50, 61))
    assert [seg.label for seg in frame.segments] == ["a", "bc"] and (frame.height, frame.width) == (50, 61)
    for seg, ids in zip(frame.segments, ({1}, {2, 3})):
        canvas = np.zeros((50, 61), bool)
        x1, y1, x2, y2 = seg.box
        canvas[y1:y2, x1:x2] = seg.mask.numpy()
        assert (canvas == np.isin(full, list(ids))).all()


def test_labels_at_work_size_shrinks_big_frames_only(monkeypatch):
    shown = []
    monkeypatch.setattr(vs, "WORK_SIDE", 64)

    def labeler(bundle, picture):
        shown.append(picture.shape[:2])
        return np.ones(picture.shape[:2], np.int64)

    full, work = vs.labels_at_work_size(labeler, {}, np.zeros((100, 200, 3), np.float32))
    assert shown == [(32, 64)] and full.shape == (100, 200) and work.shape == (32, 64)
    full, work = vs.labels_at_work_size(labeler, {}, np.zeros((30, 40, 3), np.float32))
    assert full is work and full.shape == (30, 40)


def test_garment_jewelry_with_fashionpedia_says_why_it_is_empty(monkeypatch, capsys):
    monkeypatch.setattr(vs, "load_segformer", lambda name, device: {"name": name})
    monkeypatch.setattr(vs, "predict_labels", lambda bundle, img, confidence=False: (
        np.zeros(img.shape[:2], np.int64), np.ones(img.shape[:2], np.float16)) if confidence else np.zeros(
        img.shape[:2], np.int64))
    monkeypatch.setattr(vs, "offload", lambda bundle: None)
    monkeypatch.setattr(vs, "detect_faces", lambda img: [])
    mask, _, _, _ = vs.NH_GarmentSegment().segment(torch.rand(1, 8, 8, 3), "jewelry", True, 0, 0, False, "cpu",
                                                   "fashionpedia")
    assert not mask.any() and "no jewelry class" in capsys.readouterr().out


def test_parts_frame_is_exact_for_small_and_border_parts():
    rng = np.random.default_rng(1)
    parts = {"a": {1}, "b": {2, 3}, "corner": {5}, "other corner": {6}, "missing": {9}}
    for small_h, small_w, height, width in ((13, 17, 50, 61), (9, 7, 9, 7), (20, 30, 41, 33), (5, 8, 160, 97)):
        work = np.zeros((small_h, small_w), np.int64)
        work[rng.integers(0, small_h, 4), rng.integers(0, small_w, 4)] = rng.integers(1, 4, 4)
        work[0, -1], work[-1, 0] = 5, 6  # one pixel each, on the border
        full = work[np.ix_(vs._nearest_index(height, small_h), vs._nearest_index(width, small_w))]
        frame = vs.parts_frame(work, parts, (height, width))
        assert [seg.label for seg in frame.segments] == [
            name for name, ids in parts.items() if np.isin(work, list(ids)).any()]
        for seg in frame.segments:
            wanted = np.isin(full, list(parts[seg.label]))
            canvas = np.zeros((height, width), bool)
            x1, y1, x2, y2 = seg.box
            canvas[y1:y2, x1:x2] = seg.mask.numpy()
            ys, xs = np.nonzero(wanted)
            assert (canvas == wanted).all() and seg.box == (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    # class ids past a byte take the slower comparison, with the same result
    work = np.array([[0, 300], [7, 300]])
    (seg,) = vs.parts_frame(work, {"big": {300}}, (4, 4)).segments
    assert seg.box == (2, 0, 4, 4) and seg.mask.all()


def test_fashn_gets_the_area_resize_it_was_trained_with(monkeypatch):
    shown = []

    class _Logits:
        def __init__(self, t):
            self.logits = t

    class _Model:
        device = torch.device("cpu")

        def __call__(self, **kw):
            return _Logits(torch.zeros(1, 3, 8, 8))

    class _Processor:
        def __init__(self, height, width):
            self.size = {"height": height, "width": width}

        def __call__(self, images, return_tensors):
            shown.append(images.shape[:2])
            return {"pixel_values": torch.zeros(1)}

    monkeypatch.setattr(backend, "_ensure_on_device", lambda b: None)
    fashn = {"processor": _Processor(576, 384), "model": _Model(), "name": "fashn_human_parser"}
    backend.predict_labels(fashn, np.zeros((800, 500, 3), np.uint8))  # between 1x and 2x its size
    backend.predict_labels(fashn, np.zeros((300, 200, 3), np.uint8))  # smaller than its size
    face = {"processor": _Processor(512, 512), "model": _Model(), "name": "face_parsing"}
    backend.predict_labels(face, np.zeros((700, 700, 3), np.uint8))  # under 2x: the processor resizes it
    assert shown == [(576, 384), (576, 384), (700, 700)]


def test_parser_segs_carry_their_model_and_answer_to_its_words(monkeypatch):
    segs_core = importlib.import_module("NH-Nodes.core.segs")
    work = np.array([[3, 4, 10, 5], [6, 7, 12, 1]])  # FASHN: top, dress, scarf, skirt / pants, belt, arms, face
    frame = vs.parts_frame(work, vs.FASHN_SEGS, (2, 4), "fashn_human_parser")
    assert {seg.source for seg in frame.segments} == {"fashn_human_parser"}
    upper = segs_core.evaluate(segs_core.parse_expression("upper garment"), frame).numpy()
    assert upper.tolist() == [[True, True, True, False], [False, False, False, False]]
    lower = segs_core.evaluate(segs_core.parse_expression("lower_garment"), frame).numpy()
    assert lower.tolist() == [[False, False, False, True], [True, True, False, False]]
    face = vs.parts_frame(np.array([[1, 4, 13, 17]]), vs.PORTRAIT_SEGS, (1, 4), "face_parsing")
    assert segs_core.evaluate(segs_core.parse_expression("face"), face).numpy().tolist() == [[True, True, False, False]]


def test_image_batches_of_any_float_type_are_read():
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        (picture,) = vs._image_batch_to_numpy(torch.full((1, 3, 4, 3), 0.5, dtype=dtype))
        assert picture.dtype == np.float32 and picture.shape == (3, 4, 3) and np.allclose(picture, 0.5)
