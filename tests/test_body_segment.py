"""Body Segment (Sapiens2): region words, the formula, the node with a stub parser, and the backend without weights.

The last test runs the real Sapiens2 0.4B and is skipped when its weights are not on disk.
"""

import importlib

import numpy as np
import pytest
import torch

bs = importlib.import_module("NH-Nodes.nodes.vision.body_segment_nodes")
s2 = importlib.import_module("NH-Nodes.core.sapiens2_backend")
C = {name: index for index, name in enumerate(s2.CLASSES)}


# --- region words --------------------------------------------------------------
def test_region_words_cover_the_right_classes():
    assert all(0 < i < len(s2.CLASSES) for ids in bs.REGIONS.values() for i in ids)
    skin = bs.REGIONS["skin"]
    assert C["face_neck"] in skin and C["left_hand"] in skin and C["torso"] in skin
    assert not skin & (bs.LIPS | bs.TEETH | bs.CLOTHES | {C["hair"], C["eyeglass"]})
    assert bs.REGIONS["person"] == frozenset(range(1, len(s2.CLASSES)))
    assert bs.REGIONS["mouth"] == bs.LIPS | bs.TEETH | {C["tongue"]}
    assert all(word == bs.normalize_label(word) for word in bs.REGIONS)  # looked up after normalising
    assert set().union(*bs.PARTS.values()) == set(range(1, len(s2.CLASSES)))  # every class has a segs part
    labels = np.array([[C["hair"], C["upper_lip"]], [0, C["lower_teeth"]]])
    assert bs.region_mask(labels, "Lips").tolist() == [[False, True], [False, False]]
    assert bs.region_mask(labels, "wings") is None


# --- the node, with a stub parser -------------------------------------------------
H, W = 60, 40


def _picture():
    """A person-shaped label map (the class id is written into the red channel, so any crop parses the same)."""
    labels = np.zeros((H, W), np.int64)
    labels[5:15, 15:25] = C["hair"]
    labels[15:25, 15:25] = C["face_neck"]
    labels[22:24, 18:22] = C["upper_lip"]
    labels[25:45, 10:30] = C["upper_clothing"]
    labels[25:45, 5:10] = C["left_upper_arm"]
    labels[45:58, 12:28] = C["lower_clothing"]
    image = np.zeros((H, W, 3), np.float32)
    image[..., 0] = labels * 8 / 255.0
    return labels, torch.from_numpy(image)[None]


def _run(monkeypatch, regions, show_preview=False):
    saved, shown = [], []

    def stub_predict(bundle, picture, confidence=False):
        shown.append(picture.shape[:2])
        labels = np.rint(picture[..., 0] / 8.0).astype(np.int64)
        return (labels, np.full(labels.shape, 0.9, np.float16)) if confidence else labels

    monkeypatch.setattr(bs, "find_faces", lambda picture, second_look=True: [])
    monkeypatch.setattr(bs.sapiens2, "load", lambda size, device: {"size": size})
    monkeypatch.setattr(bs.sapiens2, "predict_labels", stub_predict)
    monkeypatch.setattr(bs.sapiens2, "offload", lambda bundle: None)
    monkeypatch.setattr(bs, "save_temp_previews", lambda pictures, prefix: saved.extend(pictures) or [{"f": 1}])
    labels, image = _picture()
    out = bs.NH_BodySegment().segment(image, regions, "0.4b", 0, 0, False, "cpu", show_preview)
    return out, labels, saved, shown


def test_formula_over_body_regions(monkeypatch):
    (mask, rgba, mask_image, segs, report), labels, _, shown = _run(monkeypatch, "skin - lips")
    skin = np.isin(labels, list(bs.REGIONS["skin"])) & ~np.isin(labels, list(bs.LIPS))
    assert mask.shape == (1, H, W) and (mask[0].numpy() > 0.5).tolist() == skin.tolist()
    assert rgba.shape == (1, H, W, 4) and mask_image.shape == (1, H, W, 3)
    assert shown[0] == (H, W) and len(shown) >= 1  # a first look at the whole picture
    assert {seg.label for seg in segs[0].segments} == {"hair", "face", "lips", "arms", "top", "bottom"}
    assert "skin: " in report and "% of the picture" in report and "lips: " in report


def test_background_vietnamese_and_unknown_words(monkeypatch):
    (mask, _, _, _, report), labels, _, _ = _run(monkeypatch, "background")
    assert (mask[0].numpy() > 0.5).tolist() == (labels == 0).tolist()
    assert "background = everything outside the people" in report
    (mask, _, _, _, report), labels, _, _ = _run(monkeypatch, "da, tóc, răng, cánh")
    wanted = np.isin(labels, list(bs.REGIONS["skin"] | bs.REGIONS["hair"]))
    assert (mask[0].numpy() > 0.5).tolist() == wanted.tolist()
    assert "da -> skin:" in report and "răng -> teeth: not in the picture" in report
    assert "cánh: not a body region" in report


def test_errors_and_preview(monkeypatch):
    with pytest.raises(ValueError, match="at least one region"):
        _run(monkeypatch, " , # nothing")
    with pytest.raises(ValueError, match="not closed"):
        _run(monkeypatch, "skin - (hair")
    out, _, saved, _ = _run(monkeypatch, "hair", show_preview=True)
    assert set(out) == {"ui", "result"} and len(out["result"]) == 5 and len(saved) == 1
    assert saved[0].shape[1] == W and saved[0].shape[0] > H  # the picture plus the report below it
    required = bs.NH_BodySegment.INPUT_TYPES()["required"]
    assert list(required)[-1] == "show_preview" and required["model_size"][0] == ["0.4b", "0.8b", "1b"]


# --- backend without weights ---------------------------------------------------------
def test_seg_head_turns_patches_into_full_resolution_scores():
    head = s2.SegHead(32, deconv_channels=(16, 8, 8, 8), conv_channels=(8, 8), classes=len(s2.CLASSES))
    scores = head(torch.zeros(1, 32, 4, 3))
    assert scores.shape == (1, 29, 64, 48)  # 16 x the patch grid, like 64 x 48 patches -> 1024 x 768


def test_a_wrong_checkpoint_is_refused_with_advice(monkeypatch, tmp_path):
    from safetensors.torch import save_file

    path = tmp_path / "sapiens2_0.4b_seg.safetensors"
    save_file({"backbone.cls_token": torch.zeros(1, 1, 1024)}, str(path))
    monkeypatch.setattr(s2, "model_path", lambda size, download=True: str(path))
    monkeypatch.setattr(s2, "_LOADED", {})
    with pytest.raises(RuntimeError, match="is not the Sapiens2 0.4b segmentation checkpoint"):
        s2.load("0.4b", torch.device("cpu"))


def test_model_path_downloads_the_pinned_file_once(monkeypatch, tmp_path):
    monkeypatch.setattr(s2, "register_model_folder", lambda *a: [str(tmp_path)])
    monkeypatch.setattr(s2, "default_folder", lambda: str(tmp_path))
    calls = []

    def fake_download(repo, revision, target, patterns):
        calls.append((repo, revision, tuple(patterns)))
        (tmp_path / "sapiens2-seg-1b").mkdir()
        (tmp_path / "sapiens2-seg-1b" / s2.MODELS["1b"]["file"]).write_bytes(b"x")

    monkeypatch.setattr(s2, "download_pinned", fake_download)
    assert s2.model_path("1b", download=False) is None
    assert s2.model_path("1b").endswith(s2.MODELS["1b"]["file"]) and s2.model_path("1b")
    assert calls == [("facebook/sapiens2-seg-1b", s2.MODELS["1b"]["revision"], (s2.MODELS["1b"]["file"],))]


def test_predict_labels_shrinks_to_the_model_size_and_restores_the_picture_size(monkeypatch):
    seen = []

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, pixels):
            seen.append(tuple(pixels.shape))
            scores = torch.zeros(1, len(s2.CLASSES), 64, 48)
            scores[:, C["hair"], :32] = 1.0
            return scores

    bundle = {"model": Fake(), "device": torch.device("cpu")}
    labels = s2.predict_labels(bundle, np.zeros((3000, 2000, 3), np.uint8))
    assert seen == [(1, 3, 1024, 768)] and labels.shape == (3000, 2000) and labels.dtype == np.int64
    assert (labels[:1400] == C["hair"]).all() and (labels[1600:] == 0).all()


# --- real model (skipped without weights) ---------------------------------------------
def test_real_sapiens2_smoke():
    if s2.model_path("0.4b", download=False) is None:
        pytest.skip("Sapiens2 0.4b weights not on disk")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = s2.load("0.4b", device)
    labels = s2.predict_labels(bundle, np.full((200, 150, 3), 128, np.uint8))
    s2.offload(bundle)
    assert labels.shape == (200, 150) and labels.min() >= 0 and labels.max() < len(s2.CLASSES)
