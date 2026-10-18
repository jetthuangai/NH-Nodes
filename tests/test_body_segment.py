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
    assert bs.region_ids("Lips", "Lips", "english") == bs.LIPS and bs.region_ids("wings", "wings", "english") is None
    assert bs.region_ids("tay", "arm", "translated") == bs.ARMS | bs.HANDS  # the whole limb in Vietnamese
    assert bs.region_ids("cánh tay", "arm", "translated") == bs.ARMS


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
    (mask, _, _, _, report), labels, _, _ = _run(monkeypatch, "da, tóc, răng")
    wanted = np.isin(labels, list(bs.REGIONS["skin"] | bs.REGIONS["hair"]))
    assert (mask[0].numpy() > 0.5).tolist() == wanted.tolist()
    assert "da -> skin:" in report and "răng -> teeth: not in the picture" in report
    # "tay" and "chân" are the whole limb in Vietnamese, hand and foot included
    (mask, _, _, _, report), labels, _, _ = _run(monkeypatch, "tay")
    assert (mask[0].numpy() > 0.5).tolist() == np.isin(labels, list(bs.ARMS | bs.HANDS)).tolist()
    assert "tay: the whole limb, hands included" in report


def test_errors_and_preview(monkeypatch):
    with pytest.raises(ValueError, match="at least one region"):
        _run(monkeypatch, " , # nothing")
    with pytest.raises(ValueError, match="not closed"):
        _run(monkeypatch, "skin - (hair")
    out, _, saved, _ = _run(monkeypatch, "hair", show_preview=True)
    assert set(out) == {"ui", "result"} and len(out["result"]) == 5 and len(saved) == 1
    assert saved[0].shape[1] == W and saved[0].shape[0] > H  # the picture plus the report below it
    spec = bs.NH_BodySegment.INPUT_TYPES()
    assert list(spec["optional"]) == ["show_preview"] and "show_preview" not in spec["required"]
    assert spec["required"]["model_size"][0] == ["0.4b", "0.8b", "1b"]


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


@pytest.mark.parametrize("regions, message", [
    ("cánh", "'cánh' is not in the Vietnamese word list"),
    ("wings", "'wings' is not a body region; use skin, hair"),
    ("skin, mắt", r"'mắt' \(eye\) is not a body region: eyes, eyebrows, nose and ears are part of 'face'"),
    ("hat", "one class, 'accessories'"),
    ("skin and hair", r"combine regions with \+ or a comma"),
    ("hair*", "has a wildcard"),
])
def test_unknown_words_are_refused_before_the_model_loads(monkeypatch, regions, message):
    monkeypatch.setattr(bs.sapiens2, "load", lambda size, device: pytest.fail("the model must not load"))
    with pytest.raises(ValueError, match=message):
        bs.NH_BodySegment().segment(torch.zeros(1, 8, 8, 3), regions, "0.4b", 0, 0, False, "cpu", False)


def test_new_region_words():
    for word, region in (("t-shirt", "top"), ("jacket", "top"), ("sunglasses", "glasses"), ("face skin", "face"),
                         ("sneakers", "shoes"), ("ao dai", "dress")):
        assert bs.REGIONS[word] == bs.REGIONS[region]
    assert bs.REGIONS["head"] == bs.FACE | bs.HAIR
    assert not set(bs.REGIONS) & set(bs.NOT_REGIONS)


def test_report_explains_sides_shared_classes_and_tiny_shares():
    counts = np.zeros(len(s2.CLASSES), np.int64)
    counts[C["left_hand"]], counts[C["face_neck"]], counts[C["upper_clothing"]] = 5, 3000, 40000
    lines = bs.describe(bs.parse_concepts("left hand, neck, dress"), counts, 100000)
    assert "left hand: under 0.1% of the picture" in lines and "neck: 3.0% of the picture" in lines
    assert "dress: 40.0% of the picture" in lines
    assert bs._SIDES in lines and bs._FACE_CLASS in lines
    assert "dress: Sapiens2 labels a dress as a top (upper clothes)" in lines
    batch = bs.describe(bs.parse_concepts("hair"), counts, 100000, "pictures")
    assert batch == ["hair: not in the pictures"]


def test_preview_shows_only_the_parts_the_mask_keeps(monkeypatch):
    drawn = []
    monkeypatch.setattr(bs, "preview_picture",
                        lambda picture, frame, notes, mask=None: drawn.append(frame) or np.zeros((4, 4, 3), np.uint8))
    out, _, saved, _ = _run(monkeypatch, "skin - lips", show_preview=True)
    assert {seg.label for seg in drawn[0].segments} == {"face", "arms"} and len(saved) == 1
    drawn.clear()
    _run(monkeypatch, "background", show_preview=True)
    assert drawn[0].segments == ()  # nothing named: only the lit background


def test_a_failed_preview_keeps_the_outputs(monkeypatch, capsys):
    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(bs, "preview_picture", broken)
    out, _, _, _ = _run(monkeypatch, "hair", show_preview=True)
    assert len(out) == 5 and "could not show the preview" in capsys.readouterr().out


def test_a_batch_gets_one_mask_and_frame_per_picture(monkeypatch):
    monkeypatch.setattr(bs, "find_faces", lambda picture, second_look=True: [])
    monkeypatch.setattr(bs.sapiens2, "load", lambda size, device: {"size": size})
    monkeypatch.setattr(bs.sapiens2, "offload", lambda bundle: None)
    monkeypatch.setattr(bs.sapiens2, "predict_labels", lambda bundle, picture, confidence=False: (
        np.rint(picture[..., 0] / 8.0).astype(np.int64), np.full(picture.shape[:2], 0.9, np.float16)))
    labels, image = _picture()
    batch = torch.cat([image, torch.flip(image, dims=[2])])
    mask, _, _, segs, report = bs.NH_BodySegment().segment(batch, "hair", "0.4b", 0, 0, False, "cpu", False)
    assert mask.shape == (2, H, W) and len(segs) == 2 and "% of the pictures" in report
    assert (mask[1].numpy() > 0.5).tolist() == np.fliplr(labels == C["hair"]).tolist()
    assert all(frame.picture is not None for frame in segs)


def test_a_damaged_checkpoint_is_refused_with_advice(monkeypatch, tmp_path):
    path = tmp_path / "sapiens2_0.4b_seg.safetensors"
    path.write_bytes(b"\x10\x00\x00\x00\x00\x00\x00\x00{\"a\":")  # cut short
    monkeypatch.setattr(s2, "model_path", lambda size, download=True: str(path))
    monkeypatch.setattr(s2, "_LOADED", {"0.8b": {"model": None}})
    with pytest.raises(RuntimeError, match="cannot be read .* delete it so it is downloaded again"):
        s2.load("0.4b", torch.device("cpu"))
    assert "0.8b" not in s2._LOADED  # one size at a time


def test_cpu_fallback_is_said_once_and_cpu_runs_in_float32(monkeypatch, capsys):
    monkeypatch.setattr(s2.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(s2, "_said_cpu", False)
    assert s2.torch_device("cuda").type == "cpu" and s2.torch_device("cuda").type == "cpu"
    assert capsys.readouterr().out.count("CUDA not available") == 1
    bundle = {"model": torch.nn.Linear(2, 2).to(torch.bfloat16), "device": torch.device("cpu")}
    s2._ensure_on_device(bundle)
    assert next(bundle["model"].parameters()).dtype == torch.float32


def test_real_sapiens2_on_a_photo():
    if s2.model_path("0.4b", download=False) is None:
        pytest.skip("Sapiens2 0.4b weights not on disk")
    data = pytest.importorskip("skimage.data")
    picture = data.astronaut()  # a woman facing the camera, head and shoulders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = s2.load("0.4b", device)
    labels = bs.body_labels(bundle, picture)
    s2.offload(bundle)
    face, hair = labels == C["face_neck"], labels == C["hair"]
    assert face.mean() > 0.03 and hair.mean() > 0.03
    assert np.nonzero(face)[0].mean() < picture.shape[0] * 0.6  # the face is in the upper part


def test_vietnamese_limbs_are_whole_and_kept_apart_from_the_english_word(monkeypatch):
    (mask, _, _, _, report), labels, _, _ = _run(monkeypatch, "tay - cánh tay")
    assert (mask[0].numpy() > 0.5).tolist() == np.isin(labels, list(bs.HANDS)).tolist()  # the hands: none here
    (mask, _, _, _, report), labels, _, _ = _run(monkeypatch, "chân")
    assert (mask[0].numpy() > 0.5).tolist() == np.isin(labels, list(bs.LEGS | bs.FEET)).tolist()
    assert "chân: the whole limb, feet included" in report and "chân ->" not in report
    assert bs.WHOLE_LIMBS["tay trái"] == {C["left_lower_arm"], C["left_upper_arm"], C["left_hand"]}
    (mask, _, _, _, _), labels, _, _ = _run(monkeypatch, "tay trái")
    assert (mask[0].numpy() > 0.5).tolist() == np.isin(labels, list(bs.WHOLE_LIMBS["tay trái"])).tolist()


def test_nobody_found_is_said(monkeypatch):
    monkeypatch.setattr(bs, "find_faces", lambda picture, second_look=True: [])
    monkeypatch.setattr(bs.sapiens2, "load", lambda size, device: {"size": size})
    monkeypatch.setattr(bs.sapiens2, "offload", lambda bundle: None)
    monkeypatch.setattr(bs.sapiens2, "predict_labels", lambda bundle, picture, confidence=False: (
        np.zeros(picture.shape[:2], np.int64), np.full(picture.shape[:2], 0.9, np.float16)))
    drawn = []
    monkeypatch.setattr(bs, "preview_picture", lambda picture, frame, notes, mask=None: drawn.append(notes) or picture)
    monkeypatch.setattr(bs, "save_temp_previews", lambda pictures, prefix: [])
    labels, image = _picture()
    batch = torch.cat([image, torch.zeros_like(image)])
    out = bs.NH_BodySegment().segment(batch, "skin", "0.4b", 0, 0, False, "cpu", True)
    report = out["result"][4]
    assert f"{bs._NOBODY} in image(s) 1, 2" in report
    assert drawn[0][-1] == bs._NOBODY and drawn[1][-1] == bs._NOBODY  # each preview has its own notes


def test_batch_previews_get_their_own_picture_s_numbers(monkeypatch):
    drawn = []
    monkeypatch.setattr(bs, "preview_picture", lambda picture, frame, notes, mask=None: drawn.append(notes) or picture)
    monkeypatch.setattr(bs, "find_faces", lambda picture, second_look=True: [])
    monkeypatch.setattr(bs.sapiens2, "load", lambda size, device: {"size": size})
    monkeypatch.setattr(bs.sapiens2, "offload", lambda bundle: None)
    monkeypatch.setattr(bs.sapiens2, "predict_labels", lambda bundle, picture, confidence=False: (
        np.rint(picture[..., 0] / 8.0).astype(np.int64), np.full(picture.shape[:2], 0.9, np.float16)))
    monkeypatch.setattr(bs, "save_temp_previews", lambda pictures, prefix: [])
    labels, image = _picture()
    half = image.clone()
    half[:, :, : W // 2] = 0  # the left half of the picture is empty
    bs.NH_BodySegment().segment(torch.cat([image, half]), "hair", "0.4b", 0, 0, False, "cpu", True)
    whole = 100.0 * (labels == C["hair"]).sum() / labels.size
    cut = 100.0 * (labels[:, W // 2:] == C["hair"]).sum() / labels.size
    assert drawn[0][0] == f"hair: {whole:.1f}% of the picture" and drawn[1][0] == f"hair: {cut:.1f}% of the picture"


def _limb_labels():
    """A person with both arms, hands, legs and feet, as a Sapiens2 label map."""
    labels = np.zeros((H, W), np.int64)
    labels[5:15, 15:25] = C["face_neck"]
    labels[3:5, 15:25] = C["hair"]
    labels[15:30, 12:28] = C["torso"]
    for side, x in (("right", 4), ("left", 30)):
        labels[15:28, x:x + 6] = C[f"{side}_upper_arm"]
        labels[28:38, x:x + 6] = C[f"{side}_lower_arm"]
        labels[38:42, x:x + 6] = C[f"{side}_hand"]
    for side, x in (("left", 13), ("right", 21)):
        labels[30:42, x:x + 6] = C[f"{side}_upper_leg"]
        labels[42:52, x:x + 6] = C[f"{side}_lower_leg"]
        labels[52:56, x:x + 6] = C[f"{side}_foot"]
    return labels


def _run_on(monkeypatch, labels, regions):
    """Body Segment on a picture that parses to ``labels`` (the class id is written into the red channel)."""
    image = np.zeros((*labels.shape, 3), np.float32)
    image[..., 0] = labels * 8 / 255.0
    monkeypatch.setattr(bs, "find_faces", lambda picture, second_look=True: [])
    monkeypatch.setattr(bs.sapiens2, "load", lambda size, device: {"size": size})
    monkeypatch.setattr(bs.sapiens2, "offload", lambda bundle: None)
    monkeypatch.setattr(bs.sapiens2, "predict_labels", lambda bundle, picture, confidence=False: (
        np.rint(picture[..., 0] / 8.0).astype(np.int64), np.full(picture.shape[:2], 0.95, np.float16)))
    return bs.NH_BodySegment().segment(torch.from_numpy(image)[None], regions, "0.4b", 0, 0, False, "cpu", False)


def test_whole_limbs_on_a_picture_with_hands_and_feet(monkeypatch):
    labels = _limb_labels()
    for regions, ids in (("tay - cánh tay", bs.HANDS), ("chân", bs.LEGS | bs.FEET), ("tay", bs.ARMS | bs.HANDS),
                         ("tay trái", bs.WHOLE_LIMBS["tay trái"]), ("cánh tay", bs.ARMS), ("bàn chân", bs.FEET),
                         ("chân phải - bàn chân", bs.WHOLE_LIMBS["chân phải"] - bs.FEET)):
        wanted = np.isin(labels, list(ids))
        mask = _run_on(monkeypatch, labels, regions)[0]
        assert wanted.any() and (mask[0].numpy() > 0.5).tolist() == wanted.tolist(), regions


def test_compose_reads_body_words_on_body_segs(monkeypatch):
    segs_nodes = importlib.import_module("NH-Nodes.nodes.vision.segs_nodes")
    labels = _limb_labels()
    segs = _run_on(monkeypatch, labels, "skin")[3]
    compose = segs_nodes.NH_SegsCompose().compose
    for formula, ids in (("skin", bs.SKIN), ("da", bs.SKIN), ("arm", bs.ARMS), ("tay", bs.ARMS | bs.HANDS),
                         ("chân", bs.LEGS | bs.FEET), ("skin - face", bs.SKIN - bs.FACE), ("head", bs.FACE | bs.HAIR),
                         ("person", bs.PERSON), ("quần", frozenset()), ("váy", frozenset())):
        (mask,) = compose(segs, formula)
        assert (mask[0].numpy() > 0.5).tolist() == np.isin(labels, list(ids)).tolist(), formula


def test_compose_does_not_stand_body_parts_in_for_another_node_s_garments(monkeypatch):
    segs_nodes = importlib.import_module("NH-Nodes.nodes.vision.segs_nodes")
    vs = importlib.import_module("NH-Nodes.nodes.vision.vision_segment_nodes")
    (_, _, _, body, _), labels, _, _ = _run(monkeypatch, "skin")
    fashn = np.zeros((H, W), np.int64)
    fashn[25:45, 10:30], fashn[45:58, 12:28] = 3, 6  # FASHN: a top and pants, no dress or skirt
    garment = (vs.parts_frame(fashn, vs.FASHN_SEGS, (H, W), "fashn_human_parser"),)
    for formula in ("váy", "dress", "chân váy", "skirt"):
        (mask,) = segs_nodes.NH_SegsCompose().compose(body, formula, segs_2=garment)
        assert not mask.any(), formula
    (mask,) = segs_nodes.NH_SegsCompose().compose(body, "pants", segs_2=garment)
    assert (mask[0].numpy() > 0.5).tolist() == (fashn == 6).tolist()


def test_region_words_are_read_in_singular_or_plural():
    for word, region in (("lower leg", "lower legs"), ("upper leg", "upper legs"), ("accessory", "accessories"),
                         ("dresses", "dress"), ("Sunglass", "sunglasses"), ("calf", "calves")):
        assert bs.region_ids(word, word, "english") == bs.REGIONS[region], word
    assert bs.region_ids("cẳng chân", "lower leg", "translated") == bs.REGIONS["lower legs"]
    tree = bs.parse_concepts("cẳng chân, bắp tay, đùi")
    assert set(bs.check_regions(tree)) == {"lower leg", "upper arm", "thigh"}
    with pytest.raises(ValueError, match="part of 'face'"):
        bs.check_regions(bs.parse_concepts("eyebrow"))


def test_class_notes_only_for_words_the_picture_has_and_the_beard_advice():
    counts = np.zeros(len(s2.CLASSES), np.int64)
    assert bs.describe(bs.parse_concepts("skin, dress"), counts, 1000) == [
        "skin: not in the picture", "dress: not in the picture"]
    counts[C["face_neck"]] = 10
    assert bs._FACE_CLASS in bs.describe(bs.parse_concepts("skin"), counts, 1000)
    with pytest.raises(ValueError, match="no NH-Nodes parser has a beard class"):
        bs.check_regions(bs.parse_concepts("râu"))


def test_compose_on_body_segs_gives_no_wrong_advice_and_reads_any_accent_form(monkeypatch, capsys):
    import unicodedata

    segs_nodes = importlib.import_module("NH-Nodes.nodes.vision.segs_nodes")
    labels = _limb_labels()
    segs = _run_on(monkeypatch, labels, "skin")[3]
    compose = segs_nodes.NH_SegsCompose().compose
    for formula in ("da mặt", "face skin"):  # not a word of Body's segs: empty, with the labels it holds
        (mask,) = compose(segs, formula)
        assert not mask.any() and "The segs hold: arms, face, feet, hair, hands, legs, torso" in capsys.readouterr().out
    (mask,) = compose(segs, unicodedata.normalize("NFD", "chân"))  # some keyboards send the accents apart
    assert (mask[0].numpy() > 0.5).tolist() == np.isin(labels, list(bs.LEGS | bs.FEET)).tolist()
    (mask,) = compose(segs, "upper clothes, apparel")
    assert not mask.any()  # Body's exact words, found nowhere in this picture, and no error


def test_real_sapiens2_finds_nobody_on_pictures_without_people():
    if s2.model_path("0.4b", download=False) is None:
        pytest.skip("Sapiens2 0.4b weights not on disk")
    data = pytest.importorskip("skimage.data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = s2.load("0.4b", device)
    try:
        for name in ("rocket", "coffee", "chelsea"):  # a launch pad, a cup of coffee, a cat
            assert not bs.body_labels(bundle, np.ascontiguousarray(getattr(data, name)()[..., :3])).any(), name
    finally:
        s2.offload(bundle)
