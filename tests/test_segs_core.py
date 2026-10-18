"""NH_SEGS data model and the select / compose / convert / preview helpers (no model needed)."""

import importlib
import re

import cv2
import numpy as np
import pytest
import torch

segs = importlib.import_module("NH-Nodes.core.segs")
segs_nodes = importlib.import_module("NH-Nodes.nodes.vision.segs_nodes")


def _seg(label, score, box, height=40, width=60):
    """Segment filling ``box`` (x1, y1, x2, y2) of a height x width image."""
    mask = torch.zeros((height, width), dtype=torch.bool)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = True
    return segs.make_segment(label, score, mask)


def _frame(*segments, height=40, width=60):
    return segs.SegsFrame(height, width, tuple(segments))


# --- data model --------------------------------------------------------------
def test_make_segment_keeps_a_compact_tight_crop_with_origin_offset():
    mask = torch.zeros((10, 12), dtype=torch.bool)
    mask[2:5, 3:7] = True
    seg = segs.make_segment("car", 0.9, mask, origin=(100, 50))
    assert seg.box == (103, 52, 107, 55)
    assert tuple(seg.mask.shape) == (3, 4) and seg.mask.all()
    assert seg.area == 12 and seg.center == (105.0, 53.5)
    # a compact copy, not a view keeping the big mask alive
    assert seg.mask.untyped_storage().nbytes() == seg.mask.numel()


def test_make_segment_thresholds_float_masks_and_returns_none_when_empty():
    assert segs.make_segment("x", 1.0, torch.zeros((4, 4))) is None
    soft = torch.tensor([[0.2, 0.6], [0.5, 0.9]])
    seg = segs.make_segment("x", 1.0, soft)
    assert seg.box == (1, 0, 2, 2) and seg.mask.tolist() == [[True], [True]]  # values above 0.5, tight box
    assert segs.make_segment("x", 1.0, np.array([[0, 1]], dtype=bool)).box == (1, 0, 2, 1)


def test_union_and_instance_masks():
    frame = _frame(_seg("a", 0.9, (0, 0, 10, 10)), _seg("b", 0.8, (5, 5, 20, 15)))
    union = segs.segs_union((frame, frame))
    assert union.shape == (2, 40, 60) and union.dtype == torch.float32
    assert int(union[0].sum()) == 100 + 150 - 25
    masks, labels = segs.instance_masks((frame,))
    assert masks.shape == (2, 40, 60) and labels == ["a", "b"]
    empty, none = segs.instance_masks((_frame(),))
    assert empty.shape == (1, 40, 60) and float(empty.sum()) == 0 and none == []


def test_parse_label_list_splits_trims_dedupes_and_skips_comments():
    text = "person, Red  car\n# a comment\n\nwindow,person ,red_car"
    assert segs.parse_label_list(text) == ["person", "Red car", "window"]
    assert segs.parse_label_list("") == [] and segs.parse_label_list(None) == []


def test_label_matching_ignores_case_spacing_and_supports_wildcards():
    assert segs.label_matches("Red Car", "red_car")
    assert segs.label_matches("t-shirt", "*shirt")
    assert segs.label_matches("person", "pers?n")
    assert segs.label_matches("person", "persons") and not segs.label_matches("person", "personal")


# --- select ------------------------------------------------------------------
def test_select_filters_by_label_score_and_area():
    frame = _frame(_seg("person", 0.9, (0, 0, 10, 10)), _seg("person", 0.4, (20, 0, 30, 10)),
                   _seg("car", 0.95, (0, 20, 60, 40)))
    picked = segs.select_segments(frame, ["person"], min_score=0.5)
    assert [s.score for s in picked.segments] == [0.9]
    big = segs.select_segments(frame, min_area=0.2)  # car = 1200 / 2400 px
    assert [s.label for s in big.segments] == ["car"]
    small = segs.select_segments(frame, max_area=0.1)
    assert sorted(s.score for s in small.segments) == [0.4, 0.9]


@pytest.mark.parametrize("order, expected", [
    ("score", ["c", "a", "b"]), ("area", ["b", "a", "c"]), ("left_to_right", ["a", "c", "b"]),
    ("right_to_left", ["b", "c", "a"]), ("top_to_bottom", ["c", "a", "b"]), ("bottom_to_top", ["b", "a", "c"]),
    ("center_first", ["c", "b", "a"]), ("as_found", ["a", "b", "c"]),
])
def test_sort_orders(order, expected):
    frame = _frame(_seg("a", 0.5, (0, 10, 6, 16)), _seg("b", 0.2, (40, 20, 60, 40)), _seg("c", 0.9, (28, 0, 32, 4)))
    assert [s.label for s in segs.select_segments(frame, order=order).segments] == expected


def test_select_start_and_count_slice_after_sorting():
    frame = _frame(*[_seg(f"p{i}", 0.5, (i * 10, 0, i * 10 + 5, 5)) for i in range(5)])
    second = segs.select_segments(frame, order="left_to_right", start=1, count=1)
    assert [s.label for s in second.segments] == ["p1"]
    rest = segs.select_segments(frame, order="right_to_left", start=3, count=0)
    assert [s.label for s in rest.segments] == ["p1", "p0"]
    assert segs.select_segments(frame, start=9).segments == ()


# --- compose -----------------------------------------------------------------
def _compose_frame():
    return _frame(_seg("person", 0.9, (0, 0, 30, 40)), _seg("hair", 0.8, (0, 0, 30, 10)),
                  _seg("red car", 0.7, (40, 30, 60, 40)), _seg("t-shirt", 0.6, (0, 10, 30, 25)))


def _px(expression, frame=None):
    return int(segs.evaluate(segs.parse_expression(expression), frame or _compose_frame()).sum())


def test_compose_operators_and_keywords():
    assert _px("person") == 1200
    assert _px("person - hair") == 900
    assert _px("person | 'red car'") == _px("person + \"red car\"") == _px("person or 'red car'") == 1400
    assert _px("person & hair") == _px("person and hair") == 300
    assert _px("~person") == _px("!person") == _px("not person") == 2400 - 1200
    assert _px("all") == 1400 and _px("image") == 2400 and _px("none") == 0
    assert _px("'t-shirt'") == 450 and _px("*shirt") == 450 and _px("red_car") == 200
    assert _px("unknown") == 0


def test_compose_hyphens_and_apostrophes_inside_words_belong_to_the_label():
    assert _px("t-shirt") == 450  # no quotes needed
    assert _px("person - hair") == _px("person -hair") == _px("person- hair") == 900
    assert _px("person-hair") == 0  # one (unknown) label: subtracting needs spaces
    frame = _frame(_seg("women's dress", 0.9, (0, 0, 10, 10)))
    assert _px("\"women's dress\"", frame) == 100 and _px("women's_dress", frame) == 100


def test_compose_precedence_and_associativity():
    # & binds tighter than | and -, which read left to right
    assert _px("hair | person & 'red car'") == _px("hair | (person & 'red car')") == 300
    assert _px("person - hair - 't-shirt'") == 1200 - 300 - 450
    assert _px("person - (hair - 't-shirt')") == 900
    assert _px("~(person | 'red car')") == 1000


@pytest.mark.parametrize("expression, message", [
    ("", "empty"), ("   ", "empty"), (" , \n # only a comment", "empty"), ("'red car", "never closed"),
    ("(person | hair", "not closed"), ("person |", "ends where"), (")", "unexpected ')'"),
    ("person !", "unexpected '!' at position 8"), ("(person or)", "unexpected ')' at position 11"),
    ("person | , hair", "unexpected ','"), ("()", "holds nothing"), ("hair (person)", "unexpected '(' at position 6"),
])
def test_compose_errors_explain_where(expression, message):
    with pytest.raises(segs.ExpressionError, match=re.escape(message)):
        segs.parse_expression(expression)


def test_lists_comments_and_words_in_a_row():
    # commas and new lines join like |, after every other operator; words in a row are one label
    assert _px("person, red car") == _px("person\nred car") == _px("person | 'red car'") == 1400
    assert _px("red car") == _px("Red  Car") == 200 and _px("person hair") == 0  # one (unknown) label
    assert _px("person - hair, red car") == 900 + 200
    assert _px("(hair, red car) - person") == 200 and _px("person,,\n\nhair,") == 1200
    assert _px("hair # the hair only\n# person\n") == 300


def test_background_is_everything_outside_the_instances():
    assert _px("background") == 2400 - 1400 == _px("~all") and _px("~background") == _px("all")
    assert _px("background - red car") == 1000 and _px("background | hair") == 1300


def test_vietnamese_labels_and_keywords():
    frame = _frame(_seg("hair", 0.8, (0, 0, 30, 10)), _seg("bag", 0.7, (40, 30, 60, 40)))
    assert _px("tóc", frame) == 300 and _px("tóc, túi", frame) == 500 and _px("nền", frame) == 2400 - 500
    assert _px("tất cả - túi", frame) == 300 and _px("'nền'", frame) == 0  # quoted: a label, never a keyword
    tree = segs.parse_expression("tóc - túi, nền, áo xanh, sky")
    assert segs.expression_labels(tree) == ["hair", "bag", "áo xanh", "sky"]
    assert segs.expression_terms(tree) == [("tóc", "hair", "translated"), ("túi", "bag", "translated"),
                                           ("nền", "background", "keyword"), ("áo xanh", "áo xanh", "unknown"),
                                           ("sky", "sky", "english")]
    assert segs.uses_keyword(tree, "background") and not segs.uses_keyword(tree, "all")


def test_sam3_concepts_read_and_or_not_as_words():
    tree = segs.parse_concepts("black and white dog, person - bag")
    assert segs.expression_labels(tree) == ["black and white dog", "person", "bag"]
    assert segs.expression_labels(segs.parse_expression("person and hair")) == ["person", "hair"]


def test_compose_node_reports_unmatched_labels(capsys):
    (mask,) = segs_nodes.NH_SegsCompose().compose((_compose_frame(),), "person - ghost")
    assert mask.shape == (1, 40, 60) and int(mask.sum()) == 1200
    assert "'ghost'" in capsys.readouterr().out


def test_compose_node_catches_a_minus_without_spaces():
    with pytest.raises(ValueError, match="person - hair"):
        segs_nodes.NH_SegsCompose().compose((_compose_frame(),), "person-hair")
    # a real hyphenated label, or parts that are not both labels, stay plain labels
    assert int(segs_nodes.NH_SegsCompose().compose((_compose_frame(),), "t-shirt")[0].sum()) == 450
    assert int(segs_nodes.NH_SegsCompose().compose((_compose_frame(),), "person-ghost")[0].sum()) == 0


def test_preview_picture_shrinks_darkens_outside_the_mask_and_adds_notes():
    frame = segs.SegsFrame(400, 600, (_seg("car", 0.9, (300, 200, 500, 380), 400, 600),))
    image = torch.full((400, 600, 3), 0.8)
    mask = torch.zeros((400, 600))
    mask[200:380, 300:500] = 1.0
    picture = segs.preview_picture(image, frame, ["car: 1 found"], max_side=300, mask=mask)
    assert picture.dtype == np.uint8 and picture.shape[1] == 300 and picture.shape[0] > 200  # caption strip below
    assert abs(int(picture[20, 20, 0]) - round(204 * 0.35)) <= 1  # outside the mask: darkened
    assert int(picture[150, 200].min()) > 140  # inside: bright, lightly tinted
    plain = segs.preview_picture(image, frame)
    assert plain.shape == (400, 600, 3) and int(plain[20, 20, 0]) == 204  # no mask, no notes: as Segs Preview


def test_scale_frame_keeps_tiny_instances_visible():
    frame = segs.SegsFrame(400, 600, (_seg("dot", 0.9, (10, 10, 12, 12), 400, 600),
                                      _seg("car", 0.8, (300, 200, 500, 380), 400, 600)))
    small = segs.scale_frame(frame, 0.1)
    assert (small.height, small.width) == (40, 60) and [s.label for s in small.segments] == ["dot", "car"]
    assert small.segments[0].area == 1 and small.segments[1].box == (30, 20, 50, 38)


# --- Impact Pack SEGS --------------------------------------------------------
@pytest.mark.parametrize("box, factor, expected", [
    ((20, 10, 30, 20), 3.0, [10, 0, 40, 30]),  # 30 x 30 centred on the box
    ((0, 0, 10, 10), 3.0, [0, 0, 30, 30]),  # slid right/down to stay inside
    ((50, 30, 60, 40), 2.0, [40, 20, 60, 40]),  # slid left/up
    ((5, 5, 55, 35), 3.0, [0, 0, 60, 40]),  # larger than the image: the whole image
    ((11, 13, 12, 14), 1.0, [11, 13, 12, 14]),  # factor 1: the box itself
])
def test_detail_crop_region(box, factor, expected):
    assert segs.detail_crop_region(60, 40, box, factor) == expected


def test_detail_crop_region_always_contains_the_box_and_stays_inside():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        width, height = int(rng.integers(1, 300)), int(rng.integers(1, 300))
        x1, y1 = int(rng.integers(0, width)), int(rng.integers(0, height))
        x2, y2 = int(rng.integers(x1 + 1, width + 1)), int(rng.integers(y1 + 1, height + 1))
        factor = float(rng.uniform(1.0, 6.0))
        rx1, ry1, rx2, ry2 = segs.detail_crop_region(width, height, (x1, y1, x2, y2), factor)
        assert 0 <= rx1 <= x1 and 0 <= ry1 <= y1 and x2 <= rx2 <= width and y2 <= ry2 <= height


def test_to_impact_segs_places_each_mask_in_its_crop_region():
    frame = _frame(_seg("person", 0.9, (20, 10, 30, 20)))
    (height, width), items = segs.to_impact_segs(frame, 3.0)
    assert (height, width) == (40, 60) and len(items) == 1
    item = items[0]
    assert item._fields == ("cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label",
                            "control_net_wrapper")
    rx1, ry1, rx2, ry2 = item.crop_region
    assert item.cropped_mask.dtype == np.float32 and item.cropped_mask.shape == (ry2 - ry1, rx2 - rx1)
    full = np.zeros((40, 60), np.float32)
    full[ry1:ry2, rx1:rx2] = item.cropped_mask
    assert full[10:20, 20:30].all() and full.sum() == 100
    assert item.bbox == [20, 10, 30, 20] and item.label == "person" and item.confidence == pytest.approx(0.9)
    assert item.cropped_image is None


def test_preview_labels_stack_instead_of_hiding_each_other():
    placed = [(30, 10, 100, 30)]
    assert segs._free_spot(40, 10, 70, 20, placed, 200) == 31  # moved below the label already there
    assert segs._free_spot(40, 185, 70, 20, [(30, 180, 100, 200)], 200) == 159  # near the bottom: moved above
    assert segs._free_spot(200, 10, 70, 20, placed, 200) == 10  # no overlap: stays
    frame = _frame(_seg("person", 0.9, (10, 10, 50, 40)), _seg("hair", 0.8, (10, 10, 50, 20)), height=200, width=200)
    assert segs.render_preview(np.full((200, 200, 3), 128, np.uint8), frame).shape == (200, 200, 3)


def test_impact_node_checks_batch_index():
    node = segs_nodes.NH_SegsToImpactSEGS()
    frame = _frame(_seg("a", 0.9, (0, 0, 5, 5)))
    assert node.convert((frame,), 3.0, 0)[0][0] == (40, 60)
    with pytest.raises(ValueError, match="batch_index 1"):
        node.convert((frame,), 3.0, 1)


# --- preview and nodes -------------------------------------------------------
def test_render_preview_tints_only_instances_and_handles_any_label_text():
    image = np.full((40, 60, 3), 128, np.uint8)
    frame = _frame(_seg("áo dài", 0.91, (10, 10, 30, 30)))
    out = segs.render_preview(image, frame)
    assert out.shape == image.shape and out.dtype == np.uint8
    assert (out[20, 20] != 128).any()  # inside: tinted
    assert (out[38, 58] == 128).all()  # far outside: untouched
    assert (image == 128).all()  # the input is not modified
    assert (segs.render_preview(image, _frame()) == image).all()


def test_preview_batch_rejects_segs_from_another_image():
    frame = _frame(_seg("a", 0.9, (0, 0, 5, 5)))
    assert segs.preview_batch(torch.rand(1, 40, 60, 3), (frame,)).shape == (1, 40, 60, 3)
    assert segs.preview_batch(torch.rand(1, 40, 60, 1), (frame,)).shape == (1, 40, 60, 3)
    with pytest.raises(ValueError, match="batch"):
        segs.preview_batch(torch.rand(2, 40, 60, 3), (frame,))
    with pytest.raises(ValueError, match="60x40"):
        segs.preview_batch(torch.rand(1, 20, 30, 3), (frame,))


def test_select_and_to_masks_nodes():
    frame = _frame(_seg("person", 0.9, (0, 0, 10, 10)), _seg("person", 0.8, (40, 0, 50, 10)),
                   _seg("dog", 0.7, (20, 20, 30, 30)))
    picked, mask, count = segs_nodes.NH_SegsSelect().select((frame,), "person", 0.0, 0.0, 1.0, "right_to_left", 0, 1)
    assert count == 1 and picked[0].segments[0].box == (40, 0, 50, 10)
    assert mask.shape == (1, 40, 60) and int(mask.sum()) == 100
    masks, union, labels, n = segs_nodes.NH_SegsToMasks().convert((frame,))
    assert masks.shape == (3, 40, 60) and union.shape == (1, 40, 60)
    assert labels == "person\nperson\ndog" and n == 3


def test_to_masks_refuses_when_ram_is_short(monkeypatch):
    import psutil

    frame = _frame(*[_seg("window", 0.9, (i, 0, i + 2, 2)) for i in range(0, 20, 4)])
    monkeypatch.setattr(psutil, "virtual_memory", lambda: type("Memory", (), {"available": 4 * 40 * 60 * 4})())
    with pytest.raises(MemoryError, match="Segs Select"):
        segs_nodes.NH_SegsToMasks().convert((frame,))
    monkeypatch.setattr(psutil, "virtual_memory", lambda: type("Memory", (), {"available": 1 << 30})())
    assert segs_nodes.NH_SegsToMasks().convert((frame,))[0].shape == (5, 40, 60)


def test_frames_tell_comfyui_cache_what_they_hold():
    frame = _frame(_seg("a", 0.9, (0, 0, 10, 10)), _seg("b", 0.8, (5, 5, 20, 15)))
    tensors = frame._comfy_cache_tensors()
    assert len(tensors) == 2 and all(t is s.mask for t, s in zip(tensors, frame.segments))


def test_segs_preview_node_and_every_node_declares_matching_outputs():
    frame = _frame(_seg("a", 0.9, (0, 0, 5, 5)))
    (preview,) = segs_nodes.NH_SegsPreview().preview(torch.rand(1, 40, 60, 3), (frame,))
    assert preview.shape == (1, 40, 60, 3) and preview.dtype == torch.float32
    sam3_nodes = importlib.import_module("NH-Nodes.nodes.vision.sam3_nodes")
    classes = [*segs_nodes.NODE_CLASS_MAPPINGS.values(), *sam3_nodes.NODE_CLASS_MAPPINGS.values()]
    for cls in classes:
        required = cls.INPUT_TYPES()["required"]
        assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES) and cls.CATEGORY == "NH-Nodes/Vision"
        assert callable(getattr(cls(), cls.FUNCTION)) and required
        assert all(isinstance(spec, tuple) and spec for spec in required.values())


def test_formulas_laid_out_over_several_lines_still_read():
    # a line break next to an operator joins (as on 1.7), a line break between two labels lists them
    assert _px("person\n- hair") == _px("person -\nhair") == _px("person\r\n- hair") == 900
    assert _px("(person | 'red car')\n- hair") == 1100 and _px("~\nperson") == 1200
    assert _px("hair\n| 'red car'") == 500 and _px("person &\nhair") == 300
    assert _px("(\nperson\n)") == 1200 and _px("hair\n\n- hair") == 0
    assert _px("person\nhair") == 1200 and _px("hair\n~person") == 300 + 1200  # a list; ~person is an item


def test_comments_start_only_where_a_word_could():
    frame = _frame(_seg("car#2", 0.9, (0, 0, 10, 10)), _seg("car", 0.8, (20, 20, 30, 30)))
    assert _px("car#2", frame) == 100 and _px("car # the second\n", frame) == 100
    with pytest.raises(segs.EmptyExpressionError):
        segs.parse_expression("# nothing but a comment")
    with pytest.raises(segs.ExpressionError) as caught:
        segs.parse_expression("(person) empty")
    assert not isinstance(caught.value, segs.EmptyExpressionError)


def test_report_trusts_what_was_found_over_the_dictionary():
    frame = _frame(_seg("café", 0.9, (0, 0, 10, 10)))
    lines = segs.describe_request(segs.parse_concepts("café, bánh xèo"), (frame,), 0.5)
    assert lines[0] == "café: 1 found (best 0.90)"  # not Vietnamese after all: SAM 3 found it
    assert lines[1].startswith("bánh xèo: not understood")


def test_compose_node_catches_words_left_without_an_operator():
    with pytest.raises(ValueError, match=re.escape("person + hair")):
        segs_nodes.NH_SegsCompose().compose((_compose_frame(),), "person hair")
    assert int(segs_nodes.NH_SegsCompose().compose((_compose_frame(),), "red car")[0].sum()) == 200
    assert int(segs_nodes.NH_SegsCompose().compose((_compose_frame(),), "person ghost")[0].sum()) == 0


def test_segs_select_reads_vietnamese_labels_but_not_formula_words():
    frame = _frame(_seg("hair", 0.9, (0, 0, 10, 10)), _seg("all", 0.8, (20, 20, 30, 30)))
    picked, _, count = segs_nodes.NH_SegsSelect().select((frame,), "tóc", 0.0, 0.0, 1.0, "score", 0, 0)
    assert count == 1 and picked[0].segments[0].label == "hair"
    picked, _, count = segs_nodes.NH_SegsSelect().select((frame,), "tất cả", 0.0, 0.0, 1.0, "score", 0, 0)
    assert count == 0  # "tất cả" is the formula word all, not a label named "all"


def test_merge_segs_mixes_instances_of_the_same_pictures():
    a = (_frame(_seg("person", 0.9, (0, 0, 10, 10))),)
    b = (_frame(_seg("dress", 1.0, (2, 2, 8, 8))),)
    merged = segs.merge_segs(a, None, b)
    assert [seg.label for seg in merged[0].segments] == ["person", "dress"] and segs.merge_segs(a) is a
    with pytest.raises(ValueError, match="different pictures"):
        segs.merge_segs(a, (_frame(height=20, width=20),))
    (mask,) = segs_nodes.NH_SegsCompose().compose(a, "person - dress", segs_2=b)
    assert int(mask.sum()) == 100 - 36


def test_picture_prints_tell_photos_apart_but_not_edits_of_one():
    rng = np.random.default_rng(3)
    base = torch.from_numpy(cv2.GaussianBlur(rng.random((90, 120, 3)).astype(np.float32), (0, 0), 6))
    other = torch.from_numpy(cv2.GaussianBlur(rng.random((90, 120, 3)).astype(np.float32), (0, 0), 6))
    first = segs.picture_print(base)
    assert first.shape == (segs.PRINT_SIDE, segs.PRINT_SIDE) and first.dtype == np.float32
    for edited in (base * 1.3 + 0.1, base ** 2.0, base.mean(dim=2, keepdim=True).expand(-1, -1, 3),
                   torch.cat([base, torch.ones(90, 120, 1)], dim=2)):  # brighter, darker, grey, with alpha
        assert segs.same_picture(first, segs.picture_print(edited))
    assert not segs.same_picture(first, segs.picture_print(other))
    assert not segs.same_picture(first, segs.picture_print(base.flip(1)))  # a mirrored copy would not line up
    flat = segs.picture_print(torch.full((90, 120, 3), 0.5))
    assert segs.same_picture(flat, first) and segs.same_picture(first, flat)  # a blank picture tells nothing
    # numpy pictures (uint8 or float) give the same print up to scale
    assert segs.same_picture(first, segs.picture_print((base.numpy() * 255).astype(np.uint8)))


def test_merge_segs_refuses_segs_of_different_photos_of_the_same_size():
    rng = np.random.default_rng(4)
    one, two = (torch.from_numpy(cv2.GaussianBlur(rng.random((40, 60, 3)).astype(np.float32), (0, 0), 4))[None]
                for _ in range(2))
    a = segs.stamped((_frame(_seg("person", 0.9, (0, 0, 10, 10))),), one)
    b = segs.stamped((_frame(_seg("dress", 1.0, (2, 2, 8, 8))),), one * 0.8)  # the same photo, darker
    c = segs.stamped((_frame(_seg("bag", 1.0, (4, 4, 6, 6))),), two)
    assert a[0].picture is not None and a[0].with_segments([]).picture is a[0].picture
    assert segs.scale_frame(a[0], 0.5).picture is a[0].picture
    merged = segs.merge_segs(a, b, None)
    assert [seg.label for seg in merged[0].segments] == ["person", "dress"]
    with pytest.raises(ValueError, match="different photos of the same size"):
        segs.merge_segs(a, b, c)
    with pytest.raises(ValueError, match="different photos"):
        segs_nodes.NH_SegsCompose().compose(a, "person - bag", segs_2=c)
    # frames without a print (made by older nodes) are mixed on their size alone
    assert len(segs.merge_segs((_frame(_seg("cat", 1.0, (0, 0, 2, 2))),), c)[0].segments) == 2
    # batches: every image is checked, and the message says which one differs
    batch_a = segs.stamped(a + a, torch.cat([one, two]))
    batch_b = segs.stamped(b + b, torch.cat([one, one]))
    with pytest.raises(ValueError, match="image 2 of the batch"):
        segs.merge_segs(batch_a, batch_b)
    batch_c = segs.stamped(c + c, torch.cat([one, two]))
    merged = segs.merge_segs(batch_a, None, batch_c)
    assert [len(frame.segments) for frame in merged] == [2, 2]


def test_picture_prints_take_any_image_dtype_and_keep_every_frame():
    picture = torch.rand(20, 30, 3)
    first = segs.picture_print(picture)
    for other in (picture.half(), picture.bfloat16(), picture.double().numpy(), picture.numpy().astype(np.float16)):
        assert segs.same_picture(first, segs.picture_print(other))
    frames = (_frame(height=20, width=30), _frame(height=20, width=30))
    assert segs.stamped(frames, picture[None]) == frames  # one picture for two frames: left as they were


def test_labels_match_singular_and_plural_but_wildcards_as_written():
    for label, pattern in (("arms", "arm"), ("arm", "Arms"), ("feet", "foot"), ("teeth", "tooth"),
                           ("accessories", "accessory"), ("shoes", "shoe"), ("dresses", "dress"), ("red cars", "red car")):
        assert segs.label_matches(label, pattern) and segs.label_matches(pattern, label)
    assert not segs.label_matches("dress", "dres") and not segs.label_matches("bus", "bu")
    assert segs.label_matches("cars", "car*") and not segs.label_matches("car", "cars*")  # a wildcard is as written


def test_a_node_s_own_words_name_the_parts_of_its_segs():
    segs.add_aliases("test-parser", {"skin": {1, 2, 3}, "body": {1, 2, 3, 4}, "nothing": {9}},
                     {"face": {1}, "arms": {2, 3}, "top": {4}})
    face = segs.make_segment("face", 1.0, torch.ones(2, 2, dtype=torch.bool), (0, 0), "test-parser")
    arms = segs.make_segment("arms", 1.0, torch.ones(2, 2, dtype=torch.bool), (4, 0), "test-parser")
    top = segs.make_segment("top", 1.0, torch.ones(2, 2, dtype=torch.bool), (8, 0), "test-parser")
    other = segs.make_segment("arms", 1.0, torch.ones(2, 2, dtype=torch.bool), (0, 4), "sam3")
    frame = segs.SegsFrame(8, 12, (face, arms, top, other))
    assert segs.matching(frame.segments, "skin") == [face, arms]  # only that node's parts
    assert segs.matching(frame.segments, "arm") == [arms, other]  # a label is found on every node's instances
    assert segs.matching(frame.segments, "bodies") == [face, arms, top] and segs.matching(frame.segments, "nothing") == []
    labelled_skin = segs.make_segment("skin", 0.9, torch.ones(2, 2, dtype=torch.bool), (0, 6), "sam3")
    assert segs.matching(frame.segments + (labelled_skin,), "skin") == [labelled_skin]  # an instance's own label wins
    assert int(segs.evaluate(segs.parse_expression("skin - face"), frame).sum()) == 4
    (picked,) = segs.select_segments(frame, labels=("skin",)).segments[1:]
    assert picked is arms


def test_the_photo_check_can_be_turned_off_and_a_nan_does_not_trip_it():
    rng = np.random.default_rng(5)
    one, two = (torch.from_numpy(cv2.GaussianBlur(rng.random((40, 60, 3)).astype(np.float32), (0, 0), 4))[None]
                for _ in range(2))
    a = segs.stamped((_frame(_seg("person", 0.9, (0, 0, 10, 10))),), one)
    c = segs.stamped((_frame(_seg("bag", 1.0, (4, 4, 6, 6))),), two)
    with pytest.raises(ValueError, match="turn check_same_photo off"):
        segs_nodes.NH_SegsCompose().compose(a, "person - bag", segs_2=c)
    (mask,) = segs_nodes.NH_SegsCompose().compose(a, "person - bag", segs_2=c, check_same_photo=False)
    assert int(mask.sum()) == 100 - 4
    spoiled = one.clone()
    spoiled[0, 5, 5, 1] = float("nan")
    assert segs.same_picture(a[0].picture, segs.picture_print(spoiled[0]))
    optional = segs_nodes.NH_SegsCompose.INPUT_TYPES()["optional"]
    assert optional["check_same_photo"][1]["default"] is True
