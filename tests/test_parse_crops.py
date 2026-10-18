"""Crops for the parsers (faces, clothes) and YuNet face boxes, without models."""

import importlib

import numpy as np
import pytest

crops = importlib.import_module("NH-Nodes.core.parse_crops")
faces = importlib.import_module("NH-Nodes.core.face_detect")


def test_square_around_scales_the_longer_side_and_lifts():
    assert crops.square_around((40, 50, 60, 80), 2.0) == (20, 35, 80, 95)  # 30 px tall -> 60 px square
    assert crops.square_around((40, 50, 60, 80), 2.0, lift=0.1) == (20, 29, 80, 89)  # up by 0.1 * 60
    assert crops.square_around((0, 0, 0, 0), 3.0) == (0, 0, 1, 1)  # never empty


@pytest.mark.parametrize("box, aspect", [((100, 100, 140, 300), 1.0), ((10, 10, 50, 30), 2 / 3),
                                         ((500, 20, 590, 390), 0.75), ((0, 0, 600, 400), 1.0)])
def test_fit_box_has_the_model_shape_and_holds_the_box(box, aspect):
    region = crops.fit_box(box, aspect, 0.08, (400, 600))
    x1, y1, x2, y2 = region
    assert x1 <= box[0] and y1 <= box[1] and x2 >= box[2] and y2 >= box[3]
    assert abs((x2 - x1) / (y2 - y1) - aspect) < 0.02
    if (x2 - x1) <= 600:
        assert 0 <= x1 and x2 <= 600  # slid inside where it fits
    if (y2 - y1) <= 400:
        assert 0 <= y1 and y2 <= 400


def test_cut_padded_pads_the_outside_with_grey():
    image = np.full((10, 20, 3), 200, np.uint8)
    crop = crops.cut_padded(image, (-5, 5, 10, 15))
    assert crop.shape == (10, 15, 3)
    assert (crop[:5, 5:] == 200).all() and (crop[:, :5] == crops.PAD_VALUE).all() and (crop[5:] == crops.PAD_VALUE).all()
    assert crops.visible((-5, 5, 10, 15), (10, 20)) == (0, 5, 10, 10)


def test_nearest_owner_splits_overlapping_crops_between_their_faces():
    boxes = [(0, 0, 10, 10), (20, 0, 30, 10)]  # face centres at x = 5 and x = 25
    regions = [(0, 0, 30, 10), (0, 0, 30, 10)]
    left = crops.nearest_owner((0, 0, 30, 10), boxes, regions, 0)
    right = crops.nearest_owner((0, 0, 30, 10), boxes, regions, 1)
    assert left[:, :15].all() and not left[:, 15:].any() and (left ^ right).all()
    # a face whose crop does not reach a pixel cannot take it
    alone = crops.nearest_owner((0, 0, 30, 10), boxes, [(0, 0, 30, 10), (20, 0, 30, 10)], 0)
    assert alone[:, :20].all() and not alone[:, 20:].any()


def test_cover_box_uses_a_row_of_crops_for_a_wide_group():
    group = (100, 300, 7300, 4800)  # five people side by side on a 7360 x 4912 photo
    regions = crops.cover_box(group, 2 / 3, 0.0, (4912, 7360))
    assert len(regions) >= 3
    for x1, y1, x2, y2 in regions:
        assert abs((x2 - x1) / (y2 - y1) - 2 / 3) < 0.02 and y1 <= 300 and y2 >= 4800
    assert regions[0][0] <= 100 and regions[-1][2] >= 7300  # together they cover the group
    assert all(a[2] > b[0] for a, b in zip(regions, regions[1:]))  # and overlap
    assert crops.cover_box((100, 100, 300, 500), 2 / 3, 0.08, (4912, 7360)) == [
        crops.fit_box((100, 100, 300, 500), 2 / 3, 0.08, (4912, 7360))]  # one person: one crop
    assert len(crops.cover_box((100, 0, 200, 4900), 1.0, 0.0, (4912, 7360))) == 1  # tall: never split


def test_square_frame_pads_the_short_side():
    assert crops.square_frame((100, 200)) == (0, -50, 200, 150)
    assert crops.square_frame((300, 100)) == (-100, 0, 200, 300)


def _people_picture():
    """300 x 900 picture: a wide group (red 40) in rows 100-280, columns 50-850; a plain backdrop (red 200) elsewhere."""
    image = np.full((300, 900, 3), 200, np.uint8)
    image[100:280, 50:850, 0] = 40
    return image


def test_people_labels_refines_the_first_look_and_keeps_it_elsewhere():
    image = _people_picture()
    shown = []

    def predict(picture):
        shown.append(picture.shape[:2])
        person = picture[..., 0] == 40
        if picture.shape[:2] == (300, 900):  # the first look: right kind of thing, wrong label, unsure
            return np.where(person, 3, 0).astype(np.int64), np.full(picture.shape[:2], 0.6, np.float16)
        return np.where(person, 5, 0).astype(np.int64), np.full(picture.shape[:2], 0.9, np.float16)

    labels = crops.people_labels(predict, image, 2 / 3)
    assert shown[0] == (300, 900) and len(shown) >= 4  # a first look, then a row of crops over the group
    group = labels[100:280, 50:850]
    assert (group == 5).mean() > 0.97 and set(np.unique(group)) <= {3, 5}  # refined; a few cut edges keep the look
    assert not labels[:100].any() and not labels[:, :40].any()  # the backdrop stays empty


def test_people_labels_invents_nothing_far_from_people():
    image = np.full((300, 900, 3), 200, np.uint8)
    image[100:280, 50:150, 0] = 40  # one person at the left

    def predict(picture):
        person = picture[..., 0] == 40
        labels = np.where(person, 5, 0).astype(np.int64)
        if picture.shape[:2] != (300, 900):
            labels[picture[..., 0] == 200] = 7  # a crop "sees" clothes on the plain backdrop
        return labels, np.full(picture.shape[:2], 0.9, np.float16)

    labels = crops.people_labels(predict, image, 2 / 3)
    assert (labels[100:280, 50:150] == 5).all()
    columns = np.argwhere(labels == 7)[:, 1]
    reach = crops.NEAR_PEOPLE * 900 + 2
    assert ((columns < 150 + reach) & (columns >= 50 - reach)).all()  # only right next to the person, if at all


def test_people_labels_adds_people_found_by_their_faces():
    image = np.full((300, 900, 3), 200, np.uint8)
    image[100:280, 50:150, 0] = 40
    image[100:280, 700:800, 0] = 40  # the first look misses this one

    def predict(picture):
        person = picture[..., 0] == 40
        if picture.shape[:2] == (300, 900):
            person[:, 400:] = False
        return np.where(person, 5, 0).astype(np.int64), np.full(picture.shape[:2], 0.9, np.float16)

    assert not crops.people_labels(predict, image, 2 / 3)[:, 700:800].any()
    found = crops.people_labels(predict, image, 2 / 3, people=[(680, 90, 820, 300)])
    assert (found[100:280, 700:800] == 5).mean() > 0.95


def test_a_crop_grows_when_its_people_run_past_a_cut_edge():
    image = np.full((800, 800, 3), 200, np.uint8)
    image[300:500, 200:600, 0] = 40
    shown = []

    def predict(picture):
        shown.append(picture.shape[:2])
        person = picture[..., 0] == 40
        if len(shown) == 1:
            person[:, 320:] = False  # the first look sees only the left part
        return np.where(person, 5, 0).astype(np.int64), np.full(picture.shape[:2], 0.9, np.float16)

    labels = crops.people_labels(predict, image, 1.0)
    assert len(shown) >= 3 and shown[2][1] > shown[1][1]  # parsed again, wider
    assert (labels[300:500, 200:600] == 5).mean() > 0.9


def test_people_labels_leaves_regions_too_small_to_parse():
    image = np.full((600, 900, 3), 200, np.uint8)
    image[100:500, 100:300, 0] = 40  # a person
    image[50:70, 700:730, 0] = 40  # a speck far from them: a crop of it would be blown up 20 times
    shown = []

    def predict(picture):
        shown.append(picture.shape[:2])
        labels = np.where(picture[..., 0] == 40, 5, 0).astype(np.int64)
        if len(shown) == 1:
            labels[50:70, 700:730] = 7  # what the first look made of the speck
        return labels, np.full(picture.shape[:2], 0.9, np.float16)

    labels = crops.people_labels(predict, image, 2 / 3)
    assert len(shown) >= 2 and all(height >= crops.MIN_CROP for height, _ in shown[1:])
    assert (labels[50:70, 700:730] == 7).all()  # the speck keeps its first-look labels
    assert (labels[100:500, 100:300] == 5).all()


def test_foreground_clusters_groups_people_standing_together():
    labels = np.zeros((200, 1000), np.int64)
    labels[50:150, 100:200] = 1
    labels[50:150, 210:300] = 1  # 10 px away: same group
    labels[50:150, 800:900] = 1  # far away: its own group
    labels[5, 5] = 1  # a speck
    assert crops.foreground_clusters(labels) == [(100, 50, 300, 150), (800, 50, 900, 150)]
    assert crops.foreground_clusters(labels, extra=[(500, 40, 560, 160)]) == [
        (100, 50, 300, 150), (500, 40, 560, 160), (800, 50, 900, 150)]
    assert crops.foreground_clusters(np.zeros((5, 5), np.int64)) == []


def test_inner_mask_leaves_out_cut_edges_but_not_picture_edges():
    keep = crops.inner_mask((-50, 20, 150, 120), (200, 300), band=0.1)
    assert keep.shape == (100, 150)
    assert keep[10:90, 0].all()  # the region reaches past the picture's left edge: nothing left out there
    assert not keep[:10].any() and not keep[90:].any()  # 10% of 100 px at the cut top and bottom
    assert keep[10:90, :130].all() and not keep[:, 130:].any()  # and 10% of 200 px at the cut right edge


def test_person_box_reaches_down_from_the_face():
    assert crops.person_box((100, 50, 140, 100, 0.9), (1000, 1000)) == (20, 20, 220, 450)
    assert crops.person_box((0, 0, 40, 50), (300, 300)) == (0, 0, 120, 300)


# --- YuNet ---------------------------------------------------------------------
class _FakeYuNet:
    def __init__(self, rows):
        self.rows, self.size = rows, None

    def setInputSize(self, size):  # noqa: N802 (OpenCV's name)
        self.size = size

    def detect(self, image):
        assert image.shape[:2] == (self.size[1], self.size[0])
        return 1, (np.array(self.rows, np.float32) if self.rows else None)


def test_merge_faces_keeps_one_box_per_face():
    whole = (100, 100, 200, 220, 0.95)
    half = (100, 100, 150, 220, 0.7)  # half of the same face, found at another scale
    other = (300, 100, 380, 200, 0.8)
    assert faces.merge_faces([half, whole, other]) == [whole, other]
    assert faces.merge_faces([(0, 0, 10, 10, 0.9), (2, 2, 12, 12, 0.8)]) == [(0, 0, 10, 10, 0.9)]


def test_detect_faces_maps_boxes_back_sorts_and_filters(monkeypatch):
    # rows: x, y, w, h, 10 landmark values, score (on the copy shrunk to 2048 px)
    rows = [[10, 10, 20, 20] + [0] * 10 + [0.9], [100, 50, 60, 80] + [0] * 10 + [0.95],
            [0, 0, 5, 5] + [0] * 10 + [0.3], [2040, 0, 30, 30] + [0] * 10 + [0.8]]
    fake = _FakeYuNet(rows)
    monkeypatch.setattr(faces, "_get_detector", lambda size: (fake.setInputSize(size), fake)[1])
    found = faces.detect_faces(np.zeros((2048, 4096, 3), np.uint8), sides=(2048,))
    assert fake.size == (2048, 1024)  # looked at a copy half the size
    assert [tuple(round(v) for v in f[:4]) for f in found] == [(200, 100, 320, 260), (20, 20, 60, 60),
                                                               (4080, 0, 4096, 60)]  # largest first, clipped
    assert all(f[4] >= faces.MIN_SCORE for f in found)
    fake.rows = []
    assert faces.detect_faces(np.zeros((100, 100, 3), np.uint8)) == []
    seen = []
    monkeypatch.setattr(faces, "_detect_at", lambda img, side, score: seen.append(side) or [])
    faces.detect_faces(np.zeros((100, 100, 3), np.uint8))
    assert seen == list(faces.DETECT_SIDES)  # a sharp copy for small faces, a small one for big or grainy faces


def test_model_path_finds_existing_weights_before_downloading(monkeypatch, tmp_path):
    monkeypatch.setattr(faces, "register_model_folder", lambda *a: [str(tmp_path / "a"), str(tmp_path / "b")])
    monkeypatch.setattr(faces, "default_folder", lambda: str(tmp_path / "a"))
    downloads = []

    def fake_download(repo, revision, target, patterns):
        downloads.append((repo, revision, tuple(patterns)))
        (tmp_path / "a" / "yunet").mkdir(parents=True)
        (tmp_path / "a" / "yunet" / faces.YUNET["file"]).write_bytes(b"x")

    monkeypatch.setattr(faces, "download_pinned", fake_download)
    assert faces.model_path(download=False) is None
    path = faces.model_path()
    assert path.endswith(faces.YUNET["file"]) and downloads[0][1] == faces.YUNET["revision"]
    assert "LICENSE" in downloads[0][2]
    assert faces.model_path() == path and len(downloads) == 1  # found on disk the second time


def test_people_labels_with_a_floor_finds_nobody_where_the_parser_is_unsure():
    image = np.zeros((60, 80, 3), np.uint8)
    image[10:50, 20:40, 0] = 3 * 8  # the class id is written into the red channel, so any crop parses the same
    confidence = [0.3]

    def predict(picture):
        labels = np.rint(picture[..., 0] / 8.0).astype(np.int64)
        labels[labels != 3] = 0  # the grey padding of a crop is background
        return labels, np.full(picture.shape[:2], confidence[0], np.float16)

    assert crops.people_labels(predict, image, 0.75)[10:50, 20:40].all()  # without a floor: kept as it was
    assert not crops.people_labels(predict, image, 0.75, floor=0.5).any()
    with_face = crops.people_labels(predict, image, 0.75, people=[(22, 12, 30, 20)], floor=0.5)
    assert with_face[10:50, 20:40].all()  # a face found: the people are there
    confidence[0] = 0.9
    assert crops.people_labels(predict, image, 0.75, floor=0.5)[10:50, 20:40].all()  # sure without a face: a back
    assert not crops.people_labels(predict, np.zeros_like(image), 0.75, floor=0.5).any()  # nothing labelled


def test_the_floor_goes_by_the_share_the_parser_is_sure_of():
    image = np.zeros((60, 80, 3), np.uint8)
    image[10:50, 20:40, 0] = 3 * 8
    sure_rows = [16]

    def predict(picture):
        labels = np.rint(picture[..., 0] / 8.0).astype(np.int64)
        labels[labels != 3] = 0
        sure = np.full(picture.shape[:2], 0.7, np.float16)
        sure[10:10 + sure_rows[0], 20:40] = 0.95
        return labels, sure

    # 40% of the labelled pixels are sure: nobody, although the median confidence (0.7) is well over the floor
    assert not crops.people_labels(predict, image, 0.75, floor=0.5).any()
    sure_rows[0] = 24  # 60% sure: the people stay
    assert crops.people_labels(predict, image, 0.75, floor=0.5)[10:50, 20:40].all()


def test_the_floor_needs_the_mirror_to_agree():
    image = np.zeros((60, 80, 3), np.uint8)

    def predict(picture):  # a sure "person" at the same place in any picture: invented, so the mirror misses it
        labels = np.zeros(picture.shape[:2], np.int64)
        labels[10:50, 5:25] = 3
        return labels, np.full(picture.shape[:2], 0.95, np.float16)

    assert not crops.people_labels(predict, image, 0.75, floor=0.5).any()
    assert crops.people_labels(predict, image, 0.75)[10:50, 5:25].all()  # without the floor: kept as it was
