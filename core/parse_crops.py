"""Where to run a parsing model on a big picture, and how to put its answers back (model-free, unit tested).

Parsing models see a small fixed-size picture (512 x 512 for the SegFormer face and fashion parsers). Squeezing a
whole 8K photo into that leaves a face a few dozen pixels wide, so the parsers also run on crops: a square around
each face, or a box of the model's shape around each group of people. A crop only refines the parse of the whole
frame: it is pasted inside its own region, away from the region's cut edges, so whatever it misses keeps the
whole-frame answer and no straight crop edge shows. Regions are (x1, y1, x2, y2) in image pixels with exclusive ends,
and may reach past the image; the missing part is padded with grey.
"""

import cv2
import numpy as np

PAD_VALUE = 114  # grey, for the parts of a crop outside the picture


def square_around(box, scale, lift=0.0):
    """Square region ``scale`` times the longer side of ``box``, centred on it and moved up by ``lift`` times its
    side (a head's hair sits above its face box)."""
    x1, y1, x2, y2 = box[:4]
    side = max(1.0, scale * max(x2 - x1, y2 - y1))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0 - lift * side
    left, top = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    size = int(round(side))
    return left, top, left + size, top + size


def _span(lo, hi, size, limit):
    """[start, start + size) holding [lo, hi), slid to stay inside [0, limit) when it fits there."""
    start = (lo + hi) / 2.0 - size / 2.0
    if size <= limit:
        start = min(max(0.0, start), limit - size)
    start = int(round(start))
    return start, start + int(size)


def fit_box(box, aspect, margin, image_size):
    """Region of the model's shape (``aspect`` = width / height) around ``box`` plus ``margin`` (a fraction of its
    longer side) on each side. It stays inside the image where it fits there, and is padded where it does not."""
    height, width = image_size
    x1, y1, x2, y2 = box[:4]
    pad = margin * max(x2 - x1, y2 - y1)
    w, h = (x2 - x1) + 2 * pad, (y2 - y1) + 2 * pad
    w, h = max(w, h * aspect), max(h, w / aspect)
    w, h = int(np.ceil(w)), int(np.ceil(h))
    left, right = _span(x1, x2, w, width)
    top, bottom = _span(y1, y2, h, height)
    return left, top, right, bottom


SPLIT = 1.3  # a box this much wider than the parser's shape is covered by a row of regions instead of one


def _centres(lo, hi, tile, overlap):
    count = max(2, int(np.ceil((hi - lo - tile) / (tile * (1.0 - overlap)))) + 1)
    return np.linspace(lo + tile / 2.0, hi - tile / 2.0, count)


def cover_box(box, aspect, margin, image_size, overlap=0.2):
    """Regions of the parser's shape (``aspect`` = width / height) that together cover ``box`` plus ``margin``.

    One region (see fit_box) unless the box is much wider than that shape, as a group photo is: then a row of
    overlapping regions as tall as the box (plus a margin of its height), so each person keeps the parser's resolution
    instead of the group being squeezed into one region with grey bands above and below. A tall box keeps one region:
    splitting a standing person into halves would hide the rest of the body from the parser.
    """
    x1, y1, x2, y2 = box[:4]
    pad = margin * (y2 - y1)
    bx1, by1, bx2, by2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    if bx2 - bx1 <= (by2 - by1) * aspect * SPLIT:
        return [fit_box(box, aspect, margin, image_size)]
    tile = (by2 - by1) * aspect
    return [fit_box((c - tile / 2.0, by1, c + tile / 2.0, by2), aspect, 0.0, image_size)
            for c in _centres(bx1, bx2, tile, overlap)]


def square_frame(image_size):
    """The square region centred on the picture that holds all of it (the short side is padded)."""
    height, width = image_size
    side = max(height, width)
    left, top = (width - side) // 2, (height - side) // 2
    return left, top, left + side, top + side


def visible(region, image_size):
    """The part of ``region`` inside an image of ``image_size`` (height, width)."""
    height, width = image_size
    x1, y1, x2, y2 = region
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def cut_padded(image, region):
    """``image[region]`` (H x W x C), padded with grey where the region leaves the image."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = region
    vx1, vy1, vx2, vy2 = visible(region, (height, width))
    crop = image[vy1:vy2, vx1:vx2]
    pads = (vy1 - y1, y2 - vy2, vx1 - x1, x2 - vx2)
    if any(pads):
        crop = cv2.copyMakeBorder(np.ascontiguousarray(crop), *pads, cv2.BORDER_CONSTANT,
                                  value=(PAD_VALUE,) * image.shape[2])
    return np.ascontiguousarray(crop)


def nearest_owner(region, boxes, regions, index):
    """Bool mask over ``region`` (inside the image): the pixels that belong to ``boxes[index]`` because no other box
    whose region also covers them is nearer (distance from the box centre, in box sizes). Where the crops around two
    faces overlap, each face keeps the parse of its own crop."""
    x1, y1, x2, y2 = region
    overlapping = [other for other in range(len(boxes)) if other != index and regions[other][2] > x1
                   and regions[other][0] < x2 and regions[other][3] > y1 and regions[other][1] < y2]
    if not overlapping:
        return np.ones((y2 - y1, x2 - x1), dtype=bool)
    # decided on a grid of at most OWNER_GRID points a side, then enlarged: the border between two faces runs through
    # hair or background, where a few pixels do not matter, and an 8K crop would otherwise cost a second per face
    step = max(1, int(np.ceil(max(x2 - x1, y2 - y1) / OWNER_GRID)))
    ys = np.arange(y1, y2, step, dtype=np.float32)[:, None] + step / 2.0
    xs = np.arange(x1, x2, step, dtype=np.float32)[None, :] + step / 2.0

    def distance(box):
        bx1, by1, bx2, by2 = box[:4]
        size = max(1.0, float(max(bx2 - bx1, by2 - by1)))
        return np.hypot(xs - (bx1 + bx2) / 2.0, ys - (by1 + by2) / 2.0) / size

    own = distance(boxes[index])
    keep = np.ones(own.shape, dtype=bool)
    for other in overlapping:
        ox1, oy1, ox2, oy2 = regions[other]
        covered = (xs >= ox1) & (xs < ox2) & (ys >= oy1) & (ys < oy2)
        keep &= ~covered | (own <= distance(boxes[other]))
    if step > 1:
        keep = cv2.resize(keep.astype(np.uint8), (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST) > 0
    return keep


OWNER_GRID = 512


CUT_BAND = 0.08  # share of a crop's side, along each of its edges that cuts the picture, left to the whole frame
NEAR_PEOPLE = 0.02  # how far (share of the long side) from the whole-frame parse's people a crop may add labels


def inner_mask(region, image_size, band=CUT_BAND):
    """Bool mask over the part of ``region`` inside the picture: True except within ``band`` of the region's size
    along its edges that lie inside the picture. The parser sees things cut there, so its answer is less sure; an edge
    that is the picture's own edge (or beyond it) keeps everything."""
    height, width = image_size
    x1, y1, x2, y2 = region
    vx1, vy1, vx2, vy2 = visible(region, image_size)
    keep = np.ones((vy2 - vy1, vx2 - vx1), dtype=bool)
    cut_x, cut_y = int(round(band * (x2 - x1))), int(round(band * (y2 - y1)))
    if x1 > 0:
        keep[:, :max(0, x1 + cut_x - vx1)] = False
    if x2 < width:
        keep[:, max(0, (x2 - cut_x) - vx1):] = False
    if y1 > 0:
        keep[:max(0, y1 + cut_y - vy1), :] = False
    if y2 < height:
        keep[max(0, (y2 - cut_y) - vy1):, :] = False
    return keep


def parse_region(predict, image, region):
    """(``predict`` run on the padded crop, cut to the part of ``region`` inside the picture; that part). When
    ``predict`` returns several maps (labels and their confidence), each is cut the same way."""
    seen = visible(region, image.shape[:2])
    result = predict(cut_padded(image, region))
    x1, y1 = region[:2]
    rows, cols = slice(seen[1] - y1, seen[3] - y1), slice(seen[0] - x1, seen[2] - x1)
    if isinstance(result, tuple):
        return tuple(part[rows, cols] for part in result), seen
    return result[rows, cols], seen


GROW_ROUNDS = 3  # times a crop may grow when its people run past one of its cut edges
GROW_STEP = 0.5  # how much it grows each time, as a share of its size


def _cut_sides_with_people(labels, region, image_size, background=0, share=0.02):
    """Edges of ``region`` that cut the picture and where the parse (of the part inside the picture) still finds
    people within CUT_BAND: there the crop sliced through somebody."""
    height, width = image_size
    x1, y1, x2, y2 = region
    vx1, vy1, vx2, vy2 = visible(region, image_size)
    band_x, band_y = max(1, int(round(CUT_BAND * (x2 - x1)))), max(1, int(round(CUT_BAND * (y2 - y1))))
    found = labels != background
    sides = []
    if x1 > 0 and found[:, :band_x].mean() > share:
        sides.append("left")
    if x2 < width and found[:, max(0, (vx2 - vx1) - band_x):].mean() > share:
        sides.append("right")
    if y1 > 0 and found[:band_y, :].mean() > share:
        sides.append("top")
    if y2 < height and found[max(0, (vy2 - vy1) - band_y):, :].mean() > share:
        sides.append("bottom")
    return sides


def _grown(region, sides, aspect, image_size):
    """``region`` enlarged by GROW_STEP of its size towards ``sides``, back in the parser's shape."""
    x1, y1, x2, y2 = region
    step_x, step_y = GROW_STEP * (x2 - x1), GROW_STEP * (y2 - y1)
    box = (x1 - step_x * ("left" in sides), y1 - step_y * ("top" in sides),
           x2 + step_x * ("right" in sides), y2 + step_y * ("bottom" in sides))
    return fit_box(box, aspect, 0.0, image_size)


COARSE_TRUST = 0.5  # the first look's confidence counts this much against a crop's: a crop sees people sharper


def person_box(face, image_size):
    """Rough box of the person a face (x1, y1, x2, y2[, score]) belongs to: five face widths wide, from just above
    the head to about eight face heights below its top (a standing person), inside the picture."""
    height, width = image_size
    x1, y1, x2, y2 = face[:4]
    face_w, face_h = x2 - x1, y2 - y1
    centre = (x1 + x2) / 2.0
    return (max(0, int(centre - 2.5 * face_w)), max(0, int(y1 - 0.6 * face_h)),
            min(width, int(np.ceil(centre + 2.5 * face_w))), min(height, int(np.ceil(y1 + 8.0 * face_h))))


def people_labels(predict, image, aspect, margin=0.08, people=()):
    """Label map of the people in the uint8 RGB ``image``, from ``predict`` (picture -> (label map with 0 for
    background, confidence map)), a parser trained on pictures of the shape ``aspect`` (width / height).

    A first look parses the whole frame and is kept as the answer to refine. Each group of people (those it found,
    and the ``people`` boxes found from faces, which it may have missed) gets regions of the parser's shape (a row of
    them for a wide group, see cover_box), parsed again at the parser's full resolution; a region whose people run past
    one of its cut edges grows that way and is parsed again. A region's label replaces the current one where it is
    surer and away from the region's cut edges, for the parts of it that lie on people already found (see anchored)
    and for background near them; so nothing is invented on an empty backdrop and overlapping regions meet where their
    answers do, not along a straight line.
    """
    height, width = image.shape[:2]
    coarse, trust = predict(image)
    regions = [region for box in foreground_clusters(coarse, extra=people)
               for region in cover_box(box, aspect, margin, (height, width))]
    if not regions or regions == [(0, 0, width, height)]:
        return coarse  # nobody found, or the first look already saw the whole picture in the parser's own shape
    anchor = coarse != 0  # where the first look (or a face) puts people
    for x1, y1, x2, y2 in people:
        anchor[y1:y2, x1:x2] = True
    near = near_people(anchor.astype(np.uint8))
    canvas, trust = coarse.copy(), trust.astype(np.float32) * COARSE_TRUST
    for region in regions:
        (labels, sure), seen = parse_region(predict, image, region)
        for _ in range(GROW_ROUNDS):
            sides = _cut_sides_with_people(labels, region, (height, width))
            if not sides:
                break
            region = _grown(region, sides, aspect, (height, width))
            (labels, sure), seen = parse_region(predict, image, region)
        if labels.size == 0:
            continue
        x1, y1, x2, y2 = seen
        sure = sure.astype(np.float32)
        allowed = anchored(labels, anchor[y1:y2, x1:x2]) | ((labels == 0) & near[y1:y2, x1:x2])
        keep = inner_mask(region, (height, width)) & allowed & (sure > trust[y1:y2, x1:x2])
        canvas[y1:y2, x1:x2][keep] = labels[keep]
        trust[y1:y2, x1:x2][keep] = sure[keep]
    return canvas


ANCHOR_SHARE = 0.1  # share of a crop's part that must lie on people already found for the part to be kept


def anchored(labels, anchor, share=ANCHOR_SHARE, grid=1024):
    """Bool mask of the labelled pixels of a crop's ``labels`` whose part (the connected pixels of one label) lies on
    ``anchor`` (people already found) for at least ``share`` of its area. A crop may so extend a person the first look
    saw only in part, but anything it finds on its own (clothes "seen" on a plain backdrop, even right next to
    somebody) is dropped. Worked out on a copy at most ``grid`` pixels long."""
    height, width = labels.shape
    scale = min(1.0, grid / max(height, width, 1))
    small, held = labels, anchor
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        small = cv2.resize(labels.astype(np.int32), size, interpolation=cv2.INTER_NEAREST)
        held = cv2.resize(anchor.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    kept = np.zeros(small.shape, dtype=bool)
    for value in np.unique(small):
        if value == 0:
            continue
        count, parts = cv2.connectedComponents((small == value).astype(np.uint8), connectivity=8)
        if count <= 1:
            continue
        areas = np.bincount(parts.ravel(), minlength=count)
        on_people = np.bincount(parts[held], minlength=count)
        good = on_people >= share * areas
        good[0] = False
        kept |= good[parts]
    if scale < 1.0:
        kept = cv2.resize(kept.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    return kept & (labels != 0)


def near_people(labels, background=0, reach=NEAR_PEOPLE, grid=1024):
    """Bool map of the pixels within ``reach`` (a share of the long side) of a labelled pixel. Worked out on a copy at
    most ``grid`` pixels long and enlarged, since a few pixels do not matter here and a 300 px dilation of an 8K map
    would take seconds."""
    height, width = labels.shape
    scale = min(1.0, grid / max(height, width))
    found = (labels != background).astype(np.uint8)
    if scale < 1.0:
        found = cv2.resize(found, (max(1, round(width * scale)), max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
        found = (found > 0).astype(np.uint8)
    radius = max(1, int(round(reach * max(found.shape))))
    grown = cv2.dilate(found, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    if scale < 1.0:
        grown = cv2.resize(grown, (width, height), interpolation=cv2.INTER_NEAREST)
    return grown > 0


def foreground_clusters(labels, background=0, min_fraction=0.001, gap=0.05, extra=()):
    """Boxes around the groups of labelled pixels of a label map (and the ``extra`` boxes): specks under
    ``min_fraction`` of the picture are left out, and boxes closer to each other than ``gap`` of the picture's long
    side are merged, so people standing together form one group and people far apart get one box each."""
    found = (labels != background).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(found, connectivity=8)
    boxes = [[int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
              int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]),
              int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT])]
             for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_fraction * labels.size]
    boxes += [[int(v) for v in box[:4]] for box in extra]
    reach = gap * max(labels.shape)
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if (max(a[0], b[0]) - min(a[2], b[2]) < reach) and (max(a[1], b[1]) - min(a[3], b[3]) < reach):
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return [tuple(box) for box in sorted(boxes)]
