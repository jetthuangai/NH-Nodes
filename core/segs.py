"""NH_SEGS: the instances the NH vision nodes pass between each other.

An ``NH_SEGS`` value is a tuple with one ``SegsFrame`` per image of a batch. A frame holds the image size and the
``Segment`` instances found in that image. A segment keeps its label, score and box, and only the mask pixels inside
its box, so dozens of instances on an 8K image stay small. Full-size masks are built only when a node outputs a MASK.

Plain torch/numpy, no model: the select, compose, preview and convert nodes and their tests run anywhere.
"""

import fnmatch
import functools
from collections import namedtuple
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from .vi_words import translate

SEGS_TYPE = "NH_SEGS"


@dataclass(frozen=True, eq=False)
class Segment:
    """One instance found in an image.

    ``box`` is (x1, y1, x2, y2) in image pixels with exclusive ends, so the instance covers ``image[y1:y2, x1:x2]``;
    it is the tight box around the mask. ``mask`` is a bool [y2 - y1, x2 - x1] CPU tensor over that box.
    """

    label: str
    score: float
    box: tuple
    mask: torch.Tensor
    source: str = "sam3"

    @property
    def area(self):
        """Number of mask pixels."""
        return int(self.mask.sum())

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass(frozen=True, eq=False)
class SegsFrame:
    """The instances found in one image of ``height`` x ``width`` pixels, in the order they were found."""

    height: int
    width: int
    segments: tuple = ()

    def with_segments(self, segments):
        return SegsFrame(self.height, self.width, tuple(segments))

    def _comfy_cache_tensors(self):
        """The tensors this frame holds, so ComfyUI's RAM-pressure cache can size it."""
        return [seg.mask for seg in self.segments]


def make_segment(label, score, mask, origin=(0, 0), source="sam3"):
    """Segment for the True pixels of the bool [h, w] ``mask`` whose top-left pixel sits at ``origin`` (x, y) of the image.

    The stored box is the tight box around those pixels and the stored mask a compact CPU copy of that box. Works on
    any device, cropping before the copy to the CPU. Returns None when the mask is empty.
    """
    mask = torch.as_tensor(mask)
    if mask.dtype != torch.bool:
        mask = mask > 0.5
    rows = torch.nonzero(mask.any(dim=1)).flatten()
    if rows.numel() == 0:
        return None
    cols = torch.nonzero(mask.any(dim=0)).flatten()
    y1, y2 = int(rows[0]), int(rows[-1]) + 1
    x1, x2 = int(cols[0]), int(cols[-1]) + 1
    crop = mask[y1:y2, x1:x2].to("cpu").clone(memory_format=torch.contiguous_format)
    ox, oy = int(origin[0]), int(origin[1])
    return Segment(str(label), float(score), (ox + x1, oy + y1, ox + x2, oy + y2), crop, source)


def frame_mask(frame, segments=None):
    """Bool [H, W] union of ``segments`` (every segment of ``frame`` by default)."""
    canvas = torch.zeros((frame.height, frame.width), dtype=torch.bool)
    for seg in frame.segments if segments is None else segments:
        x1, y1, x2, y2 = seg.box
        canvas[y1:y2, x1:x2] |= seg.mask
    return canvas


def segs_union(segs):
    """MASK [B, H, W] float: the union of every instance, one mask per image."""
    return torch.stack([frame_mask(frame) for frame in segs]).float()


def check_memory(nbytes, what):
    """Raise MemoryError with advice when ``nbytes`` would take more than half of the RAM that is free now."""
    try:
        import psutil

        available = psutil.virtual_memory().available
    except Exception:  # no psutil: nothing to compare with
        return
    if nbytes > available // 2:
        raise MemoryError(
            f"[NH-Nodes] {what} would need {nbytes / 2**30:.1f} GiB but only {available / 2**30:.1f} GiB of RAM is "
            "free. Keep fewer instances with Segs Select, or combine them into one mask with Segs Compose."
        )


def instance_masks(segs):
    """(MASK [N, H, W] float with one full-size mask per instance of every image, in order; the labels).

    With no instance at all the MASK holds a single empty mask, as nodes downstream may not accept an empty batch.
    Each full-size mask takes H x W x 4 bytes (about 145 MB on an 8K image), so a MemoryError explains what to do
    when they would not fit in RAM.
    """
    labels = [seg.label for frame in segs for seg in frame.segments]
    height, width = segs[0].height, segs[0].width
    check_memory(max(1, len(labels)) * height * width * 4, f"{len(labels)} full-size instance masks")
    masks = torch.zeros((max(1, len(labels)), height, width))
    index = 0
    for frame in segs:
        for seg in frame.segments:
            x1, y1, x2, y2 = seg.box
            masks[index, y1:y2, x1:x2] = seg.mask.float()
            index += 1
    return masks, labels


def merge_segs(*all_segs):
    """One NH_SEGS from several of the same pictures (None entries are skipped): each frame gets the instances of
    all of them, in order. Raises when they do not belong to the same pictures."""
    present = [segs for segs in all_segs if segs is not None]
    first = present[0]
    for other in present[1:]:
        if len(other) != len(first) or any((a.height, a.width) != (b.height, b.width) for a, b in zip(first, other)):
            raise ValueError("[NH-Nodes] The segs to mix come from different pictures: connect segs found on the same "
                             "image batch.")
    if len(present) == 1:
        return first
    return tuple(frame.with_segments([seg for segs in present for seg in segs[index].segments])
                 for index, frame in enumerate(first))


def count_segments(segs):
    return sum(len(frame.segments) for frame in segs)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def normalize_label(text):
    """Lower case, '_' as a space, single spaces: "Red_Car " and "red  car" both become "red car"."""
    return " ".join(str(text).replace("_", " ").lower().split())


def label_matches(label, pattern):
    """True when ``label`` matches ``pattern``, ignoring case; the pattern may use * and ? wildcards."""
    return fnmatch.fnmatchcase(normalize_label(label), normalize_label(pattern))


def parse_label_list(text):
    """Commas and new lines separate entries: "person, red car" -> ["person", "red car"].

    Blank entries and lines starting with '#' are dropped; a repeated entry (ignoring case and spacing) is kept once,
    with its first spelling.
    """
    entries, seen = [], set()
    for line in str(text or "").splitlines():
        if line.strip().startswith("#"):
            continue
        for part in line.split(","):
            entry = " ".join(part.split())
            key = normalize_label(entry)
            if entry and key not in seen:
                seen.add(key)
                entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Select
# ---------------------------------------------------------------------------
SORT_ORDERS = ("score", "area", "left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top",
               "center_first", "as_found")


def sort_segments(segments, order, frame):
    """``segments`` in ``order`` (one of SORT_ORDERS); positions use each box's centre. Ties keep their found order."""
    segments = list(segments)
    if order == "score":
        return sorted(segments, key=lambda s: -s.score)
    if order == "area":
        return sorted(segments, key=lambda s: -s.area)
    if order == "left_to_right":
        return sorted(segments, key=lambda s: s.center[0])
    if order == "right_to_left":
        return sorted(segments, key=lambda s: -s.center[0])
    if order == "top_to_bottom":
        return sorted(segments, key=lambda s: s.center[1])
    if order == "bottom_to_top":
        return sorted(segments, key=lambda s: -s.center[1])
    if order == "center_first":
        cx, cy = frame.width / 2.0, frame.height / 2.0
        return sorted(segments, key=lambda s: (s.center[0] - cx) ** 2 + (s.center[1] - cy) ** 2)
    return segments


def select_segments(frame, labels=(), min_score=0.0, min_area=0.0, max_area=1.0, order="score", start=0, count=0):
    """Frame with the segments that pass the filters, sorted by ``order``, from position ``start`` (0 = first).

    ``labels`` are patterns (any one must match; none = every label). Areas are fractions of the image. ``count`` = 0
    keeps every segment from ``start`` on.
    """
    total = max(1, frame.height * frame.width)
    kept = []
    for seg in frame.segments:
        if labels and not any(label_matches(seg.label, pattern) for pattern in labels):
            continue
        if seg.score < min_score:
            continue
        if not min_area <= seg.area / total <= max_area:
            continue
        kept.append(seg)
    kept = sort_segments(kept, order, frame)
    start = max(0, int(start))
    kept = kept[start:] if count <= 0 else kept[start:start + int(count)]
    return frame.with_segments(kept)


# ---------------------------------------------------------------------------
# Compose: a small boolean language over labels
# ---------------------------------------------------------------------------
class ExpressionError(ValueError):
    """An NH_SEGS mask expression that cannot be read."""


class EmptyExpressionError(ExpressionError):
    """An expression with nothing in it (blank, or only commas and comments)."""


_OPERATORS = {"+": "|", "|": "|", "-": "-", "&": "&", "~": "~", "!": "~", "(": "(", ")": ")"}
_SEPARATORS = ",\n"  # list items, joined like '|' but read last
_WORD_OPERATORS = {"or": "|", "and": "&", "not": "~"}
_KEYWORDS = ("all", "image", "none", "background")
_QUOTES = "\"'"


def _is_word_char(ch):
    return ch.isalnum() or ch == "_"


def _word_tokens(text, word_operators):
    """[(kind, value, position, written)] with kinds "op", "sep", "word" and "quoted"; positions count from 1.

    A quote opens a quoted label only where a token starts, and a '-' between two word characters is part of the
    word, so ``women's`` and ``t-shirt`` are plain words while ``person - hair`` subtracts. A '#' where a token would
    start begins a comment that runs to the end of its line (inside a word, as in ``car#2``, it is part of the word).
    """
    tokens, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "#":
            end = text.find("\n", i)
            i = n if end < 0 else end
        elif ch in _SEPARATORS:
            tokens.append(("sep", ",", i + 1, "a new line" if ch == "\n" else ch))
            i += 1
        elif ch.isspace():
            i += 1
        elif ch in _OPERATORS:
            tokens.append(("op", _OPERATORS[ch], i + 1, ch))
            i += 1
        elif ch in _QUOTES:
            end = text.find(ch, i + 1)
            if end < 0:
                raise ExpressionError(f"quote opened at position {i + 1} is never closed")
            tokens.append(("quoted", text[i + 1:end], i + 1, text[i:end + 1]))
            i = end + 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in _SEPARATORS:
                hyphen_in_word = (text[j] == "-" and _is_word_char(text[j - 1])
                                  and j + 1 < n and _is_word_char(text[j + 1]))
                if text[j] in _OPERATORS and not hyphen_in_word:
                    break
                j += 1
            word = text[i:j]
            if word_operators and word.lower() in _WORD_OPERATORS:
                tokens.append(("op", _WORD_OPERATORS[word.lower()], i + 1, word))
            else:
                tokens.append(("word", word, i + 1, word))
            i = j
    return tokens


def _tokenize(text, word_operators=True):
    """Tokens with each run of words joined into one "label" (or "keyword") token: ``red car`` is one label.

    Labels are read through ``vi_words.translate``, so Vietnamese becomes the English the instances are named in;
    the token keeps (english, status) as its value. A run that translates to all, image, none or background, unquoted,
    is that keyword.
    """
    raw = _word_tokens(text, word_operators)
    tokens = []
    for index, token in enumerate(raw):
        kind, value, position, written = token
        if kind == "sep" and written == "a new line":
            # a formula laid out over several lines: a line break next to an operator joins, it does not list
            after = next((t for t in raw[index + 1:] if not (t[0] == "sep" and t[3] == "a new line")), None)
            before = next((t for t in reversed(tokens) if t[0] != "sep"), None) if tokens and tokens[-1][0] != "sep" \
                else None
            if (before is not None and before[0] == "op" and before[1] != ")") or (
                    after is not None and after[0] == "op" and after[1] in ("|", "-", "&", ")")):
                continue
        if kind in ("word", "quoted") and tokens and tokens[-1][0] == "run":
            _, parts, start, quoted = tokens[-1]
            tokens[-1] = ("run", parts + [value], start, quoted or kind == "quoted")
        elif kind in ("word", "quoted"):
            tokens.append(("run", [value], position, kind == "quoted"))
        else:
            tokens.append(token)
    result = []
    for token in tokens:
        if token[0] != "run":
            result.append(token)
            continue
        _, parts, position, quoted = token
        written = " ".join(parts)
        english, status = translate(written)
        if not quoted and english.lower() in _KEYWORDS:
            result.append(("keyword", english.lower(), position, written))
        else:
            result.append(("label", (english, status), position, written))
    return result


class _Parser:
    """list := expr? (',' expr?)* ; expr := term (('|' | '-') term)* ; term := unary ('&' unary)* ;
    unary := '~' unary | atom ; atom := label | keyword | '(' list ')'. Commas and new lines join the items of a list
    like '|', after every other operator. Builds a tuple tree evaluated by ``evaluate``: a label node is
    ("label", english, written, status), a keyword node ("keyword", name, written)."""

    def __init__(self, text, word_operators=True):
        self.tokens = _tokenize(text, word_operators)
        self.pos = 0

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _where(self):
        token = self._peek()
        return f"'{token[3]}' at position {token[2]}" if token else "the end"

    def parse(self):
        items = self._list(nested=False)
        if not items:
            raise EmptyExpressionError("the expression is empty; use 'all' for every instance")
        return self._union(items)

    @staticmethod
    def _union(items):
        tree = items[0]
        for item in items[1:]:
            tree = ("|", tree, item)
        return tree

    def _list(self, nested):
        """The items of a list up to the end (or the closing ')' when ``nested``); empty items are skipped."""
        items = []
        while True:
            token = self._peek()
            if token is not None and token[0] == "sep":
                self.pos += 1
                continue
            if token is None or (nested and token[:2] == ("op", ")")):
                return items
            items.append(self._expr())
            token = self._peek()
            if token is not None and token[0] != "sep" and not (nested and token[:2] == ("op", ")")):
                raise ExpressionError(f"unexpected {self._where()}")

    def _expr(self):
        tree = self._term()
        while (token := self._peek()) is not None and token[0] == "op" and token[1] in ("|", "-"):
            self.pos += 1
            tree = (token[1], tree, self._term())
        return tree

    def _term(self):
        tree = self._unary()
        while (token := self._peek()) is not None and token[0] == "op" and token[1] == "&":
            self.pos += 1
            tree = ("&", tree, self._unary())
        return tree

    def _unary(self):
        token = self._peek()
        if token is not None and token[0] == "op" and token[1] == "~":
            self.pos += 1
            return ("~", self._unary())
        return self._atom()

    def _atom(self):
        token = self._peek()
        if token is None:
            raise ExpressionError("the expression ends where a label, 'all', 'image', 'none' or '(' was expected")
        kind, value, position, written = token
        self.pos += 1
        if kind == "label":
            return ("label", value[0], written, value[1])
        if kind == "keyword":
            return ("keyword", value, written)
        if value == "(":
            items = self._list(nested=True)
            closing = self._peek()
            if closing is None or closing[:2] != ("op", ")"):
                raise ExpressionError(f"'(' at position {position} is not closed (found {self._where()})")
            if not items:
                raise ExpressionError(f"'(' at position {position} holds nothing")
            self.pos += 1
            return self._union(items)
        raise ExpressionError(f"unexpected '{written}' at position {position}")


@functools.lru_cache(maxsize=64)
def parse_expression(text, word_operators=True):
    """Parse a mask expression (see ``evaluate``) into a tree; raises ExpressionError with the position on mistakes.

    ``word_operators=False`` reads or/and/not as words, for the natural phrases SAM 3 takes ("black and white dog").
    """
    return _Parser(str(text), word_operators).parse()


def parse_concepts(text):
    """``parse_expression`` for SAM3 Segment's concepts: or/and/not are plain words there."""
    return parse_expression(text, word_operators=False)


def expression_labels(tree):
    """Label patterns (in English) used in a parsed expression, in order."""
    if tree[0] == "label":
        return [tree[1]]
    if tree[0] == "keyword":
        return []
    return [label for child in tree[1:] for label in expression_labels(child)]


def expression_terms(tree):
    """(written, english, status) of each label and keyword of a parsed expression, in order; status is "english",
    "translated" or "unknown" (Vietnamese not in vi_words) for labels and "keyword" for keywords."""
    if tree[0] == "label":
        return [(tree[2], tree[1], tree[3])]
    if tree[0] == "keyword":
        return [(tree[2], tree[1], "keyword")]
    return [term for child in tree[1:] for term in expression_terms(child)]


def uses_keyword(tree, name):
    return any(status == "keyword" and english == name for _, english, status in expression_terms(tree))


def evaluate(tree, frame):
    """Bool [H, W] mask of a parsed expression for one frame.

    A label (wildcards allowed; quotes or '_' for spaces; a '-' inside a word is part of it; words in a row make one
    label) is the union of the matching instances, ``all`` every instance, ``background`` everything outside them,
    ``image`` the whole frame, ``none`` nothing. ``a | b`` (also ``+``, ``or``, a comma or a new line) is the union,
    ``a - b`` the difference, ``a & b`` (``and``) the intersection, ``~a`` (``!``, ``not``) everything outside ``a``.
    """
    kind = tree[0]
    if kind == "label":
        return frame_mask(frame, [seg for seg in frame.segments if label_matches(seg.label, tree[1])])
    if kind == "keyword":
        if tree[1] == "all":
            return frame_mask(frame)
        if tree[1] == "background":
            return ~frame_mask(frame)
        if tree[1] == "image":
            return torch.ones((frame.height, frame.width), dtype=torch.bool)
        return torch.zeros((frame.height, frame.width), dtype=torch.bool)
    if kind == "~":
        return ~evaluate(tree[1], frame)
    left, right = evaluate(tree[1], frame), evaluate(tree[2], frame)
    if kind == "|":
        return left | right
    if kind == "&":
        return left & right
    return left & ~right


def expression_masks(tree, segs):
    """MASK [B, H, W] float of a parsed expression, one mask per image of ``segs``."""
    return torch.stack([evaluate(tree, frame) for frame in segs]).float()


# ---------------------------------------------------------------------------
# Impact Pack SEGS
# ---------------------------------------------------------------------------
# The field names of Impact Pack's SEG, so its detailers read these items; NH-Nodes does not import Impact Pack.
IMPACT_SEG = namedtuple("SEG", ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label",
                                "control_net_wrapper"], defaults=[None])


def _centred_span(lo, hi, factor, limit):
    """[start, end) of ``factor`` times the length of [lo, hi), centred on it and slid inside [0, limit).

    It is never shorter than [lo, hi) and only shorter than asked when the image is, so it always contains [lo, hi).
    """
    size = min(limit, max(hi - lo, round((hi - lo) * factor)))
    start = min(max(0, round((lo + hi - size) / 2)), limit - size)
    return start, start + size


def detail_crop_region(width, height, box, crop_factor):
    """[x1, y1, x2, y2] a detailer works on around ``box``: ``crop_factor`` times its size, centred, inside the image."""
    rx1, rx2 = _centred_span(box[0], box[2], crop_factor, width)
    ry1, ry2 = _centred_span(box[1], box[3], crop_factor, height)
    return [rx1, ry1, rx2, ry2]


def to_impact_segs(frame, crop_factor=3.0):
    """Impact Pack SEGS ``((height, width), [SEG, ...])`` for one frame; each mask is placed in its crop region."""
    items = []
    for seg in frame.segments:
        region = detail_crop_region(frame.width, frame.height, seg.box, crop_factor)
        rx1, ry1, rx2, ry2 = region
        x1, y1, x2, y2 = seg.box
        cropped = np.zeros((ry2 - ry1, rx2 - rx1), dtype=np.float32)
        cropped[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = seg.mask.numpy()
        items.append(IMPACT_SEG(None, cropped, float(seg.score), region, [x1, y1, x2, y2], seg.label, None))
    return (frame.height, frame.width), items


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------
_PALETTE = (
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40),
)


@functools.lru_cache(maxsize=16)
def _label_font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):  # both cover Vietnamese
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _free_spot(x, y, width, height, placed, limit):
    """``y`` moved down (or up near the bottom ``limit``) until a ``width`` x ``height`` label at ``x`` overlaps none
    of the ``placed`` (x1, y1, x2, y2) rectangles; labels of instances sharing a top edge (person, hair) then stack."""
    for _ in range(len(placed)):
        hit = next((r for r in placed if x < r[2] and r[0] < x + width and y < r[3] and r[1] < y + height), None)
        if hit is None:
            break
        y = hit[3] + 1 if hit[3] + 1 + height <= limit else max(0, hit[1] - height - 1)
    return y


def render_preview(image, frame, fill=0.45):
    """uint8 HxWx3 copy of ``image`` with each instance tinted, outlined and labelled "label score"."""
    canvas = np.ascontiguousarray(image[:, :, :3]).copy()
    height, width = canvas.shape[:2]
    if not frame.segments:
        return canvas
    thickness = max(1, round(min(height, width) / 700))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * thickness + 1, 2 * thickness + 1))
    for index, seg in enumerate(frame.segments):
        color = np.array(_PALETTE[index % len(_PALETTE)], dtype=np.float32)
        x1, y1, x2, y2 = seg.box
        region = canvas[y1:y2, x1:x2]
        inside = seg.mask.numpy()
        region[inside] = (region[inside] * (1.0 - fill) + color * fill).astype(np.uint8)
        padded = np.pad(inside.astype(np.uint8), thickness)  # so a box edge counts as an outline too
        edge = (padded > 0) & (cv2.erode(padded, kernel) == 0)
        region[edge[thickness:-thickness, thickness:-thickness]] = color.astype(np.uint8)

    font = _label_font(max(12, round(min(height, width) / 45)))
    picture = Image.fromarray(canvas)
    draw = ImageDraw.Draw(picture)
    placed = []  # label rectangles drawn so far, so a label never hides another one
    for index, seg in enumerate(frame.segments):
        color = _PALETTE[index % len(_PALETTE)]
        text = f"{seg.label} {seg.score:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        pad = max(2, (bottom - top) // 6)
        tw, th = right - left + 2 * pad, bottom - top + 2 * pad
        # centred over the instance's topmost pixels (a head, a roof), not at a box corner that may be far from it
        top_columns = torch.nonzero(seg.mask[0]).flatten()
        anchor = seg.box[0] + int(top_columns[top_columns.numel() // 2]) if top_columns.numel() else seg.box[0]
        x = min(max(0, anchor - tw // 2), max(0, width - tw))
        y = _free_spot(x, seg.box[1] - th if seg.box[1] - th >= 0 else seg.box[1], tw, th, placed, height)
        placed.append((x, y, x + tw, y + th))
        draw.rectangle((x, y, x + tw, y + th), fill=color)
        dark_text = (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]) > 150
        draw.text((x + pad - left, y + pad - top), text, font=font, fill=(0, 0, 0) if dark_text else (255, 255, 255))
    return np.asarray(picture)


def scale_frame(frame, scale):
    """The frame of the picture made ``scale`` times smaller (for previews): boxes scaled, masks area-resampled.

    An instance too small to keep a pixel keeps its centre pixel, so it still shows up with its label.
    """
    height, width = max(1, round(frame.height * scale)), max(1, round(frame.width * scale))
    segments = []
    for seg in frame.segments:
        x1, y1, x2, y2 = seg.box
        nx1, ny1 = min(width - 1, int(x1 * scale)), min(height - 1, int(y1 * scale))
        nx2 = max(nx1 + 1, min(width, int(np.ceil(x2 * scale))))
        ny2 = max(ny1 + 1, min(height, int(np.ceil(y2 * scale))))
        small = cv2.resize(seg.mask.numpy().astype(np.uint8) * 255, (nx2 - nx1, ny2 - ny1),
                           interpolation=cv2.INTER_AREA) >= 128
        if not small.any():
            small[small.shape[0] // 2, small.shape[1] // 2] = True
        segments.append(make_segment(seg.label, seg.score, small, (nx1, ny1), seg.source))
    return SegsFrame(height, width, tuple(segments))


def _caption(picture, notes):
    """``picture`` (uint8 RGB) with ``notes`` written on a white strip below it, one line each, long lines wrapped."""
    width = picture.shape[1]
    font = _label_font(max(13, round(width / 70)))
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines = []
    for note in notes:
        words, line = str(note).split(), ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if line and measure.textlength(candidate, font=font) > width - 16:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    left, top, right, bottom = measure.textbbox((0, 0), "Ag", font=font)
    step = bottom - top + 6
    strip = Image.new("RGB", (width, step * len(lines) + 10), (255, 255, 255))
    draw = ImageDraw.Draw(strip)
    for index, line in enumerate(lines):
        draw.text((8, 5 + index * step - top), line, font=font, fill=(20, 20, 20))
    return np.concatenate([picture, np.asarray(strip)], axis=0)


def preview_picture(image, frame, notes=(), max_side=1536, mask=None):
    """uint8 RGB preview of one picture, at most ``max_side`` px, its instances drawn and ``notes`` written below.

    ``image`` is one ComfyUI image [H, W, C] (float, 0..1). With ``mask`` (the node's [H, W] output) everything
    outside it is darkened, so the preview shows what the mask holds ("background" lights up the background) while the
    instances keep their outline and label. The picture is shrunk first (area averaging), so an 8K image with a
    hundred instances needs a few MB more, not the full-size float preview Segs Preview makes.
    """
    rgb = image[..., :3] if image.shape[-1] >= 3 else image[..., :1].expand(-1, -1, 3)
    scale = min(1.0, max_side / max(frame.height, frame.width))
    size = (max(1, round(frame.height * scale)), max(1, round(frame.width * scale)))
    if scale < 1.0:  # shrink before anything else, so no full-size copy of an 8K picture is ever made
        rgb = F.interpolate(rgb.permute(2, 0, 1)[None].float(), size=size, mode="area")[0].permute(1, 2, 0)
        frame = scale_frame(frame, scale)
    pixels = (rgb.float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
    if mask is not None:
        held = mask.float()
        if scale < 1.0:
            held = F.interpolate(held[None, None], size=size, mode="area")[0, 0]
        outside = held.cpu().numpy() < 0.5
        pixels = pixels.copy()
        pixels[outside] = (pixels[outside] * _DIM_OUTSIDE).astype(np.uint8)
    picture = render_preview(pixels, frame, fill=0.2 if mask is not None else 0.45)
    return _caption(picture, notes) if notes else picture


_DIM_OUTSIDE = 0.35  # brightness kept outside the mask in previews


def describe_request(tree, segs, threshold=None, misses=None, added=()):
    """Lines telling what each label and keyword of a parsed request found in ``segs``, for reports and previews.

    ``misses`` maps a concept to the best score its candidates reached under ``threshold``; ``added`` names concepts
    the node searched for on its own (person, for background).
    """
    instances = [seg for frame in segs for seg in frame.segments]
    lines, seen = [], set()
    for written, english, status in expression_terms(tree):
        key = (status == "keyword", normalize_label(english))
        if key in seen:
            continue
        seen.add(key)
        if status == "keyword":
            if english == "background":
                names = list(dict.fromkeys(seg.label for seg in instances))
                outside = ", ".join(names) if names else "nothing (no instance was found)"
                lines.append(f"background = everything outside {outside}")
            continue
        name = f"{written} -> {english}" if status == "translated" else written
        found = [seg for seg in instances if label_matches(seg.label, english)]
        if found:
            lines.append(f"{name}: {len(found)} found (best {max(seg.score for seg in found):.2f})")
        elif status == "unknown":
            lines.append(f"{written}: not understood; write it in English (SAM 3 reads English only)")
        elif misses and english in misses and threshold is not None:
            lines.append(f"{name}: none at threshold {threshold:.2f}; the best candidate scored "
                         f"{misses[english]:.2f}, lower the threshold to keep it")
        else:
            lines.append(f"{name}: none found; try another English word")
    for name in added:
        found = [seg for seg in instances if label_matches(seg.label, name)]
        lines.append(f"{name} (searched for background): {len(found)} found")
    return lines


def preview_batch(image, segs):
    """IMAGE batch with every instance of ``segs`` drawn on its image (tinted, outlined, labelled)."""
    if len(segs) != image.shape[0]:
        raise ValueError(
            f"[NH-Nodes] The image batch has {image.shape[0]} image(s) but the segs belong to {len(segs)}; "
            "connect the image the segs were found in."
        )
    pictures = []
    for picture, frame in zip(image, segs):
        if (picture.shape[0], picture.shape[1]) != (frame.height, frame.width):
            raise ValueError(
                f"[NH-Nodes] The image is {picture.shape[1]}x{picture.shape[0]} but the segs were found on a "
                f"{frame.width}x{frame.height} image; connect the image the segs were found in."
            )
        rgb = picture[..., :3] if picture.shape[-1] >= 3 else picture[..., :1].expand(-1, -1, 3)
        pixels = (rgb.float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
        pictures.append(torch.from_numpy(render_preview(pixels, frame).astype(np.float32) / 255.0))
    return torch.stack(pictures)
