#!/usr/bin/env python3
"""Build the NH-Nodes user guides (English and Vietnamese) as PDF files.

Usage:
    python docs/_tools/build_pdf.py [--lang en|vi|all] [--out DIR]

Reads docs/<lang>/README.md, the node pages listed in PAGE_ORDER (pages not listed are appended) and docs/CHANGELOG.md
as an appendix, converts the Markdown with a small built-in parser and lays it out with reportlab Platypus: cover,
clickable table of contents, PDF bookmarks, running header and footer, tables, callouts and code blocks. The node cards
(docs/assets/nodes/*.svg) are embedded as vector graphics through svglib. Output (default dist/ at the repo root):

    NH-Nodes-Guide-EN.pdf
    NH-Nodes-Huong-dan-VI.pdf

Needs reportlab, svglib and Pillow. Nothing in docs/ is modified and the script can be re-run at any time after the
Markdown has changed; warnings (missing images, broken links, characters no font has) are printed at the end. Fonts come
from the system (Segoe UI, Inter, Consolas on Windows; DejaVu as a fallback); set NH_PDF_FONT_DIR to add a font folder.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import io
import json
import os
import re
import sys
import unicodedata
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageDraw
from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import STATE_DEFAULTS, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily, stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Flowable,
    Frame,
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    XPreformatted,
)
from reportlab.platypus.doctemplate import PageBreakIfNotEmpty
from reportlab.platypus.tableofcontents import TableOfContents
from svglib import fonts as svg_fonts
from svglib.svglib import svg2rlg

# --------------------------------------------------------------------------------------------------------------------
# Paths, page geometry, palette
# --------------------------------------------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DOCS = REPO / "docs"
ASSETS = DOCS / "assets"

PAGE_ORDER = [
    "image-io",
    "image-compose",
    "image-preview",
    "image-retouch",
    "resolution",
    "mask",
    "vision",
    "vton",
    "logic",
    "text",
    "batch-lists",
    "loaders",
    "utils",
]

PAGE_W, PAGE_H = A4
MARGIN_X = 54.0
MARGIN_TOP = 68.0
MARGIN_BOTTOM = 62.0
FRAME_W = PAGE_W - 2 * MARGIN_X
FRAME_H = PAGE_H - MARGIN_TOP - MARGIN_BOTTOM
MAX_BLOCK_H = FRAME_H - 64  # tallest picture or card; leaves room for a heading or caption on the same page

TEAL = colors.HexColor("#14b8a6")
TEAL_DARK = colors.HexColor("#0f766e")
INK = colors.HexColor("#111827")
TEXT = colors.HexColor("#1f2937")
TEXT_SOFT = colors.HexColor("#374151")
MUTED = colors.HexColor("#6b7280")
RULE = colors.HexColor("#e3e7eb")
ZEBRA = colors.HexColor("#f6f8fa")
TH_BG = colors.HexColor("#1e2430")
CODE_BG = colors.HexColor("#f3f5f7")
CODE_BORDER = colors.HexColor("#e1e5ea")
CODE_FG = "#0b5d56"
CODE_FG_DARK = "#7ff0df"
CHIP_BG = "#edf1f4"
LINK = "#0f766e"
LINK_DARK = "#5eead4"
COVER_BG = colors.HexColor("#14171c")

CALLOUTS = {
    "tip": (colors.HexColor("#ecfdf5"), colors.HexColor("#14b8a6"), "#0f766e"),
    "note": (colors.HexColor("#eff6ff"), colors.HexColor("#3b82f6"), "#1d4ed8"),
    "warn": (colors.HexColor("#fffbeb"), colors.HexColor("#f59e0b"), "#b45309"),
    "plain": (colors.HexColor("#f3f4f6"), colors.HexColor("#9ca3af"), "#374151"),
}
CALLOUT_WORDS = {
    "tip": ("tip", "hint", "mẹo", "gợi ý"),
    "note": ("note", "info", "lưu ý", "ghi chú"),
    "warn": ("warning", "caution", "important", "licence", "license", "giấy phép", "cảnh báo", "quan trọng"),
}

CARD_SCALE_SIDE = 0.667  # node card next to its description
CARD_SCALE_WIDE = 0.92  # node card shown on its own
MAX_IMAGE_PX = 1800  # raster images wider than this are downscaled
SQUEEZE = (1.0, 0.8, 0.62)  # vertical spacing factors; a chapter whose last page is nearly empty steps down this list
MAX_ATTEMPTS = 6  # layout attempts per language (page-total settling and tail rescue)
TAIL_FILL = 0.17  # a chapter's last page filled less than this (of the frame height) counts as nearly empty

GITHUB_FALLBACK = "https://github.com/jetthuangai/NH-Nodes"
CATEGORY_ORDER = ["Image", "Resolution", "Mask", "Vision", "VTON", "Logic", "Text", "Batch", "Loaders", "Utils"]

LANGS = {
    "en": {
        "pdf": "NH-Nodes-Guide-EN.pdf",
        "tag": "en-US",
        "title": "NH-Nodes \u2014 User Guide",
        "kicker": "USER GUIDE",
        "tagline": "A complete reference to every node: what it is for, each input and output, "
        "and how it behaves in the details that matter.",
        "toc": "Contents",
        "chapter": "Chapter",
        "home": "Overview",
        "appendix": "Appendix",
        "page": "Page",
        "version": "Version",
        "author": "Author",
        "date": "Date",
        "glance": "{n} NODES IN {c} CATEGORIES",
        "changelog_note": None,
        "months": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                   "November", "December"],
        "date_fmt": "{d} {month} {y}",
    },
    "vi": {
        "pdf": "NH-Nodes-Huong-dan-VI.pdf",
        "tag": "vi-VN",
        "title": "NH-Nodes \u2014 Hướng dẫn sử dụng",
        "kicker": "HƯỚNG DẪN SỬ DỤNG",
        "tagline": "Tài liệu tra cứu đầy đủ từng node: dùng để làm gì, từng đầu vào và đầu ra, "
        "và cách node xử lý ở những chi tiết quan trọng.",
        "toc": "Mục lục",
        "chapter": "Chương",
        "home": "Tổng quan",
        "appendix": "Phụ lục",
        "page": "Trang",
        "version": "Phiên bản",
        "author": "Tác giả",
        "date": "Ngày",
        "glance": "{n} NODE TRONG {c} NHÓM",
        "changelog_note": "Nhật ký thay đổi chỉ có bản tiếng Anh.",
        "months": None,
        "date_fmt": "{d} tháng {m} năm {y}",
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------------------------------------------------
# Project metadata
# --------------------------------------------------------------------------------------------------------------------


def read_project_meta() -> dict:
    """Version and repository URL from pyproject.toml, and the node count from the node-card index."""
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    ver = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    repo = re.search(r'^Repository\s*=\s*"([^"]+)"', text, re.M)
    entries: list = []
    index = ASSETS / "nodes" / "index.json"
    if index.is_file():
        try:
            entries = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            entries = []
    counts: dict[str, int] = {}
    for entry in entries:
        parts = str(entry.get("category", "")).split("/")
        name = parts[1] if len(parts) > 1 else parts[0]
        if name:
            counts[name] = counts.get(name, 0) + 1
    categories = [(c, counts[c]) for c in CATEGORY_ORDER if c in counts]
    categories += sorted((c, n) for c, n in counts.items() if c not in CATEGORY_ORDER)
    return {
        "version": ver.group(1) if ver else "",
        "repo": (repo.group(1) if repo else GITHUB_FALLBACK).rstrip("/"),
        "nodes": len(entries) or None,
        "categories": categories,
    }


def format_date(cfg: dict, day: dt.date) -> str:
    month = cfg["months"][day.month - 1] if cfg["months"] else str(day.month)
    return cfg["date_fmt"].format(d=day.day, month=month, m=day.month, y=day.year)


# --------------------------------------------------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------------------------------------------------


def font_dirs() -> list[Path]:
    dirs: list[Path] = []
    extra = os.environ.get("NH_PDF_FONT_DIR")
    if extra:
        dirs.append(Path(extra))
    dirs.append(HERE / "fonts")
    windir = os.environ.get("WINDIR", r"C:\Windows")
    dirs.append(Path(windir) / "Fonts")
    local = os.environ.get("LOCALAPPDATA")
    if local:
        dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    dirs += [
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/dejavu"),
        Path("/usr/share/fonts/TTF"),
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts/Supplemental"),
        Path.home() / "Library" / "Fonts",
    ]
    return [d for d in dirs if d.is_dir()]


# role -> alternatives; each alternative lists the files for the faces of that role (all must exist).
FONT_FILES = {
    # regular, bold, italic, bold italic
    "body": [
        ("segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf", "segoeuiz.ttf"),
        ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf", "DejaVuSans-BoldOblique.ttf"),
    ],
    # semibold (headings), medium (small labels)
    "head": [
        ("Inter-SemiBold.ttf", "Inter-Medium.ttf"),
        ("segoeuib.ttf", "segoeuib.ttf"),
        ("DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"),
    ],
    "mono": [
        ("consola.ttf", "consolab.ttf", "consolai.ttf", "consolaz.ttf"),
        ("DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf", "DejaVuSansMono-Oblique.ttf", "DejaVuSansMono-BoldOblique.ttf"),
    ],
    # optional: used for the node cards and as glyph fallbacks
    "inter": [("Inter-Regular.ttf",)],
    "symbol": [("seguisym.ttf",), ("DejaVuSans.ttf",)],
}


FAMILY_FACES = {
    "NH-Body": ("NH-Body", "NH-Body-Bold", "NH-Body-Italic", "NH-Body-BoldItalic"),
    "NH-Head": ("NH-Head", "NH-Head-Medium"),
    "NH-Mono": ("NH-Mono", "NH-Mono-Bold", "NH-Mono-Italic", "NH-Mono-BoldItalic"),
}


def find_font_file(name: str, dirs: list[Path]) -> Path | None:
    wanted = name.lower()
    for folder in dirs:
        direct = folder / name
        if direct.is_file():
            return direct
        try:
            for entry in folder.iterdir():
                if entry.name.lower() == wanted:
                    return entry
        except OSError:
            continue
    return None


def resolve_font_set(role: str, dirs: list[Path]) -> list[Path] | None:
    for names in FONT_FILES[role]:
        found = [find_font_file(n, dirs) for n in names]
        if all(found):
            return [p for p in found if p]
    return None


class FontKit:
    """Registers the fonts and turns text into reportlab markup, falling back per glyph when a font lacks a character."""

    def __init__(self) -> None:
        self.missing: dict[str, int] = {}
        self._cmaps: dict[str, dict] = {}
        self._pick: dict[tuple[str, str], str | None] = {}
        self.fallbacks: list[str] = []
        self._register()

    def _reg(self, name: str, path: Path) -> None:
        pdfmetrics.registerFont(TTFont(name, str(path)))

    def _register(self) -> None:
        dirs = font_dirs()
        sets = {role: resolve_font_set(role, dirs) for role in FONT_FILES}
        for role in ("body", "head", "mono"):
            if not sets[role]:
                tried = "; ".join(" + ".join(alt) for alt in FONT_FILES[role])
                raise SystemExit(
                    f"error: no usable '{role}' font found (looked for: {tried}).\n"
                    "Install one of them or point NH_PDF_FONT_DIR at a folder with Vietnamese-capable TTF fonts."
                )
        body = sets["body"]
        self._reg("NH-Body", body[0])
        self._reg("NH-Body-Bold", body[1])
        self._reg("NH-Body-Italic", body[2])
        self._reg("NH-Body-BoldItalic", body[3])
        registerFontFamily("NH-Body", normal="NH-Body", bold="NH-Body-Bold", italic="NH-Body-Italic",
                           boldItalic="NH-Body-BoldItalic")
        head = sets["head"]
        self._reg("NH-Head", head[0])
        self._reg("NH-Head-Medium", head[1])
        registerFontFamily("NH-Head", normal="NH-Head", bold="NH-Head", italic="NH-Head", boldItalic="NH-Head")
        registerFontFamily("NH-Head-Medium", normal="NH-Head-Medium", bold="NH-Head", italic="NH-Head-Medium",
                           boldItalic="NH-Head")
        mono = sets["mono"]
        self._reg("NH-Mono", mono[0])
        self._reg("NH-Mono-Bold", mono[1])
        self._reg("NH-Mono-Italic", mono[2])
        self._reg("NH-Mono-BoldItalic", mono[3])
        registerFontFamily("NH-Mono", normal="NH-Mono", bold="NH-Mono-Bold", italic="NH-Mono-Italic",
                           boldItalic="NH-Mono-BoldItalic")
        # Inter regular is used by the node cards; fall back to the body font when it is not installed.
        self._reg("NH-Inter", sets["inter"][0] if sets["inter"] else body[0])
        self.fallbacks = ["NH-Body", "NH-Inter"]
        if sets["symbol"]:
            self._reg("NH-Symbol", sets["symbol"][0])
            self.fallbacks.append("NH-Symbol")
        self.fallbacks += ["NH-Mono", "NH-Head"]

    def cmap(self, name: str) -> set:
        """Code points that every face of the family `name` can draw (so <b> and <i> never hit a missing glyph)."""
        table = self._cmaps.get(name)
        if table is None:
            faces = FAMILY_FACES.get(name, (name,))
            table = set.intersection(*(set(pdfmetrics.getFont(face).face.charToGlyph) for face in faces))
            self._cmaps[name] = table
        return table

    def pick(self, base: str, ch: str) -> str | None:
        """The font to draw `ch` with: `base` when it has the glyph, else the first fallback that does, else None."""
        key = (base, ch)
        if key in self._pick:
            return self._pick[key]
        result: str | None = None
        if ord(ch) in self.cmap(base):
            result = base
        elif unicodedata.category(ch) == "Cf" or 0xFE00 <= ord(ch) <= 0xFE0F:
            result = None  # invisible format characters and variation selectors
        else:
            for name in self.fallbacks:
                if name != base and ord(ch) in self.cmap(name):
                    result = name
                    break
            if result is None:
                self.missing[ch] = self.missing.get(ch, 0) + 1
        self._pick[key] = result
        return result

    def markup(self, text: str, base: str) -> str:
        """Escape `text` for a Paragraph and wrap glyphs that `base` lacks in <font> tags of a fallback font."""
        out: list[str] = []
        run: list[str] = []
        cur = base

        def emit() -> None:
            if run:
                s = esc("".join(run))
                out.append(s if cur == base else f'<font name="{cur}">{s}</font>')
                run.clear()

        for ch in text:
            if ch in "\t\r\n":
                ch = " "
            font = self.pick(base, ch)
            if font is None:
                continue
            if font != cur:
                emit()
                cur = font
            run.append(ch)
        emit()
        return "".join(out)


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------------------------------------------------
# Markdown: blocks
# --------------------------------------------------------------------------------------------------------------------


@dataclass
class HeadingBlk:
    level: int
    text: str
    slug: str = ""
    key: str = ""


@dataclass
class ParaBlk:
    text: str
    align: str = ""


@dataclass
class CodeBlk:
    text: str
    info: str = ""


@dataclass
class ListBlk:
    items: list
    ordered: bool = False
    start: int = 1


@dataclass
class QuoteBlk:
    blocks: list


@dataclass
class TableBlk:
    header: list
    aligns: list
    rows: list


@dataclass
class RuleBlk:
    pass


@dataclass
class FigureBlk:
    src: str
    alt: str = ""
    width: str = ""
    caption: str = ""


FENCE_RE = re.compile(r"^( {0,3})(`{3,}|~{3,})\s*([^`\s]*)[^`]*$")
ATX_RE = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
HR_RE = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
QUOTE_RE = re.compile(r"^ {0,3}>[ ]?(.*)$")
LIST_RE = re.compile(r"^( {0,3})([-*+]|\d{1,9}[.)])(?:([ \t]+)(.*))?$")
TABLE_SEP_RE = re.compile(r"^ {0,3}\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$")
HTML_START_RE = re.compile(
    r"^ {0,3}<(?:p|div|center|img|br|hr|table|details|summary|h[1-6]|picture|figure|figcaption)\b", re.I)
IMG_TAG_RE = re.compile(r"<img\b[^>]*>", re.I)
ATTR_RE = re.compile(r'([\w:-]+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\')')
MD_IMG_RE = re.compile(r"^!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)$")
NAV_RE = re.compile(r"^\[←[^\]]*\]\([^)]*\)$")
LABEL_RE = re.compile(r"^\*\*[^*\n]{1,100}\*\*[:：]?(?:\s*\([^()\n]*\))?$")


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", s)]


def is_table_start(lines: list[str], i: int) -> bool:
    if i + 1 >= len(lines) or "|" not in lines[i] or "|" not in lines[i + 1]:
        return False
    if not TABLE_SEP_RE.match(lines[i + 1]):
        return False
    return len(split_row(lines[i])) == len(split_row(lines[i + 1]))


def starts_block(lines: list[str], i: int) -> bool:
    """True when line i begins a block that may interrupt a paragraph."""
    ln = lines[i]
    if FENCE_RE.match(ln) or HR_RE.match(ln) or QUOTE_RE.match(ln) or HTML_START_RE.match(ln):
        return True
    atx = ATX_RE.match(ln)
    if atx:
        return True
    lm = LIST_RE.match(ln)
    if lm and lm.group(4) and (lm.group(2)[0] not in "0123456789" or lm.group(2)[:-1] == "1"):
        return True
    return is_table_start(lines, i)


def parse_attrs(tag: str) -> dict[str, str]:
    return {m.group(1).lower(): (m.group(2) if m.group(2) is not None else m.group(3)) for m in ATTR_RE.finditer(tag)}


def parse_html_block(raw: list[str]) -> list:
    joined = " ".join(s.strip() for s in raw)
    imgs = IMG_TAG_RE.findall(joined)
    blocks: list = []
    if imgs:
        rest = IMG_TAG_RE.sub("", joined)
        rest = re.sub(r"</?(?:p|div|center)\b[^>]*>", "", rest, flags=re.I)
        rest = re.sub(r"^\s*(?:<br\s*/?>\s*)+", "", rest, flags=re.I).strip()
        for n, tag in enumerate(imgs):
            at = parse_attrs(tag)
            if at.get("src"):
                caption = rest if n == len(imgs) - 1 else ""
                blocks.append(FigureBlk(at["src"], at.get("alt", ""), at.get("width", ""), caption))
        return blocks
    align = ""
    first = re.match(r"\s*<(?:p|div|center)\b([^>]*)>", joined, re.I)
    if first:
        align = parse_attrs("<x " + first.group(1) + ">").get("align", "")
    elif re.match(r"\s*<center\b", joined, re.I):
        align = "center"
    inner = re.sub(r"</?(?:p|div|center)\b[^>]*>", "", joined, flags=re.I).strip()
    if inner:
        blocks.append(ParaBlk(inner, align))
    return blocks


def parse_list(lines: list[str], i: int) -> tuple[ListBlk, int]:
    first = LIST_RE.match(lines[i])
    assert first is not None
    marker = first.group(2)
    ordered = marker[0].isdigit()
    delim = marker[-1] if ordered else marker
    start = int(marker[:-1]) if ordered else 1
    items: list[list] = []
    n = len(lines)
    while i < n:
        m = LIST_RE.match(lines[i])
        if not m:
            break
        mk = m.group(2)
        if mk[0].isdigit() != ordered or (mk[-1] if ordered else mk) != delim:
            break
        body = m.group(4) or ""
        gap = len(m.group(3) or "")
        if gap == 0 or gap > 4 or not body:
            gap = 1
        content_col = len(m.group(1)) + len(mk) + gap
        item_lines = [body]
        i += 1
        while i < n:
            ln = lines[i]
            if not ln.strip():
                j = i
                while j < n and not lines[j].strip():
                    j += 1
                if j >= n:
                    i = j
                    break
                if indent_of(lines[j]) >= content_col:
                    item_lines.extend([""] * (j - i))
                    i = j
                    continue
                i = j
                break
            if indent_of(ln) >= content_col:
                item_lines.append(ln[content_col:])
                i += 1
                continue
            if LIST_RE.match(ln) or starts_block(lines, i):
                break
            if item_lines[-1].strip():
                item_lines.append(ln.strip())  # lazy continuation of the paragraph
                i += 1
                continue
            break
        items.append(parse_blocks(item_lines))
    return ListBlk(items, ordered, start), i


def parse_blocks(lines: list[str]) -> list:
    blocks: list = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.lstrip().startswith("<!--"):
            while i < n and "-->" not in lines[i]:
                i += 1
            i += 1
            continue
        fm = FENCE_RE.match(line)
        if fm:
            indent, fence, info = fm.group(1), fm.group(2), fm.group(3)
            close = re.compile(r"^ {0,3}" + re.escape(fence[0]) + "{%d,}\\s*$" % len(fence))
            j = i + 1
            body: list[str] = []
            while j < n and not close.match(lines[j]):
                body.append(lines[j][len(indent):] if lines[j].startswith(indent) else lines[j].lstrip())
                j += 1
            blocks.append(CodeBlk("\n".join(body), info))
            i = j + 1
            continue
        am = ATX_RE.match(line)
        if am:
            blocks.append(HeadingBlk(len(am.group(1)), (am.group(2) or "").strip()))
            i += 1
            continue
        if HR_RE.match(line):
            blocks.append(RuleBlk())
            i += 1
            continue
        if QUOTE_RE.match(line):
            inner: list[str] = []
            while i < n:
                qm = QUOTE_RE.match(lines[i])
                if qm:
                    inner.append(qm.group(1))
                    i += 1
                elif lines[i].strip() and inner and inner[-1].strip() and not starts_block(lines, i):
                    inner.append(lines[i].strip())
                    i += 1
                else:
                    break
            blocks.append(QuoteBlk(parse_blocks(inner)))
            continue
        lm = LIST_RE.match(line)
        if lm:
            blk, i = parse_list(lines, i)
            blocks.append(blk)
            continue
        if is_table_start(lines, i):
            header = split_row(line)
            aligns = []
            for cell in split_row(lines[i + 1]):
                left, right = cell.startswith(":"), cell.endswith(":")
                aligns.append("center" if left and right else "right" if right else "left")
            rows: list[list[str]] = []
            i += 2
            while i < n and lines[i].strip() and "|" in lines[i] and not ends_table(lines, i):
                row = split_row(lines[i])
                row = (row + [""] * len(header))[: len(header)]
                rows.append(row)
                i += 1
            blocks.append(TableBlk(header, aligns, rows))
            continue
        if HTML_START_RE.match(line):
            raw = [line]
            i += 1
            while i < n and lines[i].strip():
                raw.append(lines[i])
                i += 1
            blocks.extend(parse_html_block(raw))
            continue
        buf = [line.rstrip("\r")]
        i += 1
        while i < n and lines[i].strip() and not starts_block(lines, i):
            buf.append(lines[i].rstrip("\r"))
            i += 1
        text = "\n".join(buf).strip()
        im = MD_IMG_RE.match(text)
        blocks.append(FigureBlk(im.group(2), im.group(1)) if im else ParaBlk(text))
    return blocks


def ends_table(lines: list[str], i: int) -> bool:
    """True when line i starts another block, so it cannot be a table row."""
    ln = lines[i]
    return bool(FENCE_RE.match(ln) or QUOTE_RE.match(ln) or HR_RE.match(ln) or ATX_RE.match(ln))


# --------------------------------------------------------------------------------------------------------------------
# Markdown: inline -> flat token list -> reportlab markup
# --------------------------------------------------------------------------------------------------------------------

ASCII_PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
BARE_URL_RE = re.compile(r"https?://[^\s<>\"'\])]+[^\s<>\"'\]).,;:!?]")
ENTITY_RE = re.compile(r"&(#\d+|#[xX][0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);")
RUN_RE = re.compile(r"[^\\`<\[!*_&\n]+")
AUTOLINK_RE = re.compile(r"<((?:https?://|mailto:)[^<>\s]+)>")
TAG_RE = re.compile(
    r"<(/?)([A-Za-z][A-Za-z0-9]*)((?:\s+[A-Za-z_:][-\w:.]*(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s\"'=<>`]+))?)*)\s*(/?)>")
SILENT_TAGS = {"p", "div", "span", "center", "small", "font", "picture", "figure", "figcaption", "summary", "details"}
TAG_ALIAS = {"em": "i", "i": "i", "strong": "b", "b": "b", "code": "code", "kbd": "kbd", "u": "u", "sup": "sup",
             "sub": "sub", "s": "strike", "strike": "strike", "del": "strike"}


def is_space(ch: str) -> bool:
    return ch == "" or ch.isspace()


def is_punct(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "PS"


@dataclass
class Delim:
    char: str
    count: int
    orig: int
    can_open: bool
    can_close: bool
    open_tags: list = field(default_factory=list)
    close_tags: list = field(default_factory=list)


def find_close_bracket(src: str, i: int) -> int:
    depth = 0
    j = i
    n = len(src)
    while j < n:
        ch = src[j]
        if ch == "\\":
            j += 2
            continue
        if ch == "`":
            k = j
            while k < n and src[k] == "`":
                k += 1
            m = re.compile(r"(?<!`)`{%d}(?!`)" % (k - j)).search(src, k)
            j = m.end() if m else k
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return -1


def parse_destination(src: str, j: int) -> tuple[str, int] | None:
    """Parse '(dest "title")' starting at src[j] == '('. Returns (dest, index after ')') or None."""
    n = len(src)
    k = j + 1
    while k < n and src[k] in " \t\n":
        k += 1
    if k < n and src[k] == "<":
        end = src.find(">", k)
        if end == -1:
            return None
        dest = src[k + 1:end]
        k = end + 1
    else:
        depth = 0
        begin = k
        while k < n:
            ch = src[k]
            if ch == "\\" and k + 1 < n:
                k += 2
                continue
            if ch in " \t\n":
                break
            if ch == "(":
                depth += 1
            elif ch == ")":
                if depth == 0:
                    break
                depth -= 1
            k += 1
        dest = src[begin:k]
    while k < n and src[k] in " \t\n":
        k += 1
    if k < n and src[k] in "\"'(":
        closer = {'"': '"', "'": "'", "(": ")"}[src[k]]
        end = src.find(closer, k + 1)
        if end == -1:
            return None
        k = end + 1
        while k < n and src[k] in " \t\n":
            k += 1
    if k < n and src[k] == ")":
        return re.sub(r"\\([!-/:-@\[-`{-~])", r"\1", dest), k + 1
    return None


def html_tag_tokens(m: re.Match) -> list:
    closing, name, attrs, selfclose = m.group(1), m.group(2).lower(), m.group(3), m.group(4)
    if name == "br":
        return [["br"]]
    if name == "img":
        at = parse_attrs("<img" + attrs + ">")
        return [["img", at.get("alt", ""), at.get("src", "")]]
    if name == "a":
        if closing:
            return [["close", "a"]]
        href = parse_attrs("<a" + attrs + ">").get("href")
        return [["link_open", href]] if href else []
    alias = TAG_ALIAS.get(name)
    if alias is None:
        return []  # known but unneeded tags (p, div, span ...)
    if closing:
        return [["close", alias]]
    return [] if selfclose else [["open", alias]]


def parse_inline(src: str, links: bool = True) -> list:
    """Tokenise inline Markdown/HTML into ['t', s] | ['code', s] | ['open'|'close', tag] | ['br'] | link tokens."""
    src = re.sub(r" {2,}\n", "\\\n", src)
    toks: list = []
    buf: list[str] = []
    n = len(src)
    urls = {m.start(): m.end() for m in BARE_URL_RE.finditer(src)} if links else {}

    def flush() -> None:
        if buf:
            toks.append(["t", "".join(buf)])
            buf.clear()

    i = 0
    while i < n:
        c = src[i]
        if i in urls:
            url = src[i:urls[i]]
            flush()
            toks += [["link_open", url], ["t", url], ["close", "a"]]
            i = urls[i]
            continue
        if c == "\\":
            nxt = src[i + 1] if i + 1 < n else ""
            if nxt == "\n":
                flush()
                toks.append(["br"])
                i += 2
            elif nxt and nxt in ASCII_PUNCT:
                buf.append(nxt)
                i += 2
            else:
                buf.append(c)
                i += 1
            continue
        if c == "`":
            j = i
            while j < n and src[j] == "`":
                j += 1
            m = re.compile(r"(?<!`)`{%d}(?!`)" % (j - i)).search(src, j)
            if m:
                body = src[j:m.start()].replace("\n", " ")
                if len(body) > 2 and body[0] == " " and body[-1] == " " and body.strip():
                    body = body[1:-1]
                flush()
                toks.append(["code", body])
                i = m.end()
            else:
                buf.append(src[i:j])
                i = j
            continue
        if c == "<":
            m = AUTOLINK_RE.match(src, i)
            if m and links:
                flush()
                toks += [["link_open", m.group(1)], ["t", m.group(1)], ["close", "a"]]
                i = m.end()
                continue
            m = TAG_RE.match(src, i)
            if m and (m.group(2).lower() in TAG_ALIAS or m.group(2).lower() in SILENT_TAGS
                      or m.group(2).lower() in ("br", "img", "a")):
                flush()
                toks.extend(html_tag_tokens(m))
                i = m.end()
                continue
            buf.append(c)
            i += 1
            continue
        if c == "[" or (c == "!" and src.startswith("![", i)):
            is_img = c == "!"
            start = i + 1 if is_img else i
            close = find_close_bracket(src, start)
            dest = parse_destination(src, close + 1) if close != -1 and close + 1 < n and src[close + 1] == "(" else None
            if dest and (links or is_img):
                label = src[start + 1:close]
                flush()
                if is_img:
                    toks.append(["img", label, dest[0]])
                else:
                    toks.append(["link_open", dest[0]])
                    toks.extend(parse_inline(label, links=False))
                    toks.append(["close", "a"])
                i = dest[1]
                continue
            buf.append(c)
            i += 1
            continue
        if c in "*_":
            j = i
            while j < n and src[j] == c:
                j += 1
            before = src[i - 1] if i > 0 else ""
            after = src[j] if j < n else ""
            left = not is_space(after) and (not is_punct(after) or is_space(before) or is_punct(before))
            right = not is_space(before) and (not is_punct(before) or is_space(after) or is_punct(after))
            if c == "*":
                can_open, can_close = left, right
            else:
                can_open = left and (not right or is_punct(before))
                can_close = right and (not left or is_punct(after))
            flush()
            toks.append(Delim(c, j - i, j - i, can_open, can_close))
            i = j
            continue
        if c == "&":
            m = ENTITY_RE.match(src, i)
            if m:
                buf.append(html.unescape(m.group(0)))
                i = m.end()
            else:
                buf.append(c)
                i += 1
            continue
        if c == "\n":
            buf.append(" ")
            i += 1
            continue
        if not urls:
            m = RUN_RE.match(src, i)
            if m:
                buf.append(m.group(0))
                i = m.end()
                continue
        buf.append(c)
        i += 1
    flush()
    return finish_emphasis(toks)


def finish_emphasis(toks: list) -> list:
    delims = [t for t in toks if isinstance(t, Delim)]
    for ci, closer in enumerate(delims):
        if not closer.can_close:
            continue
        while closer.count > 0:
            found = -1
            for oi in range(ci - 1, -1, -1):
                op = delims[oi]
                if op.char != closer.char or not op.can_open or op.count <= 0:
                    continue
                if (op.can_close or closer.can_open) and (op.orig + closer.orig) % 3 == 0 and not (
                        op.orig % 3 == 0 and closer.orig % 3 == 0):
                    continue
                found = oi
                break
            if found == -1:
                break
            op = delims[found]
            use = 2 if op.count >= 2 and closer.count >= 2 else 1
            tag = "b" if use == 2 else "i"
            op.count -= use
            closer.count -= use
            op.open_tags.insert(0, ["open", tag])
            closer.close_tags.append(["close", tag])
            for between in delims[found + 1:ci]:
                between.can_open = between.can_close = False
    out: list = []
    for t in toks:
        if isinstance(t, Delim):
            out.extend(t.close_tags)
            if t.count:
                out.append(["t", t.char * t.count])
            out.extend(t.open_tags)
        else:
            out.append(t)
    return out


def plain_text(toks: list) -> str:
    parts = []
    for t in toks:
        if t[0] in ("t", "code"):
            parts.append(t[1])
        elif t[0] == "br":
            parts.append(" ")
        elif t[0] == "img":
            parts.append(t[1])
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def slugify(text: str) -> str:
    """GitHub-style heading anchor."""
    s = unicodedata.normalize("NFC", text).lower()
    s = "".join(ch for ch in s if ch.isalnum() or ch in " -_")
    return s.replace(" ", "-")


def norm_fragment(frag: str) -> str:
    return unicodedata.normalize("NFC", urllib.parse.unquote(frag)).lower()


class Renderer:
    """Turns inline tokens into Paragraph markup for a given base font and size."""

    def __init__(self, kit: FontKit, base: str, size: float, resolve=None, chips: bool = True, dark: bool = False) -> None:
        self.kit = kit
        self.base = base
        self.size = size
        self.resolve = resolve
        self.chips = chips and not dark
        self.code_color = CODE_FG_DARK if dark else CODE_FG
        self.code_scale = 0.98 if dark else 0.9  # light-on-dark monospace needs a little more size to stay readable
        self.link_color = LINK_DARK if dark else LINK

    def code_open(self) -> str:
        size = round(self.size * self.code_scale, 2)
        back = f' backColor="{CHIP_BG}"' if self.chips else ""
        return f'<font name="NH-Mono" size="{size}" color="{self.code_color}"{back}>'

    def markup(self, toks: list) -> str:
        out: list[str] = []
        stack: list[tuple[str, str]] = []  # (name, closing markup)
        code_depth = 0
        for t in toks:
            kind = t[0]
            if kind == "t":
                out.append(self.kit.markup(t[1], "NH-Mono" if code_depth else self.base))
            elif kind == "code":
                out.append(self.code_open() + self.kit.markup(t[1], "NH-Mono") + "</font>")
            elif kind == "br":
                out.append("<br/>")
            elif kind == "img":
                if t[1]:
                    out.append("<i>" + self.kit.markup(t[1], self.base) + "</i>")
            elif kind == "open":
                name = t[1]
                if name in ("code", "kbd"):
                    out.append(self.code_open())
                    stack.append((name, "</font>"))
                    code_depth += 1
                elif name in ("b", "i", "u"):
                    out.append(f"<{name}>")
                    stack.append((name, f"</{name}>"))
                elif name == "sup":
                    out.append("<super>")
                    stack.append((name, "</super>"))
                elif name == "sub":
                    out.append("<sub>")
                    stack.append((name, "</sub>"))
                elif name == "strike":
                    out.append("<strike>")
                    stack.append((name, "</strike>"))
            elif kind == "link_open":
                target = self.resolve(t[1]) if self.resolve else None
                if target:
                    href = "#" + target[1] if target[0] == "int" else target[1]
                    underline = "1" if target[0] == "ext" else "0"
                    href = esc(href).replace('"', "&quot;")
                    out.append(f'<a href="{href}" color="{self.link_color}" underline="{underline}">')
                    stack.append(("a", "</a>"))
                else:
                    stack.append(("a", ""))
            elif kind == "close":
                name = t[1]
                if any(entry[0] == name for entry in stack):
                    while stack:
                        top_name, closer = stack.pop()
                        out.append(closer)
                        if top_name in ("code", "kbd"):
                            code_depth -= 1
                        if top_name == name:
                            break
        while stack:
            out.append(stack.pop()[1])
        return "".join(out)


# --------------------------------------------------------------------------------------------------------------------
# Document model
# --------------------------------------------------------------------------------------------------------------------


@dataclass
class Chapter:
    stem: str
    path: Path
    kind: str  # home | page | appendix
    title: str  # raw Markdown of the H1
    blocks: list
    number: int = 0
    key: str = ""
    anchors: dict = field(default_factory=dict)


class KeyCounter:
    def __init__(self) -> None:
        self.n = 0

    def next(self, prefix: str = "a") -> str:
        self.n += 1
        return f"{prefix}{self.n}"


def is_nav_block(b) -> bool:
    return isinstance(b, ParaBlk) and bool(NAV_RE.match(b.text.strip()))


def load_chapter(path: Path, kind: str, keys: KeyCounter) -> Chapter | None:
    if not path.is_file():
        return None
    text = unicodedata.normalize("NFC", path.read_text(encoding="utf-8-sig"))
    blocks = parse_blocks(text.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4).split("\n"))
    first = next((i for i, b in enumerate(blocks) if isinstance(b, HeadingBlk) and b.level == 1), None)
    if first is None:
        title = path.stem
        body = blocks
    else:
        title = blocks[first].text
        body = blocks[first + 1:]  # anything before the H1 (banner, language switch) is web-only
    body = [b for b in body if not is_nav_block(b)]
    while body and isinstance(body[-1], RuleBlk):
        body.pop()
    chapter = Chapter(path.stem, path, kind, title, body, key=keys.next("c"))
    chapter.anchors[slugify(plain_text(parse_inline(title)))] = chapter.key
    seen: dict[str, int] = {}
    for b in body:
        if isinstance(b, HeadingBlk):
            slug = slugify(plain_text(parse_inline(b.text)))
            count = seen.get(slug, 0)
            seen[slug] = count + 1
            b.slug = slug if count == 0 else f"{slug}-{count}"
            b.key = keys.next("h")
            chapter.anchors.setdefault(b.slug, b.key)
    return chapter


def load_chapters(lang: str, keys: KeyCounter, warnings: list[str]) -> list[Chapter]:
    folder = DOCS / lang
    chapters: list[Chapter] = []
    home = load_chapter(folder / "README.md", "home", keys)
    if home is None:
        raise SystemExit(f"error: {folder / 'README.md'} not found")
    chapters.append(home)
    stems = list(PAGE_ORDER)
    extra = sorted(p.stem for p in folder.glob("*.md") if p.stem != "README" and p.stem not in PAGE_ORDER)
    for stem in extra:
        warnings.append(f"{lang}/{stem}.md is not in the page order; appended after the known pages")
    number = 0
    for stem in stems + extra:
        ch = load_chapter(folder / f"{stem}.md", "page", keys)
        if ch is None:
            warnings.append(f"{lang}/{stem}.md is missing; skipped")
            continue
        number += 1
        ch.number = number
        chapters.append(ch)
    log_ch = load_chapter(DOCS / "CHANGELOG.md", "appendix", keys)
    if log_ch is not None:
        chapters.append(log_ch)
    return chapters


class Links:
    """Resolves Markdown link targets to in-PDF destinations or external URLs."""

    def __init__(self, chapters: list[Chapter], repo_url: str, warnings: list[str]) -> None:
        self.by_path = {c.path.resolve(): c for c in chapters}
        self.repo_url = repo_url
        self.warnings = warnings
        self.warned: set[tuple[str, str]] = set()

    def warn(self, chapter: Chapter, href: str, why: str) -> None:
        if (chapter.stem, href) not in self.warned:
            self.warned.add((chapter.stem, href))
            self.warnings.append(f"{chapter.stem}.md: link '{href}' {why}")

    def resolve(self, href: str, chapter: Chapter):
        href = href.strip()
        if not href:
            return None
        if href.startswith("#"):
            key = chapter.anchors.get(norm_fragment(href[1:]))
            if key:
                return ("int", key)
            self.warn(chapter, href, "points to a missing heading")
            return None
        parts = urllib.parse.urlsplit(href)
        if parts.scheme in ("http", "https", "mailto"):
            return ("ext", href)
        if parts.scheme:
            return None
        target = (chapter.path.parent / urllib.parse.unquote(parts.path)).resolve()
        if not target.exists():
            self.warn(chapter, href, "points to a file that does not exist")
            return None
        target_chapter = self.by_path.get(target)
        if target_chapter is not None:
            if parts.fragment:
                key = target_chapter.anchors.get(norm_fragment(parts.fragment))
                if key:
                    return ("int", key)
                self.warn(chapter, href, "points to a missing heading; linked to the page start")
            return ("int", target_chapter.key)
        try:
            rel = target.relative_to(REPO.resolve()).as_posix()
        except ValueError:
            return None
        kind = "tree" if target.is_dir() else "blob"
        url = f"{self.repo_url}/{kind}/main/{rel}"
        if parts.fragment:
            url += "#" + parts.fragment
        return ("ext", url)


# --------------------------------------------------------------------------------------------------------------------
# SVG node cards
# --------------------------------------------------------------------------------------------------------------------

SVG_SANS = ("Inter", "Segoe UI", "Roboto", "Helvetica", "Arial", "sans-serif", "system-ui")
SVG_MONO = ("JetBrains Mono", "Consolas", "Courier New", "Courier", "monospace", "Menlo")


def register_card_fonts() -> None:
    """Map the font families of the SVG cards onto the embedded fonts, so svglib never falls back to Times/Helvetica."""
    table = getattr(svg_fonts.get_global_font_map(), "_map", None)
    if table is None:
        raise SystemExit("error: this svglib version has no font map to extend; install svglib 1.5.x")

    def put(family: str, weight: str, style: str, rlg: str) -> None:
        key = svg_fonts.FontMap.build_internal_name(family, weight, style)
        table[key] = {"svg_family": family, "svg_weight": weight, "svg_style": style, "rlgFont": rlg, "exact": True}

    for fam in SVG_SANS:
        for weight in ("normal", "400", "500"):
            put(fam, weight, "normal", "NH-Inter")
            put(fam, weight, "italic", "NH-Body-Italic")
        for weight in ("600", "700", "bold"):
            put(fam, weight, "normal", "NH-Head")
            put(fam, weight, "italic", "NH-Head")
    for fam in SVG_MONO:
        for weight in ("normal", "400", "500"):
            put(fam, weight, "normal", "NH-Mono")
            put(fam, weight, "italic", "NH-Mono-Italic")
        for weight in ("600", "700", "bold"):
            put(fam, weight, "normal", "NH-Mono-Bold")
            put(fam, weight, "italic", "NH-Mono-BoldItalic")
    STATE_DEFAULTS["fontName"] = "NH-Inter"  # keeps renderPDF from referencing the non-embedded Times-Roman


def fix_card_glyphs(group, kit: FontKit) -> None:
    """Split SVG text runs wherever the font lacks a glyph (arrows, symbols) so each run uses a font that has it."""
    new: list = []
    for node in group.contents:
        if isinstance(node, String) and node.text:
            runs: list[tuple[str, str]] = []
            for ch in node.text:
                font = kit.pick(node.fontName, ch)
                if font is None:
                    continue
                if runs and runs[-1][1] == font:
                    runs[-1] = (runs[-1][0] + ch, font)
                else:
                    runs.append((ch, font))
            if len(runs) == 1 and runs[0][1] == node.fontName and runs[0][0] == node.text:
                new.append(node)
                continue
            widths = [stringWidth(t, f, node.fontSize) for t, f in runs]
            total = sum(widths)
            anchor = getattr(node, "textAnchor", "start")
            x = node.x - (total if anchor == "end" else total / 2 if anchor == "middle" else 0)
            for (text, font), width in zip(runs, widths):
                piece = node.copy()
                piece.text, piece.fontName, piece.x, piece.textAnchor = text, font, x, "start"
                new.append(piece)
                x += width
        else:
            if hasattr(node, "contents"):
                fix_card_glyphs(node, kit)
            new.append(node)
    group.contents[:] = new


class SvgCard(Flowable):
    """A node card (SVG, vector) drawn at a fixed scale with a soft shadow."""

    def __init__(self, drawing, scale: float, align: str = "LEFT") -> None:
        super().__init__()
        self.drawing = drawing
        self.scale = min(scale, MAX_BLOCK_H / drawing.height)  # a card must always fit on one page
        scale = self.scale
        self.width = drawing.width * scale
        self.height = drawing.height * scale
        self.hAlign = align

    def wrap(self, aw, ah):
        return self.width, self.height

    def draw(self):
        canv = self.canv
        r = 10 * self.scale
        canv.saveState()
        for grow, alpha in ((2.6, 0.035), (1.6, 0.05), (0.8, 0.07)):
            canv.setFillColor(colors.Color(0.05, 0.08, 0.12, alpha))
            canv.roundRect(-grow + 0.6, -grow - 1.6, self.width + 2 * grow - 1.2, self.height + 2 * grow - 1.0, r + grow,
                           stroke=0, fill=1)
        canv.restoreState()
        canv.saveState()
        canv.scale(self.scale, self.scale)
        renderPDF.draw(self.drawing, canv, 0, 0)
        canv.restoreState()


# --------------------------------------------------------------------------------------------------------------------
# Custom flowables
# --------------------------------------------------------------------------------------------------------------------


def register_heading(canv, key: str, level: int, plain: str, toc_markup: str | None, y: float) -> None:
    """Create the bookmark, the PDF outline entry and (optionally) the table-of-contents entry for a heading."""
    doc = canv._doctemplate
    canv.bookmarkHorizontal(key, 0, y)
    lvl = max(0, min(level, doc.outline_level + 1))
    canv.addOutlineEntry(plain or " ", key, lvl, closed=lvl >= 1)
    doc.outline_level = lvl
    if toc_markup is not None:
        doc.notify("TOCEntry", (level, toc_markup, doc.page, key))


class Heading(Paragraph):
    """A heading that registers itself for the outline/TOC when it is drawn, with an optional accent bar."""

    def __init__(self, text, style, key, level, plain, toc_markup=None, bar=False, chapter=None, stem=""):
        super().__init__(text, style)
        self.stem = stem
        self.key = key
        self.level = level
        self.plain = plain
        self.toc_markup = toc_markup
        self.bar = bar
        self.chapter = chapter

    def split(self, aw, ah):
        return []

    def draw(self):
        canv = self.canv
        register_heading(canv, self.key, self.level, self.plain, self.toc_markup, self.height + 6)
        if self.chapter is not None:
            canv._doctemplate.header_chapter = self.chapter
            canv._doctemplate.current_stem = self.stem
            canv._doctemplate.opener_on_page = True
        canv.saveState()
        if self.bar:
            canv.setFillColor(TEAL)
            canv.roundRect(0, 1.2, 3.2, self.height - 3.0, 1.4, stroke=0, fill=1)
        canv.restoreState()
        super().draw()


class ChapterOpener(Flowable):
    """The title block that starts a chapter: label, large title, optional subtitle and a rule."""

    def __init__(self, label, title_markup, sub_markup, styles, key, plain, toc_markup, header_title, stem):
        super().__init__()
        self.label = label
        self.stem = stem
        self.title = Paragraph(title_markup, styles["ch_title"])
        self.sub = Paragraph(sub_markup, styles["ch_sub"]) if sub_markup else None
        self.key = key
        self.plain = plain
        self.toc_markup = toc_markup
        self.header_title = header_title
        self.label_font = ("NH-Head", 8.6)

    def wrap(self, aw, ah):
        self.aw = aw
        _, th = self.title.wrap(aw, 1000)
        sh = self.sub.wrap(aw, 1000)[1] if self.sub else 0
        self.th, self.sh = th, sh
        self.height = 6 + 22 + th + (6 + sh if sh else 0) + 20
        self.width = aw
        return aw, self.height

    def draw(self):
        canv = self.canv
        doc = canv._doctemplate
        register_heading(canv, self.key, 0, self.plain, self.toc_markup, self.height)
        doc.header_chapter = self.header_title
        doc.current_stem = self.stem
        doc.opener_on_page = True
        top = self.height
        canv.saveState()
        canv.setFillColor(TEAL)
        canv.roundRect(0, top - 4, 38, 3.4, 1.6, stroke=0, fill=1)
        font, size = self.label_font
        spaced_text(canv, 0, top - 22, self.label, font, size, TEAL_DARK, 1.7)
        y = top - 22 - 8 - self.th
        self.title.drawOn(canv, 0, y)
        if self.sub is not None:
            y -= 6 + self.sh
            self.sub.drawOn(canv, 0, y)
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.8)
        canv.line(0, 5, self.aw, 5)
        canv.restoreState()


def stack_flowables(canv, flowables: list, width: float):
    """Wrap `flowables` for `width`; returns ([(flowable, height, gap_before)], total_height)."""
    placed = []
    total = 0.0
    prev_after = 0.0
    for f in flowables:
        w, h = f.wrapOn(canv, width, 100000)
        before = f.getSpaceBefore()
        gap = 0.0 if not placed else max(before, prev_after)
        placed.append((f, h, gap))
        total += gap + h
        prev_after = f.getSpaceAfter()
    return placed, total


class CalloutBox(Flowable):
    """A tinted box with a coloured bar on the left (tips, notes, licence remarks)."""

    PAD_X = 11.0

    def __init__(self, flowables: list, bg, bar, pad_y: float = 7.5) -> None:
        super().__init__()
        self.content = flowables
        self.bg = bg
        self.bar = bar
        self.pad_y = pad_y

    def wrap(self, aw, ah):
        self.width = aw
        self.placed, inner = stack_flowables(self.canv, self.content, aw - 2 * self.PAD_X - 3)
        self.height = inner + 2 * self.pad_y
        return aw, self.height

    def split(self, aw, ah):
        """Break an over-long box between its paragraphs so it can continue on the next page."""
        if self.height <= ah or len(self.content) < 2:
            return []
        used = 2 * self.pad_y
        count = 0
        for _flow, h, gap in self.placed:
            if used + gap + h > ah:
                break
            used += gap + h
            count += 1
        if count == 0:
            return []
        return [CalloutBox(self.content[:count], self.bg, self.bar, self.pad_y),
                CalloutBox(self.content[count:], self.bg, self.bar, self.pad_y)]

    def draw(self):
        canv = self.canv
        canv.saveState()
        canv.setFillColor(self.bg)
        canv.roundRect(0, 0, self.width, self.height, 4.5, stroke=0, fill=1)
        path = canv.beginPath()
        path.roundRect(0, 0, self.width, self.height, 4.5)
        canv.clipPath(path, stroke=0, fill=0)
        canv.setFillColor(self.bar)
        canv.rect(0, 0, 3.4, self.height, stroke=0, fill=1)
        canv.restoreState()
        y = self.height - self.pad_y
        for f, h, gap in self.placed:
            y -= gap + h
            f.drawOn(canv, self.PAD_X + 3, y)


class RasterFigure(Flowable):
    """A raster image scaled to the frame width, with rounded corners and a hairline border."""

    def __init__(self, reader: ImageReader, max_width: float) -> None:
        super().__init__()
        iw, ih = reader.getSize()
        self.reader = reader
        self.width = min(max_width, MAX_BLOCK_H * iw / ih)  # a tall picture is scaled down to fit one page
        self.height = self.width * ih / iw
        self.hAlign = "CENTER"

    def wrap(self, aw, ah):
        if self.width > aw:
            self.height *= aw / self.width
            self.width = aw
        return self.width, self.height

    def draw(self):
        canv = self.canv
        canv.saveState()
        path = canv.beginPath()
        path.roundRect(0, 0, self.width, self.height, 4)
        canv.clipPath(path, stroke=0, fill=0)
        canv.drawImage(self.reader, 0, 0, self.width, self.height, mask="auto")  # honours PNG transparency
        canv.restoreState()
        canv.saveState()
        canv.setStrokeColor(colors.HexColor("#d5dbe1"))
        canv.setLineWidth(0.7)
        canv.roundRect(0, 0, self.width, self.height, 4, stroke=1, fill=0)
        canv.restoreState()


# --------------------------------------------------------------------------------------------------------------------
# Styles
# --------------------------------------------------------------------------------------------------------------------


def make_styles(squeeze: float = 1.0) -> dict[str, ParagraphStyle]:
    """All paragraph styles; `squeeze` < 1 tightens the vertical spacing (used to rescue near-empty last pages)."""
    s: dict[str, ParagraphStyle] = {}

    def add(name: str, parent: str | None = None, **kw) -> None:
        s[name] = ParagraphStyle(name, parent=s[parent] if parent else None, **kw)

    add("body", fontName="NH-Body", fontSize=9.6, leading=14.4, textColor=TEXT, spaceAfter=6.5, allowWidows=0,
        allowOrphans=0, alignment=TA_LEFT, splitLongWords=1)
    add("lead", "body", fontSize=10.7, leading=16.2, textColor=TEXT_SOFT, spaceAfter=10)
    add("li", "body", spaceAfter=2.6)
    add("center", "body", alignment=TA_CENTER)
    add("cap", "body", fontName="NH-Body-Italic", fontSize=8.4, leading=11.8, textColor=MUTED, alignment=TA_CENTER,
        spaceBefore=3, spaceAfter=11)
    add("callout", "body", fontSize=9.2, leading=13.4, spaceAfter=3)
    add("h2", fontName="NH-Head", fontSize=15.5, leading=19.5, textColor=INK, spaceBefore=24, spaceAfter=9,
        leftIndent=11)
    add("h3", fontName="NH-Head", fontSize=11.8, leading=15.5, textColor=TEAL_DARK, spaceBefore=14, spaceAfter=4.5)
    add("h4", fontName="NH-Head-Medium", fontSize=10.2, leading=13.5, textColor=INK, spaceBefore=10, spaceAfter=3)
    add("label", fontName="NH-Head", fontSize=9.6, leading=13, textColor=INK, spaceBefore=10, spaceAfter=3.4)
    for align_name, align in (("left", TA_LEFT), ("center", TA_CENTER), ("right", TA_RIGHT)):
        add(f"td_{align_name}", fontName="NH-Body", fontSize=8.5, leading=11.7, textColor=TEXT, alignment=align,
            splitLongWords=1)
        add(f"th_{align_name}", fontName="NH-Head", fontSize=8.1, leading=10.6, textColor=colors.white,
            alignment=align, splitLongWords=1)
    add("code", fontName="NH-Mono", fontSize=8.0, leading=11.1, textColor=colors.HexColor("#1f2937"),
        backColor=CODE_BG, borderColor=CODE_BORDER, borderWidth=0.6, borderPadding=(6, 8, 6, 8), borderRadius=4,
        leftIndent=8, rightIndent=8, spaceBefore=6, spaceAfter=13, allowWidows=0, allowOrphans=0)
    add("ch_title", fontName="NH-Head", fontSize=26, leading=31.5, textColor=INK)
    add("ch_sub", fontName="NH-Body", fontSize=12.5, leading=17, textColor=MUTED)
    add("toc_title", fontName="NH-Head", fontSize=26, leading=31.5, textColor=INK, spaceAfter=14)
    add("toc0", fontName="NH-Head", fontSize=10.6, leading=14.2, textColor=INK, spaceBefore=10, rightIndent=30)
    add("toc1", fontName="NH-Body", fontSize=9, leading=12.6, textColor=TEXT_SOFT, leftIndent=15, rightIndent=30)
    if squeeze != 1.0:
        for name, style in s.items():
            if name.startswith("toc") and name != "toc_title":
                style.leading *= 0.55 + 0.45 * squeeze
                style.spaceBefore *= squeeze
            elif not name.startswith(("td_", "th_", "ch_")):
                style.spaceBefore *= squeeze
                style.spaceAfter *= squeeze
    return s


# --------------------------------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------------------------------

CELL_PAD_X = 6.0


def cap_widths(widths: list[float], total: float) -> list[float]:
    """Lower the widest entries to one common cap so that the widths add up to `total`."""
    lo, hi = 0.0, max(widths)
    for _ in range(40):
        mid = (lo + hi) / 2
        if sum(min(w, mid) for w in widths) <= total:
            lo = mid
        else:
            hi = mid
    return [min(w, lo) for w in widths]


def table_widths(rows: list[list[Paragraph]], total: float, pad: float) -> list[float]:
    """Pick column widths: every column at least as wide as its longest word, short columns at their natural width."""
    ncols = len(rows[0])
    mins = [0.0] * ncols
    nats = [0.0] * ncols
    for row in rows:
        for j, p in enumerate(row):
            p.wrap(1e5, 1e5)
            lines = p.getActualLineWidths0() or [0.0]
            nats[j] = max(nats[j], max(lines))
            mins[j] = max(mins[j], p.minWidth())
    mins = [m + 2 * pad + 0.8 for m in mins]
    if sum(mins) > total:
        mins = cap_widths(mins, total)  # an unbreakable word wider than its share is split instead
    nats = [max(nat + 2 * pad + 1.0, mn) for nat, mn in zip(nats, mins)]
    if sum(nats) <= total:
        scale = total / sum(nats)
        return [w * scale for w in nats]

    def used(cap: float) -> float:
        return sum(min(nat, max(mn, cap)) for nat, mn in zip(nats, mins))

    lo, hi = 0.0, max(nats)
    for _ in range(40):
        mid = (lo + hi) / 2
        if used(mid) <= total:
            lo = mid
        else:
            hi = mid
    widths = [min(nat, max(mn, lo)) for nat, mn in zip(nats, mins)]
    capped = [j for j in range(ncols) if nats[j] > widths[j] + 1e-6]
    spare = total - sum(widths)
    if capped and spare > 0:
        for j in capped:
            widths[j] += spare / len(capped)
    return widths


def wrap_code_line(line: str, limit: int) -> list[str]:
    if len(line) <= limit:
        return [line]
    indent = indent_of(line)
    cont = " " * (indent + 4)
    out: list[str] = []
    cur = line
    prefix = indent
    while len(cur) > limit:
        cut = cur.rfind(" ", prefix + 1, limit + 1)
        if cut <= prefix:
            cut = limit
        out.append(cur[:cut].rstrip())
        cur = cont + cur[cut:].lstrip()
        prefix = len(cont)
    out.append(cur)
    return out


# --------------------------------------------------------------------------------------------------------------------
# Story builder
# --------------------------------------------------------------------------------------------------------------------


ATOMIC_KINDS = ("card", "figure", "callout", "code_small")


@dataclass
class Item:
    kind: str
    flows: list


class StoryBuilder:
    def __init__(self, lang: str, kit: FontKit, chapters: list[Chapter], links: Links, meta: dict,
                 warnings: list[str], squeeze: dict[str, int] | None = None, caches: tuple | None = None) -> None:
        self.lang = lang
        self.cfg = LANGS[lang]
        self.kit = kit
        self.chapters = chapters
        self.links = links
        self.meta = meta
        self.warnings = warnings
        self.squeeze = squeeze or {}
        self.style_sets = [make_styles(f) for f in SQUEEZE]
        self.styles = self.style_sets[0]
        self.sq = 1.0
        self.indent = 0.0  # horizontal room taken by the enclosing lists
        self.list_depth = 0
        self.cur: Chapter = chapters[0]
        self.svg_cache, self.img_cache = caches or ({}, {})

    # ---- inline helpers -------------------------------------------------------------------------------------------

    def base_font(self, style: ParagraphStyle) -> str:
        name = style.fontName
        if name.startswith("NH-Head"):
            return "NH-Head"
        if name.startswith("NH-Mono"):
            return "NH-Mono"
        return "NH-Body"

    def md(self, text: str, style: ParagraphStyle, resolve: bool = True, chips: bool = True, dark: bool = False) -> str:
        chapter = self.cur

        def resolver(href: str):
            return self.links.resolve(href, chapter)

        fn = resolver if resolve else None
        renderer = Renderer(self.kit, self.base_font(style), style.fontSize, fn, chips, dark)
        return renderer.markup(parse_inline(text))

    def para(self, markup: str, style: ParagraphStyle) -> Paragraph:
        try:
            return Paragraph(markup, style)
        except Exception as exc:  # reportlab raises on malformed markup; never lose the text
            self.warnings.append(f"{self.cur.stem}.md: markup error ({exc}); text added without formatting")
            plain = re.sub(r"<[^>]*>", "", markup)
            return Paragraph(esc(html.unescape(plain)), style)

    def md_para(self, text: str, style: ParagraphStyle, **kw) -> Paragraph:
        return self.para(self.md(text, style, **kw), style)

    # ---- resources ------------------------------------------------------------------------------------------------

    def asset_path(self, src: str) -> Path:
        return (self.cur.path.parent / urllib.parse.unquote(urllib.parse.urlsplit(src).path)).resolve()

    def card(self, path: Path):
        if path not in self.svg_cache:
            try:
                drawing = svg2rlg(str(path))
            except Exception as exc:
                drawing = None
                self.warnings.append(f"{path.name}: cannot convert SVG ({exc})")
            if drawing is not None:
                fix_card_glyphs(drawing, self.kit)
            self.svg_cache[path] = drawing
        return self.svg_cache[path]

    def raster(self, path: Path) -> ImageReader | None:
        if path in self.img_cache:
            return self.img_cache[path]
        reader = None
        try:
            with PILImage.open(path) as im:
                if im.width > MAX_IMAGE_PX:  # keeps the PDF small; pictures with transparency stay PNG
                    size = (MAX_IMAGE_PX, round(im.height * MAX_IMAGE_PX / im.width))
                    alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
                    small = im.convert("RGBA" if alpha else "RGB").resize(size, PILImage.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    if alpha:
                        small.save(buf, "PNG", optimize=True)
                    else:
                        small.save(buf, "JPEG", quality=88, optimize=True)
                    buf.seek(0)
                    reader = ImageReader(buf)
            if reader is None:
                reader = ImageReader(str(path))
        except (OSError, ValueError) as exc:
            self.warnings.append(f"{path.name}: cannot read image ({exc})")
        self.img_cache[path] = reader
        return reader

    # ---- blocks ---------------------------------------------------------------------------------------------------

    def build_chapter(self, ch: Chapter) -> list:
        self.cur = ch
        level = self.squeeze.get(ch.stem, 0)
        self.styles = self.style_sets[level]
        self.sq = SQUEEZE[level]
        cfg = self.cfg
        if ch.kind == "home":
            label = cfg["home"]
        elif ch.kind == "appendix":
            label = f"{cfg['appendix']} A"
        else:
            label = f"{cfg['chapter']} {ch.number}"
        title_md, gloss = ch.title, ""
        m = re.match(r"^(.*?)\s*\(([^()]+)\)\s*$", ch.title)
        if m and ch.kind == "page":
            title_md, gloss = m.group(1), m.group(2)
        st = self.styles["ch_title"]
        title_markup = self.md(title_md, st, resolve=False)
        sub_markup = self.md(gloss, self.styles["ch_sub"], resolve=False) if gloss else ""
        toc_style = self.styles["toc0"]
        toc_markup = self.md(title_md, toc_style, resolve=False, chips=False)
        if gloss:
            toc_markup += (' <font name="NH-Body" size="8.6" color="#6b7280">'
                           + self.md(gloss, self.styles["toc1"], resolve=False, chips=False) + "</font>")
        plain = plain_text(parse_inline(title_md))
        header_title = plain
        flows: list = [PageBreakIfNotEmpty()]
        flows.append(ChapterOpener(label.upper(), title_markup, sub_markup, self.styles, ch.key,
                                   plain + (f" ({plain_text(parse_inline(gloss))})" if gloss else ""), toc_markup,
                                   header_title, ch.stem))
        blocks = list(ch.blocks)
        if ch.kind == "appendix" and cfg["changelog_note"]:
            blocks.insert(0, ParaBlk(cfg["changelog_note"]))
        first_para = next((i for i, b in enumerate(blocks) if isinstance(b, ParaBlk)), None)
        items = self.build_items(blocks, lead_index=first_para)
        flows.extend(self.keep_tail(self.join(items)))
        return flows

    def keep_tail(self, flows: list) -> list:
        """Keep a chapter's closing one-liner (such as "See also") on the same page as the block above it."""
        if len(flows) < 3 or type(flows[-1]) is not Paragraph:
            return flows
        tail = [flows[-1]]
        k = len(flows) - 2
        if isinstance(flows[k], HRFlowable):
            tail.insert(0, flows[k])
            k -= 1
        prev = flows[k]
        if isinstance(prev, (Paragraph, XPreformatted)) and not isinstance(prev, Heading):
            if prev.wrap(FRAME_W, 10000)[1] < 140:
                tail.insert(0, prev)
                k -= 1
        flows[k + 1:] = [KeepTogether(tail)]
        return flows

    def build_items(self, blocks: list, lead_index: int | None = None, in_list: bool = False) -> list[Item]:
        items: list[Item] = []
        i = 0
        while i < len(blocks):
            b = blocks[i]
            nxt = blocks[i + 1] if i + 1 < len(blocks) else None
            if isinstance(b, RuleBlk):
                if isinstance(nxt, HeadingBlk) and nxt.level <= 2 or nxt is None or isinstance(nxt, RuleBlk):
                    i += 1
                    continue
                gap = 8 * self.sq
                items.append(Item("rule", [HRFlowable(width="100%", thickness=0.7, color=RULE, spaceBefore=gap,
                                                      spaceAfter=gap)]))
                i += 1
                continue
            if isinstance(b, FigureBlk) and b.src.lower().endswith(".svg") and not in_list:
                paras = []
                j = i + 1
                while j < len(blocks) and isinstance(blocks[j], ParaBlk) and len(paras) < 2 and not blocks[
                        j].align and not LABEL_RE.match(blocks[j].text.strip()):
                    paras.append(blocks[j])
                    j += 1
                item = self.card_item(b, paras)
                if item is not None:
                    items.append(item)
                    i = j
                    continue
            items.append(self.block_item(b, lead=(i == lead_index), in_list=in_list))
            i += 1
        return [it for it in items if it.flows]

    def block_item(self, b, lead: bool = False, in_list: bool = False) -> Item:
        S = self.styles
        if isinstance(b, HeadingBlk):
            return self.heading_item(b)
        if isinstance(b, ParaBlk):
            text = b.text.strip()
            if not in_list and LABEL_RE.match(text):
                return Item("label", [self.md_para(text, S["label"])])
            style = S["li"] if in_list else S["lead"] if lead else S["center"] if b.align == "center" else S["body"]
            return Item("para", [self.md_para(text, style)])
        if isinstance(b, ListBlk):
            flow = self.list_flow(b)
            return Item("list", [flow] if flow else [])
        if isinstance(b, TableBlk):
            return Item("table", [self.table_flow(b)])
        if isinstance(b, CodeBlk):
            if not b.text.strip():
                return Item("none", [])
            flow = self.code_flow(b)
            return Item("code_small" if flow.nlines <= 16 else "code", [flow])
        if isinstance(b, QuoteBlk):
            return Item("callout", [self.callout_flow(b)])
        if isinstance(b, FigureBlk):
            return self.figure_item(b)
        return Item("none", [])

    def heading_item(self, b: HeadingBlk) -> Item:
        S = self.styles
        level = b.level
        style = S["h2"] if level <= 2 else S["h3"] if level == 3 else S["h4"]
        markup = self.md(b.text, style)
        plain = plain_text(parse_inline(b.text))
        toc_markup = None
        if level <= 2:
            toc_markup = self.md(b.text, S["toc1"], resolve=False, chips=False)
        outline_level = min(level - 1, 3)
        heading = Heading(markup, style, b.key, outline_level, plain, toc_markup, bar=level <= 2)
        return Item("heading", [heading])

    def list_flow(self, b: ListBlk) -> ListFlowable | None:
        depth = self.list_depth
        indent = 15 if depth == 0 else 14
        items = []
        self.list_depth += 1
        self.indent += indent  # blocks inside the items (tables, code, images) have less room
        try:
            for blocks in b.items:
                flows = [f for it in self.build_items(blocks, in_list=True) for f in it.flows]
                if flows:
                    items.append(ListItem(flows, value=b.start) if b.ordered and not items else ListItem(flows))
        finally:
            self.list_depth -= 1
            self.indent -= indent
        if not items:
            return None
        common = dict(leftIndent=indent, bulletFontSize=9.4, bulletColor=TEAL_DARK, spaceBefore=1 * self.sq,
                      spaceAfter=(self.styles["body"].spaceAfter - 0.5) if depth == 0 else 2 * self.sq)
        if b.ordered:
            return ListFlowable(items, bulletType="1", bulletFormat="%s.", bulletFontName="NH-Head", **common)
        return ListFlowable(items, bulletType="bullet", start="•" if depth == 0 else "–", bulletFontName="NH-Body",
                            **common)

    def table_flow(self, b: TableBlk):
        S = self.styles
        ncols = len(b.header)
        aligns = (list(b.aligns) + ["left"] * ncols)[:ncols]
        head = [self.md_para(t, S[f"th_{a}"], dark=True) for t, a in zip(b.header, aligns)]
        body = [[self.md_para(t, S[f"td_{a}"]) for t, a in zip(row, aligns)] for row in b.rows]
        rows = [head] + body
        widths = table_widths(rows, FRAME_W - self.indent, CELL_PAD_X)
        table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT", spaceBefore=3 * self.sq,
                      spaceAfter=10 * self.sq)
        pad = 0.45 + 0.55 * self.sq
        cmds = [
            ("FONTNAME", (0, 0), (-1, -1), "NH-Body"),  # tables otherwise start from the non-embedded Helvetica
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), CELL_PAD_X),
            ("RIGHTPADDING", (0, 0), (-1, -1), CELL_PAD_X),
            ("TOPPADDING", (0, 0), (-1, -1), 4.3 * pad),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4.9 * pad),
            ("TOPPADDING", (0, 0), (-1, 0), 6),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6.4),
            ("BACKGROUND", (0, 0), (-1, 0), TH_BG),
            ("LINEBELOW", (0, 0), (-1, 0), 1.6, TEAL),
        ]
        if body:
            cmds += [
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ZEBRA]),
                ("LINEBELOW", (0, 1), (-1, -1), 0.45, RULE),
            ]
        table.setStyle(TableStyle(cmds))
        return table

    def code_flow(self, b: CodeBlk):
        S = self.styles
        style = S["code"]
        char_w = pdfmetrics.stringWidth("0", style.fontName, style.fontSize)
        limit = int((FRAME_W - self.indent - style.leftIndent - style.rightIndent - 2) / char_w)
        lines: list[str] = []
        for raw in b.text.split("\n"):
            lines.extend(wrap_code_line(raw.rstrip(), limit))
        markup = "\n".join(self.kit.markup(ln, "NH-Mono") if ln else "" for ln in lines) or " "
        flow = XPreformatted(markup, style)
        flow.nlines = len(lines)
        return flow

    def callout_flow(self, b: QuoteBlk):
        S = self.styles
        kind = "plain"
        first = b.blocks[0] if b.blocks else None
        label_md = body_md = None
        if isinstance(first, ParaBlk):
            m = re.match(r"^\*\*([^*]+?)\*\*\s*(.*)$", first.text.strip(), re.S)
            if m:
                label_md, body_md = m.group(1), m.group(2)
                low = label_md.strip().lower()
                for k, words in CALLOUT_WORDS.items():
                    if low.startswith(words):
                        kind = k
        bg, bar, label_color = CALLOUTS[kind]
        flows: list = []
        style = S["callout"]
        for idx, blk in enumerate(b.blocks):
            if idx == 0 and label_md is not None:
                label_markup = self.md(label_md, style)
                rest = self.md(body_md, style) if body_md else ""
                flows.append(self.para(f'<font name="NH-Head" color="{label_color}">{label_markup}</font> {rest}', style))
            elif isinstance(blk, ParaBlk):
                flows.append(self.md_para(blk.text, style))
            else:
                for it in self.build_items([blk], in_list=True):
                    flows.extend(it.flows)
        if not flows:
            return Spacer(1, 1)
        box = CalloutBox(flows, bg, bar, 7.5 * (0.45 + 0.55 * self.sq))
        box.spaceBefore = 3 * self.sq
        box.spaceAfter = 10 * self.sq
        return box

    def figure_item(self, b: FigureBlk) -> Item:
        path = self.asset_path(b.src)
        S = self.styles
        if path.suffix.lower() == ".svg":
            drawing = self.card(path) if path.is_file() else None
            if drawing is None:
                self.warnings.append(f"{self.cur.stem}.md: image '{b.src}' not found")
                return Item("none", [])
            card = SvgCard(drawing, CARD_SCALE_WIDE, "CENTER")
            card.spaceBefore = 4 * self.sq
            card.spaceAfter = 10 * self.sq
            return Item("card", [card])
        reader = self.raster(path) if path.is_file() else None
        if reader is None:
            self.warnings.append(f"{self.cur.stem}.md: image '{b.src}' not found")
            return Item("none", [])
        fig = RasterFigure(reader, FRAME_W - self.indent)
        fig.spaceBefore = 4 * self.sq
        flows: list = [fig]
        if b.caption:
            flows.append(self.md_para(b.caption, S["cap"]))
        else:
            fig.spaceAfter = 11 * self.sq
        return Item("figure", flows)

    def card_item(self, b: FigureBlk, paras: list[ParaBlk]) -> Item | None:
        """A node card with its description beside it; None when the card cannot be used that way."""
        S = self.styles
        path = self.asset_path(b.src)
        drawing = self.card(path) if path.is_file() else None
        if drawing is None or not paras:
            return None  # the caller falls back to figure_item(), which reports a missing file
        card = SvgCard(drawing, CARD_SCALE_SIDE)
        gap = 18.0
        text_w = FRAME_W - card.width - gap
        texts = [self.md_para(p.text.strip(), S["body"]) for p in paras]
        height = sum(t.wrap(text_w, 10000)[1] + S["body"].spaceAfter for t in texts)
        if height > max(card.height * 1.5, 150) or height > 420:
            return None
        table = Table([[card, texts]], colWidths=[card.width + gap, text_w], hAlign="LEFT",
                      spaceBefore=4 * self.sq, spaceAfter=10 * self.sq)
        table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "NH-Body"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (0, 0), gap),
            ("RIGHTPADDING", (1, 0), (1, 0), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        return Item("card", [table])

    # ---- keep-with-next handling ------------------------------------------------------------------------------------

    def join(self, items: list[Item]) -> list:
        """Flatten items, keeping headings and labels together with what follows them (no orphan headings)."""
        out: list = []
        i = 0
        while i < len(items):
            it = items[i]
            if it.kind not in ("heading", "label"):
                if it.kind == "code_small" or (it.kind == "figure" and len(it.flows) > 1):
                    out.append(KeepTogether(it.flows))
                else:
                    out.extend(it.flows)
                i += 1
                continue
            group: list = []
            while i < len(items) and items[i].kind in ("heading", "label"):
                group.extend(items[i].flows)
                i += 1
            nxt = items[i] if i < len(items) else None
            if nxt is None:
                out.extend(group)
                continue
            if nxt.kind in ATOMIC_KINDS:
                out.append(KeepTogether(group + nxt.flows))
                i += 1
                continue
            need = sum(f.wrap(FRAME_W, 10000)[1] + f.getSpaceBefore() + f.getSpaceAfter() for f in group)
            need += {"para": 46, "list": 64, "table": 92, "code": 56}.get(nxt.kind, 60)
            out.append(CondPageBreak(need))
            out.extend(group)
        return out


# --------------------------------------------------------------------------------------------------------------------
# Document template (cover, header/footer, outline)
# --------------------------------------------------------------------------------------------------------------------


def spaced_text(canv, x: float, y: float, text: str, font: str, size: float, color, space: float) -> None:
    """Draw letter-spaced text; the character spacing is reset so it cannot leak into later drawing."""
    obj = canv.beginText(x, y)
    obj.setFont(font, size)
    obj.setFillColor(color)
    obj.setCharSpace(space)
    obj.textOut(text)
    obj.setCharSpace(0)
    canv.drawText(obj)


def fit_text(text: str, font: str, size: float, width: float) -> str:
    if stringWidth(text, font, size) <= width:
        return text
    while text and stringWidth(text + "\u2026", font, size) > width:
        text = text[:-1]
    return text.rstrip() + "\u2026"


def prepare_banner(path: Path) -> ImageReader:
    """The banner with its bottom edge faded into the cover background."""
    with PILImage.open(path) as src:
        im = src.convert("RGB")
    w, h = im.size
    fade = int(h * 0.3)
    bg = (0x14, 0x17, 0x1C)
    mask = PILImage.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for y in range(fade):
        t = y / (fade - 1)
        alpha = int(255 * t * t * (3 - 2 * t))
        draw.line([(0, h - fade + y), (w, h - fade + y)], fill=alpha)
    im = PILImage.composite(PILImage.new("RGB", (w, h), bg), im, mask)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90, optimize=True)
    buf.seek(0)
    return ImageReader(buf)


class GuideDoc(BaseDocTemplate):
    def __init__(self, filename: str, cfg: dict, meta: dict, today: dt.date, banner: ImageReader | None,
                 prev_total: int = 0) -> None:
        title = cfg["title"]
        super().__init__(
            filename, pagesize=A4, leftMargin=MARGIN_X, rightMargin=MARGIN_X, topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM, title=title, author="jetthuang.com", subject=cfg["tagline"],
            creator="NH-Nodes docs/_tools/build_pdf.py", keywords=["ComfyUI", "NH-Nodes", "custom nodes"],
            lang=cfg["tag"], displayDocTitle=True, initialFontName="NH-Body", initialFontSize=9.6,
            initialLeading=14.4, pageCompression=1)
        self.cfg = cfg
        self.meta = meta
        self.today = today
        self.banner = banner
        self.prev_total = prev_total
        self.total_pages = prev_total
        self.max_page = 0
        self.outline_level = 0
        self.header_chapter = ""
        self.current_stem = ""
        self.tails: dict[str, list] = {}
        self.opener_on_page = False
        cover = PageTemplate("cover", [Frame(0, 0, 10, 10, id="cover_frame")], onPage=self.draw_cover, autoNextPageTemplate=None)
        main = PageTemplate(
            "main", [Frame(MARGIN_X, MARGIN_BOTTOM, FRAME_W, FRAME_H, leftPadding=0, rightPadding=0, topPadding=0,
                           bottomPadding=0, id="main_frame")],
            onPage=self.begin_page, onPageEnd=self.draw_chrome)
        self.addPageTemplates([cover, main])

    def beforeDocument(self):
        self.total_pages = self.prev_total
        self.max_page = 0
        self.outline_level = 0
        self.header_chapter = ""
        self.current_stem = ""
        self.tails = {}

    def afterPage(self):
        self.max_page = max(self.max_page, self.page)
        self.prev_total = self.max_page

    def begin_page(self, canv, doc):
        self.opener_on_page = False

    # ---- cover ------------------------------------------------------------------------------------------------------

    def draw_cover(self, canv, doc):
        cfg, meta = self.cfg, self.meta
        canv.saveState()
        canv.setFillColor(COVER_BG)
        canv.rect(-3, -3, PAGE_W + 6, PAGE_H + 6, stroke=0, fill=1)
        banner_h = PAGE_W * 560 / 1600
        if self.banner is not None:
            canv.drawImage(self.banner, -2, PAGE_H - banner_h, PAGE_W + 4, banner_h + 2)
        left = 56.0
        y = PAGE_H - banner_h - 52
        # kicker
        canv.setFillColor(TEAL)
        canv.rect(left, y - 1.5, 3, 12.5, stroke=0, fill=1)
        spaced_text(canv, left + 11, y, cfg["kicker"], "NH-Head", 9.6, TEAL, 2.4)
        # title
        size = 35.0
        title = cfg["title"]
        maxw = PAGE_W - 2 * left
        while stringWidth(title, "NH-Head", size) > maxw and size > 20:
            size -= 0.5
        y -= 20 + size
        canv.setFillColor(colors.white)
        canv.setFont("NH-Head", size)
        canv.drawString(left, y, title)
        y -= 16
        canv.setFillColor(TEAL)
        canv.roundRect(left, y, 58, 3.2, 1.6, stroke=0, fill=1)
        # tagline
        style = ParagraphStyle("tag", fontName="NH-Body", fontSize=12.6, leading=19, textColor=colors.HexColor("#aab3c0"))
        para = Paragraph(esc(cfg["tagline"]), style)
        pw, ph = para.wrap(PAGE_W - 2 * left - 70, 400)
        y -= 20 + ph
        para.drawOn(canv, left, y)
        # nodes per category
        cats = meta["categories"]
        if cats:
            y -= 58
            glance = cfg["glance"].format(n=meta["nodes"] or sum(n for _, n in cats), c=len(cats))
            spaced_text(canv, left, y, glance, "NH-Head-Medium", 7.8, colors.HexColor("#7e8897"), 1.5)
            per_row, gap, tile_h = 5, 9.0, 56.0
            tile_w = (PAGE_W - 2 * left - gap * (per_row - 1)) / per_row
            top = y - 16
            for idx, (name, count) in enumerate(cats):
                row, col = divmod(idx, per_row)
                tx = left + col * (tile_w + gap)
                ty = top - tile_h - row * (tile_h + gap)
                canv.setFillColor(colors.HexColor("#1b2028"))
                canv.setStrokeColor(colors.HexColor("#283039"))
                canv.setLineWidth(0.8)
                canv.roundRect(tx, ty, tile_w, tile_h, 7, stroke=1, fill=1)
                canv.setFillColor(colors.white)
                canv.setFont("NH-Head", 19)
                canv.drawString(tx + 12, ty + 29, str(count))
                canv.setFillColor(colors.HexColor("#8d97a6"))
                canv.setFont("NH-Body", 8.8)
                canv.drawString(tx + 12, ty + 12, name)
                canv.setFillColor(TEAL)
                canv.roundRect(tx + tile_w - 17, ty + tile_h - 14, 7, 3, 1.5, stroke=0, fill=1)
        # info columns
        base = 118.0
        cols = [
            (cfg["author"], "jetthuang.com", "https://jetthuang.com"),
            (cfg["version"], meta["version"] or "-", None),
            (cfg["date"], format_date(cfg, self.today), None),
        ]
        col_w = (PAGE_W - 2 * left) / 3
        canv.setStrokeColor(colors.HexColor("#2a303a"))
        canv.setLineWidth(0.8)
        canv.line(left, base + 38, PAGE_W - left, base + 38)
        for idx, (label, value, url) in enumerate(cols):
            cx = left + idx * col_w
            spaced_text(canv, cx, base + 18, label.upper(), "NH-Head-Medium", 7.6, colors.HexColor("#7e8897"), 1.4)
            canv.setFillColor(colors.white)
            canv.setFont("NH-Body", 11.6)
            canv.drawString(cx, base, value)
            if url:
                canv.linkURL(url, (cx, base - 4, cx + stringWidth(value, "NH-Body", 11.6), base + 12), relative=0)
        repo_label = meta["repo"].replace("https://", "")
        canv.setFillColor(colors.HexColor("#7e8897"))
        canv.setFont("NH-Body", 8.6)
        canv.drawString(left, 58, repo_label)
        canv.linkURL(meta["repo"], (left, 54, left + stringWidth(repo_label, "NH-Body", 8.6), 68), relative=0)
        canv.restoreState()
        canv.bookmarkPage("cover")
        canv.addOutlineEntry(cfg["title"], "cover", 0, closed=False)
        canv.showOutline()  # open the bookmark panel when the PDF is opened
        self.outline_level = 0

    # ---- running header and footer ----------------------------------------------------------------------------------

    def draw_chrome(self, canv, doc):
        cfg = self.cfg
        if self.current_stem:
            frame = self.frame
            top = frame._y2 - frame._topPadding
            fill = (top - frame._y) / (top - frame._y1p)
            stat = self.tails.setdefault(self.current_stem, [0, 0.0])
            stat[0] += 1
            stat[1] = fill  # ends up as the fill of the chapter's last page
        canv.saveState()
        if not self.opener_on_page:
            y = PAGE_H - 42
            canv.setFillColor(TEAL)
            canv.rect(MARGIN_X, y - 1.5, 2.6, 9.4, stroke=0, fill=1)
            canv.setFont("NH-Head-Medium", 7.9)
            canv.setFillColor(MUTED)
            canv.drawString(MARGIN_X + 8, y, cfg["title"])
            room = FRAME_W - 16 - stringWidth(cfg["title"], "NH-Head-Medium", 7.9)
            chapter = fit_text(self.header_chapter, "NH-Head-Medium", 7.9, max(room, 60))
            canv.drawRightString(PAGE_W - MARGIN_X, y, chapter)
            canv.setStrokeColor(RULE)
            canv.setLineWidth(0.7)
            canv.line(MARGIN_X, y - 8, PAGE_W - MARGIN_X, y - 8)
        y = 36
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.7)
        canv.line(MARGIN_X, y + 13, PAGE_W - MARGIN_X, y + 13)
        canv.setFont("NH-Body", 7.8)
        canv.setFillColor(MUTED)
        left = f"NH-Nodes v{self.meta['version']}  \u00b7  jetthuang.com" if self.meta["version"] else "jetthuang.com"
        canv.drawString(MARGIN_X, y, left)
        canv.linkURL("https://jetthuang.com", (MARGIN_X, y - 3, MARGIN_X + stringWidth(left, "NH-Body", 7.8), y + 9),
                     relative=0)
        total = self.total_pages or "\u2026"
        label = f"{cfg['page']} {self.page} / {total}"
        canv.setFont("NH-Head-Medium", 8)
        canv.setFillColor(TEXT_SOFT)
        canv.drawRightString(PAGE_W - MARGIN_X, y, label)
        canv.restoreState()


# --------------------------------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------------------------------


class GuideTOC(TableOfContents):
    """Table of contents whose chapter lines are never left alone at the bottom of a page."""

    BASE_STYLE = [
        ("FONTNAME", (0, 0), (-1, -1), "NH-Body"),  # tables otherwise start from the non-embedded Helvetica
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]

    def wrap(self, aw, ah):
        entries = self._lastEntries or []
        spacer = [1 if self.getLevelStyle(level).spaceBefore else 0 for level, *_ in entries]
        total_rows = len(entries) + sum(spacer)
        keep = []
        row = 0
        for (level, *_), gap in zip(entries, spacer):
            row += gap
            if level == 0 and row + 1 < total_rows:
                keep.append(("NOSPLIT", (0, row), (0, row + 1)))
            row += 1
        self.tableStyle = TableStyle(self.BASE_STYLE + keep)
        return super().wrap(aw, ah)


def make_story(builder: StoryBuilder, chapters: list[Chapter]) -> list:
    cfg = builder.cfg
    story: list = [NextPageTemplate("main"), PageBreak()]
    S = builder.style_sets[builder.squeeze.get("toc", 0)]
    toc_title = Heading(esc(cfg["toc"]), S["toc_title"], "toc", 0, cfg["toc"], None, chapter=cfg["toc"], stem="toc")
    toc = GuideTOC(levelStyles=[S["toc0"], S["toc1"]], dotsMinLevel=1, rightColumnWidth=30)
    story += [toc_title, toc]
    for ch in chapters:
        story.extend(builder.build_chapter(ch))
    return story


def build_one(lang: str, out_dir: Path, meta: dict, today: dt.date, kit: FontKit) -> tuple[Path, int, list[str]]:
    cfg = LANGS[lang]
    warnings: list[str] = []
    keys = KeyCounter()
    chapters = load_chapters(lang, keys, warnings)
    links = Links(chapters, meta["repo"], warnings)
    banner_path = ASSETS / "banner.jpg"
    banner = prepare_banner(banner_path) if banner_path.is_file() else None
    target = out_dir / cfg["pdf"]
    tmp = target.with_suffix(".pdf.tmp")
    squeeze: dict[str, int] = {}
    caches: tuple = ({}, {})  # converted SVG cards and loaded images, shared by all layout attempts
    try:
        pages = layout(lang, kit, chapters, links, meta, today, banner, tmp, warnings, squeeze, caches)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    if squeeze:
        log(f"[{lang}] tightened spacing to avoid near-empty last pages: "
            + ", ".join(f"{stem} (level {lvl})" for stem, lvl in sorted(squeeze.items())))
    return target, pages, warnings


def layout(lang, kit, chapters, links, meta, today, banner, tmp, warnings, squeeze, caches) -> int:
    """Lay the guide out and write `tmp`; returns the page count.

    Every attempt is a full build. The footers print the page total, which is only known after a build, and a chapter
    that ends on an almost empty page is retried with tighter vertical spacing (`squeeze` is updated in place).
    """
    cfg = LANGS[lang]
    prev_total = 0
    pages = 0
    for attempt in range(MAX_ATTEMPTS):
        builder = StoryBuilder(lang, kit, chapters, links, meta, warnings, squeeze, caches)
        doc = GuideDoc(str(tmp), cfg, meta, today, banner, prev_total)
        doc.multiBuild(make_story(builder, chapters), maxPasses=8)
        pages = doc.max_page
        flagged = [stem for stem, (count, fill) in doc.tails.items()
                   if count >= 2 and fill < TAIL_FILL and squeeze.get(stem, 0) < len(SQUEEZE) - 1]
        if flagged and attempt < MAX_ATTEMPTS - 2:
            for stem in flagged:
                squeeze[stem] = squeeze.get(stem, 0) + 1  # the last two attempts only settle the footer total
        elif doc.total_pages == pages:
            break
        prev_total = pages
    else:
        warnings.append("page total did not settle; footer totals may be off")
    return pages


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Build the NH-Nodes PDF user guides from docs/en and docs/vi.")
    parser.add_argument("--lang", choices=["en", "vi", "all"], default="all", help="language to build (default: all)")
    parser.add_argument("--out", type=Path, default=REPO / "dist", help="output folder (default: dist/ in the repo root)")
    args = parser.parse_args(argv)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    langs = ["en", "vi"] if args.lang == "all" else [args.lang]
    meta = read_project_meta()
    today = dt.date.today()
    kit = FontKit()
    register_card_fonts()
    status = 0
    for lang in langs:
        try:
            target, pages, warnings = build_one(lang, out_dir, meta, today, kit)
        except Exception as exc:  # keep going so one language failing does not hide the other
            log(f"[{lang}] FAILED: {type(exc).__name__}: {exc}")
            status = 1
            continue
        size_mb = target.stat().st_size / (1024 * 1024)
        log(f"[{lang}] {target}  {pages} pages  {size_mb:.2f} MB")
        for w in sorted(set(warnings)):
            log(f"[{lang}] warning: {w}")
    if kit.missing:
        shown = ", ".join(f"U+{ord(c):04X}" for c in sorted(kit.missing))
        log(f"warning: no font has these characters, they were dropped: {shown}")
    return status


if __name__ == "__main__":
    sys.exit(main())
