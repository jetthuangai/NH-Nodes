"""Vietnamese words -> the English SAM 3 and the parsers understand."""

import importlib
import unicodedata

import pytest

vi = importlib.import_module("NH-Nodes.core.vi_words")


@pytest.mark.parametrize("written, english", [
    ("tóc", "hair"), ("Tóc", "hair"), ("  tóc  ", "hair"), ("váy", "dress"), ("túi", "bag"), ("người", "person"),
    ("cửa sổ", "window"), ("cửa", "door"),  # the longer phrase wins
    ("da mặt", "face skin"), ("tóc vàng", "blonde hair"),  # a phrase of its own beats noun + colour
    ("áo đỏ", "red top"), ("áo sơ mi trắng", "white shirt"), ("tóc dài", "long hair"),
    ("áo xanh lá", "green top"), ("cây to", "large tree"), ("quần jean xanh dương", "blue jeans"),
    ("nền", "background"), ("phông nền", "background"), ("tất cả", "all"),
    ("balo", "backpack"),  # Vietnamese written without accents, listed on purpose
    ("người đàn ông", "man"), ("người phụ nữ", "woman"), ("các áo", "top"), ("những chiếc xe", "vehicle"),
    ("áo màu đỏ", "red top"), ("nền trắng", "background"), ("phông nền xám", "background"),
])
def test_translate_known_vietnamese(written, english):
    assert vi.translate(written) == (english, "translated")


def test_decomposed_accents_from_some_keyboards_are_read_the_same():
    decomposed = unicodedata.normalize("NFD", "tóc")
    assert decomposed != "tóc" and vi.translate(decomposed) == ("hair", "translated")


@pytest.mark.parametrize("written", ["person", "red car", "t-shirt", "*shirt", "Window", "to", "be"])
def test_english_passes_through_unchanged(written):
    assert vi.translate(written) == (written, "english")


@pytest.mark.parametrize("written", ["áo xanh", "con rồng", "tóc rất đẹp"])
def test_unknown_vietnamese_is_reported_not_guessed(written):
    # "xanh" alone is blue or green: the node asks instead of guessing
    assert vi.translate(written) == (written, "unknown")
    assert vi.is_vietnamese(written)


def test_every_listed_word_is_lower_case_composed_and_translates_to_ascii():
    for table in (vi.NOUNS, vi.ADJECTIVES):
        for word, english in table.items():
            assert word == vi.normalize(word), word
            assert english.isascii() and english == english.strip().lower(), english
