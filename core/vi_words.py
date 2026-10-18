"""Vietnamese words for the things NH vision nodes look for, in the English SAM 3 and the parsers understand.

SAM 3 reads English only: "tóc" finds nothing where "hair" finds the hair. ``translate`` turns the Vietnamese words
people type most for photos (people, body, clothes, accessories, rooms, buildings, nature) into English, including a
thing followed by colours or sizes ("áo sơ mi trắng" -> "white shirt"). English passes through unchanged; Vietnamese
it does not know is reported, so the node can say what to write instead.
"""

import re
import unicodedata

# Things. Longer phrases win over their first word ("cửa sổ" over "cửa"), so both may be listed.
NOUNS = {
    # people and body
    "người": "person", "con người": "person", "người mẫu": "person", "mọi người": "person",
    "phụ nữ": "woman", "người phụ nữ": "woman", "đàn bà": "woman", "người đàn bà": "woman", "cô gái": "girl",
    "con gái": "girl", "bé gái": "girl", "đàn ông": "man", "người đàn ông": "man", "chàng trai": "man",
    "con trai": "boy", "bé trai": "boy",
    "trẻ em": "child", "em bé": "baby", "cô dâu": "bride", "chú rể": "groom",
    "cơ thể": "body", "thân người": "body", "đầu": "head",
    "mặt": "face", "khuôn mặt": "face", "gương mặt": "face",
    "tóc": "hair", "mái tóc": "hair", "tóc vàng": "blonde hair",
    "da": "skin", "làn da": "skin", "da mặt": "face skin", "da người": "skin",
    "mắt": "eye", "đôi mắt": "eye", "lông mày": "eyebrow", "chân mày": "eyebrow", "lông mi": "eyelash",
    "mũi": "nose", "miệng": "mouth", "môi": "lips", "đôi môi": "lips", "răng": "teeth", "lưỡi": "tongue",
    "tai": "ear", "cổ": "neck", "vai": "shoulder", "ngực": "chest", "bụng": "belly",
    "tay": "arm", "cánh tay": "arm", "bàn tay": "hand", "ngón tay": "finger", "móng tay": "fingernail",
    "chân": "leg", "đôi chân": "leg", "bàn chân": "foot", "móng chân": "toenail",
    "râu": "beard", "ria mép": "mustache",
    # clothes and accessories
    "quần áo": "clothes", "trang phục": "outfit", "áo": "top", "áo sơ mi": "shirt", "áo thun": "t-shirt",
    "áo phông": "t-shirt", "áo khoác": "jacket", "áo vest": "suit jacket", "áo len": "sweater",
    "áo dài": "ao dai", "áo choàng": "coat", "áo măng tô": "coat", "áo ngực": "bra",
    "váy": "dress", "đầm": "dress", "váy cưới": "wedding dress", "chân váy": "skirt",
    "quần": "pants", "quần dài": "pants", "quần âu": "trousers", "quần jean": "jeans", "quần bò": "jeans",
    "quần short": "shorts", "quần đùi": "shorts",
    "giày": "shoe", "giày cao gót": "high heels", "dép": "sandal", "tất": "sock", "vớ": "sock",
    "mũ": "hat", "nón": "hat", "khăn": "scarf", "khăn quàng": "scarf", "cà vạt": "tie",
    "thắt lưng": "belt", "dây lưng": "belt", "túi": "bag", "túi xách": "handbag", "ví": "wallet",
    "ba lô": "backpack", "balo": "backpack", "kính": "glasses", "mắt kính": "glasses",
    "kính mát": "sunglasses", "kính râm": "sunglasses", "đồng hồ": "watch", "nhẫn": "ring",
    "vòng cổ": "necklace", "dây chuyền": "necklace", "vòng tay": "bracelet", "lắc tay": "bracelet",
    "bông tai": "earring", "khuyên tai": "earring", "hoa tai": "earring", "trang sức": "jewelry",
    "găng tay": "glove", "ô": "umbrella",
    # nature and outdoors
    "trời": "sky", "bầu trời": "sky", "mây": "cloud", "đám mây": "cloud", "mặt trời": "sun",
    "cây": "tree", "cây cối": "tree", "lá": "leaf", "lá cây": "leaf", "cỏ": "grass", "bãi cỏ": "lawn",
    "hoa": "flower", "bông hoa": "flower", "bụi cây": "bush", "núi": "mountain", "đồi": "hill",
    "biển": "sea", "bãi biển": "beach", "cát": "sand", "nước": "water", "sông": "river", "hồ": "lake",
    "đá": "rock", "đường": "road", "con đường": "road", "vỉa hè": "sidewalk", "cầu": "bridge",
    # buildings and rooms
    "nhà": "house", "ngôi nhà": "house", "tòa nhà": "building", "toà nhà": "building",
    "mái": "roof", "mái nhà": "roof", "tường": "wall", "bức tường": "wall", "cửa": "door",
    "cửa ra vào": "door", "cửa sổ": "window", "ban công": "balcony", "cầu thang": "stairs",
    "cột": "pillar", "hàng rào": "fence", "sàn": "floor", "sàn nhà": "floor", "nền nhà": "floor",
    "trần": "ceiling", "trần nhà": "ceiling", "đèn": "lamp", "đèn trần": "ceiling light",
    "rèm": "curtain", "rèm cửa": "curtain", "thảm": "rug", "tranh": "painting", "bức tranh": "painting",
    "gương": "mirror", "kệ": "shelf", "tủ": "cabinet", "bàn": "table", "ghế": "chair",
    "ghế sofa": "sofa", "giường": "bed", "gối": "pillow", "chăn": "blanket", "cây cảnh": "potted plant",
    "chậu cây": "potted plant", "lọ hoa": "vase", "bình hoa": "vase", "ti vi": "television",
    "tivi": "television", "máy tính": "computer", "điện thoại": "phone", "sách": "book",
    # vehicles, food, objects, animals
    "xe": "vehicle", "ô tô": "car", "xe hơi": "car", "xe máy": "motorcycle", "xe đạp": "bicycle",
    "xe buýt": "bus", "thuyền": "boat", "máy bay": "airplane",
    "cốc": "cup", "tách": "cup", "ly": "glass", "chai": "bottle", "đĩa": "plate", "bát": "bowl",
    "chén": "bowl", "bánh": "cake", "đồ ăn": "food", "thức ăn": "food", "hộp": "box",
    "sản phẩm": "product", "chữ": "text", "biển hiệu": "sign", "bảng hiệu": "sign",
    "chó": "dog", "mèo": "cat", "chim": "bird", "ngựa": "horse", "cá": "fish",
    # mask formula keywords
    "nền": "background", "phông": "background", "phông nền": "background", "hậu cảnh": "background",
    "tất cả": "all", "toàn ảnh": "image", "cả ảnh": "image",
}

# Words that describe a thing and come after it in Vietnamese ("áo đỏ" = "red top"). "xanh" alone is left out on
# purpose: it means blue or green, so the node asks for "xanh dương" / "xanh lá" (or English) instead.
ADJECTIVES = {
    "đỏ": "red", "xanh dương": "blue", "xanh da trời": "blue", "xanh lam": "blue", "xanh lá": "green",
    "xanh lá cây": "green", "xanh lục": "green", "vàng": "yellow", "trắng": "white", "đen": "black",
    "hồng": "pink", "tím": "purple", "cam": "orange", "nâu": "brown", "xám": "gray", "ghi": "gray",
    "bạc": "silver", "be": "beige", "kem": "cream",
    "lớn": "large", "to": "large", "nhỏ": "small", "dài": "long", "ngắn": "short",
}

# Words that only count or point ("các áo" = the tops) and the word for colour ("áo màu đỏ" = a red top).
_LEADING = ("các", "những", "mấy", "cái", "chiếc", "con")
_COLOUR_WORD = "màu"

_VIETNAMESE = re.compile("[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]")


def normalize(text):
    """Lower case, composed accents (what IMEs and pasted text differ in), single spaces."""
    return " ".join(unicodedata.normalize("NFC", str(text)).lower().split())


def is_vietnamese(text):
    """True when ``text`` has a letter only Vietnamese uses (an accent, đ)."""
    return bool(_VIETNAMESE.search(normalize(text)))


def _longest_prefix(words, table):
    """(English, number of words) for the longest leading phrase of ``words`` found in ``table``, or (None, 0)."""
    for size in range(len(words), 0, -1):
        english = table.get(" ".join(words[:size]))
        if english is not None:
            return english, size
    return None, 0


def translate(text):
    """(English words, status) for one thing someone typed; status is "english", "translated" or "unknown".

    A whole phrase is looked up first, then a thing followed by colours or sizes. Text without Vietnamese letters is
    taken as English and returned unchanged, as is Vietnamese that is not in the lists ("unknown").
    """
    phrase = normalize(text)
    if not is_vietnamese(phrase) and phrase not in NOUNS:
        return str(text).strip(), "english"
    words = phrase.split()
    english = _phrase(words)
    while english is None and len(words) > 1 and words[0] in _LEADING:  # "những chiếc xe": drop the counting words
        words = words[1:]
        english = _phrase(words)
    return (english, "translated") if english is not None else (str(text).strip(), "unknown")


def _phrase(words):
    """English for a thing followed by colours or sizes, or None."""
    if " ".join(words) in NOUNS:
        return NOUNS[" ".join(words)]
    noun, used = _longest_prefix(words, NOUNS)
    if noun is None:
        return None
    adjectives, rest = [], words[used:]
    while rest:
        if rest[0] == _COLOUR_WORD:
            rest = rest[1:]
            continue
        english, size = _longest_prefix(rest, ADJECTIVES)
        if english is None:
            return None
        adjectives.append(english)
        rest = rest[size:]
    if noun == "background":
        return "background"  # "nền trắng": the keyword, which needs no colour
    return " ".join([*adjectives, noun])
