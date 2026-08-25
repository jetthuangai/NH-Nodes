"""Resolution data for NH Smart Resolution Picker.

Data is kept separate from node logic so new model families can be added
without changing execution code.
"""


MODEL_SPECS = {
    "flux2_klein_4b": {
        "name": "FLUX.2 Klein 4B",
        "multiple_of": 16,
        "min_res": (64, 64),
        "max_res": (2048, 2048),
        "latent_channels": 16,
        "note": "",
    },
    "flux2_klein_9b": {
        "name": "FLUX.2 Klein 9B",
        "multiple_of": 16,
        "min_res": (64, 64),
        "max_res": (2048, 2048),
        "latent_channels": 16,
        "note": "",
    },
    "flux2_dev": {
        "name": "FLUX.2 Dev",
        "multiple_of": 16,
        "min_res": (400, 400),
        "max_res": (2048, 2048),
        "latent_channels": 16,
        "note": "",
    },
    "qwen_image_edit": {
        "name": "Qwen-Image",
        "multiple_of": 32,
        "min_res": (512, 512),
        "max_res": (2560, 2560),
        "latent_channels": 16,
        "note": "Qwen tự match aspect input image",
    },
    "z_image": {
        "name": "Z-Image",
        "multiple_of": 16,
        "min_res": (512, 512),
        "max_res": (2048, 2048),
        "latent_channels": 16,
        "note": "",
    },
}


ASPECT_SORT_ORDER = {
    "1:1": 0,
    "4:5": 1,
    "3:4": 2,
    "2:3": 3,
    "7:9": 4,
    "9:16": 5,
    "9:21": 6,
    "5:4": 7,
    "4:3": 8,
    "9:7": 9,
    "3:2": 10,
    "16:9": 11,
    "21:9": 12,
}


MODEL_SORT_ORDER = {
    "flux2_klein_4b": 0,
    "flux2_klein_9b": 1,
    "flux2_dev": 2,
    "qwen_image_edit": 3,
    "z_image": 4,
}


DEFAULT_MODEL_LABEL = "FLUX.2 Klein 9B"
DEFAULT_PRESET = "2 MP (sweet spot) ⭐ | 1:1 | 1408×1408"


def _entry(model_id, tier, aspect, width, height, reliability="verified", sweet=False, warning=""):
    model_name = MODEL_SPECS[model_id]["name"]
    tier_label = tier
    if sweet and "⭐" not in tier_label:
        tier_label = f"{tier_label} ⭐"
    return {
        "model_id": model_id,
        "model": model_name,
        "tier": tier_label,
        "aspect": aspect,
        "width": int(width),
        "height": int(height),
        "reliability": reliability,
        "warning": warning,
        "sweet": bool(sweet),
        "label": f"{tier_label} | {aspect} | {width}×{height}",
    }


def _flux_entries(model_id, tier_plan):
    buckets = {
        "1 MP": [
            ("1:1", 1024, 1024),
            ("4:3", 1152, 896),
            ("3:4", 896, 1152),
            ("3:2", 1216, 832),
            ("2:3", 832, 1216),
            ("16:9", 1344, 768),
            ("9:16", 768, 1344),
            ("21:9", 1536, 640),
            ("9:21", 640, 1536),
        ],
        "2 MP": [
            ("1:1", 1408, 1408),
            ("4:3", 1632, 1216),
            ("3:4", 1216, 1632),
            ("3:2", 1728, 1152),
            ("2:3", 1152, 1728),
            ("16:9", 1888, 1056),
            ("9:16", 1056, 1888),
        ],
        "3 MP": [
            ("1:1", 1728, 1728),
            ("4:3", 2000, 1504),
            ("3:4", 1504, 2000),
            ("3:2", 2128, 1424),
            ("2:3", 1424, 2128),
            ("16:9", 2320, 1296),
            ("9:16", 1296, 2320),
        ],
        "4 MP": [
            ("1:1", 2016, 2016),
            ("4:3", 2304, 1728),
            ("3:4", 1728, 2304),
            ("16:9", 2688, 1504),
            ("9:16", 1504, 2688),
        ],
    }

    entries = []
    for bucket_name, tier_label, reliability, sweet, warning in tier_plan:
        for aspect, width, height in buckets[bucket_name]:
            entries.append(_entry(
                model_id,
                tier_label,
                aspect,
                width,
                height,
                reliability,
                sweet and aspect == "1:1",
                warning,
            ))
    return entries


def _qwen_entries():
    entries = []
    native = [
        ("1:1", 1328, 1328),
        ("16:9", 1664, 928),
        ("9:16", 928, 1664),
        ("4:3", 1472, 1140),
        ("3:4", 1140, 1472),
        ("3:2", 1584, 1056),
        ("2:3", 1056, 1584),
    ]
    for aspect, width, height in native:
        entries.append(_entry("qwen_image_edit", "Native (1.5-1.8 MP)", aspect, width, height, "verified", aspect == "1:1"))

    community_tiers = [
        ("1 MP (preview)", [
            ("1:1", 1024, 1024),
            ("4:3", 1152, 896),
            ("3:4", 896, 1152),
            ("3:2", 1248, 832),
            ("2:3", 832, 1248),
            ("16:9", 1344, 768),
            ("9:16", 768, 1344),
        ]),
        ("2 MP", [
            ("1:1", 1408, 1408),
            ("4:3", 1632, 1216),
            ("3:4", 1216, 1632),
            ("3:2", 1728, 1152),
            ("2:3", 1152, 1728),
            ("16:9", 1888, 1056),
            ("9:16", 1056, 1888),
        ]),
        ("3 MP", [
            ("1:1", 1728, 1728),
            ("4:3", 2016, 1504),
            ("3:4", 1504, 2016),
            ("16:9", 2304, 1280),
            ("9:16", 1280, 2304),
        ]),
        ("4 MP", [
            ("1:1", 2048, 2048),
            ("4:3", 2304, 1728),
            ("3:4", 1728, 2304),
            ("16:9", 2688, 1504),
            ("9:16", 1504, 2688),
        ]),
        ("HD community", [
            ("1:1", 1920, 1920),
            ("16:9", 2560, 1440),
            ("9:16", 1440, 2560),
            ("4:3", 2560, 1920),
            ("3:4", 1920, 2560),
            ("1:1", 2560, 2560),
        ]),
    ]

    for tier, rows in community_tiers:
        reliability = "community"
        display_tier = "HD" if tier == "HD community" else tier
        warning = "Community-tested bucket; not an official Qwen preset."
        if tier == "HD community":
            warning = "HD community finding; not an official Qwen preset."
        for aspect, width, height in rows:
            entries.append(_entry("qwen_image_edit", display_tier, aspect, width, height, reliability, False, warning))
    return entries


def _z_image_entries():
    tiers = [
        ("Tier 1024", "verified", True, [
            ("1:1", 1024, 1024),
            ("9:7", 1152, 896),
            ("7:9", 896, 1152),
            ("4:3", 1152, 864),
            ("3:4", 864, 1152),
            ("3:2", 1248, 832),
            ("2:3", 832, 1248),
            ("16:9", 1280, 720),
            ("9:16", 720, 1280),
            ("21:9", 1344, 576),
            ("9:21", 576, 1344),
        ]),
        ("Tier 1280", "verified", True, [
            ("1:1", 1280, 1280),
            ("9:7", 1440, 1120),
            ("7:9", 1120, 1440),
            ("4:3", 1472, 1104),
            ("3:4", 1104, 1472),
            ("3:2", 1536, 1024),
            ("2:3", 1024, 1536),
            ("16:9", 1600, 896),
            ("9:16", 896, 1600),
            ("21:9", 1680, 720),
            ("9:21", 720, 1680),
        ]),
        ("Tier 1536", "community", False, [
            ("1:1", 1536, 1536),
            ("9:7", 1728, 1344),
            ("7:9", 1344, 1728),
            ("4:3", 1728, 1296),
            ("3:4", 1296, 1728),
            ("3:2", 1872, 1248),
            ("2:3", 1248, 1872),
            ("16:9", 2048, 1152),
            ("9:16", 1152, 2048),
        ]),
        ("Tier 2K", "extrapolation", False, [
            ("1:1", 2048, 2048),
            ("16:9", 2048, 1152),
            ("9:16", 1152, 2048),
            ("4:3", 2048, 1536),
            ("3:4", 1536, 2048),
        ]),
    ]

    entries = []
    for tier, reliability, sweet_available, rows in tiers:
        warning = ""
        if reliability == "community":
            warning = "Community-tested tier; outside official Tongyi 1024/1280 confirmation."
        elif reliability == "extrapolation":
            warning = "Extrapolation tier in max spec range; outside Z-Image sweet-spot tolerance."
        for aspect, width, height in rows:
            entries.append(_entry("z_image", tier, aspect, width, height, reliability, sweet_available and aspect == "1:1", warning))
    return entries


def _reliability_rank(entry):
    if entry["sweet"]:
        return 0
    reliability = entry["reliability"]
    if reliability == "verified":
        return 1
    if reliability == "community":
        return 2
    if reliability == "extrapolation":
        return 3
    return 4


def _sort_key(entry):
    return (
        MODEL_SORT_ORDER[entry["model_id"]],
        _reliability_rank(entry),
        entry["tier"].replace("⭐", "").strip(),
        ASPECT_SORT_ORDER.get(entry["aspect"], 99),
        entry["width"] * entry["height"],
    )


RESOLUTION_PRESETS = []
RESOLUTION_PRESETS.extend(_flux_entries("flux2_klein_4b", [
    ("1 MP", "1 MP (sweet spot)", "verified", True, ""),
    ("2 MP", "2 MP", "verified", False, ""),
    ("3 MP", "3 MP", "community", False, "Community-tested tier for Klein 4B; not official spec."),
    ("4 MP", "4 MP", "risk", False, "Risk tier for Klein 4B, especially Distilled variants."),
]))
RESOLUTION_PRESETS.extend(_flux_entries("flux2_klein_9b", [
    ("2 MP", "2 MP (sweet spot)", "verified", True, ""),
    ("1 MP", "1 MP", "verified", False, ""),
    ("3 MP", "3 MP", "verified", False, ""),
    ("4 MP", "4 MP", "risk", False, "4 MP is intended for Base 50 steps; Distilled variants have higher risk."),
]))
RESOLUTION_PRESETS.extend(_flux_entries("flux2_dev", [
    ("2 MP", "2 MP (sweet spot)", "verified", True, ""),
    ("1 MP", "1 MP", "verified", False, ""),
    ("3 MP", "3 MP", "verified", False, ""),
    ("4 MP", "4 MP", "verified", False, ""),
]))
RESOLUTION_PRESETS.extend(_qwen_entries())
RESOLUTION_PRESETS.extend(_z_image_entries())
RESOLUTION_PRESETS.sort(key=_sort_key)

MODEL_LABELS = [MODEL_SPECS[model_id]["name"] for model_id, _rank in sorted(MODEL_SORT_ORDER.items(), key=lambda item: item[1])]
MODEL_ID_BY_LABEL = {MODEL_SPECS[model_id]["name"]: model_id for model_id in MODEL_SORT_ORDER}

PRESET_LABELS_BY_MODEL = {}
PRESET_LOOKUP_BY_MODEL = {}
for entry in RESOLUTION_PRESETS:
    model_label = entry["model"]
    PRESET_LABELS_BY_MODEL.setdefault(model_label, []).append(entry["label"])
    PRESET_LOOKUP_BY_MODEL.setdefault(model_label, {})[entry["label"]] = entry

PRESET_LABELS = PRESET_LABELS_BY_MODEL[DEFAULT_MODEL_LABEL]
PRESET_LOOKUP = PRESET_LOOKUP_BY_MODEL[DEFAULT_MODEL_LABEL]
