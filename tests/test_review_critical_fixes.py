"""Regression tests for the four critical review findings (2026-10-18)."""

import importlib
import os
import sys

import pytest
import torch

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PACK_DIR))
for path in (COMFY_ROOT, os.path.dirname(PACK_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import comfy.latent_formats  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.sd  # noqa: E402

resolution_data = importlib.import_module("NH-Nodes.core.resolution_data")
picker = importlib.import_module("NH-Nodes.nodes.resolution.smart_resolution_picker")
loaders = importlib.import_module("NH-Nodes.nodes.loaders.loader_index_nodes")
batch = importlib.import_module("NH-Nodes.nodes.batch.batch_nodes")
mask_resize = importlib.import_module("NH-Nodes.nodes.image.mask_resize_image")


class _FakeModel:
    """Just enough of a ModelPatcher for comfy.sample.fix_empty_latent_channels."""

    def __init__(self, latent_format):
        self.latent_format = latent_format

    def get_model_object(self, name):
        assert name == "latent_format"
        return self.latent_format


# --- C1: Smart Resolution Picker latent must decode to the size it reports ---

FLUX2_LABELS = [spec["name"] for model_id, spec in resolution_data.MODEL_SPECS.items() if model_id.startswith("flux2")]


@pytest.mark.parametrize("model_label", FLUX2_LABELS)
@pytest.mark.parametrize("size", [(1408, 1408), (1024, 768), (768, 1344)])
def test_flux2_latent_decodes_to_reported_size(model_label, size):
    width, height = size
    out_w, out_h, latent, _info, _ar = picker.NH_SmartResolutionPicker().pick(
        model_family=model_label, use_custom=True, custom_width=width, custom_height=height,
    )
    fmt = comfy.latent_formats.Flux2()
    # Same call KSampler makes before sampling.
    samples = comfy.sample.fix_empty_latent_channels(
        _FakeModel(fmt), latent["samples"], latent.get("downscale_ratio_spacial"), latent.get("downscale_ratio_temporal"),
    )
    assert samples.shape[1] == fmt.latent_channels
    assert samples.shape[-1] * fmt.spacial_downscale_ratio == out_w
    assert samples.shape[-2] * fmt.spacial_downscale_ratio == out_h


def test_default_preset_latent_matches_flux2_format():
    width, height, latent, _info, _ar = picker.NH_SmartResolutionPicker().pick()
    assert latent["samples"].shape == (1, 128, height // 16, width // 16)


@pytest.mark.parametrize("model_id", ["qwen_image_edit", "z_image"])
def test_other_families_keep_16ch_div8_latent(model_id):
    label = resolution_data.MODEL_SPECS[model_id]["name"]
    width, height, latent, _info, _ar = picker.NH_SmartResolutionPicker().pick(model_family=label)
    assert latent["samples"].shape == (1, 16, height // 8, width // 8)
    assert latent["downscale_ratio_spacial"] == 8


# --- C2: Clip Index must resolve every listed type through CLIPType ---

def test_clip_index_types_all_resolve_to_clip_type():
    choices = loaders.NH_ClipIndex.INPUT_TYPES()["required"]["type"][0]
    assert choices[0] == "stable_diffusion"
    # Values saved by older workflows must stay valid.
    for legacy in ("stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "hunyuan_video"):
        assert legacy in choices
    for name in choices:
        assert isinstance(getattr(comfy.sd.CLIPType, name.upper()), comfy.sd.CLIPType)


def test_clip_index_passes_clip_type_to_loader(monkeypatch):
    captured = {}
    monkeypatch.setattr(loaders.folder_paths, "get_full_path_or_raise", lambda folder, name: f"/models/{name}")
    monkeypatch.setattr(comfy.sd, "load_clip", lambda **kwargs: captured.update(kwargs) or "CLIP")
    clip, name, index = loaders.NH_ClipIndex().load_clip_index(
        select_mode="index", type="flux2", index=1, clip_count=1, clip_1="te.safetensors",
    )
    assert (clip, name, index) == ("CLIP", "te.safetensors", 1)
    assert captured["clip_type"] is comfy.sd.CLIPType.FLUX2


# --- C3: Counter must re-execute every queue and keep state per graph node ---

def test_counter_is_never_cached():
    changed = batch.NH_Counter.IS_CHANGED(start=0, step=1, max_value=9, reset=False)
    assert changed != changed  # NaN never equals itself -> ComfyUI always re-runs


def test_counter_advances_per_unique_id():
    batch.NH_Counter._states.clear()
    # ComfyUI may hand each run a fresh instance; state must follow the graph node id.
    first = [batch.NH_Counter().count(0, 2, 9, False, unique_id="7")[0] for _ in range(3)]
    other = batch.NH_Counter().count(0, 1, 9, False, unique_id="8")[0]
    assert first == [0, 2, 4]
    assert other == 0


# --- C4: Mask-aware resize crop mode must keep the whole mask ---

def _bbox(mask):
    rows, cols = torch.nonzero(mask > 0.5, as_tuple=True)
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


@pytest.mark.parametrize("mode", ["crop", "pad"])
@pytest.mark.parametrize(
    "src,box,target",
    [
        ((1000, 800), (300, 400, 100, 200), (512, 512)),  # tall mask, square target (review repro)
        ((1000, 800), (100, 100, 400, 120), (512, 512)),  # wide mask
        ((640, 640), (0, 0, 640, 640), (512, 768)),      # mask = whole image, aspect mismatch
        ((900, 600), (850, 10, 50, 300), (256, 256)),    # mask touching the right edge
    ],
)
def test_mask_resize_keeps_entire_mask(mode, src, box, target):
    sw, sh = src
    x, y, bw, bh = box
    tw, th = target
    image = torch.rand((1, sh, sw, 3))
    mask = torch.zeros((1, sh, sw))
    mask[:, y:y + bh, x:x + bw] = 1.0

    out_img, out_mask = mask_resize.NH_MaskResizeImage().resize_by_mask(image, mask, tw, th, mode, "#000000")

    assert out_img.shape == (1, th, tw, 3)
    assert out_mask.shape == (1, th, tw)
    scale = min(tw / bw, th / bh)
    top, bottom, left, right = _bbox(out_mask[0])
    out_w, out_h = right - left + 1, bottom - top + 1
    # Uniformly scaled, untruncated mask: each side is the source side * scale (±2 px rounding).
    assert abs(out_w - bw * scale) <= 2, f"mask width {out_w}, expected ~{bw * scale:.1f}"
    assert abs(out_h - bh * scale) <= 2, f"mask height {out_h}, expected ~{bh * scale:.1f}"
    kept = float(out_mask.sum())
    expected = bw * bh * scale * scale
    assert kept >= expected * 0.98, f"mask lost: kept {kept:.0f} of ~{expected:.0f} px"


def test_mask_resize_crop_fallback_uses_pad_color():
    # Whole-image mask with a mismatched aspect: crop cannot fill the target, so it pads.
    image = torch.zeros((1, 64, 64, 3))
    mask = torch.ones((1, 64, 64))
    out_img, _ = mask_resize.NH_MaskResizeImage().resize_by_mask(image, mask, 128, 256, "crop", "#FF0000")
    assert torch.allclose(out_img[0, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
