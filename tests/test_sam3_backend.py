"""SAM 3 backend: geometry helpers, the two-pass pipeline with a fake sam3, package/checkpoint checks.

Everything but the last test runs without sam3, a checkpoint or a GPU. The last one runs the real model and is
skipped unless all three are present.
"""

import contextlib
import importlib
import os
import sys
import types

import numpy as np
import pytest
import torch

backend = importlib.import_module("NH-Nodes.core.sam3_backend")
sam3_nodes = importlib.import_module("NH-Nodes.nodes.vision.sam3_nodes")
segs = importlib.import_module("NH-Nodes.core.segs")


# --- geometry ----------------------------------------------------------------
def test_sample_region_maps_pixel_centres_onto_the_model_view():
    size, height, width = 16, 30, 60
    u = torch.arange(size, dtype=torch.float32).repeat(size, 1)  # value = model column index
    out = backend.sample_region(u, (height, width), (0, 0, width, height), (height, width))
    expected = ((torch.arange(width) + 0.5) * size / width - 0.5).clamp(0, size - 1)
    assert torch.allclose(out[height // 2], expected, atol=1e-4)
    # a sub-region samples the same values as the full grid
    sub = backend.sample_region(u, (height, width), (10, 5, 40, 20), (15, 30))
    assert torch.allclose(sub, out[5:20, 10:40], atol=1e-4)


def test_sample_region_chunks_give_the_same_result(monkeypatch):
    logits = torch.randn(32, 32)
    whole = backend.sample_region(logits, (90, 70), (3, 4, 66, 88), (84, 63))
    monkeypatch.setattr(backend, "_SAMPLE_CHUNK", 100)
    assert torch.equal(backend.sample_region(logits, (90, 70), (3, 4, 66, 88), (84, 63)), whole)


def test_model_box_to_image_adds_a_margin_inside_the_image():
    assert backend.model_box_to_image((16, 16, 48, 32), 64, (300, 600)) == (140, 70, 460, 155)
    assert backend.model_box_to_image((0, 0, 64, 64), 64, (300, 600)) == (0, 0, 600, 300)


@pytest.mark.parametrize("box", [(100, 100, 120, 400), (0, 0, 30, 30), (1900, 950, 2000, 1000), (10, 500, 1990, 520),
                                 (0, 0, 2000, 1000)])
def test_crop_region_contains_the_box_with_margin_and_bounded_aspect(box):
    x1, y1, x2, y2 = box
    cx1, cy1, cx2, cy2 = backend.crop_region(box, (1000, 2000))
    assert 0 <= cx1 <= x1 and 0 <= cy1 <= y1 and x2 <= cx2 <= 2000 and y2 <= cy2 <= 1000
    width, height = cx2 - cx1, cy2 - cy1
    if width < 2000 and height < 1000:  # not clamped by the image
        assert max(width, height) / min(width, height) <= 2.0 + 1e-6
        assert cx1 < x1 or cx1 == 0


def test_to_logits_inverts_probabilities_and_passes_logits_through():
    logits = torch.tensor([-5.0, -0.5, 0.0, 2.0, 6.0])
    assert torch.allclose(backend.to_logits(torch.sigmoid(logits)), logits, atol=1e-3)
    assert torch.equal(backend.to_logits(logits), logits)


def test_drop_stray_parts():
    coarse = np.zeros((20, 20), bool)
    coarse[5:15, 5:15] = True
    refined = np.zeros((20, 20), bool)
    refined[4:14, 4:14] = True
    refined[18:20, 0:2] = True  # speck far from the object
    cleaned = backend.drop_stray_parts(refined, coarse)
    assert cleaned[4:14, 4:14].all() and not cleaned[18:20, 0:2].any()


def _blob(shape, box):
    mask = np.zeros(shape, bool)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = True
    return mask


def test_close_up_agrees_keeps_sharp_thin_objects_and_rejects_other_objects():
    coarse_pole = _blob((400, 300), (130, 20, 170, 380))  # pass 1: a 40 px wide blur around a pole
    sharp_pole = _blob((400, 300), (145, 25, 155, 375))  # close-up: the 10 px pole, IoU only 0.24
    assert backend.close_up_agrees(sharp_pole, coarse_pole, tolerance=16)
    coarse_person = _blob((400, 300), (60, 40, 240, 390))
    assert backend.close_up_agrees(_blob((400, 300), (55, 35, 245, 395)), coarse_person, tolerance=16)
    assert not backend.close_up_agrees(_blob((400, 300), (0, 0, 40, 40)), coarse_person, 16)  # something else
    assert not backend.close_up_agrees(_blob((400, 300), (100, 40, 140, 80)), coarse_person, 16)  # only a part
    assert not backend.close_up_agrees(_blob((400, 300), (0, 0, 300, 400)), coarse_person, 16)  # far too much
    assert not backend.close_up_agrees(np.zeros((400, 300), bool), coarse_person, 16)


def test_close_up_agrees_on_large_masks_at_a_reduced_scale(monkeypatch):
    monkeypatch.setattr(backend, "_AGREE_PIXELS", 1000)
    coarse = _blob((400, 300), (130, 20, 170, 380))
    assert backend.close_up_agrees(_blob((400, 300), (148, 25, 151, 375)), coarse, tolerance=16)  # 3 px pole
    assert backend.close_up_agrees(_blob((400, 300), (149, 25, 150, 375)), coarse, tolerance=16)  # thinner than a step


# --- fake sam3 ---------------------------------------------------------------
SQUARE = (16, 16, 48, 32)  # model-space box (x1, y1, x2, y2) of the fake "square" instance


def _prob_map(box, size=64, inside=0.99):
    prob = torch.full((size, size), 0.01)
    x1, y1, x2, y2 = box
    prob[y1:y2, x1:x2] = inside
    return prob


class FakeProcessor:
    resolution = 64

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.calls = []

    def set_image(self, image, state=None):
        self.calls.append(("image", tuple(image.shape)))
        return {"hw": tuple(image.shape[-2:])}

    def reset_all_prompts(self, state):
        self.calls.append(("reset",))
        state.pop("prompt", None)

    def set_text_prompt(self, prompt, state):
        self.calls.append(("text", prompt))
        state["prompt"] = prompt
        if prompt == "square":
            # best instance, an exact duplicate with a lower score, and a second distinct one
            maps = [_prob_map(SQUARE), _prob_map(SQUARE), _prob_map((50, 40, 60, 60))]
            scores = [0.9, 0.6, 0.7]
        else:
            maps, scores = [], []
        masks = torch.stack(maps)[:, None] if maps else torch.zeros((0, 1, 64, 64))
        return {"masks_logits": masks, "scores": torch.tensor(scores), "boxes": torch.zeros((len(scores), 4))}


class FakeModel(torch.nn.Module):
    """predict_inst answers with a mask filling the box prompt (or the opposite corner when ``stray``)."""

    def __init__(self, stray=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.stray = stray
        self.prompts = []

    def predict_inst(self, state, box=None, mask_input=None, multimask_output=True, return_logits=False):
        height, width = state["hw"]
        self.prompts.append((tuple(np.round(box).astype(int)), tuple(mask_input.shape)))
        logits = np.full((1, height, width), -10.0, np.float32)
        if self.stray:
            logits[0, :3, :3] = 10.0
        else:
            x1, y1, x2, y2 = np.round(box).astype(int)
            logits[0, y1:y2, x1:x2] = 10.0
        return logits, np.array([0.9]), np.zeros((1, 288, 288), np.float32)


def _run(detail, model=None, concepts=("square", "nothing"), **kwargs):
    model = model or FakeModel()
    processors = []

    def make_processor(threshold):
        processors.append(FakeProcessor(threshold))
        return processors[-1]

    images = torch.rand(2, 300, 600, 3)
    frames = backend.segment_images(model, make_processor, images, list(concepts), 0.4, detail, device=torch.device("cpu"),
                                    **kwargs)
    return frames, processors, model


def test_find_instances_drops_duplicates_and_honours_the_cap():
    processor = FakeProcessor()
    state = processor.set_image(torch.zeros(3, 64, 64))
    found = backend.find_instances(processor, state, "square")
    assert [round(score, 2) for score, _, _ in found] == [0.9, 0.7]
    assert found[0][2] == SQUARE
    assert len(backend.find_instances(processor, state, "square", max_instances=1)) == 1
    assert backend.find_instances(processor, state, "nothing") == []
    assert processor.calls.count(("reset",)) == 3  # every prompt starts from a clean state


def test_fast_pass_maps_instances_to_full_resolution():
    frames, processors, model = _run("fast")
    assert len(frames) == 2 and (frames[0].height, frames[0].width) == (300, 600)
    first = frames[0].segments[0]
    assert first.label == "square" and first.score == pytest.approx(0.9)
    # model box (16..48, 16..32) of a 64 px square view -> (150..450, 75..150) in the 600 x 300 image
    assert all(abs(a - b) <= 2 for a, b in zip(first.box, (150, 75, 450, 150)))
    assert len(frames[0].segments) == 2 and model.prompts == []
    assert processors[0].threshold == 0.4 and processors[0].calls[0] == ("image", (3, 64, 64))


def test_balanced_pass_replaces_masks_with_the_close_up_result():
    frames, processors, model = _run("balanced")
    target = backend.model_box_to_image(SQUARE, 64, (300, 600))
    first = frames[0].segments[0]
    assert first.box == target  # the fake close-up fills the box prompt exactly
    assert len(model.prompts) == 4  # 2 instances x 2 images
    assert all(shape == (1, 288, 288) for _, shape in model.prompts)
    crops = [call[1] for call in processors[0].calls if call[0] == "image"][1:]
    assert all(shape[0] == 3 and shape[1] < 300 for shape in crops)  # close-ups are crops of the original


def test_balanced_pass_keeps_the_first_pass_mask_when_the_close_up_strays(capsys):
    fast, _, _ = _run("fast")
    kept, _, _ = _run("balanced", model=FakeModel(stray=True))
    assert kept[0].segments[0].box == fast[0].segments[0].box
    assert "kept the first-pass mask for 4 instance(s)" in capsys.readouterr().out


def test_interrupt_and_progress_hooks():
    seen, steps = [], []
    _run("balanced", interrupt=lambda: seen.append(1), progress=lambda done, total: steps.append((done, total)))
    assert steps == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert len(seen) == 4 + 4  # each concept of each image + each close-up

    def stop():
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        _run("fast", interrupt=stop)


def test_segment_offloads_unless_asked_to_keep(monkeypatch):
    model = FakeModel()
    moves = []
    monkeypatch.setattr(backend, "load_model", lambda checkpoint: model)
    monkeypatch.setattr(backend, "ensure_sam3", lambda: (None, lambda m, device, confidence_threshold: FakeProcessor()))
    monkeypatch.setattr(backend, "sam3_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(backend, "move_model", lambda m, device: moves.append("to"))
    monkeypatch.setattr(backend, "offload", lambda m: moves.append("off"))
    frames = backend.segment(torch.rand(1, 64, 64, 3), ["square"], keep_model_loaded=False)
    assert len(frames) == 1 and moves == ["to", "off"]
    moves.clear()
    backend.segment(torch.rand(1, 64, 64, 3), ["square"], keep_model_loaded=True)
    assert moves == ["to"]


# --- package, device and checkpoint checks -----------------------------------
def _install_fake_sam3(monkeypatch, build=None, processor=None, fail=None, flip_tf32=False):
    monkeypatch.setattr(backend, "_API", None)
    monkeypatch.setattr(backend.importlib.util, "find_spec", lambda name, *a: object() if name == "sam3" else None)

    def import_module(name):
        if fail:
            raise ImportError(fail)
        if name == "sam3.model_builder":
            if flip_tf32:  # like sam3's _setup_tf32() at import
                torch.backends.cuda.matmul.allow_tf32 = not torch.backends.cuda.matmul.allow_tf32
                torch.backends.cudnn.allow_tf32 = not torch.backends.cudnn.allow_tf32
            return types.SimpleNamespace(build_sam3_image_model=build)
        return types.SimpleNamespace(Sam3Processor=processor)

    monkeypatch.setattr(backend.importlib, "import_module", import_module)


def _good_build(bpe_path=None, checkpoint_path=None, load_from_HF=True, enable_inst_interactivity=False,
                device="cuda", eval_mode=True):
    return None


class _GoodProcessor:
    def __init__(self, model, resolution=1008, device="cuda", confidence_threshold=0.5):
        pass

    def set_image(self):
        pass

    def set_text_prompt(self):
        pass

    def reset_all_prompts(self):
        pass


def test_missing_sam3_explains_how_to_install(monkeypatch):
    monkeypatch.setattr(backend, "_API", None)
    monkeypatch.setattr(backend.importlib.util, "find_spec", lambda name, *a: None)
    with pytest.raises(RuntimeError, match="pip install --no-deps git\\+https://github.com/facebookresearch/sam3"):
        backend.ensure_sam3()


def test_broken_sam3_import_is_reported(monkeypatch):
    _install_fake_sam3(monkeypatch, fail="No module named 'pkg_resources'")
    with pytest.raises(RuntimeError, match="failed to load: No module named 'pkg_resources'"):
        backend.ensure_sam3()


def test_sam3_without_the_needed_api_is_reported(monkeypatch):
    _install_fake_sam3(monkeypatch, build=lambda checkpoint_path=None: None, processor=types.SimpleNamespace())
    with pytest.raises(RuntimeError, match="load_from_HF.*eval_mode.*Sam3Processor.set_image"):
        backend.ensure_sam3()

    class OldProcessor(_GoodProcessor):
        def __init__(self, model):
            pass

    _install_fake_sam3(monkeypatch, build=_good_build, processor=OldProcessor)
    with pytest.raises(RuntimeError, match=r"Sam3Processor\(device=\), Sam3Processor\(confidence_threshold=\)"):
        backend.ensure_sam3()


def test_importing_sam3_does_not_change_tf32(monkeypatch):
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    _install_fake_sam3(monkeypatch, build=_good_build, processor=_GoodProcessor, flip_tf32=True)
    assert backend.ensure_sam3() == (_good_build, _GoodProcessor)
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == before


def test_cpu_only_comfyui_gets_a_clear_error(monkeypatch):
    monkeypatch.setattr(backend, "_mm", types.SimpleNamespace(get_torch_device=lambda: torch.device("cpu")))
    with pytest.raises(RuntimeError, match="needs an NVIDIA GPU"):
        backend.sam3_device()


def test_missing_checkpoint_says_where_to_get_it(monkeypatch, tmp_path):
    monkeypatch.setattr(backend, "folder_paths", None)
    monkeypatch.setattr(backend, "checkpoint_folders", lambda: [str(tmp_path)])
    with pytest.raises(RuntimeError, match="huggingface.co/facebook/sam3"):
        backend.resolve_checkpoint("sam3.pt")
    (tmp_path / "sam3.pt").write_bytes(b"x")
    assert backend.resolve_checkpoint("sam3.pt") == os.path.join(str(tmp_path), "sam3.pt")
    with pytest.raises(RuntimeError, match="not a SAM 3 .pt checkpoint"):
        backend.resolve_checkpoint("model.safetensors")
    with pytest.raises(RuntimeError, match="must be a file"):
        backend.resolve_checkpoint("../secret.pt")


def test_checkpoint_list_always_has_an_entry(monkeypatch, tmp_path):
    monkeypatch.setattr(backend, "folder_paths", None)
    monkeypatch.setattr(backend, "checkpoint_folders", lambda: [str(tmp_path)])
    assert backend.checkpoint_choices() == ["sam3.pt"]
    (tmp_path / "sam3.1.pt").write_bytes(b"x")
    (tmp_path / "config.json").write_text("{}")
    assert backend.checkpoint_choices() == ["sam3.1.pt"]


# --- node ----------------------------------------------------------------------
def test_segment_node_outputs(monkeypatch):
    frame = segs.SegsFrame(8, 10, (segs.make_segment("cat", 0.8, torch.ones((2, 3), dtype=torch.bool), (1, 1)),))
    calls = []

    def fake_segment(images, concepts, *args, **kwargs):
        calls.append(concepts)
        return (frame,)

    monkeypatch.setattr(sam3_nodes.sam3, "segment", fake_segment)
    node = sam3_nodes.NH_SAM3Segment()
    result, mask, count = node.segment(torch.rand(1, 8, 10, 3), "cat, dog\ncat", 0.5, "fast", 0, "sam3.pt", False)
    assert calls == [["cat", "dog"]] and result == (frame,) and count == 1
    assert mask.shape == (1, 8, 10) and int(mask.sum()) == 6
    assert sam3_nodes.NH_SAM3Segment.RETURN_NAMES == ("segs", "mask", "count")
    with pytest.raises(ValueError, match="at least one concept"):
        node.segment(torch.rand(1, 8, 10, 3), " ,\n# none", 0.5, "fast", 0, "sam3.pt", False)
    assert sam3_nodes.NH_SAM3Segment.INPUT_TYPES()["required"]["checkpoint"][0]


# --- real model ----------------------------------------------------------------
def _real_sam3_ready():
    if not torch.cuda.is_available() or importlib.util.find_spec("sam3") is None:
        return False
    try:
        backend.resolve_checkpoint(backend.DEFAULT_CHECKPOINT)
    except RuntimeError:
        return False
    return True


@pytest.mark.skipif(not _real_sam3_ready(), reason="needs CUDA, the sam3 package and models/sam3/sam3.pt")
def test_real_sam3_finds_the_astronaut_and_releases_vram():
    from skimage import data

    image = torch.from_numpy(data.astronaut().astype(np.float32) / 255.0)[None]
    frames = backend.segment(image, ["person"], 0.5, "balanced", 0, backend.DEFAULT_CHECKPOINT, False)
    people = frames[0].segments
    assert len(people) >= 1 and people[0].label == "person" and people[0].score > 0.5
    x1, y1, x2, y2 = people[0].box
    assert 0 <= x1 < x2 <= 512 and 0 <= y1 < y2 <= 512 and people[0].area > 0.1 * 512 * 512
    assert next(backend.load_model(backend.DEFAULT_CHECKPOINT).parameters()).device.type == "cpu"
    assert "sam3" in sys.modules


def test_unreadable_tf32_switches_are_left_alone(monkeypatch):
    class Refusing:
        @property
        def allow_tf32(self):
            raise RuntimeError("mixing legacy and new precision APIs")

    monkeypatch.setattr(torch.backends.cuda, "matmul", Refusing())
    assert backend._tf32_flags() is None
    backend._restore_tf32(None)  # nothing to put back, no error


def test_checkpoint_check_rejects_foreign_and_broken_files(tmp_path):
    good = tmp_path / "sam3.pt"
    torch.save({"model": {"detector.backbone.w": torch.zeros(1), "tracker.decoder.w": torch.zeros(1)}}, good)
    backend.check_checkpoint(str(good))
    other = tmp_path / "other.pt"
    torch.save({"encoder.w": torch.zeros(1)}, other)
    with pytest.raises(RuntimeError, match="not a SAM 3 checkpoint"):
        backend.check_checkpoint(str(other))
    broken = tmp_path / "broken.pt"
    broken.write_bytes(good.read_bytes()[:100])
    with pytest.raises(RuntimeError, match="(?s)incomplete download.*huggingface.co/facebook/sam3"):
        backend.check_checkpoint(str(broken))


def test_load_model_builds_once_and_keeps_real_holes(monkeypatch, tmp_path):
    path = tmp_path / "sam3.pt"
    path.write_bytes(b"x")
    builds = []

    def build(**kwargs):
        builds.append(kwargs)
        model = FakeModel()
        model.inst_interactive_predictor = types.SimpleNamespace(_transforms=types.SimpleNamespace(max_hole_area=256.0))
        return model

    monkeypatch.setattr(backend, "_CACHE", {})
    monkeypatch.setattr(backend, "resolve_checkpoint", lambda name: str(path))
    monkeypatch.setattr(backend, "ensure_sam3", lambda: (build, None))
    monkeypatch.setattr(backend, "sam3_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(backend, "check_checkpoint", lambda p: None)
    monkeypatch.setattr(backend, "bpe_path", lambda: "vocab.txt.gz")
    monkeypatch.setattr(backend.torch.cuda, "device", lambda device: contextlib.nullcontext())
    model = backend.load_model("sam3.pt")
    assert backend.load_model("sam3.pt") is model and len(builds) == 1
    assert builds[0]["enable_inst_interactivity"] is True and builds[0]["load_from_HF"] is False
    assert builds[0]["device"] == "cpu" and builds[0]["checkpoint_path"] == str(path)
    assert builds[0]["bpe_path"] == "vocab.txt.gz"  # always passed: bundled copies of sam3 look in the wrong place
    assert model.inst_interactive_predictor._transforms.max_hole_area == 0.0
    assert not any(p.requires_grad for p in model.parameters())


def test_failed_gpu_move_puts_the_model_back(monkeypatch):
    class Unmovable(FakeModel):
        def to(self, device, *args, **kwargs):
            if torch.device(device).type != "cpu":
                raise RuntimeError("CUDA out of memory")
            return super().to(device, *args, **kwargs)

    offloaded = []
    monkeypatch.setattr(backend, "make_room_on_gpu", lambda nbytes: None)
    monkeypatch.setattr(backend, "offload", lambda model: offloaded.append(model))
    model = Unmovable()
    with pytest.raises(RuntimeError, match="out of memory"):
        backend.move_model(model, torch.device("cuda", 0))
    assert offloaded == [model]


def test_segment_offloads_when_the_run_fails(monkeypatch):
    model = FakeModel()
    offloaded = []
    monkeypatch.setattr(backend, "load_model", lambda checkpoint: model)
    monkeypatch.setattr(backend, "ensure_sam3", lambda: (None, FakeProcessor))
    monkeypatch.setattr(backend, "sam3_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(backend, "move_model", lambda m, device: None)
    monkeypatch.setattr(backend, "offload", lambda m: offloaded.append(m))

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(backend, "segment_images", broken)
    with pytest.raises(RuntimeError, match="boom"):
        backend.segment(torch.rand(1, 8, 8, 3), ["x"])
    assert offloaded == [model]
    with pytest.raises(ValueError, match="empty image batch"):
        backend.segment(torch.rand(0, 8, 8, 3), ["x"])


def test_offload_clears_position_caches_and_moves_rotary_tables():
    PositionEmbeddingSine = type("PositionEmbeddingSine", (torch.nn.Module,), {})
    model = FakeModel()
    model.pos = PositionEmbeddingSine()
    model.pos.cache = {(72, 72): torch.zeros(4)}
    model.rope = torch.nn.Module()
    model.rope.freqs_cis = torch.zeros(3)
    backend._move_tables(model, torch.device("meta"))
    assert model.rope.freqs_cis.device.type == "meta"
    model.rope.freqs_cis = torch.zeros(3)  # meta tensors hold no data to bring back
    backend.offload(model)
    assert model.pos.cache == {} and model.rope.freqs_cis.device.type == "cpu"


def test_wide_pictures_get_a_note(capsys):
    model = FakeModel()
    backend.segment_images(model, lambda threshold: FakeProcessor(threshold), torch.rand(1, 100, 600, 3), ["nothing"],
                           detail_level="fast", device=torch.device("cpu"))
    assert "wider than 4:1" in capsys.readouterr().out


def test_one_and_four_channel_images_become_rgb():
    gray = torch.rand(6, 8, 1)
    assert torch.equal(backend._image_chw(gray), gray.permute(2, 0, 1).expand(3, -1, -1))
    rgba = torch.rand(6, 8, 4)
    assert torch.equal(backend._image_chw(rgba), rgba[..., :3].permute(2, 0, 1))
    with pytest.raises(ValueError, match="1, 3 or 4 channels"):
        backend._image_chw(torch.rand(6, 8, 2))


def _fake_sam3_package(monkeypatch, folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "__init__.py").write_text("")
    monkeypatch.setitem(sys.modules, "sam3", types.SimpleNamespace(__file__=str(folder / "__init__.py"),
                                                                    __path__=[str(folder)]))


def test_bpe_path_is_found_in_the_imported_sam3_copy(monkeypatch, tmp_path):
    # ComfyUI-RMBG's layout: <pack>/models/sam3/assets/<vocab>, and its builder wrongly looks in <pack>/models/assets
    package = tmp_path / "ComfyUI-RMBG" / "models" / "sam3"
    _fake_sam3_package(monkeypatch, package)
    with pytest.raises(RuntimeError, match="has no assets/bpe_simple_vocab_16e6.txt.gz"):
        backend.bpe_path()
    (package / "assets").mkdir()
    (package / "assets" / backend.BPE_FILE).write_bytes(b"x")
    assert backend.bpe_path() == os.path.normpath(str(package / "assets" / backend.BPE_FILE))


def test_bpe_path_also_accepts_assets_next_to_the_package(monkeypatch, tmp_path):
    _fake_sam3_package(monkeypatch, tmp_path / "lib" / "sam3")
    (tmp_path / "lib" / "assets").mkdir()
    (tmp_path / "lib" / "assets" / backend.BPE_FILE).write_bytes(b"x")
    assert backend.bpe_path() == os.path.normpath(str(tmp_path / "lib" / "assets" / backend.BPE_FILE))
