"""VTON / SegFormer model handling: VRAM release (keep_model_loaded) and room-making that follow where
the ONNX Runtime sessions really run, complete and pinned downloads, and model folders that follow
extra_model_paths.yaml and list only model files.

Nothing here touches the network, a GPU or the real models folder: folder_paths points at a temp
dir, huggingface_hub is a recorder, onnxruntime is a fake and the ONNX-backed classes are stubs.
"""

import gc
import importlib
import os
import sys
import types
import weakref

import numpy as np
import pytest
import torch
from PIL import Image

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PACK_DIR))
for path in (COMFY_ROOT, os.path.dirname(PACK_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

folder_paths = importlib.import_module("folder_paths")
backend = importlib.import_module("NH-Nodes.core.segformer_backend")
ort_session = importlib.import_module("NH-Nodes.nodes.vton.preprocess.ort_session")
vton = importlib.import_module("NH-Nodes.nodes.vton.vton_preprocessor_nodes")

VTON_FILES = (
    "dwpose/yolox_l.onnx", "dwpose/dw-ll_ucoco_384.onnx",
    "humanparsing/parsing_atr.onnx", "humanparsing/parsing_lip.onnx",
)
SEGFORMER_FILES = ("config.json", "model.safetensors", "preprocessor_config.json")
GIB = 1024 ** 3


# --- helpers -------------------------------------------------------------------------------------
def _norm(path):
    return os.path.normcase(os.path.normpath(str(path)))


def _touch(root, relative_files):
    for rel in relative_files:
        target = os.path.join(str(root), *rel.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as handle:
            handle.write(b"x")


def _model_list(folder_name):
    """ComfyUI's own file list for a registered folder name (what /models and the Model Library show)."""
    return sorted(name.replace("\\", "/") for name in folder_paths.get_filename_list(folder_name))


class _FakeHub:
    """huggingface_hub stand-in: records snapshot_download calls and 'downloads' `payload`."""

    def __init__(self):
        self.calls = []
        self.payload = ()
        self.error = None

    def snapshot_download(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        _touch(kwargs["local_dir"], self.payload)
        return kwargs["local_dir"]


@pytest.fixture
def hub(monkeypatch):
    fake = _FakeHub()
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=fake.snapshot_download))
    return fake


@pytest.fixture
def models_root(tmp_path, monkeypatch):
    """A fake ComfyUI install: empty models dir and a clean folder_paths registry under tmp_path."""
    root = tmp_path / "ComfyUI" / "models"
    root.mkdir(parents=True)
    monkeypatch.setattr(folder_paths, "models_dir", str(root))
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", {})
    monkeypatch.setattr(folder_paths, "filename_list_cache", {})  # file lists of earlier tests point at deleted dirs
    monkeypatch.setattr(vton, "MODELS_DIR", str(root / "ComfyUI-Vton-Mask"))
    return root


class _FakeComfyMemory:
    """comfy.model_management stand-in that logs into a shared event list."""

    reserved = 700 * 1024 ** 2  # what extra_reserved_memory() reports: VRAM ComfyUI keeps for itself

    def __init__(self, events, device=None):
        self.events = events
        self.device = torch.device("cuda", 0) if device is None else device

    def get_torch_device(self):
        return self.device

    def extra_reserved_memory(self):
        return self.reserved

    def free_memory(self, nbytes, device):
        self.events.append(("free_memory", nbytes, device))

    def soft_empty_cache(self):
        self.events.append(("soft_empty_cache",))


@pytest.fixture
def cuda_spy(monkeypatch):
    """Pretend CUDA exists and record the cache flushes the VTON node performs."""
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    real_collect = gc.collect

    def collect(*args):
        calls.append("gc.collect")
        return real_collect(*args)

    monkeypatch.setattr(gc, "collect", collect)
    return calls


@pytest.fixture
def fake_ort(monkeypatch):
    """onnxruntime stand-in. `available`: providers the installed wheel ships. `working`: providers that
    really start (a CUDA/cuDNN mismatch ships the CUDA provider but cannot start it).

    Like the real one, a session is served by the first requested provider that works, else by the CPU.
    """

    class Session:
        def __init__(self, path, sess_options=None, providers=None):
            self.path = path
            self.active = [name for name in providers or () if name in fake.working] or ["CPUExecutionProvider"]

        def get_providers(self):
            return list(self.active)

    fake = types.SimpleNamespace(
        InferenceSession=Session,
        available=["CUDAExecutionProvider", "CPUExecutionProvider"],
        working={"CUDAExecutionProvider", "CPUExecutionProvider"},
        get_available_providers=lambda: list(fake.available),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    return fake


@pytest.fixture(autouse=True)
def isolated_ort_state(monkeypatch):
    """Each test starts with no CPU fallback on record and no GPU session alive, whatever ran before it."""
    monkeypatch.setattr(ort_session, "_cpu_fallback_seen", False)
    monkeypatch.setattr(ort_session, "_warned_cpu_fallback", False)
    monkeypatch.setattr(ort_session, "_cuda_sessions", weakref.WeakSet())


@pytest.fixture
def stub_mask_location(monkeypatch):
    """process() imports get_mask_location from utils_mask (needs scikit-image): use a trivial one."""
    monkeypatch.setitem(sys.modules, "NH-Nodes.core.utils_mask",
                        types.SimpleNamespace(get_mask_location=_fake_get_mask_location))


# --- SegFormer model folders ---------------------------------------------------------------------
def test_pack_registers_its_folder_names_with_comfyui_at_import():
    assert _norm(os.path.join(folder_paths.models_dir, "NH-Nodes")) in map(_norm, folder_paths.get_folder_paths("nh_nodes"))
    assert _norm(vton.MODELS_DIR) in map(_norm, folder_paths.get_folder_paths("vton_mask"))


def test_pack_limits_its_model_lists_to_model_files_at_import():
    registry = folder_paths.folder_names_and_paths
    assert ".onnx" in registry["vton_mask"][1]
    assert ".safetensors" in registry["nh_nodes"][1]


def test_model_lists_show_model_files_only_not_hf_cache_metadata(models_root):
    _touch(vton.MODELS_DIR, [*VTON_FILES, ".gitattributes", "README.md",
                             ".cache/huggingface/download/dwpose/yolox_l.onnx.metadata",
                             ".cache/huggingface/download/dwpose/yolox_l.onnx.lock",
                             ".cache/huggingface/.gitignore"])
    _touch(models_root / "NH-Nodes", [
        "segformer_fashion/config.json", "segformer_fashion/model.safetensors", "segformer_fashion/preprocessor_config.json",
        "segformer_fashion/.cache/huggingface/download/model.safetensors.metadata",
        "segformer_fashion/.gitattributes", "face_parsing/model.safetensors"])
    vton.resolve_vton_models_dir(download=False)  # registers vton_mask, as the pack does at start-up
    backend.model_search_dirs("segformer_fashion")  # registers nh_nodes

    assert _model_list("vton_mask") == sorted(VTON_FILES)
    assert _model_list("nh_nodes") == ["face_parsing/model.safetensors", "segformer_fashion/model.safetensors"]


def test_extension_filter_joins_a_folder_name_the_user_already_registered(models_root, tmp_path):
    shared = tmp_path / "shared" / "ComfyUI-Vton-Mask"
    _touch(shared, [*VTON_FILES, ".gitattributes"])
    folder_paths.add_model_folder_path("vton_mask", str(shared))  # what extra_model_paths.yaml does: no extension filter
    assert folder_paths.folder_names_and_paths["vton_mask"][1] == set()  # i.e. ComfyUI would list every file

    vton.resolve_vton_models_dir(download=False)

    paths, extensions = folder_paths.folder_names_and_paths["vton_mask"]
    assert [_norm(p) for p in paths] == [_norm(shared), _norm(vton.MODELS_DIR)]  # the user's location is kept, first
    assert extensions == {".onnx"}
    assert _model_list("vton_mask") == sorted(VTON_FILES)


def test_extension_filter_does_not_change_a_set_comfyui_shares_between_folders(models_root, tmp_path):
    shared_extensions = {".pt"}  # like folder_paths.supported_pt_extensions, one set behind many folder names
    folder_paths.folder_names_and_paths["vton_mask"] = ([str(tmp_path / "other")], shared_extensions)

    vton.resolve_vton_models_dir(download=False)

    assert shared_extensions == {".pt"}
    assert folder_paths.folder_names_and_paths["vton_mask"][1] == {".pt", ".onnx"}


def test_registering_the_extension_filter_twice_changes_nothing(models_root):
    default = str(models_root / "NH-Nodes")
    for _ in range(2):
        backend.register_model_folder("nh_nodes", default, {".safetensors"})
    assert folder_paths.folder_names_and_paths["nh_nodes"] == ([default], {".safetensors"})


def test_a_failing_extension_filter_does_not_lose_the_registered_locations(models_root, tmp_path, monkeypatch, capsys):
    class ReadOnlyRegistry(dict):
        def __setitem__(self, key, value):
            raise RuntimeError("registry is read-only")

    user_dir = str(tmp_path / "shared" / "NH-Nodes")
    registry = ReadOnlyRegistry()
    dict.__setitem__(registry, "nh_nodes", ([user_dir], set()))
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", registry)

    default = str(models_root / "NH-Nodes")
    assert backend.register_model_folder("nh_nodes", default, {".safetensors"}) == [user_dir, default]
    assert "read-only" in capsys.readouterr().out


def test_default_search_order_is_unchanged_for_existing_installs(models_root):
    expected = {
        "segformer_fashion": ["NH-Nodes/segformer_fashion", "RMBG/segformer_fashion"],
        "face_parsing": ["NH-Nodes/face_parsing", "face_parsing", "RMBG/segformer_face"],
    }
    for model_name, relative in expected.items():
        assert [_norm(p) for p in backend.model_search_dirs(model_name)] == [_norm(models_root / r) for r in relative]


@pytest.mark.parametrize("legacy", [("segformer_fashion", "RMBG/segformer_fashion"),
                                    ("face_parsing", "face_parsing"),
                                    ("face_parsing", "RMBG/segformer_face"),
                                    ("segformer_fashion", "NH-Nodes/segformer_fashion")])
def test_models_already_on_disk_are_found_without_download(models_root, hub, legacy):
    model_name, relative = legacy
    _touch(models_root / relative, SEGFORMER_FILES)
    assert _norm(backend.resolve_model_dir(model_name)) == _norm(models_root / relative)
    assert hub.calls == []


def test_extra_model_paths_location_is_found_and_wins_over_the_default(models_root, tmp_path, hub):
    shared = tmp_path / "shared" / "NH-Nodes"
    _touch(shared / "segformer_fashion", SEGFORMER_FILES)
    # ComfyUI applies extra_model_paths.yaml before custom nodes load, so the user's path is registered first.
    folder_paths.add_model_folder_path("nh_nodes", str(shared))
    _touch(models_root / "NH-Nodes" / "segformer_fashion", SEGFORMER_FILES)

    assert _norm(backend.resolve_model_dir("segformer_fashion")) == _norm(shared / "segformer_fashion")
    assert hub.calls == []
    # The default location stays searched, so models that already live there are never orphaned.
    assert _norm(models_root / "NH-Nodes") in map(_norm, folder_paths.get_folder_paths("nh_nodes"))


def test_registering_twice_does_not_duplicate_the_default(models_root):
    default = str(models_root / "NH-Nodes")
    assert backend.register_model_folder("nh_nodes", default) == [default]
    assert backend.register_model_folder("nh_nodes", default) == [default]
    assert folder_paths.get_folder_paths("nh_nodes") == [default]


def test_incomplete_model_folder_is_skipped_and_the_next_candidate_is_used(models_root, hub):
    _touch(models_root / "NH-Nodes" / "segformer_fashion", ["config.json"])  # interrupted download
    _touch(models_root / "RMBG" / "segformer_fashion", SEGFORMER_FILES)
    assert _norm(backend.resolve_model_dir("segformer_fashion", download=False)) == _norm(models_root / "RMBG" / "segformer_fashion")
    (models_root / "RMBG" / "segformer_fashion" / "model.safetensors").unlink()
    assert backend.resolve_model_dir("segformer_fashion", download=False) is None
    assert hub.calls == []


@pytest.mark.parametrize("model_name, repo_id, revision", [
    ("segformer_fashion", "1038lab/segformer_fashion", "e5bae2d8e3333be9f12eb6aae454c0542c1371d3"),
    ("face_parsing", "jonathandinu/face-parsing", "758b82e15a0178c9db39c1ff666a8b56e3a550c8"),
])
def test_segformer_download_is_pinned_and_lands_in_the_default_folder(models_root, tmp_path, hub, model_name, repo_id, revision):
    hub.payload = SEGFORMER_FILES
    # An extra location without the model must not change where new downloads go.
    folder_paths.add_model_folder_path("nh_nodes", str(tmp_path / "empty-shared"))

    resolved = backend.resolve_model_dir(model_name)

    assert _norm(resolved) == _norm(models_root / "NH-Nodes" / model_name)
    assert len(hub.calls) == 1
    call = hub.calls[0]
    assert call["repo_id"] == repo_id
    assert call["revision"] == revision
    assert call["allow_patterns"] == list(SEGFORMER_FILES)  # safetensors only: no pytorch_model.bin / onnx
    assert _norm(call["local_dir"]) == _norm(resolved)
    assert "local_dir_use_symlinks" not in call


def test_segformer_download_missing_files_and_failures_are_reported(models_root, hub):
    hub.payload = SEGFORMER_FILES[:2]
    with pytest.raises(RuntimeError, match="missing after download"):
        backend.resolve_model_dir("face_parsing")
    hub.error = OSError("network down")
    with pytest.raises(RuntimeError, match="1038lab/segformer_fashion") as info:
        backend.resolve_model_dir("segformer_fashion")
    assert "network down" in str(info.value) and isinstance(info.value.__cause__, OSError)


def test_extra_model_paths_yaml_end_to_end(models_root, tmp_path, hub):
    """Feed a real extra_model_paths.yaml through ComfyUI's own loader."""
    yaml = pytest.importorskip("yaml")
    extra_config = pytest.importorskip("utils.extra_config")
    if not hasattr(extra_config, "load_extra_path_config"):
        pytest.skip("another 'utils' module shadows ComfyUI's")
    shared = tmp_path / "shared"
    _touch(shared / "NH-Nodes" / "face_parsing", [*SEGFORMER_FILES, ".gitattributes"])
    _touch(shared / "ComfyUI-Vton-Mask", [*VTON_FILES, ".gitattributes"])
    config = tmp_path / "extra_model_paths.yaml"
    config.write_text(yaml.safe_dump({"my_drive": {
        "base_path": str(shared), "nh_nodes": "NH-Nodes/", "vton_mask": "ComfyUI-Vton-Mask/"}}), encoding="utf-8")
    extra_config.load_extra_path_config(str(config))

    assert _norm(backend.resolve_model_dir("face_parsing")) == _norm(shared / "NH-Nodes" / "face_parsing")
    assert _norm(vton.resolve_vton_models_dir()) == _norm(shared / "ComfyUI-Vton-Mask")
    assert hub.calls == []
    # The user's folders are listed like ours: model files only.
    assert _model_list("vton_mask") == sorted(VTON_FILES)
    assert _model_list("nh_nodes") == ["face_parsing/model.safetensors"]


# --- VTON model folders and downloads ------------------------------------------------------------
def test_vton_folder_with_missing_files_is_not_complete(models_root, hub):
    _touch(vton.MODELS_DIR, VTON_FILES[:2])  # an interrupted first download leaves a half-filled folder
    assert vton.resolve_vton_models_dir(download=False) is None
    _touch(vton.MODELS_DIR, VTON_FILES)
    assert _norm(vton.resolve_vton_models_dir(download=False)) == _norm(vton.MODELS_DIR)
    assert hub.calls == []


def test_interrupted_download_is_completed_instead_of_skipped(models_root, hub):
    _touch(vton.MODELS_DIR, VTON_FILES[:2])
    hub.payload = VTON_FILES
    resolved = vton.resolve_vton_models_dir()
    assert len(hub.calls) == 1
    assert all(os.path.isfile(os.path.join(resolved, *name.split("/"))) for name in VTON_FILES)


def test_vton_download_is_pinned_filtered_and_symlink_free(models_root, hub):
    hub.payload = VTON_FILES
    vton.resolve_vton_models_dir()
    # An exact match also proves the deprecated local_dir_use_symlinks argument is gone.
    assert hub.calls == [{
        "repo_id": "kg-09/kg-vton-mask",
        "revision": "4648778286cac2dca9aab8da167ea5257462487d",
        "allow_patterns": ["dwpose/*.onnx", "humanparsing/*.onnx"],
        "local_dir": vton.MODELS_DIR,
    }]


def test_vton_download_that_leaves_files_missing_is_an_error(models_root, hub):
    hub.payload = VTON_FILES[:3]
    with pytest.raises(RuntimeError, match="parsing_lip.onnx"):
        vton.resolve_vton_models_dir()


def test_vton_download_failure_explains_where_to_get_the_files(models_root, hub):
    hub.error = OSError("network down")
    with pytest.raises(RuntimeError, match="kg-09/kg-vton-mask") as info:
        vton.resolve_vton_models_dir()
    assert "network down" in str(info.value) and isinstance(info.value.__cause__, OSError)


def test_vton_models_in_extra_location_are_used_without_download(models_root, tmp_path, hub):
    shared = tmp_path / "shared" / "ComfyUI-Vton-Mask"
    _touch(shared, VTON_FILES)
    folder_paths.add_model_folder_path("vton_mask", str(shared))  # what an extra_model_paths.yaml entry does
    assert _norm(vton.resolve_vton_models_dir()) == _norm(shared)
    assert hub.calls == []


def test_vton_download_goes_to_the_default_folder_even_with_extra_locations(models_root, tmp_path, hub):
    hub.payload = VTON_FILES
    folder_paths.add_model_folder_path("vton_mask", str(tmp_path / "empty-shared"))
    assert _norm(vton.resolve_vton_models_dir()) == _norm(vton.MODELS_DIR)


def test_real_install_still_finds_its_models_in_the_old_places():
    """Existing installs must not need to move anything (skipped where the weights are absent)."""
    models = folder_paths.models_dir
    legacy = {
        "segformer_fashion": ("NH-Nodes/segformer_fashion", "RMBG/segformer_fashion"),
        "face_parsing": ("NH-Nodes/face_parsing", "face_parsing", "RMBG/segformer_face"),
    }
    checked = 0
    for model_name, relative_dirs in legacy.items():
        if any(all(os.path.isfile(os.path.join(models, rel, f)) for f in SEGFORMER_FILES) for rel in relative_dirs):
            assert backend.resolve_model_dir(model_name, download=False) is not None
            checked += 1
    default_vton = os.path.join(models, "ComfyUI-Vton-Mask")
    if all(os.path.isfile(os.path.join(default_vton, *name.split("/"))) for name in VTON_FILES):
        assert vton.resolve_vton_models_dir(download=False) is not None
        checked += 1
    if not checked:
        pytest.skip("no model weights on disk")


# --- keep_model_loaded: input definition ----------------------------------------------------------
def test_keep_model_loaded_is_a_new_optional_boolean_and_nothing_else_changed():
    spec = vton.NH_VTonUltimateProcessor.INPUT_TYPES()
    assert list(spec["required"]) == ["human_image", "category", "mask_feathering", "cover_shoes",
                                      "refine_hands", "refine_hair", "device"]
    assert list(spec["optional"]) == ["offset_top", "offset_bottom", "offset_left", "offset_right", "keep_model_loaded"]
    kind, options = spec["optional"]["keep_model_loaded"]
    assert kind == "BOOLEAN" and options["default"] is False
    assert "VRAM" in options["tooltip"] and "cuda" in options["tooltip"]
    assert vton.NH_VTonUltimateProcessor.RETURN_TYPES == ("MASK", "MASK", "IMAGE", "IMAGE", "IMAGE", "MASK", "MASK", "MASK")
    assert vton.NH_VTonUltimateProcessor.RETURN_NAMES == (
        "final_mask", "agnostic_mask", "densepose_image", "parsing_image", "masked_image",
        "parsing_map_raw", "hair_mask", "hands_mask")


def test_keep_model_loaded_tooltip_tells_list_users_why_to_turn_it_on():
    tooltip = vton.NH_VTonUltimateProcessor.INPUT_TYPES()["optional"]["keep_model_loaded"][1]["tooltip"]
    # ComfyUI calls a node once per item of an image list, and every call reloads the models while this is off.
    assert "list of images" in tooltip and "once per item" in tooltip
    assert "much faster" in tooltip and "batch" in tooltip  # a batch tensor is one call; lists and repeated runs benefit
    assert "CPU" in tooltip  # it has no effect on models that run on the CPU, including after a fallback


# --- keep_model_loaded: releasing the sessions ----------------------------------------------------
class _FakePose:
    """DWposeDetector stand-in. It references itself so only a real gc.collect() can free it, and it owns
    a session made by create_session, so ort_session knows whether this model lives on the GPU."""

    def __init__(self, device="cpu"):
        self.me = self
        self.session = ort_session.create_session("dwpose/yolox_l.onnx", device)

    def __call__(self, bgr):
        height, width = bgr.shape[:2]
        return np.zeros((height, width, 3), np.uint8), None, None, np.full((1, 134, 2), 0.5, np.float32)


class _ExplodingPose(_FakePose):
    def __call__(self, bgr):
        raise RuntimeError("boom")


class _FakeParsing:
    def __call__(self, image):
        labels = np.zeros(np.array(image).shape[:2], np.uint8)
        labels[:, : labels.shape[1] // 2] = 5
        return Image.fromarray(labels), Image.fromarray(labels)


def _fake_get_mask_location(category, parsing_map, keypoints, width, height, *offsets):
    return Image.fromarray(((np.array(parsing_map) == 5) * 255).astype(np.uint8)), None


@pytest.fixture
def cached_models(monkeypatch, fake_ort, stub_mask_location):
    """Factory that installs fake cached sessions for a device and returns a weakref to the pose model.

    The session is served by whatever `fake_ort.working` allows when it is installed: set that first to
    simulate a cuda request that fell back to the CPU. Only the weakref escapes, so a test never keeps
    the sessions alive by accident.
    """

    def install(device, pose_cls=_FakePose):
        pose = pose_cls(device)
        monkeypatch.setattr(vton, "LOADED_VTON_MODELS",
                            {"dwprocessor": pose, "parsing_model": _FakeParsing(), "device": device})
        return weakref.ref(pose)

    return install


def _process(device, **kwargs):
    image = torch.full((1, 48, 64, 3), 0.5)
    return vton.NH_VTonUltimateProcessor().process(image, "Upper-body", 4, False, True, True, device, **kwargs)


def test_cuda_run_releases_the_sessions_by_default(cached_models, cuda_spy):
    pose_ref = cached_models("cuda")
    assert ort_session.cuda_sessions_alive()  # served by CUDAExecutionProvider: these pin VRAM
    outputs = _process("cuda")

    assert len(outputs) == 8
    assert vton.LOADED_VTON_MODELS is None
    assert pose_ref() is None
    assert not ort_session.cuda_sessions_alive()
    assert "gc.collect" in cuda_spy and "empty_cache" in cuda_spy
    assert cuda_spy.index("gc.collect") < cuda_spy.index("empty_cache")  # collect first, then hand VRAM back


def test_keep_model_loaded_keeps_the_cache(cached_models, cuda_spy):
    pose_ref = cached_models("cuda")
    cached = vton.LOADED_VTON_MODELS
    _process("cuda", keep_model_loaded=True)
    assert vton.LOADED_VTON_MODELS is cached and pose_ref() is not None
    assert cuda_spy == []
    _process("cuda", keep_model_loaded=True)  # still cached and usable
    assert vton.LOADED_VTON_MODELS is cached


def test_cuda_run_with_keep_off_after_a_keep_on_run_releases(cached_models, cuda_spy):
    pose_ref = cached_models("cuda")
    _process("cuda", keep_model_loaded=True)
    assert pose_ref() is not None
    _process("cuda", keep_model_loaded=False)
    assert pose_ref() is None and vton.LOADED_VTON_MODELS is None


@pytest.mark.parametrize("keep", [False, True])
def test_cpu_behaviour_is_unchanged_models_stay_cached(cached_models, cuda_spy, keep):
    pose_ref = cached_models("cpu")
    cached = vton.LOADED_VTON_MODELS
    _process("cpu", keep_model_loaded=keep)
    assert vton.LOADED_VTON_MODELS is cached and pose_ref() is not None
    assert cuda_spy == []


@pytest.mark.parametrize("keep", [False, True])
def test_cuda_request_that_onnxruntime_served_from_the_cpu_keeps_its_cache(cached_models, fake_ort, cuda_spy, keep):
    fake_ort.working = {"CPUExecutionProvider"}  # onnxruntime-gpu is installed but CUDA/cuDNN do not start
    pose_ref = cached_models("cuda")
    cached = vton.LOADED_VTON_MODELS
    assert ort_session.cpu_fallback_seen() and not ort_session.cuda_sessions_alive()

    _process("cuda", keep_model_loaded=keep)
    _process("cuda", keep_model_loaded=keep)  # the next item of a list: still the same, usable sessions

    # Nothing sits in VRAM: dropping the sessions would only make every list item reload four models.
    assert vton.LOADED_VTON_MODELS is cached and pose_ref() is not None
    assert cuda_spy == []  # no gc.collect(), no torch.cuda.empty_cache()


def test_failed_cuda_run_served_from_the_cpu_keeps_the_cache(cached_models, fake_ort, cuda_spy):
    fake_ort.working = {"CPUExecutionProvider"}
    pose_ref = cached_models("cuda", pose_cls=_ExplodingPose)
    with pytest.raises(RuntimeError, match="boom"):
        _process("cuda")
    assert vton.LOADED_VTON_MODELS is not None and pose_ref() is not None
    assert cuda_spy == []


def test_a_gpu_cache_is_still_released_once_cpu_sessions_exist_next_to_it(cached_models, fake_ort, cuda_spy):
    """The decision follows the sessions that are alive, not a flag of the last session created."""
    pose_ref = cached_models("cuda")  # on the GPU
    fake_ort.working = {"CPUExecutionProvider"}
    ort_session.create_session("other.onnx", "cuda")  # a later session that fell back
    _process("cuda")
    assert vton.LOADED_VTON_MODELS is None and pose_ref() is None
    assert "empty_cache" in cuda_spy


def test_sessions_are_released_even_when_processing_fails(cached_models, cuda_spy):
    pose_ref = cached_models("cuda", pose_cls=_ExplodingPose)
    with pytest.raises(RuntimeError, match="boom"):
        _process("cuda")  # the traceback is still alive in this block: it must not pin the sessions
    assert vton.LOADED_VTON_MODELS is None
    assert pose_ref() is None
    assert "empty_cache" in cuda_spy


def test_failed_run_on_cpu_keeps_the_cache(cached_models, cuda_spy):
    pose_ref = cached_models("cpu", pose_cls=_ExplodingPose)
    with pytest.raises(RuntimeError, match="boom"):
        _process("cpu")
    assert pose_ref() is not None and cuda_spy == []


def test_a_failing_cache_flush_does_not_hide_the_original_error(cached_models, cuda_spy, monkeypatch, capsys):
    def broken_flush():
        raise RuntimeError("CUDA error: device-side assert")

    pose_ref = cached_models("cuda", pose_cls=_ExplodingPose)
    monkeypatch.setattr(torch.cuda, "empty_cache", broken_flush)
    with pytest.raises(RuntimeError, match="boom"):  # the run's own error, not the flush error
        _process("cuda")
    assert vton.LOADED_VTON_MODELS is None and pose_ref() is None
    assert "device-side assert" in capsys.readouterr().out


def test_release_without_cuda_skips_empty_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(vton, "LOADED_VTON_MODELS", {"device": "cuda"})
    vton.release_vton_models()
    assert vton.LOADED_VTON_MODELS is None and calls == []


def test_dropping_sessions_that_ran_on_the_cpu_leaves_torchs_cache_alone(cached_models, fake_ort, cuda_spy):
    fake_ort.working = {"CPUExecutionProvider"}
    pose_ref = cached_models("cuda")  # a cuda request that onnxruntime served from the CPU
    vton.release_vton_models()
    assert vton.LOADED_VTON_MODELS is None and pose_ref() is None  # still dropped (and collected) ...
    assert "gc.collect" in cuda_spy and "empty_cache" not in cuda_spy  # ... but there is no VRAM to hand back


def test_dropping_sessions_that_ran_on_the_gpu_hands_the_vram_back(cached_models, cuda_spy):
    pose_ref = cached_models("cuda")
    vton.release_vton_models()
    assert vton.LOADED_VTON_MODELS is None and pose_ref() is None
    assert cuda_spy == ["gc.collect", "empty_cache"]


# --- making room before creating CUDA sessions ----------------------------------------------------
@pytest.fixture
def stub_loaders(monkeypatch, models_root, hub, fake_ort, stub_mask_location):
    """load_model() with stub ONNX classes and no model files yet; returns the event log.

    The stubs open their sessions with create_session, so the fake onnxruntime decides where they run.
    """
    events = []
    monkeypatch.setattr(vton, "LOADED_VTON_MODELS", None)
    monkeypatch.setattr(ort_session, "ensure_onnxruntime", lambda: None)

    class Detector(_FakePose):
        def __init__(self, model_root, device):
            super().__init__(device)
            events.append(("detector", device, model_root))

    class Parsing(_FakeParsing):
        def __init__(self, model_root, device):
            self.session = ort_session.create_session("humanparsing/parsing_atr.onnx", device)
            events.append(("parsing", device, model_root))

    monkeypatch.setitem(sys.modules, "NH-Nodes.nodes.vton.preprocess.dwpose", types.SimpleNamespace(DWposeDetector=Detector))
    monkeypatch.setitem(sys.modules, "NH-Nodes.nodes.vton.preprocess.humanparsing.run_parsing", types.SimpleNamespace(Parsing=Parsing))
    return events


@pytest.fixture
def ready_loaders(stub_loaders):
    """stub_loaders plus a complete model folder in the default location."""
    _touch(vton.MODELS_DIR, VTON_FILES)
    return stub_loaders


def test_cuda_load_asks_comfyui_for_room_before_creating_sessions(ready_loaders, monkeypatch, cuda_spy):
    comfy = _FakeComfyMemory(ready_loaders)
    monkeypatch.setattr(backend, "_mm", comfy)
    vton.NH_VTonUltimateProcessor().load_model("cuda")

    assert [event[0] for event in ready_loaders] == ["free_memory", "soft_empty_cache", "detector", "parsing"]
    _, nbytes, device = ready_loaders[0]
    assert vton._VTON_VRAM_ESTIMATE >= 2 * GIB  # the four sessions hold about 2.9 GB after a run
    assert nbytes == vton._VTON_VRAM_ESTIMATE + comfy.reserved  # on top of the reserve ComfyUI keeps for itself
    assert device == torch.device("cuda", 0)  # ComfyUI's device, with the index its loaded models carry


def test_cpu_load_never_touches_comfyui_memory(ready_loaders, monkeypatch, cuda_spy):
    monkeypatch.setattr(backend, "_mm", _FakeComfyMemory(ready_loaders))
    vton.NH_VTonUltimateProcessor().load_model("cpu")
    assert [event[0] for event in ready_loaders] == ["detector", "parsing"]


def test_cuda_request_without_the_cuda_provider_does_not_evict_comfyui_models(ready_loaders, fake_ort, monkeypatch, cuda_spy):
    monkeypatch.setattr(backend, "_mm", _FakeComfyMemory(ready_loaders))
    fake_ort.available = ["CPUExecutionProvider"]  # CPU-only onnxruntime wheel
    fake_ort.working = {"CPUExecutionProvider"}
    vton.NH_VTonUltimateProcessor().load_model("cuda")
    assert [event[0] for event in ready_loaders] == ["detector", "parsing"]


def test_cuda_run_that_falls_back_asks_for_room_once_then_stops_evicting(ready_loaders, fake_ort, monkeypatch, cuda_spy):
    monkeypatch.setattr(backend, "_mm", _FakeComfyMemory(ready_loaders))
    fake_ort.working = {"CPUExecutionProvider"}  # onnxruntime-gpu is installed, but CUDA/cuDNN do not start

    # First run: nothing says yet that the GPU is unusable, so ComfyUI is asked, once.
    assert len(_process("cuda")) == 8
    assert [event[0] for event in ready_loaders] == ["free_memory", "soft_empty_cache", "detector", "parsing"]
    assert ort_session.cpu_fallback_seen()
    assert vton.LOADED_VTON_MODELS is not None and cuda_spy == []  # CPU sessions are not released

    # Every later load in this process skips it: whatever it would evict cannot help sessions on the CPU.
    ready_loaders.clear()
    monkeypatch.setattr(vton, "LOADED_VTON_MODELS", None)
    _process("cuda")
    vton.NH_VTonUltimateProcessor().load_model("cpu")
    monkeypatch.setattr(vton, "LOADED_VTON_MODELS", None)
    vton.NH_VTonUltimateProcessor().load_model("cuda")
    assert [event[0] for event in ready_loaders] == ["detector", "parsing"] * 3


def test_cuda_runs_on_the_gpu_keep_asking_for_room_and_release(ready_loaders, monkeypatch, cuda_spy):
    monkeypatch.setattr(backend, "_mm", _FakeComfyMemory(ready_loaders))
    for _ in range(2):
        ready_loaders.clear()
        _process("cuda")
        assert [event[0] for event in ready_loaders] == ["free_memory", "soft_empty_cache", "detector", "parsing"]
        assert vton.LOADED_VTON_MODELS is None  # released after the run, so the next one reloads
    assert not ort_session.cpu_fallback_seen() and not ort_session.cuda_sessions_alive()
    assert "empty_cache" in cuda_spy


def test_load_model_uses_the_resolved_folder_and_caches_per_device(stub_loaders, tmp_path):
    shared = tmp_path / "shared" / "ComfyUI-Vton-Mask"
    _touch(shared, VTON_FILES)  # the default folder stays empty
    folder_paths.add_model_folder_path("vton_mask", str(shared))

    node = vton.NH_VTonUltimateProcessor()
    first = node.load_model("cpu")
    assert [_norm(event[2]) for event in stub_loaders] == [_norm(shared)] * 2
    assert node.load_model("cpu") is first and len(stub_loaders) == 2  # cache hit: nothing is rebuilt


def test_switching_device_drops_the_old_sessions_first(ready_loaders, monkeypatch, cuda_spy):
    monkeypatch.setattr(backend, "_mm", _FakeComfyMemory(ready_loaders))
    node = vton.NH_VTonUltimateProcessor()
    cuda_ref = weakref.ref(node.load_model("cuda")["dwprocessor"])
    ready_loaders.clear()
    cpu_models = node.load_model("cpu")
    assert cpu_models["device"] == "cpu" and cuda_ref() is None
    assert [event[0] for event in ready_loaders] == ["detector", "parsing"]


def test_switching_device_away_from_cpu_fallback_sessions_does_not_flush_torchs_cache(ready_loaders, fake_ort, monkeypatch, cuda_spy):
    monkeypatch.setattr(backend, "_mm", _FakeComfyMemory(ready_loaders))
    fake_ort.working = {"CPUExecutionProvider"}
    node = vton.NH_VTonUltimateProcessor()
    cuda_ref = weakref.ref(node.load_model("cuda")["dwprocessor"])
    cuda_spy.clear()

    assert node.load_model("cpu")["device"] == "cpu"
    assert cuda_ref() is None and "empty_cache" not in cuda_spy


def test_missing_onnxruntime_still_fails_before_any_download(models_root, hub, monkeypatch):
    def not_installed():
        raise RuntimeError("ONNX Runtime missing")

    monkeypatch.setattr(vton, "LOADED_VTON_MODELS", None)
    monkeypatch.setattr(ort_session, "ensure_onnxruntime", not_installed)
    with pytest.raises(RuntimeError, match="ONNX Runtime"):
        vton.NH_VTonUltimateProcessor().load_model("cuda")
    assert hub.calls == []


def test_cuda_provider_available_reflects_the_onnxruntime_build(monkeypatch):
    fake = types.SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    assert ort_session.cuda_provider_available() is False
    fake.get_available_providers = lambda: ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    assert ort_session.cuda_provider_available() is True


# --- where the sessions really run -------------------------------------------------------------------
def test_create_session_records_which_sessions_the_gpu_serves(fake_ort):
    gpu = ort_session.create_session("a.onnx", "cuda")
    cpu = ort_session.create_session("b.onnx", "cpu")

    assert gpu.get_providers()[0] == "CUDAExecutionProvider" and cpu.get_providers() == ["CPUExecutionProvider"]
    assert ort_session.cuda_sessions_alive() and not ort_session.cpu_fallback_seen()
    del gpu
    gc.collect()
    assert not ort_session.cuda_sessions_alive()  # the record never keeps a session (and its VRAM) alive


def test_a_cuda_request_served_by_the_cpu_is_a_fallback_not_a_gpu_session(fake_ort, capsys):
    fake_ort.working = {"CPUExecutionProvider"}
    session = ort_session.create_session("a.onnx", "cuda")

    assert session.get_providers() == ["CPUExecutionProvider"]
    assert ort_session.cpu_fallback_seen() and not ort_session.cuda_sessions_alive()
    assert "a.onnx on CPUExecutionProvider" in capsys.readouterr().out  # the existing warning is still printed


def test_the_fallback_is_remembered_even_after_its_warning_was_printed(fake_ort, monkeypatch):
    fake_ort.working = {"CPUExecutionProvider"}
    monkeypatch.setattr(ort_session, "_warned_cpu_fallback", True)  # the one-time warning is already used up
    ort_session.create_session("a.onnx", "cuda")
    assert ort_session.cpu_fallback_seen()


def test_a_session_that_reports_no_provider_counts_as_a_fallback(fake_ort):
    fake_ort.InferenceSession.get_providers = lambda self: []
    ort_session.create_session("a.onnx", "cuda")
    assert ort_session.cpu_fallback_seen() and not ort_session.cuda_sessions_alive()


def test_cpu_requests_are_never_a_fallback(fake_ort):
    fake_ort.working = {"CPUExecutionProvider"}
    ort_session.create_session("a.onnx", "cpu")
    assert not ort_session.cpu_fallback_seen() and not ort_session.cuda_sessions_alive()


# --- the shared GPU helper -------------------------------------------------------------------------
def test_make_room_is_a_noop_without_cuda_in_comfy_cpu_mode_or_outside_comfyui(monkeypatch):
    events = []
    monkeypatch.setattr(backend, "_mm", _FakeComfyMemory(events))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    backend.make_room_on_gpu(GIB)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(backend, "_mm", _FakeComfyMemory(events, device=torch.device("cpu")))  # ComfyUI --cpu
    backend.make_room_on_gpu(GIB)

    monkeypatch.setattr(backend, "_mm", None)
    backend.make_room_on_gpu(GIB)
    assert events == []


def test_make_room_asks_for_the_size_plus_the_reserve_comfyui_keeps_for_itself(monkeypatch):
    events = []
    comfy = _FakeComfyMemory(events)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(backend, "_mm", comfy)
    backend.make_room_on_gpu(GIB)
    # ComfyUI's own loaders ask for their size plus extra_reserved_memory(); asking for less would leave us short.
    assert events == [("free_memory", GIB + comfy.reserved, torch.device("cuda", 0)), ("soft_empty_cache",)]


def test_make_room_on_a_comfyui_without_extra_reserved_memory_asks_for_the_size_alone(monkeypatch):
    events = []

    class OlderComfy(_FakeComfyMemory):
        extra_reserved_memory = None  # the function does not exist in older ComfyUI builds

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(backend, "_mm", OlderComfy(events))
    backend.make_room_on_gpu(GIB)
    assert events[0] == ("free_memory", GIB, torch.device("cuda", 0))


def test_make_room_reports_comfyui_errors_instead_of_failing_the_run(monkeypatch, capsys):
    class Broken(_FakeComfyMemory):
        def free_memory(self, nbytes, device):
            raise RuntimeError("model list changed")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(backend, "_mm", Broken([]))
    backend.make_room_on_gpu(GIB)  # must not raise
    assert "model list changed" in capsys.readouterr().out


class _FakeSegformer:
    def __init__(self, device):
        self.device = device
        self.moved_to = []

    def parameters(self):
        return iter([torch.zeros(1000)])  # 4000 bytes

    def to(self, device):
        self.moved_to.append(device)
        self.device = device
        return self


def test_segformer_asks_comfyui_for_room_using_comfyuis_own_device(monkeypatch):
    events = []
    comfy = _FakeComfyMemory(events)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(backend, "_mm", comfy)
    model = _FakeSegformer(torch.device("cpu"))
    backend._ensure_on_device({"model": model, "device": torch.device("cuda")})

    # ComfyUI's loaded models carry cuda:0; an index-less cuda device matches none of them.
    # The weights (4000 bytes) are asked for twice over, plus the reserve ComfyUI keeps for itself.
    assert events[0] == ("free_memory", 8000 + comfy.reserved, torch.device("cuda", 0))
    assert model.moved_to == [torch.device("cuda")]


def test_segformer_already_on_the_gpu_is_not_moved_or_evicted_again(monkeypatch):
    events = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(backend, "_mm", _FakeComfyMemory(events))
    model = _FakeSegformer(torch.device("cuda", 0))  # what .to("cuda") leaves behind
    backend._ensure_on_device({"model": model, "device": torch.device("cuda")})
    assert events == [] and model.moved_to == []
