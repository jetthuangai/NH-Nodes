"""Wave 3 (caching / cost): Layer Stack, Load Images Matching, indexed loaders and the large-image viewer."""

import asyncio
import copy
import gc
import importlib
import json
import math
import os
import shutil
import subprocess
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import comfy.cli_args
import comfy.sd
import comfy.utils
import folder_paths
from comfy_execution.graph_utils import ExecutionBlocker

folder_loaders = importlib.import_module("NH-Nodes.nodes.image.folder_loaders")
image_folders = importlib.import_module("NH-Nodes.core.image_folders")
layer_composite = importlib.import_module("NH-Nodes.nodes.image.layer_composite")
loaders = importlib.import_module("NH-Nodes.nodes.loaders.loader_index_nodes")

PACK_DIR = Path(__file__).resolve().parent.parent


def _norm(path):
    return os.path.normcase(os.path.normpath(str(path)))


def _inside(path, folder):
    path, folder = _norm(path), _norm(folder)
    return path == folder or path.startswith(folder.rstrip(os.sep) + os.sep)


# --- 1. Layer Stack Composite is a pure function: it must not opt out of ComfyUI's cache ---

def test_layer_stack_has_no_is_changed_so_it_can_be_cached():
    assert not hasattr(layer_composite.NH_LayerStackComposite, "IS_CHANGED")


def test_layer_stack_depends_only_on_its_inputs():
    node = layer_composite.NH_LayerStackComposite()
    background = torch.zeros(1, 8, 8, 3)
    layer = torch.ones(1, 4, 4, 3)
    first = node.composite(background, 1, "2,2", layer_1=layer)
    second = node.composite(background, 1, "2,2", layer_1=layer)
    assert torch.equal(first[0], second[0]) and first[1:] == second[1:]
    assert float(background.sum()) == 0.0 and float(layer.min()) == 1.0  # inputs are not modified


# --- 2. Load Images Matching ---

def _png(path, value=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (value, value, value)).save(path)
    return path


def _touch(path):
    """Give the file a new modification time, the way an edit does."""
    stat = os.stat(path)
    new = stat.st_mtime_ns + 5_000_000_000
    os.utime(path, ns=(new, new))


def _args(**overrides):
    args = {
        "source_folder_path": "", "matching_root_path": "", "match_text": "", "match_mode": "contains",
        "recursive": True, "index": 0, "step": 1, "repeat_count": 1,
        "context_folder_names": "Nam\nNu", "source_file_names": "",
    }
    args.update(overrides)
    return args


def _load(**overrides):
    return folder_loaders.NH_LoadImagesMatching().load_matching(**_args(**overrides))


def _changed(**overrides):
    return folder_loaders.NH_LoadImagesMatching.IS_CHANGED(**_args(**overrides))


# The node only reads the names in the source path, never the folder, so tests use one that does not exist (shared
# parent folders with the matching tree no longer matter, see tests/test_folder_matching_context.py).
SOURCES = Path(os.path.abspath(os.sep)) / "nh_nodes_test_source_photos"


@pytest.fixture
def studio(tmp_path):
    """The layout the node is made for: matching images in <root>/<context>/<person>, and a source person folder."""
    root = tmp_path / "matching"
    for context, base in (("Nam", 10), ("Nu", 110)):
        for person, shade in (("Anh", 0), ("Binh", 20)):
            _png(root / context / person / "01.png", base + shade)
            _png(root / context / person / "02.png", base + shade + 5)
    return SimpleNamespace(base=tmp_path, root=root, source=SOURCES / "Nu" / "Anh")


class _Fs:
    """Records every folder the code under test lists or walks, to prove where a search did and did not look."""

    def __init__(self, monkeypatch):
        self.listed = []
        real_scandir, real_listdir = os.scandir, os.listdir

        def scandir(path="."):
            self._record(path)
            return real_scandir(path)

        def listdir(path="."):
            self._record(path)
            return real_listdir(path)

        monkeypatch.setattr(os, "scandir", scandir)
        monkeypatch.setattr(os, "listdir", listdir)

    def _record(self, path):
        try:
            self.listed.append(_norm(os.path.abspath(os.fsdecode(path))))
        except (TypeError, ValueError):
            pass

    def touched(self, folder):
        return [path for path in self.listed if _inside(path, folder)]


# 2a. Existing lookups keep working

def test_success_path_keeps_its_outputs_and_reports_found(studio):
    images, filename, path, matched_text, found = _load(
        source_folder_path=str(studio.source), matching_root_path=str(studio.root),
    )
    assert found is True
    assert tuple(images.shape) == (1, 8, 8, 3)
    assert filename == "01.png"
    assert _norm(path) == _norm(studio.root / "Nu" / "Anh")  # the match text is the source folder name
    assert matched_text == "Anh"


def test_context_folder_of_the_source_decides_between_equal_names(studio):
    for context in ("Nu", "Nam"):
        source = SOURCES / context / "Anh"
        _, _, path, _, found = _load(source_folder_path=str(source), matching_root_path=str(studio.root))
        assert found and _norm(path) == _norm(studio.root / context / "Anh")


def test_explicit_match_text_with_contains_and_exact(studio):
    _png(studio.root / "Nu" / "Binh Duong" / "01.png")
    kwargs = {"source_folder_path": str(studio.source), "matching_root_path": str(studio.root)}
    _, _, path, text, found = _load(match_text="duong", **kwargs)
    assert found and _norm(path) == _norm(studio.root / "Nu" / "Binh Duong") and text == "duong"
    _, _, path, _, found = _load(match_text="binh", match_mode="exact", **kwargs)
    assert found and _norm(path) == _norm(studio.root / "Nu" / "Binh")
    _, _, _, _, found = _load(match_text="duong", match_mode="exact", **kwargs)
    assert found is False


def test_non_recursive_search_looks_at_the_context_folders_below_the_root(studio):
    # <root>/Nu/Anh is two levels down, but the source's context ("Nu") makes <root>/Nu a place to look.
    kwargs = {"matching_root_path": str(studio.root), "recursive": False, "match_text": "Anh"}
    _, _, path, _, found = _load(source_folder_path=str(studio.source), **kwargs)
    assert found and _norm(path) == _norm(studio.root / "Nu" / "Anh")
    _, _, _, _, found = _load(source_folder_path=str(SOURCES / "neutral"), **kwargs)
    assert found is False


def test_source_file_names_give_the_match_text_and_the_context(studio):
    _, filename, path, text, found = _load(
        source_file_names="Nu/Anh/photo.png", matching_root_path=str(studio.root),
    )
    assert found and text == "Anh" and filename == "01.png"
    assert _norm(path) == _norm(studio.root / "Nu" / "Anh")


def test_legacy_person_folder_path_still_names_the_source(studio):
    _, _, path, _, found = folder_loaders.NH_LoadImagesMatching().load_matching(
        **_args(matching_root_path=str(studio.root)), person_folder_path=str(studio.source),
    )
    assert found and _norm(path) == _norm(studio.root / "Nu" / "Anh")


def test_index_step_and_repeat_pick_and_repeat_files(studio):
    images, filename, _, _, found = _load(
        source_folder_path=str(studio.source), matching_root_path=str(studio.root), index=1, step=2, repeat_count=2,
    )
    assert found
    assert filename == "02.png\n01.png"  # wraps round
    assert images.shape[0] == 4


def test_a_root_inside_one_context_also_searches_the_sibling_context(studio):
    # The root points at ".../Nam" but the source belongs to "Nu": the matching Nu folder is used.
    _, _, path, _, found = _load(source_folder_path=str(studio.source), matching_root_path=str(studio.root / "Nam"))
    assert found and _norm(path) == _norm(studio.root / "Nu" / "Anh")
    # ... and without a source context the root itself is all that is searched.
    _, _, path, _, found = _load(
        source_folder_path=str(SOURCES / "neutral" / "Anh"), matching_root_path=str(studio.root / "Nam"),
    )
    assert found and _norm(path) == _norm(studio.root / "Nam" / "Anh")


def test_the_sibling_context_keeps_the_rest_of_a_deeper_root(tmp_path):
    root = tmp_path / "match"
    _png(root / "Nam" / "Tops" / "Anh" / "01.png", 1)
    _png(root / "Nu" / "Tops" / "Anh" / "01.png", 2)
    deeper_root = str(root / "Nam" / "Tops")
    # Not recursive, only the same place in the sibling context ("Nu/Tops") lists the persons directly.
    _, _, path, _, found = _load(
        source_folder_path=str(SOURCES / "Nu" / "Tops" / "Anh"), matching_root_path=deeper_root, recursive=False,
    )
    assert found and _norm(path) == _norm(root / "Nu" / "Tops" / "Anh")
    # Recursive, the sibling context as a whole is searched, so a person kept one level higher is found as well.
    shutil.rmtree(root / "Nu" / "Tops")
    _png(root / "Nu" / "Anh" / "01.png", 3)
    _, _, path, _, found = _load(source_folder_path=str(SOURCES / "Nu" / "Anh"), matching_root_path=deeper_root)
    assert found and _norm(path) == _norm(root / "Nu" / "Anh")


# 2b. The search is bounded

def test_search_stays_inside_the_matching_root_and_its_context_siblings(tmp_path, monkeypatch):
    lab = tmp_path / "lab"
    _png(lab / "work" / "Nam" / "Anh" / "01.png")
    for n in range(4):  # a tree the old ancestor scan walked top to bottom
        _png(lab / "decoys" / f"d{n}" / "deep" / "Anh" / "01.png")
    source = _png(lab / "decoys" / "Nu" / "Anh" / "p.png").parent  # shares "decoys" with the decoy tree
    root = lab / "work" / "Nam"
    fs = _Fs(monkeypatch)
    _, _, path, _, found = _load(source_folder_path=str(source), matching_root_path=str(root))
    assert found and _norm(path) == _norm(root / "Anh")
    assert fs.touched(lab / "decoys") == []
    # Only the root's own tree, plus one listing of its parent to look for a sibling context folder.
    outside = [p for p in fs.touched(tmp_path) if not _inside(p, root)]
    assert set(outside) <= {_norm(lab / "work")}


def test_a_drive_root_is_never_walked(studio, monkeypatch):
    drive = os.path.abspath(os.sep)
    fs = _Fs(monkeypatch)
    with pytest.raises(ValueError, match="drive"):
        _load(source_folder_path=str(studio.source), matching_root_path=drive, match_text="Anh")
    assert _norm(drive) not in fs.listed


def test_the_models_folder_is_never_entered(tmp_path, monkeypatch):
    work = tmp_path / "work"
    _png(work / "Nam" / "Anh" / "01.png", 10)
    models = work / "models"
    _png(models / "previews" / "Anh" / "01.png", 99)
    monkeypatch.setattr(folder_paths, "models_dir", str(models))
    fs = _Fs(monkeypatch)
    source = _png(tmp_path / "src" / "Anh" / "p.png").parent
    _, _, path, _, found = _load(source_folder_path=str(source), matching_root_path=str(work))
    assert found and _norm(path) == _norm(work / "Nam" / "Anh")
    assert fs.touched(models) == []


def test_collecting_images_skips_the_models_folder(tmp_path, monkeypatch):
    _png(tmp_path / "work" / "Anh" / "01.png")
    _png(tmp_path / "work" / "models" / "previews" / "02.png")
    _png(tmp_path / "work" / "models" / "03.png")
    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path / "work" / "models"))
    fs = _Fs(monkeypatch)
    # No pictures directly in "work", so the walk goes down, as it always did - but not into the models folder.
    assert image_folders.collect_image_files(str(tmp_path / "work")) == [os.path.join("Anh", "01.png")]
    assert fs.touched(tmp_path / "work" / "models") == []
    # A folder inside the models folder, named on purpose, is still read.
    assert image_folders.collect_image_files(str(tmp_path / "work" / "models"), recursive=True) == [
        "03.png", os.path.join("previews", "02.png"),
    ]


def test_the_models_folder_is_refused_as_a_search_root(tmp_path, monkeypatch):
    models = tmp_path / "models"
    _png(models / "Anh" / "01.png")
    monkeypatch.setattr(folder_paths, "models_dir", str(models))
    with pytest.raises(ValueError, match="models"):
        _load(source_folder_path=str(tmp_path / "Anh"), matching_root_path=str(models), match_text="Anh")


# 2c. found

def test_nothing_matching_gives_found_false_and_a_message(studio, capsys):
    images, filename, path, matched_text, found = _load(
        source_folder_path=str(studio.source), matching_root_path=str(studio.root), match_text="Zed",
    )
    assert found is False
    assert isinstance(images, ExecutionBlocker)  # whatever needs the images is held back, with the reason
    assert "Zed" in images.message and str(studio.root) in images.message
    assert (filename, path, matched_text) == ("", "", "Zed")
    assert "Zed" in capsys.readouterr().out


def test_a_matching_folder_without_images_is_not_found(studio):
    (studio.root / "Nu" / "Empty").mkdir()
    images, _, path, text, found = _load(
        source_folder_path=str(studio.source), matching_root_path=str(studio.root), match_text="Empty",
    )
    assert found is False and path == "" and text == "Empty"
    assert isinstance(images, ExecutionBlocker) and "No images" in images.message


def test_one_missing_item_makes_the_whole_request_not_found(studio):
    both = {"matching_root_path": str(studio.root), "source_file_names": "Nu/Anh/p.png\nNu/Binh/q.png"}
    images, filename, _, text, found = _load(**both)
    assert found and images.shape[0] == 2 and filename == "01.png\n01.png" and text == "Anh\nBinh"
    images, filename, path, text, found = _load(**{**both, "source_file_names": "Nu/Anh/p.png\nNu/Zed/q.png"})
    assert found is False and isinstance(images, ExecutionBlocker)
    assert (filename, path, text) == ("", "", "Zed")


@pytest.mark.parametrize("bad_root", ["", "no-such-folder"])
def test_unusable_settings_still_raise(studio, bad_root):
    root = bad_root and str(studio.base / bad_root)
    with pytest.raises(ValueError):
        _load(source_folder_path=str(studio.source), matching_root_path=root)


# 2d. IS_CHANGED

def test_is_changed_is_a_stable_signature_instead_of_nan(studio):
    kwargs = {"source_folder_path": str(studio.source), "matching_root_path": str(studio.root)}
    first = _changed(**kwargs)
    assert isinstance(first, str) and first == _changed(**kwargs)


def test_is_changed_follows_the_selected_files_only(studio):
    kwargs = {"source_folder_path": str(studio.source), "matching_root_path": str(studio.root)}
    before = _changed(**kwargs)
    _touch(studio.root / "Nu" / "Anh" / "02.png")   # not selected (index 0, step 1)
    _touch(studio.root / "Nam" / "Binh" / "01.png")  # another person
    assert _changed(**kwargs) == before
    _touch(studio.root / "Nu" / "Anh" / "01.png")   # selected
    assert _changed(**kwargs) != before


def test_is_changed_notices_a_new_image_that_becomes_the_selected_one(studio):
    kwargs = {"source_folder_path": str(studio.source), "matching_root_path": str(studio.root)}
    before = _changed(**kwargs)
    _png(studio.root / "Nu" / "Anh" / "00.png")  # sorts first, so it is now what index 0 selects
    assert _changed(**kwargs) != before


def test_is_changed_notices_a_better_matching_folder(studio):
    shutil.rmtree(studio.root / "Nu" / "Anh")  # only Nam/Anh is left for a source in Nu
    kwargs = {"source_folder_path": str(studio.source), "matching_root_path": str(studio.root)}
    before = _changed(**kwargs)
    _png(studio.root / "Nu" / "Anh" / "01.png")
    assert _changed(**kwargs) != before


def test_is_changed_is_nan_while_nothing_matches_so_a_miss_is_not_cached(studio):
    value = _changed(source_folder_path=str(studio.source), matching_root_path=str(studio.root), match_text="Zed")
    assert math.isnan(value)


def test_is_changed_with_a_wired_source_fingerprints_the_matching_tree(studio):
    # ComfyUI passes None for an input that comes from another node, so the folder cannot be resolved up front.
    wired = {"source_folder_path": None, "source_file_names": None, "matching_root_path": str(studio.root)}
    before = _changed(**wired)
    assert isinstance(before, str) and before == _changed(**wired)
    _touch(studio.root / "Nam" / "Binh" / "01.png")
    after_edit = _changed(**wired)
    assert after_edit != before
    _png(studio.root / "Nu" / "Chi" / "01.png")
    assert _changed(**wired) != after_edit


def test_is_changed_with_a_wired_root_cannot_know_what_to_look_at(studio):
    assert math.isnan(_changed(source_folder_path=str(studio.source), matching_root_path=None))


def test_is_changed_gives_up_on_a_tree_too_big_to_fingerprint(studio, monkeypatch):
    wired = {"source_folder_path": None, "source_file_names": None, "matching_root_path": str(studio.root)}
    assert isinstance(_changed(**wired), str)
    monkeypatch.setattr(folder_loaders, "_SIGNATURE_ENTRY_LIMIT", 3)
    assert math.isnan(_changed(**wired))


# 2e. The same through ComfyUI's own cache key (what decides whether a node and everything after it runs again)

class _Text:
    """A stand-in for any node that feeds a STRING into another node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"

    def run(self, value):
        return (value,)


def _cache_key(monkeypatch, prompt, node_id):
    import nodes as comfy_nodes
    from comfy_execution.caching import CacheKeySetInputSignature
    from comfy_execution.graph import DynamicPrompt
    from execution import IsChangedCache

    for name, cls in (
        ("NH_LayerStackComposite", layer_composite.NH_LayerStackComposite),
        ("NH_LoadImagesMatching", folder_loaders.NH_LoadImagesMatching),
        ("NH_LoadImagesFromFolder", folder_loaders.NH_LoadImagesFromFolder),
        ("NH_Text", _Text),
    ):
        monkeypatch.setitem(comfy_nodes.NODE_CLASS_MAPPINGS, name, cls)

    async def compute():
        dynprompt = DynamicPrompt(copy.deepcopy(prompt))  # a queued prompt is new data; IsChangedCache writes into it
        keys = CacheKeySetInputSignature(dynprompt, [node_id], IsChangedCache("test", dynprompt, None))
        await keys.add_keys([node_id])
        return keys.get_data_key(node_id)

    return asyncio.run(compute())


def _matching_prompt(studio, wired=False, **overrides):
    settings = _args(source_folder_path=str(studio.source), matching_root_path=str(studio.root), **overrides)
    prompt = {"2": {"class_type": "NH_LoadImagesMatching", "inputs": settings}}
    if wired:
        prompt["1"] = {"class_type": "NH_Text", "inputs": {"value": settings["source_folder_path"]}}
        settings["source_folder_path"] = ["1", 0]
    return prompt


def test_layer_stack_with_unchanged_inputs_gets_the_same_cache_key(monkeypatch):
    prompt = {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 8, "height": 8, "batch_size": 1, "color": 0}},
        "2": {"class_type": "NH_LayerStackComposite", "inputs": {"background_image": ["1", 0], "layer_count": 0, "positions": "0,0"}},
    }
    first = _cache_key(monkeypatch, prompt, "2")
    assert first == _cache_key(monkeypatch, prompt, "2")  # NaN would make every queue a new key
    changed = {**prompt, "2": {"class_type": "NH_LayerStackComposite", "inputs": {**prompt["2"]["inputs"], "positions": "1,1"}}}
    assert first != _cache_key(monkeypatch, changed, "2")


@pytest.mark.parametrize("wired", [False, True])
def test_matching_cache_key_is_stable_until_a_file_it_could_read_changes(studio, monkeypatch, wired):
    prompt = _matching_prompt(studio, wired=wired)
    first = _cache_key(monkeypatch, prompt, "2")
    assert first == _cache_key(monkeypatch, prompt, "2")
    _touch(studio.root / "Nu" / "Anh" / "01.png")
    assert first != _cache_key(monkeypatch, prompt, "2")


def test_matching_cache_key_does_not_hold_a_miss(studio, monkeypatch):
    prompt = _matching_prompt(studio, match_text="Zed")
    assert _cache_key(monkeypatch, prompt, "2") != _cache_key(monkeypatch, prompt, "2")


# 2f. fallback_images: a lazy input that is asked for only when nothing matches

def _lazy(**overrides):
    """check_lazy_status the way ComfyUI calls it: every setting, and fallback_images None until it has run."""
    return folder_loaders.NH_LoadImagesMatching().check_lazy_status(**_args(**overrides))


def _where(studio, **overrides):
    return {"source_folder_path": str(studio.source), "matching_root_path": str(studio.root), **overrides}


def test_fallback_images_is_the_only_new_input_and_it_is_optional_and_lazy():
    spec = folder_loaders.NH_LoadImagesMatching.INPUT_TYPES()
    assert list(spec["required"]) == [
        "source_folder_path", "matching_root_path", "match_text", "match_mode", "recursive", "index", "step",
        "repeat_count", "context_folder_names",
    ]
    assert list(spec["optional"]) == ["source_file_names", "fallback_images"]
    kind, options = spec["optional"]["fallback_images"]
    assert kind == "IMAGE" and options["lazy"] is True
    assert folder_loaders.NH_LoadImagesMatching.RETURN_TYPES == ("IMAGE", "STRING", "STRING", "STRING", "BOOLEAN")
    assert folder_loaders.NH_LoadImagesMatching.RETURN_NAMES == ("images", "filename", "path", "matched_text", "found")


def test_description_and_tooltip_send_users_to_fallback_images_instead_of_a_switch():
    description = folder_loaders.NH_LoadImagesMatching.DESCRIPTION
    tooltip = folder_loaders.NH_LoadImagesMatching.INPUT_TYPES()["optional"]["fallback_images"][1]["tooltip"]
    for text in (description, tooltip):
        assert "instead of routing images through a switch" in text
    assert "fallback_images" in description and "nothing matches" in description and "found" in description
    assert "nothing matches" in tooltip


def test_the_fallback_is_asked_for_only_when_nothing_matches(studio):
    assert _lazy(fallback_images=None, **_where(studio)) == []  # a match: the fallback never has to run
    assert _lazy(fallback_images=None, **_where(studio, match_text="Zed")) == ["fallback_images"]
    (studio.root / "Nu" / "Empty").mkdir()  # a folder that matches but holds no images is a miss too
    assert _lazy(fallback_images=None, **_where(studio, match_text="Empty")) == ["fallback_images"]


def test_one_item_without_a_match_asks_for_the_fallback(studio):
    both = {"matching_root_path": str(studio.root), "fallback_images": None}
    assert _lazy(source_file_names="Nu/Anh/p.png\nNu/Binh/q.png", **both) == []
    assert _lazy(source_file_names="Nu/Anh/p.png\nNu/Zed/q.png", **both) == ["fallback_images"]


def test_the_legacy_person_folder_path_input_counts_for_the_fallback_too(studio):
    legacy = {
        "source_folder_path": "", "matching_root_path": str(studio.root), "person_folder_path": str(studio.source),
        "fallback_images": None,
    }
    assert _lazy(**legacy) == []
    assert _lazy(match_text="Zed", **legacy) == ["fallback_images"]


def test_a_fallback_that_is_not_connected_or_has_already_run_is_never_asked_for(studio):
    miss = _where(studio, match_text="Zed")
    assert _lazy(**miss) == []  # not connected: ComfyUI sends no key, and asking for it would be an error
    assert _lazy(fallback_images=torch.zeros(1, 4, 4, 3), **miss) == []  # connected and evaluated
    assert _lazy(fallback_images=torch.zeros(0, 4, 4, 3), **miss) == []  # an empty batch is still a value


def test_checking_for_the_fallback_never_raises_for_a_setting_that_cannot_work(studio):
    for bad_root in ("", str(studio.base / "no-such-folder")):
        assert _lazy(fallback_images=None, **_where(studio, matching_root_path=bad_root)) == []
        with pytest.raises(ValueError):  # the node itself reports it, fallback or not
            _load(fallback_images=torch.zeros(1, 4, 4, 3), **_where(studio, matching_root_path=bad_root))


def test_a_miss_returns_the_fallback_as_images_and_keeps_the_other_outputs(studio, capsys):
    fallback = torch.full((1, 4, 4, 3), 0.25)
    images, filename, path, matched_text, found = _load(fallback_images=fallback, **_where(studio, match_text="Zed"))
    assert images is fallback  # as it is, not repeated or copied
    assert (filename, path, matched_text, found) == ("", "", "Zed", False)
    assert "Zed" in capsys.readouterr().out  # the reason is still printed


def test_a_miss_returns_the_fallback_whatever_repeat_count_says(studio):
    fallback = torch.zeros(2, 4, 4, 3)
    images = _load(fallback_images=fallback, repeat_count=3, **_where(studio, match_text="Zed"))[0]
    assert images is fallback


def test_a_hit_does_not_use_the_fallback(studio):
    fallback = torch.full((1, 4, 4, 3), 0.25)
    images, filename, _, matched_text, found = _load(fallback_images=fallback, **_where(studio))
    assert found is True and images is not fallback and tuple(images.shape) == (1, 8, 8, 3)
    assert (filename, matched_text) == ("01.png", "Anh")


def test_a_miss_without_a_fallback_still_holds_the_images_back(studio):
    for fallback in ({}, {"fallback_images": None}):  # not connected / connected to something that gave nothing
        images, filename, path, matched_text, found = _load(**fallback, **_where(studio, match_text="Zed"))
        assert isinstance(images, ExecutionBlocker) and "Zed" in images.message
        assert (filename, path, matched_text, found) == ("", "", "Zed", False)


def test_is_changed_does_not_look_at_the_fallback_input(studio):
    # ComfyUI hands IS_CHANGED None for every input that is wired from another node.
    assert _changed(fallback_images=None, **_where(studio)) == _changed(**_where(studio))
    assert math.isnan(_changed(fallback_images=None, **_where(studio, match_text="Zed")))


def test_cache_key_with_a_fallback_wired_is_stable_on_a_hit_and_never_holds_a_miss(studio, monkeypatch):
    def prompt(**overrides):
        settings = _args(fallback_images=["1", 0], **_where(studio, **overrides))
        return {
            "1": {"class_type": "EmptyImage", "inputs": {"width": 8, "height": 8, "batch_size": 1, "color": 0}},
            "2": {"class_type": "NH_LoadImagesMatching", "inputs": settings},
        }

    hit = prompt()
    first = _cache_key(monkeypatch, hit, "2")
    assert first == _cache_key(monkeypatch, hit, "2")
    _touch(studio.root / "Nu" / "Anh" / "01.png")
    assert first != _cache_key(monkeypatch, hit, "2")
    miss = prompt(match_text="Zed")
    assert _cache_key(monkeypatch, miss, "2") != _cache_key(monkeypatch, miss, "2")


# 2g. Load Images Folder (a different node): IS_CHANGED fingerprints the folder it reads, and gives up on an unknown one

def _folder_changed(folder, **overrides):
    args = {"folder_path": folder, "index": 0, "step": 1, "seed_mode": "fixed"}
    args.update(overrides)
    return folder_loaders.NH_LoadImagesFromFolder.IS_CHANGED(**args)


def test_folder_is_changed_follows_the_selected_files_only(tmp_path):
    folder = tmp_path / "pics"
    _png(folder / "01.png")
    _png(folder / "02.png")
    before = _folder_changed(str(folder))
    assert isinstance(before, str) and before == _folder_changed(str(folder))
    _touch(folder / "02.png")  # not selected (index 0, step 1)
    assert _folder_changed(str(folder)) == before
    _touch(folder / "01.png")  # selected
    assert _folder_changed(str(folder)) != before
    assert _folder_changed(str(folder), index=1) != before  # another selection is another fingerprint


def test_folder_is_changed_with_a_wired_folder_is_a_constant_not_the_input_folder(tmp_path, monkeypatch):
    # ComfyUI passes None for an input that comes from another node, so the folder that will be read is unknown. None
    # used to resolve to ComfyUI's input folder: the node hashed that folder and never saw a change in the real one.
    inputs = tmp_path / "input"
    _png(inputs / "01.png")
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(inputs))
    input_signature = _folder_changed("")  # no text at all still means the input folder
    assert isinstance(input_signature, str)
    wired = _folder_changed(None)
    assert isinstance(wired, str) and wired == _folder_changed(None) and wired != input_signature


def test_folder_cache_key_is_stable_until_a_selected_file_or_the_wired_path_changes(tmp_path, monkeypatch):
    folder = tmp_path / "pics"
    _png(folder / "01.png")
    settings = {
        "folder_path": str(folder), "index": 0, "step": 1, "seed_mode": "fixed", "recursive": False,
        "filename_extension": True, "path_output": "file_path",
    }
    typed = {"2": {"class_type": "NH_LoadImagesFromFolder", "inputs": settings}}
    first = _cache_key(monkeypatch, typed, "2")
    assert first == _cache_key(monkeypatch, typed, "2")
    _touch(folder / "01.png")
    assert first != _cache_key(monkeypatch, typed, "2")

    def wired(path):
        return {
            "1": {"class_type": "NH_Text", "inputs": {"value": str(path)}},
            "2": {"class_type": "NH_LoadImagesFromFolder", "inputs": {**settings, "folder_path": ["1", 0]}},
        }

    # Same upstream path: cached (nodes downstream are not re-run); another path: runs again.
    assert _cache_key(monkeypatch, wired(folder), "2") == _cache_key(monkeypatch, wired(folder), "2")
    assert _cache_key(monkeypatch, wired(folder), "2") != _cache_key(monkeypatch, wired(tmp_path / "other"), "2")


# --- 3. Indexed loaders keep what they loaded ---

# ComfyUI's output cache: it holds what a loader node returned until it evicts it. The helpers below file every result
# here, the way the executor does; a test that wants an eviction empties the list.
COMFY_CACHE = []


@pytest.fixture
def loads(tmp_path, monkeypatch):
    """Counting stand-ins for the comfy.sd / comfy.utils loaders and real files for the model names.

    The fixture holds no reference to what the stand-ins return (it records ids and the arguments instead), so a test
    can tell whether the node under test is the one keeping an object alive.
    """
    calls = SimpleNamespace(diffusion=[], torch_files=[], vae=[], vae_metadata=[], checked=[], clip=[])
    monkeypatch.setattr(comfy.cli_args.args, "cache_classic", False)  # the default cache; tests of --cache-classic set it
    COMFY_CACHE.clear()

    def full_path(folder, name):
        path = tmp_path / "models" / folder / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"model " + name.encode())
        return str(path)

    class Loaded:
        """What the real loaders return (a ModelPatcher, a VAE, a CLIP) is a plain object: it can be weakly referenced."""

        def __init__(self, **attributes):
            self.__dict__.update(attributes)

    class FakeVAE:
        invalid = False

        def __init__(self, sd=None, metadata=None, **kwargs):
            calls.vae.append(sd)
            calls.vae_metadata.append(metadata)
            self.sd = sd
            self.metadata = metadata

        def throw_exception_if_invalid(self):
            calls.checked.append(id(self))
            if type(self).invalid:
                raise RuntimeError("ERROR: VAE is invalid: None")

    def load_diffusion_model(path, model_options=None):
        calls.diffusion.append((path, dict(model_options or {})))
        return Loaded(path=path)

    def load_torch_file(path, *args, return_metadata=False, **kwargs):
        calls.torch_files.append((path, return_metadata))
        state_dict = {"file": path}
        return (state_dict, {"read_from": path}) if return_metadata else state_dict

    def load_clip(**kwargs):
        calls.clip.append(kwargs)
        return Loaded(**kwargs)

    monkeypatch.setattr(loaders.folder_paths, "get_full_path_or_raise", full_path)
    monkeypatch.setattr(comfy.sd, "load_diffusion_model", load_diffusion_model)
    monkeypatch.setattr(comfy.sd, "VAE", FakeVAE)
    monkeypatch.setattr(comfy.sd, "load_clip", load_clip)
    monkeypatch.setattr(comfy.utils, "load_torch_file", load_torch_file)
    calls.vae_class = FakeVAE
    calls.path = full_path
    yield calls
    COMFY_CACHE.clear()


def _cached_by_comfyui(result):
    COMFY_CACHE.append(result)
    return result


def _diffusion(node, index=2, weight_dtype="default", select_mode="index", **slots):
    slots = {"diffusion_model_1": "a.safetensors", "diffusion_model_2": "b.safetensors",
             "diffusion_model_3": "c.safetensors", **slots}
    return _cached_by_comfyui(node.load_diffusion_model_index(
        select_mode=select_mode, index=index, model_count=3, weight_dtype=weight_dtype, **slots,
    ))


def test_diffusion_model_is_not_reloaded_when_an_unselected_slot_changes(loads):
    node = loaders.NH_DiffusionModelIndex()
    first = _diffusion(node)
    second = _diffusion(node, diffusion_model_1="other.safetensors", diffusion_model_3="")
    assert len(loads.diffusion) == 1
    assert second[0] is first[0]
    assert second[1:] == ("b.safetensors", 2)


def test_diffusion_model_reloads_for_another_file_option_or_changed_file(loads):
    node = loaders.NH_DiffusionModelIndex()
    _diffusion(node)
    _diffusion(node, index=3)  # another selection
    assert [Path(path).name for path, _ in loads.diffusion] == ["b.safetensors", "c.safetensors"]
    _diffusion(node, index=3, weight_dtype="fp8_e5m2")  # same file, other options
    assert loads.diffusion[-1][1] == {"dtype": torch.float8_e5m2}
    assert len(loads.diffusion) == 3
    _diffusion(node, index=3, weight_dtype="fp8_e5m2")
    assert len(loads.diffusion) == 3
    path = loads.path("diffusion_models", "c.safetensors")  # the file is replaced under the same name
    with open(path, "wb") as handle:
        handle.write(b"a different and longer model")
    _diffusion(node, index=3, weight_dtype="fp8_e5m2")
    assert len(loads.diffusion) == 4


def test_diffusion_model_boolean_mode_uses_the_same_cache(loads):
    node = loaders.NH_DiffusionModelIndex()
    first = _diffusion(node, select_mode="boolean", enable_2=True)
    second = _diffusion(node, select_mode="boolean", enable_2=True, diffusion_model_3="z.safetensors")
    assert second[0] is first[0] and len(loads.diffusion) == 1


def test_diffusion_model_nothing_selected_loads_nothing(loads):
    result = _diffusion(loaders.NH_DiffusionModelIndex(), select_mode="boolean")
    assert isinstance(result[0], ExecutionBlocker) and loads.diffusion == []


def test_loaded_objects_belong_to_the_node_instance(loads):
    _diffusion(loaders.NH_DiffusionModelIndex())
    _diffusion(loaders.NH_DiffusionModelIndex())  # ComfyUI keeps one instance per graph node; a new one starts empty
    assert len(loads.diffusion) == 2


def _vae(node, index=1, **slots):
    slots = {"vae_1": "a.safetensors", "vae_2": "b.safetensors", **slots}
    return _cached_by_comfyui(node.load_vae_index(select_mode="index", index=index, vae_count=2, **slots))


def test_vae_is_not_reloaded_when_an_unselected_slot_changes(loads):
    node = loaders.NH_VAEIndex()
    first = _vae(node)
    second = _vae(node, vae_2="other.safetensors")
    assert second[0] is first[0] and len(loads.vae) == 1 and len(loads.torch_files) == 1
    _vae(node, index=2)
    assert len(loads.vae) == 2


def test_vae_is_checked_like_the_core_loader_does(loads):
    vae = _vae(loaders.NH_VAEIndex())[0]
    assert loads.checked == [id(vae)]


def test_vae_is_built_with_the_metadata_of_its_file_like_the_core_loader_does(loads):
    vae = _vae(loaders.NH_VAEIndex())[0]
    path = loads.path("vae", "a.safetensors")
    assert loads.torch_files == [(path, True)]  # comfy.utils.load_torch_file(path, return_metadata=True)
    assert vae.sd == {"file": path}
    assert vae.metadata == {"read_from": path} and loads.vae_metadata == [{"read_from": path}]


def test_invalid_vae_raises_and_is_not_cached(loads):
    node = loaders.NH_VAEIndex()
    loads.vae_class.invalid = True
    with pytest.raises(RuntimeError, match="invalid"):
        _vae(node)
    loads.vae_class.invalid = False
    _vae(node)
    assert len(loads.vae) == 2  # the failed one was not kept


def test_vae_without_the_check_method_still_loads(loads, monkeypatch):
    class OldVAE:
        def __init__(self, sd=None, **kwargs):
            self.sd = sd

    monkeypatch.setattr(comfy.sd, "VAE", OldVAE)
    assert isinstance(_vae(loaders.NH_VAEIndex())[0], OldVAE)


def _clip(node, type="flux2", index=1, **slots):
    slots = {"clip_1": "a.safetensors", "clip_2": "b.safetensors", **slots}
    return _cached_by_comfyui(node.load_clip_index(select_mode="index", type=type, index=index, clip_count=2, **slots))


def test_clip_is_not_reloaded_when_an_unselected_slot_changes(loads):
    node = loaders.NH_ClipIndex()
    first = _clip(node)
    second = _clip(node, clip_2="other.safetensors")
    assert second[0] is first[0] and len(loads.clip) == 1
    assert loads.clip[0]["clip_type"] is comfy.sd.CLIPType.FLUX2


def test_clip_reloads_for_another_type_or_file(loads):
    node = loaders.NH_ClipIndex()
    _clip(node)
    _clip(node, type="sd3")
    assert [call["clip_type"] for call in loads.clip] == [comfy.sd.CLIPType.FLUX2, comfy.sd.CLIPType.SD3]
    _clip(node, type="sd3", index=2)
    assert len(loads.clip) == 3


# 3b. The loaded object is held weakly: ComfyUI's cache decides how long it lives

LOADERS = {
    # kind: (node class, run the node, the stand-in loader's calls, a selection of another file)
    "diffusion": (loaders.NH_DiffusionModelIndex, _diffusion, lambda loads: loads.diffusion, {"index": 3}),
    "vae": (loaders.NH_VAEIndex, _vae, lambda loads: loads.vae, {"index": 2}),
    "clip": (loaders.NH_ClipIndex, _clip, lambda loads: loads.clip, {"index": 2}),
}


@pytest.mark.parametrize("kind", LOADERS)
def test_a_loader_reuses_the_object_while_comfyui_holds_it_and_lets_it_go_after(loads, kind):
    node_class, run, calls, _ = LOADERS[kind]
    node = node_class()
    first = run(node)[0]
    probe = weakref.ref(first)
    assert run(node)[0] is first and len(calls(loads)) == 1  # ComfyUI's cache still holds it: no second read from disk
    del first
    COMFY_CACHE.clear()  # the RAM-pressure cache evicts the entry
    gc.collect()
    assert probe() is None  # nothing of the node's keeps a big model alive
    assert run(node)[0] is not None and len(calls(loads)) == 2  # asked for again: loaded again, never a dead object


@pytest.mark.parametrize("kind", LOADERS)
def test_with_cache_classic_the_loader_keeps_the_object_itself(loads, monkeypatch, kind):
    monkeypatch.setattr(comfy.cli_args.args, "cache_classic", True)  # --high-ram switches it on as well
    node_class, run, calls, _ = LOADERS[kind]
    node = node_class()
    probe = weakref.ref(run(node)[0])
    COMFY_CACHE.clear()  # the classic cache drops an output as soon as the prompt stops using it
    gc.collect()
    assert probe() is not None  # the node still has it ...
    assert run(node)[0] is probe() and len(calls(loads)) == 1  # ... and the next run reuses it


@pytest.mark.parametrize("kind", LOADERS)
def test_with_cache_classic_choosing_another_file_drops_the_old_object(loads, monkeypatch, kind):
    monkeypatch.setattr(comfy.cli_args.args, "cache_classic", True)
    node_class, run, calls, other_file = LOADERS[kind]
    node = node_class()
    probe = weakref.ref(run(node)[0])
    run(node, **other_file)
    COMFY_CACHE.clear()
    gc.collect()
    assert probe() is None and len(calls(loads)) == 2  # two big models never sit in memory together


def test_an_object_that_cannot_be_weakly_referenced_is_loaded_again_each_time(loads, monkeypatch):
    class Slotted:
        __slots__ = ()  # no __weakref__ slot

    def load_slotted(path, model_options=None):
        loads.diffusion.append((path, {}))
        return Slotted()

    monkeypatch.setattr(comfy.sd, "load_diffusion_model", load_slotted)
    node = loaders.NH_DiffusionModelIndex()
    first, second = _diffusion(node)[0], _diffusion(node)[0]
    assert first is not second and len(loads.diffusion) == 2  # no error, and nothing remembered

    monkeypatch.setattr(comfy.cli_args.args, "cache_classic", True)  # a strong reference works for any object
    third = _diffusion(node)[0]
    assert _diffusion(node)[0] is third and len(loads.diffusion) == 3


def test_with_cache_classic_the_old_object_is_gone_before_the_next_one_loads(loads, monkeypatch):
    monkeypatch.setattr(comfy.cli_args.args, "cache_classic", True)
    node = loaders.NH_DiffusionModelIndex()
    probe = weakref.ref(_diffusion(node)[0])
    COMFY_CACHE.clear()
    real_load = comfy.sd.load_diffusion_model
    old_model_alive = []

    def load(path, model_options=None):
        old_model_alive.append(probe() is not None)  # is the previous model still in memory while this one loads?
        return real_load(path, model_options)

    monkeypatch.setattr(comfy.sd, "load_diffusion_model", load)
    _diffusion(node, index=3)
    assert old_model_alive == [False]


# --- 4. Large image viewer: step through the batch ---

JS_IMPORTS = {
    'import { app } from "../../../scripts/app.js";': "const app = globalThis.__app;",
    'import { api } from "../../../scripts/api.js";': "const api = globalThis.__api;",
}

JS_HARNESS = r"""
import { pathToFileURL } from "node:url";

const extensionPath = process.argv[2];

// ---- a very small DOM: enough for the viewer to build, draw into, and tear down -----------------------------
const created = [];
const frames = new Map();
let nextFrame = 1;
const requested = [];
const opened = [];
const alerts = [];
const store = {};
const context = new Proxy({}, {
  get: (_target, name) => (name in store ? store[name] : () => {}),
  set: (_target, name, value) => { store[name] = value; return true; },
});

class Listeners {
  constructor() { this.map = new Map(); }
  add(type, fn) { if (!this.map.has(type)) this.map.set(type, []); this.map.get(type).push(fn); }
  remove(type, fn) { this.map.set(type, (this.map.get(type) || []).filter((item) => item !== fn)); }
  list(type) { return this.map.get(type) || []; }
  total() { let n = 0; for (const list of this.map.values()) n += list.length; return n; }
}

class Element {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.parentNode = null;
    this.style = {};
    this.listeners = new Listeners();
    this.text = "";
    this.disabled = false;
    this.clientWidth = 800;
    this.clientHeight = 600;
    this.title = "";
    this.value = "";
  }
  get textContent() { return this.text; }
  set textContent(value) { this.text = String(value); }
  appendChild(child) { child.parentNode = this; this.children.push(child); return child; }
  remove() {
    if (this.parentNode) this.parentNode.children = this.parentNode.children.filter((item) => item !== this);
    this.parentNode = null;
  }
  contains(node) { for (let n = node; n; n = n.parentNode) if (n === this) return true; return false; }
  addEventListener(type, fn) { this.listeners.add(type, fn); }
  removeEventListener(type, fn) { this.listeners.remove(type, fn); }
  getContext() { return context; }
  getBoundingClientRect() { return { left: 0, top: 0, width: this.clientWidth, height: this.clientHeight }; }
  focus() { document.activeElement = this; }
  blur() { if (document.activeElement === this) document.activeElement = null; }
  click() { if (!this.disabled && this.onclick) this.onclick({ target: this }); }
}

const windowListeners = new Listeners();
const documentListeners = new Listeners();
const document = {
  body: new Element("body"),
  activeElement: null,
  createElement(tag) { const element = new Element(tag); created.push(element); return element; },
  addEventListener: (type, fn) => documentListeners.add(type, fn),
  removeEventListener: (type, fn) => documentListeners.remove(type, fn),
};
globalThis.document = document;
globalThis.window = {
  devicePixelRatio: 1,
  addEventListener: (type, fn) => windowListeners.add(type, fn),
  removeEventListener: (type, fn) => windowListeners.remove(type, fn),
  open: (url) => { opened.push(url); },
  // what the browser does when the mouse button is released anywhere on the page
  releaseMouse: () => windowListeners.list("mouseup").slice().forEach((fn) => fn({})),
};
globalThis.alert = (message) => { alerts.push(message); };
globalThis.requestAnimationFrame = (fn) => { const id = nextFrame++; frames.set(id, fn); return id; };
globalThis.cancelAnimationFrame = (id) => { frames.delete(id); };
globalThis.Image = class {
  set src(value) { this._src = value; if (value) requested.push(value); }
  get src() { return this._src; }
};

const registered = [];
globalThis.__app = { registerExtension(extension) { registered.push(extension); } };
globalThis.__api = { apiURL: (path) => path };
await import(pathToFileURL(extensionPath).href);
const extension = registered[0];

// ---- helpers ------------------------------------------------------------------------------------------------
function flush() {
  for (let round = 0; round < 5 && frames.size; round++) {
    const pending = [...frames.values()];
    frames.clear();
    pending.forEach((fn) => fn());
  }
}

function level(index, scale, columns, rows, width, height) {
  return { level: index, scale, columns, rows, width, height, tile_size: 512 };
}

function manifest(id, width, height) {
  return {
    cache_id: id, title: "T", width, height, tile_size: 512,
    thumb_url: `/thumb/${id}`, tile_url_template: `/tile/${id}/{level}/{x}_{y}.jpg`,
    levels: [level(0, 0.25, 2, 2, width / 4, height / 4), level(1, 1, 8, 6, width, height)],
  };
}

function pair(i) { return { title: "C", compare_mode: "slider", a: manifest(`a${i}`, 3000 + 100 * i, 2000), b: manifest(`b${i}`, 3000, 2000 + 100 * i) }; }

function makeNode(nodeName, executedKey, items) {
  const NodeType = function () {};
  const node = { widgets: [], addWidget(type, name, value, callback) { const widget = { type, name, value, callback }; this.widgets.push(widget); return widget; } };
  return extension.beforeRegisterNodeDef(NodeType, { name: nodeName }).then(() => {
    NodeType.prototype.onNodeCreated.call(node);
    NodeType.prototype.onExecuted.call(node, { [executedKey]: items });
    return node;
  });
}

function find(text) { return created.filter((item) => item.text === text && document.body.contains(item)); }
function labelText() { const label = created.filter((item) => /^\d+ \/ \d+$/.test(item.text) && document.body.contains(item)); return label.length ? label[label.length - 1].text : null; }
function statusText() { const status = created.filter((item) => /\| tiles /.test(item.text) && document.body.contains(item)); return status.length ? status[status.length - 1].text : null; }
function zoomPercent() { const m = /(\d+)%/.exec(statusText() || ""); return m ? Number(m[1]) : null; }
function titleText() { const t = created.filter((item) => /^(T|C) \|/.test(item.text) && document.body.contains(item)); return t.length ? t[t.length - 1].text : null; }
function press(key, extra = {}) {
  const event = { key, target: document.activeElement || document.body, altKey: false, ctrlKey: false, metaKey: false, shiftKey: false, prevented: 0, stopped: 0, ...extra };
  event.preventDefault = () => { event.prevented++; };
  event.stopPropagation = () => { event.stopped++; };
  documentListeners.list("keydown").slice().forEach((fn) => fn(event));
  flush();
  return event;
}
function counts() { return { document: documentListeners.total(), window: windowListeners.total(), overlays: document.body.children.length }; }
function tilesFor(prefix) { return requested.filter((url) => url.includes(prefix)).length; }

async function openViewer(nodeName, executedKey, items) {
  const node = await makeNode(nodeName, executedKey, items);
  node.widgets[0].callback();
  flush();
  return node;
}

const out = {};

// ---- preview viewer, 3 images -------------------------------------------------------------------------------
{
  await openViewer("NH_LargeImagePreview", "nh_large_preview", [manifest("p0", 4000, 3000), manifest("p1", 5000, 3000), manifest("p2", 6000, 3000)]);
  const s = {};
  s.label0 = labelText();
  s.prevDisabled0 = find("\u25C0")[0].disabled;
  s.nextDisabled0 = find("\u25B6")[0].disabled;
  s.title0 = titleText();
  s.tiles0 = [tilesFor("/p0/"), tilesFor("/p1/")];
  s.fit0 = zoomPercent();

  const right = press("ArrowRight");
  s.label1 = labelText();
  s.title1 = titleText();
  s.consumed = [right.prevented, right.stopped];
  s.tiles1 = [tilesFor("/p0/"), tilesFor("/p1/")];
  s.fit1 = zoomPercent();

  find("+")[0].click(); flush();
  s.zoomedIn = zoomPercent();
  press("ArrowRight"); flush();
  s.label2 = labelText();
  s.zoomAfterNavigation = zoomPercent();
  find("Fit")[0].click(); flush();
  s.fitOfThird = zoomPercent();
  s.nextDisabledAtEnd = find("\u25B6")[0].disabled;
  s.prevDisabledAtEnd = find("\u25C0")[0].disabled;

  const beyond = press("ArrowRight");
  s.labelBeyond = labelText();
  s.beyondConsumed = beyond.prevented;

  press("ArrowLeft");
  s.labelBack = labelText();
  find("\u25C0")[0].click(); flush();
  s.labelFirst = labelText();
  find("\u25C0")[0].click(); flush();   // disabled at the start: nothing happens
  s.labelStill = labelText();
  find("\u25B6")[0].click(); flush();
  s.labelButton = labelText();

  // keys that must not flip images
  const ignored = [];
  for (const extra of [{ ctrlKey: true }, { altKey: true }, { metaKey: true }, { shiftKey: true }, { target: { tagName: "INPUT" } }, { target: { tagName: "TEXTAREA" } }]) {
    const event = press("ArrowRight", extra);
    ignored.push([labelText(), event.prevented]);
  }
  s.ignored = ignored;
  press("ArrowUp");
  s.labelAfterOtherKey = labelText();

  s.listenersOpen = counts();
  press("Escape");
  s.listenersClosed = counts();
  press("ArrowRight");   // nothing is listening any more, nothing may throw
  s.afterClose = labelText();
  out.preview = s;
}

// ---- Close button / overlay click also clean up ----------------------------------------------------------------
{
  await openViewer("NH_LargeImagePreview", "nh_large_preview", [manifest("q0", 4000, 3000), manifest("q1", 4000, 3000)]);
  find("Close")[0].click();
  out.closeButton = counts();
}

// ---- a single image: nothing to flip --------------------------------------------------------------------------
{
  await openViewer("NH_LargeImagePreview", "nh_large_preview", [manifest("s0", 4000, 3000)]);
  const s = {};
  s.nav = [find("\u25C0").length, find("\u25B6").length, labelText()];
  const event = press("ArrowRight");
  s.prevented = event.prevented;
  press("Escape");
  s.closed = counts();
  out.single = s;
}

// ---- compare viewer, 3 pairs ------------------------------------------------------------------------------------
{
  await openViewer("NH_LargeImageCompare", "nh_large_compare", [pair(0), pair(1), pair(2)]);
  const s = {};
  s.label0 = labelText();
  s.tiles0 = [tilesFor("/a0/"), tilesFor("/b0/"), tilesFor("/a1/")];
  s.title0 = titleText();
  find("Side")[0].click(); flush();
  s.mode0 = statusText().split(" | ")[0];
  find("+")[0].click(); flush();
  s.zoomedIn = zoomPercent();
  press("ArrowRight");
  s.label1 = labelText();
  s.title1 = titleText();
  s.tiles1 = [tilesFor("/a1/"), tilesFor("/b1/")];
  s.mode1 = statusText().split(" | ")[0];
  s.zoomAfterNavigation = zoomPercent();
  find("Fit")[0].click(); flush();
  s.fit = zoomPercent();
  find("\u25B6")[0].click(); flush();
  s.label2 = labelText();
  s.nextDisabledAtEnd = find("\u25B6")[0].disabled;
  press("ArrowRight");
  s.labelBeyond = labelText();
  press("ArrowLeft"); press("ArrowLeft"); press("ArrowLeft");
  s.labelStart = labelText();
  s.prevDisabledAtStart = find("\u25C0")[0].disabled;
  // the split slider keeps the arrow keys while it has the focus
  const slider = created.filter((item) => item.tagName === "INPUT" && document.body.contains(item))[0];
  slider.focus();
  press("ArrowRight");
  s.labelWithSliderFocused = labelText();
  window.releaseMouse();
  s.sliderBlurred = document.activeElement !== slider;
  press("ArrowRight");
  s.labelAfterBlur = labelText();
  s.listenersOpen = counts();
  press("Escape");
  s.listenersClosed = counts();
  out.compare = s;
}

// ---- no tile data: falls back to the thumbnail tab, like before ----------------------------------------------------
{
  const bare = { cache_id: "n0", width: 10, height: 10, thumb_url: "/thumb/n0" };
  const before = counts();
  await openViewer("NH_LargeImagePreview", "nh_large_preview", [bare, bare]);
  out.fallback = { opened, overlays: counts().overlays - before.overlays };
}

console.log("RESULT:" + JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def viewer_run(tmp_path_factory):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    source = (PACK_DIR / "web" / "nh_large_image_preview.js").read_text(encoding="utf-8")
    for original, replacement in JS_IMPORTS.items():
        assert original in source, f"the viewer no longer has the import this test replaces: {original}"
        source = source.replace(original, replacement)
    work = tmp_path_factory.mktemp("large_preview_js")
    (work / "viewer.mjs").write_text(source, encoding="utf-8")
    (work / "harness.mjs").write_text(JS_HARNESS, encoding="utf-8")
    run = subprocess.run(
        [node, str(work / "harness.mjs"), str(work / "viewer.mjs")],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert run.returncode == 0, run.stderr
    line = next(line for line in run.stdout.splitlines() if line.startswith("RESULT:"))
    return json.loads(line[len("RESULT:"):])


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_viewer_js_is_valid_syntax():
    check = subprocess.run(
        [shutil.which("node"), "--check", str(PACK_DIR / "web" / "nh_large_image_preview.js")],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert check.returncode == 0, check.stderr


def test_preview_viewer_starts_on_the_first_image_with_a_counter(viewer_run):
    s = viewer_run["preview"]
    assert s["label0"] == "1 / 3"
    assert (s["prevDisabled0"], s["nextDisabled0"]) == (True, False)
    assert "4000x3000" in s["title0"]
    assert s["tiles0"][0] > 0 and s["tiles0"][1] == 0


def test_preview_viewer_right_arrow_shows_the_next_image(viewer_run):
    s = viewer_run["preview"]
    assert s["label1"] == "2 / 3"
    assert "5000x3000" in s["title1"]
    assert s["tiles1"][1] > 0  # tiles of the second image are requested
    assert s["consumed"] == [1, 1]  # the key is not also passed on to the ComfyUI canvas behind


def test_preview_viewer_zoom_is_reset_for_every_image(viewer_run):
    s = viewer_run["preview"]
    assert s["zoomedIn"] != s["fit1"]  # the zoom button did something ...
    assert s["label2"] == "3 / 3"
    assert s["zoomAfterNavigation"] == s["fitOfThird"] != s["zoomedIn"]  # ... and the next image starts fitted


def test_preview_viewer_stops_at_both_ends_and_buttons_follow(viewer_run):
    s = viewer_run["preview"]
    assert (s["prevDisabledAtEnd"], s["nextDisabledAtEnd"]) == (False, True)
    assert s["labelBeyond"] == "3 / 3" and s["beyondConsumed"] == 1
    assert s["labelBack"] == "2 / 3"
    assert s["labelFirst"] == "1 / 3"
    assert s["labelStill"] == "1 / 3"
    assert s["labelButton"] == "2 / 3"


def test_preview_viewer_ignores_modified_keys_and_text_fields(viewer_run):
    s = viewer_run["preview"]
    assert all(label == "2 / 3" and prevented == 0 for label, prevented in s["ignored"])
    assert s["labelAfterOtherKey"] == "2 / 3"


def test_preview_viewer_removes_its_listeners_when_it_closes(viewer_run):
    s = viewer_run["preview"]
    assert s["listenersOpen"]["document"] == 1 and s["listenersOpen"]["window"] >= 3 and s["listenersOpen"]["overlays"] == 1
    assert s["listenersClosed"] == {"document": 0, "window": 0, "overlays": 0}
    assert viewer_run["closeButton"] == {"document": 0, "window": 0, "overlays": 0}


def test_single_image_viewer_has_no_navigation(viewer_run):
    s = viewer_run["single"]
    assert s["nav"] == [0, 0, None]
    assert s["prevented"] == 0
    assert s["closed"] == {"document": 0, "window": 0, "overlays": 0}


def test_compare_viewer_steps_through_the_pairs(viewer_run):
    s = viewer_run["compare"]
    assert s["label0"] == "1 / 3"
    assert s["tiles0"][0] > 0 and s["tiles0"][1] > 0 and s["tiles0"][2] == 0
    assert s["label1"] == "2 / 3"
    assert "A 3100x2000" in s["title1"] and "B 3000x2100" in s["title1"]
    assert s["tiles1"][0] > 0 and s["tiles1"][1] > 0  # both sides of the second pair
    assert s["label2"] == "3 / 3" and s["nextDisabledAtEnd"] is True and s["labelBeyond"] == "3 / 3"
    assert s["labelStart"] == "1 / 3" and s["prevDisabledAtStart"] is True


def test_compare_viewer_keeps_the_mode_and_resets_the_zoom(viewer_run):
    s = viewer_run["compare"]
    assert s["mode0"] == "side_by_side" and s["mode1"] == "side_by_side"
    assert s["zoomedIn"] != s["fit"]
    assert s["zoomAfterNavigation"] == s["fit"]


def test_compare_viewer_leaves_the_arrow_keys_to_a_focused_slider(viewer_run):
    s = viewer_run["compare"]
    assert s["labelWithSliderFocused"] == "1 / 3"
    assert s["sliderBlurred"] is True
    assert s["labelAfterBlur"] == "2 / 3"


def test_compare_viewer_removes_its_listeners_when_it_closes(viewer_run):
    s = viewer_run["compare"]
    assert s["listenersOpen"]["document"] == 1 and s["listenersOpen"]["overlays"] == 1
    assert s["listenersClosed"] == {"document": 0, "window": 0, "overlays": 0}


def test_manifests_without_tiles_still_open_the_thumbnail(viewer_run):
    s = viewer_run["fallback"]
    assert len(s["opened"]) == 1 and s["opened"][0].startswith("/thumb/n0?") and s["overlays"] == 0
