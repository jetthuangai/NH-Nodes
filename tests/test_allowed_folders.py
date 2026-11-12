"""The image nodes and the folder-count route only read and write inside ComfyUI's folders and the declared ones.

core/allowed_folders.py decides; Save Image Path, Load Images Folder, Load images matching and
GET /nh-nodes/load-images-folder/count ask it before they touch the disk.
"""

import asyncio
import importlib
import json
import os

import pytest
import torch
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

import folder_paths

allowed_folders = importlib.import_module("NH-Nodes.core.allowed_folders")
image_folders = importlib.import_module("NH-Nodes.core.image_folders")
folder_routes = importlib.import_module("NH-Nodes.core.folder_routes")
folder_loaders = importlib.import_module("NH-Nodes.nodes.image.folder_loaders")
save_image_path = importlib.import_module("NH-Nodes.nodes.image.save_image_path")

FolderNotAllowed = allowed_folders.FolderNotAllowed


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A ComfyUI with input/output/temp folders, one declared folder (photos) and one that is not (private)."""
    for name in ("input", "output", "temp", "photos", "private"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path / "input"))
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path / "output"))
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path / "temp"))
    monkeypatch.setattr(allowed_folders, "_declared", [str(tmp_path / "photos")])
    return tmp_path


def _png(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (10, 20, 30)).save(path)
    return path


def _link_or_skip(link, target):
    """A directory link ``link`` -> ``target``: a symlink, or a junction where Windows refuses symlinks."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except OSError:
        pass
    if os.name != "nt":
        pytest.skip("cannot create a directory link here")
    import subprocess

    done = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
    if done.returncode != 0:
        pytest.skip("cannot create a directory link here")


# --- the rule ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["input", "output", "temp", "photos"])
def test_comfyui_folders_and_declared_folders_are_readable(machine, name):
    assert allowed_folders.can_read(str(machine / name))
    assert allowed_folders.can_read(str(machine / name / "sub" / "deeper"))  # need not exist


@pytest.mark.parametrize("name, writable", [("output", True), ("photos", True), ("input", False), ("temp", False)])
def test_only_output_and_declared_folders_are_writable(machine, name, writable):
    target = str(machine / name / "sub")
    if writable:
        assert allowed_folders.require_writable(target) == os.path.abspath(target)
    else:
        with pytest.raises(FolderNotAllowed):
            allowed_folders.require_writable(target)


@pytest.mark.parametrize("relative", ["private", "output/../private", "photos/../../x", "outputs", "photos2"])
def test_other_folders_are_refused(machine, relative):
    path = os.path.join(str(machine), *relative.split("/"))
    assert not allowed_folders.can_read(path)
    with pytest.raises(FolderNotAllowed):
        allowed_folders.require_readable(path)
    with pytest.raises(FolderNotAllowed):
        allowed_folders.require_writable(path)


def test_a_refusal_is_a_permission_error_that_says_how_to_allow_the_folder(machine):
    with pytest.raises(PermissionError) as error:
        allowed_folders.require_writable(str(machine / "private"), "folder_path")
    message = str(error.value)
    assert "folder_path" in message and "nh_image_folders" in message and "extra_model_paths.yaml" in message
    assert "restart ComfyUI" in message


@pytest.mark.skipif(os.name != "nt", reason="Windows paths")
def test_windows_spellings_of_an_allowed_folder_are_the_same_folder(machine):
    assert allowed_folders.can_read(str(machine / "PHOTOS" / "a").upper())
    assert allowed_folders.can_read(str(machine / "photos").replace("\\", "/") + "/a")


@pytest.mark.skipif(os.name != "nt", reason="Windows paths")
@pytest.mark.parametrize("path", ["\\\\server\\share\\x", "//server/share/x", "\\\\?\\UNC\\server\\share\\x"])
def test_a_network_path_is_refused_without_resolving_it(machine, monkeypatch, path):
    # Resolving a \\server\share path makes Windows contact that server, so a refused path is never resolved.
    # (The allowed folders themselves are resolved: they are the owner's.)
    real_realpath = os.path.realpath

    def no_network_resolving(value, *args, **kwargs):
        if "server" in str(value).lower():
            raise AssertionError(f"resolved {value!r}")
        return real_realpath(value, *args, **kwargs)

    monkeypatch.setattr(os.path, "realpath", no_network_resolving)
    assert not allowed_folders.can_read(path)


def test_another_drive_is_refused(machine):
    other = "Q:\\x" if os.name == "nt" else "/nh-nodes-test-other-root/x"
    assert not allowed_folders.can_read(other)


def test_a_link_inside_an_allowed_folder_cannot_lead_out_of_it(machine):
    _link_or_skip(machine / "photos" / "way_out", machine / "private")
    assert not allowed_folders.can_read(str(machine / "photos" / "way_out"))
    assert not allowed_folders.can_read(str(machine / "photos" / "way_out" / "deeper"))
    with pytest.raises(FolderNotAllowed):
        allowed_folders.require_writable(str(machine / "photos" / "way_out"))


def test_an_allowed_folder_reached_through_a_link_still_works(machine, monkeypatch):
    # ComfyUI's own folders are often links (a models or output folder on another disk).
    _link_or_skip(machine / "linked_output", machine / "output")
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(machine / "linked_output"))
    assert allowed_folders.require_writable(str(machine / "linked_output" / "day1"))


def test_nothing_is_declared_by_default(machine, monkeypatch):
    monkeypatch.setattr(allowed_folders, "_declared", [])
    assert allowed_folders.declared_folders() == []
    assert not allowed_folders.can_read(str(machine / "photos"))
    assert allowed_folders.can_read(str(machine / "input"))


def test_folders_of_extra_model_paths_are_taken_and_not_left_as_a_model_folder(machine, monkeypatch):
    # ComfyUI registers every key of extra_model_paths.yaml as a model folder and lists such folders at /models.
    registry = dict(folder_paths.folder_names_and_paths)
    registry[allowed_folders.FOLDER_KEY] = ([str(machine / "private"), str(machine / "photos")], set())
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", registry)
    monkeypatch.setattr(allowed_folders, "_declared", [])

    assert allowed_folders.declared_folders() == [str(machine / "private"), str(machine / "photos")]
    assert allowed_folders.FOLDER_KEY not in folder_paths.folder_names_and_paths
    assert allowed_folders.can_read(str(machine / "private"))
    assert allowed_folders.declared_folders() == [str(machine / "private"), str(machine / "photos")]  # kept


# --- Save Image Path ---------------------------------------------------------------------------------------------

def _save(folder_path, **kwargs):
    settings = dict(filename_prefix="img", create_new_folder=False, new_folder_name="")
    settings.update(kwargs)
    return save_image_path.NH_SaveImagePath().save_images(torch.rand((1, 4, 4, 3)), folder_path=folder_path, **settings)


def test_save_refuses_a_folder_that_is_not_allowed_and_creates_nothing(machine):
    target = machine / "private" / "new" / "deeper"
    with pytest.raises(FolderNotAllowed, match="folder_path"):
        _save(str(target))
    assert not (machine / "private" / "new").exists()
    assert list(machine.rglob("*.png")) == []


@pytest.mark.parametrize("folder_path", [
    "../private",
    "a/../../private",
    # A backslash separates folders on Windows only; elsewhere this is one odd folder name inside output.
    pytest.param("..\\private", marks=pytest.mark.skipif(os.name != "nt", reason="Windows separator")),
])
def test_save_refuses_a_relative_path_that_leaves_the_output_folder(machine, folder_path):
    with pytest.raises(FolderNotAllowed):
        _save(folder_path)
    assert list(machine.rglob("*.png")) == []


def test_save_refuses_the_input_folder(machine):
    with pytest.raises(FolderNotAllowed):
        _save(str(machine / "input"))


def test_save_works_in_the_output_folder_and_in_a_declared_folder(machine):
    assert _save("")[1] == str(machine / "output")
    assert _save("day1")[1] == str(machine / "output" / "day1")
    assert _save(str(machine / "photos" / "exports"))[1] == str(machine / "photos" / "exports")
    saved = sorted(str(path.relative_to(machine)).replace("\\", "/") for path in machine.rglob("*.png"))
    assert saved == ["output/day1/img0001.png", "output/img0001.png", "photos/exports/img0001.png"]


def test_save_next_to_an_existing_file_of_an_allowed_folder(machine):
    source = _png(machine / "photos" / "shoot" / "a.png")
    assert _save(str(source))[1] == str(machine / "photos" / "shoot")


def test_save_never_looks_at_a_file_outside_the_allowed_folders(machine, monkeypatch):
    secret = _png(machine / "private" / "a.png")
    looked = []
    real_isfile = os.path.isfile
    monkeypatch.setattr(os.path, "isfile", lambda path: looked.append(str(path)) or real_isfile(path))
    with pytest.raises(FolderNotAllowed):
        _save(str(secret))
    assert looked == []


# --- Load Images Folder ------------------------------------------------------------------------------------------

def _load_folder(folder_path, **kwargs):
    settings = dict(index=0, step=1, seed_mode="fixed")
    settings.update(kwargs)
    return folder_loaders.NH_LoadImagesFromFolder().load_images(folder_path, **settings)


def test_load_folder_refuses_a_folder_that_is_not_allowed(machine, monkeypatch):
    _png(machine / "private" / "a.png")
    listed = []
    real_listdir = os.listdir
    monkeypatch.setattr(os, "listdir", lambda path: listed.append(str(path)) or real_listdir(path))
    with pytest.raises(FolderNotAllowed, match="folder_path"):
        _load_folder(str(machine / "private"))
    with pytest.raises(FolderNotAllowed):
        _load_folder("../private")
    assert listed == []


def test_load_folder_change_check_does_not_raise_for_a_refused_folder(machine):
    _png(machine / "private" / "a.png")
    result = folder_loaders.NH_LoadImagesFromFolder.IS_CHANGED(str(machine / "private"), 0, 1, "fixed")
    assert isinstance(result, str) and result.startswith("error:FolderNotAllowed")


@pytest.mark.parametrize("name", ["input", "output", "temp", "photos"])
def test_load_folder_reads_comfyui_folders_and_declared_folders(machine, name):
    _png(machine / name / "set" / "a.png")
    images, filename, path = _load_folder(str(machine / name / "set"))
    assert images.shape == (1, 4, 4, 3) and filename == "a.png"
    assert path == str(machine / name / "set" / "a.png")


def test_load_folder_relative_path_stays_below_the_input_folder(machine):
    _png(machine / "input" / "set" / "a.png")
    assert _load_folder("set")[1] == "a.png"


# --- Load images matching ----------------------------------------------------------------------------------------

def _matching_args(**overrides):
    settings = dict(
        source_folder_path="", matching_root_path="", match_text="", match_mode="contains", recursive=True,
        index=0, step=1, repeat_count=1, context_folder_names="Nam\nNu", source_file_names="",
    )
    settings.update(overrides)
    return settings


def test_matching_refuses_a_root_that_is_not_allowed(machine):
    _png(machine / "private" / "Anh" / "a.png")
    node = folder_loaders.NH_LoadImagesMatching()
    args = _matching_args(matching_root_path=str(machine / "private"), match_text="Anh")
    with pytest.raises(FolderNotAllowed, match="matching_root_path"):
        node.load_matching(**args)
    assert folder_loaders.NH_LoadImagesMatching.IS_CHANGED(**args) != folder_loaders.NH_LoadImagesMatching.IS_CHANGED(**args)  # NaN
    assert node.check_lazy_status(fallback_images=None, **args) == []


def test_matching_reads_a_declared_root_and_the_source_is_names_only(machine, monkeypatch):
    _png(machine / "photos" / "library" / "Anh" / "a.png")
    probed = []
    real_isfile = os.path.isfile
    monkeypatch.setattr(os.path, "isfile", lambda path: probed.append(str(path)) or real_isfile(path))
    source = machine / "private" / "shoot" / "Anh"  # not allowed, and never touched: only its names count
    _, filename, path, matched_text, found = folder_loaders.NH_LoadImagesMatching().load_matching(
        **_matching_args(source_folder_path=str(source), matching_root_path=str(machine / "photos" / "library"))
    )
    assert found and filename == "a.png" and matched_text == "Anh"
    assert path == str(machine / "photos" / "library" / "Anh")
    assert [item for item in probed if "private" in item] == []


def test_matching_takes_the_folder_of_a_source_image_path_from_the_text(machine):
    _png(machine / "photos" / "library" / "Anh" / "a.png")
    source_file = machine / "private" / "shoot" / "Anh" / "portrait.JPG"  # does not exist
    _, _, _, matched_text, found = folder_loaders.NH_LoadImagesMatching().load_matching(
        **_matching_args(source_folder_path=str(source_file), matching_root_path=str(machine / "photos" / "library"))
    )
    assert found and matched_text == "Anh"


def test_matching_does_not_list_a_parent_that_is_not_allowed_for_twins(machine, monkeypatch):
    # The root sits in a context folder (Nam) whose parent is not allowed: the sibling Nu is not looked for.
    monkeypatch.setattr(allowed_folders, "_declared", [str(machine / "private" / "Nam")])
    _png(machine / "private" / "Nam" / "Anh" / "a.png")
    _png(machine / "private" / "Nu" / "Anh" / "b.png")
    scanned = []
    real_scandir = os.scandir
    monkeypatch.setattr(os, "scandir", lambda path: scanned.append(os.path.normcase(str(path))) or real_scandir(path))
    _, filename, _, _, found = folder_loaders.NH_LoadImagesMatching().load_matching(
        **_matching_args(source_folder_path="x/Nu/Anh", matching_root_path=str(machine / "private" / "Nam"))
    )
    assert found and filename == "a.png"
    assert os.path.normcase(str(machine / "private")) not in scanned
    assert not any(os.path.normcase(str(machine / "private" / "Nu")) in item for item in scanned)


# --- the count route ---------------------------------------------------------------------------------------------

def _count(**params):
    async def run():
        routes = web.RouteTableDef()
        folder_routes.register_routes(routes)
        app = web.Application()
        app.add_routes(routes)
        async with TestClient(TestServer(app)) as client:
            response = await client.get(folder_routes.COUNT_PATH, params=params)
            return response.status, await response.read()

    status, body = asyncio.run(run())
    return status, json.loads(body), body.decode("utf-8")


def test_count_route_refuses_a_folder_that_is_not_allowed(machine):
    _png(machine / "private" / "a.png")
    for folder_path in (str(machine / "private"), "../private", str(machine / "private" / "missing")):
        status, body, raw = _count(folder_path=folder_path)
        assert (status, body) == (403, {"error": "folder is not allowed"})
        assert "private" not in raw and str(machine) not in raw


def test_count_route_counts_in_allowed_folders(machine):
    _png(machine / "photos" / "a.png")
    _png(machine / "input" / "b.png")
    assert _count(folder_path=str(machine / "photos"))[:2] == (200, {"count": 1})
    assert _count(folder_path="")[:2] == (200, {"count": 1})


# --- nothing outside the pack is touched at import ----------------------------------------------------------------

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_image_label_reads_the_font_folder_only_when_the_node_list_is_built(monkeypatch):
    import importlib.util
    from pathlib import Path

    def no_listing(self):
        raise AssertionError(f"listed {self} at import")

    # A second copy of the module, so the registered node class is left alone.
    spec = importlib.util.spec_from_file_location(
        "NH-Nodes.nodes.image.image_label_import_probe", os.path.join(PACK_DIR, "nodes", "image", "image_label.py")
    )
    probe = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "iterdir", no_listing)
        spec.loader.exec_module(probe)  # importing the module lists no folder
    fonts = probe.NH_ImageLabel.INPUT_TYPES()["required"]["font_name"][0]
    assert fonts[0] == "default" and len(fonts) == len(set(fonts))


def test_the_pack_does_not_change_the_environment_of_the_process():
    import re

    changes = re.compile(r"os\.environ\[[^\]]*\]\s*=(?!=)|os\.environ\.(setdefault|update|pop)\(|os\.putenv\(")
    offenders = []
    for folder in ("core", "nodes"):
        for root, _dirs, files in os.walk(os.path.join(PACK_DIR, folder)):
            for name in files:
                if name.endswith(".py"):
                    with open(os.path.join(root, name), encoding="utf-8") as handle:
                        if changes.search(handle.read()):
                            offenders.append(os.path.relpath(os.path.join(root, name), PACK_DIR))
    assert offenders == []


def test_resolution_notes_shown_in_the_ui_are_english():
    resolution_data = importlib.import_module("NH-Nodes.core.resolution_data")
    notes = [spec.get("note", "") for spec in resolution_data.MODEL_SPECS.values()]
    assert any(notes) and all(note.isascii() for note in notes)


# --- links below an allowed folder ---------------------------------------------------------------------------------

@pytest.fixture
def way_out(machine):
    """photos/way_out is a link to private/, which holds a picture in a folder named like a match."""
    _png(machine / "private" / "secret.png")
    _png(machine / "private" / "Anh" / "secret.png")
    _png(machine / "photos" / "own.png")
    _link_or_skip(machine / "photos" / "way_out", machine / "private")
    return machine


def test_reading_sub_folders_skips_a_link_that_leads_out(way_out):
    files = image_folders.collect_image_files(str(way_out / "photos"), recursive=True)
    assert files == ["own.png"]
    assert _count(folder_path=str(way_out / "photos"), recursive="true")[:2] == (200, {"count": 1})


def test_matching_does_not_search_through_a_link_that_leads_out(way_out):
    node = folder_loaders.NH_LoadImagesMatching()
    for match_text in ("way", "Anh"):
        result = node.load_matching(**_matching_args(matching_root_path=str(way_out / "photos"), match_text=match_text))
        assert result[4] is False  # found
    scanned = []
    real_scandir = os.scandir
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "scandir", lambda path: scanned.append(os.path.normcase(str(path))) or real_scandir(path))
        assert folder_loaders._tree_signature([str(way_out / "photos")]) is not None
    assert scanned == [os.path.normcase(str(way_out / "photos"))]  # the wired-input change check stays out too


def test_a_twin_context_folder_that_is_a_link_out_is_skipped_not_an_error(machine):
    _png(machine / "photos" / "Nam" / "Anh" / "a.png")
    _png(machine / "private" / "Anh" / "b.png")
    _link_or_skip(machine / "photos" / "Nu", machine / "private")
    _, filename, _, _, found = folder_loaders.NH_LoadImagesMatching().load_matching(
        **_matching_args(source_folder_path="x/Nu/Anh", matching_root_path=str(machine / "photos" / "Nam"))
    )
    assert found and filename == "a.png"


def test_an_image_file_that_is_a_link_out_is_not_opened(machine):
    secret = _png(machine / "private" / "secret.png")
    try:
        os.symlink(secret, machine / "photos" / "linked.png")
    except OSError:
        pytest.skip("cannot create a file link here")
    with pytest.raises(FolderNotAllowed):
        _load_folder(str(machine / "photos"))


@pytest.mark.parametrize("settings", [
    dict(create_new_folder=True, new_folder_name="way_out"),
    dict(filename_prefix="way_out/c"),
    dict(filename_prefix="way_out\\deeper\\c"),
])
def test_save_does_not_write_through_a_link_in_the_new_folder_or_the_prefix(way_out, settings):
    before = sorted(way_out.joinpath("private").rglob("*"))
    with pytest.raises(FolderNotAllowed):
        _save(str(way_out / "photos"), **settings)
    assert sorted(way_out.joinpath("private").rglob("*")) == before


# --- more spellings ------------------------------------------------------------------------------------------------

def test_a_declared_folder_may_end_in_a_slash_or_differ_in_case(machine, monkeypatch):
    spelled = str(machine / "photos") + "/"
    if os.name == "nt":
        spelled = spelled.upper()
    monkeypatch.setattr(allowed_folders, "_declared", [spelled])
    assert allowed_folders.can_read(str(machine / "photos" / "set"))
    assert _save(str(machine / "photos" / "exports"))[2] == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows paths")
@pytest.mark.parametrize("path", ["Q:foo", "\\\\?\\C:\\Windows", "\\\\.\\C:\\Windows", "\\\\?\\GLOBALROOT\\Device\\x"])
def test_windows_device_and_drive_relative_paths_are_refused(machine, path):
    assert not allowed_folders.can_read(path)
    with pytest.raises(FolderNotAllowed):
        _save(path)


@pytest.mark.skipif(os.name != "nt", reason="Windows paths")
def test_a_drive_relative_path_lands_in_the_output_folder_or_is_refused(machine):
    # "C:foo" is no absolute path: on the drive of the output folder it is the sub-folder foo of it, on another
    # drive it is refused. Nothing is written anywhere else either way.
    drive = os.path.splitdrive(str(machine))[0]
    assert _save(f"{drive}foo")[1] == str(machine / "output" / "foo")
    assert [path.relative_to(machine).parts[0] for path in machine.rglob("*.png")] == ["output"]


@pytest.mark.skipif(os.name != "nt", reason="Windows paths")
def test_a_device_prefix_cannot_hide_dots_that_leave_an_allowed_folder(machine):
    path = "\\\\?\\" + str(machine / "photos") + "\\..\\private"
    assert not allowed_folders.can_read(path)


def test_matching_change_check_with_wired_inputs_does_not_raise_for_a_refused_root(machine):
    args = _matching_args(matching_root_path=str(machine / "private"), match_text=None)  # match_text is wired
    result = folder_loaders.NH_LoadImagesMatching.IS_CHANGED(**args)
    assert result != result  # NaN: unknown, run again


# --- Image Label fonts ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("font_name", ["..\\..\\secret.ttf", "C:\\secret.ttf", "\\\\server\\share\\x.ttf", "missing.ttf"])
def test_image_label_only_opens_fonts_of_its_list(monkeypatch, font_name):
    image_label = importlib.import_module("NH-Nodes.nodes.image.image_label")
    opened = []
    real = image_label._cached_truetype
    monkeypatch.setattr(image_label, "_cached_truetype", lambda path, size: opened.append(path) or real(path, size))
    assert image_label._load_font(font_name, 20) is not None
    assert opened == ["arial.ttf"]
