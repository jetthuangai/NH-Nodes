"""Load images matching: the source's context folder (Nu / Nam …) decides between same-named folders.

Folders that every candidate shares with the source (drive, D:/Photos …) must not outweigh that context.
"""

import importlib
import os

import pytest

folder_loaders = importlib.import_module("NH-Nodes.nodes.image.folder_loaders")


def _tree(root, *relative_dirs):
    for relative in relative_dirs:
        os.makedirs(os.path.join(root, relative), exist_ok=True)


def _pick(root, source, match_text="subject_01", recursive=True):
    picked = folder_loaders._resolve_best_matching_folder(
        str(root), match_text, "exact", recursive, source_context_paths=[str(source)],
    )
    return os.path.relpath(picked, str(root)).replace("\\", "/")


@pytest.mark.parametrize("group", ["Nu", "Nam"])
def test_library_next_to_the_source_follows_the_source_group(tmp_path, group):
    # D:/Photos/Source/<group>/subject_01 and D:/Photos/Library/{Nam,Nu}/subject_01 share D:/Photos.
    photos = tmp_path / "Photos"
    _tree(photos / "Library", "Nam/subject_01", "Nu/subject_01")
    source = photos / "Source" / group / "subject_01"
    _tree(source)
    assert _pick(photos / "Library", source) == f"{group}/subject_01"


@pytest.mark.parametrize("group", ["Nu", "Nam"])
def test_source_inside_the_library_tree_follows_its_group(tmp_path, group):
    library = tmp_path / "Photos" / "Library"
    _tree(library, "Nam/subject_01", "Nu/subject_01", f"{group}/subject_01/edits")
    source = library / group / "subject_01" / "edits"
    assert _pick(library, source) == f"{group}/subject_01"


def test_without_context_the_first_folder_in_natural_order_wins(tmp_path):
    library = tmp_path / "Library"
    _tree(library, "Nam/subject_01", "Nu/subject_01")
    picked = folder_loaders._resolve_best_matching_folder(str(library), "subject_01", "exact", True)
    assert os.path.relpath(picked, str(library)).replace("\\", "/") == "Nam/subject_01"


def test_context_scores_ignore_ancestors_all_candidates_share(tmp_path):
    shared = tmp_path / "Photos" / "Library"
    paths = [str(shared / "Nam" / "subject_01"), str(shared / "Nu" / "subject_01")]
    source = [str(tmp_path / "Photos" / "Source" / "Nu" / "subject_01")]
    scores = folder_loaders._context_scores(paths, source)
    assert scores[paths[1]] > scores[paths[0]]
    assert folder_loaders._context_scores(paths, None) == {paths[0]: 0, paths[1]: 0}
