"""Load Images Folder (NH) and Load images matching: read the images of a folder, or of the folder that matches a name."""

import hashlib
import os
import re
import unicodedata
from pathlib import Path

import torch
from PIL import Image, ImageOps

import folder_paths
import node_helpers
from comfy_execution.graph_utils import ExecutionBlocker

from ...core.image_folders import IMAGE_EXTENSIONS, collect_image_files, natural_key, resolve_load_dir
from ...core.image_ops import pil_to_image_tensor, resize_pil


def _normalize_match_value(value):
    value = unicodedata.normalize("NFD", str(value or ""))
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    value = re.sub(r"[_\-]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip().casefold()


def _path_parts(value):
    return [part.strip() for part in re.split(r"[\\/]+", str(value or "")) if part.strip()]


def _normalized_path_parts(value):
    return [part for part in (_normalize_match_value(item) for item in _path_parts(value)) if part]


def _longest_common_path_run(candidate_parts, source_parts):
    if not candidate_parts or not source_parts:
        return 0

    previous = [0] * (len(source_parts) + 1)
    best = 0
    for candidate_part in candidate_parts:
        current = [0] * (len(source_parts) + 1)
        for index, source_part in enumerate(source_parts, start=1):
            if candidate_part == source_part:
                current[index] = previous[index - 1] + 1
                best = max(best, current[index])
        previous = current
    return best


def _ordered_path_overlap(candidate_parts, source_parts):
    if not candidate_parts or not source_parts:
        return 0

    count = 0
    cursor = 0
    for candidate_part in candidate_parts:
        while cursor < len(source_parts) and source_parts[cursor] != candidate_part:
            cursor += 1
        if cursor >= len(source_parts):
            break
        count += 1
        cursor += 1
    return count


def _candidate_context_score(candidate_path, source_context_paths):
    if not source_context_paths:
        return 0

    candidate_parts = _normalized_path_parts(candidate_path)
    best_score = 0
    for source_path in source_context_paths:
        source_parts = _normalized_path_parts(source_path)
        run = _longest_common_path_run(candidate_parts, source_parts)
        ordered = _ordered_path_overlap(candidate_parts, source_parts)
        score = run * 100 + ordered
        if candidate_parts and len(source_parts) >= 2 and candidate_parts[-1] == source_parts[-2]:
            score += 10
        elif candidate_parts and source_parts and candidate_parts[-1] == source_parts[-1]:
            score += 5
        best_score = max(best_score, score)
    return best_score


def _source_segment_set(source_context_paths):
    segments = set()
    for path_value in source_context_paths or []:
        segments.update(_normalized_path_parts(path_value))
    return segments


def _add_unique_path(paths, path_value):
    if not path_value:
        return
    key = os.path.normcase(os.path.normpath(path_value))
    if key not in {os.path.normcase(os.path.normpath(path)) for path in paths}:
        paths.append(path_value)


_SIGNATURE_ENTRY_LIMIT = 20000  # most folders + files Load Images Matching hashes for IS_CHANGED before it gives up


class _NoMatchError(ValueError):
    """The search itself came up empty: no matching folder, or no images in it. A bad setting is a plain ValueError."""

    def __init__(self, message, match_text=""):
        super().__init__(message)
        self.match_text = match_text


def _hash_parts(hasher, *parts):
    for part in parts:
        hasher.update(str(part).encode("utf-8", errors="replace"))
        hasher.update(b"\0")


def _same_path(first, second):
    return os.path.normcase(os.path.abspath(first)) == os.path.normcase(os.path.abspath(second))


def _is_inside(path, folder):
    """True when ``path`` lies below ``folder`` (``folder`` itself does not count)."""
    path = os.path.normcase(os.path.abspath(path))
    folder = os.path.normcase(os.path.abspath(folder))
    return path != folder and path.startswith(folder.rstrip(os.sep) + os.sep)


def _is_drive_root(path):
    path = os.path.abspath(path)
    return os.path.dirname(path) == path


def _entry_is_dir(entry, follow_symlinks=True):
    try:
        return entry.is_dir(follow_symlinks=follow_symlinks)
    except OSError:
        return False


def _absolute_root(root_path):
    """The matching root as an absolute path; a relative one is taken relative to ComfyUI's input folder."""
    root_path = (root_path or "").strip()
    if root_path and not os.path.isabs(root_path):
        root_path = os.path.join(folder_paths.get_input_directory(), root_path)
    return root_path


def _check_search_root(root_path, recursive):
    """Refuse the roots a search must not scan: a whole drive (when it would be walked) and the models folder."""
    if recursive and _is_drive_root(root_path):
        raise ValueError(
            f"Matching root is a drive root ({root_path}); refusing to scan a whole drive. "
            "Point matching_root_path at the folder that holds the images, or turn recursive off."
        )
    if _same_path(root_path, folder_paths.models_dir):
        raise ValueError(f"Matching root is the ComfyUI models folder ({root_path}), which holds no images to match.")


def _list_folder(folder_path):
    """The os.DirEntry objects in ``folder_path`` in natural order, without the ComfyUI models folder; [] if unreadable."""
    try:
        with os.scandir(folder_path) as scan:
            entries = sorted(scan, key=lambda entry: natural_key(entry.name))
    except OSError:
        return []
    models_dir = os.path.normcase(os.path.abspath(folder_paths.models_dir))
    return [
        entry for entry in entries
        if not (_entry_is_dir(entry) and os.path.normcase(os.path.normpath(entry.path)) == models_dir)
    ]


def _child_folders(folder_path):
    """Paths of the sub-folders directly inside ``folder_path``, in natural order."""
    return [entry.path for entry in _list_folder(folder_path) if _entry_is_dir(entry)]


def _walk_folders(root_path, recursive=True):
    """Yield ``(folder, entries)`` for ``root_path`` and, when ``recursive``, for every folder below it.

    The matching search's stand-in for os.walk, with limits: it refuses the roots ``_check_search_root`` names,
    never enters the ComfyUI models folder and, like os.walk, does not follow symlinks into other trees.
    """
    _check_search_root(root_path, recursive)
    pending = [root_path]
    while pending:
        folder = pending.pop()
        entries = _list_folder(folder)
        yield folder, entries
        if recursive:
            pending.extend(entry.path for entry in reversed(entries) if _entry_is_dir(entry, follow_symlinks=False))


def _context_twins(root_path, context_names, wanted=None):
    """Folders that stand in for ``root_path`` under another context folder.

    When the root sits inside a context folder (``.../Nam/...``), the same place under a sibling context folder
    (``.../Nu/...``) is a twin, and so is the sibling context folder as a whole. ``wanted`` limits the siblings to
    those names (normalized); all other configured context names count when it is None.
    """
    context = {_normalize_match_value(name) for name in context_names} - {""}
    if not context:
        return []

    parts = list(Path(os.path.normpath(root_path)).parts)
    twins = []
    for position, part in enumerate(parts):
        own_name = _normalize_match_value(part)
        if position == 0 or own_name not in context:
            continue
        rest = parts[position + 1:]
        for sibling_path in _child_folders(str(Path(*parts[:position]))):
            sibling_name = _normalize_match_value(os.path.basename(sibling_path))
            if sibling_name == own_name or sibling_name not in context:
                continue
            if wanted is not None and sibling_name not in wanted:
                continue
            twins.append(sibling_path)
            if rest and os.path.isdir(os.path.join(sibling_path, *rest)):
                twins.append(os.path.join(sibling_path, *rest))
    return twins


def _context_root_candidates(root_path, source_context_paths, context_names=()):
    """Folders to search for the match, best context first.

    That is the matching root, the sub-folders of it that the source path names (``Nu`` when the source lives in a
    ``Nu`` folder) and the context twins of the root. Nothing else above the root is listed, let alone walked.
    """
    root_path = _absolute_root(root_path)
    if not root_path:
        return []

    candidates = []
    _add_unique_path(candidates, root_path)

    source_segments = _source_segment_set(source_context_paths)
    if not source_segments:
        return candidates

    for child_path in _child_folders(root_path):
        if _normalize_match_value(os.path.basename(child_path)) in source_segments:
            _add_unique_path(candidates, child_path)
    for twin_path in _context_twins(root_path, context_names, source_segments):
        _add_unique_path(candidates, twin_path)

    candidates.sort(key=lambda item: (-_candidate_context_score(item, source_context_paths), natural_key(item)))
    return candidates


def _resolve_matching_folder(root_path, match_text, match_mode, recursive, source_context_paths=None):
    root_path = _absolute_root(root_path)
    if not root_path:
        raise ValueError("Matching root path is empty")
    if not os.path.isdir(root_path):
        raise ValueError(f"Matching root folder does not exist: {root_path}")

    normalized_match = _normalize_match_value(match_text)
    if not normalized_match:
        raise ValueError("Match text is empty")

    candidates = [root_path]
    for _folder, entries in _walk_folders(root_path, recursive):
        candidates.extend(entry.path for entry in entries if _entry_is_dir(entry))

    exact_matches = []
    contains_matches = []
    for candidate in candidates:
        normalized_name = _normalize_match_value(os.path.basename(candidate))
        if normalized_name == normalized_match:
            exact_matches.append(candidate)
        if normalized_match in normalized_name:
            contains_matches.append(candidate)

    matches = exact_matches if match_mode == "exact" else contains_matches or exact_matches
    if not matches:
        raise _NoMatchError(f"No matching folder found for '{match_text}' under: {root_path}", match_text)

    if source_context_paths:
        matches.sort(key=lambda item: (
            -_candidate_context_score(item, source_context_paths),
            natural_key(os.path.relpath(item, root_path)),
        ))
    else:
        matches.sort(key=lambda item: natural_key(os.path.relpath(item, root_path)))
    return matches[0]


def _resolve_best_matching_folder(root_path, match_text, match_mode, recursive, source_context_paths=None, context_names=()):
    root_path = _absolute_root(root_path)
    if not root_path:
        raise ValueError("Matching root path is empty")
    _check_search_root(root_path, recursive)

    search_roots = _context_root_candidates(root_path, source_context_paths, context_names)
    if recursive:
        # Walking a folder covers everything below it, so a search root inside another one is not walked twice.
        search_roots = [path for path in search_roots if not any(_is_inside(path, other) for other in search_roots)]

    matched_folders = []
    errors = []
    for search_root in search_roots:
        try:
            matched_folder = _resolve_matching_folder(
                search_root,
                match_text,
                match_mode,
                recursive,
                source_context_paths=source_context_paths,
            )
            _add_unique_path(matched_folders, matched_folder)
        except ValueError as exc:
            errors.append(exc)

    if not matched_folders:
        # A setting that cannot work (missing root, empty text) is reported as such; otherwise nothing matched.
        for exc in errors:
            if not isinstance(exc, _NoMatchError):
                raise exc
        raise _NoMatchError(f"No matching folder found for '{match_text}' under: {root_path}", match_text)

    matched_folders.sort(key=lambda item: (-_candidate_context_score(item, source_context_paths), natural_key(item)))
    return matched_folders[0]


def _parse_context_folder_names(value):
    names = [item.strip() for item in re.split(r"[\n,;|]+", str(value or "")) if item.strip()]
    return names


def _detect_context_folder(paths, context_folder_names):
    if not context_folder_names:
        return ""

    normalized_context = {_normalize_match_value(name): name for name in context_folder_names}
    for path_value in paths:
        for part in re.split(r"[\\/]+", str(path_value or "")):
            match = normalized_context.get(_normalize_match_value(part))
            if match:
                return match
    return ""


def _matching_selection(
    source_folder_path,
    matching_root_path,
    match_text,
    match_mode,
    recursive,
    index,
    step,
    context_folder_names,
    source_file_names,
    person_folder_path=None,
):
    """Work out what Load images matching reads: one ``(match_text, folder, [file names])`` per match item.

    Raises _NoMatchError when an item finds no folder, or a folder without images, and ValueError for a setting
    that cannot work (missing root, empty match text, a root that must not be scanned).
    """
    if not source_folder_path and person_folder_path:
        source_folder_path = person_folder_path

    source_paths = [line.strip() for line in str(source_folder_path or "").splitlines() if line.strip()]
    first_source_path = source_paths[0] if source_paths else ""
    resolved_source_folder = resolve_load_dir(first_source_path)
    selected_match_text = (match_text or "").strip()
    source_names = [line.strip() for line in str(source_file_names or "").splitlines() if line.strip()]
    context_names = _parse_context_folder_names(context_folder_names)
    match_items = []

    if selected_match_text:
        item_count = max(len(source_paths), len(source_names), 1)
        for item_index in range(item_count):
            match_items.append((
                selected_match_text,
                source_paths[item_index] if item_index < len(source_paths) else "",
                source_names[item_index] if item_index < len(source_names) else "",
            ))
    elif source_names:
        for item_index, source_name in enumerate(source_names):
            source_dir = os.path.dirname(source_name)
            source_match_text = os.path.basename(os.path.normpath(source_dir)) if source_dir else os.path.basename(os.path.normpath(resolved_source_folder))
            match_items.append((
                source_match_text,
                source_paths[item_index] if item_index < len(source_paths) else "",
                source_name,
            ))
    else:
        selected_match_text = os.path.basename(os.path.normpath(resolved_source_folder))
        match_items.append((selected_match_text, first_source_path, ""))

    plan = []
    for current_match_text, current_source_path, current_source_name in match_items:
        context_paths = [current_source_path, current_source_name, resolved_source_folder]
        context_folder = _detect_context_folder(context_paths, context_names)
        if context_folder:
            context_paths.append(context_folder)
        matched_folder = _resolve_best_matching_folder(
            matching_root_path,
            current_match_text,
            match_mode,
            recursive,
            source_context_paths=context_paths,
            context_names=context_names,
        )
        try:
            image_files = collect_image_files(matched_folder)
        except ValueError as exc:  # no images in the folder that matched
            raise _NoMatchError(str(exc), current_match_text) from exc
        start_index = index % len(image_files)
        selected_files = [image_files[(start_index + offset) % len(image_files)] for offset in range(step)]
        plan.append((current_match_text, matched_folder, selected_files))
    return plan


def _selection_signature(plan):
    """Digest of what a matching plan reads: the folders chosen, and the selected files with size and mtime."""
    hasher = hashlib.sha256()
    for current_match_text, matched_folder, selected_files in plan:
        _hash_parts(hasher, current_match_text, os.path.abspath(matched_folder))
        for file_name in selected_files:
            stat = os.stat(os.path.join(matched_folder, file_name))
            _hash_parts(hasher, file_name, stat.st_size, stat.st_mtime_ns)
    return hasher.hexdigest()


def _tree_signature(roots, limit=None):
    """Digest of every folder and image below ``roots`` (names, sizes, mtimes); None beyond ``limit`` entries."""
    limit = _SIGNATURE_ENTRY_LIMIT if limit is None else limit
    hasher = hashlib.sha256()
    seen = 0
    for root in roots:
        _hash_parts(hasher, "root", os.path.abspath(root))
        for folder, entries in _walk_folders(root, recursive=True):
            _hash_parts(hasher, "folder", os.path.relpath(folder, root))
            for entry in entries:
                seen += 1
                if seen > limit:
                    return None
                if _entry_is_dir(entry):
                    _hash_parts(hasher, "dir", entry.name)
                elif Path(entry.name).suffix.lower() in IMAGE_EXTENSIONS:
                    try:
                        stat = entry.stat()
                    except OSError:
                        _hash_parts(hasher, "image", entry.name)
                    else:
                        _hash_parts(hasher, "image", entry.name, stat.st_size, stat.st_mtime_ns)
    return hasher.hexdigest()


def _load_image_batch(paths):
    """The images at ``paths`` as one RGB IMAGE batch; every image after the first is scaled to the size of the first."""
    images = []
    target_size = None
    for path in paths:
        img = node_helpers.pillow(Image.open, path)
        img = node_helpers.pillow(ImageOps.exif_transpose, img).convert("RGB")
        if target_size is None:
            target_size = img.size
        elif img.size != target_size:
            img = resize_pil(img, target_size, "lanczos")
        images.append(pil_to_image_tensor(img))
    return torch.cat(images, dim=0)


class NH_LoadImagesFromFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "multiline": False}),
                "index": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "step": ("INT", {"default": 1, "min": 1, "max": 9999}),
                "seed_mode": (["fixed", "increment", "decrement", "random"],),
                "recursive": ("BOOLEAN", {"default": False}),
                "filename_extension": ("BOOLEAN", {"default": True}),
                "path_output": (["file_path", "folder_path"], {"default": "file_path"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "filename", "path")
    FUNCTION = "load_images"
    CATEGORY = "NH-Nodes/Image"

    @classmethod
    def IS_CHANGED(cls, folder_path, index, step, seed_mode, recursive=False, filename_extension=True, path_output="file_path", **kwargs):
        # An input wired from another node arrives here as None, so the folder is unknown (resolving None would
        # fingerprint ComfyUI's input folder instead). The upstream value is already part of ComfyUI's cache key, so a
        # constant keeps an unchanged graph cached while a new path still runs the node again.
        if folder_path is None:
            return "wired-folder-path"
        try:
            resolved_folder = resolve_load_dir(folder_path)
            image_files = collect_image_files(resolved_folder, recursive=recursive)
            step = max(1, int(step))
            start_index = int(index) % len(image_files)

            if seed_mode == "decrement":
                indices = [((start_index - offset) % len(image_files)) for offset in range(step)]
            else:
                indices = [((start_index + offset) % len(image_files)) for offset in range(step)]

            hasher = hashlib.sha256()
            _hash_parts(
                hasher,
                os.path.abspath(resolved_folder),
                start_index,
                step,
                seed_mode,
                bool(recursive),
                bool(filename_extension),
                path_output,
            )
            for selected_index in indices:
                rel_path = image_files[selected_index]
                stat = os.stat(os.path.join(resolved_folder, rel_path))
                _hash_parts(hasher, rel_path, stat.st_size, stat.st_mtime_ns)

            return hasher.hexdigest()
        except Exception as exc:
            return f"error:{type(exc).__name__}:{exc}"

    def load_images(self, folder_path, index, step, seed_mode, recursive=False, filename_extension=True, path_output="file_path", **kwargs):
        resolved_folder = resolve_load_dir(folder_path)
        image_files = collect_image_files(resolved_folder, recursive=recursive)
        start_index = index % len(image_files)

        if seed_mode == "decrement":
            indices = [((start_index - offset) % len(image_files)) for offset in range(step)]
        else:
            indices = [((start_index + offset) % len(image_files)) for offset in range(step)]

        selected_files = [image_files[i] for i in indices]
        full_paths = [os.path.join(resolved_folder, file_name) for file_name in selected_files]
        if path_output == "folder_path":
            output_paths = [os.path.dirname(full_path) for full_path in full_paths]
        else:
            output_paths = full_paths

        display_filenames = []
        for file_name in selected_files:
            base_name = os.path.basename(file_name)
            if not filename_extension:
                base_name = os.path.splitext(base_name)[0]
            display_filenames.append(base_name)

        return (_load_image_batch(full_paths), "\n".join(display_filenames), "\n".join(output_paths))


class NH_LoadImagesMatching:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_folder_path": ("STRING", {"default": "", "multiline": False}),
                "matching_root_path": ("STRING", {"default": "", "multiline": False}),
                "match_text": ("STRING", {"default": "", "multiline": False}),
                "match_mode": (["contains", "exact"],),
                "recursive": ("BOOLEAN", {"default": True}),
                "index": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "step": ("INT", {"default": 1, "min": 1, "max": 9999}),
                "repeat_count": ("INT", {"default": 1, "min": 1, "max": 9999}),
                "context_folder_names": ("STRING", {"default": "Nam\nNu", "multiline": True}),
            },
            "optional": {
                "source_file_names": ("STRING", {"default": "", "multiline": True}),
                "fallback_images": ("IMAGE", {
                    "lazy": True,
                    "tooltip": (
                        "Comes out as images when nothing matches, and is only evaluated then. Connect it here "
                        "instead of routing images through a switch or If/Else: a node that receives the held-back "
                        "images never runs."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("images", "filename", "path", "matched_text", "found")
    FUNCTION = "load_matching"
    CATEGORY = "NH-Nodes/Image"
    DESCRIPTION = (
        "Load the images of the folder under matching_root_path whose name matches the source folder (or match_text). "
        "When nothing matches, found is False and images are held back: every node that receives them is skipped, "
        "a switch or If/Else included. To carry on with another image, connect it to fallback_images instead of "
        "routing images through a switch or If/Else: it is only evaluated when nothing matches, and then comes out "
        "as images (found stays False)."
    )

    @classmethod
    def IS_CHANGED(cls, source_folder_path, matching_root_path, match_text, match_mode, recursive, index, step, repeat_count, context_folder_names="Nam\nNu", source_file_names="", **kwargs):
        # ComfyUI's cache key already holds every setting and every upstream result; this adds what is on disk:
        # the folder(s) the search picks and the selected files with their size and time, so an edit re-runs the node.
        # An input wired from another node arrives here as None, which leaves the folder unknown: the whole matching
        # tree is fingerprinted instead. A search that finds nothing, or a tree too big to hash cheaply, gives NaN:
        # that result is recomputed every time (so a "not found" is never remembered).
        unknown = float("nan")
        try:
            settings = (source_folder_path, matching_root_path, match_text, match_mode, recursive, index, step, context_folder_names, source_file_names)
            if None not in settings:
                plan = _matching_selection(
                    source_folder_path, matching_root_path, match_text, match_mode, recursive, index, step,
                    context_folder_names, source_file_names, kwargs.get("person_folder_path"),
                )
                return _selection_signature(plan)
            root_path = _absolute_root(matching_root_path)
            if not root_path or context_folder_names is None:
                return unknown
            roots = [root_path] + _context_twins(root_path, _parse_context_folder_names(context_folder_names))
            signature = _tree_signature(roots)
            return unknown if signature is None else signature
        except (ValueError, OSError):
            return unknown

    def check_lazy_status(self, source_folder_path, matching_root_path, match_text, match_mode, recursive, index, step, repeat_count, context_folder_names="Nam\nNu", source_file_names="", **kwargs):
        # fallback_images is asked for only when it is wired (ComfyUI sends its key, with None until the input has
        # run) and the search finds nothing; on a match its upstream nodes never run. The search is the one
        # load_matching makes. A setting that cannot work asks for nothing: load_matching raises it right after.
        if "fallback_images" not in kwargs or kwargs["fallback_images"] is not None:
            return []
        try:
            _matching_selection(
                source_folder_path, matching_root_path, match_text, match_mode, recursive, index, step,
                context_folder_names, source_file_names, kwargs.get("person_folder_path"),
            )
        except _NoMatchError:
            return ["fallback_images"]
        except (ValueError, OSError):
            pass
        return []

    def load_matching(self, source_folder_path, matching_root_path, match_text, match_mode, recursive, index, step, repeat_count, context_folder_names="Nam\nNu", source_file_names="", fallback_images=None, **kwargs):
        try:
            plan = _matching_selection(
                source_folder_path, matching_root_path, match_text, match_mode, recursive, index, step,
                context_folder_names, source_file_names, kwargs.get("person_folder_path"),
            )
        except _NoMatchError as exc:
            # Nothing to load for this request. `found` lets a workflow carry on with something else, and so does
            # fallback_images, which comes out as the images. Without it the images are held back (whatever needs
            # them is told why). A setting that cannot work still raises.
            message = str(exc)
            if fallback_images is not None:
                print(f"[NH-Nodes] Load images matching: {message} (using fallback_images)")
                return (fallback_images, "", "", exc.match_text, False)
            print(f"[NH-Nodes] Load images matching: {message}")
            return (ExecutionBlocker(message), "", "", exc.match_text, False)

        selected_files = []
        matched_folders = []
        matched_texts = []

        for current_match_text, matched_folder, file_names in plan:
            for file_name in file_names:
                selected_files.append(file_name)
                matched_folders.append(matched_folder)
                matched_texts.append(current_match_text)

        images = _load_image_batch(
            [os.path.join(folder, file_name) for folder, file_name in zip(matched_folders, selected_files)]
        )
        if repeat_count > 1:
            images = images.repeat(repeat_count, 1, 1, 1)

        return (images, "\n".join(selected_files), "\n".join(matched_folders), "\n".join(matched_texts), True)


NODE_CLASS_MAPPINGS = {
    "NH_LoadImagesFromFolder": NH_LoadImagesFromFolder,
    "NH_LoadImagesMatching": NH_LoadImagesMatching,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_LoadImagesFromFolder": "Load Images Folder (NH)",
    "NH_LoadImagesMatching": "Load images matching",
}
