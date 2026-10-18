"""Image folder helpers shared by the folder loader nodes and the folder-count HTTP route.

``resolve_load_dir`` turns the folder text of a loader into a directory, ``collect_image_files`` lists the images in
it in natural order. Nothing here needs torch or Pillow.
"""

import os
import re
from pathlib import Path

import folder_paths

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".bmp", ".tif", ".tiff"}


def natural_key(value):
    """Sort key that puts 'img2' before 'img10' and ignores case."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def resolve_load_dir(folder_path, image="", upload_probe=""):
    """The folder a loader reads.

    ``folder_path`` when given (relative to ComfyUI's input folder unless absolute; a file stands for its folder),
    else the folder of the uploaded ``image`` / ``upload_probe``, else the input folder.
    """
    folder_path = (folder_path or "").strip()
    if folder_path:
        resolved = folder_path
        if not os.path.isabs(resolved):
            resolved = os.path.join(folder_paths.get_input_directory(), resolved)
        if os.path.isfile(resolved):
            resolved = os.path.dirname(resolved)
        return resolved

    upload_name = image or upload_probe
    if upload_name:
        return os.path.dirname(folder_paths.get_annotated_filepath(upload_name))

    return folder_paths.get_input_directory()


def collect_image_files(folder_path, recursive=False):
    """The images of ``folder_path`` in natural order, as names (paths relative to the folder below it).

    Sub-folders are read when ``recursive`` is set, and also when the folder holds no image itself. ValueError for a
    folder that does not exist or has no images.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder does not exist: {folder_path}")

    files = []
    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        if os.path.isfile(full_path) and Path(entry).suffix.lower() in IMAGE_EXTENSIONS:
            files.append(entry)

    if recursive or not files:
        seen = set(files)
        models_dir = os.path.normcase(os.path.abspath(folder_paths.models_dir))
        for current_root, dir_names, file_names in os.walk(folder_path):
            # Natural order, and never into the ComfyUI models folder (a walk from one of its parents would crawl it).
            dir_names[:] = sorted(
                (name for name in dir_names if os.path.normcase(os.path.abspath(os.path.join(current_root, name))) != models_dir),
                key=natural_key,
            )
            for file_name in sorted(file_names, key=natural_key):
                full_path = os.path.join(current_root, file_name)
                if os.path.isfile(full_path) and Path(file_name).suffix.lower() in IMAGE_EXTENSIONS:
                    rel_path = os.path.relpath(full_path, folder_path)
                    if rel_path not in seen:
                        seen.add(rel_path)
                        files.append(rel_path)

    files.sort(key=natural_key)
    if not files:
        raise ValueError(f"No images found in folder: {folder_path}")
    return files
