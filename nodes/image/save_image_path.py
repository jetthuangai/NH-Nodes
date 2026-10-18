"""Save Image Path (NH): numbered image files in any folder, with an optional preview."""

import os
import re
import uuid

import folder_paths

from ...core.utils import tensor_to_pil

_WINDOWS = os.name == "nt"


def _save_dir_from_text(folder_path):
    folder_path = (folder_path or "").strip()
    if not folder_path:
        return folder_paths.get_output_directory()
    if os.path.isfile(folder_path):
        return os.path.dirname(folder_path)
    if os.path.isabs(folder_path):
        return folder_path
    return os.path.join(folder_paths.get_output_directory(), folder_path)


def _contained_path(base_dir, relative, what):
    """Absolute path of ``relative`` below ``base_dir``; ValueError (naming ``what``) if it would leave ``base_dir``.

    Both slash styles count as separators on every OS, so a workflow written on Windows is judged the same elsewhere.
    """
    base_dir = os.path.abspath(base_dir)
    path = os.path.abspath(os.path.join(base_dir, str(relative).replace("\\", "/")))
    try:
        inside = os.path.commonpath([base_dir, path]) == base_dir
    except ValueError:  # another drive or a UNC share
        inside = False
    if not inside:
        raise ValueError(f"{what} must stay inside the target folder: {relative!r}")
    return path


def _save_child_dir(save_dir, folder_name):
    folder_name = str(folder_name or "").strip()
    if not folder_name:
        return save_dir
    return _contained_path(save_dir, folder_name, "new_folder_name")


def _split_filename_prefix(save_dir, prefix):
    """('sub/dir/', 'name', directory) for a filename_prefix such as 'sub/dir/name'.

    Subfolders are allowed, but the folder part must stay inside ``save_dir`` (no '..', no absolute path) and the
    file name must be a plain name. Without a separator the name is not looked at by the folder check, and on
    Windows a colon turns it into a path: os.path.join(save_dir, 'D:x0001.png') is a file on drive D:, 'C:x' is
    relative to the current folder of drive C: (join drops the 'C:', so the reported name is wrong and the
    numbering never finds the file again) and 'name:stream' writes an NTFS alternate data stream. A Windows file
    name cannot contain a colon anyway, so such a prefix is refused whole, before anything is created.
    """
    if _WINDOWS and ":" in prefix:
        raise ValueError(
            "filename_prefix cannot contain ':' on Windows (it selects a drive or an NTFS stream, "
            f"not a file name): {prefix!r}"
        )
    folder_part, name = re.match(r"(.*[\\/])?([^\\/]*)\Z", prefix, re.DOTALL).groups()
    if not folder_part:
        return "", name, save_dir
    return folder_part, name, _contained_path(save_dir, folder_part, "filename_prefix")


def _save_pil_image(image, save_path, file_format, dpi):
    image = image.convert("RGB")
    save_kwargs = {"dpi": (dpi, dpi)}

    if file_format == "png":
        pil_format = "PNG"
        save_kwargs["compress_level"] = 4
    elif file_format == "jpg":
        pil_format = "JPEG"
        save_kwargs["quality"] = 95
        save_kwargs["subsampling"] = 0
    elif file_format == "webp":
        pil_format = "WEBP"
        save_kwargs["quality"] = 95
        save_kwargs["method"] = 6
    elif file_format == "avif":
        pil_format = "AVIF"
        save_kwargs["quality"] = 90
    else:
        raise ValueError(f"Unsupported format: {file_format}")

    try:
        image.save(save_path, format=pil_format, **save_kwargs)
    except TypeError:
        if file_format == "avif" and "dpi" in save_kwargs:
            save_kwargs.pop("dpi", None)
            image.save(save_path, format=pil_format, **save_kwargs)
            print("[NH-Nodes] Save Image: AVIF DPI metadata is not supported by the current Pillow build.")
        else:
            raise


class NH_SaveImagePath:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "folder_path": ("STRING", {"default": "", "multiline": False}),
                "create_new_folder": ("BOOLEAN", {"default": False}),
                "new_folder_name": ("STRING", {"default": "", "multiline": False}),
                "filename_prefix": ("STRING", {"default": "NH", "multiline": False}),
                "number_suffix": ("BOOLEAN", {"default": True}),
                "preview_image": ("BOOLEAN", {"default": False}),
                "file_format": (["png", "jpg", "webp", "avif"],),
                "dpi": ("INT", {"default": 300, "min": 1, "max": 2400, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("filenames", "path", "saved_count")
    FUNCTION = "save_images"
    CATEGORY = "NH-Nodes/Image"

    def save_images(self, images, folder_path, filename_prefix, number_suffix=True, preview_image=False, file_format="png", dpi=300, create_new_folder=False, new_folder_name=""):
        save_dir = _save_dir_from_text(folder_path)
        if create_new_folder:
            save_dir = _save_child_dir(save_dir, new_folder_name)

        # A prefix may name subfolders ("renders/img"); they live below save_dir and are where files are counted.
        prefix = (filename_prefix or "NH").strip() or "NH"
        prefix_folder, prefix_name, target_dir = _split_filename_prefix(save_dir, prefix)
        os.makedirs(target_dir, exist_ok=True)

        numbered_pattern = re.compile(rf"^{re.escape(prefix_name)}(\d+)\.(?:png|jpe?g|webp|avif)$", re.IGNORECASE)
        legacy_numbered_pattern = re.compile(rf"^(?:{re.escape(prefix_name)})?(\d+)\.(?:png|jpe?g|webp|avif)$", re.IGNORECASE)
        bare_pattern = re.compile(rf"^{re.escape(prefix_name)}\.(?:png|jpe?g|webp|avif)$", re.IGNORECASE)

        used_numbers = set()
        bare_exists = False
        for name in os.listdir(target_dir):
            if bare_pattern.match(name):
                bare_exists = True
                continue
            match = numbered_pattern.match(name)
            if match is None and number_suffix:
                match = legacy_numbered_pattern.match(name)
            if match:
                used_numbers.add(int(match.group(1)))

        saved_files = []
        preview_results = []
        preview_dir = folder_paths.get_temp_directory()
        next_number = 1
        for batch_index, image in enumerate(images):
            if number_suffix:
                while next_number in used_numbers:
                    next_number += 1
                filename = f"{prefix_name}{next_number:04d}.{file_format}"
                used_numbers.add(next_number)
                next_number += 1
            elif not bare_exists and not used_numbers:
                filename = f"{prefix_name}.{file_format}"
                bare_exists = True
            else:
                while next_number in used_numbers:
                    next_number += 1
                filename = f"{prefix_name}{next_number}.{file_format}"
                used_numbers.add(next_number)
                next_number += 1

            pil_image = tensor_to_pil(image, rgb=True)
            _save_pil_image(pil_image, os.path.join(target_dir, filename), file_format, int(dpi))
            saved_files.append(prefix_folder + filename)

            if preview_image:
                preview_filename = f"NH_preview_{uuid.uuid4().hex}_{batch_index}.png"
                _save_pil_image(pil_image, os.path.join(preview_dir, preview_filename), "png", int(dpi))
                preview_results.append({
                    "filename": preview_filename,
                    "subfolder": "",
                    "type": "temp",
                })

        result = ("\n".join(saved_files), save_dir, len(saved_files))
        if preview_image:
            return {"ui": {"images": preview_results}, "result": result}
        return result


NODE_CLASS_MAPPINGS = {
    "NH_SaveImagePath": NH_SaveImagePath,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_SaveImagePath": "Save Image Path (NH)",
}
