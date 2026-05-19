import hashlib
import os

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import folder_paths
import node_helpers


def _intermediate_dtype():
    if not torch.cuda.is_available():
        return torch.float32

    try:
        import comfy.model_management

        return comfy.model_management.intermediate_dtype()
    except Exception:
        return torch.float32


def _input_image_files():
    input_dir = folder_paths.get_input_directory()
    if not os.path.isdir(input_dir):
        return []

    files = [
        file_name
        for file_name in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, file_name))
    ]
    return sorted(folder_paths.filter_files_content_types(files, ["image"]))


class NH_LoadImageInfo:
    """Load one input image and return image metadata strings."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (_input_image_files(), {"image_upload": True}),
                "filename_extension": ("BOOLEAN", {"default": True}),
                "path_include_filename": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "file_name", "path")
    FUNCTION = "load_image"
    CATEGORY = "NH-Nodes/Image"
    SEARCH_ALIASES = ["load image", "load image path", "image file name", "image metadata"]

    def load_image(self, image, filename_extension=True, path_include_filename=True):
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        width = None
        height = None
        dtype = _intermediate_dtype()

        for frame in ImageSequence.Iterator(img):
            frame = node_helpers.pillow(ImageOps.exif_transpose, frame)

            if frame.mode == "I":
                frame = frame.point(lambda value: value * (1 / 255))

            rgb_image = frame.convert("RGB")
            if not output_images:
                width, height = rgb_image.size

            if rgb_image.size != (width, height):
                continue

            image_array = np.array(rgb_image).astype(np.float32) / 255.0
            output_images.append(torch.from_numpy(image_array)[None,].to(dtype=dtype))

            if "A" in frame.getbands():
                mask_array = np.array(frame.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask_array)
            elif frame.mode == "P" and "transparency" in frame.info:
                mask_array = np.array(frame.convert("RGBA").getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask_array)
            else:
                mask = torch.zeros((height, width), dtype=torch.float32, device="cpu")
            output_masks.append(mask.unsqueeze(0).to(dtype=dtype))

            if img.format == "MPO":
                break

        if not output_images:
            raise ValueError(f"No usable image frames found in: {image_path}")

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        file_name = os.path.basename(image_path)
        if not filename_extension:
            file_name = os.path.splitext(file_name)[0]

        path = image_path if path_include_filename else os.path.dirname(image_path)
        return (output_image, output_mask, file_name, path)

    @classmethod
    def IS_CHANGED(cls, image, filename_extension=True, path_include_filename=True):
        image_path = folder_paths.get_annotated_filepath(image)
        hasher = hashlib.sha256()
        with open(image_path, "rb") as image_file:
            hasher.update(image_file.read())
        hasher.update(str(bool(filename_extension)).encode("utf-8"))
        hasher.update(str(bool(path_include_filename)).encode("utf-8"))
        return hasher.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, image, **kwargs):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


NODE_CLASS_MAPPINGS = {
    "NH_LoadImageInfo": NH_LoadImageInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_LoadImageInfo": "Load Image Info (NH)",
}
