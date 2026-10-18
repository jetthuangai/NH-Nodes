"""Large image save node for NH-Nodes."""

import json
import os

import folder_paths
from comfy.cli_args import args
from PIL.PngImagePlugin import PngInfo

from ...core.utils import tensor_to_pil

# Cạnh dài nhất (px) mà từng định dạng ghi được; PNG/TIFF không bị giới hạn trong thực tế.
# JPEG: spec cho tới 65535 nhưng libjpeg (Pillow) từ chối > 65500 (JPEG_MAX_DIMENSION).
_MAX_SIDE_BY_EXTENSION = {"webp": 16383, "jpg": 65500}


def _check_size_limit(extension, width, height):
    limit = _MAX_SIDE_BY_EXTENSION.get(extension)
    if limit is not None and max(width, height) > limit:
        raise ValueError(
            f"NH Save Large Image: {width}x{height} px is too large for {extension.upper()} "
            f"(max {limit} px per side). Use png or tiff for images this large."
        )


def _extension_for_save_format(file_format):
    file_format = str(file_format or "png").lower()
    if file_format in {"jpeg", "jpg"}:
        return "jpg"
    if file_format == "webp":
        return "webp"
    if file_format in {"tif", "tiff"}:
        return "tiff"
    return "png"


def _metadata_pnginfo(prompt=None, extra_pnginfo=None):
    metadata = PngInfo()
    if prompt is not None:
        metadata.add_text("prompt", json.dumps(prompt))
    if extra_pnginfo is not None:
        for key, value in extra_pnginfo.items():
            metadata.add_text(key, json.dumps(value))
    return metadata


def _save_image(image, path, file_format, quality=95, embed_metadata=True, prompt=None, extra_pnginfo=None):
    file_format = _extension_for_save_format(file_format)
    quality = max(1, min(100, int(quality)))
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if file_format == "png":
        metadata = _metadata_pnginfo(prompt, extra_pnginfo) if embed_metadata else None
        image.save(path, format="PNG", pnginfo=metadata, compress_level=4)
    elif file_format == "jpg":
        image.convert("RGB").save(path, format="JPEG", quality=quality, subsampling=0, optimize=False)
    elif file_format == "webp":
        image.convert("RGB").save(path, format="WEBP", quality=quality, method=6)
    elif file_format == "tiff":
        image.save(path, format="TIFF", compression="tiff_lzw")
    else:
        raise ValueError(f"Unsupported large image save format: {file_format}")


class NH_SaveLargeImage:
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "NH_Large", "multiline": False}),
                "file_format": (["png", "jpg", "webp", "tiff"], {"default": "png"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "embed_metadata": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("image", "file_path", "saved_count")
    FUNCTION = "save"
    CATEGORY = "NH-Nodes/Image"

    def save(self, image, filename_prefix="NH_Large", file_format="png", quality=95, embed_metadata=True, prompt=None, extra_pnginfo=None):
        extension = _extension_for_save_format(file_format)
        # Kiểm tra giới hạn trước khi ghi bất kỳ file nào (mọi ảnh trong batch cùng kích thước), kể cả tạo thư mục.
        _check_size_limit(extension, int(image.shape[2]), int(image.shape[1]))
        # --disable-metadata của ComfyUI thắng ô embed_metadata: không nhúng prompt/workflow.
        embed_metadata = bool(embed_metadata) and not args.disable_metadata
        first = image[0]
        full_output_folder, filename, counter, _subfolder, _filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            self.output_dir,
            first.shape[1],
            first.shape[0],
        )

        saved_paths = []
        for batch_number, image_tensor in enumerate(image):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file_name = f"{filename_with_batch_num}_{counter:05}_.{extension}"
            file_path = os.path.join(full_output_folder, file_name)
            pil_image = tensor_to_pil(image_tensor, rgb=True, keep_alpha=True)
            _save_image(pil_image, file_path, extension, quality, embed_metadata, prompt, extra_pnginfo)
            saved_paths.append(file_path)
            counter += 1

        return {
            "ui": {
                "nh_large_saved": [
                    {
                        "path": path,
                        "filename": os.path.basename(path),
                    }
                    for path in saved_paths
                ],
            },
            "result": (image, "\n".join(saved_paths), len(saved_paths)),
        }


NODE_CLASS_MAPPINGS = {
    "NH_SaveLargeImage": NH_SaveLargeImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_SaveLargeImage": "NH Save Large Image",
}
