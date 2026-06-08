"""Thumbnail-only large image preview node for NH-Nodes."""

import json

from .large_preview_cache import create_preview_cache


class NH_LargeImagePreview:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preview_megapixels": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 8.0, "step": 0.05}),
                "preview_format": (["jpg", "webp", "png"], {"default": "jpg"}),
                "preview_quality": ("INT", {"default": 90, "min": 1, "max": 100, "step": 1}),
                "viewer_mode": (["modal", "side_panel", "new_tab"], {"default": "modal"}),
                "title": ("STRING", {"default": "NH Large Image Preview", "multiline": False}),
            },
            "optional": {
                "tile_size": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
                "tile_format": (["jpg", "webp"], {"default": "jpg"}),
                "tile_quality": ("INT", {"default": 90, "min": 1, "max": 100, "step": 1}),
                "cache_policy": (["temp"], {"default": "temp"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "manifest")
    FUNCTION = "preview"
    CATEGORY = "NH-Nodes/Image"

    def preview(
        self,
        image,
        preview_megapixels=1.0,
        preview_format="jpg",
        preview_quality=90,
        viewer_mode="modal",
        title="NH Large Image Preview",
        tile_size=512,
        tile_format="jpg",
        tile_quality=90,
        cache_policy="temp",
    ):
        batch_count = int(image.shape[0])
        manifests = []
        ui_images = []

        for batch_index in range(batch_count):
            manifest = create_preview_cache(
                image[batch_index],
                preview_megapixels=preview_megapixels,
                preview_format=preview_format,
                preview_quality=preview_quality,
                tile_size=tile_size,
                tile_format=tile_format,
                tile_quality=tile_quality,
                viewer_mode=viewer_mode,
                cache_policy=cache_policy,
                title=title,
                batch_index=batch_index,
                batch_count=batch_count,
            )
            manifests.append(manifest)
            ui_images.append(manifest["thumb"])

        manifest_json = json.dumps(manifests, ensure_ascii=False)
        return {
            "ui": {
                "images": ui_images,
                "nh_large_preview": manifests,
            },
            "result": (image, manifest_json),
        }


NODE_CLASS_MAPPINGS = {
    "NH_LargeImagePreview": NH_LargeImagePreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_LargeImagePreview": "NH Large Image Preview",
}
