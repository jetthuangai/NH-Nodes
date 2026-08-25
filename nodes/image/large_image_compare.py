"""Large image compare node for NH-Nodes."""

import json

from ...core.large_preview_cache import create_compare_thumbnail, create_preview_cache


def _match_batches(image_a, image_b):
    count_a = int(image_a.shape[0])
    count_b = int(image_b.shape[0])

    if count_a == count_b:
        return image_a, image_b
    if count_a == 1:
        return image_a.repeat(count_b, 1, 1, 1), image_b
    if count_b == 1:
        return image_a, image_b.repeat(count_a, 1, 1, 1)

    shared = min(count_a, count_b)
    print(f"[NH-Nodes] Large Image Compare: batch mismatch ({count_a} vs {count_b}), truncating to {shared}")
    return image_a[:shared], image_b[:shared]


class NH_LargeImageCompare:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "compare_mode": (["slider", "side_by_side", "difference"], {"default": "slider"}),
                "sync_pan_zoom": ("BOOLEAN", {"default": True}),
                "preview_megapixels": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 8.0, "step": 0.05}),
                "preview_format": (["jpg", "webp", "png"], {"default": "jpg"}),
                "preview_quality": ("INT", {"default": 90, "min": 1, "max": 100, "step": 1}),
                "title": ("STRING", {"default": "NH Large Image Compare", "multiline": False}),
            },
            "optional": {
                "tile_size": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
                "tile_format": (["jpg", "webp"], {"default": "jpg"}),
                "tile_quality": ("INT", {"default": 90, "min": 1, "max": 100, "step": 1}),
                "cache_policy": (["temp"], {"default": "temp"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image_a", "image_b", "manifest")
    FUNCTION = "compare"
    CATEGORY = "NH-Nodes/Image"

    def compare(
        self,
        image_a,
        image_b,
        compare_mode="slider",
        sync_pan_zoom=True,
        preview_megapixels=1.0,
        preview_format="jpg",
        preview_quality=90,
        title="NH Large Image Compare",
        tile_size=512,
        tile_format="jpg",
        tile_quality=90,
        cache_policy="temp",
    ):
        matched_a, matched_b = _match_batches(image_a, image_b)
        batch_count = int(matched_a.shape[0])
        compare_manifests = []
        ui_images = []

        for batch_index in range(batch_count):
            item_title = title or "NH Large Image Compare"
            manifest_a = create_preview_cache(
                matched_a[batch_index],
                preview_megapixels=preview_megapixels,
                preview_format=preview_format,
                preview_quality=preview_quality,
                tile_size=tile_size,
                tile_format=tile_format,
                tile_quality=tile_quality,
                viewer_mode="compare",
                cache_policy=cache_policy,
                title=f"{item_title} A",
                batch_index=batch_index,
                batch_count=batch_count,
            )
            manifest_b = create_preview_cache(
                matched_b[batch_index],
                preview_megapixels=preview_megapixels,
                preview_format=preview_format,
                preview_quality=preview_quality,
                tile_size=tile_size,
                tile_format=tile_format,
                tile_quality=tile_quality,
                viewer_mode="compare",
                cache_policy=cache_policy,
                title=f"{item_title} B",
                batch_index=batch_index,
                batch_count=batch_count,
            )
            compare_thumb = create_compare_thumbnail(
                manifest_a,
                manifest_b,
                compare_mode=compare_mode,
                preview_format=preview_format,
                preview_quality=preview_quality,
            )
            compare_manifest = {
                "version": 1,
                "type": "compare",
                "title": item_title,
                "compare_mode": compare_mode,
                "sync_pan_zoom": bool(sync_pan_zoom),
                "batch_index": int(batch_index),
                "batch_count": int(batch_count),
                "width": int(max(manifest_a["width"], manifest_b["width"])),
                "height": int(max(manifest_a["height"], manifest_b["height"])),
                "thumb": compare_thumb,
                "a": manifest_a,
                "b": manifest_b,
            }
            compare_manifests.append(compare_manifest)
            ui_images.append(compare_thumb)

        manifest_json = json.dumps(compare_manifests, ensure_ascii=False)
        return {
            "ui": {
                "images": ui_images,
                "nh_large_compare": compare_manifests,
            },
            "result": (matched_a, matched_b, manifest_json),
        }


NODE_CLASS_MAPPINGS = {
    "NH_LargeImageCompare": NH_LargeImageCompare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_LargeImageCompare": "NH Large Image Compare",
}
