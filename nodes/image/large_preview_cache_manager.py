"""Cache manager node for NH large preview cache."""

import json

from ...core.large_preview_cache import cache_info, cleanup_cache, delete_cache


class NH_LargePreviewCacheManager:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["info", "cleanup_by_age", "enforce_size_limit", "cleanup_age_and_size", "delete_cache_id"], {"default": "info"}),
                "max_age_hours": ("FLOAT", {"default": 24.0, "min": 0.0, "max": 8760.0, "step": 0.5}),
                "max_total_mb": ("FLOAT", {"default": 2048.0, "min": 1.0, "max": 1048576.0, "step": 64.0}),
                "cache_id": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("summary", "removed", "freed_mb", "cache_root")
    FUNCTION = "run"
    CATEGORY = "NH-Nodes/Image"

    def run(self, action="info", max_age_hours=24.0, max_total_mb=2048.0, cache_id=""):
        action = str(action or "info")
        before = cache_info()
        cleanup_result = {"removed": 0, "freed_bytes": 0, "root": before["root"]}

        if action == "cleanup_by_age":
            cleanup_result = cleanup_cache(max_age_hours=max_age_hours, max_total_mb=0)
        elif action == "enforce_size_limit":
            cleanup_result = cleanup_cache(max_age_hours=None, max_total_mb=max_total_mb)
        elif action == "cleanup_age_and_size":
            cleanup_result = cleanup_cache(max_age_hours=max_age_hours, max_total_mb=max_total_mb)
        elif action == "delete_cache_id":
            if not str(cache_id or "").strip():
                raise ValueError("cache_id is required for delete_cache_id")
            cleanup_result = delete_cache(cache_id.strip())
        elif action != "info":
            raise ValueError(f"Unsupported cache manager action: {action}")

        after = cache_info()
        summary = {
            "action": action,
            "before": before,
            "cleanup": cleanup_result,
            "after": after,
        }
        freed_mb = round(float(cleanup_result.get("freed_bytes", 0)) / (1024 * 1024), 3)
        result = (
            json.dumps(summary, ensure_ascii=False, indent=2),
            int(cleanup_result.get("removed", 0)),
            freed_mb,
            after["root"],
        )
        return {
            "ui": {
                "nh_large_cache": [
                    {
                        "action": action,
                        "removed": int(cleanup_result.get("removed", 0)),
                        "freed_mb": freed_mb,
                        "cache_count": int(after["cache_count"]),
                        "total_mb": float(after["total_mb"]),
                        "root": after["root"],
                    }
                ],
            },
            "result": result,
        }


NODE_CLASS_MAPPINGS = {
    "NH_LargePreviewCacheManager": NH_LargePreviewCacheManager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_LargePreviewCacheManager": "NH Large Preview Cache Manager",
}
