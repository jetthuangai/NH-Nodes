"""HTTP routes for NH large image preview cache."""

from aiohttp import web

from .large_preview_cache import cache_info, cleanup_cache, delete_cache, get_thumb_path, get_tile_path, read_manifest


try:
    from server import PromptServer

    @PromptServer.instance.routes.get("/nh-nodes/large-preview/manifest/{cache_id}")
    async def nh_large_preview_manifest(request):
        try:
            manifest = read_manifest(request.match_info["cache_id"])
            return web.json_response(manifest)
        except FileNotFoundError:
            return web.json_response({"error": "manifest not found"}, status=404)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @PromptServer.instance.routes.get("/nh-nodes/large-preview/thumb/{cache_id}")
    async def nh_large_preview_thumb(request):
        try:
            return web.FileResponse(get_thumb_path(request.match_info["cache_id"]))
        except FileNotFoundError:
            return web.json_response({"error": "thumbnail not found"}, status=404)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @PromptServer.instance.routes.get("/nh-nodes/large-preview/tile/{cache_id}/{level}/{tile}")
    async def nh_large_preview_tile(request):
        try:
            return web.FileResponse(get_tile_path(
                request.match_info["cache_id"],
                request.match_info["level"],
                request.match_info["tile"],
            ))
        except FileNotFoundError:
            return web.json_response({"error": "tile not available in thumbnail-only preview"}, status=404)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @PromptServer.instance.routes.get("/nh-nodes/large-preview/cache-info")
    async def nh_large_preview_cache_info(request):
        try:
            return web.json_response(cache_info())
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @PromptServer.instance.routes.post("/nh-nodes/large-preview/cleanup")
    async def nh_large_preview_cleanup(request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        try:
            cache_id = payload.get("cache_id", "")
            if cache_id:
                result = delete_cache(cache_id)
            else:
                result = cleanup_cache(payload.get("max_age_hours", 24), payload.get("max_total_mb", 0))
            return web.json_response(result)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
except Exception as exc:
    print(f"[NH-Nodes] Failed to register large preview routes: {exc}")


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
