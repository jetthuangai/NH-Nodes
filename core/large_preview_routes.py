"""HTTP routes for NH large image preview cache.

Importing this module registers the routes on ComfyUI's PromptServer. The handlers are plain module-level functions,
so a test can serve them from any aiohttp app through ``register_routes``.
"""

import asyncio
import json

from aiohttp import web

from .large_preview_cache import cache_info, cleanup_cache, delete_cache, get_thumb_path, get_tile_path, read_manifest


def _without_paths(value):
    """Answers go to the browser: drop absolute filesystem paths (the Cache Manager node still shows them)."""
    if isinstance(value, dict):
        return {key: _without_paths(item) for key, item in value.items() if key not in ("path", "root")}
    if isinstance(value, list):
        return [_without_paths(item) for item in value]
    return value


async def _read_json_object(request):
    """The JSON object in the request body ({} when the body is empty). ValueError for anything else."""
    try:
        text = await request.text()
        payload = json.loads(text) if text.strip() else {}
    except ValueError:  # not JSON, or not UTF-8 text: json.JSONDecodeError and UnicodeDecodeError both are ValueErrors
        raise ValueError("request body is not valid JSON") from None
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


async def nh_large_preview_manifest(request):
    try:
        manifest = read_manifest(request.match_info["cache_id"])
        return web.json_response(manifest)
    except FileNotFoundError:
        return web.json_response({"error": "manifest not found"}, status=404)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def nh_large_preview_thumb(request):
    try:
        return web.FileResponse(get_thumb_path(request.match_info["cache_id"]))
    except FileNotFoundError:
        return web.json_response({"error": "thumbnail not found"}, status=404)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)


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


async def nh_large_preview_cache_info(request):
    try:
        # Directory walks run in a worker thread so a large cache never stalls the server's event loop.
        return web.json_response(_without_paths(await asyncio.to_thread(cache_info)))
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def nh_large_preview_cleanup(request):
    try:
        payload = await _read_json_object(request)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    try:
        cache_id = payload.get("cache_id", "")
        if cache_id:
            result = await asyncio.to_thread(delete_cache, cache_id)
        else:
            result = await asyncio.to_thread(cleanup_cache, payload.get("max_age_hours", 24), payload.get("max_total_mb", 0))
        return web.json_response(_without_paths(result))
    except (ValueError, TypeError, OverflowError) as exc:  # a bad id, or a limit that is not a number
        return web.json_response({"error": str(exc)}, status=400)


def register_routes(routes):
    """Attach the handlers to an aiohttp RouteTableDef (ComfyUI's ``PromptServer.instance.routes``)."""
    routes.get("/nh-nodes/large-preview/manifest/{cache_id}")(nh_large_preview_manifest)
    routes.get("/nh-nodes/large-preview/thumb/{cache_id}")(nh_large_preview_thumb)
    routes.get("/nh-nodes/large-preview/tile/{cache_id}/{level}/{tile}")(nh_large_preview_tile)
    routes.get("/nh-nodes/large-preview/cache-info")(nh_large_preview_cache_info)
    routes.post("/nh-nodes/large-preview/cleanup")(nh_large_preview_cleanup)


try:
    from server import PromptServer

    register_routes(PromptServer.instance.routes)
except Exception as exc:
    print(f"[NH-Nodes] Failed to register large preview routes: {exc}")
