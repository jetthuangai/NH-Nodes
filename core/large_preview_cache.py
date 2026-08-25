"""Cache helpers for NH large image preview nodes."""

import json
import math
import os
import re
import shutil
import time
import uuid
from pathlib import Path

import folder_paths
import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw


CACHE_SUBDIR = "nh_large_preview"
CACHE_ID_RE = re.compile(r"^[a-f0-9]{32}$")
TILE_NAME_RE = re.compile(r"^[0-9]+_[0-9]+\.(?:jpg|jpeg|webp|png)$", re.IGNORECASE)
_RESAMPLING = getattr(Image, "Resampling", Image)


def validate_cache_id(cache_id):
    if not CACHE_ID_RE.match(str(cache_id or "")):
        raise ValueError("Invalid NH large preview cache id")
    return str(cache_id)


def validate_level(level):
    level_text = "" if level is None else str(level)
    if not level_text.isdigit():
        raise ValueError("Invalid NH large preview level")
    return int(level_text)


def validate_tile_name(tile_name):
    tile_name = str(tile_name or "")
    if not TILE_NAME_RE.match(tile_name):
        raise ValueError("Invalid NH large preview tile name")
    return tile_name


def get_cache_root(cache_policy="temp"):
    # Phase 1 keeps all preview cache under ComfyUI temp. The policy is stored
    # in the manifest so Phase 2 can add session/persistent roots cleanly.
    root = Path(folder_paths.get_temp_directory()) / CACHE_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_cache_dir(cache_id, cache_policy="temp"):
    cache_id = validate_cache_id(cache_id)
    root = get_cache_root(cache_policy).resolve()
    cache_dir = (root / cache_id).resolve()
    try:
        cache_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("Resolved NH preview cache path escaped cache root") from exc
    return cache_dir


def _directory_size(path):
    size = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        try:
            size += item.stat().st_size
        except OSError:
            pass
    return size


def _cache_items(root):
    items = []
    if not root.exists():
        return items

    for item in root.iterdir():
        if not item.is_dir() or not CACHE_ID_RE.match(item.name):
            continue
        try:
            stat = item.stat()
        except OSError:
            continue
        items.append({
            "cache_id": item.name,
            "path": item,
            "mtime": float(stat.st_mtime),
            "bytes": int(_directory_size(item)),
        })
    return items


def cache_info():
    root = get_cache_root().resolve()
    items = _cache_items(root)
    total_bytes = sum(item["bytes"] for item in items)
    now = time.time()
    sorted_items = sorted(items, key=lambda item: item["mtime"])
    return {
        "root": str(root),
        "cache_count": len(items),
        "total_bytes": int(total_bytes),
        "total_mb": round(total_bytes / (1024 * 1024), 3),
        "oldest_age_hours": round((now - sorted_items[0]["mtime"]) / 3600, 3) if sorted_items else 0,
        "newest_age_hours": round((now - sorted_items[-1]["mtime"]) / 3600, 3) if sorted_items else 0,
        "items": [
            {
                "cache_id": item["cache_id"],
                "bytes": int(item["bytes"]),
                "mb": round(item["bytes"] / (1024 * 1024), 3),
                "age_hours": round((now - item["mtime"]) / 3600, 3),
            }
            for item in sorted_items
        ],
    }


def delete_cache(cache_id):
    cache_dir = get_cache_dir(cache_id).resolve()
    if not cache_dir.exists():
        return {"removed": 0, "freed_bytes": 0, "cache_id": validate_cache_id(cache_id), "path": str(cache_dir)}

    root = get_cache_root().resolve()
    try:
        cache_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("Resolved NH preview cache path escaped cache root") from exc

    freed_bytes = _directory_size(cache_dir)
    shutil.rmtree(cache_dir, ignore_errors=True)
    return {
        "removed": 1,
        "freed_bytes": int(freed_bytes),
        "cache_id": validate_cache_id(cache_id),
        "path": str(cache_dir),
    }


def _tensor_to_pil(image_tensor):
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError(f"Expected IMAGE tensor, got {type(image_tensor).__name__}")

    array = image_tensor.detach().cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return Image.fromarray(array).convert("RGB")
    if array.ndim == 3 and array.shape[-1] == 1:
        return Image.fromarray(array[:, :, 0]).convert("RGB")
    if array.ndim == 3 and array.shape[-1] == 4:
        return Image.fromarray(array).convert("RGBA")
    return Image.fromarray(array).convert("RGB")


def _preview_dimensions(width, height, preview_megapixels):
    width = max(1, int(width))
    height = max(1, int(height))
    max_pixels = max(1, int(float(preview_megapixels) * 1_000_000))
    if width * height <= max_pixels:
        return width, height

    scale = math.sqrt(max_pixels / float(width * height))
    preview_width = max(1, int(math.floor(width * scale)))
    preview_height = max(1, int(math.floor(height * scale)))
    while preview_width * preview_height > max_pixels:
        if preview_width >= preview_height and preview_width > 1:
            preview_width -= 1
        elif preview_height > 1:
            preview_height -= 1
        else:
            break
    return preview_width, preview_height


def _fit_to_canvas(image, size, fill=(16, 16, 16)):
    canvas = Image.new("RGB", size, fill)
    fitted = image.convert("RGB")
    fitted.thumbnail(size, _RESAMPLING.LANCZOS)
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def _extension_for_format(image_format):
    image_format = str(image_format or "jpg").lower()
    if image_format in {"jpeg", "jpg"}:
        return "jpg"
    if image_format == "webp":
        return "webp"
    if image_format == "png":
        return "png"
    return "jpg"


def _save_preview_image(image, path, image_format, quality):
    image_format = _extension_for_format(image_format)
    quality = max(1, min(100, int(quality)))
    path.parent.mkdir(parents=True, exist_ok=True)

    if image_format == "jpg":
        image.convert("RGB").save(path, format="JPEG", quality=quality, optimize=False)
    elif image_format == "webp":
        image.convert("RGB").save(path, format="WEBP", quality=quality, method=4)
    elif image_format == "png":
        image.save(path, format="PNG", compress_level=3)
    else:
        image.convert("RGB").save(path, format="JPEG", quality=quality, optimize=False)


def _build_level_sizes(width, height):
    sizes = []
    level_width = max(1, int(width))
    level_height = max(1, int(height))
    while True:
        sizes.append((level_width, level_height))
        if level_width <= 1 and level_height <= 1:
            break
        if max(level_width, level_height) <= 512:
            break
        level_width = max(1, int(math.ceil(level_width / 2)))
        level_height = max(1, int(math.ceil(level_height / 2)))
    return sizes


def _tile_level(level_image, level_index, cache_dir, tile_size, tile_format, tile_quality, source_width):
    level_width, level_height = level_image.size
    tile_size = max(64, int(tile_size))
    tile_format = _extension_for_format(tile_format)
    level_dir = cache_dir / f"level_{level_index}"
    level_dir.mkdir(parents=True, exist_ok=True)

    columns = int(math.ceil(level_width / float(tile_size)))
    rows = int(math.ceil(level_height / float(tile_size)))
    for tile_y in range(rows):
        top = tile_y * tile_size
        bottom = min(level_height, top + tile_size)
        for tile_x in range(columns):
            left = tile_x * tile_size
            right = min(level_width, left + tile_size)
            tile = level_image.crop((left, top, right, bottom))
            tile_path = level_dir / f"{tile_x}_{tile_y}.{tile_format}"
            _save_preview_image(tile, tile_path, tile_format, tile_quality)

    return {
        "level": int(level_index),
        "width": int(level_width),
        "height": int(level_height),
        "scale": float(level_width) / max(float(source_width), 1.0),
        "tile_size": int(tile_size),
        "columns": int(columns),
        "rows": int(rows),
    }


def _build_tile_pyramid(source, cache_dir, tile_size, tile_format, tile_quality):
    source_rgb = source.convert("RGB")
    source_width, source_height = source_rgb.size
    levels = []

    for level_index, (level_width, level_height) in enumerate(_build_level_sizes(source_width, source_height)):
        if level_index == 0:
            level_image = source_rgb
        else:
            level_image = source_rgb.resize((level_width, level_height), _RESAMPLING.LANCZOS)
        levels.append(_tile_level(
            level_image,
            level_index,
            cache_dir,
            tile_size,
            tile_format,
            tile_quality,
            source_width,
        ))

    return levels


def create_preview_cache(
    image_tensor,
    preview_megapixels=1.0,
    preview_format="jpg",
    preview_quality=90,
    tile_size=512,
    tile_format="jpg",
    tile_quality=90,
    viewer_mode="modal",
    cache_policy="temp",
    title="",
    batch_index=0,
    batch_count=1,
):
    source = _tensor_to_pil(image_tensor)
    source_width, source_height = source.size
    preview_width, preview_height = _preview_dimensions(source_width, source_height, preview_megapixels)
    preview = source.copy()
    if (preview_width, preview_height) != source.size:
        preview.thumbnail((preview_width, preview_height), _RESAMPLING.LANCZOS)

    cache_id = uuid.uuid4().hex
    cache_dir = get_cache_dir(cache_id, cache_policy)
    cache_dir.mkdir(parents=True, exist_ok=False)

    preview_ext = _extension_for_format(preview_format)
    thumb_filename = f"thumb.{preview_ext}"
    thumb_path = cache_dir / thumb_filename
    _save_preview_image(preview, thumb_path, preview_ext, preview_quality)
    tile_ext = _extension_for_format(tile_format)
    levels = _build_tile_pyramid(source, cache_dir, tile_size, tile_ext, tile_quality)

    subfolder = f"{CACHE_SUBDIR}/{cache_id}"
    manifest = {
        "version": 1,
        "phase": "tile-pyramid",
        "cache_id": cache_id,
        "cache_policy": cache_policy,
        "title": title or "NH Large Image Preview",
        "batch_index": int(batch_index),
        "batch_count": int(batch_count),
        "width": int(source_width),
        "height": int(source_height),
        "preview_width": int(preview.size[0]),
        "preview_height": int(preview.size[1]),
        "preview_megapixels": float(preview_megapixels),
        "preview_format": preview_ext,
        "preview_quality": int(preview_quality),
        "viewer_mode": viewer_mode,
        "thumb": {
            "filename": thumb_filename,
            "subfolder": subfolder,
            "type": "temp",
        },
        "thumb_url": f"/nh-nodes/large-preview/thumb/{cache_id}",
        "manifest_url": f"/nh-nodes/large-preview/manifest/{cache_id}",
        "tile_size": int(tile_size),
        "tile_format": tile_ext,
        "tile_quality": int(tile_quality),
        "levels": levels,
        "tile_url_template": f"/nh-nodes/large-preview/tile/{cache_id}/{{level}}/{{x}}_{{y}}.{tile_ext}",
    }

    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def get_manifest_path(cache_id):
    return get_cache_dir(cache_id) / "manifest.json"


def read_manifest(cache_id):
    path = get_manifest_path(cache_id)
    if not path.is_file():
        raise FileNotFoundError("NH large preview manifest not found")
    return json.loads(path.read_text(encoding="utf-8"))


def get_thumb_path(cache_id):
    manifest = read_manifest(cache_id)
    thumb_name = manifest.get("thumb", {}).get("filename", "")
    thumb_path = get_cache_dir(cache_id) / os.path.basename(thumb_name)
    if not thumb_path.is_file():
        raise FileNotFoundError("NH large preview thumbnail not found")
    return thumb_path


def create_compare_thumbnail(manifest_a, manifest_b, compare_mode="slider", preview_format="jpg", preview_quality=90):
    cache_id = validate_cache_id(manifest_a.get("cache_id"))
    thumb_a = Image.open(get_thumb_path(cache_id)).convert("RGB")
    thumb_b = Image.open(get_thumb_path(manifest_b.get("cache_id"))).convert("RGB")
    mode = str(compare_mode or "slider").lower().replace("-", "_")
    preview_ext = _extension_for_format(preview_format)

    if mode == "side_by_side":
        target_height = max(thumb_a.height, thumb_b.height)
        a = _fit_to_canvas(thumb_a, (thumb_a.width, target_height))
        b = _fit_to_canvas(thumb_b, (thumb_b.width, target_height))
        result = Image.new("RGB", (a.width + b.width, target_height), (16, 16, 16))
        result.paste(a, (0, 0))
        result.paste(b, (a.width, 0))
        draw = ImageDraw.Draw(result)
        draw.line((a.width, 0, a.width, target_height), fill=(230, 230, 230), width=2)
    else:
        size = (max(thumb_a.width, thumb_b.width), max(thumb_a.height, thumb_b.height))
        a = _fit_to_canvas(thumb_a, size)
        b = _fit_to_canvas(thumb_b, size)

        if mode == "difference":
            result = ImageChops.difference(a, b)
            result = result.point(lambda value: min(255, value * 3))
        else:
            split_x = size[0] // 2
            result = Image.new("RGB", size, (16, 16, 16))
            result.paste(a.crop((0, 0, split_x, size[1])), (0, 0))
            result.paste(b.crop((split_x, 0, size[0], size[1])), (split_x, 0))
            draw = ImageDraw.Draw(result)
            draw.line((split_x, 0, split_x, size[1]), fill=(230, 230, 230), width=2)

    thumb_filename = f"compare_thumb.{preview_ext}"
    cache_dir = get_cache_dir(cache_id)
    thumb_path = cache_dir / thumb_filename
    _save_preview_image(result, thumb_path, preview_ext, preview_quality)
    return {
        "filename": thumb_filename,
        "subfolder": f"{CACHE_SUBDIR}/{cache_id}",
        "type": "temp",
    }


def get_tile_path(cache_id, level, tile_name):
    level = validate_level(level)
    tile_name = validate_tile_name(tile_name)
    tile_path = get_cache_dir(cache_id) / f"level_{level}" / tile_name
    if not tile_path.is_file():
        raise FileNotFoundError("NH large preview tile not found")
    return tile_path


def cleanup_cache(max_age_hours=24, max_total_mb=0):
    root = get_cache_root().resolve()
    removed = 0
    freed_bytes = 0

    if max_age_hours is not None:
        max_age_seconds = max(0.0, float(max_age_hours)) * 3600.0
        cutoff = time.time() - max_age_seconds
        for item in _cache_items(root):
            if item["mtime"] > cutoff:
                continue
            shutil.rmtree(item["path"], ignore_errors=True)
            removed += 1
            freed_bytes += item["bytes"]

    max_total_bytes = max(0.0, float(max_total_mb or 0)) * 1024 * 1024
    if max_total_bytes > 0:
        remaining = _cache_items(root)
        total_bytes = sum(item["bytes"] for item in remaining)
        for item in sorted(remaining, key=lambda entry: entry["mtime"]):
            if total_bytes <= max_total_bytes:
                break
            shutil.rmtree(item["path"], ignore_errors=True)
            removed += 1
            freed_bytes += item["bytes"]
            total_bytes -= item["bytes"]

    return {"removed": removed, "freed_bytes": freed_bytes, "root": str(root)}
