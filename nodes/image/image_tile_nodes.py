import numpy as np
import torch
import torch.nn.functional as F


def _positive_int(value, default=1):
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _round_to_multiple(value, multiple):
    return max(multiple, int(round(value / multiple)) * multiple)


def _tile_size_for_mode(image_width, image_height, tile_mode, tile_size, tile_width, tile_height):
    image_width = _positive_int(image_width)
    image_height = _positive_int(image_height)

    if tile_mode == "custom":
        width = _positive_int(tile_width, 512)
        height = _positive_int(tile_height, 512)
    elif tile_mode == "square":
        width = height = _positive_int(tile_size, 1024)
    else:
        long_side = _positive_int(tile_size, 1024)
        if image_width >= image_height:
            width = long_side
            height = max(1, round(long_side * image_height / image_width))
        else:
            height = long_side
            width = max(1, round(long_side * image_width / image_height))

    return width, height


def _auto_overlap_for_axis(image_size, tile_size):
    if image_size <= tile_size:
        return 0

    raw = tile_size * 0.125
    lower = 16 if tile_size >= 128 else max(1, tile_size // 8)
    overlap = _round_to_multiple(max(lower, min(raw, 128)), 8)
    return min(overlap, max(0, (tile_size - 1) // 2))


def _axis_positions(image_size, tile_size, overlap):
    if image_size <= tile_size:
        return [0]

    step = max(1, tile_size - overlap)
    last_position = image_size - tile_size
    positions = []
    current = 0

    while current < last_position:
        positions.append(current)
        current += step

    positions.append(last_position)
    deduped = []
    for position in positions:
        position = int(position)
        if not deduped or deduped[-1] != position:
            deduped.append(position)
    return deduped


def _extract_tile(image, batch_index, x, y, tile_width, tile_height):
    image_height = int(image.shape[1])
    image_width = int(image.shape[2])
    crop_width = max(1, min(tile_width, image_width - x))
    crop_height = max(1, min(tile_height, image_height - y))
    tile = image[batch_index:batch_index + 1, y:y + crop_height, x:x + crop_width, :]
    pad_right = tile_width - crop_width
    pad_bottom = tile_height - crop_height

    if pad_right <= 0 and pad_bottom <= 0:
        return tile, crop_width, crop_height

    tile = tile.movedim(-1, 1)
    tile = F.pad(tile, (0, pad_right, 0, pad_bottom), mode="replicate")
    return tile.movedim(1, -1), crop_width, crop_height


def _tile_weight(width, height, x, y, image_width, image_height, overlap_x, overlap_y):
    weight_x = np.ones((width,), dtype=np.float32)
    weight_y = np.ones((height,), dtype=np.float32)

    if overlap_x > 0 and x > 0:
        fade = min(overlap_x, width)
        weight_x[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
    if overlap_x > 0 and x + width < image_width:
        fade = min(overlap_x, width)
        weight_x[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

    if overlap_y > 0 and y > 0:
        fade = min(overlap_y, height)
        weight_y[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
    if overlap_y > 0 and y + height < image_height:
        fade = min(overlap_y, height)
        weight_y[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

    return weight_y[:, None, None] * weight_x[None, :, None]


def _infer_processed_scale(tiles, image_entries):
    first_entry = None
    for image_entry in image_entries:
        if image_entry["tiles"]:
            first_entry = image_entry["tiles"][0]
            break

    if first_entry is None:
        raise ValueError("Tile data does not contain any tile entries")

    original_tile_width = int(first_entry["width"])
    original_tile_height = int(first_entry["height"])
    processed_tile_height = int(tiles.shape[1])
    processed_tile_width = int(tiles.shape[2])
    scale_x = processed_tile_width / max(1, original_tile_width)
    scale_y = processed_tile_height / max(1, original_tile_height)
    return scale_x, scale_y


class NH_ImageTile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_mode": (["original_ratio", "custom", "square"],),
                "tile_size": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "tile_width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "tile_height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE", "NH_TILE_DATA", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("tiles", "tile_data", "tile_count", "overlap_x", "overlap_y", "tile_info")
    FUNCTION = "tile"
    CATEGORY = "NH-Nodes/Image"
    SEARCH_ALIASES = ["tile image", "split image tile", "image tiling"]

    def tile(self, image, tile_mode, tile_size, tile_width, tile_height):
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("NH Image Tile expects IMAGE tensor with shape [B, H, W, C]")

        batch_size, image_height, image_width, channels = image.shape
        tile_width, tile_height = _tile_size_for_mode(
            int(image_width),
            int(image_height),
            tile_mode,
            tile_size,
            tile_width,
            tile_height,
        )
        overlap_x = _auto_overlap_for_axis(int(image_width), tile_width)
        overlap_y = _auto_overlap_for_axis(int(image_height), tile_height)
        x_positions = _axis_positions(int(image_width), tile_width, overlap_x)
        y_positions = _axis_positions(int(image_height), tile_height, overlap_y)

        tile_tensors = []
        image_entries = []
        all_tile_entries = []

        for batch_index in range(int(batch_size)):
            tile_entries = []
            for y in y_positions:
                for x in x_positions:
                    tile_tensor, crop_width, crop_height = _extract_tile(
                        image,
                        batch_index,
                        int(x),
                        int(y),
                        int(tile_width),
                        int(tile_height),
                    )
                    tile_tensors.append(tile_tensor)
                    entry = {
                        "batch_index": batch_index,
                        "x": int(x),
                        "y": int(y),
                        "width": int(tile_width),
                        "height": int(tile_height),
                        "crop_width": int(crop_width),
                        "crop_height": int(crop_height),
                    }
                    tile_entries.append(entry)
                    all_tile_entries.append(entry)

            image_entries.append({
                "width": int(image_width),
                "height": int(image_height),
                "tile_count": len(tile_entries),
                "tiles": tile_entries,
            })

        tiles = torch.cat(tile_tensors, dim=0)
        tile_data = {
            "version": 1,
            "tile_mode": tile_mode,
            "tile_width": int(tile_width),
            "tile_height": int(tile_height),
            "overlap_x": int(overlap_x),
            "overlap_y": int(overlap_y),
            "batch_size": int(batch_size),
            "channels": int(channels),
            "images": image_entries,
        }

        tile_count = len(all_tile_entries)
        info_lines = [
            f"mode={tile_mode}, image={int(image_width)}x{int(image_height)}, tile={tile_width}x{tile_height}",
            f"grid={len(x_positions)}x{len(y_positions)}, overlap={overlap_x}x{overlap_y}, tiles={tile_count}",
        ]
        return (tiles, tile_data, tile_count, int(overlap_x), int(overlap_y), "\n".join(info_lines))


class NH_ImageUntile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tile_data": ("NH_TILE_DATA",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "untile_info")
    FUNCTION = "untile"
    CATEGORY = "NH-Nodes/Image"
    SEARCH_ALIASES = ["untile image", "merge tiles", "stitch tiles"]

    def untile(self, tiles, tile_data):
        if not isinstance(tiles, torch.Tensor) or tiles.ndim != 4:
            raise ValueError("NH Image Untile expects IMAGE tiles with shape [B, H, W, C]")
        if not isinstance(tile_data, dict) or "images" not in tile_data:
            raise ValueError("Invalid NH_TILE_DATA input")

        image_entries = tile_data["images"]
        expected_tiles = sum(int(entry["tile_count"]) for entry in image_entries)
        if int(tiles.shape[0]) < expected_tiles:
            raise ValueError(f"Not enough tiles: got {int(tiles.shape[0])}, expected {expected_tiles}")

        widths = {int(entry["width"]) for entry in image_entries}
        heights = {int(entry["height"]) for entry in image_entries}
        if len(widths) != 1 or len(heights) != 1:
            raise ValueError("Untile currently requires all original images in the batch to have the same size")

        scale_x, scale_y = _infer_processed_scale(tiles, image_entries)
        overlap_x = max(0, int(round(int(tile_data.get("overlap_x", 0)) * scale_x)))
        overlap_y = max(0, int(round(int(tile_data.get("overlap_y", 0)) * scale_y)))
        channels = int(tiles.shape[-1])
        outputs = []
        tile_index = 0

        for image_entry in image_entries:
            source_image_width = int(image_entry["width"])
            source_image_height = int(image_entry["height"])
            image_width = max(1, int(round(source_image_width * scale_x)))
            image_height = max(1, int(round(source_image_height * scale_y)))
            accum = np.zeros((image_height, image_width, channels), dtype=np.float32)
            weights = np.zeros((image_height, image_width, 1), dtype=np.float32)

            for tile_entry in image_entry["tiles"]:
                x = int(round(int(tile_entry["x"]) * scale_x))
                y = int(round(int(tile_entry["y"]) * scale_y))
                tile_array = tiles[tile_index].detach().cpu().numpy().astype(np.float32)
                tile_index += 1
                width = min(
                    int(tile_array.shape[1]),
                    max(1, int(round(int(tile_entry.get("crop_width", tile_entry["width"])) * scale_x))),
                )
                height = min(
                    int(tile_array.shape[0]),
                    max(1, int(round(int(tile_entry.get("crop_height", tile_entry["height"])) * scale_y))),
                )

                left = max(0, x)
                top = max(0, y)
                right = min(image_width, x + width)
                bottom = min(image_height, y + height)
                if right <= left or bottom <= top:
                    continue

                weight = _tile_weight(width, height, x, y, image_width, image_height, overlap_x, overlap_y)
                crop_left = left - x
                crop_top = top - y
                crop_right = crop_left + (right - left)
                crop_bottom = crop_top + (bottom - top)
                accum[top:bottom, left:right, :] += (
                    tile_array[crop_top:crop_bottom, crop_left:crop_right, :]
                    * weight[crop_top:crop_bottom, crop_left:crop_right, :]
                )
                weights[top:bottom, left:right, :] += weight[crop_top:crop_bottom, crop_left:crop_right, :]

            output = accum / np.maximum(weights, 1e-8)
            outputs.append(torch.from_numpy(output).to(device=tiles.device, dtype=tiles.dtype).unsqueeze(0))

        image = torch.cat(outputs, dim=0)
        info = (
            f"images={len(outputs)}, tiles={expected_tiles}, "
            f"scale={scale_x:.4g}x{scale_y:.4g}, overlap={overlap_x}x{overlap_y}"
        )
        return (image, info)


NODE_CLASS_MAPPINGS = {
    "NH_ImageTile": NH_ImageTile,
    "NH_ImageUntile": NH_ImageUntile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_ImageTile": "Image Tile (NH)",
    "NH_ImageUntile": "Image Untile (NH)",
}
