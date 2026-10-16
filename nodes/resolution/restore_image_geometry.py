"""Restore a processed resize result to its original image coordinates."""

import math

import torch
import torch.nn.functional as F


class NH_RestoreImageGeometry:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "original_image": ("IMAGE",),
                "resize_data": ("NH_RESIZE_DATA",),
                "resize_method": (["bicubic", "bilinear", "nearest"], {"default": "bicubic"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "restore"
    CATEGORY = "NH-Nodes/Resolution"
    DESCRIPTION = (
        "Place a processed or upscaled image back into its original frame using resize_data. "
        "White mask pixels mark restored content; black pixels retain the original image. "
        "Processing must preserve the resized image's framing."
    )

    def restore(self, image, original_image, resize_data, resize_method="bicubic"):
        if not isinstance(resize_data, dict) or resize_data.get("version") != 1:
            raise ValueError("Use version-1 resize_data from an NH image resize node.")
        required = (
            "source_width", "source_height", "target_width", "target_height",
            "resized_width", "resized_height", "crop_left", "crop_top",
            "pad_left", "pad_top", "resize_mode",
        )
        missing = [key for key in required if key not in resize_data]
        if missing:
            raise ValueError(f"resize_data is missing: {', '.join(missing)}")
        source_width, source_height, target_width, target_height, resized_width, resized_height = (
            int(resize_data[key]) for key in required[:6]
        )
        if min(source_width, source_height, target_width, target_height, resized_width, resized_height) <= 0:
            raise ValueError("resize_data image dimensions must be positive.")
        if tuple(original_image.shape[1:3]) != (source_height, source_width):
            raise ValueError(
                f"original_image must match resize_data source size {source_width}x{source_height}."
            )
        if image.shape[-1] != original_image.shape[-1]:
            raise ValueError("image and original_image must have the same number of channels.")
        if image.shape[0] != original_image.shape[0] and min(image.shape[0], original_image.shape[0]) != 1:
            raise ValueError("Image batches must match, or one batch must contain a single image.")
        mode = resize_data["resize_mode"]
        if mode not in {"cover_crop", "contain_pad", "stretch"}:
            raise ValueError(f"Unsupported resize_mode in resize_data: {mode}")
        if resize_method not in {"bicubic", "bilinear", "nearest"}:
            raise ValueError(f"Unsupported resize_method: {resize_method}")

        image_height, image_width = image.shape[1:3]
        crop_left, crop_top = int(resize_data["crop_left"]), int(resize_data["crop_top"])
        pad_left, pad_top = int(resize_data["pad_left"]), int(resize_data["pad_top"])
        sample_left, sample_top, sample_right, sample_bottom = 0, 0, image_width, image_height
        if mode == "contain_pad":
            # Exclude padding samples without rounding the transform's content bounds.
            sample_left = max(0, min(image_width - 1, math.ceil(pad_left * image_width / target_width - 0.5)))
            sample_top = max(0, min(image_height - 1, math.ceil(pad_top * image_height / target_height - 0.5)))
            sample_right = max(sample_left + 1, min(image_width, math.ceil((pad_left + resized_width) * image_width / target_width - 0.5)))
            sample_bottom = max(sample_top + 1, min(image_height, math.ceil((pad_top + resized_height) * image_height / target_height - 0.5)))

        batch = max(image.shape[0], original_image.shape[0])
        device = original_image.device
        compute_dtype = torch.float64 if torch.float64 in (image.dtype, original_image.dtype) else torch.float32
        samples = image[:, sample_top:sample_bottom, sample_left:sample_right].movedim(-1, 1)
        samples = samples.to(device=device, dtype=compute_dtype).expand(batch, -1, -1, -1)
        result = original_image.expand(batch, -1, -1, -1).clone()
        mask = torch.zeros((batch, source_height, source_width), device=device, dtype=torch.float32)
        # Decide coverage in source coordinates, avoiding float32 rounding at crop boundaries.
        valid_left = max(0, min(source_width, math.ceil((crop_left - pad_left) * source_width / resized_width - 0.5)))
        valid_top = max(0, min(source_height, math.ceil((crop_top - pad_top) * source_height / resized_height - 0.5)))
        valid_right = max(0, min(source_width, math.ceil((target_width + crop_left - pad_left) * source_width / resized_width - 0.5)))
        valid_bottom = max(0, min(source_height, math.ceil((target_height + crop_top - pad_top) * source_height / resized_height - 0.5)))
        mask[:, valid_top:valid_bottom, valid_left:valid_right] = 1.0

        # Pixel centers travel through the rounded intermediate image, then the crop/pad offset.
        target_x = (torch.arange(source_width, device=device, dtype=compute_dtype) + 0.5) * (resized_width / source_width)
        target_x = target_x + pad_left - crop_left
        grid_x = 2.0 * (target_x * (image_width / target_width) - sample_left) / (sample_right - sample_left) - 1.0
        # Bound temporary grids for large originals instead of allocating a full-resolution grid.
        chunk_rows = max(1, 1_048_576 // source_width)
        for top in range(0, source_height, chunk_rows):
            bottom = min(top + chunk_rows, source_height)
            target_y = (torch.arange(top, bottom, device=device, dtype=compute_dtype) + 0.5) * (resized_height / source_height)
            target_y = target_y + pad_top - crop_top
            grid_y = 2.0 * (target_y * (image_height / target_height) - sample_top) / (sample_bottom - sample_top) - 1.0
            grid = torch.empty((1, bottom - top, source_width, 2), device=device, dtype=compute_dtype)
            grid[0, :, :, 0] = grid_x.unsqueeze(0)
            grid[0, :, :, 1] = grid_y.unsqueeze(1)
            restored = F.grid_sample(
                samples, grid.expand(batch, -1, -1, -1), mode=resize_method,
                padding_mode="border", align_corners=False,
            ).clamp(0.0, 1.0).movedim(1, -1).to(dtype=original_image.dtype)
            valid = mask[0, top:bottom].bool()
            result[:, top:bottom] = torch.where(valid.unsqueeze(-1), restored, result[:, top:bottom])

        return (result, mask)


NODE_CLASS_MAPPINGS = {"NH_RestoreImageGeometry": NH_RestoreImageGeometry}
NODE_DISPLAY_NAME_MAPPINGS = {"NH_RestoreImageGeometry": "NH Restore Image Geometry"}
