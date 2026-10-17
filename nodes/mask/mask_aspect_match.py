import torch
import numpy as np

def get_bounding_box_and_ratio(mask_tensor):
    """Tính toán bounding box và tỷ lệ (width / height) từ một mask tensor."""
    m_np = mask_tensor.cpu().numpy()
    rows, cols = np.where(m_np > 0.5)

    if len(rows) == 0:
        return None, 1.0 # Trả về None và tỷ lệ 1:1 nếu mask rỗng

    ymin, ymax = np.min(rows), np.max(rows)
    xmin, xmax = np.min(cols), np.max(cols)

    bbox = (xmin, ymin, xmax, ymax)
    width = xmax - xmin + 1
    height = ymax - ymin + 1

    aspect_ratio = width / height if height > 0 else 1.0
    return bbox, aspect_ratio


def _fit_window(start, length, limit):
    """Cửa sổ [start, start + length) trượt vào trong [0, limit); lớn hơn limit thì cắt ở mép cuối."""
    start = max(0, min(start, limit - length))
    return start, min(limit, start + length)


def _adjust_mask(mask_np, bbox, target_size, mode):
    """Chỉnh hộp (bbox) của một mask [H, W] về đúng tỷ lệ target_size = (rộng, cao).
    Kích thước hộp mới tính bằng số nguyên nên không bị lệch 1px do số thực. Ngoài hộp luôn là 0."""
    height, width = mask_np.shape
    xmin, ymin, xmax, ymax = (int(v) for v in bbox)
    cur_w, cur_h = xmax - xmin + 1, ymax - ymin + 1
    target_w, target_h = target_size
    target_is_wider = target_w * cur_h > cur_w * target_h
    output = np.zeros_like(mask_np)

    if mode in ("stretch", "pad"):
        # Nới một chiều: target rộng hơn thì nới ngang, ngược lại nới dọc
        new_w, new_h = cur_w, cur_h
        if target_is_wider:
            new_w = max(1, cur_h * target_w // target_h)
        else:
            new_h = max(1, cur_w * target_h // target_w)

        # Hộp mới nới đối xứng quanh hộp cũ. Chạm mép ảnh thì trượt vào trong để vẫn đủ kích thước.
        x0, x1 = _fit_window(xmin - (new_w - cur_w) // 2, new_w, width)
        y0, y1 = _fit_window(ymin - (new_h - cur_h) // 2, new_h, height)

        if mode == "stretch":
            # Kéo dãn nội dung (nearest) cho vừa hộp mới. Hộp lớn hơn cả ảnh thì chỉ hiện phần giữa.
            off_x = (new_w - (x1 - x0)) // 2
            off_y = (new_h - (y1 - y0)) // 2
            cols = (np.arange(x1 - x0) + off_x) * cur_w // new_w
            rows = (np.arange(y1 - y0) + off_y) * cur_h // new_h
            output[y0:y1, x0:x1] = mask_np[ymin:ymax + 1, xmin:xmax + 1][np.ix_(rows, cols)]
        else:
            # Pad: phần đệm thuộc vùng mask (1.0), nội dung gốc giữ nguyên vị trí cũ
            output[y0:y1, x0:x1] = 1.0
            output[ymin:ymax + 1, xmin:xmax + 1] = mask_np[ymin:ymax + 1, xmin:xmax + 1]

    elif mode == "crop":
        # Cắt bớt một chiều quanh tâm; phần giữ lại không bị dịch chỗ
        new_w, new_h = cur_w, cur_h
        if target_is_wider:
            new_h = max(1, cur_w * target_h // target_w)
        else:
            new_w = max(1, cur_h * target_w // target_h)
        x0 = xmin + (cur_w - new_w) // 2
        y0 = ymin + (cur_h - new_h) // 2
        output[y0:y0 + new_h, x0:x0 + new_w] = mask_np[y0:y0 + new_h, x0:x0 + new_w]

    else:
        raise ValueError(f"NH Mask Aspect Ratio Match: unknown mode '{mode}'")

    return output


class NH_MaskAspectRatioMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_ratio_mask": ("MASK",), # Mask A (Cái áo)
                "mask_to_adjust": ("MASK",),    # Mask B (Hộp vuông)
                # THÊM LỰA CHỌN MỚI Ở ĐÂY
                "mode": (["stretch", "pad", "crop"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "match_aspect_ratio"
    CATEGORY = "NH-Nodes/Mask"
    DESCRIPTION = (
        "Resizes the bounding box of mask_to_adjust to the aspect ratio of target_ratio_mask, around the same centre. "
        "stretch: scale the content to the new box. "
        "pad: keep the content and add mask pixels (1.0) around it until the box has the ratio. "
        "crop: keep the middle part of the content. "
        "A box that would leave the image slides back inside it. Every mask of the batch is processed; "
        "a shorter target batch repeats."
    )

    def match_aspect_ratio(self, target_ratio_mask, mask_to_adjust, mode):
        # Chấp nhận mask [H, W] hoặc [B, H, W]; xử lý từng mask trong batch
        targets = target_ratio_mask.reshape(-1, *target_ratio_mask.shape[-2:])
        masks = mask_to_adjust.reshape(-1, *mask_to_adjust.shape[-2:])
        if targets.shape[0] == 0 or masks.shape[0] == 0:
            return (mask_to_adjust,)

        adjusted = []
        for index in range(masks.shape[0]):
            # Batch của mask A ngắn hơn thì lặp lại vòng tròn
            bbox_a, _ = get_bounding_box_and_ratio(targets[index % targets.shape[0]])
            bbox_b, _ = get_bounding_box_and_ratio(masks[index])
            if bbox_b is None:
                adjusted.append(masks[index]) # Mask B rỗng: giữ nguyên
                continue

            # Mask A rỗng thì coi như tỷ lệ 1:1
            target_size = (1, 1) if bbox_a is None else (bbox_a[2] - bbox_a[0] + 1, bbox_a[3] - bbox_a[1] + 1)
            output_np = _adjust_mask(masks[index].cpu().numpy(), bbox_b, target_size, mode)
            adjusted.append(torch.from_numpy(output_np).to(masks.device))

        return (torch.stack(adjusted),)


# Ánh xạ để ComfyUI nhận diện class
NODE_CLASS_MAPPINGS = {
    "NH_MaskAspectRatioMatch": NH_MaskAspectRatioMatch
}

# Tên sẽ hiển thị trong menu của ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_MaskAspectRatioMatch": "NH Mask Aspect Ratio Match"
}
