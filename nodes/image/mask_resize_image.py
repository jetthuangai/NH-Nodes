import torch
import numpy as np
import cv2

from ...core.mask_ops import mask_bbox


class NH_MaskResizeImage:
    """Resize ảnh theo kích thước target, đảm bảo vùng mask không bị mất.

    - Crop: scale để mask vừa khít target, crop phần ảnh thừa (pad nếu ảnh nhỏ hơn target).
    - Pad: scale vừa đủ để mask nằm trong target, pad phần thừa bằng màu tùy chọn.
    - Mask rỗng: coi như mask phủ cả ảnh (crop: lấp đầy target rồi cắt giữa; pad: cả ảnh nằm gọn trong target).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "mode": (["crop", "pad"],),
                "pad_color_hex": ("STRING", {"default": "#000000"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "resize_by_mask"
    CATEGORY = "NH-Nodes/Image"

    def resize_by_mask(self, image, mask, width, height, mode, pad_color_hex):
        # Parse pad color
        pad_color = self._hex_to_rgb(pad_color_hex)

        # image: (B, H, W, C) float32 [0,1], mask: (B, H, W) float32 [0,1]
        results_img = []
        results_mask = []

        for i in range(image.shape[0]):
            img_np = image[i].cpu().numpy()  # (H, W, C)
            m_np = mask[i].cpu().numpy() if i < mask.shape[0] else mask[0].cpu().numpy()  # (H, W)

            out_img, out_mask = self._process_single(img_np, m_np, width, height, mode, pad_color)
            results_img.append(out_img)
            results_mask.append(out_mask)

        # Lanczos có thể vọt ra ngoài [0,1] (ringing) quanh cạnh tương phản cao -> kẹp lại
        out_images = torch.from_numpy(np.clip(np.stack(results_img, axis=0), 0.0, 1.0)).float()
        out_masks = torch.from_numpy(np.stack(results_mask, axis=0)).float()
        return (out_images, out_masks)

    def _process_single(self, img_np, mask_np, tw, th, mode, pad_color):
        sh, sw = img_np.shape[:2]

        # Tìm bounding box của mask
        bbox = mask_bbox(mask_np)
        if bbox is None:
            # Mask rỗng: resize bình thường
            return self._simple_resize(img_np, mask_np, tw, th, mode, pad_color)

        mx1, my1, mx2, my2 = bbox
        mw = mx2 - mx1 + 1
        mh = my2 - my1 + 1

        # Tính scale cần thiết để vùng mask vừa trong target
        scale_for_mask_w = tw / mw
        scale_for_mask_h = th / mh

        # Cả hai mode: scale lớn nhất mà toàn bộ mask vẫn nằm trong target.
        # (Dùng max ở đây sẽ làm mask tràn target ở một chiều và bị crop mất.)
        scale = min(scale_for_mask_w, scale_for_mask_h)

        # Kích thước ảnh sau khi scale
        new_sw = max(1, round(sw * scale))
        new_sh = max(1, round(sh * scale))

        # Resize ảnh và mask
        img_scaled = cv2.resize(img_np, (new_sw, new_sh), interpolation=cv2.INTER_LANCZOS4)
        mask_scaled = cv2.resize(mask_np, (new_sw, new_sh), interpolation=cv2.INTER_NEAREST)

        # Vị trí mask center sau khi scale
        # Pixel mask chiếm [mx1, mx2 + 1) -> tâm là (mx1 + mx2 + 1) / 2
        mask_cx = round((mx1 + mx2 + 1) / 2.0 * scale)
        mask_cy = round((my1 + my2 + 1) / 2.0 * scale)

        if mode == "crop":
            return self._crop_around_center(img_scaled, mask_scaled, tw, th, mask_cx, mask_cy, pad_color)
        else:
            return self._pad_around_center(img_scaled, mask_scaled, tw, th, mask_cx, mask_cy, pad_color)

    def _crop_around_center(self, img, mask, tw, th, cx, cy, pad_color):
        h, w = img.shape[:2]

        # Ảnh scaled nhỏ hơn target: không thể crop mà giữ đủ mask -> pad bằng màu user chọn
        if w < tw or h < th:
            return self._pad_around_center(img, mask, tw, th, cx, cy, pad_color)

        # Tính vùng crop, giữ mask center ở giữa target
        x1 = cx - tw // 2
        y1 = cy - th // 2

        # Clamp để không vượt ra ngoài ảnh
        x1 = max(0, min(x1, w - tw))
        y1 = max(0, min(y1, h - th))
        x2 = x1 + tw
        y2 = y1 + th

        return img[y1:y2, x1:x2], mask[y1:y2, x1:x2]

    def _pad_around_center(self, img, mask, tw, th, cx, cy, pad_color):
        h, w = img.shape[:2]

        # Vị trí đặt ảnh trên canvas target, giữ mask center ở giữa target
        paste_x = tw // 2 - cx
        paste_y = th // 2 - cy

        # Clamp: đảm bảo ảnh không bị đặt quá xa khiến vùng mask bị cắt
        paste_x = max(min(paste_x, tw - 1), -(w - 1))
        paste_y = max(min(paste_y, th - 1), -(h - 1))

        # Tạo canvas
        canvas_img = np.zeros((th, tw, img.shape[2]), dtype=img.dtype)
        r, g, b = pad_color
        canvas_img[:, :, 0] = r / 255.0
        canvas_img[:, :, 1] = g / 255.0
        canvas_img[:, :, 2] = b / 255.0

        canvas_mask = np.zeros((th, tw), dtype=mask.dtype)

        # Tính vùng overlap
        src_x1 = max(0, -paste_x)
        src_y1 = max(0, -paste_y)
        dst_x1 = max(0, paste_x)
        dst_y1 = max(0, paste_y)
        copy_w = min(w - src_x1, tw - dst_x1)
        copy_h = min(h - src_y1, th - dst_y1)

        if copy_w > 0 and copy_h > 0:
            canvas_img[dst_y1:dst_y1 + copy_h, dst_x1:dst_x1 + copy_w] = \
                img[src_y1:src_y1 + copy_h, src_x1:src_x1 + copy_w]
            canvas_mask[dst_y1:dst_y1 + copy_h, dst_x1:dst_x1 + copy_w] = \
                mask[src_y1:src_y1 + copy_h, src_x1:src_x1 + copy_w]

        return canvas_img, canvas_mask

    def _simple_resize(self, img_np, mask_np, tw, th, mode, pad_color):
        """Fallback khi mask rỗng: coi như mask phủ cả ảnh, không méo ảnh.

        crop = scale lớn nhất (lấp đầy target) rồi cắt giữa; pad = scale nhỏ nhất (cả ảnh nằm trong target), phần thừa pad_color.
        """
        sh, sw = img_np.shape[:2]
        scale = max(tw / sw, th / sh) if mode == "crop" else min(tw / sw, th / sh)
        new_sw = max(1, round(sw * scale))
        new_sh = max(1, round(sh * scale))
        img_scaled = cv2.resize(img_np, (new_sw, new_sh), interpolation=cv2.INTER_LANCZOS4)
        mask_scaled = cv2.resize(mask_np, (new_sw, new_sh), interpolation=cv2.INTER_NEAREST)

        # Tâm ảnh sau khi scale
        cx = round(sw / 2.0 * scale)
        cy = round(sh / 2.0 * scale)
        if mode == "crop":
            return self._crop_around_center(img_scaled, mask_scaled, tw, th, cx, cy, pad_color)
        return self._pad_around_center(img_scaled, mask_scaled, tw, th, cx, cy, pad_color)

    @staticmethod
    def _hex_to_rgb(hex_str):
        hex_str = hex_str.strip().lstrip('#')
        if len(hex_str) != 6:
            return (0, 0, 0)
        try:
            return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))
        except ValueError:
            return (0, 0, 0)


NODE_CLASS_MAPPINGS = {
    "NH_MaskResizeImage": NH_MaskResizeImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_MaskResizeImage": "NH Mask-Aware Resize"
}
