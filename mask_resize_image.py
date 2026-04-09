import torch
import numpy as np
import cv2


class NH_MaskResizeImage:
    """Resize ảnh theo kích thước target, đảm bảo vùng mask không bị mất.

    - Crop: scale lên để fill target, sau đó crop bớt, giữ mask ở trung tâm.
    - Pad: scale vừa đủ để mask nằm trong target, pad phần thừa bằng màu tùy chọn.
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

        out_images = torch.from_numpy(np.stack(results_img, axis=0)).float()
        out_masks = torch.from_numpy(np.stack(results_mask, axis=0)).float()
        return (out_images, out_masks)

    def _process_single(self, img_np, mask_np, tw, th, mode, pad_color):
        sh, sw = img_np.shape[:2]

        # Tìm bounding box của mask
        rows, cols = np.where(mask_np > 0.5)
        if len(rows) == 0:
            # Mask rỗng: resize bình thường
            return self._simple_resize(img_np, mask_np, tw, th, mode, pad_color)

        my1, my2 = int(np.min(rows)), int(np.max(rows))
        mx1, mx2 = int(np.min(cols)), int(np.max(cols))
        mw = mx2 - mx1 + 1
        mh = my2 - my1 + 1

        # Tính scale cần thiết để vùng mask vừa trong target
        scale_for_mask_w = tw / mw
        scale_for_mask_h = th / mh

        if mode == "pad":
            # Scale nhỏ nhất để toàn bộ mask vừa trong target
            scale = min(scale_for_mask_w, scale_for_mask_h)
        else:
            # Crop: scale lớn nhất để mask vẫn vừa, sau đó crop phần thừa
            scale = max(scale_for_mask_w, scale_for_mask_h)

        # Kích thước ảnh sau khi scale
        new_sw = max(1, round(sw * scale))
        new_sh = max(1, round(sh * scale))

        # Resize ảnh và mask
        img_scaled = cv2.resize(img_np, (new_sw, new_sh), interpolation=cv2.INTER_LANCZOS4)
        mask_scaled = cv2.resize(mask_np, (new_sw, new_sh), interpolation=cv2.INTER_NEAREST)

        # Vị trí mask center sau khi scale
        mask_cx = round((mx1 + mx2) / 2.0 * scale)
        mask_cy = round((my1 + my2) / 2.0 * scale)

        if mode == "crop":
            return self._crop_around_center(img_scaled, mask_scaled, tw, th, mask_cx, mask_cy)
        else:
            return self._pad_around_center(img_scaled, mask_scaled, tw, th, mask_cx, mask_cy, pad_color)

    def _crop_around_center(self, img, mask, tw, th, cx, cy):
        h, w = img.shape[:2]

        # Tính vùng crop, giữ mask center ở giữa target
        x1 = cx - tw // 2
        y1 = cy - th // 2

        # Clamp để không vượt ra ngoài ảnh
        x1 = max(0, min(x1, w - tw))
        y1 = max(0, min(y1, h - th))

        # Đảm bảo đủ kích thước (trường hợp ảnh scaled nhỏ hơn target)
        x2 = x1 + tw
        y2 = y1 + th

        if x2 > w or y2 > h:
            # Fallback: pad nếu thiếu
            return self._pad_around_center(img, mask, tw, th, cx, cy, (0, 0, 0))

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
        """Fallback khi mask rỗng."""
        img_resized = cv2.resize(img_np, (tw, th), interpolation=cv2.INTER_LANCZOS4)
        mask_resized = cv2.resize(mask_np, (tw, th), interpolation=cv2.INTER_NEAREST)
        return img_resized, mask_resized

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
