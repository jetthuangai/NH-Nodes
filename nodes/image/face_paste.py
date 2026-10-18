import torch
import numpy as np
import cv2 # Cần thư viện OpenCV

from ...core.mask_ops import mask_bbox

def _to_uint8(image):
    # Clip trước khi ép uint8: giá trị lệch khỏi [0,1] (vd. do resample) sẽ bị wrap về gần 0 nếu không clip.
    return (np.clip(image.cpu().numpy(), 0.0, 1.0) * 255).astype(np.uint8)

def _as_mask_batch(mask):
    # MASK 2-D [H,W] không có chiều batch: coi là [1,H,W].
    return mask.unsqueeze(0) if mask.dim() == 2 else mask

class NH_SimpleFacePaste:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dest_image": ("IMAGE",),
                "dest_mask": ("MASK",),
                "source_image": ("IMAGE",),
                "source_mask": ("MASK",),
                "feathering": ("INT", {"default": 20, "min": 0, "max": 200, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paste_face"
    CATEGORY = "NH-Nodes/Image"

    def paste_face(self, dest_image, dest_mask, source_image, source_mask, feathering):
        dest_mask = _as_mask_batch(dest_mask)
        source_mask = _as_mask_batch(source_mask)
        # Output theo batch của dest_image; input khác có batch 1 (hoặc ngắn hơn) được lặp lại theo vòng.
        output = dest_image.clone()
        for index in range(dest_image.shape[0]):
            pasted = self._paste_single(
                dest_image[index], dest_mask[index % dest_mask.shape[0]],
                source_image[index % source_image.shape[0]], source_mask[index % source_mask.shape[0]],
                feathering,
            )
            if pasted is not None:  # mask rỗng: giữ nguyên ảnh đích của item này
                output[index] = pasted
        return (output,)

    @staticmethod
    def _paste_single(dest_image, dest_mask, source_image, source_mask, feathering):
        """Dán mặt của một ảnh nguồn [H,W,C] lên một ảnh đích [H,W,C]; trả None nếu một trong hai mask rỗng."""
        src_img_np = _to_uint8(source_image)
        src_mask_np = source_mask.cpu().numpy()

        dest_img_np = _to_uint8(dest_image)
        dest_mask_np = dest_mask.cpu().numpy()

        src_box = mask_bbox(src_mask_np)
        dest_box = mask_bbox(dest_mask_np)

        if src_box is None or dest_box is None:
            return None

        src_x1, src_y1, src_x2, src_y2 = src_box
        dest_x1, dest_y1, dest_x2, dest_y2 = dest_box

        src_face = src_img_np[src_y1:src_y2+1, src_x1:src_x2+1]
        src_face_mask = (np.clip(src_mask_np[src_y1:src_y2+1, src_x1:src_x2+1], 0.0, 1.0) * 255).astype(np.uint8)

        dest_width = dest_x2 - dest_x1 + 1
        dest_height = dest_y2 - dest_y1 + 1
        resized_face = cv2.resize(src_face, (dest_width, dest_height), interpolation=cv2.INTER_AREA)
        resized_face_mask = cv2.resize(src_face_mask, (dest_width, dest_height), interpolation=cv2.INTER_NEAREST)

        if feathering > 0:
            k_size = feathering * 2 + 1
            resized_face_mask = cv2.GaussianBlur(resized_face_mask, (k_size, k_size), 0)

        alpha_mask = resized_face_mask / 255.0
        alpha_mask = np.expand_dims(alpha_mask, axis=2)

        paste_area = dest_img_np[dest_y1:dest_y1+dest_height, dest_x1:dest_x1+dest_width]
        blended_area = paste_area * (1 - alpha_mask) + resized_face * alpha_mask
        
        output_img_np = dest_img_np.copy()
        output_img_np[dest_y1:dest_y1+dest_height, dest_x1:dest_x1+dest_width] = blended_area.astype(np.uint8)

        return torch.from_numpy(output_img_np.astype(np.float32) / 255.0)

# --- PHẦN BỊ THIẾU TRƯỚC ĐÂY ĐÃ ĐƯỢC THÊM VÀO ĐÂY ---
# Ánh xạ để ComfyUI nhận diện class
NODE_CLASS_MAPPINGS = {
    "NH_SimpleFacePaste": NH_SimpleFacePaste
}

# Tên sẽ hiển thị trong menu của ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_SimpleFacePaste": "NH Simple Face Paste"
}