import torch
import numpy as np
import cv2 # Cần thư viện OpenCV

def get_bounding_box(mask_np):
    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None, None, None, None
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax, ymax

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
        src_img_np = (source_image[0].cpu().numpy() * 255).astype(np.uint8)
        src_mask_np = source_mask[0].cpu().numpy()
        
        dest_img_np = (dest_image[0].cpu().numpy() * 255).astype(np.uint8)
        dest_mask_np = dest_mask[0].cpu().numpy()

        src_x1, src_y1, src_x2, src_y2 = get_bounding_box(src_mask_np > 0.5)
        dest_x1, dest_y1, dest_x2, dest_y2 = get_bounding_box(dest_mask_np > 0.5)

        if src_x1 is None or dest_x1 is None:
            return (dest_image,)

        src_face = src_img_np[src_y1:src_y2+1, src_x1:src_x2+1]
        src_face_mask = (src_mask_np[src_y1:src_y2+1, src_x1:src_x2+1] * 255).astype(np.uint8)

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
        
        output_image = torch.from_numpy(output_img_np.astype(np.float32) / 255.0).unsqueeze(0)

        return (output_image,)

# --- PHẦN BỊ THIẾU TRƯỚC ĐÂY ĐÃ ĐƯỢC THÊM VÀO ĐÂY ---
# Ánh xạ để ComfyUI nhận diện class
NODE_CLASS_MAPPINGS = {
    "NH_SimpleFacePaste": NH_SimpleFacePaste
}

# Tên sẽ hiển thị trong menu của ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_SimpleFacePaste": "NH Simple Face Paste"
}