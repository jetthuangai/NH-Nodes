import torch
import numpy as np
import cv2


class NH_AgnosticImageGenerator:
    """Xóa vùng mask trên ảnh và fill bằng gray, noise, hoặc blur.

    Output:
    - agnostic_img: ảnh đã fill vùng mask (dùng làm input cho inpaint/VTON)
    - masked_img: ảnh gốc với vùng mask bị xóa trắng (transparent)
    - composite: ảnh gốc với viền mask overlay để preview
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "fill_mode": (["gray", "noise", "blur"],),
                "blur_radius": ("INT", {"default": 65, "min": 3, "max": 255, "step": 2}),
                "noise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "gray_value": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feathering": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("agnostic_img", "masked_img", "composite")
    FUNCTION = "generate"
    CATEGORY = "NH-Nodes/Image"

    def generate(self, image, mask, fill_mode, blur_radius, noise_strength, gray_value, feathering):
        results_agnostic = []
        results_masked = []
        results_composite = []

        for i in range(image.shape[0]):
            img_np = image[i].cpu().numpy()  # (H, W, C) float32 [0,1]
            m_np = mask[i].cpu().numpy() if i < mask.shape[0] else mask[0].cpu().numpy()  # (H, W)

            h, w, c = img_np.shape

            # Tạo alpha mask [0,1]
            alpha = (m_np > 0.5).astype(np.float32)

            # Feathering: làm mềm biên mask
            if feathering > 0:
                kernel_size = feathering * 2 + 1
                alpha = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), 0)

            alpha_3ch = alpha[:, :, np.newaxis]  # (H, W, 1)

            # --- Fill content ---
            if fill_mode == "gray":
                fill = np.full_like(img_np, gray_value)

            elif fill_mode == "noise":
                noise = np.random.rand(h, w, c).astype(np.float32)
                # Blend noise với gray trung tâm để bớt harsh
                fill = noise * noise_strength + gray_value * (1.0 - noise_strength)

            elif fill_mode == "blur":
                # Blur radius phải lẻ
                kr = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
                fill = cv2.GaussianBlur(img_np, (kr, kr), 0)

            # --- agnostic_img: blend fill vào vùng mask ---
            agnostic = img_np * (1.0 - alpha_3ch) + fill * alpha_3ch
            agnostic = np.clip(agnostic, 0.0, 1.0)

            # --- masked_img: vùng mask = trắng ---
            masked = img_np * (1.0 - alpha_3ch) + np.ones_like(img_np) * alpha_3ch
            masked = np.clip(masked, 0.0, 1.0)

            # --- composite: overlay viền mask lên ảnh gốc ---
            composite = self._create_composite(img_np, alpha, agnostic)

            results_agnostic.append(agnostic)
            results_masked.append(masked)
            results_composite.append(composite)

        out_agnostic = torch.from_numpy(np.stack(results_agnostic)).float()
        out_masked = torch.from_numpy(np.stack(results_masked)).float()
        out_composite = torch.from_numpy(np.stack(results_composite)).float()
        return (out_agnostic, out_masked, out_composite)

    @staticmethod
    def _create_composite(img_np, alpha, agnostic):
        """Tạo preview: nửa trái = ảnh gốc với viền mask, nửa phải = agnostic."""
        h, w = img_np.shape[:2]

        # Tìm contour của mask để vẽ viền
        mask_uint8 = (alpha * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Nửa trái: ảnh gốc + viền xanh
        left = (img_np * 255).astype(np.uint8)
        cv2.drawContours(left, contours, -1, (0, 255, 0), 2)

        # Tô nhẹ vùng mask bằng màu xanh semi-transparent
        overlay = left.copy()
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), -1)
        left = cv2.addWeighted(left, 0.7, overlay, 0.3, 0)

        # Nửa phải: agnostic
        right = (agnostic * 255).astype(np.uint8)

        # Ghép 2 nửa
        mid = w // 2
        composite = left.copy()
        composite[:, mid:] = right[:, mid:]

        # Vẽ đường chia giữa
        cv2.line(composite, (mid, 0), (mid, h), (255, 255, 255), 1)

        return composite.astype(np.float32) / 255.0


NODE_CLASS_MAPPINGS = {
    "NH_AgnosticImageGenerator": NH_AgnosticImageGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_AgnosticImageGenerator": "NH Agnostic Image Generator"
}
