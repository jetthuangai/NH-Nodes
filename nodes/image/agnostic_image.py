import torch
import torch.nn.functional as F
import numpy as np
import cv2


class NH_AgnosticImageGenerator:
    """Erase the mask region of the image and fill it with gray, noise or blur.

    Output:
    - agnostic_img: the image with the mask region filled (used as input for inpaint/VTON)
    - masked_img: the original image with the mask region blanked to white (transparent)
    - composite: the original image with the mask outline overlaid, for preview

    The mask is a soft alpha: intermediate values are blended partially, and feathering softens further starting
    from those same values.
    A 2-D mask [H,W] is treated as [1,H,W]; a mask whose size differs from the image is resized to the image size.
    The "noise" fill is deterministic: the i-th image of the batch uses seed = i.
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

        # Normalise the mask to [B,H,W] at the image size (a batch of 1 is reused for every image in the loop below).
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[-2:] != image.shape[1:3]:
            mask = F.interpolate(
                mask.unsqueeze(1).float(), size=tuple(image.shape[1:3]), mode="bilinear", align_corners=False, antialias=True,
            ).squeeze(1)

        for i in range(image.shape[0]):
            img_np = image[i].cpu().numpy()  # (H, W, C) float32 [0,1]
            m_np = mask[i].cpu().numpy() if i < mask.shape[0] else mask[0].cpu().numpy()  # (H, W)

            h, w, c = img_np.shape

            # Alpha keeps the soft values of the mask (no thresholding), clamped to [0,1]
            alpha = np.clip(m_np, 0.0, 1.0).astype(np.float32)
            # The preview outline needs a crisp shape: taken at the 0.5 level of the original mask (as before),
            # not from the soft alpha
            outline_mask = (m_np > 0.5).astype(np.float32)

            # Feathering: soften the mask edge
            if feathering > 0:
                kernel_size = feathering * 2 + 1
                alpha = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), 0)
                outline_mask = cv2.GaussianBlur(outline_mask, (kernel_size, kernel_size), 0)

            alpha_3ch = alpha[:, :, np.newaxis]  # (H, W, 1)

            # --- Fill content ---
            if fill_mode == "gray":
                fill = np.full_like(img_np, gray_value)

            elif fill_mode == "noise":
                # The seed is fixed by the index in the batch: the same input always gives the same noise
                noise = np.random.default_rng(i).random((h, w, c), dtype=np.float32)
                # Blend the noise with the centre gray to make it less harsh
                fill = noise * noise_strength + gray_value * (1.0 - noise_strength)

            elif fill_mode == "blur":
                # The blur radius must be odd
                kr = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
                fill = cv2.GaussianBlur(img_np, (kr, kr), 0)

            # --- agnostic_img: blend the fill into the mask region ---
            agnostic = img_np * (1.0 - alpha_3ch) + fill * alpha_3ch
            agnostic = np.clip(agnostic, 0.0, 1.0)

            # --- masked_img: mask region = white ---
            masked = img_np * (1.0 - alpha_3ch) + np.ones_like(img_np) * alpha_3ch
            masked = np.clip(masked, 0.0, 1.0)

            # --- composite: overlay the mask outline on the original image ---
            composite = self._create_composite(img_np, outline_mask, agnostic)

            results_agnostic.append(agnostic)
            results_masked.append(masked)
            results_composite.append(composite)

        out_agnostic = torch.from_numpy(np.stack(results_agnostic)).float()
        out_masked = torch.from_numpy(np.stack(results_masked)).float()
        out_composite = torch.from_numpy(np.stack(results_composite)).float()
        return (out_agnostic, out_masked, out_composite)

    @staticmethod
    def _create_composite(img_np, outline_mask, agnostic):
        """Build the preview: left half = original image with the mask outline, right half = agnostic."""
        h, w = img_np.shape[:2]

        # Find the mask contours to draw the outline
        mask_uint8 = np.ascontiguousarray((outline_mask * 255).astype(np.uint8))
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Left half: original image + green outline
        left = np.ascontiguousarray((np.clip(img_np, 0.0, 1.0) * 255).astype(np.uint8))
        cv2.drawContours(left, contours, -1, (0, 255, 0), 2)

        # Lightly tint the mask region with semi-transparent green
        overlay = np.ascontiguousarray(left.copy())
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), -1)
        left = cv2.addWeighted(left, 0.7, overlay, 0.3, 0)

        # Right half: agnostic
        right = np.ascontiguousarray((np.clip(agnostic, 0.0, 1.0) * 255).astype(np.uint8))

        # Join the 2 halves
        mid = w // 2
        composite = np.ascontiguousarray(left.copy())
        composite[:, mid:] = right[:, mid:]

        # Draw the dividing line in the middle
        cv2.line(composite, (mid, 0), (mid, h), (255, 255, 255), 1)

        return composite.astype(np.float32) / 255.0


NODE_CLASS_MAPPINGS = {
    "NH_AgnosticImageGenerator": NH_AgnosticImageGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_AgnosticImageGenerator": "NH Agnostic Image Generator"
}
