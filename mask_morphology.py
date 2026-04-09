import torch
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter, binary_dilation, binary_erosion

class NH_MaskMorphology:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "horizontal_expand": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "vertical_expand": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "fill_holes": ("BOOLEAN", {"default": False}),
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = "NH-Nodes/Mask"

    def process(self, mask, horizontal_expand, vertical_expand, fill_holes, blur_radius):
        masks_out = []
        
        for m in mask:
            m_np = m.cpu().numpy()
            
            # 1. Co giãn theo chiều ngang
            if horizontal_expand != 0:
                kernel_width = abs(horizontal_expand)
                # Bỏ qua nếu kernel = 0
                if kernel_width > 0:
                    structure_h = np.ones((1, kernel_width))
                    if horizontal_expand > 0:
                        m_np = binary_dilation(m_np, structure=structure_h).astype(m_np.dtype)
                    else:
                        m_np = binary_erosion(m_np, structure=structure_h).astype(m_np.dtype)

            # 2. Co giãn theo chiều dọc
            if vertical_expand != 0:
                kernel_height = abs(vertical_expand)
                # Bỏ qua nếu kernel = 0
                if kernel_height > 0:
                    structure_v = np.ones((kernel_height, 1))
                    if vertical_expand > 0:
                        m_np = binary_dilation(m_np, structure=structure_v).astype(m_np.dtype)
                    else:
                        m_np = binary_erosion(m_np, structure=structure_v).astype(m_np.dtype)

            # 3. Lấp đầy lỗ trống (nếu cần)
            if fill_holes:
                m_bool = m_np > 0.5
                m_filled_bool = binary_fill_holes(m_bool)
                m_np = m_filled_bool.astype(m_np.dtype)

            # 4. Làm mờ
            if blur_radius > 0:
                m_np = gaussian_filter(m_np, sigma=blur_radius)
            
            masks_out.append(torch.from_numpy(m_np))

        return (torch.stack(masks_out),)

NODE_CLASS_MAPPINGS = { "NH_MaskAdvancedMorphology": NH_MaskMorphology }
NODE_DISPLAY_NAME_MAPPINGS = { "NH_MaskAdvancedMorphology": "Mask Morphology (NH)" }