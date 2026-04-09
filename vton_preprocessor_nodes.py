import torch
import numpy as np
from PIL import Image, ImageFilter
import os
import folder_paths
from .utils import resize_image, tensor_to_pil, pil_to_tensor, pil_to_mask
from scipy.ndimage import shift

# Import logic cục bộ
from .preprocess.humanparsing.run_parsing import Parsing
from .preprocess.dwpose import DWposeDetector
from .src.utils_mask import get_mask_location

MODELS_DIR = os.path.join(folder_paths.models_dir, "ComfyUI-Vton-Mask")
LOADED_VTON_MODELS = None

LIP_PALETTE = [
    0,0,0, 128,0,0, 255,0,0, 0,85,0, 170,0,51, 255,85,0, 0,0,85, 0,119,221, 85,85,0, 0,85,85,
    85,51,0, 52,86,128, 0,128,0, 0,0,255, 51,170,221, 0,255,255, 85,255,170, 170,255,85, 255,255,0, 255,170,85
]

class NH_VTonUltimateProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "human_image": ("IMAGE",),
                "category": (["Upper-body", "Lower-body", "Dresses", "Upper-body (Sleeveless)", "Lower-body (Shorts/Skirt)"],),
                "mask_feathering": ("INT", {"default": 4, "min": 0, "max": 50, "step": 1}),
                "cover_shoes": ("BOOLEAN", {"default": False}),
                "refine_hands": ("BOOLEAN", {"default": True}),
                "refine_hair": ("BOOLEAN", {"default": True}),
                "device": (["cpu", "cuda"],),
            },
             "optional": {
                "offset_top": ("INT", {"default": 0, "min": -200, "max": 200, "step": 1}),
                "offset_bottom": ("INT", {"default": 0, "min": -200, "max": 200, "step": 1}),
                "offset_left": ("INT", {"default": 0, "min": -200, "max": 200, "step": 1}),
                "offset_right": ("INT", {"default": 0, "min": -200, "max": 200, "step": 1}),
             }
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE", "IMAGE", "IMAGE", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("final_mask", "agnostic_mask", "densepose_image", "parsing_image", "masked_image", "parsing_map_raw", "hair_mask", "hands_mask")
    FUNCTION = "process"
    CATEGORY = "NH-Nodes/VTON"

    def load_model(self, device):
        global LOADED_VTON_MODELS
        if LOADED_VTON_MODELS is not None and LOADED_VTON_MODELS.get("device") == device: return LOADED_VTON_MODELS
        if not os.path.exists(MODELS_DIR):
            from huggingface_hub import snapshot_download
            os.makedirs(MODELS_DIR, exist_ok=True)
            snapshot_download(repo_id="kg-09/kg-vton-mask", local_dir=MODELS_DIR, local_dir_use_symlinks=False)
        LOADED_VTON_MODELS = {
            'dwprocessor': DWposeDetector(model_root=MODELS_DIR, device=device),
            'parsing_model': Parsing(model_root=MODELS_DIR, device=device),
            'device': device
        }
        return LOADED_VTON_MODELS

    def process(self, human_image, category, mask_feathering, cover_shoes, refine_hands, refine_hair, device, 
                offset_top=0, offset_bottom=0, offset_left=0, offset_right=0):
        vton_model = self.load_model(device)
        human_img_pil = tensor_to_pil(human_image)
        human_img_resized = resize_image(human_img_pil)

        pose_img_np, _, _, candidate = vton_model['dwprocessor'](np.array(human_img_resized)[:,:,::-1])
        pose_image_pil = Image.fromarray(pose_img_np[:,:,::-1])
        parsing_map_pil, parsing_map_colored_pil = vton_model['parsing_model'](human_img_resized)

        candidate = candidate[0]
        keypoints_original = candidate.copy()
        keypoints_original[:, 0] *= human_img_pil.width
        keypoints_original[:, 1] *= human_img_pil.height
        keypoints_resized = candidate.copy()
        keypoints_resized[:, 0] *= human_img_resized.width
        keypoints_resized[:, 1] *= human_img_resized.height
        
        base_category = category.split(" ")[0]
        final_mask_pil, _ = get_mask_location(base_category, parsing_map_pil, keypoints_resized, 
                                              human_img_resized.width, human_img_resized.height,
                                              offset_top, offset_bottom, offset_left, offset_right)
        
        parsing_map_full_np = np.array(parsing_map_pil.resize(human_img_pil.size, Image.NEAREST))
        final_mask_pil = final_mask_pil.resize(human_img_pil.size, Image.NEAREST)
        
        mask_np = np.array(final_mask_pil).astype(np.float32) / 255.0
        if mask_np.ndim == 3: mask_np = mask_np[:,:,0]

        # --- CÁC LOGIC TINH CHỈNH ---
        
        if category in ["Lower-body", "Lower-body (Shorts/Skirt)"]:
            body_cols = np.any((parsing_map_full_np > 0), axis=0)
            if np.any(body_cols):
                xmin_person, xmax_person = np.where(body_cols)[0][[0, -1]]
                mask_rows = np.any(mask_np, axis=1)
                if np.any(mask_rows):
                    ymin_mask, ymax_mask = np.where(mask_rows)[0][[0, -1]]
                    new_mask_np = np.zeros_like(mask_np)
                    new_mask_np[ymin_mask:ymax_mask+1, xmin_person:xmax_person+1] = 1.0
                    mask_np = new_mask_np

            shoulder_pts = keypoints_original[[2,5], :]
            hip_pts = keypoints_original[[8,11], :]
            if np.all(shoulder_pts[:,1]>0) and np.all(hip_pts[:,1]>0):
                limit_y = int((np.mean(shoulder_pts[:, 1]) + np.mean(hip_pts[:, 1])) / 2)
                mask_np[0:limit_y, :] = 0.0
            
            if category == "Lower-body (Shorts/Skirt)":
                knee_pts = keypoints_original[[9, 12], :]
                if np.all(knee_pts[:,1]>0):
                    y_bottom_limit = int(np.mean(knee_pts[:, 1]))
                    mask_np[y_bottom_limit:, :] = 0
        
        if category == "Upper-body (Sleeveless)":
             arm_mask = np.isin(parsing_map_full_np, [3, 14, 15]).astype(np.float32)
             mask_np = np.clip(mask_np - arm_mask, 0, 1)

        hair_mask_np = np.isin(parsing_map_full_np, [2]).astype(np.float32)
        arm_mask_np = np.isin(parsing_map_full_np, [3, 14, 15]).astype(np.float32)

        if refine_hands: mask_np = np.clip(mask_np - arm_mask_np, 0, 1)
        if refine_hair: mask_np = np.clip(mask_np - hair_mask_np, 0, 1)

        if cover_shoes:
            hip_pts = keypoints_original[[8, 11], :]
            ankle_pts = keypoints_original[[15, 16], :]
            if np.all(hip_pts[:,1]>0) and np.all(ankle_pts[:,1]>0):
                y_start = int(max(np.mean(hip_pts[:, 1]), np.min(ankle_pts[:, 1])))
                body_cols = np.any((parsing_map_full_np > 0), axis=0)
                if np.any(body_cols):
                    xmin, xmax = np.where(body_cols)[0][[0, -1]]
                    mask_np[y_start:human_img_pil.height, xmin:xmax+1] = 1.0
        
        if mask_feathering > 0:
            feathered_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
            feathered_pil = feathered_pil.filter(ImageFilter.GaussianBlur(radius=mask_feathering))
            mask_np = np.array(feathered_pil).astype(np.float32) / 255.0
        
        # --- Tạo các output cuối cùng ---
        person_silhouette_np = (parsing_map_full_np > 0).astype(np.float32)
        agnostic_mask_np = np.clip(person_silhouette_np - mask_np, 0, 1)
        agnostic_mask_pil = Image.fromarray((agnostic_mask_np * 255).astype(np.uint8))
        
        final_mask_for_composite = Image.fromarray((mask_np * 255).astype(np.uint8))
        masked_image = Image.composite(final_mask_for_composite.convert("L"), human_img_pil, final_mask_for_composite)
        
        pose_image_final = pose_image_pil.resize(human_img_pil.size, Image.LANCZOS)
        parsing_map_colored_pil = Image.fromarray(parsing_map_full_np.astype(np.uint8), 'P')
        parsing_map_colored_pil.putpalette(LIP_PALETTE)
        
        parsing_map_raw_tensor = torch.from_numpy(parsing_map_full_np.astype(np.float32) / 19.0).unsqueeze(0)

        return (
            torch.from_numpy(mask_np).unsqueeze(0),
            pil_to_mask(agnostic_mask_pil),
            pil_to_tensor(pose_image_final),
            pil_to_tensor(parsing_map_colored_pil.convert("RGB")),
            pil_to_tensor(masked_image),
            parsing_map_raw_tensor,
            torch.from_numpy(hair_mask_np).unsqueeze(0),
            torch.from_numpy(arm_mask_np).unsqueeze(0),
        )

NODE_CLASS_MAPPINGS = {"NH_VTonUltimateProcessor": NH_VTonUltimateProcessor}
NODE_DISPLAY_NAME_MAPPINGS = {"NH_VTonUltimateProcessor": "VTON Ultimate Processor (NH)"}