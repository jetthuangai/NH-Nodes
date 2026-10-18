# VTON Ultimate Processor (NH). Built on ComfyUI-Vton-Mask's mask generator and FitDiT
# preprocessing: CC BY-NC-SA 4.0 (non-commercial). See THIRD_PARTY_NOTICES.md.
import gc
import traceback
import torch
import numpy as np
from PIL import Image, ImageFilter
import os
import folder_paths
from ...core.model_files import download_pinned, make_room_on_gpu, register_model_folder
from ...core.utils import resize_image, tensor_to_pil, pil_to_tensor, pil_to_mask

# onnxruntime (via preprocess/) and scikit-image (via utils_mask) are imported
# inside load_model()/process(), not here: they are native libraries that only
# this node needs, and loading them at ComfyUI start-up makes the pack import
# heavier and breaks process memory snapshots (Modal restores segfault).

# Default location: where the models are downloaded and where existing installs already keep them.
MODELS_DIR = os.path.join(folder_paths.models_dir, "ComfyUI-Vton-Mask")
LOADED_VTON_MODELS = None

# extra_model_paths.yaml key. Each location holds dwpose/ and humanparsing/, like MODELS_DIR.
VTON_FOLDER_NAME = "vton_mask"
VTON_FOLDER_EXTENSIONS = (".onnx",)  # what ComfyUI lists for VTON_FOLDER_NAME: the models, not the HF cache metadata
VTON_REPO_ID = "kg-09/kg-vton-mask"
# The repo is third-party: pin it so a moving "main" cannot change what gets downloaded.
VTON_REVISION = "4648778286cac2dca9aab8da167ea5257462487d"
VTON_MODEL_FILES = (
    "dwpose/yolox_l.onnx", "dwpose/dw-ll_ucoco_384.onnx",
    "humanparsing/parsing_atr.onnx", "humanparsing/parsing_lip.onnx",
)
_VTON_DOWNLOAD_PATTERNS = ("dwpose/*.onnx", "humanparsing/*.onnx")  # skips the repo's other files
# Measured with the four CUDA sessions (RTX 5090): ~2.9 GB held after one run, ~0.9 GB of it weights
# and the rest ONNX Runtime's arena. Used to ask ComfyUI for that much room (plus the reserve it keeps
# for itself, see make_room_on_gpu) before they are created.
_VTON_VRAM_ESTIMATE = 3 * 1024 ** 3

register_model_folder(VTON_FOLDER_NAME, MODELS_DIR, VTON_FOLDER_EXTENSIONS)  # make the folder name known to ComfyUI at load time


def _missing_vton_files(root):
    return [name for name in VTON_MODEL_FILES if not os.path.isfile(os.path.join(root, *name.split("/")))]


def resolve_vton_models_dir(download=True):
    """Folder holding all four VTON .onnx files, downloading them into MODELS_DIR if none does.

    Searched: MODELS_DIR (models/ComfyUI-Vton-Mask) and every `vton_mask` location from
    extra_model_paths.yaml. A folder missing any of the four files, as an interrupted first
    download leaves it, does not count: the next call finishes the download.
    """
    for root in register_model_folder(VTON_FOLDER_NAME, MODELS_DIR, VTON_FOLDER_EXTENSIONS):
        if not _missing_vton_files(root):
            return root
    if not download:
        return None
    download_pinned(VTON_REPO_ID, VTON_REVISION, MODELS_DIR, _VTON_DOWNLOAD_PATTERNS)
    missing = _missing_vton_files(MODELS_DIR)
    if missing:
        raise RuntimeError(f"[NH-Nodes] VTON model files missing after download in {MODELS_DIR}: {', '.join(missing)}")
    return MODELS_DIR


def release_vton_models():
    """Drop the cached ONNX Runtime sessions, which frees the GPU memory they hold.

    ONNX Runtime allocates outside PyTorch, so ComfyUI's model management cannot see or unload
    these sessions: their VRAM only comes back when the last reference to them goes away.
    Sessions that ran on the CPU are dropped the same way, but hold no VRAM: torch's cache is left alone.
    """
    global LOADED_VTON_MODELS
    from .preprocess.ort_session import cuda_sessions_alive

    on_gpu = cuda_sessions_alive()  # asked before the sessions go: afterwards nothing is left to ask about
    LOADED_VTON_MODELS = None
    gc.collect()
    if on_gpu and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as exc:  # runs in a finally: after a CUDA error this must not replace the run's own error
            print(f"[NH-Nodes] VTON: released the models but torch.cuda.empty_cache() failed: {exc}")


def _should_release(device, keep_model_loaded):
    """True when a run on `device` must hand its cached sessions back once it is over.

    That is the case for sessions ONNX Runtime serves from the GPU while the user has not asked to
    keep them. What counts is where they really run, not the device widget: a cuda request that
    fell back to the CPU (CUDA/cuDNN mismatch, CPU-only wheel) holds plain RAM, no VRAM, so it keeps
    its cache like device=cpu does: dropping it would only reload four models for every item of a list.
    Called after the run, once the sessions exist.
    """
    from .preprocess.ort_session import cuda_sessions_alive

    return device == "cuda" and not keep_model_loaded and cuda_sessions_alive()


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
                "keep_model_loaded": ("BOOLEAN", {"default": False, "tooltip": (
                    "Off (default): after each run on cuda the four ONNX models (about 3 GB of VRAM) are released, "
                    "so whatever runs next gets the whole GPU; the next run reloads them. "
                    "On: keep them in VRAM for faster repeated runs. With a list of images ComfyUI calls this node "
                    "once per item (a batch tensor is a single call), so keeping the models loaded is much faster "
                    "for lists and repeated runs. "
                    "ComfyUI cannot see or free that memory, so Free memory does nothing for it "
                    "and a large model run afterwards may run out of VRAM. "
                    "Ignored when the models run on the CPU (device cpu, or cuda without a working ONNX Runtime GPU "
                    "build): they always stay cached."
                )}),
             }
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE", "IMAGE", "IMAGE", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("final_mask", "agnostic_mask", "densepose_image", "parsing_image", "masked_image", "parsing_map_raw", "hair_mask", "hands_mask")
    FUNCTION = "process"
    CATEGORY = "NH-Nodes/VTON"

    def load_model(self, device):
        global LOADED_VTON_MODELS
        if LOADED_VTON_MODELS is not None and LOADED_VTON_MODELS.get("device") == device:
            return LOADED_VTON_MODELS
        from .preprocess.ort_session import cpu_fallback_seen, cuda_provider_available, ensure_onnxruntime
        ensure_onnxruntime()
        from .preprocess.dwpose import DWposeDetector
        from .preprocess.humanparsing.run_parsing import Parsing
        model_root = resolve_vton_models_dir()
        if LOADED_VTON_MODELS is not None:
            release_vton_models()  # sessions cached for the other device go before the new ones are allocated
        # Room is only worth making when the sessions can really use the GPU. Not with the CPU-only wheel, and not
        # once a cuda request was seen served from the CPU in this process (the GPU wheel is installed but CUDA/cuDNN
        # do not start; see create_session): evicting ComfyUI's models would not help those sessions. The very first
        # run cannot know yet and asks once.
        if device == "cuda" and cuda_provider_available() and not cpu_fallback_seen():
            make_room_on_gpu(_VTON_VRAM_ESTIMATE)
        LOADED_VTON_MODELS = {
            'dwprocessor': DWposeDetector(model_root=model_root, device=device),
            'parsing_model': Parsing(model_root=model_root, device=device),
            'device': device
        }
        return LOADED_VTON_MODELS

    def process(self, human_image, category, mask_feathering, cover_shoes, refine_hands, refine_hair, device,
                offset_top=0, offset_bottom=0, offset_left=0, offset_right=0, keep_model_loaded=False):
        # Import trước load_model: thiếu scikit-image thì báo lỗi ngay, không phải tải model xong mới lỗi.
        from ...core.utils_mask import get_mask_location

        if human_image.shape[0] == 0:
            raise ValueError("VTON Ultimate Processor: human_image is an empty batch")
        # Only sessions served by the GPU pin VRAM that ComfyUI cannot manage. Asked after the run, because only
        # then it is known where load_model's sessions ended up: on the CPU (device=cpu, or a cuda request that fell
        # back to it) the cache stays whatever the flag says.
        try:
            return self._process_batch(
                human_image, get_mask_location, category, mask_feathering, cover_shoes, refine_hands, refine_hair,
                device, offset_top, offset_bottom, offset_left, offset_right,
            )
        except BaseException as exc:
            if _should_release(device, keep_model_loaded):
                # The traceback keeps the failed frames alive and they reference the sessions: clear them,
                # or the release below frees nothing until the exception has been handled.
                traceback.clear_frames(exc.__traceback__)
            raise
        finally:
            if _should_release(device, keep_model_loaded):
                release_vton_models()

    def _process_batch(self, human_image, get_mask_location, category, mask_feathering, cover_shoes, refine_hands,
                       refine_hair, device, offset_top, offset_bottom, offset_left, offset_right):
        # The models are only referenced from this frame and _process_image's, so release_vton_models() can free them.
        vton_model = self.load_model(device)
        # Mỗi ảnh trong batch chạy độc lập với cùng setting, rồi nối các output theo chiều batch.
        results = [
            self._process_image(
                vton_model, get_mask_location, human_image[index], index, category, mask_feathering, cover_shoes,
                refine_hands, refine_hair, offset_top, offset_bottom, offset_left, offset_right,
            )
            for index in range(human_image.shape[0])
        ]
        return tuple(torch.cat(outputs, dim=0) for outputs in zip(*results))

    def _process_image(self, vton_model, get_mask_location, human_image, index, category, mask_feathering, cover_shoes,
                       refine_hands, refine_hair, offset_top, offset_bottom, offset_left, offset_right):
        """Xử lý đúng một ảnh [H,W,C]; `index` là vị trí trong batch (chỉ dùng cho thông báo lỗi)."""
        # DWPose và parsing cần RGB 3 kênh: ảnh xám làm hỏng bước đảo BGR, RGBA đưa 4 kênh vào DWPose (alpha bị bỏ).
        human_img_pil = tensor_to_pil(human_image).convert("RGB")
        human_img_resized = resize_image(human_img_pil)

        pose_img_np, _, _, candidate = vton_model['dwprocessor'](np.array(human_img_resized)[:,:,::-1])
        # DWPose không trả mảng rỗng khi không thấy người: nó vẫn chạy pose trên cả ảnh và đánh dấu mọi keypoint là -1.
        if len(candidate) == 0 or not np.any(candidate[0] != -1):
            raise ValueError(
                f"VTON Ultimate Processor: DWPose found no person in the image at batch index {index}. "
                "Use a photo where the person is clearly visible."
            )
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
        if mask_np.ndim == 3:
            mask_np = mask_np[:,:,0]

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

        if refine_hands:
            mask_np = np.clip(mask_np - arm_mask_np, 0, 1)
        if refine_hair:
            mask_np = np.clip(mask_np - hair_mask_np, 0, 1)

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
        # No fromarray(mode="P") (deprecated in Pillow 11.3): the uint8 map is read as "L", and putpalette turns it into "P".
        parsing_map_colored_pil = Image.fromarray(parsing_map_full_np.astype(np.uint8))
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