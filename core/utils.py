import torch
import numpy as np
from PIL import Image

def resize_image(image, target_size=768):
    """Resize image while maintaining aspect ratio"""
    w, h = image.size
    if w > h:
        new_w, new_h = target_size, int(h * target_size / w)
    else:
        new_h, new_w = target_size, int(w * target_size / h)
    return image.resize((new_w, new_h), Image.LANCZOS)

def tensor_to_pil(tensor_image):
    """Convert tensor to PIL image"""
    if tensor_image.dim() == 4:
        tensor_image = tensor_image.squeeze(0)
    np_image = (tensor_image.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_image)

def pil_to_tensor(pil_image):
    """Convert PIL image to tensor in ComfyUI format for IMAGE"""
    np_image = np.array(pil_image).astype(np.float32) / 255.0
    if len(np_image.shape) == 2: # Grayscale to RGB
        np_image = np.stack([np_image]*3, axis=-1)
    tensor = torch.from_numpy(np_image).unsqueeze(0)
    return tensor

def pil_to_mask(pil_image):
    """Convert a single-channel PIL image to a MASK tensor"""
    np_image = np.array(pil_image).astype(np.float32) / 255.0
    if len(np_image.shape) == 3: # Ensure single channel
        np_image = np_image[:,:,0]
    tensor = torch.from_numpy(np_image).unsqueeze(0)
    return tensor