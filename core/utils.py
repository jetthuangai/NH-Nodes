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
    """Convert ONE ComfyUI image ([H,W,C] or [1,H,W,C]) to PIL.

    Values are clipped to [0, 1] first, so overshoot from resampling cannot wrap around
    in uint8. 1/3/4 channels become L/RGB/RGBA. Batches must be converted per image.
    """
    if tensor_image.dim() == 4:
        if tensor_image.shape[0] != 1:
            raise ValueError(
                f"tensor_to_pil expects a single image but got a batch of {tensor_image.shape[0]}; "
                "convert each image of the batch separately"
            )
        tensor_image = tensor_image[0]
    np_image = np.clip(tensor_image.detach().cpu().float().numpy() * 255.0, 0, 255).astype(np.uint8)
    if np_image.ndim == 3 and np_image.shape[2] == 1:
        np_image = np_image[:, :, 0]
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

def parse_hex_color(value, default=(0, 0, 0, 255)):
    """Parse '#RGB', '#RRGGBB' or '#RRGGBBAA' (the '#' is optional) into an (r, g, b, a) tuple of 0-255 ints.

    Without an alpha part the colour is opaque. Text that is not a valid colour returns ``default``.
    """
    text = str(value or "").strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) == 6:
        text += "ff"
    if len(text) != 8:
        return tuple(default)
    try:
        return tuple(int(text[i:i + 2], 16) for i in range(0, 8, 2))
    except ValueError:
        return tuple(default)
