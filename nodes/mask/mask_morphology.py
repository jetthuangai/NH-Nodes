import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes, gaussian_filter

# Moi lan chi xu ly toi da ~16 trieu diem anh (64 MB moi buffer float32): batch dai (mask video) khong lam bo nho tang vot
_CHUNK_PIXELS = 1 << 24


def _running_extreme(x, radius, dim, reduce_fn):
    """Max (reduce_fn = torch.maximum) hoac min (torch.minimum) trong cua so 2*radius+1 diem tam la moi diem,
    truot doc theo `dim` cua x [B, 1, H, W] (-1 = chieu rong, -2 = chieu cao).

    Bien anh duoc nhan ban (replicate): ngoai mep coi nhu bang diem mep, nen mask cham mep khong bi an tu mep.
    Khong quet tung diem cua cua so (O(radius), max_pool2d tren CPU mat vai giay o kernel 255) ma nhan doi cua so:
    moi buoc la mot reduce_fn giua hai lat cat lech nhau `step` diem, cua so dai 1, 2, 4, ... -> O(log radius) buoc,
    chay tren thiet bi cua tensor. Ket qua chinh xac (max/min khong lam tron).
    """
    size = x.shape[dim]
    radius = min(radius, size - 1)  # cua so rong hon anh thi cung chi thay het truc do
    if radius <= 0:
        return x
    window = 2 * radius + 1
    pad = (radius, radius, 0, 0) if dim == -1 else (0, 0, radius, radius)
    run = F.pad(x, pad, mode="replicate")
    step = 1
    while 2 * step <= window:
        length = run.shape[dim] - step
        run = reduce_fn(run.narrow(dim, 0, length), run.narrow(dim, step, length))
        step *= 2
    # run[i] = reduce(padded[i : i + step]); hai doan dai `step` chong nhau phu dung cua so `window`
    return reduce_fn(run.narrow(dim, 0, size), run.narrow(dim, window - step, size))


def _expand(x, amount, dim):
    """amount > 0: no ra (dilation, max); amount < 0: co lai (erosion, min); 0: giu nguyen (tra ve chinh x).
    |amount| la do dai kernel theo truc do: le thi dung nguyen (5 -> moi ben 2 diem). Chan thi lam tron len so
    le ke tiep (10 -> 11, moi ben 5 diem) de ket qua doi xung; ban cu kernel chan lech nua diem ve mot phia."""
    if amount == 0:
        return x
    reduce_fn = torch.maximum if amount > 0 else torch.minimum
    return _running_extreme(x, abs(amount) // 2, dim, reduce_fn)


class NH_MaskMorphology:
    @classmethod
    def INPUT_TYPES(cls):
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
    DESCRIPTION = (
        "Grow (positive) or shrink (negative) the mask along each axis; the value is the kernel length in pixels, "
        "so 5 moves each edge by 2 px. An even length is rounded up to the next odd one (10 acts like 11) so the "
        "result stays centred. Soft masks keep their gradient and the image border never erodes the mask. "
        "Order: horizontal, vertical, fill_holes (mask > 0.5), then Gaussian blur_radius."
    )

    def process(self, mask, horizontal_expand, vertical_expand, fill_holes, blur_radius):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() != 3:
            raise ValueError(f"Mask Morphology expects a MASK [batch, height, width], got shape {tuple(mask.shape)}")

        # Theo lo tren thiet bi cua tensor (CPU hoac GPU): anh nho gop chung mot lan chay, anh lon tung cai
        source = (mask if mask.is_floating_point() else mask.float()).unsqueeze(1)  # [B, 1, H, W]
        count, _, height, width = source.shape
        if height == 0 or width == 0:
            # Khong co diem anh de no / co / lap / lam mo: tra ve nguyen ven (ban sao) nhu ban scipy cu. F.pad(replicate)
            # khong pad duoc tensor rong nen khong de no chay.
            return (source.squeeze(1).clone(),)
        per_chunk = max(1, _CHUNK_PIXELS // max(1, height * width))
        if count <= per_chunk:
            grown = self._grow(source, horizontal_expand, vertical_expand)
        else:
            grown = torch.empty_like(source)
            for start in range(0, count, per_chunk):
                chunk = source[start : start + per_chunk]
                grown[start : start + per_chunk] = self._grow(chunk, horizontal_expand, vertical_expand)
        result = grown.squeeze(1)

        # 3. Lap day lo trong, 4. lam mo: scipy, tung anh tren CPU (giu nguyen ket qua cu)
        if fill_holes or blur_radius > 0:
            result = self._fill_and_blur(result, fill_holes, blur_radius)
        elif grown is source:
            result = result.clone()  # khong doi gi: van tra ban sao, node phia sau co the sua tai cho
        return (result,)

    @staticmethod
    def _grow(x, horizontal_expand, vertical_expand):
        x = _expand(x, horizontal_expand, -1)  # 1. chieu ngang truoc
        return _expand(x, vertical_expand, -2)  # 2. roi chieu doc

    @staticmethod
    def _fill_and_blur(x, fill_holes, blur_radius):
        arrays = x.detach().cpu()
        if arrays.dtype not in (torch.float32, torch.float64):
            arrays = arrays.float()  # numpy/scipy khong nhan half/bfloat16
        arrays = arrays.numpy()
        out = np.empty_like(arrays)
        for i, m_np in enumerate(arrays):
            if fill_holes:
                m_np = binary_fill_holes(m_np > 0.5).astype(m_np.dtype)
            if blur_radius > 0:
                m_np = gaussian_filter(m_np, sigma=blur_radius)
            out[i] = m_np
        return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)


NODE_CLASS_MAPPINGS = { "NH_MaskAdvancedMorphology": NH_MaskMorphology }
NODE_DISPLAY_NAME_MAPPINGS = { "NH_MaskAdvancedMorphology": "Mask Morphology (NH)" }
