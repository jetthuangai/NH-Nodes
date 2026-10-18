"""SAM 3 for the NH vision nodes: finding the model, loading it, and segmenting concepts in two passes.

NH-Nodes does not ship SAM 3. It uses Meta's ``sam3`` package, installed separately (see ``INSTALL_HINT``), and the
``sam3.pt`` checkpoint, which Hugging Face hands out only after Meta's license is accepted, so it is never
downloaded automatically. Checkpoints are looked up in the ``sam3`` model folder (``models/sam3`` by default, more
locations in ``extra_model_paths.yaml``); other SAM 3 packs use the same folder, so one copy serves all of them.

Pass 1 looks at the whole image once: the image is resized to the model's square input, encoded once and asked for
every concept. Each instance's mask is sampled back to full resolution inside its own box only.
Pass 2 (``detail_level="balanced"``) cuts each instance out of the full-resolution image with a margin and segments it
again with SAM 3's box-prompt decoder, the pass-1 mask given as a hint. Objects that are small or medium in the frame
then get their edges from the original pixels instead of a ~1000 px view of the whole frame. SAM 3 draws masks on a
grid of about 288 cells across whatever it looks at, so a subject that fills most of an 8K frame still has edges at
about 1/300 of its size; hair-level edges need matting. A close-up that does not show the pass-1 instance is dropped
and the pass-1 mask kept.

The official sam3 code builds some of its lookup tables on the GPU, so SAM 3 runs on CUDA only.
"""

import contextlib
import importlib
import importlib.util
import inspect
import itertools
import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .model_files import make_room_on_gpu, models_dir, register_model_folder
from .segs import SegsFrame, make_segment

try:
    import folder_paths
except Exception:  # running outside ComfyUI (tests)
    folder_paths = None

try:
    import comfy.model_management as _mm
except Exception:
    _mm = None

FOLDER_NAME = "sam3"  # shared with the other SAM 3 packs
# Listed in ComfyUI for the folder: SAM 3 checkpoints in the formats SAM 3 packs use, not Hugging Face cache files.
FOLDER_EXTENSIONS = (".pt", ".pth", ".safetensors")
CHECKPOINT_EXTENSIONS = (".pt", ".pth")  # what sam3's own loader reads (torch.load)
DEFAULT_CHECKPOINT = "sam3.pt"
BPE_FILE = "bpe_simple_vocab_16e6.txt.gz"  # sam3's tokenizer vocabulary, in the package's assets folder
DETAIL_LEVELS = ("balanced", "fast")

SAM3_COMMIT = "b26a5f330e05d321afb39d01d3d4881f258f65ff"  # the sam3 build NH-Nodes is tested with
INSTALL_HINT = (
    "SAM 3 needs Meta's sam3 package, which NH-Nodes does not install for you. In ComfyUI's Python environment run:\n"
    f"  pip install --no-deps git+https://github.com/facebookresearch/sam3@{SAM3_COMMIT}\n"
    "  pip install timm ftfy regex iopath\n"
    "--no-deps keeps your numpy: sam3 asks for numpy<2, which ComfyUI does not need. "
    "Restart ComfyUI afterwards."
)
CHECKPOINT_HINT = (
    "Ask for access at https://huggingface.co/facebook/sam3 (Meta has you accept the SAM License first), "
    "download sam3.pt from that page and put it in {folder}. Then refresh ComfyUI (press R) and pick it."
)

DEDUP_IOU = 0.9  # a second instance of the same concept overlapping a better one this much is a duplicate
# A close-up replaces the pass-1 mask when it shows the same object: at least REFINE_MIN_INSIDE of it lies within the
# pass-1 blur of the pass-1 mask, and it covers at least REFINE_MIN_CORE of the pass-1 core (see close_up_agrees).
REFINE_MIN_INSIDE = 0.8
REFINE_MIN_CORE = 0.6
_PASS1_GRID = 288  # SAM 3's detector draws masks on a 288 x 288 grid of its square view of the whole image
_AGREE_PIXELS = 1 << 21  # masks larger than this are compared at a reduced scale
_WIDE_ASPECT = 4.0  # beyond this aspect ratio SAM 3's square view squeezes the picture enough to miss small things
_CROP_MARGIN = 0.2  # pass-2 crop margin on each side, as a fraction of the instance's longer side
_CROP_MIN_MARGIN = 16  # px
_CROP_MAX_ASPECT = 2.0  # pass-2 crops are widened to at most 2:1, the model squeezes them into a square
_SAMPLE_CHUNK = 1 << 22  # output pixels per grid_sample call when resampling masks
_WORKING_BYTES = 3 << 30  # VRAM for activations and the full-resolution image, on top of the weights

_API = None
_CACHE = {}


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def default_folder():
    """<models dir>/sam3, the folder ComfyUI and the other SAM 3 packs use."""
    return os.path.join(models_dir(), FOLDER_NAME)


def checkpoint_folders():
    """Every folder searched for checkpoints, best first (registers the folder name with ComfyUI)."""
    return register_model_folder(FOLDER_NAME, default_folder(), FOLDER_EXTENSIONS)


def checkpoint_choices():
    """Checkpoint names for the node's list: .pt/.pth files in the sam3 folders, or the default name if none yet."""
    names = set()
    folders = checkpoint_folders()
    if folder_paths is not None:
        try:
            names.update(folder_paths.get_filename_list(FOLDER_NAME))
        except Exception as exc:
            print(f"[NH-Nodes] Could not list the {FOLDER_NAME} model folder: {exc}")
    else:
        for folder in folders:
            if os.path.isdir(folder):
                names.update(os.listdir(folder))
    names = sorted(name for name in names if name.lower().endswith(CHECKPOINT_EXTENSIONS))
    return names or [DEFAULT_CHECKPOINT]


def resolve_checkpoint(name):
    """Full path of checkpoint ``name`` in the sam3 folders; a RuntimeError says where to get it when it is missing."""
    name = str(name or DEFAULT_CHECKPOINT)
    parts = name.replace("\\", "/").split("/")
    if os.path.isabs(name) or ".." in parts:
        raise RuntimeError(f"[NH-Nodes] SAM 3 checkpoint name must be a file in the {FOLDER_NAME} folder: {name!r}")
    if not name.lower().endswith(CHECKPOINT_EXTENSIONS):
        raise RuntimeError(
            f"[NH-Nodes] {name} is not a SAM 3 .pt checkpoint. The safetensors export on Hugging Face is for the "
            "transformers library and uses other weight names; download sam3.pt instead."
        )
    folders = checkpoint_folders()
    path = folder_paths.get_full_path(FOLDER_NAME, name) if folder_paths is not None else None
    if path is None:
        path = next((os.path.join(folder, *parts) for folder in folders
                     if os.path.isfile(os.path.join(folder, *parts))), None)
    if path is None or not os.path.isfile(path):
        raise RuntimeError(
            f"[NH-Nodes] SAM 3 checkpoint {name} not found in: {', '.join(folders)}.\n"
            + CHECKPOINT_HINT.format(folder=folders[0] if folders else default_folder())
        )
    return path


# Make the folder name known to ComfyUI as soon as the pack loads.
checkpoint_folders()


# ---------------------------------------------------------------------------
# The sam3 package and the model
# ---------------------------------------------------------------------------
def _tf32_flags():
    """(matmul, cudnn) TF32 switches, or None when torch refuses to report them (its newer precision API in use)."""
    try:
        return torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    except Exception:
        return None


def _restore_tf32(flags):
    if flags is not None:
        try:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags
        except Exception:
            pass


def _missing_parameters(func, names):
    """Names from ``names`` that ``func`` does not accept (none when its signature cannot be read or takes **kwargs)."""
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return []
    if any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
        return []
    return [name for name in names if name not in parameters]


def ensure_sam3():
    """(build_sam3_image_model, Sam3Processor) from the installed sam3 package.

    A RuntimeError explains what to do when the package is missing, fails to import or lacks the image API
    NH-Nodes uses. Importing sam3's model builder switches TF32 on for the whole process; the previous setting is
    put back, so ComfyUI's own models keep theirs.
    """
    global _API
    if _API is not None:
        return _API
    if importlib.util.find_spec("sam3") is None:
        raise RuntimeError("[NH-Nodes] " + INSTALL_HINT)
    flags = _tf32_flags()
    try:
        builder = importlib.import_module("sam3.model_builder")
        processors = importlib.import_module("sam3.model.sam3_image_processor")
    except Exception as exc:
        raise RuntimeError(f"[NH-Nodes] The sam3 package is installed but failed to load: {exc}\n{INSTALL_HINT}") from exc
    finally:
        _restore_tf32(flags)

    build = getattr(builder, "build_sam3_image_model", None)
    processor_cls = getattr(processors, "Sam3Processor", None)
    missing = []
    if not callable(build):
        missing.append("build_sam3_image_model")
    else:
        missing += [f"build_sam3_image_model({name}=)" for name in _missing_parameters(
            build, ("bpe_path", "checkpoint_path", "load_from_HF", "enable_inst_interactivity", "device", "eval_mode"))]
    if processor_cls is None:
        missing.append("Sam3Processor")
    else:
        missing += [f"Sam3Processor({name}=)" for name in
                    _missing_parameters(processor_cls, ("device", "confidence_threshold"))]
        missing += [f"Sam3Processor.{name}" for name in ("set_image", "set_text_prompt", "reset_all_prompts")
                    if not callable(getattr(processor_cls, name, None))]
    if missing:
        raise RuntimeError(
            f"[NH-Nodes] The installed sam3 package lacks {', '.join(missing)}; it is older or newer than the "
            f"versions NH-Nodes supports. Reinstall it:\n{INSTALL_HINT}"
        )
    _API = (build, processor_cls)
    return _API


def sam3_device():
    """ComfyUI's torch device, which must be a CUDA GPU for SAM 3."""
    if _mm is not None:
        device = _mm.get_torch_device()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError(
            f"[NH-Nodes] SAM 3 needs an NVIDIA GPU with CUDA (ComfyUI runs on {device}): the official sam3 code "
            "builds part of the model on the GPU."
        )
    return device


def bpe_path():
    """The tokenizer vocabulary inside the sam3 package that was imported, whichever copy that is.

    Other packs bundle their own copy of sam3 and may put it first on sys.path (ComfyUI-RMBG does), so "sam3" is not
    always Meta's install. Such a copy can look for the vocabulary in the wrong folder when it is not given one
    (ComfyUI-RMBG's looks one level too high), so NH-Nodes always finds it here and passes it.
    """
    package = sys.modules.get("sam3") or importlib.import_module("sam3")
    folders = [os.path.dirname(package.__file__)] if getattr(package, "__file__", None) else []
    folders += [folder for folder in getattr(package, "__path__", []) if folder not in folders]
    for folder in folders:
        for candidate in (os.path.join(folder, "assets", BPE_FILE), os.path.join(folder, os.pardir, "assets", BPE_FILE)):
            if os.path.isfile(candidate):
                return os.path.normpath(candidate)
    where = folders[0] if folders else "its folder"
    raise RuntimeError(f"[NH-Nodes] The sam3 package in {where} has no assets/{BPE_FILE}; reinstall it:\n{INSTALL_HINT}")


def check_checkpoint(path):
    """Fail early, with the download advice, when ``path`` is not a readable SAM 3 checkpoint.

    sam3 loads weights non-strictly and only prints the names it misses, so another model's .pt in the shared folder
    would run with random weights, and a truncated download fails deep inside torch. Reading the names is quick:
    the file is memory-mapped, not loaded.
    """
    advice = CHECKPOINT_HINT.format(folder=os.path.dirname(path))
    try:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as exc:
        raise RuntimeError(f"[NH-Nodes] {path} cannot be read as a SAM 3 checkpoint ({exc}); it may be an "
                           f"incomplete download.\n{advice}") from exc
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    names = list(state.keys()) if isinstance(state, dict) else []
    del state
    if not any(name.startswith("detector.") for name in names) or not any(name.startswith("tracker.") for name in names):
        raise RuntimeError(f"[NH-Nodes] {path} is not a SAM 3 checkpoint (it has no SAM 3 detector and tracker "
                           f"weights).\n{advice}")


def load_model(checkpoint):
    """The SAM 3 image model for ``checkpoint`` (a name in the sam3 folders); kept in RAM between runs."""
    path = resolve_checkpoint(checkpoint)
    stamp = (os.path.normcase(os.path.abspath(path)), os.path.getmtime(path), os.path.getsize(path))
    if _CACHE.get("model") is not None and _CACHE.get("stamp") == stamp:
        return _CACHE["model"]
    unload()
    build, _ = ensure_sam3()
    device = sam3_device()
    check_checkpoint(path)
    print(f"[NH-Nodes] Loading SAM 3 from {path}")
    with torch.cuda.device(device):  # the tables sam3 builds on "cuda" go to ComfyUI's GPU
        model = build(bpe_path=bpe_path(), checkpoint_path=path, load_from_HF=False,
                      enable_inst_interactivity=True, device="cpu", eval_mode=True)
    if not callable(getattr(model, "predict_inst", None)) or getattr(model, "inst_interactive_predictor", None) is None:
        raise RuntimeError("[NH-Nodes] This sam3 build has no box-prompt decoder (predict_inst). " + INSTALL_HINT)
    transforms = getattr(model.inst_interactive_predictor, "_transforms", None)
    if getattr(transforms, "max_hole_area", 0):
        # sam3 fills holes of close-up masks by default; keep real ones (between an arm and the body), as pass 1 does
        transforms.max_hole_area = 0.0
    model.requires_grad_(False)
    _CACHE.update(model=model, stamp=stamp)
    return model


def _model_device(model):
    return next(model.parameters()).device


def _is_on(current, wanted):
    """True when ``current`` satisfies ``wanted``; an index-less "cuda" accepts any cuda:N."""
    return current.type == wanted.type and (wanted.index is None or current.index == wanted.index)


_TABLE_ATTRIBUTES = ("freqs_cis", "freqs_cis_real", "freqs_cis_imag")  # sam3's rotary tables, plain attributes


def _all_on(model, device):
    return all(_is_on(p.device, device) for p in model.parameters())


def _move_tables(model, device):
    """Move sam3's rotary tables, which model.to() leaves behind because they are not buffers."""
    for module in model.modules():
        for name in _TABLE_ATTRIBUTES:
            value = getattr(module, name, None)
            if isinstance(value, torch.Tensor) and not _is_on(value.device, device):
                setattr(module, name, value.to(device))


def move_model(model, device):
    """Put the weights on ``device``, asking ComfyUI to free VRAM first; back to the CPU if the move fails."""
    if device.type == "cuda":
        weights = 0 if _all_on(model, device) else sum(
            t.numel() * t.element_size() for t in itertools.chain(model.parameters(), model.buffers()))
        make_room_on_gpu(weights + _WORKING_BYTES)
    try:
        model.to(device)
        _move_tables(model, device)
    except Exception:
        offload(model)  # a half-moved model would fail every later run
        raise


def offload(model):
    """Move the weights and sam3's GPU-side tables to the CPU, so nothing holds VRAM between runs.

    The position-encoding caches sam3 fills at build time (about 200 MB on the GPU) are emptied; sam3 rebuilds them
    on the GPU when it needs them again.
    """
    cpu = torch.device("cpu")
    if not _all_on(model, cpu):
        model.to(cpu)
    _move_tables(model, cpu)
    for module in model.modules():
        if type(module).__name__ == "PositionEmbeddingSine" and isinstance(getattr(module, "cache", None), dict):
            module.cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload():
    """Forget the cached model (another checkpoint was chosen)."""
    model = _CACHE.pop("model", None)
    _CACHE.pop("stamp", None)
    if model is not None:
        offload(model)


def _native_bf16():
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # torch without the including_emulation switch
        return torch.cuda.is_bf16_supported()


def _autocast(device):
    """bfloat16 autocast on GPUs with native bfloat16 (Ampere and newer), as Meta runs SAM 3; full precision elsewhere."""
    if device.type == "cuda" and _native_bf16():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# Geometry helpers (model-free, unit tested)
# ---------------------------------------------------------------------------
def to_logits(masks):
    """sam3 returns its masks after a sigmoid (under the key "masks_logits"); turn them back into logits.

    Values outside [0, 1] are taken as logits already, in case a later sam3 returns them as the key says.
    """
    masks = masks.float()
    if masks.numel() and (float(masks.min()) < 0.0 or float(masks.max()) > 1.0):
        return masks
    masks = masks.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(masks) - torch.log1p(-masks)


def sample_region(logits, image_size, region, out_size):
    """Resample the square model-space map ``logits`` [S, S] over ``region`` of the image, bilinearly.

    ``region`` is (x1, y1, x2, y2) in pixels of an image of ``image_size`` (height, width), which the model saw
    squeezed into S x S; the result is [h, w] for ``out_size`` (h, w), each value taken at the centre of its output
    pixel. Runs in row chunks, so a full 8K region needs a few hundred MB at most.
    """
    height, width = image_size
    size = logits.shape[-1]
    x1, y1, x2, y2 = region
    out_h, out_w = int(out_size[0]), int(out_size[1])
    device = logits.device
    xs = x1 + (torch.arange(out_w, device=device, dtype=torch.float32) + 0.5) * ((x2 - x1) / out_w)
    ys = y1 + (torch.arange(out_h, device=device, dtype=torch.float32) + 0.5) * ((y2 - y1) / out_h)
    # model-map pixel index of each centre, as grid_sample coordinates (align_corners=True: -1/+1 = first/last centre)
    grid_x = (xs * (size / width) - 0.5) * (2.0 / max(1, size - 1)) - 1.0
    grid_y = (ys * (size / height) - 0.5) * (2.0 / max(1, size - 1)) - 1.0
    source = logits.float()[None, None]
    result = torch.empty((out_h, out_w), device=device, dtype=torch.float32)
    rows = max(1, _SAMPLE_CHUNK // max(1, out_w))
    for start in range(0, out_h, rows):
        gy = grid_y[start:start + rows]
        grid = torch.stack(torch.broadcast_tensors(grid_x[None, :], gy[:, None]), dim=-1)[None]
        result[start:start + rows] = F.grid_sample(source, grid, mode="bilinear", padding_mode="border",
                                                   align_corners=True)[0, 0]
    return result


def mask_box(mask):
    """Tight (x1, y1, x2, y2) box of a bool [H, W] tensor with exclusive ends; None when empty."""
    rows = torch.nonzero(mask.any(dim=1)).flatten()
    if rows.numel() == 0:
        return None
    cols = torch.nonzero(mask.any(dim=0)).flatten()
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def model_box_to_image(box, size, image_size, margin=1):
    """An S x S model-space box as an image-space region, ``margin`` model pixels wider on each side, in the image."""
    height, width = image_size
    sx, sy = width / size, height / size
    x1, y1, x2, y2 = box
    return (max(0, int((x1 - margin) * sx)), max(0, int((y1 - margin) * sy)),
            min(width, int(math.ceil((x2 + margin) * sx))), min(height, int(math.ceil((y2 + margin) * sy))))


def _fit_span(center, size, limit):
    size = int(round(min(size, limit)))
    start = int(round(center - size / 2.0))
    start = min(max(0, start), limit - size)
    return start, start + size


def crop_region(box, image_size):
    """Pass-2 crop around an image-space box: a margin on each side, widened to at most 2:1, kept inside the image."""
    height, width = image_size
    x1, y1, x2, y2 = box
    margin = max(_CROP_MIN_MARGIN, _CROP_MARGIN * max(x2 - x1, y2 - y1))
    crop_w, crop_h = (x2 - x1) + 2 * margin, (y2 - y1) + 2 * margin
    crop_w, crop_h = max(crop_w, crop_h / _CROP_MAX_ASPECT), max(crop_h, crop_w / _CROP_MAX_ASPECT)
    left, right = _fit_span((x1 + x2) / 2.0, crop_w, width)
    top, bottom = _fit_span((y1 + y2) / 2.0, crop_h, height)
    # the box itself always stays inside, whatever the rounding did
    return min(left, x1), min(top, y1), max(right, x2), max(bottom, y2)


def drop_stray_parts(refined, coarse):
    """``refined`` without the connected parts that do not touch ``coarse`` at all (specks around the object)."""
    count, labels = cv2.connectedComponents(refined.astype(np.uint8), connectivity=8)
    if count <= 2:
        return refined
    touching = np.bincount(labels[coarse], minlength=count) > 0
    touching[0] = False
    return touching[labels]


def close_up_agrees(refined, coarse, tolerance):
    """True when the close-up mask ``refined`` shows the same object as the pass-1 mask ``coarse`` (bool arrays).

    The pass-1 mask is blurred by up to ``tolerance`` px, so plain IoU would turn down a sharp mask of a thin pole or
    cable. Instead most of the close-up must lie within ``tolerance`` of the pass-1 mask, and it must cover most of
    the pass-1 core (pixels deeper than twice ``tolerance`` inside it); thin objects have no core and need only the
    first. Large masks are compared at a reduced scale.
    """
    if not refined.any() or not coarse.any():
        return False
    step = max(1, int(math.sqrt(refined.size / _AGREE_PIXELS)))
    if step > 1 and not refined[::step, ::step].any():
        step = 1  # something thinner than the step: compare at full scale
    refined, coarse = refined[::step, ::step], coarse[::step, ::step]
    tolerance = max(1.0, tolerance / step)
    to_coarse = cv2.distanceTransform((~coarse).astype(np.uint8), cv2.DIST_L2, 3)
    inside = np.count_nonzero(refined & (to_coarse <= tolerance)) / np.count_nonzero(refined)
    core = cv2.distanceTransform(coarse.astype(np.uint8), cv2.DIST_L2, 3) > 2.0 * tolerance
    covered = np.count_nonzero(refined & core) / np.count_nonzero(core) if core.any() else 1.0
    return inside >= REFINE_MIN_INSIDE and covered >= REFINE_MIN_CORE


def _box_overlap(a, b):
    """Intersection box of two (x1, y1, x2, y2) boxes, or None."""
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return (x1, y1, x2, y2) if x1 < x2 and y1 < y2 else None


def find_instances(processor, state, concept, max_instances=0):
    """[(score, logits [S, S], model-space box)] for one concept on an encoded image, best first, duplicates dropped."""
    processor.reset_all_prompts(state)
    output = processor.set_text_prompt(concept, state)
    masks, scores = output["masks_logits"], output["scores"]
    if masks is None or len(scores) == 0:
        return []
    scores = scores.float().flatten()
    found, kept = [], []  # kept: (bool mask, box, area) of accepted instances
    for i in torch.argsort(scores, descending=True).tolist():
        logits = to_logits(masks[i].reshape(masks.shape[-2], masks.shape[-1]))
        inside = logits > 0
        box = mask_box(inside)
        if box is None:
            continue
        area = int(inside.sum())
        duplicate = False
        for other, other_box, other_area in kept:
            overlap = _box_overlap(box, other_box)
            if overlap is None:
                continue
            ox1, oy1, ox2, oy2 = overlap
            inter = int((inside[oy1:oy2, ox1:ox2] & other[oy1:oy2, ox1:ox2]).sum())
            if inter / max(1, area + other_area - inter) >= DEDUP_IOU:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append((inside, box, area))
        found.append((float(scores[i]), logits, box))
        if max_instances and len(found) >= max_instances:
            break
    return found


# ---------------------------------------------------------------------------
# Segmenting
# ---------------------------------------------------------------------------
def _hint_size(model):
    """Side of the low-resolution mask the box-prompt decoder takes as a hint (288 for SAM 3)."""
    encoder = getattr(getattr(getattr(model, "inst_interactive_predictor", None), "model", None),
                      "sam_prompt_encoder", None)
    size = getattr(encoder, "mask_input_size", (288, 288))
    return int(size[0]), int(size[1])


def _upsample(concept, score, logits, box, image_size):
    """Pass 1 only: the instance's mask at full resolution, inside its own region."""
    region = model_box_to_image(box, logits.shape[-1], image_size)
    x1, y1, x2, y2 = region
    mask = sample_region(logits, image_size, region, (y2 - y1, x2 - x1)) > 0
    return make_segment(concept, score, mask, origin=(x1, y1)), False


def _refine(model, processor, image, concept, score, logits, box, image_size, hint_size):
    """Pass 2: segment the instance again on a full-resolution crop, the pass-1 mask as a hint.

    Returns (segment, refined) where ``refined`` is False when the pass-1 mask was kept.
    """
    target = model_box_to_image(box, logits.shape[-1], image_size)
    cx1, cy1, cx2, cy2 = crop_region(target, image_size)
    coarse = (sample_region(logits, image_size, (cx1, cy1, cx2, cy2), (cy2 - cy1, cx2 - cx1)) > 0).cpu().numpy()
    hint = sample_region(logits, image_size, (cx1, cy1, cx2, cy2), hint_size).clamp(-20.0, 20.0)
    state = processor.set_image(image[:, cy1:cy2, cx1:cx2])
    x1, y1, x2, y2 = target
    prompt_box = np.array([x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1], dtype=np.float32)
    masks, _, _ = model.predict_inst(state, box=prompt_box, mask_input=hint[None], multimask_output=False,
                                     return_logits=True)
    refined = np.asarray(masks).reshape(-1, cy2 - cy1, cx2 - cx1)[0] > 0
    refined = drop_stray_parts(refined, coarse)
    accepted = close_up_agrees(refined, coarse, max(image_size) / _PASS1_GRID + 2.0)
    return make_segment(concept, score, refined if accepted else coarse, origin=(cx1, cy1)), accepted


def _image_chw(image):
    """One ComfyUI image [H, W, C] -> float [3, H, W] in [0, 1]."""
    channels = image.shape[-1]
    if channels == 1:
        image = image.expand(-1, -1, 3)
    elif channels not in (3, 4):
        raise ValueError(f"[NH-Nodes] SAM 3 expects an IMAGE with 1, 3 or 4 channels, got {channels}")
    return image[..., :3].permute(2, 0, 1).float().clamp(0.0, 1.0)


def segment_images(model, make_processor, images, concepts, threshold=0.5, detail_level="balanced",
                   max_per_concept=0, device=None, interrupt=None, progress=None):
    """NH_SEGS (one SegsFrame per image) for ``concepts`` found in the ComfyUI IMAGE batch ``images``.

    ``make_processor(threshold)`` returns a Sam3Processor for ``model`` on ``device``; the model must already be
    there. Instances come in concept order, best score first within a concept. ``interrupt()`` runs before each
    concept and each close-up (it raises to stop), ``progress(done, total)`` after each concept of each image.
    """
    device = device or _model_device(model)
    refine = detail_level != "fast"
    hint_size = _hint_size(model) if refine else None
    frames, kept_pass1, warned_wide = [], 0, False
    done, total = 0, len(images) * len(concepts)
    with torch.inference_mode(), _autocast(device):
        for image in images:
            height, width = int(image.shape[0]), int(image.shape[1])
            if not warned_wide and max(height, width) > _WIDE_ASPECT * min(height, width):
                warned_wide = True
                print(f"[NH-Nodes] SAM 3 sees the whole {width}x{height} picture squeezed into a square, so on images "
                      "wider than 4:1 it can miss small things; split such a picture into tiles for better results")
            pixels = _image_chw(image).to(device)
            processor = make_processor(threshold)
            size = int(getattr(processor, "resolution", 1008))
            square = F.interpolate(pixels[None], size=(size, size), mode="bilinear", antialias=True,
                                   align_corners=False)[0].clamp_(0.0, 1.0)
            state = processor.set_image(square)
            segments = []
            for concept in concepts:
                if interrupt is not None:
                    interrupt()
                for score, logits, box in find_instances(processor, state, concept, max_per_concept):
                    if refine:
                        if interrupt is not None:
                            interrupt()
                        seg, refined = _refine(model, processor, pixels, concept, score, logits, box,
                                               (height, width), hint_size)
                        kept_pass1 += 0 if refined else 1
                    else:
                        seg, _ = _upsample(concept, score, logits, box, (height, width))
                    if seg is not None:
                        segments.append(seg)
                done += 1
                if progress is not None:
                    progress(done, total)
            frames.append(SegsFrame(height, width, tuple(segments)))
            del pixels, square, state
    if kept_pass1:
        print(f"[NH-Nodes] SAM 3: kept the first-pass mask for {kept_pass1} instance(s) whose close-up disagreed")
    return tuple(frames)


def segment(images, concepts, threshold=0.5, detail_level="balanced", max_per_concept=0, checkpoint=DEFAULT_CHECKPOINT,
            keep_model_loaded=False, interrupt=None, progress=None):
    """Load SAM 3 and run ``segment_images`` on ComfyUI's GPU; the weights go back to RAM afterwards unless kept."""
    if images.shape[0] == 0:
        raise ValueError("[NH-Nodes] SAM 3 got an empty image batch")
    model = load_model(checkpoint)
    _, processor_cls = ensure_sam3()
    device = sam3_device()
    try:
        move_model(model, device)
        return segment_images(
            model, lambda value: processor_cls(model, device=device, confidence_threshold=value), images, concepts,
            threshold, detail_level, max_per_concept, device, interrupt, progress,
        )
    finally:
        if not keep_model_loaded:
            offload(model)
