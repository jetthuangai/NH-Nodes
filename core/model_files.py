"""Model-folder, download and VRAM helpers shared by the SegFormer and VTON model loaders.

Both model families follow the same rules:

* Model folders are registered with ComfyUI's ``folder_paths`` under the names
  ``nh_nodes`` (SegFormer) and ``vton_mask`` (VTON). Users can add more
  locations with those names in ``extra_model_paths.yaml``; every registered
  location is searched before anything is downloaded. Each name is registered
  with the extension of its model files (``.safetensors``, ``.onnx``), so
  ComfyUI's model lists show those files and not the Hugging Face cache
  metadata that downloads leave in the same folder.
* Downloads always go to the default folder under the models dir, pinned to a
  fixed Hugging Face revision.
"""

import os

import torch

try:
    import folder_paths
except Exception:  # running outside ComfyUI (tests)
    folder_paths = None

try:
    import comfy.model_management as _mm
except Exception:
    _mm = None


def models_dir():
    """ComfyUI's models directory, read on every call (--models-directory, tests)."""
    if folder_paths is not None:
        return folder_paths.models_dir
    return os.path.join(os.path.dirname(__file__), "models")


def unique_dirs(paths):
    """Drop repeats (same folder spelled differently) but keep the first spelling and the order."""
    seen, unique = set(), []
    for path in paths:
        key = os.path.normcase(os.path.abspath(path))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _list_only(folder_name, extensions):
    """Add `extensions` to the file types ComfyUI lists for the registered `folder_name`.

    add_model_folder_path registers a name with an empty extension set, which ComfyUI reads as "list
    every file": /models and the Model Library then show the .cache/huggingface metadata of our
    downloads and a repo's .gitattributes. The set is replaced, never changed in place (ComfyUI shares
    one set between many folder names), and the locations list stays the very same object, so every
    extra_model_paths.yaml entry survives.
    """
    if not extensions:
        return
    registry = folder_paths.folder_names_and_paths
    paths, current = registry[folder_name]
    merged = {*current, *extensions}
    if merged != set(current):
        registry[folder_name] = (paths, merged)


def register_model_folder(folder_name, default_dir, extensions=()):
    """Register `folder_name` with ComfyUI and return every folder searched for it, best first.

    ComfyUI applies extra_model_paths.yaml before custom nodes load, so locations the user
    configured under `folder_name` are already registered; they come first. `default_dir`
    is added too (a no-op when already present) and is always part of the result, so models
    that live there are found whether or not the user configured anything. Safe to call repeatedly.

    `extensions` (".onnx") are the file types of the models in these folders: ComfyUI's lists for
    `folder_name` then show only those, also for the locations the user added.
    """
    registered = []
    if folder_paths is not None:
        try:
            folder_paths.add_model_folder_path(folder_name, default_dir)
            registered = folder_paths.get_folder_paths(folder_name)
            _list_only(folder_name, extensions)  # last: if it fails, the locations above are still returned
        except Exception as exc:  # an unusual ComfyUI build: fall back to the default location alone
            print(f"[NH-Nodes] Could not register model folder '{folder_name}' with ComfyUI: {exc}")
    return unique_dirs([*registered, default_dir])


def download_pinned(repo_id, revision, target, allow_patterns):
    """snapshot_download of `allow_patterns` from `repo_id` at `revision` into `target`.

    Finished files are kept and partial ones resume, so calling it again after an interrupted
    download completes the folder. Failures are re-raised with where to get the files by hand.
    """
    from huggingface_hub import snapshot_download

    os.makedirs(target, exist_ok=True)
    print(f"[NH-Nodes] Downloading {repo_id}@{revision[:8]} -> {target}")
    try:
        snapshot_download(repo_id=repo_id, revision=revision, local_dir=target, allow_patterns=list(allow_patterns))
    except Exception as exc:
        raise RuntimeError(
            f"[NH-Nodes] Downloading {repo_id} (revision {revision[:8]}) into {target} failed: {exc}\n"
            f"Run again once the connection works (finished files are kept), or download "
            f"{', '.join(allow_patterns)} from https://huggingface.co/{repo_id}/tree/{revision} into that folder."
        ) from exc


def make_room_on_gpu(nbytes):
    """Ask ComfyUI to unload its own models until about `nbytes` of VRAM are free.

    ComfyUI's own loaders ask for their size plus the VRAM it keeps in reserve (extra_reserved_memory(),
    set by --reserve-vram or its built-in default), so that reserve is added here as well: asking for
    `nbytes` alone would leave us that much short. Also hands PyTorch's cached blocks back to the
    driver: libraries that allocate outside PyTorch (ONNX Runtime) cannot use them. Nothing happens
    without a CUDA device, when ComfyUI itself runs on the CPU, or outside ComfyUI. A failure is
    reported, not raised: the caller still tries its allocation.
    """
    if _mm is None or not torch.cuda.is_available():
        return
    try:
        # ComfyUI's loaded models carry an indexed device (cuda:0); an index-less
        # torch.device("cuda") compares unequal to it and would match none of them.
        device = _mm.get_torch_device()
        if device.type != "cuda":
            return
        reserved = getattr(_mm, "extra_reserved_memory", None)  # absent in older ComfyUI builds
        _mm.free_memory(nbytes + (reserved() if reserved else 0), device)
        _mm.soft_empty_cache()
    except Exception as exc:
        print(f"[NH-Nodes] Could not ask ComfyUI to free VRAM ({exc}); continuing")
