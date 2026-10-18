# Changelog

All notable changes to NH-Nodes. Versions follow the Comfy Registry releases.
"Behaviour change" marks fixes that can change the output of an existing workflow.

## 1.6.2 — Load images matching picks the right group

- `Load images matching` (behaviour change): when the source and the matching library share parent folders (for example
  `D:/Photos/Source/Nu/…` and `D:/Photos/Library/{Nam,Nu}/…`), the shared folders no longer outweigh the source's
  group, so a source in `Nu` picks `Nu/<match>` instead of the first folder in alphabetical order. Same-drive and
  cross-drive setups now rank identically, and folders elsewhere in the library no longer change which match wins.

## 1.6.1 — Cleanup and hardening

- Internal restructuring with no change to node ids, inputs or outputs: the large image-tools module is split into one
  module per node family, shared helpers live in `core/` (`image_folders`, `image_ops`, `mask_ops`, `model_files`), and
  the frontend extensions share one widget helper module.
- Widgets are hidden with the frontend's own `hidden` property when available (older frontends keep the previous
  method); values are still saved with the workflow.
- `Load Images Folder`: the image count used for auto increment now honours `recursive`; a wired `folder_path` no
  longer fingerprints ComfyUI's input folder by mistake.
- HTTP routes: the folder-count and large-preview routes no longer return absolute paths, answer with proper status
  codes (malformed JSON → 400) and do their file work off the server's event loop.
- `NH Save Large Image` accepts single-channel images; bfloat16 IMAGE tensors convert correctly; image conversions no
  longer use Pillow's deprecated `fromarray(mode=…)`.

## 1.6.0 — Performance, caching and VRAM

### Faster graphs
- **Lazy switches:** `If/Else (NH)`, `Switch N (NH)` and `Random Choice (NH)` now ask ComfyUI only for the branch they pick, so the other branches (for example a second sampler) are never executed. The choice itself is unchanged. A connected input whose upstream returns nothing still counts as connected.
- `Layer Stack Composite (NH)` is cached like any pure node instead of re-running on every queue.
- `Load images matching`: results are cached until the matched files change; the search stays inside the matching root (no more scanning parent folders, drive roots or the models folder). When nothing matches, `found` is `False` and `images` returns the new optional `fallback_images` input (evaluated only on a miss); without a fallback, `images` is blocked with a message and the rest of the workflow keeps running (behaviour change: it used to raise an error). Use `fallback_images` rather than routing `images` through a switch — a blocked input stops the switch too. A drive or network-share root cannot be searched recursively any more.
- Indexed loaders (`Load Diffusion Model Index`, `Load VAE Index`, `Load Clip Index`) reuse the model ComfyUI still holds, so editing a slot you did not select no longer reloads it from disk; they keep only a weak reference, so ComfyUI's RAM-pressure cache can still free the model (`--cache-classic` keeps a normal reference). VAEs are loaded with their file metadata and validated like the core loader does.
- `Mask Morphology (NH)`: rewritten on torch (CPU or GPU). Soft masks keep their gradient, masks touching the border no longer shrink from the edge, and large kernels are up to 100× faster. Even kernel lengths act as the next odd length (behaviour change).

### VRAM and models
- `VTON Ultimate Processor (NH)`: new optional `keep_model_loaded` (default **off**). Off frees the ONNX Runtime GPU memory after each run that really used the GPU (about 1.5 s slower per run); on keeps today's speed and is much faster for image lists, which ComfyUI runs one item at a time. ComfyUI is asked to make room before the models are loaded on the GPU, except once ONNX Runtime has fallen back to the CPU.
- SegFormer nodes now really make ComfyUI unload models when VRAM is tight (a device-index mismatch prevented it).
- Model downloads are pinned to tested Hugging Face revisions and only fetch the needed files; a half-downloaded VTON folder is repaired instead of trusted.
- `extra_model_paths.yaml` support: the folder names `nh_nodes` (SegFormer models, `.safetensors`) and `vton_mask` (VTON models, `.onnx`) can point to extra locations; existing installs keep working without moving anything.

### Viewer
- `NH Large Image Preview` / `NH Large Image Compare`: browse every image of a batch with ◀ n / N ▶ or the ←/→ keys.

## 1.5.0 — Correctness fixes

### Image
- **Batches everywhere:** `VTON Ultimate Processor`, `NH Simple Face Paste` and `NH Agnostic Image Generator` now process every image of a batch (smaller inputs with batch 1 are reused for every item). Behaviour change: Face Paste returns the full destination batch instead of one image. VTON stops with a clear error naming the batch index when no person is detected (it used to return a meaningless mask or crash).
- **Values are clipped before 8-bit conversion**, so overshoot from resampling no longer wraps bright pixels to black (Face Paste, Mask-Aware Resize, shared helpers).
- `NH Mask-Aware Resize`: an empty mask now respects the mode (pad = fit and pad with `pad_color_hex`, crop = cover and centre) instead of stretching.
- `NH Agnostic Image Generator`: masks of a different size are resized to the image, 2-D masks are accepted, soft masks keep their gradient, and the `noise` fill is deterministic (behaviour change).
- `Layer Layout Composite` / `Layer Stack Composite`: RGBA layers are pasted with their own alpha instead of as opaque rectangles. RGBA backgrounds give RGBA output.
- Padding colours accept `#RRGGBBAA`; without an alpha part the padding is opaque (it used to copy the blue value into alpha).
- `Save Image Path (NH)`: `filename_prefix` and `new_folder_name` can no longer write outside the chosen folder (on Windows a prefix may not contain `:`, which blocks drive prefixes such as `D:name`); a prefix such as `renders/img` creates the sub-folder. An absolute `folder_path` is still allowed.
- `NH Save Large Image`: WEBP (16383 px) and JPEG (65500 px) size limits are checked before anything is written; `--disable-metadata` is respected.
- `Image Label (NH)`: much faster font fitting (cached fonts + binary search), same sizes as before.
- `Image Grid Composite (NH)`: refuses a canvas that would not fit in the available memory with a clear message instead of crashing.
- `NH Large Preview Cache Manager`: actions run on every queue (it used to be cached).

### Resolution
- `NH Remove Resize Padding`: images resized with `cover_crop` are no longer stretched back (that distorted them). With `original_size` the picture is scaled back uniformly into the original frame and the cropped-away borders are filled from the edges; use `NH Restore Image Geometry` to composite onto the original instead.
- Resize nodes keep float precision for Lanczos/box/hamming (no 8-bit quantisation).
- `NH Smart Resolution Picker`: a stale preset or model name is reported as a warning in `info`.

### Logic, text and lists
- `Math Eval (NH)`: uses a shared safe evaluator. Behaviour change: an invalid expression now shows an error that names the expression instead of silently returning 0. `x / 0`, `x // 0` and `x % 0` give 0 with a warning. numpy/torch scalar inputs work.
- `Random Choice (NH)`: the weight list follows the input slots (`weights` item *i* belongs to `input_i`). Behaviour change when some slots are empty.
- Seeds are used literally in `Random Choice`, `Prompt Scheduler`, `Text Split Lines` and `Text Random Line`, so seed 0 is reproducible (behaviour change).
- `Prompt Template (NH)`: an empty variable no longer glues the neighbouring words together.
- `Regex Extract (NH)`: optional groups that did not match give an empty string instead of "no match".
- `Pack Universal (NH)`: values are stored by slot, so an empty `input_0` no longer shifts the other indices (behaviour change).
- `Compare (NH)` operator `in`: plain membership / substring test (no more fuzzy numeric matching).
- `Batch Merge (NH)`: RGB and RGBA batches can be merged; letterbox bars are transparent for both; very thin inputs no longer crash `pad`.
- `NH Mask Aspect Ratio Match`: `pad`/`crop` placement fixed (no crash at the image edge, requested size honoured) and the whole batch is processed.
- VTON garment masks: undetected keypoints and missing keypoint groups no longer stretch the mask to the image edges; small parsing islands are removed without cutting rectangles out of the garment.
- `Universal Slider Builder (NH)` is marked deprecated (it never produced output); existing workflows still load.

## 1.4.3 — Tooling, licensing and dependencies
- CI runs ruff and the test suite on every pull request; publishing to the Comfy Registry only happens when it passes.
- `onnxruntime` is no longer installed automatically (the CPU wheel overwrote `onnxruntime-gpu`). `VTON Ultimate Processor` explains how to install it and falls back to the CPU with a warning.
- License notices: the VTON code and parsing models are CC BY-NC-SA 4.0 and the face-parsing model is non-commercial — see `THIRD_PARTY_NOTICES.md`.

## 1.4.2
- `NH Smart Resolution Picker`: FLUX.2 latents are 128 channels at 1/16 (images came out twice the size).
- `Load Clip Index (NH)`: works with current ComfyUI (`CLIPType`) and lists every CLIP type.
- `Counter (NH)`: counts on every queue.
- `NH Mask-Aware Resize`: `crop` keeps the whole mask.
