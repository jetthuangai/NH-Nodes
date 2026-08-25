# NH-Nodes for ComfyUI

Production-focused custom nodes for ComfyUI workflows: image loading, tiling,
compositing, mask utilities, smart resize, logic routing, prompt/text tooling,
batch helpers, indexed model loaders, garment/portrait segmentation, and VTON
preprocessing.

[![ComfyUI Registry](https://img.shields.io/badge/ComfyUI_Registry-nh--nodes-2563eb)](https://registry.comfy.org/nodes/nh-nodes)
[![GitHub](https://img.shields.io/badge/GitHub-jetthuangai%2FNH--Nodes-111827)](https://github.com/jetthuangai/NH-Nodes)
[![License: MIT](https://img.shields.io/badge/License-MIT-16a34a.svg)](LICENSE)

- **Author:** [jetthuang.com](https://jetthuang.com)
- **Hugging Face:** [nhathoangfoto](https://huggingface.co/nhathoangfoto)
- **Support:** [PayPal](https://paypal.me/nhathoangfoto)

---

## Why NH-Nodes

NH-Nodes is built for practical workflow construction. It focuses on reusable
building blocks that remove repetitive graph work:

- Load images with filename/path metadata.
- Split images into tiles, upscale or process tiles, then stitch them back.
- Build image grids, layer composites, comparisons, and labeled previews.
- Resize images and masks with predictable aspect-ratio behavior.
- Route workflow branches with boolean logic and generic data switches.
- Generate prompts from templates, regex, lists, schedules, and text utilities.
- Pick indexed LoRA/diffusion models for larger configurable workflows.
- Prepare VTON masks, agnostic images, DensePose/parsing outputs, and related masks.

---

## Installation

### ComfyUI Manager

Search for `NH-Nodes`, install, then restart ComfyUI.

### Comfy CLI

```bash
comfy node install nh-nodes
```

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jetthuangai/NH-Nodes.git
cd NH-Nodes
pip install -r requirements.txt
```

Restart ComfyUI after installation.

---

## Featured Workflows

### Tile, Process, Untile

Use this when an image is too large for a model, or when every tile needs the
same enhancement/upscale step.

```text
Load Image -> Image Tile (NH) -> upscale/process tiles -> Image Untile (NH)
```

`Image Tile (NH)` supports:

- `original_ratio`: derive tile dimensions from the original image aspect ratio.
- `custom`: use explicit tile width and height.
- `square`: force square tiles, for example `1024 x 1024`.
- Automatic overlap calculation.
- Edge padding so square/custom tile outputs keep the requested tile size.

`Image Untile (NH)` uses `tile_data` from the tile node to stitch results back.
It also handles tiles that were resized or upscaled, such as 2x or 3x per tile.

### Grid Contact Sheet

```text
Image batch -> Image Grid Composite (NH) -> Preview Image
```

Use manual cell size or first-image cell size. Resize modes include `stretch`,
`pad`, and `fill`.

### Metadata-Aware Image Load

```text
Load Image Info (NH) -> image / mask / file_name / path
```

Useful for naming outputs, logging source images, or routing by filename.

### Smart Resolution Resize

```text
Load Image -> NH Smart Ratio Image Resize -> (your pipeline) -> NH Remove Resize Padding
```

`NH Smart Ratio Image Resize` picks the closest model preset itself from
`model_family` + `resolution_level`, so it does not need `NH Smart Resolution
Picker` upstream. Resize with cover crop, contain pad, or stretch behavior.

In `contain_pad` mode it also emits `resize_data` describing the content region,
padding, and scale factor. Feed that into `NH Remove Resize Padding` at the end
of the pipeline to crop the padded border back off (`content_region`) or to
restore the original input size (`original_size`).

Use `NH Smart Resolution Picker` when you need target dimensions and a latent
for text-to-image, instead of resizing an existing image.

### Garment and Portrait Segmentation

```text
Load Image -> Garment Segment (NH) [part=upper_garment] -> mask / cutout_rgba / mask_image
Load Image -> Portrait Segment (NH) [hair + face] -> mask / cutout_rgba / mask_image
```

SegFormer-based masks at input resolution. `cutout_rgba` is the input image
with the mask as alpha (RGB is left untouched outside the mask); `mask_image`
is the mask as an RGB preview. Dresses and jumpsuits belong to `upper_garment`. Both nodes share `mask_expand`, `mask_blur`, and `fill_holes`.
Models are auto-downloaded to `models/NH-Nodes/` on first use (existing
`models/face_parsing` or `models/RMBG/segformer_fashion` copies are reused).

---

## Node Catalog

### Image Loading and Saving

| Node | Purpose |
|---|---|
| `Load Image Info (NH)` | Load one image and output `IMAGE`, `MASK`, file name, and path. |
| `Save Image Path (NH)` | Save image batches to a chosen folder and output saved filenames/path/count. |
| `NH Save Large Image` | Save full-resolution images without attaching heavy full-res previews to the canvas. |
| `Load Images Folder (NH)` | Load images from a folder by index/step, with filename and path outputs. |
| `Load images matching` | Load source/matching images by text, folder context, and matching rules. |

### Image Tiling, Grid, and Composition

| Node | Purpose |
|---|---|
| `Image Tile (NH)` | Split image batches into overlapping tiles with metadata for reconstruction. |
| `Image Untile (NH)` | Rebuild full images from processed/upscaled tiles and tile metadata. |
| `Image Grid Composite (NH)` | Combine an image batch into a configurable grid. |
| `NH Large Image Preview` | Preview large images with lightweight canvas thumbnails and passthrough output. |
| `NH Large Image Compare` | Compare large A/B images with lightweight canvas proxy and tiled slider/side/difference viewer. |
| `NH Large Preview Cache Manager` | Inspect and clean the NH large preview tile cache by age, size, or cache id. |
| `Layer Layout Composite (NH)` | Place layers into preset, grid, or custom layout slots. |
| `Layer Stack Composite (NH)` | Paste multiple layers onto a background using explicit coordinates. |
| `Image Resize Unit (NH)` | Resize by pixel/mm/cm using stretch, pad, crop, or lock-ratio modes. |
| `Image Compare (NH)` | Compare two images side-by-side, top/bottom, split, difference, or overlay. |
| `Image Label (NH)` | Add header/bottom labels with font, padding, color, and outline controls. |
| `NH Agnostic Image Generator` | Create agnostic/masked/composite images for inpaint and VTON workflows. |
| `NH Simple Face Paste` | Paste a source face region into a target using mask feathering. |

### Smart Resolution

| Node | Purpose |
|---|---|
| `NH Smart Resolution Picker` | Pick model/preset-aware dimensions and latent setup. |
| `NH Smart Ratio Image Resize` | Auto-pick the closest model preset for the source aspect, then resize with cover, contain, or stretch; outputs `resize_data` for padding removal. |
| `NH Ratio Preset Image Resize` | Resize to a chosen model preset ratio with crop, pad, or fill behavior. |
| `NH Remove Resize Padding` | Remove the padded border from processed images using resize metadata. |

### Vision Segmentation

| Node | Purpose |
|---|---|
| `Garment Segment (NH)` | Mask one of `upper_garment` (tops, jackets, coats, dresses, ties, scarves, collars, sleeves), `lower_garment` (pants, shorts, skirts, belts, tights; no shoes), `footwear`, or `headwear`. Attached parts (pockets, zippers, bows...) merge only when touching the selected garment. |
| `Portrait Segment (NH)` | Checklist mask of face, neck, hair, eyes, lips, nose, eyebrows, ears (+ optional mouth interior). `face` is the solid face region including features. |

### Mask Tools

| Node | Purpose |
|---|---|
| `Mask Morphology (NH)` | Expand/shrink masks, fill holes, and blur edges. |
| `Mask Properties (NH)` | Output mask bounding box, dimensions, and related measurements. |
| `NH Create Box Mask` | Convert a mask area into a rectangular box mask. |
| `NH Mask Aspect Ratio Match` | Adjust one mask to match another mask's aspect ratio. |
| `NH Mask-Aware Resize` | Resize an image/mask pair while preserving the masked subject region. |

### Logic and Flow Control

| Node | Purpose |
|---|---|
| `Compare (NH)` | Compare values with typed operators. |
| `Logic Gate (NH)` | AND, OR, NOT, XOR, NAND, and NOR boolean logic. |
| `If/Else (NH)` | Select between two inputs by condition. |
| `Switch N (NH)` | Pick one of several generic inputs by index. |
| `Any Switch (NH)` | Boolean switch helper. |
| `Any Branch Switch (NH)` | Route generic data through selected branches. |
| `Gate Switch (NH)` | Pass or block a generic input by boolean state. |
| `Value Match Index (NH)` | Return index of matching configured value. |
| `Math Eval (NH)` | Safe math expression evaluator with numeric/string outputs. |
| `Random Choice (NH)` | Weighted random choice from generic inputs. |

### Text and Prompt Tools

| Node | Purpose |
|---|---|
| `String Operations (NH)` | Strip, case convert, replace, slice, inspect, and test strings. |
| `Prompt Join (NH)` | Join text fragments with separators and empty-line handling. |
| `Text Split (NH)` | Split text by delimiter and expose count/first/last values. |
| `Regex Extract (NH)` | Extract, replace, split, and match with regex patterns. |
| `Text Index (NH)` | Select a line/item from text by index. |
| `Text Concatenate (NH)` | Concatenate multiple text inputs. |
| `Text Split Lines (NH)` | Split multiline text into selected parts and metadata. |
| `Text Random Line (NH)` | Pick a random line from multiline text. |
| `Prompt Template (NH)` | Fill `{placeholder}` variables from connected text values. |
| `Prompt Scheduler (NH)` | Select prompts by step, sequence, ping-pong, or seed. |

### Lists and Batch Helpers

| Node | Purpose |
|---|---|
| `List Create (NH)` | Build an `NH_LIST` from multiline or delimited text. |
| `List Index (NH)` | Select an item from an `NH_LIST`. |
| `List Filter (NH)` | Filter lists with contains, regex, equals, length, and related conditions. |
| `Batch Index (NH)` | Extract one or more images from an image batch. |
| `Batch Merge (NH)` | Merge image batches with resize handling. |
| `Counter (NH)` | Stateful counter for repeated queue runs. |
| `Any List Split (NH)` | Split any incoming list into an `All` output plus up to 10 indexed outputs. |

### Indexed Loaders

| Node | Purpose |
|---|---|
| `Load LoRA Model Index (NH)` | Apply one selected LoRA to `MODEL` by 1-based index. |
| `Load LoRA Clip Index (NH)` | Apply one selected LoRA to `CLIP` by 1-based index. |
| `Load Diffusion Model Index (NH)` | Load one diffusion model by 1-based index. |
| `Load VAE Index (NH)` | Load one VAE by 1-based index or boolean toggle. |
| `Load Clip Index (NH)` | Load one CLIP by 1-based index or boolean toggle, with CLIP type selection. |

### Workflow Utilities

| Node | Purpose |
|---|---|
| `Multi-Slider (FLOAT)` | Five float controls in one node. |
| `Multi-Slider (INT)` | Five integer controls in one node. |
| `Universal Slider Builder (NH)` | Generate dynamic slider outputs from text config. |
| `Boolean Switch (NH)` | Simple boolean toggle. |
| `Pack Universal (NH)` | Bundle generic values into an `NH_UNIVERSAL_PIPE`. |
| `Unpack Universal (NH)` | Extract values from an `NH_UNIVERSAL_PIPE`. |

### VTON Preprocessing

| Node | Purpose |
|---|---|
| `VTON Ultimate Processor (NH)` | Generate masks, agnostic image data, DensePose/parsing outputs, hair/hands masks, and related VTON preprocessing assets. |

---

## Menu Structure

```text
NH-Nodes/
  Image/
    Load/save helpers
    Tile/untile
    Grid and layer composition
    Resize, compare, label
  Mask/
    Morphology, properties, bbox, aspect matching
  Vision/
    Garment and portrait segmentation
  Logic/
    Compare, gates, switches, branching, math, random choice
  Text/
    String, regex, split, concatenate, templates, scheduling
  Batch/
    Image batch, list helpers, any-list split
  Resolution/
    Smart resolution picker, ratio resize, preset resize, padding removal
  Loaders/
    Indexed LoRA, diffusion model, VAE, and CLIP loaders
  Utils/
    Sliders, booleans, universal pipe
  VTON/
    VTON Ultimate Processor
```

---

## Requirements

```text
numpy
Pillow
scipy
opencv-python
scikit-image>=0.18.0
onnxruntime>=1.8.0
transformers>=4.20.0
tqdm>=4.62.0
huggingface_hub>=0.16.0
```

Most image/text/logic nodes use common ComfyUI dependencies. VTON and Vision
nodes download model assets on first use (`transformers` SegFormer weights for
Vision: ~190 MB fashion, ~340 MB face).

---

## Development Notes

- Layout: `core/` holds shared helpers and model backends (no nodes); `nodes/<category>/`
  mirrors the ComfyUI menu, and every module there that defines `NODE_CLASS_MAPPINGS`
  is registered automatically by `__init__.py` - drop a file in the right folder
  and restart. VTON's vendored preprocessing lives in `nodes/vton/preprocess/`.
- Tests: `python -m pytest custom_nodes/NH-Nodes/tests -q` from the ComfyUI root.
- Restart ComfyUI after installing or updating nodes.
- If a node appears cached in the browser, refresh the ComfyUI page after restart.
- `Image Tile (NH)` and `Image Untile (NH)` must pass `tile_data` directly through
  the workflow so reconstruction uses the original tile layout.

---

## License

[MIT](LICENSE). Free for personal and commercial use.

If NH-Nodes helps your workflow, consider starring the repository:
[github.com/jetthuangai/NH-Nodes](https://github.com/jetthuangai/NH-Nodes)
