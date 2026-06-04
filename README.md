# NH-Nodes for ComfyUI

Production-focused custom nodes for ComfyUI workflows: image loading, tiling,
compositing, mask utilities, smart resize, logic routing, prompt/text tooling,
batch helpers, indexed model loaders, and VTON preprocessing.

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
Load Image -> NH Smart Resolution Picker -> NH Smart Ratio Image Resize
```

Pick model-oriented target sizes and resize with cover crop, contain pad, or
stretch behavior.

---

## Node Catalog

### Image Loading and Saving

| Node | Purpose |
|---|---|
| `Load Image Info (NH)` | Load one image and output `IMAGE`, `MASK`, file name, and path. |
| `Save Image Path (NH)` | Save image batches to a chosen folder and output saved filenames/path/count. |
| `Load Images Folder (NH)` | Load images from a folder by index/step, with filename and path outputs. |
| `Load images matching` | Load source/matching images by text, folder context, and matching rules. |

### Image Tiling, Grid, and Composition

| Node | Purpose |
|---|---|
| `Image Tile (NH)` | Split image batches into overlapping tiles with metadata for reconstruction. |
| `Image Untile (NH)` | Rebuild full images from processed/upscaled tiles and tile metadata. |
| `Image Grid Composite (NH)` | Combine an image batch into a configurable grid. |
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
| `NH Smart Ratio Image Resize` | Resize images to picked dimensions with cover, contain, or stretch behavior. |

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

### Indexed Loaders

| Node | Purpose |
|---|---|
| `Load LoRA Model Index (NH)` | Apply one selected LoRA to `MODEL` by 1-based index. |
| `Load LoRA Clip Index (NH)` | Apply one selected LoRA to `CLIP` by 1-based index. |
| `Load Diffusion Model Index (NH)` | Load one diffusion model by 1-based index. |

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
  Logic/
    Compare, gates, switches, branching, math, random choice
  Text/
    String, regex, split, concatenate, templates, scheduling
  Batch/
    Image batch and list helpers
  Loaders/
    Indexed LoRA and diffusion model loaders
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

Most image/text/logic nodes use common ComfyUI dependencies. VTON-related nodes
may download or require additional model assets on first use.

---

## Development Notes

- Restart ComfyUI after installing or updating nodes.
- If a node appears cached in the browser, refresh the ComfyUI page after restart.
- `Image Tile (NH)` and `Image Untile (NH)` must pass `tile_data` directly through
  the workflow so reconstruction uses the original tile layout.

---

## License

[MIT](LICENSE). Free for personal and commercial use.

If NH-Nodes helps your workflow, consider starring the repository:
[github.com/jetthuangai/NH-Nodes](https://github.com/jetthuangai/NH-Nodes)
