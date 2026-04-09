# NH-Nodes for ComfyUI

**32 custom nodes** to supercharge your ComfyUI workflows — from mask editing and image processing to logic control, prompt building, and batch automation.

[![Registry](https://img.shields.io/badge/ComfyUI_Registry-NH--Nodes-blue)](https://registry.comfy.org/nodes/nh-nodes)
[![GitHub](https://img.shields.io/github/stars/jetthuangai/NH-Nodes?style=social)](https://github.com/jetthuangai/NH-Nodes)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Author:** [jetthuang.com](https://jetthuang.com) | **Hugging Face:** [nhathoangfoto](https://huggingface.co/nhathoangfoto) | **Support:** [PayPal](https://paypal.me/nhathoangfoto)

---

## Installation

**Option A — ComfyUI Manager (recommended):**
> Search `NH-Nodes` in Custom Nodes Manager -> Install -> Restart

**Option B — CLI:**
```bash
comfy node install nh-nodes
```

**Option C — Manual:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jetthuangai/NH-Nodes.git
cd NH-Nodes
pip install -r requirements.txt
# Restart ComfyUI
```

---

## All 32 Nodes at a Glance

### Mask Operations — `NH-Nodes/Mask`

| # | Node | What it does | Inputs | Outputs |
|---|------|-------------|--------|---------|
| 1 | **Mask Morphology** | Expand/shrink mask independently in H/V direction, fill holes, blur edges | `mask`, `horizontal_expand`, `vertical_expand`, `fill_holes`, `blur_radius` | `MASK` |
| 2 | **Mask Properties** | Read bounding box coordinates and dimensions from a mask | `mask` | `width`, `height`, `x1`, `y1`, `x2`, `y2`, `bbox` |
| 3 | **Create Box Mask** | Convert any shape mask into a solid rectangle | `mask` | `box_mask` |
| 4 | **Mask Aspect Ratio Match** | Adjust one mask to match another's aspect ratio | `target_ratio_mask`, `mask_to_adjust`, `mode` | `MASK` |

### Image Processing — `NH-Nodes/Image`

| # | Node | What it does | Inputs | Outputs |
|---|------|-------------|--------|---------|
| 5 | **Agnostic Image Generator** | Remove masked region and fill with gray / noise / blur | `image`, `mask`, `fill_mode`, `blur_radius`, `noise_strength`, `gray_value`, `feathering` | `agnostic_img`, `masked_img`, `composite` |
| 6 | **Mask-Aware Resize** | Resize image to exact W x H while keeping the mask region intact | `image`, `mask`, `width`, `height`, `mode`, `pad_color_hex` | `IMAGE`, `MASK` |
| 7 | **Simple Face Paste** | Paste a face onto another image with feathered blending | `dest_image`, `dest_mask`, `source_image`, `source_mask`, `feathering` | `IMAGE` |

### Logic & Control — `NH-Nodes/Logic`

| # | Node | What it does | Inputs | Outputs |
|---|------|-------------|--------|---------|
| 8 | **Compare** | Compare two values (`==`, `!=`, `>`, `<`, `>=`, `<=`) | `a`, `b`, `op`, `type_cast` | `result` (BOOL), `a_passthrough`, `b_passthrough` |
| 9 | **Logic Gate** | Combine booleans: AND, OR, NOT, XOR, NAND, NOR | `a`, `op`, `b` (optional) | `result` (BOOL) |
| 10 | **If/Else** | Route data based on a condition | `condition`, `if_true`, `if_false` | `result`, `condition_out` |
| 11 | **Switch N** | Pick 1 of up to 10 inputs by index | `index`, `input_0`..`input_9` | `result`, `count` |
| 12 | **Math Eval** | Safe math expressions: `a * 2 + sqrt(b)` | `expression`, `a`, `b`, `c`, `d`, `round_to` | `result_float`, `result_int`, `result_string` |
| 13 | **Random Choice** | Weighted random pick from up to 6 inputs | `seed`, `weights`, `input_0`..`input_5` | `result`, `picked_index`, `probabilities` |

> **Math Eval** supports: `+ - * / // % **`, functions `min`, `max`, `abs`, `round`, `sqrt`, `floor`, `ceil`, `clamp`. No `eval()` — uses safe AST parsing.

### Text & Prompt — `NH-Nodes/Text`

| # | Node | What it does | Inputs | Outputs |
|---|------|-------------|--------|---------|
| 14 | **String Operations** | upper, lower, strip, title, replace, contains, startswith, endswith, length, slice | `text`, `operation`, `param_a`, `param_b` | `text_out`, `bool_out`, `int_out` |
| 15 | **Prompt Join** | Merge up to 5 text inputs with a separator | `separator`, `skip_empty`, `text_1`..`text_5` | `result`, `count` |
| 16 | **Text Split** | Split text by delimiter | `text`, `delimiter`, `max_splits` | `items_joined`, `count`, `first`, `last` |
| 17 | **Regex Extract** | Match, find all, replace, or split with regex | `text`, `pattern`, `mode`, `replacement` | `result`, `matched`, `groups`, `count` |
| 18 | **Prompt Template** | Fill `{placeholders}` in a template string | `template`, `var_names`, `var1`..`var6` | `result`, `missing_vars` |
| 19 | **Prompt Scheduler** | Cycle through prompts by step (sequential / pingpong / random) | `prompts`, `current_step`, `total_steps`, `mode`, `seed` | `current_prompt`, `progress`, `step_index` |

> **Prompt Template** example: `"a {color} {garment}, {style}"` + var_names `"color,garment,style"` -> connect text nodes to var1, var2, var3.

### List & Batch — `NH-Nodes/Batch`

| # | Node | What it does | Inputs | Outputs |
|---|------|-------------|--------|---------|
| 20 | **List Create** | Create a list from multiline text | `text`, `delimiter` | `items` (NH_LIST), `count`, `first`, `last` |
| 21 | **List Index** | Get an item from a list by index | `items`, `index`, `wrap` | `item`, `is_first`, `is_last`, `count` |
| 22 | **List Filter** | Filter a list by condition | `items`, `condition`, `mode` | `passed`, `rejected`, `passed_count`, `rejected_count` |
| 23 | **Batch Index** | Extract slice from an IMAGE batch | `batch`, `index`, `end_index`, `step` | `result`, `original_count` |
| 24 | **Batch Merge** | Concatenate multiple IMAGE batches | `batch_a`, `batch_b`, `batch_c`, `resize_mode` | `result`, `count` |
| 25 | **Counter** | Auto-increment counter across Queue runs | `start`, `step`, `max_value`, `reset` | `current`, `is_done`, `progress`, `remaining` |

> **List Filter** conditions: `contains:dress`, `startswith:img_`, `endswith:.png`, `regex:^\d+`, `equals:red`, `len>5`, `len<=10`.

### Workflow Utilities — `NH-Nodes/Utils`

| # | Node | What it does | Inputs | Outputs |
|---|------|-------------|--------|---------|
| 26 | **Multi-Slider (FLOAT)** | 5 float sliders in one node | `float_1`..`float_5` | 5x `FLOAT` |
| 27 | **Multi-Slider (INT)** | 5 integer sliders in one node | `int_1`..`int_5` | 5x `INT` |
| 28 | **Universal Slider Builder** | Create sliders from text config | `config` (multiline) | Dynamic outputs |
| 29 | **Boolean Switch** | Simple on/off toggle | `boolean_switch` | `boolean` |
| 30 | **Pack Universal** | Bundle up to 10 values of any type into one pipe | `input_0`..`input_9` | `NH_UNIVERSAL_PIPE` |
| 31 | **Unpack Universal** | Extract one value from a pipe by index | `pipe`, `index` | `*` (any type) |

### Preprocessing — `NH-Nodes/VTON`

| # | Node | What it does | Inputs | Outputs |
|---|------|-------------|--------|---------|
| 32 | **VTON Ultimate Processor** | Generate garment masks using DWPose + human parsing | `human_image`, `category`, `mask_feathering`, `cover_shoes`, `refine_hands`, `refine_hair`, `device`, offsets | `final_mask`, `agnostic_mask`, `densepose_image`, `parsing_image`, `masked_img`, `parsing_map_raw`, `hair_mask`, `hands_mask` |

> Categories: Upper-body, Lower-body, Dresses, Upper-body (Sleeveless), Lower-body (Shorts/Skirt). Models auto-download on first use.

---

## Common Workflow Examples

### Example 1: Conditional image routing
```
[Load Image A] -> [Compare (a > b)] -> [If/Else] -> [Save Image]
[Load Image B] ----^                      ^--- if_true / if_false
```

### Example 2: Prompt building with template
```
[String "red"]   -> var1 --\
[String "dress"] -> var2 ---+--> [Prompt Template: "a {color} {garment}, 8k"] -> "a red dress, 8k"
```

### Example 3: Batch processing with counter
```
[Counter (0 to 9)] -> [List Index] -> [Prompt Scheduler] -> [KSampler]
                          ^--- [List Create: "cat\ndog\nbird"]
```

### Example 4: Mask-aware resize for inpainting
```
[Load Image] ---> [Mask-Aware Resize (pad, 512x512, #000000)] -> [Inpaint Model]
[Load Mask]  ---^                                                      ^--- mask output
```

---

## Node Menu Structure in ComfyUI

```
NH-Nodes/
  +-- Batch/
  |     List Create, List Index, List Filter
  |     Batch Index, Batch Merge, Counter
  +-- Image/
  |     Agnostic Image Generator, Mask-Aware Resize, Simple Face Paste
  +-- Logic/
  |     Compare, Logic Gate, If/Else, Switch N, Math Eval, Random Choice
  +-- Mask/
  |     Mask Morphology, Mask Properties, Create Box Mask, Mask Aspect Ratio Match
  +-- Text/
  |     String Operations, Prompt Join, Text Split, Regex Extract
  |     Prompt Template, Prompt Scheduler
  +-- Utils/
  |     Multi-Slider (FLOAT), Multi-Slider (INT)
  |     Universal Slider Builder, Boolean Switch
  |     +-- Pipe/
  |           Pack Universal, Unpack Universal
  +-- VTON/
        VTON Ultimate Processor
```

---

## Requirements

```
numpy, Pillow, scipy, opencv-python
scikit-image >= 0.18.0
onnxruntime >= 1.8.0
transformers >= 4.20.0
tqdm >= 4.62.0
huggingface_hub >= 0.16.0
```

Most are already included with a standard ComfyUI installation.

---

## License

[MIT](LICENSE) - Free for personal and commercial use.

---

**If NH-Nodes helps your workflow, a star on [GitHub](https://github.com/jetthuangai/NH-Nodes) means a lot!**
