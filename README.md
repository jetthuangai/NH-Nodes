# NH-Nodes for ComfyUI

A collection of utility nodes for ComfyUI focused on mask operations, image processing, and workflow helpers.

**Author:** [jetthuang.com](https://jetthuang.com)
**Hugging Face:** [huggingface.co/nhathoangfoto](https://huggingface.co/nhathoangfoto)
**Support this work:** [paypal.me/nhathoangfoto](https://paypal.me/nhathoangfoto)

---

## Installation

Copy this folder into `ComfyUI/custom_nodes/`, then install dependencies:

```bash
pip install -r requirements.txt
```

Restart ComfyUI.

---

## Node List

### Mask Operations

| Node | Description |
|---|---|
| **Mask Morphology (NH)** | Advanced morphological operations — independent horizontal/vertical dilation & erosion, hole filling, and Gaussian blur. |
| **Mask Properties (NH)** | Analyzes a mask and outputs bounding box coordinates (x1, y1, x2, y2), width, height, and BBOX. |
| **NH Create Box Mask** | Converts any shape mask into a solid rectangular bounding box mask. |
| **NH Mask Aspect Ratio Match** | Adjusts one mask's bounding box to match the aspect ratio of another. Supports stretch, pad, and crop modes. |

### Image Processing

| Node | Description |
|---|---|
| **NH Agnostic Image Generator** | Removes the masked region and fills it with gray, noise, or blur. Outputs agnostic image, masked image (white fill), and a side-by-side composite preview. Configurable feathering, noise strength, gray value, and blur radius. |
| **NH Mask-Aware Resize** | Resizes image to target width/height while preserving the mask region. Crop mode fills and trims; Pad mode fits and fills padding with a custom color (hex picker). Outputs both resized image and mask. |
| **NH Simple Face Paste** | Pastes a face from a source image onto a destination using mask-guided bounding boxes with feathered alpha blending. |

### Preprocessing

| Node | Description |
|---|---|
| **VTON Ultimate Processor (NH)** | Generates clothing-region masks from a human image using DWPose and human parsing. Supports multiple garment categories, mask refinement (hands, hair, shoes), and directional offset controls. Auto-downloads models on first use. |

### Workflow Utilities

| Node | Description |
|---|---|
| **Multi-Slider (FLOAT)** | 5 float sliders in a single node (-1.0 to 1.0). |
| **Multi-Slider (INT)** | 5 integer sliders in a single node (-100 to 100). |
| **Universal Slider Builder (NH)** | Dynamically creates sliders from a text config. Define name, type, range, and step per line. |
| **Boolean Switch (NH)** | Simple on/off toggle outputting a boolean value. |
| **Pack Universal (NH)** | Packs up to 10 inputs of any type into a single pipe for cleaner workflows. |
| **Unpack Universal (NH)** | Extracts a single item from a universal pipe by index. |

---

## Node Details

### Mask Morphology (NH)

Perform directional mask expansion/contraction with optional hole filling and blur.

- **Inputs:** mask, horizontal_expand (-512~512), vertical_expand (-512~512), fill_holes, blur_radius (0~100)
- **Output:** MASK

### Mask Properties (NH)

Extract geometric info from a mask's non-zero region.

- **Inputs:** mask
- **Outputs:** width, height, x1, y1, x2, y2, bbox

### NH Create Box Mask

Convert an irregular mask shape into a filled rectangle covering its bounding box.

- **Inputs:** mask
- **Output:** box_mask (MASK)

### NH Mask Aspect Ratio Match

Match the aspect ratio of one mask's bounding box to another's.

- **Inputs:** target_ratio_mask, mask_to_adjust, mode (stretch / pad / crop)
- **Output:** MASK

### NH Agnostic Image Generator

Remove the masked region from an image and fill it using one of three modes. Useful as input for inpainting or virtual try-on pipelines.

- **Inputs:** image, mask, fill_mode (gray / noise / blur), blur_radius (3~255), noise_strength (0~1), gray_value (0~1), feathering (0~100)
- **Outputs:**
  - **agnostic_img** — image with masked area filled by chosen mode
  - **masked_img** — image with masked area filled white
  - **composite** — side-by-side preview: left half shows original with green mask overlay, right half shows agnostic result

### NH Mask-Aware Resize

Resize an image to exact dimensions while guaranteeing the mask region stays intact.

- **Inputs:** image, mask, width (64~8192), height (64~8192), mode (crop / pad), pad_color_hex
- **Outputs:** IMAGE, MASK
- **Crop:** Scales up to fill target, then crops around the mask center.
- **Pad:** Scales to fit the mask inside target, pads the rest with the chosen color.

### NH Simple Face Paste

Composite a face region from one image onto another with smooth blending.

- **Inputs:** dest_image, dest_mask, source_image, source_mask, feathering (0~200)
- **Output:** IMAGE

### VTON Ultimate Processor (NH)

Full preprocessing pipeline for garment mask generation.

- **Inputs:** human_image, category (Upper-body / Lower-body / Dresses / Sleeveless / Shorts-Skirt), mask_feathering, cover_shoes, refine_hands, refine_hair, device, offset controls
- **Outputs:** final_mask, agnostic_mask, densepose_image, parsing_image, masked_image, parsing_map_raw, hair_mask, hands_mask

### Multi-Slider (FLOAT) / Multi-Slider (INT)

Compact nodes providing 5 adjustable value sliders each.

### Universal Slider Builder (NH)

Define custom sliders via text config:
```
# name, type, default, min, max, step
strength, FLOAT, 0.5, 0.0, 1.0, 0.01
steps, INT, 20, 1, 100, 1
```

### Boolean Switch (NH)

Simple toggle with labeled on/off states.

### Pack / Unpack Universal (NH)

Bundle up to 10 heterogeneous values into one pipe connection, then extract by index. Reduces wire clutter in complex workflows.

---

## License

MIT
