# Third-Party Notices

NH-Nodes' own code is released under the [MIT License](LICENSE). The components
listed here come from other projects and keep their own licenses. Some of them are
**non-commercial**, so the features that depend on them are not covered by the
MIT license.

> This is a good-faith summary, not legal advice. Check the upstream licenses
> yourself before using these features commercially.

## At a glance

| Feature | Commercial use | Why |
|---------|----------------|-----|
| Every node not listed below | Yes, under MIT | Original NH-Nodes code |
| `VTON Ultimate Processor (NH)` | **No** | Its code and parsing models are CC BY-NC-SA 4.0 (FitDiT / OOTDiffusion) |
| `Portrait Segment (NH)` | **No** | The face-parsing model card restricts use to non-commercial research and education |
| `Garment Segment (NH)` | Unclear | The model uploader does not declare a license |
| `SAM3 Segment (NH)` | Yes, within trade-control limits | Meta's SAM License is royalty-free and has no non-commercial clause, but it forbids ITAR-regulated activities and uses that trade controls prohibit (military or warfare, nuclear, espionage, guns and illegal weapons). The model and the `sam3` package are not part of NH-Nodes: you get them from Meta |

## Source code bundled in this repository

| Path | Origin | License |
|------|--------|---------|
| `nodes/vton/preprocess/humanparsing/` | [OOTDiffusion](https://github.com/levihsu/OOTDiffusion) preprocessing, reused by [FitDiT](https://github.com/BoyuanJiang/FitDiT) and [ComfyUI-Vton-Mask](https://github.com/karthikg-09/ComfyUI-Vton-Mask). Includes parts of [Self-Correction-Human-Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) (MIT) and HRNet utilities (Microsoft, MIT). | CC BY-NC-SA 4.0 — see the `LICENSE` file in that folder |
| `nodes/vton/preprocess/dwpose/` | [DWPose](https://github.com/IDEA-Research/DWPose) ONNX inference as used by ControlNet annotators and FitDiT. Drawing helpers trace back to OpenPose and pytorch-openpose (see file headers). | Apache-2.0 — full text in that folder |
| `core/utils_mask.py` | FitDiT `src/utils_mask.py`, received via ComfyUI-Vton-Mask | CC BY-NC-SA 4.0 |
| `nodes/vton/vton_preprocessor_nodes.py` | NH-Nodes node built on ComfyUI-Vton-Mask's mask generator. That project's `LICENSE` is MIT and says it is based on FitDiT; its README says it follows FitDiT's licensing terms. | Treat as CC BY-NC-SA 4.0 together with the code above |
| `core/utils.py` (`resize_image`, `tensor_to_pil`, `pil_to_tensor`) | Small helpers adapted from ComfyUI-Vton-Mask `utils.py` | MIT |

Every file NH-Nodes changed carries a `Modified by NH-Nodes` header line (for
example, ONNX Runtime sessions in the vendored code are created through
`nodes/vton/preprocess/ort_session.py`).

### License texts

| File | Covers |
|------|--------|
| [`nodes/vton/preprocess/humanparsing/LICENSE`](nodes/vton/preprocess/humanparsing/LICENSE) | CC BY-NC-SA 4.0 notice for the human-parsing code |
| [`nodes/vton/preprocess/dwpose/LICENSE`](nodes/vton/preprocess/dwpose/LICENSE) | Apache License 2.0 (DWPose) |
| [`licenses/Self-Correction-Human-Parsing-MIT.txt`](licenses/Self-Correction-Human-Parsing-MIT.txt) | MIT, Peike Li |
| [`licenses/Microsoft-human-pose-estimation-MIT.txt`](licenses/Microsoft-human-pose-estimation-MIT.txt) | MIT, Microsoft (HRNet transforms) |
| [`licenses/ComfyUI-Vton-Mask-MIT.txt`](licenses/ComfyUI-Vton-Mask-MIT.txt) | MIT, ComfyUI-Vton-Mask (`core/utils.py` helpers) |

## Models downloaded at runtime

These files are **not** stored in this repository. The nodes download them from
Hugging Face on first use, pinned to the revisions below (only the files the nodes need).
They are looked up first in the folders registered as `vton_mask` and `nh_nodes`
(configurable in `extra_model_paths.yaml`).

| Node | Hugging Face repository | Contents | Terms |
|------|-------------------------|----------|-------|
| VTON Ultimate Processor | [`kg-09/kg-vton-mask`](https://huggingface.co/kg-09/kg-vton-mask) @ `4648778` | `dwpose/yolox_l.onnx`, `dwpose/dw-ll_ucoco_384.onnx` | DWPose / YOLOX weights, Apache-2.0 upstream |
| VTON Ultimate Processor | same | `humanparsing/parsing_atr.onnx`, `humanparsing/parsing_lip.onnx` | From the OOTDiffusion/FitDiT checkpoints (CC BY-NC-SA 4.0); trained on the ATR and LIP datasets, which are distributed for research use |
| Portrait Segment | [`jonathandinu/face-parsing`](https://huggingface.co/jonathandinu/face-parsing) @ `758b82e` | SegFormer fine-tuned on CelebAMask-HQ | Model card: "non-commercial research and educational purposes" |
| Garment Segment | [`1038lab/segformer_fashion`](https://huggingface.co/1038lab/segformer_fashion) @ `e5bae2d` | SegFormer trained for Fashionpedia classes | No license declared by the uploader (Fashionpedia annotations are CC BY 4.0) |

## Software and models you install yourself

`SAM3 Segment (NH)` runs Meta's SAM 3. NH-Nodes neither bundles nor downloads it, and the
nodes only load it when you use them. You get both parts from Meta and accept Meta's terms
yourself.

| What | Where it comes from | Terms |
|------|---------------------|-------|
| `sam3` package (code) | [`facebookresearch/sam3`](https://github.com/facebookresearch/sam3), installed by the user with pip | SAM License |
| SAM 3 checkpoint `sam3.pt` | [`facebook/sam3`](https://huggingface.co/facebook/sam3) on Hugging Face, a gated repository: access is granted after the user accepts the license. It goes in the `sam3` model folder (`models/sam3`, or any location added under `sam3` in `extra_model_paths.yaml`), which other SAM 3 packs share | SAM License |

The [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE) (Meta, last
updated 19 November 2025) is a non-exclusive, worldwide, non-transferable, royalty-free
limited license to use, reproduce, distribute, copy, modify and create derivative works of
the SAM Materials. It does not mention commercial use. What matters to a user of NH-Nodes:

- **Trade controls.** You must not be a target of trade controls and must comply with them.
  The materials may not be used for activities subject to the International Traffic in Arms
  Regulations (ITAR), or for end uses that trade controls prohibit, including military or
  warfare purposes, nuclear industries or applications, espionage, and the development or
  use of guns or illegal weapons.
- Whoever redistributes the SAM Materials must include a copy of the license. NH-Nodes does
  not redistribute them.
- Research publications that use the SAM Materials must acknowledge it.
- The materials may not be used to reverse engineer or decompile them, nor to encourage
  others to.
- They are provided "as is", and Meta may terminate the license if you breach it.

## Python dependencies

Dependencies listed in `requirements.txt` are installed from PyPI and are not
redistributed here; each keeps its own license. ONNX Runtime (MIT) is optional
and only needed by the VTON node. The `sam3` package (SAM License) and its extras
(`timm`, `ftfy`, `regex`, `iopath`) are optional too: only `SAM3 Segment (NH)` needs
them, and you install them yourself.
