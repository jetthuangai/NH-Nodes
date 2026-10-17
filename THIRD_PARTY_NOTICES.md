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
Hugging Face on first use.

| Node | Hugging Face repository | Contents | Terms |
|------|-------------------------|----------|-------|
| VTON Ultimate Processor | [`kg-09/kg-vton-mask`](https://huggingface.co/kg-09/kg-vton-mask) | `dwpose/yolox_l.onnx`, `dwpose/dw-ll_ucoco_384.onnx` | DWPose / YOLOX weights, Apache-2.0 upstream |
| VTON Ultimate Processor | same | `humanparsing/parsing_atr.onnx`, `humanparsing/parsing_lip.onnx` | From the OOTDiffusion/FitDiT checkpoints (CC BY-NC-SA 4.0); trained on the ATR and LIP datasets, which are distributed for research use |
| Portrait Segment | [`jonathandinu/face-parsing`](https://huggingface.co/jonathandinu/face-parsing) | SegFormer fine-tuned on CelebAMask-HQ | Model card: "non-commercial research and educational purposes" |
| Garment Segment | [`1038lab/segformer_fashion`](https://huggingface.co/1038lab/segformer_fashion) | SegFormer trained for Fashionpedia classes | No license declared by the uploader (Fashionpedia annotations are CC BY 4.0) |

## Python dependencies

Dependencies listed in `requirements.txt` are installed from PyPI and are not
redistributed here; each keeps its own license. ONNX Runtime (MIT) is optional
and only needed by the VTON node.
