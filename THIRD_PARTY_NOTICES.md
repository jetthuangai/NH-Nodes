# Third-Party Notices

NH-Nodes' own code is released under the [MIT License](LICENSE). The components
listed here come from other projects and keep their own licenses. Some of them are
**non-commercial** and others restrict how they may be used, so the features that
depend on them are not covered by the MIT license.

> This is a good-faith summary, not legal advice. Check the upstream licenses
> yourself before using these features commercially.

## At a glance

| Feature | Commercial use | Why |
|---------|----------------|-----|
| Every node not listed below | Yes, under MIT | Original NH-Nodes code |
| `VTON Ultimate Processor (NH)` | **No** | Its code and parsing models are CC BY-NC-SA 4.0 (FitDiT / OOTDiffusion) |
| `Portrait Segment (NH)` | **No** | The face-parsing model card restricts use to non-commercial research and education |
| `Garment Segment (NH)` | **No** | Non-commercial, which its license defines as research or evaluation only (see below). The default model, FASHN Human Parser, is under the NVIDIA Source Code License for SegFormer. The `fashionpedia` model declares no license, but it is fine-tuned from `nvidia/mit-b3`, so the same license plausibly applies |
| `Body Segment (NH)` | Yes, with use restrictions | Meta's Sapiens2 License is royalty-free and has no non-commercial clause, but it prohibits surveillance, biometric processing, identifying people, deepfakes and other uses (listed below), requires you to follow privacy, biometric and trade-control laws, and lets Meta audit your use, end the license and change its terms |
| `SAM3 Segment (NH)` | Yes, within trade-control limits | Meta's SAM License is royalty-free and has no non-commercial clause, but it forbids ITAR-regulated activities and uses that trade controls prohibit (military or warfare, nuclear, espionage, guns and illegal weapons). The model and the `sam3` package are not part of NH-Nodes: you get them from Meta |

## Source code bundled in this repository

| Path | Origin | License |
|------|--------|---------|
| `nodes/vton/preprocess/humanparsing/` | [OOTDiffusion](https://github.com/levihsu/OOTDiffusion) preprocessing, reused by [FitDiT](https://github.com/BoyuanJiang/FitDiT) and [ComfyUI-Vton-Mask](https://github.com/karthikg-09/ComfyUI-Vton-Mask). Includes parts of [Self-Correction-Human-Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) (MIT) and HRNet utilities (Microsoft, MIT). | CC BY-NC-SA 4.0 — see the `LICENSE` file in that folder |
| `nodes/vton/preprocess/dwpose/` | [DWPose](https://github.com/IDEA-Research/DWPose) ONNX inference as used by ControlNet annotators and FitDiT. Drawing helpers trace back to OpenPose and pytorch-openpose (see file headers). | Apache-2.0 — full text in that folder |
| `core/utils_mask.py` | FitDiT `src/utils_mask.py`, received via ComfyUI-Vton-Mask | CC BY-NC-SA 4.0 |
| `nodes/vton/vton_preprocessor_nodes.py` | NH-Nodes node built on ComfyUI-Vton-Mask's mask generator. That project's `LICENSE` is MIT and says it is based on FitDiT; its README says it follows FitDiT's licensing terms. | Treat as CC BY-NC-SA 4.0 together with the code above |
| `core/utils.py` (`resize_image`, `tensor_to_pil`, `pil_to_tensor`) | Small helpers adapted from ComfyUI-Vton-Mask `utils.py` | MIT |
| `core/sapiens2/` | Meta's [Sapiens2](https://github.com/facebookresearch/sapiens2) `sapiens/backbones/standalone/sapiens2.py` at commit `7e5bae8`, kept as `backbone.py` with a `Modified by NH-Nodes` header line | Sapiens2 License — the copy is [`core/sapiens2/LICENSE.md`](core/sapiens2/LICENSE.md) |
| `core/sapiens2_backend.py` (`SegHead`) | NH-Nodes' inference version of Meta's `SegHead` (`sapiens/dense/src/models/heads/seg_head.py` at the same commit): the same layers and layer names, so it loads Meta's released weights | Treated like the file above: Sapiens2 License |

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
| [`core/sapiens2/LICENSE.md`](core/sapiens2/LICENSE.md) | Sapiens2 License (Meta): the bundled Sapiens2 model code and the Sapiens2 weights that Body Segment downloads |

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
| Garment Segment (`model`: `fashn`, the default) | [`fashn-ai/fashn-human-parser`](https://huggingface.co/fashn-ai/fashn-human-parser) @ `1f80c34` | SegFormer-B4 fine-tuned from `nvidia/mit-b4` for 18 human-parsing classes (about 256 MB) | NVIDIA Source Code License for SegFormer: research or evaluation only (see below). Model card: "This model inherits the NVIDIA Source Code License for SegFormer." |
| Garment Segment (`model`: `fashionpedia`) | [`1038lab/segformer_fashion`](https://huggingface.co/1038lab/segformer_fashion) @ `e5bae2d` | SegFormer trained for Fashionpedia classes (about 190 MB); its `config.json` records `_name_or_path: nvidia/mit-b3` | No license declared by the uploader (Fashionpedia annotations are CC BY 4.0). The NVIDIA license plausibly applies, so treat it as research or evaluation only (see below) |
| Body Segment | [`facebook/sapiens2-seg-0.4b`](https://huggingface.co/facebook/sapiens2-seg-0.4b) @ `449b3c5`, [`-0.8b`](https://huggingface.co/facebook/sapiens2-seg-0.8b) @ `196a627`, [`-1b`](https://huggingface.co/facebook/sapiens2-seg-1b) @ `60cf0b3` | `sapiens2_<size>_seg.safetensors` (1.6 / 3.3 / 5.9 GB): Sapiens2 body-part segmentation, 28 classes plus background | Sapiens2 License (see below) |
| Portrait Segment, Garment Segment, Body Segment | [`opencv/face_detection_yunet`](https://huggingface.co/opencv/face_detection_yunet) @ `3cc26e7` | `yunet/face_detection_yunet_2023mar.onnx`, the YuNet face detector (about 230 KB), and its `LICENSE` file | MIT, Copyright (c) 2020 Shiqi Yu (OpenCV Zoo) |

YuNet is the face detector that Portrait Segment, Garment Segment and Body Segment use to find the
people in a picture. It runs on the CPU through OpenCV, and its MIT license does not change the "At a
glance" table: the terms of those nodes come from their models.

### NVIDIA Source Code License for SegFormer (Garment Segment)

Garment Segment's default model, FASHN Human Parser, is a SegFormer-B4 fine-tuned from
`nvidia/mit-b4`. Its [model card](https://huggingface.co/fashn-ai/fashn-human-parser) says: "This
model inherits the NVIDIA Source Code License for SegFormer." Section 3.3 of that
[license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) reads:

> The Work and any derivative works thereof only may be used or intended for use non-commercially.
> Notwithstanding the foregoing, NVIDIA and its affiliates may use the Work and any derivative works
> commercially. As used herein, "non-commercially" means for research or evaluation purposes only.

This is narrower than the everyday meaning of "non-commercial": the license allows research or
evaluation, and nothing else. NH-Nodes' MIT license does not change it.

The `fashionpedia` model (`1038lab/segformer_fashion`) has no model card and its uploader declares no
license, so there is nothing to quote. Its `config.json` records `_name_or_path: nvidia/mit-b3`,
which points to a fine-tune of NVIDIA's SegFormer encoder, so the same license plausibly applies.
Treat it as research or evaluation only too.

### Sapiens2 License (Body Segment)

Body Segment runs Meta's Sapiens2 (April 2026). NH-Nodes bundles the model code with the license
text (`core/sapiens2/`) and downloads the weights from Meta's repositories on first use. It does not
redistribute the weights. The [Sapiens2 License](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md)
(`core/sapiens2/LICENSE.md` is identical to Meta's file at commit `7e5bae8`) is a non-exclusive,
worldwide, non-transferable, royalty-free limited license to use, reproduce, distribute, copy, create
derivative works of and modify the "Sapiens Materials": Meta's models, code and trained weights. It
has no non-commercial clause. You agree to it by using or distributing any part of the materials.
What matters to a user of NH-Nodes:

- **Prohibited uses (section 1.b.vi).** Your use of the materials must not involve or encourage
  others to use them:
  - for purposes of surveillance, including any research or development relating to surveillance;
  - for biometric processing;
  - in any manner that infringes, misappropriates or otherwise violates third-party rights;
  - for attempting to identify or re-identify any individual included in the Sapiens Materials;
  - to create deepfakes, surveillance, or other misleading, deceptive or inappropriate content,
    including impersonating any individual or facilitating false online engagement;
  - to collect, process, disclose, generate or infer health, demographic or other sensitive personal
    or private information about individuals without the rights and consents that applicable laws
    require;
  - in a pornographic or defamatory manner;
  - to engage in the unauthorized or unlicensed practice of any profession, including but not
    limited to financial, legal, medical/health, or related professional practices;
  - to engage in, promote, incite, facilitate, or assist in the planning or development of any
    activities subject to the International Traffic in Arms Regulations (ITAR) or end uses prohibited
    by trade controls, or any other activities that present a risk of death or bodily harm to
    individuals, including those related to military or warfare purposes, nuclear industries or
    applications, espionage, the development or use of guns or illegal weapons or illegal drugs, and
    the operation of critical infrastructure or machinery.
- **Laws (sections 1.b.iii and 1.b.v).** Your use must comply with applicable laws, including privacy,
  data protection and security laws: the EU General Data Protection Regulation, the California
  Consumer Privacy Act, the US data security regulations at 28 C.F.R. Part 202, Brazil's Lei Geral de
  Proteção de Dados, and any law governing the processing of biometric information. It must comply
  with trade controls too, and you must not be the target of trade controls.
- **No reverse engineering (1.b.iv).** Your use must not involve or encourage others to reverse
  engineer, decompile or discover the underlying components of the materials.
- **What Meta may do.** If Meta reasonably believes that you are or are likely to be in violation, it
  may audit your use, storage and distribution of the materials and require you to delete every copy
  (1.c). It may terminate the license if you breach it, and you must then delete the materials and
  stop using them (6). It may also change the terms from time to time, provided the changes are
  similar in spirit to the current ones: they take effect immediately, and continuing to use the
  materials counts as agreeing (8).
- **Redistribution and publications (1.b.i and 1.b.ii).** Whoever distributes the materials, or
  derivative works of them, must do so under this license and include a copy of it. Research
  publications that use the materials must acknowledge it.
- **Liability (2 to 5).** The materials and any output or results from them are provided "as is", and
  you are solely responsible for deciding whether using them is appropriate. If you start proceedings
  against Meta or any entity alleging that the materials or their outputs infringe your intellectual
  property, the license ends, and you must indemnify Meta against third-party claims that arise from
  your use or distribution of the materials.
- **Governing law (7).** California law applies, and the courts of California have exclusive
  jurisdiction.

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
