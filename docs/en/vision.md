[← Back to guide](./README.md)

# Vision

Two nodes that cut parts out of a photo of a person with SegFormer models: clothing with Garment Segment, and face, hair and neck with Portrait Segment. Each returns a mask at the size of the input image, a cutout with transparency and a mask preview. Find them under **NH-Nodes → Vision**.

**On this page**

- [Before you start](#before-you-start) — models, first run, licences, memory
- [Refining the mask](#refining-the-mask) — the options both nodes share
- [Garment Segment (NH)](#garment-segment-nh) — mask upper garments, lower garments, footwear or headwear
- [Portrait Segment (NH)](#portrait-segment-nh) — mask face, hair, neck and facial features

## Before you start

| | Garment Segment (NH) | Portrait Segment (NH) |
|---|---|---|
| Model | SegFormer trained for Fashionpedia clothing classes | SegFormer fine-tuned on CelebAMask-HQ face parts |
| Hugging Face repository | `1038lab/segformer_fashion` | `jonathandinu/face-parsing` |
| Download | about 190 MB | about 340 MB |
| Licence | The uploader declares none | Non-commercial research and education only |

**First run.** The first time a node runs, it downloads its model from Hugging Face. Each download is pinned to a fixed revision of the repository, so a later change upstream cannot alter what you get. The files go to `ComfyUI/models/NH-Nodes/`, one sub-folder per model. A complete copy that already exists is used instead: in `models/RMBG/segformer_fashion` for the garment model, in `models/face_parsing` or `models/RMBG/segformer_face` for the face model, or in any folder you add under the `nh_nodes` key in `extra_model_paths.yaml` (see [Installation](../../README.md#installation) for an example). If a download is interrupted, run the node again: finished files are kept.

**Memory.** A model is read from disk once and kept in system memory. It moves to the GPU only while a node runs, and goes back to system memory afterwards, so it does not hold VRAM between runs. Before it moves to the GPU, the node asks ComfyUI to unload some of its own models if VRAM is tight.

> **Note:** If a node reports unexpected labels, a different checkpoint sits in one of the folders it searches. Remove that copy and the right model is downloaded.

> **Licence:** The model behind Portrait Segment may only be used for non-commercial research and education. The model behind Garment Segment declares no licence, so its terms are unclear. NH-Nodes' own code is MIT. Read [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) before you use either node commercially.

## Refining the mask

Both nodes turn the model's label map into a mask, then refine it in this order: `fill_holes`, `mask_expand`, `mask_blur`. The mask always has the size of the input image.

| Option | Range | Garment Segment | Portrait Segment | What it does |
|---|---|---|---|---|
| `fill_holes` | on / off | off | on | Closes holes that the mask completely encloses. It is off for garments by default because an arm across the torso would otherwise be filled in. |
| `mask_expand` | -64 to 64 | `0` | `0` | Grows (positive) or shrinks (negative) the mask by that many pixels. |
| `mask_blur` | 0 to 64 | `0` | `0` | Feathers the edge by about that many pixels. |
| `device` | `cuda`, `cpu` | `cuda` | `cuda` | Where the model runs. Without CUDA the node falls back to the CPU and prints a note once in the console. |

All three outputs are shared by both nodes:

| Output | Type | Description |
|---|---|---|
| `mask` | MASK | The mask, from 0 to 1. Edges are soft when `mask_blur` is above 0. |
| `cutout_rgba` | IMAGE | The input image with the mask as its alpha channel. The colours are left untouched outside the mask. |
| `mask_image` | IMAGE | The mask as a grey RGB image, for previews or for nodes that need an `IMAGE`. |

Batches work: every image is segmented in turn and the outputs keep the batch order.

---

## Garment Segment (NH)

<p align="center"><img src="../assets/nodes/NH_GarmentSegment.svg" alt="Garment Segment (NH) node" width="360"></p>

Masks one group of clothing, or all groups together, on a person in the image.

**Use it when** you want a clean mask of a shirt, trousers, shoes or a hat for inpainting, recolouring or a try-on workflow.

| Input | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | A photo of a person. |
| `part` | COMBO | `upper_garment` | `upper_garment`, `lower_garment`, `footwear`, `headwear` or `all_garments`. See the table below. |
| `attach_parts` | BOOLEAN | `true` | Merges pockets, zippers, bows, ruffles and similar decorations that touch the selected garment. |
| `mask_expand` | INT | `0` | Range -64 to 64. Grows (+) or shrinks (-) the mask by N pixels. |
| `mask_blur` | INT | `0` | Range 0 to 64. Feathers the mask edge by N pixels. |
| `fill_holes` | BOOLEAN | `false` | Fills enclosed holes. |
| `device` | COMBO | `cuda` | `cuda` or `cpu`. |

| Output | Type | Description |
|---|---|---|
| `mask` | MASK | The mask of the selected part. |
| `cutout_rgba` | IMAGE | The input with the mask as alpha. |
| `mask_image` | IMAGE | The mask as a grey RGB preview. |

**What each `part` covers**

| `part` | Included |
|---|---|
| `upper_garment` | Shirts and blouses, tops and T-shirts, sweaters, cardigans, jackets, vests, coats, dresses, jumpsuits, capes, ties, scarves, hoods, collars, lapels, epaulettes, sleeves and necklines. |
| `lower_garment` | Pants, shorts, skirts, belts, buckles, leg warmers, tights and stockings. Shoes are not included. |
| `footwear` | Shoes and socks. |
| `headwear` | Hats, headbands, head coverings and hair accessories. |
| `all_garments` | All four groups together in one mask. |

**How it works**

- **Dresses and jumpsuits** belong to `upper_garment`. There is no separate dress option. For a shirt and trousers together, or a dress and shoes, choose `all_garments`.
- **Attached parts** are pockets, zippers, appliqués, beads, bows, flowers, fringe, ribbons, rivets, ruffles, sequins and tassels. With `attach_parts` on, a piece is added only when it touches the selected garment (within a few pixels). A pocket on a jacket joins an `upper_garment` mask, but a bow on a bag does not.
- **Never selected:** glasses, gloves, watches, bags, wallets and umbrellas have no `part`.

<p align="center"><img src="../assets/examples/garment-segment.jpg" alt="Garment Segment masks overlaid on a sample photo" width="720"><br><em>Garment Segment on a sample photo: the coloured areas are the masks of the selected parts.</em></p>

**Example**

```text
Load Image → Garment Segment (NH, upper_garment) → mask → Mask Morphology (NH) → inpaint
```

Grow the mask a few pixels with [Mask Morphology (NH)](./mask.md#mask-morphology-nh) so the inpainted garment can overlap the old outline.

> **Licence:** the garment model declares no licence. See [Before you start](#before-you-start).

---

## Portrait Segment (NH)

<p align="center"><img src="../assets/nodes/NH_PortraitSegment.svg" alt="Portrait Segment (NH) node" width="360"></p>

Masks the face, hair, neck and facial features of a portrait. You tick the regions you want.

**Use it when** you want to retouch or recolour hair, work on skin and facial features, or protect the face while you edit the rest of the picture.

| Input | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | A portrait photo. |
| `face` | BOOLEAN | `true` | The solid face region: skin, nose, eyes, eyebrows, eyeglasses, lips and the inside of the mouth. |
| `neck` | BOOLEAN | `true` | The neck. |
| `hair` | BOOLEAN | `true` | The hair. |
| `eyes` | BOOLEAN | `true` | Both eyes. |
| `lips` | BOOLEAN | `true` | Upper and lower lip. |
| `nose` | BOOLEAN | `true` | The nose. |
| `eyebrows` | BOOLEAN | `true` | Both eyebrows. |
| `ears` | BOOLEAN | `true` | Both ears. |
| `mouth_interior` | BOOLEAN | `false` | The inside of the mouth, between the lips. |
| `mask_expand` | INT | `0` | Range -64 to 64. Grows (+) or shrinks (-) the mask by N pixels. |
| `mask_blur` | INT | `0` | Range 0 to 64. Feathers the mask edge by N pixels. |
| `fill_holes` | BOOLEAN | `true` | Fills enclosed holes. |
| `device` | COMBO | `cuda` | `cuda` or `cpu`. |

| Output | Type | Description |
|---|---|---|
| `mask` | MASK | The combined mask of every ticked region. |
| `cutout_rgba` | IMAGE | The input with the mask as alpha. |
| `mask_image` | IMAGE | The mask as a grey RGB preview. |

**How it works**

- The ticked regions are combined into one mask. With nothing ticked, the mask is empty.
- `face` already contains eyes, eyebrows, nose, lips and the inside of the mouth. Those options change the result only when `face` is off.
- Hats, earrings, necklaces and clothing exist in the model, but the node does not offer them.

<p align="center"><img src="../assets/examples/portrait-segment.jpg" alt="Portrait Segment masks overlaid on a sample photo" width="720"><br><em>Portrait Segment on a sample photo: the coloured areas are the masks of the ticked regions.</em></p>

**Example: hair only**

```text
Load Image → Portrait Segment (NH) → mask → recolour or inpaint
```

Switch off every region except `hair`. With the defaults you get face, ears, neck and hair together in one mask.

> **Tip:** The model was fine-tuned on CelebAMask-HQ, a set of close-up face photos. It works best when the face fills a good part of the frame. For a small face in a large photo, crop around the face first.

> **Licence:** the face-parsing model is for non-commercial research and education only. See [Before you start](#before-you-start).

---

[← Back to guide](./README.md)
