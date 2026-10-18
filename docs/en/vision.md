[← Back to guide](./README.md)

# Vision

Eight nodes that cut parts out of a photo. Two use SegFormer models that know people: clothing with Garment Segment, and face, hair and neck with Portrait Segment. Each returns a mask at the size of the input image, a cutout with transparency and a mask preview. The other six work with SAM 3, Meta's segmentation model: SAM3 Segment finds every instance of anything you can name in plain English (people, a wine glass, windows, the sky), and five Segs nodes pick, combine, preview and convert what it found. Find them under **NH-Nodes → Vision**.

**On this page**

- [Which node do I need?](#which-node-do-i-need) — the SegFormer nodes or SAM 3
- **SegFormer nodes: garments and portraits**
  - [Before you start](#before-you-start) — models, first run, licences, memory
  - [Refining the mask](#refining-the-mask) — the options both nodes share
  - [Garment Segment (NH)](#garment-segment-nh) — mask upper garments, lower garments, footwear or headwear
  - [Portrait Segment (NH)](#portrait-segment-nh) — mask face, hair, neck and facial features
- **SAM 3 nodes: anything you can name**
  - [Before you start with SAM 3](#before-you-start-with-sam-3) — GPU, package, checkpoint, memory, licence
  - [How the two passes work](#how-the-two-passes-work) — what `balanced` does on 4K to 8K photos, and where it stops
  - [Instances: the NH_SEGS type](#instances-the-nh_segs-type) — what flows between the nodes
  - [SAM3 Segment (NH)](#sam3-segment-nh) — find every instance of the concepts you name
  - [Segs Select (NH)](#segs-select-nh) — keep instances by label, score, size and position
  - [Segs Compose (NH)](#segs-compose-nh) — build a mask from labels with `|`, `-`, `&` and `~`
  - [Segs To Masks (NH)](#segs-to-masks-nh) — one mask per instance, the union and the labels
  - [Segs To Impact SEGS (NH)](#segs-to-impact-segs-nh) — hand the instances to Impact Pack's detailers
  - [Segs Preview (NH)](#segs-preview-nh) — draw instances on their picture
  - [SAM 3 workflows](#sam-3-workflows) — the background behind people, the biggest object, a facade without windows
  - [If something goes wrong with SAM 3](#if-something-goes-wrong-with-sam-3)

## Which node do I need?

| You want | Use | What it needs |
|---|---|---|
| The clothing of one person: shirt, trousers, shoes, hat | [Garment Segment](#garment-segment-nh) | A SegFormer model that downloads itself on the first run. Runs on the CPU or the GPU. |
| The face, hair, neck and facial features of a portrait | [Portrait Segment](#portrait-segment-nh) | The same. |
| Anything you can name (people, objects, sky, walls, windows), every instance on its own, also on 4K to 8K photos | [SAM3 Segment](#sam3-segment-nh) and the Segs nodes | An NVIDIA GPU, Meta's `sam3` package and a checkpoint that you install and download yourself. See [Before you start with SAM 3](#before-you-start-with-sam-3). |

You can ask SAM3 Segment for `dress` or `hair` too. The SegFormer nodes stay the easier choice for their fixed sets of clothing and face parts: they set themselves up and also run without a GPU.

## Before you start

This section is about Garment Segment and Portrait Segment. SAM3 Segment and the Segs nodes have their own: [Before you start with SAM 3](#before-you-start-with-sam-3).

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

Garment Segment and Portrait Segment turn the model's label map into a mask, then refine it in this order: `fill_holes`, `mask_expand`, `mask_blur`. The mask always has the size of the input image.

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

## Before you start with SAM 3

SAM 3 is Meta's model for finding things by name: you write what you want (`person`, `window`, `wine glass`) and it returns every instance of it, each with a score and its own mask. NH-Nodes does not include the model and never downloads it, so SAM3 Segment needs three things from you, once. If another SAM 3 pack already works in your ComfyUI, steps 2 and 3 are usually done: that pack has installed the package and put `sam3.pt` where this node looks.

**1. An NVIDIA GPU with CUDA.** SAM 3 runs only on CUDA: the official sam3 code builds part of the model on the GPU, so there is no CPU mode and no `device` input. Plan for about 5 GB of VRAM at the peak.

**2. Meta's `sam3` package.** NH-Nodes does not install it for you. In the Python that runs ComfyUI, run these two lines, then restart ComfyUI:

```text
python -m pip install --no-deps git+https://github.com/facebookresearch/sam3@b26a5f330e05d321afb39d01d3d4881f258f65ff
python -m pip install timm ftfy regex iopath
```

The first line installs the sam3 commit that NH-Nodes was tested with; newer commits with the same image API also work. It needs Git on your computer. `--no-deps` keeps your numpy: sam3 asks for numpy below 2, which ComfyUI does not need. Meta's page for the package lists Python 3.12 or newer, PyTorch 2.7 or newer and CUDA 12.6 or newer. If another SAM 3 pack already provides `sam3`, skip the first line: installing it again could replace the copy that pack uses.

**3. The checkpoint `sam3.pt`.** Meta hands out the weights only after you accept its SAM License, so NH-Nodes cannot download them for you. Request access at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), download `sam3.pt` from that page and put it in `ComfyUI/models/sam3/`. Press R in ComfyUI to refresh the lists, then pick the file in `checkpoint`. To keep it somewhere else, add the folder under the `sam3` key in `extra_model_paths.yaml` (see [Installation](../../README.md#installation) for the file):

```yaml
my_models:
    base_path: D:/ai-models/
    sam3: sam3/                    # SAM 3 checkpoints, shared with other SAM 3 packs
```

Only the PyTorch checkpoint `sam3.pt` works. The safetensors export on Hugging Face is made for the transformers library and uses other weight names.

**Memory.** The weights are read from disk once and kept in system memory (about 3.6 GB). They move to the GPU while SAM3 Segment runs and go back afterwards, so ComfyUI gets its VRAM back. Before they move, the node asks ComfyUI to unload some of its own models if VRAM is tight. Turn on `keep_model_loaded` to leave them on the GPU when you run the node again and again.

> **Licence:** SAM 3 is released under Meta's SAM License. It is royalty-free and has no non-commercial clause, but it forbids uses that fall under ITAR or that trade controls prohibit, including military or warfare, nuclear, espionage, and guns or illegal weapons. NH-Nodes' own code is MIT, and it contains neither the model nor the `sam3` package. Read [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) before you use the node commercially.

---

## How the two passes work

SAM 3 looks at a picture of about 1000 px. On a 6000 px photo that is a sixth of the width, so small objects and thin edges would come out soft. SAM3 Segment therefore looks twice.

1. **First pass: the whole image.** The picture is shrunk to the model's square input (1008 px) and encoded once. Every concept you listed is asked against that one encoding, which is cheap. This gives each instance a score, a box and a first mask, which is resampled to full resolution inside its own box only.
2. **Second pass: a close-up of each instance** (`balanced` only). The instance is cut out of the full-resolution image with a margin (20% of its longer side, at least 16 px) and segmented again with SAM 3's box-prompt decoder, with the first mask as a hint. Its edges now come from the original pixels.
3. **A safety net.** A close-up is kept when it matches the first-pass instance to within the blur of the first pass, so thin things such as poles and cables survive. When it does not match, the first-pass mask stays, and the console says for how many instances. Specks of the new mask that do not touch the first-pass mask are dropped.

| `detail_level` | Passes | Use it for |
|---|---|---|
| `balanced` (default) | Both | Crisp edges on small and medium objects, also on 4K to 8K photos. About 0.1 s per instance on a fast GPU. |
| `fast` | The first only | Small pictures, and quick tries while you look for the right words for the concepts. |

**What `balanced` can and cannot do.** Small and medium objects (a window, a glass, one person in a group photo) get crisp edges even on a 4K to 8K photo, because each one is judged on a close-up of the original pixels. The limit is the model: SAM 3 describes whatever it looks at on a grid of about 288 cells. A subject that fills most of an 8K frame is its own close-up, so its edges are accurate to about 1/300 of its size, around 25 px for a subject 7000 px across. That is fine for the outline of a person or a building, and soft on hair, lace and fur. Hair-level edges need a matting step that NH-Nodes does not have yet; a Matte Refine node is planned.

**Measured on one machine** (an NVIDIA RTX 5090): the first run loads the model in about 9 s. A 7360×4912 photo with the concepts `person, wine glass, dress` gave 13 instances in about 3.6 s with `balanced`. A 6720×4480 interior with 8 concepts gave 29 instances in about 6 s. The VRAM peak was about 5 GB. Other GPUs will differ.

---

## Instances: the NH_SEGS type

`NH_SEGS` is the data that SAM3 Segment hands to the Segs nodes. For every image of a batch it holds a list of instances. Each instance keeps its **label** (the concept you wrote), its **score**, a tight **box** and the mask **only inside that box**. A full-size mask is built only when a node outputs a `MASK`, so dozens of instances on an 8K photo stay light.

- Instances come in the order of your concepts, and by score within a concept.
- The Segs nodes treat every image of a batch separately.
- ComfyUI's own nodes do not read `NH_SEGS`. To get masks out, use the `mask` outputs, [Segs Compose](#segs-compose-nh) or [Segs To Masks](#segs-to-masks-nh). To use Impact Pack's detailers, convert with [Segs To Impact SEGS](#segs-to-impact-segs-nh).

---

## SAM3 Segment (NH)

<p align="center"><img src="../assets/nodes/NH_SAM3Segment.svg" alt="SAM3 Segment (NH) node" width="360"></p>

Finds every instance of the things you name and returns them as `NH_SEGS`, together with their combined mask and a count. To see what it found, connect [Segs Preview](#segs-preview-nh).

**Use it when** you want masks of "all the people", "every window", "the sky" or "the wine glasses" in a photo, also a 4K to 8K one, without painting anything. Then use the [Segs nodes](#segs-select-nh) to pick instances, subtract one from another, or turn them into plain masks.

Combo inputs without a stated default start on their first option.

| Input | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | The picture. Colour (RGB) is used and any alpha is dropped. Every image of a batch is processed in turn. |
| `concepts` | STRING | `person` | What to find: short English noun phrases, one per line or separated by commas (`person, red car, window, sky`). Lines that start with `#` are ignored. Each phrase becomes the label of the instances it finds. Write at least one. |
| `threshold` | FLOAT | `0.5` | Range 0.05 to 0.95. The minimum confidence. Lower finds more instances and more mistakes; higher keeps only the sure ones. |
| `detail_level` | COMBO | `balanced` | `balanced` or `fast`. `balanced` segments every instance a second time on a close-up of the full-resolution image, for crisp edges on small and medium objects. `fast` keeps the first, whole-image pass. See [How the two passes work](#how-the-two-passes-work). |
| `max_per_concept` | INT | `0` | Range 0 to 200. Keep at most this many instances of each concept, best score first. `0` keeps all. |
| `checkpoint` | COMBO | — | A SAM 3 `.pt` or `.pth` file from `models/sam3` or another folder registered as `sam3`. Other SAM 3 packs use the same folder. When it holds no file, the list shows `sam3.pt` and the node tells you where to get it when you run it. |
| `keep_model_loaded` | BOOLEAN | `false` | Off: after the run the weights go back to system memory and ComfyUI gets the VRAM back. On: they stay on the GPU (about 3.6 GB), which is faster when the node runs again, but ComfyUI cannot use that VRAM meanwhile. |

| Output | Type | Description |
|---|---|---|
| `segs` | NH_SEGS | The instances of every image, each with its label, score, box and mask. Connect it to the Segs nodes. |
| `mask` | MASK | The union of every instance, one mask per image, at the size of the image. |
| `count` | INT | The number of instances found, over the whole batch. |

<p align="center"><img src="../assets/examples/sam3-segment.jpg" alt="SAM3 Segment instances drawn by Segs Preview, and three masks built from them on a sample photo" width="720"><br><em>SAM3 Segment on a sample photo with the concepts "person, helmet, face", drawn by Segs Preview, then three masks made from the instances with Segs Compose and Segs Select.</em></p>

**How it works**

- **Labels.** The concept you wrote is the label of its instances. The Segs nodes ignore case when they match labels, so spelling is not a problem.
- **Order and duplicates.** Instances come in the order of your concepts, best score first within a concept. Two instances of the same concept whose masks overlap almost completely (90% or more) count as one, and the better score stays. `max_per_concept` then keeps the best of what is left.
- **Seeing the result.** Connect `segs` and the same picture to [Segs Preview](#segs-preview-nh): every instance is tinted, outlined and labelled with its label and score.
- **Nothing found** is not an error: you get no instances, an empty mask and a `count` of 0.
- **Batches.** Every image is processed in turn and has its own list of instances. A progress bar counts the concepts, and you can cancel the run between concepts and between close-ups.
- **Very wide or tall pictures.** SAM 3 first sees the whole picture squeezed into a square. On a picture wider or taller than 4:1 (a panorama), it can miss small things, and the console says so. Split such a picture into tiles with [Image Tile (NH)](./image-compose.md#image-tile-nh) and run the tiles.
- **Setup problems** stop the node with a message that says what to do: a missing package, a missing checkpoint, no CUDA GPU. See [If something goes wrong with SAM 3](#if-something-goes-wrong-with-sam-3).

> **Tip:** Write simple English nouns: `window`, `wine glass`, `red car`. If a concept finds nothing, try another word (an unusual building may answer to `structure` or `monument` rather than `building`) or lower `threshold`.

> **Tip:** Ask for a whole and its parts together (`person, hair, face, dress`), then take them apart with [Segs Compose](#segs-compose-nh): `person - hair`.

> **Licence:** SAM 3 is under Meta's SAM License. See [Before you start with SAM 3](#before-you-start-with-sam-3).

---

## Segs Select (NH)

<p align="center"><img src="../assets/nodes/NH_SegsSelect.svg" alt="Segs Select (NH) node" width="360"></p>

Keeps the instances you want, by label, score, size and position, in the order you choose.

**Use it when** you need only some of what SAM3 Segment found: the biggest object, the second person from the left, only the sure detections, or only the large windows.

| Input | Type | Default | Description |
|---|---|---|---|
| `segs` | NH_SEGS | — | The instances, from SAM3 Segment or another Segs node. |
| `labels` | STRING | empty | The labels to keep, separated by commas. `*` stands for any run of characters and `?` for one character (`*shirt`). Case does not matter, and `_` counts as a space (`red_car` finds `red car`). Labels with spaces need no quotes here: `wine glass, red car`. Empty keeps every label. |
| `min_score` | FLOAT | `0.0` | Range 0 to 1. Drops instances with a lower score. |
| `min_area` | FLOAT | `0.0` | Range 0 to 1. Drops instances smaller than this fraction of the image (`0.01` is 1%). |
| `max_area` | FLOAT | `1.0` | Range 0 to 1. Drops instances larger than this fraction of the image. |
| `sort_by` | COMBO | `score` | The order of the kept instances. See the table below. |
| `start` | INT | `0` | Range 0 to 4096. Skips this many instances of the sorted list. `0` starts at the first. |
| `count` | INT | `0` | Range 0 to 4096. Keeps this many instances from `start` on. `0` keeps all of them. |

| Output | Type | Description |
|---|---|---|
| `segs` | NH_SEGS | The kept instances, in the sorted order. |
| `mask` | MASK | The union of the kept instances, one mask per image. |
| `count` | INT | The number of instances kept, over the whole batch. |

| `sort_by` | Order |
|---|---|
| `score` | Highest score first. |
| `area` | Largest first. |
| `left_to_right`, `right_to_left` | By the horizontal centre of the box. |
| `top_to_bottom`, `bottom_to_top` | By the vertical centre of the box. |
| `center_first` | The instance whose box centre is nearest the centre of the image first. |
| `as_found` | The order SAM3 Segment found them in: your concepts in turn, best score first within each. |

**How it works**

- **Three steps, in this order:** the filters (`labels`, `min_score`, `min_area`, `max_area`), then the sorting, then the slice (`start` and `count`).
- **Per image.** Every image of a batch is handled on its own, so `start` `1` skips the first instance of each image.
- **Area** is the share of the image's pixels that the mask covers, not the size of its box. Positions use the centre of the box. Ties keep the order the instances were found in.
- The kept instances stay sorted, so [Segs To Masks](#segs-to-masks-nh) after this node returns its masks in that order.

| To get | Set |
|---|---|
| The second person from the left | `labels` `person`, `sort_by` `left_to_right`, `start` `1`, `count` `1` |
| The biggest object | `sort_by` `area`, `count` `1` |
| The three surest instances | `sort_by` `score`, `count` `3` |
| Nothing under 1% of the picture | `min_area` `0.01` |
| The instance nearest the middle | `sort_by` `center_first`, `count` `1` |

---

## Segs Compose (NH)

<p align="center"><img src="../assets/nodes/NH_SegsCompose.svg" alt="Segs Compose (NH) node" width="360"></p>

Builds one mask from the labels of the instances with a small formula: union, minus, overlap and "everything except".

**Use it when** you want "the background", "the person without the hair", "the wall without the windows", or any other combination of the things SAM3 Segment found.

| Input | Type | Default | Description |
|---|---|---|---|
| `segs` | NH_SEGS | — | The instances, from SAM3 Segment or Segs Select. |
| `expression` | STRING | `all` | The formula. See below. The field has several lines, and a line break counts as a space. |

| Output | Type | Description |
|---|---|---|
| `mask` | MASK | The result of the formula, one mask per image, at the size of the image. |

**What you can write**

| Write | Meaning |
|---|---|
| `person` | A label: the union of every instance with that label. Case does not matter, `_` counts as a space, and `*` and `?` are wildcards (`*shirt`). |
| `"red car"` | Put a label that contains spaces or an operator character in quotes, `"` or `'`, or write the space as `_` (`red_car`). Without quotes, `red car` is an error. Quotes also turn a word such as `all` or `image` into an ordinary label. |
| `t-shirt` | A hyphen between two letters or digits belongs to the label, so `t-shirt` and `x-ray` need no quotes. An apostrophe inside a word is fine too (`women's`). A label with a space needs quotes, and double quotes when it also has an apostrophe: `"women's dress"`. |
| `a \| b`, `a + b`, `a or b` | Union: everything in a or in b. |
| `a - b` | a without b. Put spaces around the minus: `person-hair` is read as one label. When both `person` and `hair` were found, the node stops with an error that shows `person - hair`; otherwise the unknown label gives an empty part and the console names it. |
| `a & b`, `a and b` | Overlap: only where a and b both are. |
| `~a`, `!a`, `not a` | Everything except a: the whole picture minus a. |
| `( … )` | Grouping. |
| `all` | Every instance in the segs. |
| `image` | The whole picture. |
| `none` | An empty mask. |

`~` binds first, then `&`, then `|` and `-` together, left to right: `a - b | c` is `(a - b) | c`. When in doubt, add parentheses.

| Expression | You get |
|---|---|
| `person - hair` | The people without their hair (ask SAM3 Segment for `person, hair`). |
| `~person` | Everything except the people: the background. |
| `(wall \| ceiling) - window` | The walls and the ceiling with the window openings cut out. |
| `"wine glass" \| person` | The glasses and the people together. |
| `all - person` | Everything that was found except the people. Unlike `~person`, it leaves out the background that nothing found. |

**How it works**

- **A label that no instance has** gives an empty part, and the console says so (`no instance is labelled 'x'; those parts are empty`). A typo in `~persn` therefore gives the whole picture: check the console when a mask looks wrong.
- **Every image** of a batch gets its own mask, built from its own instances.
- **A formula that cannot be read** stops the node with a message that gives the position of the problem, for example `unexpected 'car' at position 5`.

---

## Segs To Masks (NH)

<p align="center"><img src="../assets/nodes/NH_SegsToMasks.svg" alt="Segs To Masks (NH) node" width="360"></p>

Turns the instances into ordinary masks: one mask per instance, the union per image, and the list of labels.

**Use it when** you need each instance as a mask of its own: to process them one by one, to save them, or to read their labels.

| Input | Type | Default | Description |
|---|---|---|---|
| `segs` | NH_SEGS | — | The instances, from SAM3 Segment or Segs Select. |

| Output | Type | Description |
|---|---|---|
| `masks` | MASK | One mask per instance, image after image, in the order of `labels`. When there is no instance at all, a single empty mask, so that nodes after it do not fail. |
| `union` | MASK | One mask per image: the union of its instances. |
| `labels` | STRING | The label of every instance, one per line, in the order of `masks`. |
| `count` | INT | The number of instances. It is 0 when there are none, although `masks` then holds one empty mask. |

> **Note:** Every mask has the size of the picture and 4 bytes per pixel: about 145 MB on a 7360×4912 photo. When the masks would need more than about half of your free RAM (for example hundreds of instances on an 8K photo), the node stops with a message instead of filling the memory. Use [Segs Select](#segs-select-nh) first to keep only the instances you need, or [Segs Compose](#segs-compose-nh) to get one mask.

---

## Segs To Impact SEGS (NH)

<p align="center"><img src="../assets/nodes/NH_SegsToImpactSEGS.svg" alt="Segs To Impact SEGS (NH) node" width="360"></p>

Converts the instances of one image to Impact Pack's `SEGS`, so that its detailers and SEGS tools can work on them.

**Use it when** you already use Impact Pack and want its detailers to refine what SAM3 Segment found. NH-Nodes needs Impact Pack for nothing else, so skip this node if you do not use it.

| Input | Type | Default | Description |
|---|---|---|---|
| `segs` | NH_SEGS | — | The instances, from SAM3 Segment or Segs Select. |
| `crop_factor` | FLOAT | `3.0` | Range 1 to 10. The size of the region a detailer works on around each instance, as a multiple of its box (Impact Pack's `crop_factor`). The region is moved to stay inside the image and always contains the box. |
| `batch_index` | INT | `0` | Range 0 to 4095. Impact Pack's SEGS hold one image, so this picks which image of the batch to convert. A number past the last image stops the node with a message. |

| Output | Type | Description |
|---|---|---|
| `segs` | SEGS | Impact Pack's `SEGS` for that image. Each instance has its box, crop region, mask, label and score (as `confidence`). |

**How it works**

- The output is Impact Pack's own type, not `NH_SEGS`. NH-Nodes builds it with the same fields and does not import Impact Pack, so the conversion needs no Impact Pack. Using the result does.
- To convert several images of a batch, use one node per `batch_index`.

---

## Segs Preview (NH)

<p align="center"><img src="../assets/nodes/NH_SegsPreview.svg" alt="Segs Preview (NH) node" width="360"></p>

Draws the instances on their picture: each one tinted in its own colour, outlined and labelled with its label and score.

**Use it when** you want to see what SAM3 Segment found, or what Segs Select or Segs Compose works with. Put it after SAM3 Segment to see everything, or after Segs Select to see only the selection.

| Input | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | The picture the segs were found in: the same number of images and the same size. Otherwise the node stops and says so. |
| `segs` | NH_SEGS | — | The instances to draw. |

| Output | Type | Description |
|---|---|---|
| `preview` | IMAGE | The picture with the instances drawn on it, at the same size. |

The colours follow the order of the instances, each label sits above the highest point of its instance, and the size of the labels follows the size of the picture. An 8K picture gives an 8K preview.

---

## SAM 3 workflows

In each of these, check what SAM3 Segment found before you edit anything. Connect the same picture and the `segs` to [Segs Preview (NH)](#segs-preview-nh): every instance you want should have an outline and a label. If one is missing, lower `threshold` (for example to 0.3) or try another word for the concept.

### The background behind people

```text
Load Image → SAM3 Segment (concepts: person)
           → Segs Compose (expression: ~person)
           → Mask Morphology (NH)
           → edit the background
```

`~person` is the whole picture except the people. Add what people hold or sit on to both fields, for example the concepts `person, wine glass, chair` and the expression `~(person | "wine glass" | chair)`. Move the edge of the mask a few pixels with [Mask Morphology (NH)](./mask.md#mask-morphology-nh): grow it so that the edit covers the old outline, or shrink it if the edit eats into the people. Then use the mask to blur, relight, recolour or inpaint the background.

### The biggest object, inpainted

```text
Load Image → SAM3 Segment (concepts: person, dog, car)
           → Segs Select (sort_by: area, count: 1)
           → Mask Morphology (NH, grow)
           → inpaint
```

With `labels` empty, Segs Select picks the biggest of everything you asked for. Set `labels` (for example `car`) to pick the biggest of one kind, `start` `1` for the second biggest, and raise `min_score` so that a doubtful detection cannot be picked. Grow the mask a few pixels so that the inpainted area covers the old outline.

### Architecture: a facade without windows

```text
Load Image → SAM3 Segment (concepts: building, window, door)
           → Segs Compose (expression: building - (window | door))
           → recolour, restyle or inpaint the wall
```

`building` covers the whole facade with its windows and doors; the formula cuts them out and leaves the wall. For a room, ask for `wall, ceiling, window` and use `(wall | ceiling) - window`. If `building` finds nothing on an unusual building, try `structure` or `monument` and change the formula to match. On a 4K to 8K photo, keep `balanced`: every window is segmented again on a close-up of the original pixels. The outline of a facade that fills the whole frame is accurate to about 1/300 of its width (see [How the two passes work](#how-the-two-passes-work)), so check the roofline in Segs Preview.

---

## If something goes wrong with SAM 3

| What you see | What to do |
|---|---|
| An error that SAM 3 needs Meta's sam3 package | Install it as in [Before you start with SAM 3](#before-you-start-with-sam-3), then restart ComfyUI. |
| An error that the sam3 package is installed but failed to load, or lacks something | The installed version is not one NH-Nodes supports, or one of its extras is missing. Reinstall with the two lines in [Before you start with SAM 3](#before-you-start-with-sam-3). The message shows the cause. |
| An error that the checkpoint was not found | Download `sam3.pt` as in [Before you start with SAM 3](#before-you-start-with-sam-3), press R in ComfyUI and pick the file. The message lists the folders that were searched. |
| An error that a file is not a SAM 3 `.pt` checkpoint | Use `sam3.pt`. The safetensors files on Hugging Face do not work with this node. |
| An error that a file cannot be read as a SAM 3 checkpoint, or is not one | The file in the `sam3` folder is an incomplete download or another model's `.pt`. Download `sam3.pt` again as in [Before you start with SAM 3](#before-you-start-with-sam-3). |
| An error that SAM 3 needs an NVIDIA GPU with CUDA | ComfyUI is running on the CPU or on another kind of GPU. SAM3 Segment has no CPU mode. |
| An error that you must write at least one concept | `concepts` is empty or holds only comments. Write a word such as `person`. |
| `count` is 0, or an instance is missing | Lower `threshold`, try another word for the concept, and look at the result with Segs Preview. |
| The console says it kept the first-pass mask for some instances | Their close-up did not match the first pass, so those instances keep the first pass's edges. Nothing is wrong. |
| An error from Segs Preview about sizes or the number of images | Connect the picture that the segs were found in. |
| An error from Segs Compose that gives a position | Check the formula at that position. Put labels with spaces in quotes: `"red car"`. |
| A Segs Compose mask is empty, and the console says no instance has a label | Check the spelling of the label. Put spaces around a minus: `person - hair`, not `person-hair`. |
| An error from Segs To Masks that the full-size instance masks would need more memory than is free | There are too many instances for the size of the picture. Use Segs Select to keep only the ones you need, or Segs Compose to get one mask. |
| The console says that SAM 3 sees the picture squeezed into a square and can miss small things | The picture is wider or taller than 4:1. Split it into tiles with Image Tile (NH) and run the tiles. |
| ComfyUI runs out of VRAM after the node | Leave `keep_model_loaded` off, so the weights leave the GPU after each run. |

---

[← Back to guide](./README.md)
