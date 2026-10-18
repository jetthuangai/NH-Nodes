[← Back to guide](./README.md)

# Vision

Eight nodes that cut parts out of a photo. Two use SegFormer models that know people: clothing with Garment Segment, and face, hair and neck with Portrait Segment. Each returns a mask at the size of the input image, a cutout with transparency and a mask preview. The other six work with SAM 3, Meta's segmentation model: you type what you want into SAM3 Segment (people, a wine glass, windows, the sky, or a formula such as `person - bag` or `background`) and it returns the mask and shows it on the node. It also keeps every instance it found, and five Segs nodes pick, combine, preview and convert them. Find them under **NH-Nodes → Vision**.

**On this page**

- [Which node do I need?](#which-node-do-i-need) — the SegFormer nodes or SAM 3
- **SegFormer nodes: garments and portraits**
  - [Before you start](#before-you-start) — models, the face detector, first run, licences, memory
  - [How the SegFormer nodes look at a photo](#how-the-segformer-nodes-look-at-a-photo) — why small faces and people get close-ups
  - [Refining the mask](#refining-the-mask) — the options both nodes share
  - [Garment Segment (NH)](#garment-segment-nh) — mask upper garments, lower garments, footwear or headwear
  - [Portrait Segment (NH)](#portrait-segment-nh) — mask face, hair, neck and facial features
- **SAM 3 nodes: anything you can name**
  - [Before you start with SAM 3](#before-you-start-with-sam-3) — GPU, package, checkpoint, memory, licence
  - [How the two passes work](#how-the-two-passes-work) — what `balanced` does on 4K to 8K photos, and where it stops
  - [Instances: the NH_SEGS type](#instances-the-nh_segs-type) — what flows between the nodes
  - [Writing a formula](#writing-a-formula) — what `concepts` and `expression` accept: lists, `-`, `&`, `~`, `background`, Vietnamese words
  - [SAM3 Segment (NH)](#sam3-segment-nh) — type what you want, get its mask and see it on the node
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
| The clothing of one person or of a group: shirt, trousers, shoes, hat | [Garment Segment](#garment-segment-nh) | A SegFormer model and a small face detector that download themselves on the first run. Runs on the CPU or the GPU. |
| The face, hair, neck and facial features of a portrait, a half-body shot or every face of a group photo | [Portrait Segment](#portrait-segment-nh) | The same. |
| Anything you can name (people, objects, sky, walls, windows) as a mask from a short formula (`person - bag`, `background`), or every instance on its own, also on 4K to 8K photos | [SAM3 Segment](#sam3-segment-nh) and the Segs nodes | An NVIDIA GPU, Meta's `sam3` package and a checkpoint that you install and download yourself. See [Before you start with SAM 3](#before-you-start-with-sam-3). |

You can ask SAM3 Segment for `dress` or `hair` too. The SegFormer nodes stay the easier choice for their fixed sets of clothing and face parts: they set themselves up and also run without a GPU.

## Before you start

This section is about Garment Segment and Portrait Segment. SAM3 Segment and the Segs nodes have their own: [Before you start with SAM 3](#before-you-start-with-sam-3).

| | Garment Segment (NH) | Portrait Segment (NH) |
|---|---|---|
| Model | SegFormer trained for Fashionpedia clothing classes | SegFormer fine-tuned on CelebAMask-HQ face parts |
| Hugging Face repository | `1038lab/segformer_fashion` | `jonathandinu/face-parsing` |
| Download | about 190 MB | about 340 MB |
| Licence | The uploader declares none | Non-commercial research and education only |

**First run.** The first time a node runs, it downloads its model from Hugging Face, together with the face detector described below. Each download is pinned to a fixed revision of the repository, so a later change upstream cannot alter what you get. The files go to `ComfyUI/models/NH-Nodes/`, one sub-folder per model. A complete copy that already exists is used instead: in `models/RMBG/segformer_fashion` for the garment model, in `models/face_parsing` or `models/RMBG/segformer_face` for the face model, or in any folder you add under the `nh_nodes` key in `extra_model_paths.yaml` (see [Installation](../../README.md#installation) for an example). If a download is interrupted, run the node again: finished files are kept.

**Face detector.** Both nodes also use YuNet, a small face detector (about 230 KB, MIT licence, from OpenCV Zoo), to find where the people are before they parse a picture. It is downloaded once, pinned to a fixed revision of `opencv/face_detection_yunet`, into `ComfyUI/models/NH-Nodes/yunet/`. It runs on the CPU with the OpenCV that NH-Nodes already needs, so it takes no VRAM. A copy that already sits in a `yunet` sub-folder of any `nh_nodes` folder (`models/NH-Nodes/`, or one you add in `extra_model_paths.yaml`) is used instead; to run offline, put `face_detection_yunet_2023mar.onnx` there yourself. If the detector cannot be loaded (offline on the first run, or an OpenCV that cannot run it), the node prints one warning in the console and parses whole pictures, without the close-ups described in [How the SegFormer nodes look at a photo](#how-the-segformer-nodes-look-at-a-photo). Restart ComfyUI to try again.

**Memory.** A model is read from disk once and kept in system memory. It moves to the GPU only while a node runs, and goes back to system memory afterwards, so it does not hold VRAM between runs. Before it moves to the GPU, the node asks ComfyUI to unload some of its own models if VRAM is tight.

> **Note:** If a node reports unexpected labels, a different checkpoint sits in one of the folders it searches. Remove that copy and the right model is downloaded.

> **Licence:** The model behind Portrait Segment may only be used for non-commercial research and education. The model behind Garment Segment declares no licence, so its terms are unclear. NH-Nodes' own code is MIT. The YuNet face detector is MIT-licensed and changes none of this. Read [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) before you use either node commercially.

## How the SegFormer nodes look at a photo

Both models look at a 512 × 512 picture. On a 6000 px photo, a face that is 600 px wide is only about 50 px wide to the model, so the nodes do not hand it the photo in one piece.

1. **The whole picture first.** A picture with a side longer than twice the model's 512 px is shrunk with area averaging, which is faster than the model's own resize and keeps fine detail clean, and then parsed. This first answer is kept wherever nothing better is found, so whatever the later steps miss keeps at least this result.
2. **People are found.** The face detector looks for faces on two sizes of the picture. Garment Segment also takes the people that the first parse found.
3. **Close-ups refine the answer.** Each face (Portrait Segment) or group of people (Garment Segment) is parsed again on a crop that the model sees at its full 512 px. The crop's labels go into the first answer, but not along the crop's cut edges, where the model sees things cut off, so no straight edge shows in hair or clothes. How far a crop is trusted differs by node, as the Garment Segment and Portrait Segment sections describe.

On a photo longer than 4096 px, all of this is worked out on a copy 4096 px long and the label map is enlarged to the photo at the end: the model sees 512 px at a time, so its labels hold no finer detail, and merging crops at full 8K size would take seconds.

What each node does with this is in its own section: [Garment Segment](#garment-segment-nh) and [Portrait Segment](#portrait-segment-nh).

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

Masks one group of clothing, or all groups together, on the people in the image.

**Use it when** you want a clean mask of a shirt, trousers, shoes or a hat for inpainting, recolouring or a try-on workflow.

| Input | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | A photo of a person or of a group of people. |
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
- **Close-ups of people.** The node parses the whole picture first and keeps that answer. It then finds the people (in that parse and with the face detector) and cuts square crops, the model's own input shape, around each group of them: a row of overlapping crops for a wide group, one crop for a person or a tight group. Each crop is parsed at the model's full 512 px, and a crop grows when people run past one of its edges. This helps people who are small in a large photo. See [How the SegFormer nodes look at a photo](#how-the-segformer-nodes-look-at-a-photo).
- **A crop only refines.** Its labels replace the first answer only where the model is surer, away from the crop's cut edges, and only for parts on people already found. Nothing is invented on an empty backdrop, and no straight crop edge cuts through a garment.

<p align="center"><img src="../assets/examples/garment-segment.jpg" alt="Garment Segment masks overlaid on a sample photo" width="720"><br><em>Garment Segment on a sample photo: the coloured areas are the masks of the selected parts.</em></p>

**Example**

```text
Load Image → Garment Segment (NH, upper_garment) → mask → Mask Morphology (NH) → inpaint
```

Grow the mask a few pixels with [Mask Morphology (NH)](./mask.md#mask-morphology-nh) so the inpainted garment can overlap the old outline.

> **Note:** The Fashionpedia model misses some tops, for example a blouse under a big hat. A better garment model is planned.

> **Licence:** the garment model declares no licence. See [Before you start](#before-you-start).

---

## Portrait Segment (NH)

<p align="center"><img src="../assets/nodes/NH_PortraitSegment.svg" alt="Portrait Segment (NH) node" width="360"></p>

Masks the face, hair, neck and facial features of a portrait, or of every face in a group photo. You tick the regions you want.

**Use it when** you want to retouch or recolour hair, work on skin and facial features, or protect the face while you edit the rest of the picture.

| Input | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | A portrait, a half-body shot or a group photo. |
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
| `mask` | MASK | The combined mask of every ticked region, of every face found. |
| `cutout_rgba` | IMAGE | The input with the mask as alpha. |
| `mask_image` | IMAGE | The mask as a grey RGB preview. |

**How it works**

- The ticked regions are combined into one mask. With nothing ticked, the mask is empty.
- `face` already contains eyes, eyebrows, nose, lips and the inside of the mouth. Those options change the result only when `face` is off.
- Hats, earrings, necklaces and clothing exist in the model, but the node does not offer them.
- **The whole picture first.** It is parsed as a padded square (it keeps its proportions, grey fills the rest) instead of being squeezed into the model's square. This answer stays wherever the crops below do not reach, so a picture in which no face is found, or in which the detector is not available, keeps it as the result. Once faces are found, the face parts (skin, eyes, eyebrows, nose, lips, ears) that this answer shows far from all of them are dropped as mistakes, such as a reflection or a pattern; its hair, neck and clothes stay.
- **Faces.** The face detector finds the faces (at most 32, the largest ones). When it finds none, the picture gets a grey margin and is searched once more, which finds a face that the frame cuts. The parser then checks each face on a tight crop: a hand or a pattern that the detector took for a face is turned down and changes nothing.
- **Two crops per face.** A wide crop shows hair and neck whole. A tight crop, framed like the photos the model learnt on, gives eyes, eyebrows, nose and lips. The wide crop refines what is inside it away from its cut edges, and the tight crop decides only inside an ellipse around the face, so no straight crop edge shows in long hair. Where the crops of two faces overlap, each pixel goes to the nearer face. In a group photo the mask therefore holds the ticked regions of every person.
- **Large faces.** A face that already fills a large part of the frame is left to the whole-picture parse, which shows it large enough. See [How the SegFormer nodes look at a photo](#how-the-segformer-nodes-look-at-a-photo).

<p align="center"><img src="../assets/examples/portrait-segment.jpg" alt="Portrait Segment masks overlaid on a sample photo" width="720"><br><em>Portrait Segment on a sample photo: the coloured areas are the masks of the ticked regions.</em></p>

**Example: hair only**

```text
Load Image → Portrait Segment (NH) → mask → recolour or inpaint
```

Switch off every region except `hair`. With the defaults you get face, ears, neck and hair together in one mask.

> **Tip:** The model was fine-tuned on CelebAMask-HQ, a set of close-up face photos, so the node parses each face it finds on a close-up crop. A small face in a large photo, a half-body shot or a group photo needs no cropping first. A face the detector misses (very small, hidden or turned away) keeps the whole-picture parse, which is coarser: crop around it yourself when you need its finest detail.

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

**Memory.** The weights are read from disk once and kept in system memory (about 3.6 GB). They move to the GPU while SAM3 Segment runs and go back afterwards, so ComfyUI gets its VRAM back. Before they move, the node asks ComfyUI to unload some of its own models if VRAM is tight. Turn on `keep_model_loaded` to leave them on the GPU when you run the node again and again. While it is on, ComfyUI's **Unload Models** button does not release them: turn `keep_model_loaded` off and run the node once, or restart ComfyUI.

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

`NH_SEGS` is the data that SAM3 Segment hands to the Segs nodes. For every image of a batch it holds a list of instances. Each instance keeps its **label** (the English words SAM 3 was asked for: `áo đỏ` is stored as `red top`), its **score**, a tight **box** and the mask **only inside that box**. A full-size mask is built only when a node outputs a `MASK`, so dozens of instances on an 8K photo stay light.

- Instances come in the order the labels appear in your concepts, and by score within a label.
- The Segs nodes treat every image of a batch separately.
- ComfyUI's own nodes do not read `NH_SEGS`. To get masks out, use the `mask` outputs, [Segs Compose](#segs-compose-nh) or [Segs To Masks](#segs-to-masks-nh). To use Impact Pack's detailers, convert with [Segs To Impact SEGS](#segs-to-impact-segs-nh).

---

## Writing a formula

SAM3 Segment's `concepts` and Segs Compose's `expression` read the same small language. You name the things you want, combine them with operators, and the result is one mask per image. In SAM3 Segment the words you write are also what SAM 3 looks for. In Segs Compose they pick instances that were found already.

| Write | Meaning |
|---|---|
| `person` | A label: the instances with that name. Case does not matter. |
| `red car`, `women's dress` | Several words in a row are one label, with no quotes. A hyphen between two letters or digits belongs to the label (`t-shirt`, `x-ray`), and so does an apostrophe inside a word. |
| `person, hair` | A list: the union of its items. A new line works like a comma. A list is read after every other operator, so `person - hair, dress` is `person - hair` plus `dress`. Empty items are skipped. |
| `a + b`, `a \| b` | Union: everything in a or in b. |
| `a - b` | a without b. Put spaces around the minus: `person-hair` is read as one label. |
| `a & b` | Overlap: only where a and b both are. |
| `~a`, `!a` | Everything except a: the whole picture minus a. |
| `( … )` | Grouping. |
| `background` | Everything outside every instance that was found. The same as `~all`. |
| `all` | Every instance that was found. |
| `image` | The whole picture. |
| `none` | An empty mask. |
| `# …` | A comment, to the end of the line. It starts where a word could start; inside a word, as in `car#2`, the `#` belongs to the word. |
| `"salt & pepper"` | Quotes keep together a label that contains operator characters. A quoted `all`, `image`, `none` or `background` is an ordinary label. |

`~` binds first, then `&`, then `+`, `|` and `-` together, left to right, and last the list: `a - b + c` is `(a - b) + c`. When in doubt, add parentheses.

**Line breaks.** A line break lists, like a comma. It joins instead when it is next to an operator (the line before ends with one, other than `)`, or the next line starts with `+`, `|`, `-`, `&` or `)`), so a long formula can be laid out over several lines:

```text
(wall + ceiling)
  - window
```

**Vietnamese words.** SAM 3 reads English only, so the nodes translate the common Vietnamese words you type: `tóc` is `hair`, `váy` is `dress`, `cửa sổ` is `window`. The list covers people, body, clothes, accessories, nature, rooms and buildings, vehicles, food and animals. The full list is in `core/vi_words.py`.

- A thing followed by colours or sizes is put into English order: `áo đỏ` is `red top`, `tóc dài` is `long hair`, `ghế sofa lớn` is `large sofa`. `màu` (colour) may stand between them: `áo màu đỏ`. `tóc vàng` is `blonde hair`.
- `xanh` after a thing means blue or green, so it is not guessed. Write `xanh dương` (blue) or `xanh lá` (green), as in `áo xanh dương`, or write the English word.
- `các`, `những`, `mấy`, `cái`, `chiếc` and `con` in front of a thing are skipped when the phrase is not in the list as written: `những chiếc xe` is `vehicle`.
- `nền`, `phông`, `phông nền` and `hậu cảnh` are the keyword `background`, also with a colour (`nền trắng`). `tất cả` is `all`, and `toàn ảnh` and `cả ảnh` are `image`.
- Write Vietnamese with its accents. Text without Vietnamese accents is taken as English (a few short words, such as `xe`, are in the list as written).
- A Vietnamese phrase that is not in the list is searched for as written, and SAM 3 usually finds nothing. The report then says that it was not understood. The labels of the instances are always the English words.
- Quotes do not stop the translation. A quoted `nền` is the label `background`, not the keyword.

**Where the two nodes differ**

| | SAM3 Segment, `concepts` | Segs Compose, `expression` |
|---|---|---|
| `or`, `and`, `not` | Plain words, so a phrase such as `black and white dog` reaches SAM 3 as written. | Operators: `a or b`, `a and b`, `not a`. |
| Wildcards `*` and `?` | Refused: SAM 3 needs plain words. | Match labels (`*shirt`). |
| What a label does | It is what SAM 3 is asked to find. | It picks the instances that have that label. A label that no instance has gives an empty part. |
| An empty field | Stops with a message to write at least one concept. A field with only comments, or only `all`, `image` or `none`, counts as empty. | Stops with a message to use `all` for every instance. |

---

## SAM3 Segment (NH)

<p align="center"><img src="../assets/nodes/NH_SAM3Segment.svg" alt="SAM3 Segment (NH) node" width="360"></p>

Type what you want to cut out and get its mask: one word (`person`), a list (`hair, face`) or a formula (`person - bag`, `background`). SAM 3 finds every instance of the things you name. The node returns the mask of your formula, the instances as `NH_SEGS`, a count and a report of what was and was not found, and it shows what it found on itself.

**Use it when** you want masks of "all the people", "every window", "the sky", "the background" or "the person without the bag" in a photo, also a 4K to 8K one, without painting anything. For finer picks (the second person from the left, the biggest object), or to make several masks from one run, connect `segs` to the [Segs nodes](#segs-select-nh).

Combo inputs without a stated default start on their first option.

| Input | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | The picture. Colour (RGB) is used and any alpha is dropped. Every image of a batch is processed in turn. |
| `concepts` | STRING | `person` | What to find and which mask to return, written as a formula (see [Writing a formula](#writing-a-formula)): short English words (`person`, `red car`, `window`), commas or new lines to list several, and `+`, `-`, `&`, `~`, parentheses and `background` to combine them (`person - bag`). Common Vietnamese words work too (`tóc`, `áo đỏ`, `nền`). Write at least one word. |
| `threshold` | FLOAT | `0.5` | Range 0.05 to 0.95. The minimum confidence. Lower finds more instances and more mistakes; higher keeps only the sure ones. A score exactly at the threshold is not kept. When a concept finds nothing, the report says how close the best candidate came. |
| `detail_level` | COMBO | `balanced` | `balanced` or `fast`. `balanced` segments every instance a second time on a close-up of the full-resolution image, for crisp edges on small and medium objects. `fast` keeps the first, whole-image pass. See [How the two passes work](#how-the-two-passes-work). |
| `max_per_concept` | INT | `0` | Range 0 to 200. Keep at most this many instances of each concept, best score first. `0` keeps all. |
| `checkpoint` | COMBO | — | A SAM 3 `.pt` or `.pth` file from `models/sam3` or another folder registered as `sam3`. Other SAM 3 packs use the same folder. When it holds no file, the list shows `sam3.pt` and the node tells you where to get it when you run it. |
| `keep_model_loaded` | BOOLEAN | `false` | Off: after the run the weights go back to system memory and ComfyUI gets the VRAM back. On: they stay on the GPU (about 3.6 GB), which is faster when the node runs again, but ComfyUI cannot use that VRAM meanwhile, and its **Unload Models** button does not release it (turn this off and run the node once, or restart ComfyUI). |
| `show_preview` | BOOLEAN | `true` | Optional. Shows the result on the node: the mask bright and the rest darkened, the instances outlined and labelled, and the report lines below. At most 1536 px on the longer side, and the first 16 pictures of a batch. Off: nothing is drawn, and the node still runs (see below). |

| Output | Type | Description |
|---|---|---|
| `segs` | NH_SEGS | The instances of every image, each with its label, score, box and mask. The labels are the English words SAM 3 was asked for. When the formula uses `background`, they include the `person` instances that were searched for it. Connect it to the Segs nodes. |
| `mask` | MASK | The mask of your formula, one per image, at the size of the image. A plain list of words gives the union of every instance. |
| `count` | INT | The number of instances found, over the whole batch, including the `person` instances that `background` searched for. |
| `report` | STRING | What the request found, one line per word: counts and best scores, translations, near misses, words that were not understood, and what `background` meant. See below. |

<p align="center"><img src="../assets/examples/sam3-formula.jpg" alt="What SAM3 Segment shows on the node for four formulas on a sample photo: background, person - face, hair and face, and everything except the person and the helmets" width="720"><br><em>SAM3 Segment on a sample photo, as the node shows it. Top: "background" lights up everything outside the person, and the report is under the picture. Bottom: three more formulas; the bright part is each one's mask output.</em></p>

**The report.** The first line of `report` gives the totals. Each line after it (they are also written under the preview) starts with what you wrote:

| Line | Meaning |
|---|---|
| `person: 2 found (best 0.91)` | The number of instances found and the best score. |
| `áo đỏ -> red top: 1 found (best 0.64)` | What a Vietnamese phrase was translated to. |
| `hair: none at threshold 0.50; the best candidate scored 0.31, lower the threshold to keep it` | Nothing passed `threshold`, but a candidate came close. Lower `threshold` below its score to keep it. |
| `bag: none found; try another English word` | No candidate scored even 0.1. |
| `áo xanh: not understood; write it in English (SAM 3 reads English only)` | A Vietnamese phrase that is not in the list and found nothing. |
| `background = everything outside person, red top` | What `background` meant: everything outside the instances of these labels. |
| `person (searched for background): 2 found` | The `person` search that the node added because the formula uses `background` and does not name `person`. |

**How it works**

- **Labels.** The English words SAM 3 was asked for are the labels of the instances: `áo đỏ` is looked for as `red top`, and its instances are labelled `red top`. The Segs nodes ignore case when they match labels, so spelling is not a problem.
- **The mask.** `mask` is the result of the formula. For `person - bag`, SAM 3 looks for `person` and for `bag`, `mask` is the people without the bags, and `segs` still holds both kinds of instance.
- **`background`.** It is everything outside every instance that was found, so the node needs to know where the subject is. When the formula uses it and does not name `person`, SAM 3 also looks for `person`; those instances are part of `segs` and `count` and are outlined in the preview. To keep other things out of the background too, name them: `~(person + wine glass + chair)`. A list is a union, so `person, background` is the whole picture.
- **Order and duplicates.** Instances come in the order the labels appear in your formula (the `person` that `background` added comes last), best score first within a label. Two instances of the same label whose masks overlap almost completely (90% or more) count as one, and the better score stays. `max_per_concept` then keeps the best of what is left. Instances with exactly equal scores keep the order SAM 3 gave them.
- **Seeing the result.** With `show_preview` on, the node draws the preview itself: no other node is needed. To see the picture at full size (the preview is at most 1536 px), or to see what Segs Select works with, connect `segs` and the same picture to [Segs Preview](#segs-preview-nh).
- **It runs on its own.** SAM3 Segment is an output node (the dot on its card is orange): ComfyUI runs it even when nothing is connected to its outputs. A copy that you no longer need therefore keeps running SAM 3 whenever its inputs change, and turning `show_preview` off does not stop it. Mute or delete it. While its inputs stay the same, ComfyUI reuses the last result.
- **Nothing found** is not an error: you get no instances and a `count` of 0. The mask is empty for a plain list, and the whole picture for a formula such as `background` or `~person`, because nothing was found to leave out.
- **Batches.** Every image is processed in turn and has its own list of instances. A progress bar counts the concepts, and you can cancel the run between concepts and between close-ups.
- **Things cut by the frame.** SAM 3 fades its masks over a band along the border of the square view it works in, about 0.3% of the picture's side wide. The mask of a thing that runs off the picture still reaches the image edge: the band is lifted where the object reaches it from inside. The price is that a thing which ends within that band just above the edge is drawn down to the edge.
- **Very wide or tall pictures.** SAM 3 first sees the whole picture squeezed into a square. On a picture wider or taller than 4:1 (a panorama), it can miss small things, and the console says so. Split such a picture into tiles with [Image Tile (NH)](./image-compose.md#image-tile-nh) and run the tiles.
- **Setup problems** stop the node with a message that says what to do: a missing package, a missing checkpoint, no CUDA GPU. See [If something goes wrong with SAM 3](#if-something-goes-wrong-with-sam-3).

> **Tip:** Write simple English nouns: `window`, `wine glass`, `red car`. If a concept finds nothing, try another word (an unusual building may answer to `structure` or `monument` rather than `building`) or lower `threshold`. The report says which case it is: nothing found, a candidate that came close, or a word that was not understood.

> **Tip:** Ask for a whole and its parts in one formula: `person - hair`. The node looks for both and `mask` is the difference. `segs` still holds both, so [Segs Compose](#segs-compose-nh) can make other masks from the same run.

> **Licence:** SAM 3 is under Meta's SAM License. See [Before you start with SAM 3](#before-you-start-with-sam-3).

---

## Segs Select (NH)

<p align="center"><img src="../assets/nodes/NH_SegsSelect.svg" alt="Segs Select (NH) node" width="360"></p>

Keeps the instances you want, by label, score, size and position, in the order you choose.

**Use it when** you need only some of what SAM3 Segment found: the biggest object, the second person from the left, only the sure detections, or only the large windows.

| Input | Type | Default | Description |
|---|---|---|---|
| `segs` | NH_SEGS | — | The instances, from SAM3 Segment or another Segs node. |
| `labels` | STRING | empty | The labels to keep, separated by commas. `*` stands for any run of characters and `?` for one character (`*shirt`). Case does not matter, and `_` counts as a space (`red_car` finds `red car`). Labels with spaces need no quotes here: `wine glass, red car`. Common Vietnamese words are translated (`tóc` finds `hair`); the formula words `nền` and `tất cả` are not labels here. Empty keeps every label. |
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

Builds one mask from the labels of the instances, with the same small formula as SAM3 Segment's `concepts`: union, minus, overlap and "everything except".

**Use it when** you already have instances and want "the background", "the person without the hair", "the wall without the windows", or another combination of what SAM3 Segment found: after [Segs Select](#segs-select-nh), or to make several masks from one SAM 3 run. For a single mask you can also write the same formula straight into SAM3 Segment's `concepts`.

| Input | Type | Default | Description |
|---|---|---|---|
| `segs` | NH_SEGS | — | The instances, from SAM3 Segment or Segs Select. |
| `expression` | STRING | `all` | The formula. See [Writing a formula](#writing-a-formula). The field has several lines: a line break lists, like a comma, and joins when it is next to an operator. |

| Output | Type | Description |
|---|---|---|
| `mask` | MASK | The result of the formula, one mask per image, at the size of the image. |

**What is different here.** The language is the one in [Writing a formula](#writing-a-formula), with these differences from SAM3 Segment:

- `or`, `and` and `not` are operators: `a or b` is `a | b`, `a and b` is `a & b`, `not a` is `~a`.
- A label picks the instances that have it. Case does not matter, `_` counts as a space (`red_car` finds `red car`), and `*` and `?` are wildcards (`*shirt`).
- Put a label that contains an operator character in quotes, `"` or `'` (`"salt & pepper"`), and use double quotes when it also has an apostrophe. Several words in a row need no quotes.
- Vietnamese labels are translated (`tóc` finds `hair`), also inside quotes. `nền` is the keyword `background`.

| Expression | You get |
|---|---|
| `person - hair` | The people without their hair (ask SAM3 Segment for `person, hair`). |
| `~person` | Everything except the people. |
| `background` | Everything outside every instance in the segs. Unlike `~person`, it also leaves out windows, chairs or anything else that was found. |
| `(wall \| ceiling) - window` | The walls and the ceiling with the window openings cut out. |
| `wine glass, person` | The glasses and the people together (a list, the same as `wine glass \| person`). |
| `all - person` | Everything that was found except the people. Unlike `~person`, it leaves out the background that nothing found. |

**How it works**

- **A label that no instance has** gives an empty part, and the console says so (`no instance is labelled 'x'; those parts are empty`). A typo in `~persn` therefore gives the whole picture: check the console when a mask looks wrong.
- **Words in a row are one label.** When no instance has that label but each word is the label of an instance (`person hair`), the node stops with an error that suggests `person + hair`. The same goes for a hyphen: when both `person` and `hair` were found, `person-hair` stops with an error that shows `person - hair`.
- **Every image** of a batch gets its own mask, built from its own instances.
- **A formula that cannot be read** stops the node with a message that gives the position of the problem, for example `unexpected '(' at position 8` for `person (adult)`.

<p align="center"><img src="../assets/examples/sam3-segment.jpg" alt="SAM3 Segment instances drawn by Segs Preview, and three masks built from them with Segs Compose and Segs Select on a sample photo" width="720"><br><em>The instances of "person, helmet, face" on a sample photo, drawn by Segs Preview, then three masks made from them with Segs Compose and Segs Select.</em></p>

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

**Use it when** you want to see the instances at the full size of your picture, or to see what Segs Select picked. SAM3 Segment already shows a small preview on itself (at most 1536 px, with its report). Put this node after SAM3 Segment to see everything, or after Segs Select to see only the selection.

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

In each of these, check what SAM3 Segment found before you edit anything. Its preview on the node shows every instance with an outline and a label, and its report says what was not found. If an instance you want is missing, lower `threshold` (for example to 0.3) or try another word for the concept.

### The background behind people

```text
Load Image → SAM3 Segment (concepts: background)
           → Mask Morphology (NH)
           → edit the background
```

`background` is everything outside the people that SAM 3 found: the node looks for `person` on its own. When people hold or sit on things, name those too, so that they stay out of the background, for example the concepts `~(person + wine glass + chair)`. Move the edge of the mask a few pixels with [Mask Morphology (NH)](./mask.md#mask-morphology-nh): grow it so that the edit covers the old outline, or shrink it if the edit eats into the people. Then use the mask to blur, relight, recolour or inpaint the background.

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
Load Image → SAM3 Segment (concepts: building - (window + door))
           → recolour, restyle or inpaint the wall
```

`building` covers the whole facade with its windows and doors; the formula cuts them out and leaves the wall. For a room, use `(wall + ceiling) - window`. If `building` finds nothing on an unusual building, try `structure` or `monument` and change the formula to match. On a 4K to 8K photo, keep `balanced`: every window is segmented again on a close-up of the original pixels. The outline of a facade that fills the whole frame is accurate to about 1/300 of its width (see [How the two passes work](#how-the-two-passes-work)), so check the roofline in the preview.

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
| An error that you must write at least one concept | `concepts` is empty, holds only comments, or holds only the keywords `all`, `image` or `none`. Write a word such as `person`. |
| An error that a concept has a wildcard | Remove the `*` or `?`: SAM 3 needs plain words. Wildcards work in Segs Select and Segs Compose. |
| An error from SAM3 Segment that gives a position, such as `unexpected '(' at position 8` | Check the formula at that position. A bracket in a label is read as grouping. See [Writing a formula](#writing-a-formula). |
| `count` is 0, or an instance is missing | Read the report: it says whether the word found nothing, a candidate came close (lower `threshold` to its score), or the word was not understood (write it in English). The preview on the node shows what was found. |
| The report says a word was not understood | It is a Vietnamese phrase that is not in the list. Write it in English. After a thing, `xanh` is ambiguous: write `xanh dương` (blue) or `xanh lá` (green). |
| The mask is the whole picture, or not what you expected | A list is a union: `person, background` is the person plus everything outside it. Write `background` alone, or a formula such as `person - bag`, and look at the preview: the bright part is the mask. |
| SAM3 Segment runs although nothing is connected to it | It is an output node. Mute or delete it when you do not need it; turning `show_preview` off does not stop it. |
| The console says it kept the first-pass mask for some instances | Their close-up did not match the first pass, so those instances keep the first pass's edges. Nothing is wrong. |
| An error from Segs Preview about sizes or the number of images | Connect the picture that the segs were found in. |
| An error from Segs Compose that gives a position | Check the formula at that position. See [Writing a formula](#writing-a-formula). |
| An error from Segs Compose that no instance is labelled `person hair`, but `person` and `hair` are | Words in a row are one label. Join them: `person + hair`. |
| A Segs Compose mask is empty, and the console says no instance has a label | Check the spelling of the label. Put spaces around a minus: `person - hair`, not `person-hair`. |
| An error from Segs To Masks that the full-size instance masks would need more memory than is free | There are too many instances for the size of the picture. Use Segs Select to keep only the ones you need, or Segs Compose to get one mask. |
| The console says that SAM 3 sees the picture squeezed into a square and can miss small things | The picture is wider or taller than 4:1. Split it into tiles with Image Tile (NH) and run the tiles. |
| ComfyUI runs out of VRAM after the node | Leave `keep_model_loaded` off, so the weights leave the GPU after each run. |
| ComfyUI's Unload Models button does not free the VRAM of SAM 3 | `keep_model_loaded` is on. Turn it off and run the node once, or restart ComfyUI. |

---

[← Back to guide](./README.md)
