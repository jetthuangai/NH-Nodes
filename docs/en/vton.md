[← Back to guide](./README.md)

# VTON

One node that prepares a photo of a person for a virtual try-on or a garment replacement. It finds the area to repaint, the part of the person to keep, the pose and the body parts, so you do not have to paint masks by hand. Find it under **NH-Nodes → VTON**.

**On this page**

- [Before you start](#before-you-start) — ONNX Runtime, first-run download, memory, licence
- [VTON Ultimate Processor (NH)](#vton-ultimate-processor-nh) — the node, its inputs and outputs
  - [Categories](#categories) — what each garment category masks
  - [Outputs in detail](#outputs-in-detail) — what you get and how to read it
  - [Adjusting the mask](#adjusting-the-mask) — offsets, feathering, hands, hair, shoes
  - [Memory use](#memory-use) — `keep_model_loaded` and VRAM
  - [If something goes wrong](#if-something-goes-wrong)

## Before you start

**1. Install ONNX Runtime.** NH-Nodes does not install it for you, because the CPU package would overwrite an existing GPU one. In the Python that runs ComfyUI, install exactly one of these, then restart ComfyUI:

```text
python -m pip install onnxruntime-gpu     # NVIDIA GPU
python -m pip install onnxruntime         # CPU only
```

The node checks this first, before it downloads anything. If `device` is `cuda` but the installed package has no GPU support, the node runs on the CPU and prints a hint in the console.

**2. Let the first run download four models (about 0.9 GB).** The first run fetches two DWPose models (a person detector and a pose model) and two human-parsing models from the Hugging Face repository `kg-09/kg-vton-mask`. The download is pinned to a fixed revision, so a later change upstream cannot alter what you get. Files go to `ComfyUI/models/ComfyUI-Vton-Mask/`, the folder earlier installs already use, so copies you already have are found. You can add more locations with the `vton_mask` key in `extra_model_paths.yaml` (see [Installation](../../README.md#installation) for an example). A folder that misses any of the four files does not count: run the node again and it finishes the download.

**3. Plan for about 3 GB of VRAM** if you run on `cuda`. See [Memory use](#memory-use).

**4. Use a photo of one person** who is clearly visible. The node uses the first person it detects.

> **Licence:** VTON Ultimate Processor is for **non-commercial use only**. Its code and human-parsing models come from FitDiT and OOTDiffusion under CC BY-NC-SA 4.0. The DWPose models are Apache-2.0 upstream, but treat the node as a whole as non-commercial. NH-Nodes' own code is MIT. Read [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) before any commercial use.

---

## VTON Ultimate Processor (NH)

<p align="center"><img src="../assets/nodes/NH_VTonUltimateProcessor.svg" alt="VTON Ultimate Processor (NH) node" width="360"></p>

Creates the masks and guide images for a virtual try-on from one photo of a person.

**Use it when** you build a try-on or garment-replacement workflow and need the area to repaint, and the part of the person to keep, without drawing masks by hand.

Combo inputs without a stated default start on their first option.

| Input | Type | Default | Description |
|---|---|---|---|
| `human_image` | IMAGE | — | Photo of one person. Colour (RGB) is used and any alpha is dropped. Every image of a batch is processed. |
| `category` | COMBO | `Upper-body` | `Upper-body`, `Lower-body`, `Dresses`, `Upper-body (Sleeveless)` or `Lower-body (Shorts/Skirt)`. See [Categories](#categories). |
| `mask_feathering` | INT | `4` | Range 0 to 50. Blur radius, in pixels of your photo, for the edge of the final mask. `0` gives a hard edge. |
| `cover_shoes` | BOOLEAN | `false` | Extends the mask down to the bottom of the image across the width of the person, so shoes and feet are repainted too. It needs enough of the body to be detected; if it is not, the option does nothing. |
| `refine_hands` | BOOLEAN | `true` | Takes the arms out of the mask, so the original arms stay visible. |
| `refine_hair` | BOOLEAN | `true` | Takes the hair out of the mask, so hair in front of the garment stays. |
| `device` | COMBO | `cpu` | `cpu` or `cuda`. Choose `cuda` for an NVIDIA GPU with `onnxruntime-gpu` installed. |
| `offset_top` *(optional)* | INT | `0` | Range -200 to 200. Moves the top edge of the mask rectangle by N pixels. Positive moves it down. |
| `offset_bottom` *(optional)* | INT | `0` | Range -200 to 200. Moves the bottom edge. Positive moves it down. |
| `offset_left` *(optional)* | INT | `0` | Range -200 to 200. Moves the left edge. Positive moves it to the right. |
| `offset_right` *(optional)* | INT | `0` | Range -200 to 200. Moves the right edge. Positive moves it to the right. |
| `keep_model_loaded` *(optional)* | BOOLEAN | `false` | Off: after a run that really used the GPU, the models are released from VRAM. On: they stay loaded, which is much faster for repeated runs and for lists of images. See [Memory use](#memory-use). |

| Output | Type | Description |
|---|---|---|
| `final_mask` | MASK | The area to repaint, after offsets, refinements and feathering. |
| `agnostic_mask` | MASK | The person without the repaint area: the part to keep. |
| `densepose_image` | IMAGE | The detected pose as a skeleton drawing (body, hands and face) on black. |
| `parsing_image` | IMAGE | The body-part map in colours, one colour per part. |
| `masked_image` | IMAGE | Your photo with the final mask painted white. |
| `parsing_map_raw` | MASK | The body-part map as numbers: each pixel holds its label divided by 19. |
| `hair_mask` | MASK | The hair. |
| `hands_mask` | MASK | The arms. |

All outputs have the size of the input photo, and a batch gives a batch.

<p align="center"><img src="../assets/examples/vton-masks.jpg" alt="VTON Ultimate Processor final mask, parsing map and pose on a sample photo" width="720"><br><em>The final mask, the body-part map and the pose drawing for one photo.</em></p>

**How it works**

- **Working size.** The photo is analysed on a copy whose longest side is 768 px. The pose model finds keypoints and a parsing model labels every pixel (hair, face, clothes, arms and so on). The results are scaled back to your photo.
- **A generous rectangle.** The mask is built from the keypoints and the parsed clothing, then padded and with the head cut out. It is not the exact outline of the garment that is on the person now.
- **Refinement order.** Category rules, then `refine_hands`, `refine_hair`, `cover_shoes`, and last `mask_feathering`.
- **Batches.** Each image is processed on its own with the same settings. If no person is found in one of them, the node stops and names the batch index.

**Example**

```text
Load Image (person) → VTON Ultimate Processor (Upper-body, cuda) → final_mask → your try-on or inpaint model
                                                                  → masked_image → Preview Image (check the area)
```

Look at `masked_image` first. The white area is what will be repainted. If it is too big or too small, change the offsets.

### Categories

| `category` | The mask covers | Good to know |
|---|---|---|
| `Upper-body` | Tops, shirts and jackets: a padded rectangle from the chin or shoulder line (whichever is higher) down to the hips, reaching out to the hands. The head is cut out. | With `refine_hands` on, the arms are taken out. |
| `Lower-body` | Trousers and skirts: a rectangle across the full width of the person, from about the waist (never higher than the middle of the torso) down to the feet. | `offset_left` and `offset_right` have no effect, because the width always follows the person. |
| `Dresses` | Dresses and jumpsuits: a wide padded rectangle from the chin or shoulder line down to the feet. The head, shoes and bag are cut out. | |
| `Upper-body (Sleeveless)` | The same area as `Upper-body`, with the arms always taken out. | The arms are removed even when `refine_hands` is off. |
| `Lower-body (Shorts/Skirt)` | The same area as `Lower-body`, but it stops at the knee line. | If the knees are not detected, it behaves like `Lower-body`. |

### Outputs in detail

- **`densepose_image`** keeps this name, but it is a DWPose skeleton drawing, not a DensePose map.
- **`agnostic_mask`** is everything the parser sees as part of the person (hair, face, skin, clothes, bags), minus `final_mask`.
- **`masked_image`** blends towards white where `final_mask` is soft.
- **`parsing_map_raw`** is the body-part map as numbers. Multiply a pixel by 19 and round to get its label: 0 background, 1 hat, 2 hair, 3 sunglasses, 4 upper clothes, 5 skirt, 6 pants, 7 dress, 8 belt, 9 left shoe, 10 right shoe, 11 face, 12 left leg, 13 right leg, 14 left arm, 15 right arm, 16 bag, 17 scarf, 18 neck.
- **`hair_mask`** is label 2.
- **`hands_mask`** is labels 14 and 15 (both arms) plus label 3 (sunglasses). It is the region that `refine_hands` takes out of the final mask. The parser has no separate hand class, so the whole arm is covered.

### Adjusting the mask

- **Offsets.** Each offset moves one edge of the mask rectangle in image coordinates: positive values move the edge right or down. To grow the mask on every side, use negative `offset_top` and `offset_left` and positive `offset_bottom` and `offset_right`.
- **Offsets use the working size.** They are measured on the 768 px copy, not on your photo. On a photo whose longest side is 3072 px, an offset of 10 moves the edge by 40 px.
- **`mask_feathering`** blurs the finished mask in pixels of your photo, so a high value on a small photo softens the whole mask.
- **`refine_hands` and `refine_hair`** subtract the arms and the hair from the mask. Turn `refine_hands` off if the new garment has long sleeves that must cover the arms.
- **`cover_shoes`** is meant for categories whose mask already reaches the legs, such as `Lower-body`, `Dresses` and `Lower-body (Shorts/Skirt)`.

### Memory use

The four models together use about 3 GB of VRAM on `cuda`. ONNX Runtime allocates this memory itself, so ComfyUI cannot see or free it, and ComfyUI's **Free memory** button does nothing for it. `keep_model_loaded` decides what the node does with it after a run:

| `keep_model_loaded` | Models that run on the GPU | Models that run on the CPU |
|---|---|---|
| `false` (default) | Released after each run, so the next node gets the whole GPU. The next run loads them again, which costs about 1.5 seconds per run. | They stay cached. |
| `true` | They stay in VRAM, so repeated runs are much faster. A large model that runs afterwards may run out of VRAM. | They stay cached. |

- **Lists of images.** ComfyUI calls the node once per item of a list, but only once for a whole batch. With the setting off and the models on the GPU, a list of 20 images loads the models 20 times, so turn it on for lists and for repeated runs.
- **"Runs on the GPU" means what ONNX Runtime really did**, not what `device` says. If `device` is `cuda` but ONNX Runtime fell back to the CPU (only the CPU package is installed, or CUDA and cuDNN do not match), the models sit in ordinary memory, stay cached, and the setting has no effect.
- Before loading the models on the GPU, the node asks ComfyUI to unload some of its own models to make room. It skips this when the GPU package of ONNX Runtime is not installed, and once ONNX Runtime has fallen back to the CPU, because freeing VRAM would not help those models.
- If you change `device` between runs, the models of the old device are released first.

### If something goes wrong

| What you see | What to do |
|---|---|
| An error that ONNX Runtime is not installed | Install one package as in [Before you start](#before-you-start), then restart ComfyUI. |
| It runs slowly although `device` is `cuda` | The console says why. Either only the CPU package of ONNX Runtime is installed (install `onnxruntime-gpu` instead), or the GPU package cannot start (check that its CUDA and cuDNN versions fit your setup). |
| An error that no person was found at a batch index | Use a photo where the person is clearly visible. |
| The next model runs out of VRAM | Turn `keep_model_loaded` off. |
| A list of images takes very long | Turn `keep_model_loaded` on, so the models load once instead of once per image. |
| The download stopped half way | Run the node again. It continues and keeps finished files. |

---

[← Back to guide](./README.md)
