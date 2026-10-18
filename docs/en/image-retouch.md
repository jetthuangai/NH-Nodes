[← Back to guide](./README.md)

# Image — Mask-Based Retouch

Three nodes that work with a mask. One resizes an image to an exact size and keeps the masked subject whole. One erases the masked area, so a try-on or inpainting model has to fill it in. One pastes the masked face of one image onto another. Find them under **NH-Nodes → Image** in the node menu.

**On this page**

| Node | Use it to |
|---|---|
| [NH Mask-Aware Resize](#nh-mask-aware-resize) | Resize to an exact size while the whole masked subject stays in the picture |
| [NH Agnostic Image Generator](#nh-agnostic-image-generator) | Erase the masked area with grey, noise or blur |
| [NH Simple Face Paste](#nh-simple-face-paste) | Paste the masked face of one image onto the masked area of another |

A `MASK` is white (`1`) where the node should work and black (`0`) elsewhere. Get one from a segmentation node such as [SAM3 Segment (NH)](./vision.md#sam3-segment-nh) (any object you can name, or a formula such as `person - bag` or `background`), [Garment Segment (NH)](./vision.md#garment-segment-nh) or [Portrait Segment (NH)](./vision.md#portrait-segment-nh), or from any mask tool in ComfyUI.

---

## NH Mask-Aware Resize

<p align="center"><img src="../assets/nodes/NH_MaskResizeImage.svg" alt="NH Mask-Aware Resize node" width="360"></p>

Resizes an image to an exact width and height and keeps everything inside the mask. The masked subject is enlarged as far as the target allows, and the image is cropped or padded around it.

**Use it when** a model takes a fixed size (for example 832 × 1216) and the face, garment or product you mask must fit in it completely.

| Input | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | The image or batch to resize. |
| `mask` | MASK | — | The area to keep. Pixels above `0.5` count. It must have the same width and height as `image`. A single mask is used for every image of a batch. |
| `width` | INT | `512` | `64` to `8192`, step 8. The width of the result. |
| `height` | INT | `512` | `64` to `8192`, step 8. The height of the result. |
| `mode` | COMBO | `crop` | `crop`: cut a window out of the scaled image. `pad`: centre the subject and fill the empty space with a colour. |
| `pad_color_hex` | STRING | `#000000` | The padding colour, written as `#RRGGBB`. Any other form is read as black, including `#RGB` and `#RRGGBBAA`. |

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | The resized image, `width` × `height`. |
| `mask` | MASK | The mask, moved and scaled in the same way, so it still lines up with the new image. Padded areas are `0`. |

**How it works**

1. The node finds the box around the mask.
2. It scales the whole image by the largest factor at which that box still fits inside `width` × `height`. A small subject can be enlarged a lot.
3. It puts the centre of the box in the centre of the result.
   - `crop` takes a window of the scaled image. The window slides back inside the image when needed, so a mask near an edge ends up off-centre, but nothing is padded. If the scaled image is smaller than the result, `crop` pads like `pad`.
   - `pad` keeps the box exactly centred. Wherever the result reaches past the image, it is filled with `pad_color_hex`.

The whole mask is always in the result. The picture is never stretched, and the image uses Lanczos resampling.

Example: a 200 × 100 px mask in the top-left corner of a 1000 × 800 px image, with a 512 × 512 target. The scale is 2.56 in both modes.

| Mode | Where the mask lands | Padding |
|---|---|---|
| `crop` | At the top of the result, in rows 0 to 255. | None. |
| `pad` | Centred, in rows 128 to 383. | 128 rows of `pad_color_hex` above the image. |

With a mask near the middle of the image, both modes give the same picture. With a mask that covers the whole image, both fit the image inside the target and pad the rest.

<p align="center"><img src="../assets/examples/mask-aware-resize.jpg" alt="Crop and pad results of Mask-Aware Resize with the mask outlined" width="720"><br><em>The <code>crop</code> and <code>pad</code> results, with the mask outlined.</em></p>

**Example — prepare a garment for a fixed-size model**

```text
Load Image → Garment Segment (NH) → NH Mask-Aware Resize → your model
```

Connect the image to both Garment Segment and `image`, and the `mask` output of Garment Segment to `mask`. Set `width` to `832`, `height` to `1216` and `mode` to `crop`.

**Common mistakes**

- Using a mask with another size than the image. The node does not resize it, so the box lands in the wrong place.
- Expecting the whole image to survive. Only the mask is guaranteed. `crop` removes the surroundings that do not fit.
- Passing a soft mask whose values stay at `0.5` or below. Those pixels are ignored.

> **Note:** An empty mask counts as "the whole image". `crop` then covers the target and cuts the middle. `pad` fits the whole image inside the target. Nothing is stretched.

---

## NH Agnostic Image Generator

<p align="center"><img src="../assets/nodes/NH_AgnosticImageGenerator.svg" alt="NH Agnostic Image Generator node" width="360"></p>

Erases the masked area of an image and fills it with grey, noise or a heavy blur, so that a try-on or inpainting model has to invent what goes there. It also returns a white-masked copy and a before and after preview.

**Use it when** you prepare an "agnostic" person image for virtual try-on, where the garment area is removed, or you need a neutral base for an inpainting model.

| Input | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | The image or batch to work on. |
| `mask` | MASK | — | The area to erase. Soft values blend partly. A single mask is used for every image of a batch. A 2-D mask is accepted. A mask of another size is resized to the image. |
| `fill_mode` | COMBO | `gray` | What fills the area: `gray`, `noise` or `blur`. |
| `blur_radius` | INT | `65` | `3` to `255`, step 2. The blur size for `blur`. An even value is raised to the next odd number. Larger means blurrier. |
| `noise_strength` | FLOAT | `1.0` | `0` to `1`, step 0.05. Used by `noise`: `0` gives flat grey, `1` gives pure noise. |
| `gray_value` | FLOAT | `0.5` | `0` to `1`, step 0.01. The brightness of the grey, from black to white. `noise` mixes with this grey. |
| `feathering` | INT | `0` | `0` to `100`. Softens the edge of the mask with a blur. Larger means a wider, softer edge. |

| Output | Type | Description |
|---|---|---|
| `agnostic_img` | IMAGE | The image with the masked area replaced by the fill. |
| `masked_img` | IMAGE | The image with the masked area painted white. |
| `composite` | IMAGE | A preview. The left half is the original with a green outline and tint around the mask, the right half is the agnostic image, and a white line marks the middle. |

| `fill_mode` | The masked area becomes |
|---|---|
| `gray` | Flat grey of brightness `gray_value`. |
| `noise` | Random noise mixed with the grey by `noise_strength`. The noise is repeatable: image *n* of a batch always gets the same noise. |
| `blur` | A heavily blurred copy of the whole image, so the colours stay but the detail is gone. Use a large `blur_radius` if the model must not see the original. |

**How it works**

- The mask works as a soft alpha. A value of `0.5` blends half of the fill with half of the image. `feathering` blurs that alpha further.
- The outline in `composite` follows the `0.5` level of the original mask.
- A single mask is reused for every image of a batch.

<p align="center"><img src="../assets/examples/agnostic.jpg" alt="An agnostic image made from a garment mask" width="720"><br><em>An agnostic image made from a garment mask.</em></p>

**Example — a try-on input**

```text
Load Image → Garment Segment (NH) → NH Agnostic Image Generator → agnostic_img → your try-on model
```

Use `fill_mode` `gray` with `gray_value` `0.5`, and `feathering` `8` for a soft edge.

> **Tip:** Grow the mask a little before this node (for example with [Mask Morphology (NH)](./mask.md#mask-morphology-nh)) so that no sliver of the old garment stays at the edge.

---

## NH Simple Face Paste

<p align="center"><img src="../assets/nodes/NH_SimpleFacePaste.svg" alt="NH Simple Face Paste node" width="360"></p>

Cuts the masked region out of a source image, resizes it to the masked region of a destination image, and blends it in with a soft edge.

**Use it when** you want to put one face onto another image, or restore a face from the original after other edits.

| Input | Type | Default | Description |
|---|---|---|---|
| `dest_image` | IMAGE | — | The image to paste onto. Its batch size sets the batch size of the result. |
| `dest_mask` | MASK | — | Marks where the face goes. Only the box around the mask (pixels above `0.5`) is used, for the position and the size. It must have the same size as `dest_image`. |
| `source_image` | IMAGE | — | The image that supplies the face. |
| `source_mask` | MASK | — | Marks the face to take. The box around it is cut out, and its shape, soft values included, decides what is pasted. It must have the same size as `source_image`. |
| `feathering` | INT | `20` | `0` to `200`. Blurs the edge of the pasted shape. `0` gives a hard edge. Larger means a softer edge. |

| Output | Type | Description |
|---|---|---|
| `IMAGE` | IMAGE | The destination images with the face pasted in. |

**How it works**

1. The node cuts the box around `source_mask` out of the source image.
2. It resizes that cut-out to the exact size of the box around `dest_mask`. The proportions are not kept: the face is stretched to fit.
3. It blends the cut-out into the destination with the soft `source_mask` as the opacity, after `feathering` has blurred it.

- **Batches.** The result has the batch size of `dest_image`. A source image, source mask or destination mask with fewer items is reused in turn, so a single source can be pasted onto a whole batch.
- **Empty masks.** If either mask of an item is empty, that image is returned unchanged.
- The work is done in 8 bits, so values are clipped to the range 0 to 1 first.

**Example — put a face onto several images**

```text
Load Image (source) → Portrait Segment (NH) → NH Simple Face Paste.source_mask
Load Image (source) → NH Simple Face Paste.source_image
Load Images Folder (NH) → NH Simple Face Paste.dest_image
Portrait Segment (NH) on the destination → NH Simple Face Paste.dest_mask
```

Set `feathering` to `20`. Because a single source is reused, every image of the batch gets the same face.

**Common mistakes**

- Using masks whose boxes have very different proportions. The face looks squashed or stretched.
- Giving a mask that has another size than its image.
- Using a [Portrait Segment (NH)](./vision.md#portrait-segment-nh) mask of a photo with several faces. The mask holds every face, so the box around it covers them all.
- Setting `feathering` very high for a small face. The blur can eat the whole shape.

> **Tip:** Make `dest_mask` slightly larger than the face you replace, and keep `source_mask` tight. The soft edge of the source mask then blends into the destination.

---

**See also:** [Vision](./vision.md) for the segmentation nodes that make masks, [Mask](./mask.md) for masks you can grow, shrink and blur, and [VTON](./vton.md) for the full try-on preprocessing node.

[← Back to guide](./README.md)
