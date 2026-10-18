[← Back to guide](./README.md)

# Resolution

Image models give their best results at sizes they were trained or tested on. The five nodes under **NH-Nodes → Resolution** take care of that: one picks a size for text-to-image, two resize an existing image to a model-friendly size, and two bring the result back to the original frame afterwards.

**On this page**

- [Which node do I need?](#which-node-do-i-need)
- [Before you start](#before-you-start) — model families, resolution levels, resize modes, resize methods, undoing a resize
- [NH Smart Resolution Picker](#nh-smart-resolution-picker) — a preset size and an empty latent for text-to-image
- [NH Smart Ratio Image Resize](#nh-smart-ratio-image-resize) — resize an image to the preset closest to its own shape
- [NH Ratio Preset Image Resize](#nh-ratio-preset-image-resize) — resize an image to a ratio you choose
- [NH Remove Resize Padding](#nh-remove-resize-padding) — take padding bars off again
- [NH Restore Image Geometry](#nh-restore-image-geometry) — put a processed image back into the original frame

## Which node do I need?

| I want to… | Use |
|---|---|
| Get a width, a height and an empty latent for text-to-image | [NH Smart Resolution Picker](#nh-smart-resolution-picker) |
| Resize a photo to the preset that fits its own shape | [NH Smart Ratio Image Resize](#nh-smart-ratio-image-resize) |
| Resize a photo to a ratio I choose, such as 16:9 or 9:16 | [NH Ratio Preset Image Resize](#nh-ratio-preset-image-resize) |
| Remove padding bars, or return to the original size | [NH Remove Resize Padding](#nh-remove-resize-padding) |
| Put an edited image back into the original frame and keep the original pixels outside the crop | [NH Restore Image Geometry](#nh-restore-image-geometry) |

## Before you start

### Model families and presets

Every node on this page asks for a `model_family`. Each family has a list of **presets**: width × height pairs for common aspect ratios, grouped in tiers by size in megapixels (MP).

| Family | Sweet spot (recommended) | Other tiers | Empty latent |
|---|---|---|---|
| `FLUX.2 Klein 4B` | 1 MP, 1024×1024 | 2 MP verified · 3 MP community · 4 MP risk | 128 channels, 1/16 size |
| `FLUX.2 Klein 9B` (default) | 2 MP, 1408×1408 | 1 MP and 3 MP verified · 4 MP risk | 128 channels, 1/16 size |
| `FLUX.2 Dev` | 2 MP, 1408×1408 | 1 MP, 3 MP and 4 MP verified | 128 channels, 1/16 size |
| `Qwen-Image` | Native, 1328×1328 | 1 MP, 2 MP, 3 MP, 4 MP and HD, all community | 16 channels, 1/8 size |
| `Z-Image` | Tier 1024 and Tier 1280 | Tier 1536 community · Tier 2K extrapolation | 16 channels, 1/8 size |

Each tier carries a trust label. Anything other than `verified` shows a warning in the `info` output.

- `verified` — no warning.
- `community` — found by community testing, not an official size.
- `risk` — may fail, especially on distilled variants of the model.
- `extrapolation` — outside the range that was tested.

The full list of sizes lives in [`core/resolution_data.py`](../../core/resolution_data.py).

### Resolution levels

`resolution_level` (`1 MP` to `12 MP`) sets the target area for the two image-resize nodes.

- **1 MP to 4 MP:** the node uses the family's presets from the matching tier. `Z-Image` names its tiers by side length, so `1 MP` uses Tier 1024, `2 MP` Tier 1280, `3 MP` Tier 1536 and `4 MP` Tier 2K. `Qwen-Image` uses its `1 MP (preview)`, `2 MP`, `3 MP` and `4 MP` tiers; its Native and HD sizes can only be picked in [NH Smart Resolution Picker](#nh-smart-resolution-picker).
- **5 MP to 12 MP:** no presets exist. The node calculates a size with the area and ratio you asked for, rounds it to the family's multiple (16, or 32 for `Qwen-Image`) and labels it `extrapolated`. The `info` output warns that it may be outside the model's tested range.

### Resize modes

| Mode | What happens | Proportions | Cost |
|---|---|---|---|
| `cover_crop` | Scales the image until it covers the target, then cuts the overflow evenly from both sides. | Kept | The edges are cut off. |
| `contain_pad` | Scales the image until all of it fits, centres it and fills the rest with `pad_color_hex`. | Kept | Bars are added. |
| `stretch` | Scales width and height separately to the exact target size. | Not kept | The picture looks squashed or stretched. |

[NH Ratio Preset Image Resize](#nh-ratio-preset-image-resize) names the same three modes `crop`, `pad` and `fill`.

<p align="center"><img src="../assets/examples/resize-modes.jpg" alt="One photo resized to 16:9 with cover_crop, contain_pad and stretch" width="720"><br><em>The same photo resized to 16:9: <code>cover_crop</code>, <code>contain_pad</code> and <code>stretch</code>.</em></p>

### Resize methods

`resize_method` is the resampling filter. The image-resize nodes and [NH Remove Resize Padding](#nh-remove-resize-padding) offer the same eight; [NH Restore Image Geometry](#nh-restore-image-geometry) has its own short list.

| Method | Character |
|---|---|
| `lanczos` (default) | Sharp, high quality. Works in float precision, with no 8-bit rounding. |
| `bicubic`, `bilinear` | Smoother than `lanczos`. |
| `area`, `box`, `hamming` | Average the source pixels. Good for strong reductions. |
| `nearest`, `nearest-exact` | Copy pixels without smoothing. Keeps hard edges. |

### Undoing a resize

Each resize node outputs `resize_data` (`NH_RESIZE_DATA`). It records what the resize did: source and target size, mode, scale, content region, padding and crop. Wire it to the node that undoes the resize.

| You resized with | You want | Use |
|---|---|---|
| `contain_pad` | The picture without the bars | [NH Remove Resize Padding](#nh-remove-resize-padding), `content_region` |
| `contain_pad` or `stretch` | The original size back | [NH Remove Resize Padding](#nh-remove-resize-padding), `original_size` |
| `cover_crop` | The original size, with the cropped-away borders taken from the original image | [NH Restore Image Geometry](#nh-restore-image-geometry) |
| Any mode | A mask of where the new content sits | [NH Restore Image Geometry](#nh-restore-image-geometry), `mask` |

A full round trip with `cover_crop`:

```text
Load Image → NH Smart Ratio Image Resize (cover_crop) → … your processing … → NH Restore Image Geometry → Save Image
```

| Connect | To |
|---|---|
| Load Image `IMAGE` | Smart Ratio Image Resize `image` **and** Restore Image Geometry `original_image` |
| Smart Ratio Image Resize `image` | your processing nodes |
| the processed `IMAGE` | Restore Image Geometry `image` |
| Smart Ratio Image Resize `resize_data` | Restore Image Geometry `resize_data` |

<p align="center"><img src="../assets/examples/restore-geometry.jpg" alt="Round trip: resize, process, then Restore Image Geometry with its mask" width="720"><br><em>Resize, process, restore. The mask is white where the processed content was placed.</em></p>

> **Note:** `resize_data` describes the resize only. Extra crops, rotations or outpainting in your processing, and any other change a model makes to the framing, cannot be reversed with it.

---

## NH Smart Resolution Picker

<p align="center"><img src="../assets/nodes/NH_SmartResolutionPicker.svg" alt="NH Smart Resolution Picker node" width="360"></p>

Picks a width and height that suit your model family, and builds an empty latent of that size.

**Use it when** you start a text-to-image workflow and want a size the model handles well, or when several nodes need the same width and height.

| Input | Type | Default | Description |
|---|---|---|---|
| `model_family` | COMBO | `FLUX.2 Klein 9B` | `FLUX.2 Klein 4B`, `FLUX.2 Klein 9B`, `FLUX.2 Dev`, `Qwen-Image` or `Z-Image`. Changing it refreshes the `preset` list. |
| `preset` | COMBO | `2 MP (sweet spot) ⭐ \| 1:1 \| 1408×1408` | A size from the family's list, written as `tier \| ratio \| width×height`. The star marks the recommended square size. |
| `use_custom` *(optional)* | BOOLEAN | `false` | Ignore `preset` and use `custom_width` × `custom_height`. |
| `custom_width` *(optional)* | INT | `1024` | Range 64 to 4096, step 16. Used when `use_custom` is on. |
| `custom_height` *(optional)* | INT | `1024` | Range 64 to 4096, step 16. Used when `use_custom` is on. |
| `swap_orientation` *(optional)* | BOOLEAN | `false` | Swaps width and height, for example landscape to portrait. |
| `batch_size` *(optional)* | INT | `1` | Range 1 to 64. Number of empty latents in the output. |

| Output | Type | Description |
|---|---|---|
| `width` | INT | Final width in pixels. |
| `height` | INT | Final height in pixels. |
| `latent` | LATENT | An empty (all zeros) latent with the right shape for the family, ready for a sampler. |
| `info` | STRING | One-line summary: family, tier, ratio, size and megapixels. May add `Warning:`, `Note:` and `Adjustment:` lines. |
| `aspect_ratio` | FLOAT | Width divided by height. |

**How it works**

- **Presets:** the `preset` list shows the sizes of the selected family only. Some presets, such as the 4 MP tiers, are larger than the custom-size limit below.
- **Custom sizes:** the node rounds each side to the nearest multiple of 16 (32 for `Qwen-Image`), then clamps it to the family's range: 64 to 2048 px for `FLUX.2 Klein`, 400 to 2048 for `FLUX.2 Dev`, 512 to 2560 for `Qwen-Image`, 512 to 2048 for `Z-Image`. `info` lists every change it made.
- **Warnings:** a community, risk or extrapolation tier, or a size the node adjusted, puts a warning mark at the start of `info`.
- **Latent shape:** `FLUX.2` latents have 128 channels at 1/16 of the image size. `Qwen-Image` and `Z-Image` use 16 channels at 1/8.

**Example**

```text
NH Smart Resolution Picker → latent → KSampler (latent_image)
                           → width, height → any node that needs a size
```

> **Tip:** To edit an existing image instead of generating one, use [NH Smart Ratio Image Resize](#nh-smart-ratio-image-resize). It chooses the preset from the image itself.

---

## NH Smart Ratio Image Resize

<p align="center"><img src="../assets/nodes/NH_SmartRatioImageResize.svg" alt="NH Smart Ratio Image Resize node" width="360"></p>

Resizes an image to the preset whose aspect ratio is closest to the image's own.

**Use it when** you send a photo to an edit or image-to-image model and want a size the model handles well without choosing the ratio yourself, or when you will undo the resize later with `resize_data`.

| Input | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | The image or batch to resize. |
| `model_family` | COMBO | `FLUX.2 Klein 9B` | Whose presets to choose from: `FLUX.2 Klein 4B`, `FLUX.2 Klein 9B`, `FLUX.2 Dev`, `Qwen-Image` or `Z-Image`. |
| `resolution_level` | COMBO | `1 MP` | Target size, `1 MP` to `12 MP`. See [Resolution levels](#resolution-levels). |
| `resize_mode` | COMBO | `cover_crop` | `cover_crop`, `contain_pad` or `stretch`. See [Resize modes](#resize-modes). |
| `resize_method` | COMBO | `lanczos` | Resampling filter. See [Resize methods](#resize-methods). |
| `pad_color_hex` | STRING | `#FFFFFF` | Padding colour for `contain_pad`: `#RGB`, `#RRGGBB` or `#RRGGBBAA`. The alpha part sets transparency on RGBA images. Text that is not a colour gives white. |

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | The resized image or batch, exactly `width` × `height`. |
| `width` | INT | Width of the chosen preset. |
| `height` | INT | Height of the chosen preset. |
| `info` | STRING | Summary: preset, size, megapixels, source aspect ratio, mode and method. |
| `source_aspect_ratio` | FLOAT | Width divided by height of the input. |
| `target_aspect_ratio` | FLOAT | Width divided by height of the chosen preset. |
| `selected_preset` | STRING | Label of the chosen preset, for example `2 MP (sweet spot) \| 3:2 \| 1728×1152`. |
| `resize_data` | NH_RESIZE_DATA | Record of the resize, for [NH Remove Resize Padding](#nh-remove-resize-padding) and [NH Restore Image Geometry](#nh-restore-image-geometry). |

**How it works**

- The node compares the image's width ÷ height with every preset of the family at the chosen level and takes the closest ratio. On a tie it prefers the more trusted tier, then the size closest to the level.
- Presets are not always exact ratios. At 1 MP the `4:3` preset is 1152×896, so `cover_crop` can trim a few pixels even from a 4:3 photo.
- Every image of a batch gets the same size and shares one `resize_data`.
- `Qwen-Image` presets from the Native and HD tiers are not reachable here. Use [NH Smart Resolution Picker](#nh-smart-resolution-picker) for those.

**Example**

```text
Load Image → NH Smart Ratio Image Resize (FLUX.2 Klein 9B, 2 MP, cover_crop) → edit model → NH Restore Image Geometry → Save Image
```

A 1500×1000 photo becomes the `3:2` preset of the 2 MP tier, 1728×1152.

---

## NH Ratio Preset Image Resize

<p align="center"><img src="../assets/nodes/NH_RatioPresetImageResize.svg" alt="NH Ratio Preset Image Resize node" width="360"></p>

Resizes an image to a preset size for the aspect ratio you choose.

**Use it when** you need a fixed shape, such as 16:9 for video frames or 9:16 for phone screens, whatever the shape of the source image.

| Input | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | The image or batch to resize. |
| `model_family` | COMBO | `FLUX.2 Klein 9B` | Whose presets to choose from. Same list as above. |
| `resolution_level` | COMBO | `1 MP` | Target size, `1 MP` to `12 MP`. See [Resolution levels](#resolution-levels). |
| `target_ratio` | COMBO | `1:1` | `1:1`, `3:4`, `2:3`, `7:9`, `9:16`, `9:21`, `4:3`, `9:7`, `3:2`, `16:9` or `21:9`. The list in the node shows only the ratios the family offers at the chosen level. |
| `resize_mode` | COMBO | `crop` | `crop`, `pad` or `fill`. These are `cover_crop`, `contain_pad` and `stretch` under other names. |
| `resize_method` | COMBO | `lanczos` | Resampling filter. See [Resize methods](#resize-methods). |
| `pad_color_hex` | STRING | `#FFFFFF` | Padding colour for `pad`. Same rules as in [NH Smart Ratio Image Resize](#nh-smart-ratio-image-resize). |

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | The resized image or batch, exactly `width` × `height`. |
| `width` | INT | Width of the chosen preset. |
| `height` | INT | Height of the chosen preset. |
| `info` | STRING | Summary: requested and selected ratio, size, megapixels, mode and method. |
| `source_aspect_ratio` | FLOAT | Width divided by height of the input. |
| `target_aspect_ratio` | FLOAT | Width divided by height of the chosen preset. |
| `selected_preset` | STRING | Label of the chosen preset. |
| `resize_data` | NH_RESIZE_DATA | Record of the resize, for the nodes that undo it. |

**How it works**

- If a saved workflow asks for a ratio the family does not offer at that level, the node uses the closest available ratio and says so in `info`.
- `fill` stretches the image to the target. It does not fill with a colour. Use `pad` for coloured bars.
- Above 4 MP the sizes are calculated and labelled `extrapolated`. See [Resolution levels](#resolution-levels).

**Example**

```text
Load Image → NH Ratio Preset Image Resize (FLUX.2 Klein 9B, 2 MP, 16:9, crop) → …
```

The output is 1888×1056. `contain_pad` would keep the whole photo and add bars of `pad_color_hex`.

---

## NH Remove Resize Padding

<p align="center"><img src="../assets/nodes/NH_RemoveResizePadding.svg" alt="NH Remove Resize Padding node" width="360"></p>

Removes the padding that a resize added, and can scale the result back to the original size.

**Use it when** you resized with `contain_pad`, processed or upscaled the image, and now want the bars gone.

| Input | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | The image to clean. It can be a processed or upscaled version of the resize output. |
| `resize_data` | NH_RESIZE_DATA | — | The `resize_data` from the resize node that made the padding. |
| `output_size` | COMBO | `content_region` | `content_region` keeps the picture at its own size. `original_size` scales it to the exact width and height of the source image. |
| `resize_method` | COMBO | `lanczos` | Filter for the scaling that `original_size` needs. See [Resize methods](#resize-methods). |

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | The image without padding, or at the original size. |
| `width` | INT | Width of the result. |
| `height` | INT | Height of the result. |
| `info` | STRING | What the node did: padding removed or not, the crop box and the output size. |

**How it works**

| Resize mode used | `content_region` | `original_size` |
|---|---|---|
| `contain_pad` | Padding cropped away. | Padding cropped away, then the picture is scaled to the original size. |
| `stretch` | Unchanged, there is nothing to remove. | Stretched back to the original size. |
| `cover_crop` | Unchanged. The cropped part cannot be recovered, and `info` says so. | Scaled back evenly into the original frame. The borders that were cut off repeat the nearest edge pixels. |

- The crop box follows the image if it was upscaled after the resize, so a 2× upscaled result still loses exactly its padding.
- Without usable `resize_data` the image passes through unchanged and `info` says so.

**Example**

```text
Load Image → NH Smart Ratio Image Resize (contain_pad) → upscale or edit → NH Remove Resize Padding (original_size) → Save Image
```

> **Tip:** After `cover_crop`, prefer [NH Restore Image Geometry](#nh-restore-image-geometry). It keeps the real borders of the original instead of repeating edge pixels.

---

## NH Restore Image Geometry

<p align="center"><img src="../assets/nodes/NH_RestoreImageGeometry.svg" alt="NH Restore Image Geometry node" width="360"></p>

Puts a processed image back into the frame of the original, and keeps the original pixels wherever the processed image does not reach.

**Use it when** you resized with `cover_crop`, ran the image through a model or upscaler, and want a result at the exact original size, or you need a mask for blending the processed area into the original.

| Input | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | The processed image. It may be larger than the resize output, even by a non-integer factor, as long as its framing is unchanged. |
| `original_image` | IMAGE | — | The image that went into the resize node. Its size must match the source size stored in `resize_data`. |
| `resize_data` | NH_RESIZE_DATA | — | The `resize_data` from [NH Smart Ratio Image Resize](#nh-smart-ratio-image-resize) or [NH Ratio Preset Image Resize](#nh-ratio-preset-image-resize). |
| `resize_method` | COMBO | `bicubic` | Sampling used to bring the processed image back: `bicubic`, `bilinear` or `nearest`. |

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | The original-size image: processed content in its original region, original pixels everywhere else. |
| `mask` | MASK | White where processed content was placed, black where the original was kept. |

**How it works**

- **Cropped result (`cover_crop`):** the processed picture goes back to the region it came from. The borders that were cropped away stay as they were in `original_image`, and the mask is black there.
- **Padded result (`contain_pad`):** the padding is ignored and only the picture is restored, so the mask covers the whole frame. `stretch` also covers the whole frame.
- **Batches:** both inputs need the same batch size, or one of them a single image, which is reused. The channel count (RGB or RGBA) must match.
- **Limits:** the metadata cannot undo extra crops, rotations or content a model added at the edges. Restoring does not bring back detail lost when the image was reduced.
- **Errors:** the node stops with a clear message if `resize_data` is not from an NH resize node or `original_image` has the wrong size.

**Example**

```text
Load Image → NH Smart Ratio Image Resize (cover_crop) → edit or upscale → NH Restore Image Geometry → Save Image
```

Also connect Load Image to `original_image` and `resize_data` to `resize_data`, as in [Undoing a resize](#undoing-a-resize).

> **Tip:** The mask has hard edges. For a soft transition, shrink it and then blur it with [Mask Morphology (NH)](./mask.md#mask-morphology-nh), so the fade stays inside the processed area. Then composite `image` over `original_image` with that mask, for example with ComfyUI's `ImageCompositeMasked`.

---

[← Back to guide](./README.md)
