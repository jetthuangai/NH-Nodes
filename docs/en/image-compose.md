[← Back to guide](./README.md)

# Image — Tile, Compose and Resize

Cut a large image into tiles and rebuild it, lay images out in grids and page layouts, paste layers at exact positions, compare two images, add a text label, and resize to millimetres or centimetres. These nodes work on the pixels of `IMAGE` batches. No model is involved. Find them under **NH-Nodes → Image** in the node menu.

**On this page**

| Node | Use it to |
|---|---|
| [Image Tile (NH)](#image-tile-nh) | Cut an image into overlapping tiles |
| [Image Untile (NH)](#image-untile-nh) | Put processed tiles back together, blending the overlaps |
| [Image Grid Composite (NH)](#image-grid-composite-nh) | Build a contact sheet from a batch |
| [Layer Layout Composite (NH)](#layer-layout-composite-nh) | Place images on a page from a preset, a grid or your own slots |
| [Layer Stack Composite (NH)](#layer-stack-composite-nh) | Paste layers on a background at positions you calculate |
| [Image Compare (NH)](#image-compare-nh) | Compare two images side by side, split, blended or as a difference |
| [Image Label (NH)](#image-label-nh) | Add a text band, or text over the picture |
| [Image Resize Unit (NH)](#image-resize-unit-nh) | Resize to pixels, millimetres or centimetres |

**Colour fields** (`background_color`, `frame_color_hex`, `pad_color_hex`, …) take `#RRGGBB`, `#RGB`, `#RRGGBBAA` or a colour name such as `red`. Always start a hex value with `#`. A value the node cannot read falls back to the default colour of that field. The alpha part counts only where the result can be transparent.

---

## Image Tile (NH)

<p align="center"><img src="../assets/nodes/NH_ImageTile.svg" alt="Image Tile (NH) node" width="360"></p>

Cuts an image into overlapping tiles of the same size, and records where every tile came from, so that Image Untile can rebuild the image.

**Use it when** an image is too large for a model or an upscaler, or when every part of the image needs the same treatment, such as an upscale or a detail pass per tile.

| Input | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | The image or batch to cut. Every image of a batch gets the same tile layout. |
| `tile_mode` | COMBO | `original_ratio` | How the tile size is chosen. `original_ratio`: tiles have the shape of the image and `tile_size` is their long side. `square`: `tile_size` × `tile_size`. `custom`: `tile_width` × `tile_height`. |
| `tile_size` | INT | `1024` | `16` to `16384`, step 8. Used by `original_ratio` and `square`. |
| `tile_width` | INT | `1024` | `16` to `16384`, step 8. Used by `custom` only. |
| `tile_height` | INT | `1024` | `16` to `16384`, step 8. Used by `custom` only. |

| Output | Type | Description |
|---|---|---|
| `tiles` | IMAGE | All tiles as one batch, all the same size, in reading order: left to right, top to bottom. For a batch, the tiles of the first image come first, then those of the second. |
| `tile_data` | NH_TILE_DATA | The tile layout. Connect it to Image Untile. |
| `tile_count` | INT | The total number of tiles. |
| `overlap_x` | INT | The horizontal overlap between neighbouring tiles, in pixels. |
| `overlap_y` | INT | The vertical overlap, in pixels. |
| `tile_info` | STRING | A two-line summary, such as `grid=3x2, overlap=128x128, tiles=6`. |

**How it works**

- **Overlap is automatic.** It is about one eighth of the tile side, rounded to a multiple of 8 and kept between 16 and 128 px (tiles under 128 px use proportionally less). On an axis where the image is not larger than the tile, there is no overlap.
- **Layout.** Tiles advance by the tile size minus the overlap. The last tile of each row and column is moved back so that it ends exactly on the image edge, so it can overlap its neighbour by more than `overlap_x` or `overlap_y`. Because of the overlap, a 2048 px side needs three 1024 px tiles, not two.
- **Small images.** If the image is smaller than a tile on one axis, the tile covers the whole side and the rest is filled by repeating the edge pixels. Image Untile cuts that filling off again.

Tile counts you can expect:

| Image | Settings | Tile | Grid | Tiles |
|---|---|---|---|---|
| 2048 × 1536 | `square`, `tile_size` 1024 | 1024 × 1024 | 3 × 2 | 6 |
| 3000 × 2000 | `original_ratio`, `tile_size` 1024 | 1024 × 683 | 4 × 4 | 16 |
| 1920 × 1080 | `custom`, 768 × 512 | 768 × 512 | 3 × 3 | 9 |
| 4096 × 4096 | `square`, `tile_size` 1024 | 1024 × 1024 | 5 × 5 | 25 |
| 4096 × 4096 | `square`, `tile_size` 512 | 512 × 512 | 9 × 9 | 81 |
| 800 × 600 | `square`, `tile_size` 1024 | 1024 × 1024 | 1 × 1 | 1 |

<p align="center"><img src="../assets/examples/tile-untile.jpg" alt="An image with its tile grid and overlap, the tiles, and the reassembled image" width="720"><br><em>An image with its tile grid and overlap, the tiles, and the image rebuilt by Image Untile.</em></p>

**Example — upscale in tiles**

```text
Load Image → Image Tile (NH) → Upscale Image (using Model) → Image Untile (NH) → Save Image
Image Tile (NH).tile_data → Image Untile (NH).tile_data
```

For a detail pass instead, put `VAE Encode → KSampler (low denoise) → VAE Decode` between the two tile nodes. Image Untile reads the size of the processed tiles, so the result is scaled by the same factor as the tiles.

**Common mistakes**

- Not connecting `tile_data` straight from Image Tile to Image Untile, or using the `tile_data` of a different Image Tile node.
- Changing the number or the order of the tiles between the two nodes, for example with a batch filter.
- Cropping or padding the tiles. Scaling them is fine, but all tiles must keep the same shape.

> **Tip:** A wide strip along the right or bottom edge where two tiles overlap more than usual is averaged, which can soften generated detail. Choose a size so that the image side is the tile size plus a whole number of (tile size − overlap), for example 1920 px with 1024 px tiles.

---

## Image Untile (NH)

<p align="center"><img src="../assets/nodes/NH_ImageUntile.svg" alt="Image Untile (NH) node" width="360"></p>

Puts processed tiles back together into the full image, blending the overlaps, using the layout from Image Tile.

**Use it when** you cut an image with Image Tile and want it whole again, at the original size or enlarged.

| Input | Type | Default | Description |
|---|---|---|---|
| `tiles` | IMAGE | — | The processed tiles, in the same number and order that Image Tile produced. Extra tiles are ignored. All tiles must have the same size, and they can all be enlarged by the same factor, such as 2× or 3×. |
| `tile_data` | NH_TILE_DATA | — | The layout from Image Tile. |

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | The rebuilt image, one for each image that went into Image Tile. |
| `untile_info` | STRING | A summary, such as `images=1, tiles=6, scale=2x2, overlap=256x256`. |

**How it works**

- **Scale.** The node divides the size of the processed tiles by the original tile size, for width and height separately. Upscaled tiles give an upscaled image with no extra setting. The overlap is scaled the same way.
- **Feathering.** In every overlap strip, neighbouring tiles cross-fade with a linear ramp as wide as the overlap. The outer border of the image is not faded. There is no setting for this.
- **Edge filling.** The padding that Image Tile added to small images is cropped off.
- **Errors.** If you give fewer tiles than expected, the node stops with `Not enough tiles: got N, expected M`.

**Example**

```text
Image Tile (NH).tiles → Upscale Image (using Model) → Image Untile (NH).tiles
Image Tile (NH).tile_data → Image Untile (NH).tile_data
```

A 2048 × 1536 image cut into 6 tiles and upscaled 2× comes back as 4096 × 3072.

---

## Image Grid Composite (NH)

<p align="center"><img src="../assets/nodes/NH_ImageGridComposite.svg" alt="Image Grid Composite (NH) node" width="360"></p>

Puts a batch of images into a grid, a contact sheet, with cells of one size, gaps and a background colour.

**Use it when** you want to see several results side by side in one picture, or deliver a sheet of thumbnails.

| Input | Type | Default | Description |
|---|---|---|---|
| `images` | IMAGE | — | The images to place, left to right, top to bottom. It takes a batch, or several batches or lists, and puts all images in one sequence. |
| `columns` | INT | `2` | `1` to `64`. |
| `rows` | INT | `2` | `1` to `64`. |
| `cell_size_mode` | COMBO | `manual` | `manual`: use `cell_width` and `cell_height`. `first_image`: every cell gets the size of the first image, and the two size fields are ignored. |
| `cell_width` | INT | `512` | `1` to `16384`. |
| `cell_height` | INT | `512` | `1` to `16384`. |
| `resize_mode` | COMBO | `stretch` | How an image fits its cell. `stretch`: fill the cell, changing the proportions. `pad`: keep the proportions and fill the rest with the background colour. `fill`: keep the proportions, cover the cell and cut the overflow evenly from the centre. |
| `gap_x` | INT | `0` | `0` to `4096` px between columns. |
| `gap_y` | INT | `0` | `0` to `4096` px between rows. |
| `background_color_hex` | STRING | `#000000` | The colour of the gaps and of the `pad` areas. |

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | The finished grid, one image. |
| `slot_count` | INT | `columns` × `rows`. |
| `used_count` | INT | How many images were placed: the smaller of the image count and `slot_count`. |
| `grid_info` | STRING | A summary, then one line per placed image with its position and size. |

**How it works**

- The sheet is `columns × cell_width + (columns − 1) × gap_x` wide and `rows × cell_height + (rows − 1) × gap_y` high. There is no outer margin.
- Images beyond `columns × rows` are left out. Empty cells show the background colour.
- The result is RGB. Transparency in the inputs is dropped.
- **Memory guard.** The node needs about 28 bytes of RAM for each pixel of the sheet. If the sheet would not fit in the free memory, it stops with a message asking you to lower the columns, rows, cell size or gaps.

<p align="center"><img src="../assets/examples/grid-composite.jpg" alt="A 2 by 2 contact sheet with labels" width="720"><br><em>A 2 × 2 contact sheet with labels.</em></p>

**Example — a labelled contact sheet**

```text
4 × Image Label (NH) → Image Batch → Image Grid Composite (NH) → Preview Image
```

Use any node that joins images into one batch, such as `Image Batch` or [Batch Merge (NH)](./batch-lists.md#batch-merge-nh). Set `columns` and `rows` to `2`, `resize_mode` to `pad`, and `gap_x` and `gap_y` to `8`.

> **Tip:** With `first_image`, all cells take the size of the first image. Put the image whose size you want first in the batch.

---

## Layer Layout Composite (NH)

<p align="center"><img src="../assets/nodes/NH_LayerLayoutComposite.svg" alt="Layer Layout Composite (NH) node" width="360"></p>

Places each image of a batch into the slots of a page layout, from a preset, a grid or your own coordinates, on a canvas of any size.

**Use it when** you build a poster, a photo sheet or a catalogue page, such as six photos on an A4 page.

| Input | Type | Default | Description |
|---|---|---|---|
| `images` | IMAGE | — | The images for the slots. Image 1 goes to slot 1, image 2 to slot 2, and so on. |
| `canvas_width` | INT | `2480` | `1` to `20000` px. The default canvas, 2480 × 3508 px, is an A4 page at 300 DPI. |
| `canvas_height` | INT | `3508` | `1` to `20000` px. |
| `background_color` | STRING | `#FFFFFF` | The canvas colour. A colour with alpha below `FF` (for example `#FFFFFF00`) gives a transparent RGBA result. |
| `layout_mode` | COMBO | `preset` | Where the slots come from. `preset`: from `layout_preset`. `grid`: from `columns` and `rows`. `custom`: from `custom_layout`. |
| `layout_preset` | COMBO | `2x3` | The preset for `preset` mode. See the table below. |
| `resize_mode` | COMBO | `cover` | How an image fits its slot. `cover`: fill the slot and cut the overflow from the centre. `contain`: fit the whole image, centred, with the background around it. `stretch`: fill the slot, changing the proportions. |
| `columns` | INT | `2` | `1` to `20`. Used in `grid` mode. |
| `rows` | INT | `3` | `1` to `20`. Used in `grid` mode. |
| `margin_x` | INT | `150` | `0` to `10000` px of empty space at the left and right edges. |
| `margin_y` | INT | `150` | `0` to `10000` px at the top and bottom edges. |
| `gap_x` | INT | `150` | `0` to `10000` px between slots in a row. |
| `gap_y` | INT | `150` | `0` to `10000` px between slots in a column. |
| `custom_units` | COMBO | `auto` | The unit of the numbers in `custom_layout`: `auto`, `relative`, `percent` or `pixels`. |
| `custom_layout` | STRING | six relative slots | One slot per line. See "Custom layout" below. |
| `background_fit` | COMBO | `stretch` | How `background_image` fits the canvas: `stretch`, `cover` or `contain`. |
| `background_image` *(optional)* | IMAGE | — | A picture drawn on the canvas before the slots. Only the first image of the batch is used. |

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | The finished page, one image. |
| `slot_count` | INT | How many slots the layout has. |
| `used_count` | INT | How many slots were filled: the smaller of the image count and `slot_count`. |
| `layout_info` | STRING | One line per placed image, such as `0: x=10, y=10, w=323, h=687, mode=contain`. |

**Which inputs matter in which mode**

| Input | `preset` | `grid` | `custom` |
|---|:---:|:---:|:---:|
| `layout_preset` | yes | — | — |
| `columns`, `rows` | — | yes | — |
| `margin_x`, `margin_y`, `gap_x`, `gap_y` | yes | yes | — |
| `custom_layout`, `custom_units` | — | — | yes |

**Presets**

Margins shrink the area for the slots, and gaps separate them.

| Preset | Slots | Layout |
|---|---|---|
| `2x3` | 6 | 2 columns × 3 rows, all slots the same size |
| `2x3_mixed` | 6 | 2 columns × 3 rows, with rows of about 24 %, 24 % and 52 % of the height |
| `2_top_1_bottom` | 3 | two slots side by side on top (about 42 % of the height) and one wide slot below |
| `1_full` | 1 | one slot that fills the area inside the margins |
| `2_vertical` | 2 | 1 column × 2 rows, one above the other |
| `2_horizontal` | 2 | 2 columns × 1 row, side by side |
| `3_vertical` | 3 | 1 column × 3 rows |
| `4_grid` | 4 | 2 × 2 |

**Custom layout**

Write one slot per line as `x, y, width, height`.

```text
# x, y, width, height, relative to the canvas
0.05, 0.05, 0.90, 0.60     # a large photo on top
0.05, 0.70, 0.43, 0.25     # two small photos below
0.52, 0.70, 0.43, 0.25
```

- Separate the numbers with commas, semicolons or spaces. Everything after `#` is a comment. Empty lines are skipped.
- If a line has more than four numbers, the last four are used, so you can start a line with its own number.
- Width and height must be above zero. A slot can overlap another one or run past the canvas edge. The parts outside are cut off.
- Slot *n* gets image *n*. In this mode `margin_*` and `gap_*` are ignored.

| `custom_units` | The numbers mean |
|---|---|
| `relative` | A fraction of the canvas size, from `0` to `1`. |
| `percent` | A percentage of the canvas size, from `0` to `100`. |
| `pixels` | Pixels. |
| `auto` | `relative` when every number in the line is between −1 and 1, otherwise `pixels`. `auto` never picks `percent`. |

**How it works**

- If you give more images than slots, the extra images are left out. If you give fewer, the empty slots show the canvas.
- Images with an alpha channel are pasted with their own transparency, so the canvas shows through. The same goes for an RGBA `background_image`: its transparent areas show `background_color`.
- The result is RGB. It is RGBA only when `background_color` has an alpha below `FF`.

<p align="center"><img src="../assets/examples/layer-layout.jpg" alt="A page built with Layer Layout Composite" width="720"><br><em>A page built with Layer Layout Composite.</em></p>

**Example — six photos on an A4 page**

```text
Image Batch (6 images) → Layer Layout Composite (NH) → Save Image Path (NH)
```

Keep the defaults: `preset`, `2x3`, `cover`, 150 px margins and gaps. In Save Image Path, set `dpi` to `300` so the page prints at A4 size.

> **Tip:** Use `contain` when no part of a photo may be cut, and `cover` when a full slot looks better than a complete picture.

---

## Layer Stack Composite (NH)

<p align="center"><img src="../assets/nodes/NH_LayerStackComposite.svg" alt="Layer Stack Composite (NH) node" width="360"></p>

Pastes up to 64 layers on a background image. You write each position as numbers or as a formula.

**Use it when** you place a logo, a watermark or stickers at exact spots. A spot can depend on the size of the background or of another layer, such as "centred" or "bottom right with a 40 px margin".

| Input | Type | Default | Description |
|---|---|---|---|
| `background_image` | IMAGE | — | The base picture. Only the first image of the batch is used. The result has its size. |
| `layer_count` | INT | `1` | `0` to `64`. How many layers to place. The node shows that many `layer_N` inputs. |
| `positions` | STRING | `0,0` | One line per layer: `x, y` in pixels from the top-left corner. Formulas are allowed. See below. |
| `layer_1` … `layer_64` *(optional)* | IMAGE | — | The layers. Only the first image of each is used. `layer_1` is pasted first, so higher numbers sit on top. |

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | The background with the layers pasted on it. |
| `used_count` | INT | How many layers were pasted, meaning at least partly inside the picture. |
| `layer_info` | STRING | One line per layer: its position and size, `missing` when no image is connected, or `outside canvas`. |

**Writing positions**

One line is one layer: line 1 is `layer_1`, line 2 is `layer_2`, and so on. Each line has an x formula and a y formula, separated by a comma or a semicolon. Empty lines and `#` comments do not count. A layer without a line goes to `0, 0`.

```text
(a - w1) / 2, (b - h1) / 2     # layer 1, centred
x1 + w1 + 20, y1               # layer 2, to the right of layer 1 with a 20 px gap
a - w3 - 40, b - h3 - 40       # layer 3, bottom right with a 40 px margin
```

Formulas use numbers, parentheses, `+ - * / // % **` and a leading minus. Decimal numbers use a dot. Functions such as `min` or `max` are not available, and `^` is not a power (use `**`). The result is rounded to whole pixels. Division by zero is an error.

| Variable | Value |
|---|---|
| `a`, `bg_w`, `width` | The width of the background. |
| `b`, `bg_h`, `height` | The height of the background. |
| `w1`, `w2`, … | The width of layer 1, 2, …. It is `0` when that layer has no image. |
| `h1`, `h2`, … | The height of layer 1, 2, …. |
| `x1`, `y1`, `x2`, … | The position already worked out for an earlier line. A line can use only the lines above it. |

Names are not case-sensitive. A formula the node cannot read stops the run with `Invalid position expression` and the text of the formula.

**How it works**

- The part of a layer outside the picture is cut off. A layer completely outside is not pasted, and `layer_info` says `outside canvas`.
- Layers with an alpha channel are pasted with their own transparency. The result has the channels of the background: RGB stays RGB, and an RGBA background gives an RGBA result.
- The node shows only the inputs up to `layer_count`. When you lower `layer_count`, the inputs above it, and their links, are removed. The **update layers** button on the node refreshes the inputs, for example after you load a workflow.
- The result is cached like that of any other node: the node runs again only when the background, a layer, `layer_count` or `positions` changes.

**Example — a watermark in the bottom right corner**

```text
Load Image (photo).image → Layer Stack Composite (NH).background_image
Load Image (logo).image + .mask → Join Image with Alpha → Layer Stack Composite (NH).layer_1
```

Set `layer_count` to `1` and `positions` to `a - w1 - 40, b - h1 - 40`.

> **Note:** Load Image gives the picture in RGB and its transparency as a separate `MASK`. To paste a logo with transparent areas, join the two again with the core node `Join Image with Alpha`, as above. The same works with the `image` and `mask` outputs of [Load Image Info (NH)](./image-io.md#load-image-info-nh).

**Common mistakes**

- Changing `layer_count` but not connecting the new inputs. The extra layers show `missing` in `layer_info`.
- Expecting a batch of layers. Only the first image of each input is used.
- Writing a formula with `min()` or `^`. Rewrite it with the operators above.

---

## Image Compare (NH)

<p align="center"><img src="../assets/nodes/NH_ImageCompare.svg" alt="Image Compare (NH) node" width="360"></p>

Combines two images into one picture so you can compare them: side by side, stacked, split, blended, or as a difference.

**Use it when** you want a before and after, or to see exactly what a step changed.

| Input | Type | Default | Description |
|---|---|---|---|
| `image_a` | IMAGE | — | The first image. It sets the size of the result. |
| `image_b` | IMAGE | — | The second image. It is resized to the size of `image_a`. |
| `compare_mode` | COMBO | `side_by_side` | How to combine them. See the table below. |
| `split_position` | FLOAT | `0.5` | `0` to `1`, step 0.01. Where the dividing line sits in the two `split_*` modes. |
| `overlay_opacity` | FLOAT | `0.5` | `0` to `1`, step 0.01. The blend in `overlay` mode: `0` shows only A, `1` shows only B. |
| `resize_method` | COMBO | `lanczos` | How `image_b` is resized to `image_a`: `lanczos`, `bicubic`, `bilinear` or `nearest`. |

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | The combined picture. |

| `compare_mode` | Result |
|---|---|
| `side_by_side` | A on the left, B on the right. Twice as wide as A. |
| `top_bottom` | A on top, B below. Twice as tall as A. |
| `split_vertical` | One picture divided by a vertical line: A to the left of `split_position`, B to the right. |
| `split_horizontal` | One picture divided by a horizontal line: A above `split_position`, B below. |
| `difference` | The absolute difference of A and B in every channel. Black where they are the same. |
| `overlay` | A and B blended with `overlay_opacity`. |

**How it works**

- **Batches.** If both batches have the same size, image *n* of A is compared with image *n* of B. If one has a single image, it is repeated for every image of the other. Otherwise both are cut to the shorter length, and a message appears in the console.
- The result is RGB. Transparency is dropped.

<p align="center"><img src="../assets/examples/image-compare.jpg" alt="Output of Image Compare" width="720"><br><em>Output of Image Compare.</em></p>

**Example — before and after**

```text
Load Image → Image Compare (NH).image_a
KSampler → VAE Decode → Image Compare (NH).image_b
Image Compare (NH) → Preview Image
```

Set `compare_mode` to `split_vertical` and `split_position` to `0.5`.

> **Tip:** The difference is not amplified, so a small change looks almost black. Use `overlay` or a split to judge subtle changes. For images of tens of megapixels, use [NH Large Image Compare](./image-preview.md#nh-large-image-compare).

---

## Image Label (NH)

<p align="center"><img src="../assets/nodes/NH_ImageLabel.svg" alt="Image Label (NH) node" width="360"></p>

Adds a text band above or below an image, or writes the text over the picture, with a choice of font, colours and outline.

**Use it when** you want captions on a contact sheet, a "before" and "after" tag on a comparison, or the model or seed printed on a result.

| Input | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | The image or batch to label. Every image gets the same label. |
| `label_text` | STRING | `NH Label` | The text. Start a new line to write several lines. |
| `position` | COMBO | `header` | `header`: at the top. `bottom`: at the bottom. |
| `frame_height` | INT | `80` | `1` to `4096`. The height of the band in pixels. Make it taller than the text. |
| `fill_enabled` | BOOLEAN | `true` | `true`: fill the band with `frame_color_hex`. `false`: build the band from the top pixel row (header) or the bottom pixel row (bottom) of the image, stretched, so the band blends with the picture. |
| `frame_color_hex` | STRING | `#111111` | The band colour. |
| `text_color_hex` | STRING | `#FFFFFF` | The text colour. |
| `font_name` | COMBO | `default` | `default`, or a font file from the Windows `Fonts` folder. On other systems only `default` is offered. `default` uses Arial when it is available, and Pillow's built-in font otherwise. |
| `font_size` | INT | `32` | `6` to `512`. |
| `fit_text` | BOOLEAN | `true` | Shrink the text, down to size 8, until it fits the image width minus both side paddings. The text is never wrapped. |
| `overlay_on_image` | BOOLEAN | `false` | `false`: add the band outside, so the image grows taller by `frame_height`. `true`: draw the band over the edge of the picture and keep the size. |
| `align` | COMBO | `left` | `left`, `center` or `right`. |
| `padding_x` | INT | `20` | `0` to `2048` px of space at the left and right of the text. |
| `outline_width` | INT | `0` | `0` to `32` px. The width of a line around the letters. |
| `outline_color_hex` | STRING | `#000000` | The outline colour. |

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | The labelled image, in RGB. |

**How it works**

- The text is centred vertically in the band. If the band is shorter than the text, the text runs past the band.
- With `overlay_on_image` on, the band covers part of the picture. Its height is limited to the image height.
- The list of fonts is read when ComfyUI starts. Restart ComfyUI after you install a font.

<p align="center"><img src="../assets/examples/image-label.jpg" alt="Output of Image Label" width="720"><br><em>Output of Image Label.</em></p>

**Example — tag a result**

```text
VAE Decode → Image Label (NH) → Preview Image
```

Set `label_text` to `After`, `position` to `bottom`, `align` to `center` and `outline_width` to `2`.

> **Tip:** To print text from elsewhere in your workflow, such as a file name, connect a STRING output to `label_text`.

---

## Image Resize Unit (NH)

<p align="center"><img src="../assets/nodes/NH_ImageResizeByUnit.svg" alt="Image Resize Unit (NH) node" width="360"></p>

Resizes an image to a width and height given in pixels, millimetres or centimetres. A DPI value converts physical units to pixels.

**Use it when** the result must have a print size, such as A4 or 10 × 15 cm, or when you want to set one side and let the other follow.

| Input | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | The image or batch to resize. |
| `unit` | COMBO | `pixel` | The unit of `width_value` and `height_value`: `pixel`, `mm` or `cm`. |
| `width_value` | FLOAT | `1024.0` | `0` to `100000`, step 0.1. The target width in `unit`. `0` means: work it out from the height and keep the proportions. |
| `height_value` | FLOAT | `1024.0` | `0` to `100000`, step 0.1. The target height. `0` means: work it out from the width. |
| `dpi` | INT | `300` | `1` to `2400`. Used only for `mm` and `cm`. Pixels = size ÷ 25.4 × `dpi` for millimetres, size ÷ 2.54 × `dpi` for centimetres. |
| `resize_mode` | COMBO | `stretch` | How the image reaches the size. See the table below. |
| `resize_method` | COMBO | `nearest` | The resampling filter: `nearest`, `bilinear`, `bicubic`, `area`, `nearest-exact` or `lanczos`. |
| `pad_color_hex` | STRING | `#000000` | The padding colour for `pad`. The alpha part applies to RGBA images. |

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | The resized image. |
| `width_px` | INT | The real width in pixels. |
| `height_px` | INT | The real height in pixels. |

| `resize_mode` | Result |
|---|---|
| `stretch` | Exactly width × height. The proportions change if the shape differs. |
| `pad` | The whole image fits inside, centred, on a canvas of exactly width × height filled with `pad_color_hex`. |
| `crop` | The image covers the target, and the overflow is cut evenly from the centre. The size is exactly width × height. |
| `lock_ratio` | The proportions are kept and the image fits inside width × height. One side can be smaller than requested. |

**How it works**

- If both `width_value` and `height_value` are `0`, the image keeps its size.
- `nearest` and `nearest-exact` give the same result.
- The `dpi` here only converts units. To store a DPI value in a PNG or JPG file, set `dpi` in [Save Image Path (NH)](./image-io.md#save-image-path-nh).

Examples with an 800 × 600 image:

| Settings | Result |
|---|---|
| `mm`, 210 × 297, `dpi` 300, `stretch`, `pad` or `crop` | 2480 × 3508 px (A4) |
| `cm`, 10 × 15, `dpi` 300, `lock_ratio` | 1181 × 886 px |
| `pixel`, 1024 × 0, `stretch` | 1024 × 768 px |
| `pixel`, 0 × 0 | 800 × 600 px |

**Example — an A4 page**

```text
Load Image → Image Resize Unit (NH) → Save Image Path (NH)
```

Set `unit` to `mm`, `width_value` to `210`, `height_value` to `297`, `dpi` to `300` and `resize_mode` to `pad`. In Save Image Path, set `dpi` to `300`.

> **Tip:** The default `resize_method` is `nearest`, which is fast and keeps hard edges. For photos, choose `lanczos` or `bicubic`. To resize for a model, see [Resolution](./resolution.md).

---

**See also:** [Image — Load and Save](./image-io.md) for saving the result, [Image — Mask-Based Retouch](./image-retouch.md) for the mask-aware nodes, and [Resolution](./resolution.md) for model-aware resizing.

[← Back to guide](./README.md)
