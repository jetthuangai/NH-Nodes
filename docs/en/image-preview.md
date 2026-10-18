[← Back to guide](./README.md)

# Image — Large Image Preview

The browser canvas struggles with images of tens of megapixels. These nodes keep a small thumbnail on the node and open a zoomable viewer when you ask for it. The full-size pixels wait in a temporary cache on disk. A third node shows what is in that cache and cleans it. Find them under **NH-Nodes → Image** in the node menu.

**On this page**

| Node | Use it to |
|---|---|
| [NH Large Image Preview](#nh-large-image-preview) | Look at a large image with a thumbnail on the node and a zoomable viewer |
| [NH Large Image Compare](#nh-large-image-compare) | Compare two large images with a slider, side by side or as a difference |
| [NH Large Preview Cache Manager](#nh-large-preview-cache-manager) | See what the preview cache holds, and delete old or oversized entries |

## How the viewer works

Each time a preview or compare node runs, it writes one cache entry for every image:

- a **thumbnail**, no larger than `preview_megapixels`, which appears on the node;
- **tiles** of the image at full size and at several smaller sizes, so the viewer loads only the part you look at, and sharper tiles as you zoom in;
- a **manifest**, a small JSON text that describes the entry. The node returns it in its `manifest` output.

Entries live in the ComfyUI `temp` folder, in a sub-folder named `nh_large_preview`. Each entry has a 32-character id. ComfyUI normally clears its `temp` folder when it starts, so the cache does not survive a restart. Every run adds new entries, so the cache grows until you clean it with [NH Large Preview Cache Manager](#nh-large-preview-cache-manager).

To open the viewer, run the workflow once, then click **Open Viewer** (preview) or **Open Compare** (compare) on the node. Before the first run, the button tells you to queue the node first.

| To | Do this |
|---|---|
| Pan | Drag with the mouse. |
| Zoom | Use the mouse wheel, or the **+** and **−** buttons. The zoom goes from a quarter of the fitted size up to 400 %. |
| Fit the image in the window | Click **Fit**. |
| See the real pixels | Click **100%**. |
| Move to the next or the previous image of a batch | Click ◀ or ▶, or press the Left or Right arrow key. These controls appear when the run produced more than one image, with a counter such as `2 / 5` between them. Each image starts fitted. |
| Open the thumbnail in a new browser tab | Click **Thumb** (preview viewer only). |
| Close | Click **Close**, press Esc, or click the dark area outside the window. |

---

## NH Large Image Preview

<p align="center"><img src="../assets/nodes/NH_LargeImagePreview.svg" alt="NH Large Image Preview node" width="360"></p>

Shows a thumbnail of a large image on the node, opens a zoomable viewer on demand, and passes the image on unchanged.

**Use it when** you need to inspect a large result, such as a tiled upscale of 50 megapixels, without sending the full image to the browser canvas.

| Input | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | The image or batch to preview. |
| `preview_megapixels` | FLOAT | `1.0` | `0.05` to `8`, step 0.05. The largest size of the thumbnail on the node, in megapixels. A smaller image keeps its size. |
| `preview_format` | COMBO | `jpg` | The thumbnail format: `jpg`, `webp` or `png`. |
| `preview_quality` | INT | `90` | `1` to `100`. The thumbnail quality. |
| `viewer_mode` | COMBO | `modal` | `modal`, `side_panel` or `new_tab`. The value is stored in the manifest. The viewer currently always opens as a window over the page, whatever you choose. |
| `title` | STRING | `NH Large Image Preview` | The title in the viewer header. |
| `tile_size` *(optional)* | INT | `512` | `128` to `2048`, step 64. The edge of a tile, in pixels. |
| `tile_format` *(optional)* | COMBO | `jpg` | The tile format: `jpg` or `webp`. |
| `tile_quality` *(optional)* | INT | `90` | `1` to `100`. The tile quality. |
| `cache_policy` *(optional)* | COMBO | `temp` | Where the cache is kept. Only `temp` exists. |

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | The input image, unchanged. |
| `manifest` | STRING | JSON text with one entry for each image of the batch, including its cache id. |

**How it works**

- The node can sit in the middle of a workflow, because it passes the image through.
- The tiles are lossy at the default quality, so the viewer is for inspection. Use [NH Save Large Image](./image-io.md#nh-save-large-image) to keep the pixels.
- Every run creates new cache entries. See [How the viewer works](#how-the-viewer-works).

**Example — inspect a tiled upscale**

```text
Image Untile (NH) → NH Large Image Preview → NH Save Large Image
```

Run the workflow, then click **Open Viewer** on the preview node and zoom to **100%**.

> **Tip:** `preview_megapixels` changes only the thumbnail on the node. The viewer always shows the image at full size.

---

## NH Large Image Compare

<p align="center"><img src="../assets/nodes/NH_LargeImageCompare.svg" alt="NH Large Image Compare node" width="360"></p>

Compares two large images in a zoomable viewer with a slider, a side-by-side view or a difference view. Only a small thumbnail goes to the canvas.

**Use it when** you want to check a large upscale or edit against the original at pixel level, where [Image Compare (NH)](./image-compose.md#image-compare-nh) would produce a picture too large for the canvas.

| Input | Type | Default | Description |
|---|---|---|---|
| `image_a` | IMAGE | — | The first image. |
| `image_b` | IMAGE | — | The second image. |
| `compare_mode` | COMBO | `slider` | The view the viewer opens in and the thumbnail shows: `slider`, `side_by_side` or `difference`. |
| `sync_pan_zoom` | BOOLEAN | `true` | Stored in the manifest. The viewer currently always pans and zooms both images together. |
| `preview_megapixels` | FLOAT | `1.0` | `0.05` to `8`, step 0.05. The largest size of the thumbnails, in megapixels. |
| `preview_format` | COMBO | `jpg` | `jpg`, `webp` or `png`. |
| `preview_quality` | INT | `90` | `1` to `100`. |
| `title` | STRING | `NH Large Image Compare` | The title in the viewer header. |
| `tile_size` *(optional)* | INT | `512` | `128` to `2048`, step 64. |
| `tile_format` *(optional)* | COMBO | `jpg` | `jpg` or `webp`. |
| `tile_quality` *(optional)* | INT | `90` | `1` to `100`. |
| `cache_policy` *(optional)* | COMBO | `temp` | Only `temp` exists. |

| Output | Type | Description |
|---|---|---|
| `image_a` | IMAGE | The first image. If one batch had a single image, it is repeated to match the other. |
| `image_b` | IMAGE | The second image, matched in the same way. |
| `manifest` | STRING | JSON text with one entry for each pair, including both cache ids. |

**How it works**

- **Batches.** If both batches have the same size, image *n* of A is paired with image *n* of B. If one has a single image, it is repeated for every image of the other. Otherwise both are cut to the shorter length, and a message appears in the console. Use ◀ and ▶ in the viewer to move through the pairs.
- **Different sizes.** In the viewer, each image is scaled to fit one common frame, so images of different sizes are compared in the same area.
- **Views.** In the viewer, the **Slider**, **Side** and **Diff** buttons switch the view at any time. In slider view, a bar in the header moves the dividing line between 5 % and 95 %.
- **Thumbnail.** The thumbnail on the node shows the chosen view. In `difference` it is brightened three times so small changes are easier to see. The viewer's difference view is not brightened.

**Example — check an upscale**

```text
Load Image → NH Large Image Compare.image_a
Image Untile (NH) → NH Large Image Compare.image_b
```

Click **Open Compare**, drag the slider bar, and zoom in on an edge.

> **Tip:** If the two images have different sizes, the comparison is still lined up by their frames, not by their pixels. Resize one of them first when you need a pixel-exact check.

---

## NH Large Preview Cache Manager

<p align="center"><img src="../assets/nodes/NH_LargePreviewCacheManager.svg" alt="NH Large Preview Cache Manager node" width="360"></p>

Shows what the large-preview cache holds on disk, and deletes old entries, entries over a size limit, or one entry by id.

**Use it when** the `temp` folder grows because of many previews, or you want to free the space without restarting ComfyUI.

| Input | Type | Default | Description |
|---|---|---|---|
| `action` | COMBO | `info` | What to do. See the table below. |
| `max_age_hours` | FLOAT | `24.0` | `0` to `8760`, step 0.5. Entries older than this are deleted by the actions that use age. `0` deletes every entry. |
| `max_total_mb` | FLOAT | `2048.0` | `1` to `1048576`, step 64. The size the cache may keep, in megabytes, for the actions that use size. |
| `cache_id` | STRING | empty | The 32-character id of one entry. Used by `delete_cache_id` only. You find the id in the `manifest` output of a preview node. |

| `action` | What it does |
|---|---|
| `info` | Deletes nothing. Reports the number of entries, their total size and their age. |
| `cleanup_by_age` | Deletes the entries older than `max_age_hours`. |
| `enforce_size_limit` | Deletes the oldest entries until the cache is no larger than `max_total_mb`. |
| `cleanup_age_and_size` | Does both: age first, then size. |
| `delete_cache_id` | Deletes the one entry named in `cache_id`. The id is required. |

| Output | Type | Description |
|---|---|---|
| `summary` | STRING | JSON text: the action, the state of the cache before and after, and what was removed. |
| `removed` | INT | How many entries were deleted. |
| `freed_mb` | FLOAT | How many megabytes were freed. |
| `cache_root` | STRING | The folder that holds the cache. |

**How it works**

- The node runs on every queue. It is never skipped because its settings did not change.
- An entry is the cache of one image from one run. Deleting an entry also breaks the viewer for that run: tiles stop loading, and **Open Viewer** shows nothing for it. Run the preview node again to rebuild it.
- It is an output node, so it runs even when nothing is connected to its outputs.

**Example — keep the cache small**

```text
NH Large Preview Cache Manager
```

Add the node on its own, set `action` to `cleanup_age_and_size`, `max_age_hours` to `24` and `max_total_mb` to `2048`. Each run removes entries older than a day and trims the rest to 2 GB.

> **Note:** Do not set `max_age_hours` to `0` in a workflow that also shows a preview. The entries made in the same run would be deleted too.

---

**See also:** [NH Save Large Image](./image-io.md#nh-save-large-image) to write the full-size file, and [Image Compare (NH)](./image-compose.md#image-compare-nh) for a comparison picture you can save.

[← Back to guide](./README.md)
