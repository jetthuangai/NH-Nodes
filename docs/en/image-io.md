[← Back to guide](./README.md)

# Image — Load and Save

These nodes move images in and out of a workflow. Load one image together with its file name and path, step through a folder one image per run, pull the images of a matching folder by name, and save results to the folder and file name you choose. One node saves very large images without sending a heavy preview to the canvas. Find them under **NH-Nodes → Image** in the node menu.

**On this page**

| Node | Use it to |
|---|---|
| [Load Image Info (NH)](#load-image-info-nh) | Load one image and get its file name and path as text |
| [Load Images Folder (NH)](#load-images-folder-nh) | Load images from a folder by position, and move to the next one on every run |
| [Load images matching](#load-images-matching) | Find the folder that matches a subject name and load its images |
| [Save Image Path (NH)](#save-image-path-nh) | Save to any folder with your own prefix, numbering, format and DPI |
| [NH Save Large Image](#nh-save-large-image) | Save very large images without a heavy preview on the canvas |

## Paths and file types

| Where | A relative path starts in |
|---|---|
| `folder_path` of Load Images Folder, `matching_root_path` and `source_folder_path` of Load images matching | the ComfyUI `input` folder |
| `folder_path` of Save Image Path | the ComfyUI `output` folder |
| NH Save Large Image | always the ComfyUI `output` folder |

The folder loaders read `.png`, `.jpg`, `.jpeg`, `.webp`, `.avif`, `.bmp`, `.tif` and `.tiff` files. They sort names in natural order, so `img2` comes before `img10`, and they return RGB images.

---

## Load Image Info (NH)

<p align="center"><img src="../assets/nodes/NH_LoadImageInfo.svg" alt="Load Image Info (NH) node" width="360"></p>

Loads one image from the ComfyUI `input` folder, like the built-in Load Image, and also returns its file name and path as text.

**Use it when** you want to name or route your results after the source file, or record which file a run used.

| Input | Type | Default | Description |
|---|---|---|---|
| `image` | COMBO | — | The image to load. Lists the image files in the `input` folder. Use the upload button to add one. |
| `filename_extension` | BOOLEAN | `true` | `true`: `file_name` keeps its extension. `false`: the extension is removed. |
| `path_include_filename` | BOOLEAN | `true` | `true`: `path` is the full path of the file. `false`: `path` is only the folder that holds it. |

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | The image in RGB. The frames of an animated file (GIF, WebP) become a batch. |
| `mask` | MASK | Built from the transparency of the file: opaque pixels are `0`, transparent pixels are `1`, as in Load Image. All `0` when the file has no transparency. |
| `file_name` | STRING | The name of the file. |
| `path` | STRING | The full path, or only the folder. See `path_include_filename`. |

**How it works**

- Loading follows ComfyUI's Load Image. The EXIF rotation is applied. In an animation, frames whose size differs from the first frame are skipped. For MPO files only the first frame is used.
- Transparency is not kept in `image`. It goes to `mask`.
- The node runs again when the content of the file changes, so you can edit and save the image in place.

**Example — name the result after the source file**

```text
Load Image Info (NH).image → … your workflow … → Save Image Path (NH).images
Load Image Info (NH).file_name → Save Image Path (NH).filename_prefix
```

Set `filename_extension` to `false` so the prefix does not end in `.png`.

> **Tip:** Set `path_include_filename` to `false` and connect `path` to `folder_path` of Save Image Path to save the result in the folder of the source image.

---

## Load Images Folder (NH)

<p align="center"><img src="../assets/nodes/NH_LoadImagesFromFolder.svg" alt="Load Images Folder (NH) node" width="360"></p>

Loads one or more images from a folder by position, and can move to the next image after every run.

**Use it when** you want to run a workflow over every image of a folder, one image per run (for example to upscale a whole folder), or to load a few consecutive images as a batch.

| Input | Type | Default | Description |
|---|---|---|---|
| `folder_path` | STRING | empty | The folder to read. An absolute path, or a path relative to the `input` folder. Empty means the `input` folder. A path to a file uses the folder of that file. |
| `index` | INT | `0` | Position of the first image to load, from `0` to `999999`. It wraps around: with 10 images, `12` loads image `2`. |
| `step` | INT | `1` | How many images to load per run (`1` to `9999`), one after the other from `index`. It is also the distance `index` moves between runs in `increment` and `decrement` modes. |
| `seed_mode` | COMBO | `fixed` | What happens to `index` each time you queue: `fixed` (stays), `increment` (adds `step`), `decrement` (subtracts `step`; the images are also read backwards), `random` (a new random position). |
| `recursive` | BOOLEAN | `false` | Also read the images in sub-folders. |
| `filename_extension` | BOOLEAN | `true` | Keep the extension in `filename`. |
| `path_output` | COMBO | `file_path` | What `path` lists: `file_path` (the full path of each file) or `folder_path` (the folder of each file). |

| Output | Type | Description |
|---|---|---|
| `images` | IMAGE | The loaded images as one batch. If their sizes differ, every image is resized to the size of the first one. |
| `filename` | STRING | The file names, one per line, in loading order. |
| `path` | STRING | The paths, one per line. |

**How it works**

- If the folder holds no images of its own, its sub-folders are searched even when `recursive` is off. A search of sub-folders skips the ComfyUI `models` folder.
- The browser moves `index` when you queue the workflow from the ComfyUI interface. It counts the images with the node's own `recursive` setting, so `increment`, `decrement` and `random` wrap and pick over the same images that the node loads. If you send the workflow through the API, set `index` yourself.
- The node runs again when one of the selected files changes on disk (name, size or date). If `folder_path` is wired from another node, the node cannot check the files in advance: it then runs again when the wired path or another setting changes, but not when a file inside the same folder changes.
- A folder that does not exist, or has no images, stops the run with an error.

**Example — upscale a whole folder**

```text
Load Images Folder (NH) → Upscale Image (using Model) → Save Image Path (NH)
```

Set `seed_mode` to `increment` and `step` to `1`. Each queue loads the next file. When `index` passes the last image, it starts again from the first.

> **Tip:** `step` `4` with `fixed` loads the same four images on every run. With `increment`, each run moves four images forward.

---

## Load images matching

<p align="center"><img src="../assets/nodes/NH_LoadImagesMatching.svg" alt="Load images matching node" width="360"></p>

Finds the folder under a root whose name matches a subject, and loads its images. The subject comes from your source folder, or from text you type.

**Use it when** your matching images are grouped in folders by subject, and you want the folder that belongs to the current source picked for you. This works well for libraries split into groups, such as `Nam` and `Nu` (Vietnamese for male and female), with one folder per person inside each group:

```text
matching_root_path
├── Nam
│   ├── subject_01
│   └── subject_02
└── Nu
    ├── subject_01     ← loaded when the source path ends in …/Nu/subject_01
    └── subject_02
```

| Input | Type | Default | Description |
|---|---|---|---|
| `source_folder_path` | STRING | empty | The source folder of the subject. Only the folder **names** in the path are used, so the folder does not have to exist. The last name is the match text, unless you fill in `match_text`. A context folder in the path (see `context_folder_names`) tells the node which branch to prefer. |
| `matching_root_path` | STRING | empty | The root of the folder tree to search. An absolute path, or a path relative to the `input` folder. Required. |
| `match_text` | STRING | empty | The folder name to look for. Empty means: use the name of the source folder. |
| `match_mode` | COMBO | `contains` | `contains`: the folder name contains the text. `exact`: the folder name equals the text. Both ignore case and accents, and treat `_` and `-` as spaces. |
| `recursive` | BOOLEAN | `true` | `true`: search every folder below the root. `false`: check only the root and its direct sub-folders. |
| `index` | INT | `0` | Position of the first image in the matched folder, from `0` to `999999`. It wraps around. |
| `step` | INT | `1` | How many images to load from the matched folder (`1` to `9999`), one after the other from `index`. |
| `repeat_count` | INT | `1` | Repeats the loaded images this many times in the output batch (`1` to `9999`). The text outputs are not repeated. |
| `context_folder_names` | STRING | `Nam` and `Nu`, one per line | The names of the context folders. Separate names with new lines, commas, semicolons or vertical bars. Replace the defaults with the group names of your own library. |
| `source_file_names` *(optional)* | STRING | empty | One source file path per line. Each line is one match item: the match text is the name of the file's folder (unless `match_text` is set), and its path supplies the context. Use it to load the matches of several subjects at once. |
| `fallback_images` *(optional)* | IMAGE | — | The images to use when nothing matches. They are only evaluated on a miss, so the nodes that feed this input do not run when a folder matches. See "When nothing is found". |

| Output | Type | Description |
|---|---|---|
| `images` | IMAGE | The loaded images as one batch. If their sizes differ, every image is resized to the size of the first one. When nothing is found it carries `fallback_images` if that input is connected, and is blocked otherwise (see "When nothing is found"). |
| `filename` | STRING | The file names, one per line, one line per loaded image. |
| `path` | STRING | The matched folder of each loaded image, one per line. |
| `matched_text` | STRING | The text that was searched, one line per loaded image. |
| `found` | BOOLEAN | `true` when every item found a folder with images. Otherwise `false`, also when `fallback_images` supplies the images. |

**How it works**

- **Match text:** `match_text`, or else the name of the folder that holds each `source_file_names` line, or else the name of the source folder. Fill in at least one of `source_folder_path`, `source_file_names` and `match_text`. With none of them, the node looks for the name of ComfyUI's `input` folder, which rarely matches.
- **Context:** if the source path contains one of the `context_folder_names`, matches under the same context rank first. A source in `…/Nu/subject_01` picks `Nu/subject_01` before `Nam/subject_01`. If your root sits inside one context folder (`…/Nam`) and the source belongs to another (`Nu`), the same place under the other context folder is searched too.
- **Ties:** among equal matches, the folder whose path is most like the source path wins, then natural name order.
- **Limits:** the search stays inside the root. It does not scan parent folders, apart from the sibling context folders described above, and it never enters the ComfyUI `models` folder. With `recursive` on, a root that is a drive or a network share (such as `D:\` or `\\server\share`) is refused: it cannot be searched recursively. The `models` folder is refused as a root whatever `recursive` says.
- **Files:** the images of the matched folder are sorted in natural order. `index` and `step` choose which ones to load.
- **Caching:** the result is reused until the matched folder or one of the selected files changes (name, size or date). A search that finds nothing is never remembered, so a folder you add later is found on the next run. If a setting is wired from another node, the node cannot know the folder in advance. It then watches the whole tree below the root, and in a very large tree it runs on every queue.

**When nothing is found**

> **Note:** if no folder matches, or the matched folder has no images, the run does not crash. `found` is `false`, `filename` and `path` are empty, `matched_text` holds the text that was searched, and the reason is printed in the console. With several items, one miss makes the whole request "not found". A setting that cannot work still stops the run with an error: an empty or missing `matching_root_path`, a root that is a drive or a network share (with `recursive` on), or the `models` folder.

What `images` does on a miss depends on `fallback_images`:

- **Connected:** `images` outputs the fallback images, and the nodes that feed `fallback_images` run only now. `found` stays `false`, so you can still tell that nothing matched.
- **Not connected:** `images` is blocked. Nodes that need it are skipped, and ComfyUI shows the reason ("Execution Blocked"). The rest of the workflow keeps running.

> **Tip:** To carry on with another image when nothing matches, connect it to `fallback_images`. Do not route `images` through an If/Else or a switch: a blocked input blocks that node too, so it never gets to choose.

**Example — load the matching images for a source folder**

```text
Load Images Folder (NH).path → Load images matching.source_folder_path
Load images matching.images → … your workflow …
```

On Load Images Folder, set `path_output` to `folder_path`, so `path` is the folder of the source image. On Load images matching, set `matching_root_path` to the root of your library. Connect a default picture to `fallback_images`, or use `found` to decide what happens when no matching folder exists.

> **Tip:** Keep `matching_root_path` as close to the images as you can. With `recursive` on, every folder below the root is scanned.

---

## Save Image Path (NH)

<p align="center"><img src="../assets/nodes/NH_SaveImagePath.svg" alt="Save Image Path (NH) node" width="360"></p>

Saves an image batch to a folder you choose, with your own file name prefix and automatic numbering.

**Use it when** the results belong in a project folder outside the default `output` folder, or you need a format or a DPI value that the built-in Save Image does not offer.

| Input | Type | Default | Description |
|---|---|---|---|
| `images` | IMAGE | — | The images to save. Every image of the batch becomes a file. |
| `folder_path` | STRING | empty | Where to save. Empty means the `output` folder. A relative path is placed inside `output`. An absolute path is used as it is, even outside ComfyUI. Missing folders are created. A path to an existing file uses the folder of that file. |
| `create_new_folder` | BOOLEAN | `false` | Save into a sub-folder of `folder_path`. The `new_folder_name` field only shows while this is on. |
| `new_folder_name` | STRING | empty | The sub-folder, such as `day1` or `batch/day1`. It must stay inside `folder_path`. |
| `filename_prefix` | STRING | `NH` | The start of every file name. An empty value becomes `NH`. A `/` puts the files in a sub-folder, as in `renders/img`. |
| `number_suffix` | BOOLEAN | `true` | `true`: add a four-digit number. `false`: use the bare prefix first (see the table below). |
| `preview_image` | BOOLEAN | `false` | Also show the saved images on the node. |
| `file_format` | COMBO | `png` | `png`, `jpg`, `webp` or `avif`. |
| `dpi` | INT | `300` | `1` to `2400`. Print-size metadata that PNG and JPG files keep. WEBP and AVIF files do not store it. The pixels are not changed. |

| Output | Type | Description |
|---|---|---|
| `filenames` | STRING | The saved file names, one per line, including a sub-folder from `filename_prefix`. |
| `path` | STRING | The folder the files were saved to (`folder_path` plus the new folder). Join it with `filenames` for the full path. |
| `saved_count` | INT | How many files were written. |

**Where the files go**

| `folder_path` | New folder | `filename_prefix` | First file |
|---|---|---|---|
| empty | off | `NH` | `output/NH0001.png` |
| `renders` | off | `shot` | `output/renders/shot0001.png` |
| `D:\Projects\ClientA` | on, `day1` | `portrait` | `D:\Projects\ClientA\day1\portrait0001.png` |
| `D:\Projects\ClientA` | off | `final/portrait` | `D:\Projects\ClientA\final\portrait0001.png` |

**How numbering works**

The node never overwrites a file. It looks at the files that already use the prefix in the target folder (any of `png`, `jpg`, `webp`, `avif`) and takes the next free number.

| `number_suffix` | Two images, prefix `shot` | Next run |
|---|---|---|
| `true` | `shot0001.png`, `shot0002.png` | `shot0003.png`, `shot0004.png` |
| `false` | `shot.png`, `shot1.png` | `shot2.png`, `shot3.png` |

**Rules for names**

- `filename_prefix` and `new_folder_name` must stay inside the target folder. A `..` part or an absolute path stops the run with an error before anything is written. An absolute `folder_path` is allowed.
- On Windows, `filename_prefix` cannot contain `:`. It would select a drive instead of naming a file.

**How it works**

- Quality is fixed: `png` is lossless, `jpg` and `webp` use quality 95, `avif` uses quality 90. `avif` needs a Pillow build with AVIF support.
- Files are saved in RGB, so transparency is dropped. No ComfyUI workflow is embedded in the file.
- `dpi` is stored in `png` and `jpg` files. `webp` and `avif` files accept the value but do not keep it.
- This is an output node: it runs even when nothing is connected to its outputs.

> **Tip:** To drag a saved image back into ComfyUI and restore its workflow, use the built-in Save Image, or [NH Save Large Image](#nh-save-large-image) with a PNG.

---

## NH Save Large Image

<p align="center"><img src="../assets/nodes/NH_SaveLargeImage.svg" alt="NH Save Large Image node" width="360"></p>

Saves full-resolution images to the ComfyUI `output` folder and returns their paths. It sends no heavy preview of the saved image to the canvas.

**Use it when** the image is very large (tens of megapixels) and the browser struggles to show it, or you want TIFF output or a JPG or WEBP quality value.

| Input | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | The images to save. Every image of the batch is saved. |
| `filename_prefix` | STRING | `NH_Large` | Works like the prefix of the built-in Save Image. It can include a sub-folder (`project/shot`) and the tokens `%width%`, `%height%`, `%year%`, `%month%`, `%day%`, `%hour%`, `%minute%`, `%second%`. It cannot leave the `output` folder. |
| `file_format` | COMBO | `png` | `png`, `jpg`, `webp` or `tiff`. |
| `quality` | INT | `95` | `1` to `100`. Used by `jpg` and `webp`. Ignored by `png` and `tiff`, which are lossless. |
| `embed_metadata` | BOOLEAN | `true` | PNG only: store the prompt and workflow in the file, so you can drag it back into ComfyUI. It is always off when ComfyUI runs with `--disable-metadata`. |

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | The input images, unchanged. |
| `file_path` | STRING | The full path of each saved file, one per line. |
| `saved_count` | INT | How many files were saved. |

**How it works**

- File names look like `NH_Large_00001_.png`: the prefix, a five-digit counter, then the extension. The counter continues from the highest number already used for the prefix, whatever the format.
- **Size limits:** WEBP allows up to 16383 px per side and JPEG up to 65500 px. The node checks this before it writes anything, and the error suggests PNG or TIFF. PNG and TIFF have no practical limit.
- **Transparency:** PNG and TIFF keep it. JPG and WEBP are saved in RGB. A single-channel (grey) image is accepted and saved as RGB.
- TIFF files use LZW compression.
- The node draws nothing on the canvas after saving. Read `file_path`, or open the `output` folder.

**Example — save an untiled result as TIFF**

```text
Image Untile (NH) → NH Save Large Image
```

Set `file_format` to `tiff` for a lossless file you can open in an editor. To look at the image first, add [NH Large Image Preview](./image-preview.md#nh-large-image-preview).

---

**See also:** [Image — Tile, Compose and Resize](./image-compose.md) for the tile nodes, and [Image — Large Image Preview](./image-preview.md) for the viewer.

[← Back to guide](./README.md)
