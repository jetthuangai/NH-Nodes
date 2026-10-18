[← Back to guide](./README.md)

# Batch & Lists

Nodes for working with groups of things. Pick or join images in a batch, count how many times you have queued a workflow, and build, filter and index lists of text. One more node splits any ComfyUI list into separate outputs. Find them under **NH-Nodes → Batch** in the node menu.

The most common mistake with these nodes is mixing counting from 0 with counting from 1, so read [Counting rules](#counting-rules) first.

**On this page**

| Node | Use it to |
|---|---|
| [Batch Index (NH)](#batch-index-nh) | Pick one image, or a range of images, from a batch |
| [Batch Merge (NH)](#batch-merge-nh) | Join up to three batches into one |
| [Counter (NH)](#counter-nh) | Get a number that goes up on every queue |
| [List Create (NH)](#list-create-nh) | Turn text into a list |
| [List Index (NH)](#list-index-nh) | Take one item from a list |
| [List Filter (NH)](#list-filter-nh) | Split a list into matches and non-matches |
| [Any List Split (NH)](#any-list-split-nh) | Split any ComfyUI list into separate outputs |

In the tables, `*` means any data type.

---

## Counting rules

Some nodes count from 0 and some from 1. Check this table when you wire a number from one node into another.

| Where | The first item is |
|---|---|
| `Batch Index (NH)` — `index`, `end_index` | `0` |
| `List Index (NH)` — `index` | `0` |
| `Any List Split (NH)` — outputs | `index 1` |
| `Counter (NH)` — `current` | your `start` value (default `0`) |
| Indexed loaders — `index` (see [Loaders](./loaders.md)) | `1` |

To drive a loader with a Counter, set the Counter's `start` to `1`.

---

## Batch Index (NH)

<p align="center"><img src="../assets/nodes/NH_BatchIndex.svg" alt="Batch Index (NH) node" width="360"></p>

Picks one image, or a range of images, out of an image batch and tells you how many images the batch had.

**Use it when** you want a single image from a generated batch (for example the third of four), the last frame of a sequence, or only part of a batch.

| Input | Type | Default | Description |
|---|---|---|---|
| `batch` | IMAGE | — | The image batch to pick from. |
| `index` | INT | `0` | Position of the first image to return, starting at 0. A negative value counts from the end (`-1` is the last image). A value past the end gives the last image. Range -100 to 1000. |
| `end_index` | INT | `-1` | Optional. `-1` returns only the image at `index`. Any other value returns a range (see below). Range -100 to 1000. |
| `step` | INT | `1` | Optional. Take every n-th image of the range. Used only with a range. Range 1 to 10. |

| Output | Type | Description |
|---|---|---|
| `result` | IMAGE | The selected image or images. |
| `original_count` | INT | How many images the input batch had. |

**How it works**

- **One image (default):** with `end_index` at `-1`, you get the image at `index`.
- **A range:** a positive `end_index` is exclusive, like a list slice. `index` 2 and `end_index` 5 return images 2, 3 and 4. A negative `end_index` counts from the end and includes that image: `-2` stops at the second-to-last image, so only the last one is left out. `-1` is reserved for the one-image mode.
- **Empty range:** if `end_index` is not after `index`, the node prints a warning in the console and returns the first image of the batch.
- Any `end_index` larger than the batch simply means "to the end".

Results for a batch of 10 images, numbered 0 to 9:

| `index` | `end_index` | `step` | Result |
|---|---|---|---|
| `3` | `-1` | — | image 3 |
| `-1` | `-1` | — | image 9 (the last one) |
| `2` | `5` | `1` | images 2, 3, 4 |
| `2` | `5` | `2` | images 2, 4 |
| `0` | `-2` | `1` | images 0 to 8 (all but the last) |
| `-3` | `1000` | `1` | images 7, 8, 9 (the last three) |
| `0` | `1000` | `3` | images 0, 3, 6, 9 |

**Example — keep the third image of a batch**

```text
KSampler (batch of 4) → VAE Decode → Batch Index (NH) → Save Image
```

Set `index` to `2`. The node returns the third image, and `original_count` reads `4`. Use `index` `-1` to always get the last image, however long the batch is.

> **Tip:** Wire a `Counter (NH)` into `index` to step through a batch one image per queue. Set the Counter's `max_value` to the batch size minus one.

---

## Batch Merge (NH)

<p align="center"><img src="../assets/nodes/NH_BatchMerge.svg" alt="Batch Merge (NH) node" width="360"></p>

Joins up to three image batches into one longer batch, resizing the extra batches to match the first.

**Use it when** you want images from separate branches in one batch (to preview, save or send them on together), or when the batches you are combining have different sizes.

| Input | Type | Default | Description |
|---|---|---|---|
| `batch_a` | IMAGE | — | The first batch. Its width, height and channels (RGB or RGBA) set the format of the result. It is never resized. |
| `batch_b` | IMAGE | — | Optional. A second batch, added after `batch_a`. |
| `batch_c` | IMAGE | — | Optional. A third batch, added last. |
| `resize_mode` | COMBO | `stretch` | Optional. How images from `batch_b` and `batch_c` are fitted to the size of `batch_a`: `stretch`, `crop` or `pad`. Images that already have the right size are left alone. |

| Output | Type | Description |
|---|---|---|
| `result` | IMAGE | All images in order: `batch_a`, then `batch_b`, then `batch_c`. |
| `count` | INT | The number of images in `result`. |

**Resize modes**

| Mode | What it does | Proportions | Cost |
|---|---|---|---|
| `stretch` | Scales the image to exactly the target size. | Not kept — the image can look squashed or stretched. | Nothing is lost. |
| `crop` | Scales the image until the target is fully covered, then cuts the overflow evenly from the centre. | Kept. | The edges are cut off. |
| `pad` | Scales the image until all of it fits, centres it and fills the rest with bars. | Kept. | Bars are added: black in an RGB result, transparent in an RGBA result. |

**RGB and RGBA**

The result has the channels of `batch_a`. Every other batch is converted to match before it is joined.

| `batch_a` | Other batch | Result |
|---|---|---|
| RGB | RGB | RGB |
| RGBA | RGBA | RGBA |
| RGB | RGBA | RGB — the other batch loses its alpha channel. |
| RGBA | RGB | RGBA — the other batch gets a fully opaque alpha channel. With `pad`, the bars are transparent. |

Batches with any other channel count (for example a single channel) cannot be mixed with a different channel count. The node stops with an error that names both counts.

**Example — combine two images of different sizes**

```text
Load Image A + Load Image B → Batch Merge (NH) → Preview Image
```

Set `resize_mode` to `pad`. Image B keeps its proportions and sits in the centre of a frame the size of image A. `count` reads `2`.

> **Tip:** Connect the image whose size you want to keep to `batch_a`. To merge more than three batches, chain two nodes: send the `result` of the first into `batch_a` of the second.

---

## Counter (NH)

<p align="center"><img src="../assets/nodes/NH_Counter.svg" alt="Counter (NH) node" width="360"></p>

A counter that goes up by `step` every time you queue the workflow and tells you when it has passed `max_value`.

**Use it when** you want to go through a set of things one per queue: test several LoRAs, rotate through prompts, or process a batch image by image.

| Input | Type | Default | Description |
|---|---|---|---|
| `start` | INT | `0` | The first value of the count. |
| `step` | INT | `1` | How much the count goes up on each queue. Minimum 1. |
| `max_value` | INT | `9` | The last value to count to. Minimum 0. |
| `reset` | BOOLEAN | `false` | While on, every queue restarts the count at `start`. |

| Output | Type | Description |
|---|---|---|
| `current` | INT | The value for this queue. Never higher than `max_value`. |
| `is_done` | BOOLEAN | `true` once the count has passed `max_value`. |
| `progress` | FLOAT | `current` divided by `max_value`, from 0.0 to 1.0. |
| `remaining` | INT | Values left up to `max_value`, counting `current` itself (`max_value - current + 1`). Never below 0. |

**How it works**

- **It runs on every queue.** The node is never served from ComfyUI's cache, so each queue moves the count on: every click on Run, every repeat of the batch count, every auto-queue pass.
- **Counting:** the first queue returns `start`, the next one `start + step`, and so on. When the next value would pass `max_value`, you get `max_value` with `is_done` on, and the count stops. Extra queues repeat that result. With `step` 2 and `max_value` 9 the sequence is 0, 2, 4, 6, 8, then 9 with `is_done` on.
- **`reset`:** switch it on and queue once to rewind. That queue returns `start`. Switch `reset` off again and the next queue returns `start + step`. While `reset` stays on, every queue returns `start`.
- **Changing settings mid-count:** `step` and `max_value` apply at once. `start` is only read when counting begins and after a reset.
- **Where the count lives:** in memory, per node ID. Restarting ComfyUI starts over, and two Counter nodes count independently. Workflows that contain a Counter with the same node ID share one count, so use `reset` after switching between them.

With the default settings (`start` 0, `step` 1, `max_value` 9):

| Queue | 1 | 2 | … | 10 | 11 | 12 |
|---|---|---|---|---|---|---|
| `current` | 0 | 1 | … | 9 | 9 | 9 |
| `is_done` | false | false | … | false | true | true |
| `remaining` | 10 | 9 | … | 1 | 0 | 0 |
| `progress` | 0.0 | 0.11 | … | 1.0 | 1.0 | 1.0 |

**Example — test five LoRAs, one per queue**

```text
Counter (NH) → Load LoRA Model Index (NH) → KSampler
```

| Node | Settings |
|---|---|
| `Counter (NH)` | `start` 1, `step` 1, `max_value` 5. |
| `Load LoRA Model Index (NH)` | Wire `current` into `index`. Set `lora_count` to 5 and choose a LoRA in `lora_1` to `lora_5`. |

Set the batch count to 5 and press Run. Each queue uses the next LoRA, and the `filename` output tells you which one. The Counter starts at 1 because the loaders count from 1. See [Loaders](./loaders.md) for the full node.

> **Tip:** `is_done` is a plain BOOLEAN. Connect it to `If/Else (NH)` or `Logic Gate (NH)` (see [Logic](./logic.md)) to react when the sequence has finished. Turn `reset` on for one queue to run the sequence again.

---

## List Create (NH)

<p align="center"><img src="../assets/nodes/NH_ListCreate.svg" alt="List Create (NH) node" width="360"></p>

Turns text into a list of items, one per line or split on any delimiter you choose.

**Use it when** you keep a set of prompts, style names or keywords in one text box and want to pick, filter or loop through them with the List nodes.

| Input | Type | Default | Description |
|---|---|---|---|
| `text` | STRING | three lines: `item 1`, `item 2`, `item 3` | The text to split. Multiline text box. |
| `delimiter` | STRING | `\n` | Optional. What separates the items. `\n` is a new line and `\t` is a tab. Anything else is used exactly as typed, for example `,` or `;`. |

| Output | Type | Description |
|---|---|---|
| `items` | NH_LIST | The list. Only `List Index (NH)` and `List Filter (NH)` accept it. |
| `count` | INT | How many items the list has. |
| `first` | STRING | The first item. Empty when the list is empty. |
| `last` | STRING | The last item. Empty when the list is empty. |

**How it works**

- Every item is trimmed (spaces at both ends removed), and empty items are dropped, so blank lines never create items.
- The delimiter is plain text, not a pattern. It can be longer than one character, such as `||`.
- If nothing is left after trimming, `items` is an empty list, `count` is 0, and `first` and `last` are empty.
- Keep something in `delimiter`. An empty field makes the node stop with an "empty separator" error.

**Example — split one line into three items**

Set `text` to `red, green, blue` and `delimiter` to `,`. The list holds `red`, `green` and `blue`; `count` is `3`, `first` is `red` and `last` is `blue`.

> **Note:** `NH_LIST` is the pack's own list type. It travels as a single value, so it does not make ComfyUI run the next node once per item. To get one run per item, use `Text Split Lines (NH)` from [Text](./text.md) together with [Any List Split (NH)](#any-list-split-nh).

---

## List Index (NH)

<p align="center"><img src="../assets/nodes/NH_ListIndex.svg" alt="List Index (NH) node" width="360"></p>

Takes one item from a list by its position and tells you whether it is the first or the last one.

**Use it when** you want to pick a prompt or style by number, or step through a list one item per queue with a `Counter (NH)`.

| Input | Type | Default | Description |
|---|---|---|---|
| `items` | NH_LIST | — | The list, from `List Create (NH)` or `List Filter (NH)`. |
| `index` | INT | `0` | Position of the item, starting at 0. Range -100 to 1000. |
| `wrap` | BOOLEAN | `true` | Optional. What happens when `index` is outside the list (see below). |

| Output | Type | Description |
|---|---|---|
| `item` | STRING | The chosen item. |
| `is_first` | BOOLEAN | `true` if the item is the first of the list. |
| `is_last` | BOOLEAN | `true` if the item is the last of the list. |
| `count` | INT | How many items the list has. |

**How it works**

- **`wrap` on:** the position loops around the list, so a Counter can keep running and the list simply repeats. Negative values count back from the end.
- **`wrap` off:** a position past the end gives the last item. A negative position counts from the end and stops at the first item.
- **Empty list:** `item` is empty, `is_first` and `is_last` are both `true`, `count` is `0`, and the console shows a warning.

For a list of 4 items (0 to 3):

| `index` | `wrap` on | `wrap` off |
|---|---|---|
| `0` | item 0 | item 0 |
| `3` | item 3 | item 3 |
| `4` | item 0 | item 3 |
| `5` | item 1 | item 3 |
| `-1` | item 3 | item 3 |
| `-5` | item 3 | item 0 |

**Example — rotate through prompts**

```text
List Create (NH) → List Index (NH) → CLIP Text Encode
```

Write one prompt per line in `List Create (NH)`. Wire a `Counter (NH)` `current` into `index`. With `start` 0 and `max_value` set to `count` minus one, each queue uses the next prompt. Leave `wrap` on if you want the list to repeat instead of stopping.

---

## List Filter (NH)

<p align="center"><img src="../assets/nodes/NH_ListFilter.svg" alt="List Filter (NH) node" width="360"></p>

Splits a list in two: the items that match a condition and the items that do not.

**Use it when** you want to keep only the prompts that mention a word, drop items you do not need, or keep only short or long items.

| Input | Type | Default | Description |
|---|---|---|---|
| `items` | NH_LIST | — | The list to filter. |
| `condition` | STRING | `contains:dress` | The test to apply to every item (see below). |
| `mode` | COMBO | `include` | `include`: `passed` holds the items that match. `exclude`: `passed` holds the items that do not match. |

| Output | Type | Description |
|---|---|---|
| `passed` | NH_LIST | The items that passed. |
| `rejected` | NH_LIST | The items that did not pass. |
| `passed_count` | INT | How many items passed. |
| `rejected_count` | INT | How many items did not pass. |

**Conditions**

| Write | An item matches when… | Case |
|---|---|---|
| `contains:text` | it contains `text` | ignored |
| `startswith:text` | it starts with `text` | respected |
| `endswith:text` | it ends with `text` | respected |
| `equals:text` | it is exactly `text` | ignored |
| `regex:pattern` | the regular expression finds a match anywhere in it | respected (start the pattern with `(?i)` to ignore case) |
| `len>5`, `len>=5`, `len<5`, `len<=5`, `len==5`, `len!=5` | its length in characters passes the test | — |
| anything else, for example `dress` | it contains the whole condition text | ignored |

**How it works**

- The keyword before the colon is not case sensitive (`Contains:hat` works), but the word `len` must be written in lowercase. Spaces around the value and around the `len` operator are ignored.
- A condition with an unknown keyword, such as `foo:bar`, is treated as plain text and searched for as a whole.
- A broken regular expression matches nothing and prints a message in the console.
- There is one condition per node. To require two conditions, chain two nodes: send `passed` of the first into `items` of the second. To accept either of two words, use a single regular expression such as `regex:red|blue`.

**Example — keep only the prompts that mention a dress**

```text
List Create (NH) → List Filter (NH) → List Index (NH)
```

For the items `Red Dress`, `blue jeans`, `dress shirt` and `Hat`, the condition `contains:dress` with `include` gives `passed` = `Red Dress`, `dress shirt` (`passed_count` 2) and `rejected` = `blue jeans`, `Hat` (`rejected_count` 2). With `exclude` the two lists swap.

---

## Any List Split (NH)

<p align="center"><img src="../assets/nodes/NH_AnyListSplit.svg" alt="Any List Split (NH) node" width="360"></p>

Splits any ComfyUI list into a full-list output (`All`) plus up to ten separate outputs, one per item.

**Use it when** a node gives you a list (for example the `lines` of `Text Split Lines (NH)`, or a batch of images) and you want to send the first, second and third item to different places.

| Input | Type | Default | Description |
|---|---|---|---|
| `items` | `*` | — | The list or batch to split. Any type is accepted. |
| `index_count` | INT | `4` | How many `index` outputs the node shows and uses. Range 0 to 10. `All` is always there. |

| Output | Type | Description |
|---|---|---|
| `All` | `*` (list) | Every item as a ComfyUI list (the `≡` mark on the card). Nodes connected here run once per item. |
| `index 1` … `index 10` | `*` | One item each. `index 1` is the first item. Only the first `index_count` outputs are shown. |

**How it works**

- **Outputs follow `index_count`.** Change it, or press the `update outputs` button, and the node adds or removes outputs to match. Lowering it disconnects the links of the outputs that disappear. The card above shows all ten outputs; on the canvas you see `index_count` of them.
- **Image batches are split into single images.** A batch of 3 images followed by a batch of 2 gives 5 items. Other values are kept as they are, and empty entries are skipped.
- **Blocked outputs.** An `index N` output with no matching item is blocked: the list is shorter than N, or N is above `index_count`. The nodes after a blocked output are skipped silently, with no error. If the incoming list is empty, `All` is blocked too.

**Example — send two lines of one text box to two places**

```text
Text Split Lines (NH) → Any List Split (NH)
```

Write the positive prompt on line 1 and the negative prompt on line 2 of `Text Split Lines (NH)`. Connect `lines` to `items`, set `index_count` to `2`, then connect `index 1` to the positive prompt and `index 2` to the negative prompt.

> **Tip:** Use `All` when the next node should run once per item, and the `index` outputs when you need a specific item.

---

**See also:** [Loaders](./loaders.md) for the nodes a Counter can drive, [Text](./text.md) for `Text Split Lines (NH)`, and [Logic](./logic.md) for switches and branches.

[← Back to guide](./README.md)
