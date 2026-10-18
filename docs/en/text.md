[← Back to guide](./README.md)

# Text

Text nodes build and reshape the strings that drive your prompts and file names. Fill prompt templates, join and split text, run string operations and regular expressions, and pick lines by position or at random. Most nodes return STRING. Some also return counts or true/false flags that you can feed into the [Logic](./logic.md) nodes. Find the nodes under **NH-Nodes → Text** in the node menu.

**On this page**

- [Prompt Template (NH)](#prompt-template-nh) — fill `{placeholders}` in a prompt and drop empty ones cleanly
- [Prompt Scheduler (NH)](#prompt-scheduler-nh) — move through a list of prompts as a step number grows
- [String Operations (NH)](#string-operations-nh) — change case, trim, replace, test or slice text
- [Prompt Join (NH)](#prompt-join-nh) — join up to five fragments with a separator
- [Text Split (NH)](#text-split-nh) — split text at a delimiter and get the count, first and last item
- [Regex Extract (NH)](#regex-extract-nh) — find, replace or split text with a regular expression
- [Text Index (NH)](#text-index-nh) — pick one of up to 64 typed texts by number
- [Text Concatenate (NH)](#text-concatenate-nh) — join up to 64 strings, for example into a path
- [Text Split Lines (NH)](#text-split-lines-nh) — turn multiline text into a list that runs once per line
- [Text Random Line (NH)](#text-random-line-nh) — pick a random line of text

**Reading the examples**

- `\n` in an example marks a line break, and `—` marks an empty or unconnected value.
- In the delimiter fields of Text Split, Text Concatenate and Text Split Lines, typing `\n` or `\t` means a line break or a tab. Prompt Join uses its separator exactly as typed.
- Random picks use the seed as given, 0 included. The same seed and the same text always give the same pick.

---

## Prompt Template (NH)

<p align="center"><img src="../assets/nodes/NH_PromptTemplate.svg" alt="Prompt Template (NH) node" width="360"></p>

Fills `{name}` placeholders in a prompt with up to six text values. Empty values are removed cleanly, without stray commas or extra spaces.

**Use it when** one base prompt serves many variants, such as colour, garment and style, and some fields may stay empty.

| Input | Type | Default | Description |
|---|---|---|---|
| `template` | STRING | `a {color} {garment}, {style}, 8k photo` | Multiline prompt with `{name}` placeholders. |
| `var_names` | STRING | `color,garment,style` | Comma-separated placeholder names. The Nth name takes the value of `varN`. |
| `var1` … `var6` *(optional)* | STRING | `""` | The values. Connection only: there is no text box, so connect a text node. |

| Output | Type | Description |
|---|---|---|
| `result` | STRING | The finished prompt. |
| `missing_vars` | STRING | Names of placeholders that were empty, joined with `, `. |

**How it works**

- A placeholder is `{name}` with the exact name from `var_names` (case-sensitive; spaces around names are ignored). `var1` fills the first name, `var2` the second, and so on. Only six variables exist, so a seventh name always counts as empty.
- Text in braces that is not listed in `var_names` stays as typed and is not reported.
- An empty or unconnected variable removes its placeholder together with the space before it (and the space after it when the placeholder starts a line). The name is added to `missing_vars`.
- Every line of the result is then tidied: `, ,` becomes `,`, and a comma at the start or end of a line is dropped. This also tidies commas that were already in the template.

With the default template and names:

| `var1` | `var2` | `var3` | `result` | `missing_vars` |
|---|---|---|---|---|
| `red` | `dress` | `studio` | `a red dress, studio, 8k photo` | — |
| `red` | `dress` | — | `a red dress, 8k photo` | `style` |
| — | `dress` | — | `a dress, 8k photo` | `color, style` |
| — | — | `studio` | `a, studio, 8k photo` | `color, garment` |

With one empty variable at different positions:

| `template` | `var_names` | `var1` | `result` |
|---|---|---|---|
| `red dress, {style}, 8k photo` | `style` | — | `red dress, 8k photo` |
| `{style}, red dress` | `style` | — | `red dress` |
| `red dress, {style}` | `style` | — | `red dress` |
| `a {color} dress` | `color` | — | `a dress` |

**Example — a prompt with optional parts**

```text
Text "red"                  → Prompt Template (NH).var1
Text "dress"                → Prompt Template (NH).var2
Prompt Template (NH).result → CLIP Text Encode.text
```

---

## Prompt Scheduler (NH)

<p align="center"><img src="../assets/nodes/NH_PromptScheduler.svg" alt="Prompt Scheduler (NH) node" width="360"></p>

Picks one line from a list of prompts according to a step number you provide, and reports the progress.

**Use it when** a looped workflow, for example one driven by `Counter (NH)`, should walk through prompts: the first prompt for the first part of the sequence, then the next.

| Input | Type | Default | Description |
|---|---|---|---|
| `prompts` | STRING | Three lines: `prompt 1`, `prompt 2`, `prompt 3` | Multiline, one prompt per line. Blank lines are ignored and each line is trimmed. |
| `current_step` | INT | `0` | Position in the schedule, 0 or more. Usually connected to a counter. |
| `total_steps` | INT | `10` | Length of the schedule, 1 or more. |
| `mode` | COMBO | `sequential` | `sequential`, `pingpong` or `random`. |
| `seed` *(optional)* | INT | `0` | Used by `random` only. |

| Output | Type | Description |
|---|---|---|
| `current_prompt` | STRING | The chosen line. Empty when `prompts` has no lines. |
| `progress` | FLOAT | `current_step / total_steps`, never above 1.0. |
| `step_index` | INT | Zero-based position of the chosen line. |

| `mode` | Behaviour |
|---|---|
| `sequential` | Divides `total_steps` evenly between the prompts: `total_steps // number of prompts` steps each, at least 1. After the last block it stays on the last prompt. |
| `pingpong` | Goes forward, then backward: 1, 2, 3, 2, 1, 2, 3 and so on. `total_steps` only affects `progress`. |
| `random` | Picks a line for each step with a generator seeded by `seed + current_step`. The same seed and step always give the same line. |

Three prompts (`cat`, `dog`, `bird`) and `total_steps` 12:

```text
current_step   0    1    2    3    4    5    6    7    8    9   10   11
sequential    cat  cat  cat  cat  dog  dog  dog  dog bird bird bird bird
pingpong      cat  dog bird  dog  cat  dog bird  dog  cat  dog bird  dog
```

**Example — one prompt per queue**

```text
Counter (NH).current                 → Prompt Scheduler (NH).current_step   (Counter: start 0, step 1, max_value 11)
Prompt Scheduler (NH).current_prompt → CLIP Text Encode.text
```

Set `total_steps` to the number of queues in the sequence, here 12.

> **Note:** This node does not read the sampler's steps. `current_step` is simply a number you supply.

---

## String Operations (NH)

<p align="center"><img src="../assets/nodes/NH_StringOps.svg" alt="String Operations (NH) node" width="360"></p>

Runs one text operation, such as changing case, trimming, replacing, testing or slicing, and returns the text, a true/false result and a length.

**Use it when** you need to clean up a name, test a file name (`endswith .png`), or cut a piece out of a string.

| Input | Type | Default | Description |
|---|---|---|---|
| `text` | STRING | `""` | Multiline text to process. |
| `operation` | COMBO | `upper` | `upper`, `lower`, `strip`, `title`, `replace`, `contains`, `startswith`, `endswith`, `length` or `slice`. |
| `param_a` *(optional)* | STRING | `""` | The text to search for, the prefix or suffix, or the slice. |
| `param_b` *(optional)* | STRING | `""` | The replacement text for `replace`. |

| Output | Type | Description |
|---|---|---|
| `text_out` | STRING | The processed text. |
| `bool_out` | BOOLEAN | The result of a test; `True` for operations that are not tests. |
| `int_out` | INT | Number of characters. |

| `operation` | `text_out` | `bool_out` | `int_out` |
|---|---|---|---|
| `upper`, `lower`, `title` | Converted text | `True` | Length of `text` |
| `strip` | `text` without leading and trailing whitespace | `True` | Length of the result |
| `replace` | Every `param_a` replaced by `param_b` | `True` | Length of the result |
| `contains` | `text`, unchanged | `True` if `param_a` is found, ignoring case | Length of `text` |
| `startswith` | `text`, unchanged | `True` if `text` starts with `param_a` (case-sensitive) | Length of `text` |
| `endswith` | `text`, unchanged | `True` if `text` ends with `param_a` (case-sensitive) | Length of `text` |
| `length` | `text`, unchanged | `True` if `text` is not empty | Length of `text` |
| `slice` | The part chosen by `param_a` | `True` | Length of the result |

**Slice syntax.** `param_a` is `start:end`, where either side may be empty and negative numbers count from the end: `2:5`, `:3`, `5:` or `-3:`. A single number returns one character, counted from 0. An invalid slice, or one with a step such as `1:6:2`, returns the text unchanged.

| `text` | `operation` | `param_a` | `param_b` | `text_out` | `bool_out` | `int_out` |
|---|---|---|---|---|---|---|
| `  Red Dress  ` | `strip` | — | — | `Red Dress` | `True` | 9 |
| `red dress` | `upper` | — | — | `RED DRESS` | `True` | 9 |
| `red dress` | `title` | — | — | `Red Dress` | `True` | 9 |
| `red dress` | `replace` | `red` | `blue` | `blue dress` | `True` | 10 |
| `Red Dress` | `contains` | `DRESS` | — | `Red Dress` | `True` | 9 |
| `portrait_01.png` | `endswith` | `.png` | — | `portrait_01.png` | `True` | 15 |
| `portrait_01.png` | `startswith` | `Portrait` | — | `portrait_01.png` | `False` | 15 |
| `abcdefgh` | `slice` | `2:5` | — | `cde` | `True` | 3 |
| `abcdefgh` | `slice` | `-3:` | — | `fgh` | `True` | 3 |

**Example — branch on a file extension**

```text
Load Image Info (NH).file_name  → String Operations (NH).text    (operation: endswith, param_a: .png)
String Operations (NH).bool_out → If/Else (NH).condition
```

> **Tip:** `replace` with an empty `param_a` puts `param_b` between every character: `abc` with `-` becomes `-a-b-c-`.

---

## Prompt Join (NH)

<p align="center"><img src="../assets/nodes/NH_PromptJoin.svg" alt="Prompt Join (NH) node" width="360"></p>

Joins up to five text fragments with a separator and tells you how many were joined.

**Use it when** you build a prompt from separate pieces, such as subject, style and quality tags, and some pieces may be empty.

| Input | Type | Default | Description |
|---|---|---|---|
| `separator` | STRING | `", "` | Text placed between fragments (comma and space). Used exactly as typed. |
| `skip_empty` | BOOLEAN | `True` | Drop fragments that are empty after trimming. |
| `text_1` … `text_5` *(optional)* | STRING | `""` | The fragments. Connection only: there is no text box. |

| Output | Type | Description |
|---|---|---|
| `result` | STRING | The joined text. |
| `count` | INT | Number of fragments that were joined. |

**How it works**

- Each connected fragment is trimmed. Inputs that are not connected are always ignored.
- With `skip_empty` on, empty fragments are dropped. With it off, an empty fragment still gets its separator.
- The separator is not converted: `\n` stays two characters. For a real line break, use [Text Concatenate](#text-concatenate-nh).

| `separator` | `skip_empty` | Fragments | `result` | `count` |
|---|---|---|---|---|
| `", "` | on | `masterpiece`, —, `8k photo` | `masterpiece, 8k photo` | 2 |
| `", "` | off | `masterpiece`, —, `8k photo` | `masterpiece, , 8k photo` | 3 |
| `" / "` | on | `text_1` `a`, `text_5` `e` | `a / e` | 2 |

**Example — add quality tags to a prompt**

```text
Prompt Template (NH).result → Prompt Join (NH).text_1
Text "masterpiece, 8k"      → Prompt Join (NH).text_2
Prompt Join (NH).result     → CLIP Text Encode.text
```

---

## Text Split (NH)

<p align="center"><img src="../assets/nodes/NH_TextSplit.svg" alt="Text Split (NH) node" width="360"></p>

Splits text at a delimiter and returns the items, how many there are, and the first and last item.

**Use it when** you receive a list written on one line, such as `red, green, blue`, and need its first or last entry or its length.

| Input | Type | Default | Description |
|---|---|---|---|
| `text` | STRING | `""` | Multiline text to split. |
| `delimiter` | STRING | `", "` | Where to cut (comma and space). `\n` means a line break and `\t` a tab. |
| `max_splits` *(optional)* | INT | `-1` | -1 to 100. -1 splits everywhere; N makes at most N cuts. |

| Output | Type | Description |
|---|---|---|
| `items_joined` | STRING | The items, one per line. |
| `count` | INT | Number of items. |
| `first` | STRING | First item. |
| `last` | STRING | Last item. |

**How it works**

- The delimiter is matched exactly, spaces included. Each item is trimmed and empty items are dropped.
- `max_splits` N makes at most N cuts, so there are at most N + 1 items and the rest of the text stays in the last one. `0` makes no cut.
- Empty or blank text gives empty outputs and `count` 0. An empty `delimiter` is an error.

| `text` | `delimiter` | `max_splits` | `items_joined` | `count` | `first` | `last` |
|---|---|---|---|---|---|---|
| `red, green, blue` | `", "` | -1 | `red\ngreen\nblue` | 3 | `red` | `blue` |
| `red,green,blue` | `", "` | -1 | `red,green,blue` | 1 | `red,green,blue` | `red,green,blue` |
| `red,green,blue` | `","` | -1 | `red\ngreen\nblue` | 3 | `red` | `blue` |
| `red, green, blue, pink` | `", "` | 1 | `red\ngreen, blue, pink` | 2 | `red` | `green, blue, pink` |
| `a,, b ,,c` | `","` | -1 | `a\nb\nc` | 3 | `a` | `c` |

**Example — take the first tag of a list**

```text
Text "red, green, blue" → Text Split (NH).text
Text Split (NH).first   → CLIP Text Encode.text
```

> **Tip:** If a list has no spaces after the commas, set the delimiter to `,` without a space, as in the third row.

---

## Regex Extract (NH)

<p align="center"><img src="../assets/nodes/NH_RegexExtract.svg" alt="Regex Extract (NH) node" width="360"></p>

Finds, replaces or splits text with a regular expression and reports the result, whether it matched, the captured groups and a count.

**Use it when** you need to pull a number out of a file name, rewrite part of a string, or cut text at a pattern.

| Input | Type | Default | Description |
|---|---|---|---|
| `text` | STRING | `""` | Multiline text to search. |
| `pattern` | STRING | `(\d+)` | The regular expression, in Python syntax. |
| `mode` | COMBO | `match` | `match`, `findall`, `replace` or `split`. |
| `replacement` *(optional)* | STRING | `""` | Used by `replace`; `\1` and `\g<name>` insert groups. |

| Output | Type | Description |
|---|---|---|
| `result` | STRING | The main result; see the table of modes. |
| `matched` | BOOLEAN | `True` when the pattern was found (for `split`, when the text was cut). |
| `groups` | STRING | Captured groups, or the original text for `replace`. |
| `count` | INT | Number of matches or pieces. |

| `mode` | What it does | `result` | `groups` | `count` |
|---|---|---|---|---|
| `match` | Finds the first match anywhere in `text`, not only at the start. | The whole match | Capture groups, one per line (the whole match if the pattern has no groups) | 1, or 0 |
| `findall` | Finds every match. | One match per line (several groups: joined with `, `) | Same as `result` | Number of matches |
| `replace` | Replaces every match with `replacement`. | The new text | The original text | Number of matches |
| `split` | Cuts the text at each match, trims the pieces and drops empty ones. | One piece per line | Same as `result` | Number of pieces |

| `text` | `pattern` | `mode` | `result` | `groups` | `count` |
|---|---|---|---|---|---|
| `size 1024x768` | `(\d+)x(\d+)` | `match` | `1024x768` | `1024\n768` | 1 |
| `size 1024x768` | `\d+` | `match` | `1024` | `1024` | 1 |
| `no digits` | `\d+` | `match` | — (`matched` is `False`) | — | 0 |
| `a1b22c333` | `\d+` | `findall` | `1\n22\n333` | `1\n22\n333` | 3 |
| `a1b22c333` | `\d+` | `replace` with `#` | `a#b#c#` | `a1b22c333` | 3 |
| `1024x768` | `(\d+)x(\d+)` | `replace` with `\2x\1` | `768x1024` | `1024x768` | 1 |
| `red; blue ;green` | `\s*;\s*` | `split` | `red\nblue\ngreen` | `red\nblue\ngreen` | 3 |

**Example — read the number in a file name**

```text
Load Image Info (NH).file_name → Regex Extract (NH).text      (pattern: (\d+), mode: match)
Regex Extract (NH).result      → the number inside the file name, as text
```

> **Tip:** Add `(?i)` at the start of the pattern to ignore case, and `(?m)` to make `^` and `$` match at every line instead of only at the ends of the whole text.

> **Note:** A pattern with an error does not stop the workflow. `result` is the unchanged text, `matched` is `False`, `count` is 0, and the console shows the message.

---

## Text Index (NH)

<p align="center"><img src="../assets/nodes/NH_TextIndex.svg" alt="Text Index (NH) node" width="360"></p>

Outputs one of up to 64 typed texts, chosen by a 1-based number.

**Use it when** you keep several prompts in one node and want to select one by number, for example from a counter.

| Input | Type | Default | Description |
|---|---|---|---|
| `index` | INT | `1` | Which text to output, 1 to 64. |
| `text_count` | INT | `5` | How many text boxes are used, 1 to 64. |
| `text_1` … `text_64` *(optional)* | STRING | `""` | Multiline text boxes. Only the first `text_count` are used. |

| Output | Type | Description |
|---|---|---|
| `text` | STRING | The text in box number `index`. |

**How it works**

- Numbering starts at 1, so `index` 1 is `text_1`.
- The node shows only `text_count` boxes. If the layout looks stale after loading a workflow, press `update texts`.
- If the chosen box is empty, or `index` is outside 1 to `text_count`, the output is blocked. Nodes after Text Index are skipped without an error. If a sampler after it does not run, check for an empty box or an `index` out of range.

| `index` | `text_count` | Boxes | `text` |
|---|---|---|---|
| 2 | 3 | `red`, `blue`, `green` | `blue` |
| 3 | 3 | `red`, `blue`, — | Blocked: the next nodes are skipped |
| 4 | 3 | `red`, `blue`, `green`, `pink` | Blocked: `index` is above `text_count` |

**Example — one prompt per queue**

```text
Counter (NH).current → Text Index (NH).index      (Counter: start 1, step 1, max_value 5)
Text Index (NH).text → CLIP Text Encode.text
```

> **Tip:** Value Match Index also counts from 1, so its `index` output connects here directly.

---

## Text Concatenate (NH)

<p align="center"><img src="../assets/nodes/NH_TextConcatenate.svg" alt="Text Concatenate (NH) node" width="360"></p>

Joins up to 64 strings with a delimiter, skipping the empty ones.

**Use it when** you build a folder path or a file name from parts, such as `renders/portraits/final`.

| Input | Type | Default | Description |
|---|---|---|---|
| `string_count` | INT | `2` | How many string fields are used, 2 to 64. |
| `delimiter` | STRING | `/` | Placed between strings. `\n` means a line break and `\t` a tab. |
| `clean_whitespace` | BOOLEAN | `True` | Collapse runs of whitespace into one space and trim each string. |
| `string_1` … `string_64` *(optional)* | STRING | `""` | Single-line fields. Only the first `string_count` are used. |

| Output | Type | Description |
|---|---|---|
| `text` | STRING | The joined text. |

**How it works**

- Strings are taken in order, from `string_1` to `string_<string_count>`. Empty ones are skipped, so no delimiter is left behind.
- With `clean_whitespace` on, a string that holds only whitespace counts as empty. With it off, strings are used exactly as typed.
- The node shows only `string_count` fields. If the layout looks stale after loading a workflow, press `update strings`.

| `delimiter` | `clean_whitespace` | Strings | `text` |
|---|---|---|---|
| `/` | on | `renders`, `portraits`, —, `final` (`string_count` 4) | `renders/portraits/final` |
| `", "` | on | `"  red   dress "`, `" studio   light "` | `red dress, studio light` |
| `\n` | on | `a`, `b`, `c` (`string_count` 3) | `a`, `b` and `c` on three lines |
| empty | on | `a`, `b`, `c` (`string_count` 3) | `abc` |

**Example — build a save prefix**

```text
Text Concatenate (NH).text → Save Image.filename_prefix      (strings: renders, portraits, final)
```

---

## Text Split Lines (NH)

<p align="center"><img src="../assets/nodes/NH_TextSplitLines.svg" alt="Text Split Lines (NH) node" width="360"></p>

Turns multiline text into a list of lines. Connect `lines` to a STRING input and ComfyUI runs the connected nodes once per line, like batch processing for text. The node also outputs one chosen line as `selected`.

**Use it when** you want one image per prompt from a list of prompts, or you want to pick a single line by position.

| Input | Type | Default | Description |
|---|---|---|---|
| `text` | STRING | Three lines: `first prompt`, `second prompt`, `third prompt` | Multiline text, one entry per line. |
| `index` | INT | `0` | Which line `selected` returns, counted from 0. -1 picks a random line using `seed`. Range -1 to 9999. |
| `mode` | COMBO | `sequential` | `sequential` keeps the order, `shuffle` reorders with the seed, `reverse` flips the order. |
| `seed` | INT | `0` | 0 to 4294967295. Used by `shuffle` and by `index` -1. |
| `delimiter` *(optional)* | STRING | `\n` | Where to cut. `\n` means a line break and `\t` a tab. |
| `skip_empty` *(optional)* | BOOLEAN | `True` | Drop empty lines. |
| `strip_whitespace` *(optional)* | BOOLEAN | `True` | Trim each line. |
| `wrap_index` *(optional)* | BOOLEAN | `True` | When `index` is past the last line, wrap around. Off: use the last line. |

| Output | Type | Description |
|---|---|---|
| `lines` | STRING (list) | Every line, in the chosen order. Nodes connected here run once per line. |
| `selected` | STRING | The line chosen by `index`. |
| `count` | INT | Number of lines. |
| `preview` | STRING | All lines numbered from 0, with `>>>` after the selected one. |
| `is_first` | BOOLEAN | `True` when `selected` is the first line of the ordered list. |
| `is_last` | BOOLEAN | `True` when `selected` is the last line of the ordered list. |

**How it works**

- The text is cut at `delimiter`, then trimmed and cleaned of empty lines as set by `strip_whitespace` and `skip_empty`.
- Only `lines` is a list. The other outputs are single values and do not trigger extra runs.
- `mode` reorders the lines first, and `selected` is picked from the reordered list. `shuffle` is repeatable: the same seed gives the same order.
- With `wrap_index` on, `index` 5 with 3 lines becomes 5 mod 3 = 2. With it off, 5 is limited to the last line.
- Empty text gives `count` 0, an empty `selected` and `(empty)` as `preview`.

With `text` `cat\ndog\nbird`:

| `index` | `mode` | `seed` | `selected` | Why |
|---|---|---|---|---|
| 1 | `sequential` | any | `dog` | Line 1, counted from 0. |
| 5 | `sequential` | any | `bird` | 5 wraps to line 2. With `wrap_index` off it is limited to line 2 as well. |
| 0 | `reverse` | any | `bird` | `lines` becomes `bird`, `dog`, `cat`. |
| 0 | `shuffle` | 0 | `cat` | `lines` becomes `cat`, `bird`, `dog`. |
| -1 | `sequential` | 0 | `dog` | A random line chosen with seed 0. |
| -1 | `sequential` | 1 | `cat` | Seed 1 chooses another line. |

`preview` for `index` 1 in `sequential` mode:

```text
[0] cat
[1] dog >>>
[2] bird
```

**Example — one image per line**

```text
Text Split Lines (NH).lines → CLIP Text Encode.text → KSampler → VAE Decode → Save Image
```

With three lines of text, ComfyUI runs the chain three times and saves three images.

---

## Text Random Line (NH)

<p align="center"><img src="../assets/nodes/NH_TextRandomLine.svg" alt="Text Random Line (NH) node" width="360"></p>

Picks one random line from multiline text. The same seed always gives the same line.

**Use it when** you want a random prompt, style or location from a list and need nothing more than that. For reordering, wrapping or a list output, use [Text Split Lines](#text-split-lines-nh).

| Input | Type | Default | Description |
|---|---|---|---|
| `text` | STRING | Three lines: `option A`, `option B`, `option C` | Multiline text, one option per line. Blank lines are ignored and each line is trimmed. |
| `seed` | INT | `0` | 0 to 4294967295. Chooses the line. |

| Output | Type | Description |
|---|---|---|
| `text` | STRING | The picked line. Empty when there are no lines. |
| `picked_index` | INT | Position of the picked line, counted from 0. -1 when there are no lines. |
| `total_lines` | INT | Number of lines to choose from. |

With `text` `option A\noption B\noption C\noption D`:

| `seed` | `text` | `picked_index` | `total_lines` |
|---|---|---|---|
| 0 | `option D` | 3 | 4 |
| 1 | `option B` | 1 | 4 |
| 2 | `option A` | 0 | 4 |
| 5 | `option C` | 2 | 4 |

**Example — a random location per queue**

```text
Text Random Line (NH).text → CLIP Text Encode.text
```

> **Tip:** The `seed` widget has ComfyUI's `control after generate` setting. `randomize` gives a new line each time you queue; `fixed` repeats the same line.

---

**See also:** [Logic](./logic.md) for switches that pick between texts and for `Compare (NH)`, which turns a text test into a condition; [Batch & Lists](./batch-lists.md) for `Counter (NH)` and the `NH_LIST` nodes (`List Create (NH)` and others) that also build lists from text.

[← Back to guide](./README.md)
