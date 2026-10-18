[← Back to guide](./README.md)

# Logic

Logic nodes let a workflow make decisions. Compare values, combine true/false results, choose between inputs, send data down one of several paths, calculate simple formulas, or pick at random. `If/Else`, `Switch N`, `Random Choice` and `Any Switch` use lazy evaluation: ComfyUI runs only the branch that is chosen, so a sampler on an unchosen branch never executes. Find the nodes under **NH-Nodes → Logic** in the node menu.

**On this page**

- [Which switch do I need?](#which-switch-do-i-need) — how each switch treats the branch you did not choose
- [Compare (NH)](#compare-nh) — turn two values into `True` or `False`
- [Logic Gate (NH)](#logic-gate-nh) — combine true/false values with AND, OR, NOT, XOR, NAND, NOR
- [If/Else (NH)](#ifelse-nh) — pick one of two inputs by a condition
- [Switch N (NH)](#switch-n-nh) — pick one of up to 10 inputs by number
- [Any Switch (NH)](#any-switch-nh) — pick one of two inputs by a condition, same logic as If/Else
- [Any Branch Switch (NH)](#any-branch-switch-nh) — send one input to one of five outputs
- [Gate Switch (NH)](#gate-switch-nh) — let data through, or stop it and everything after it
- [Value Match Index (NH)](#value-match-index-nh) — turn text, such as a file name, into a number for a switch
- [Math Eval (NH)](#math-eval-nh) — calculate a formula with the variables `a` to `d`
- [Random Choice (NH)](#random-choice-nh) — pick one of up to six inputs at random, with weights

In the tables, `*` means any data type.

## Which switch do I need?

Two mechanisms keep unused work from running. With **lazy evaluation**, nodes that only feed an unchosen input never run, so a sampler there costs no time. With **blocking**, the unchosen output carries a block signal: every node after it is skipped, silently and without an error, including Save and Preview nodes.

| Node | Chooses by | Inputs → outputs | The side that is not chosen |
|---|---|---|---|
| If/Else, Any Switch | `True` / `False` | 2 → 1 | Never runs (lazy). |
| Switch N | Number 0 to 9 | up to 10 → 1 | Never runs (lazy). |
| Random Choice | Seed and weights | up to 6 → 1 | Never runs (lazy). |
| Any Branch Switch | Number 0 to 4 | 1 → 5 | The other outputs are blocked. |
| Gate Switch | `True` / `False` | 1 → 1 | The input never runs (lazy). When disabled, the output is blocked. |

> **Note:** Lazy evaluation skips only what nothing else needs. If a node also feeds something that does run, it still runs.

---

## Compare (NH)

<p align="center"><img src="../assets/nodes/NH_Compare.svg" alt="Compare (NH) node" width="360"></p>

Compares `a` with `b` and outputs `True` or `False`. Both values also pass through unchanged.

**Use it when** you need a condition for If/Else or Gate Switch, such as "is the width larger than 1024?" or "does the file name contain `portrait`?".

| Input | Type | Default | Description |
|---|---|---|---|
| `a` | `*` | — | Left value, any type. Connect a node; there is no widget. |
| `b` | `*` | — | Right value, any type. Connect a node. |
| `op` | COMBO | `is` | `is`, `not`, `>`, `<`, `>=`, `<=` or `in`. |
| `type_cast` *(optional)* | COMBO | `auto` | `auto`, `int`, `float` or `string`. `in` ignores it. |

| Output | Type | Description |
|---|---|---|
| `result` | BOOLEAN | `True` when the comparison holds. |
| `a_passthrough` | `*` | `a`, unchanged. |
| `b_passthrough` | `*` | `b`, unchanged. |

**How it works**

- `auto` reads both values as numbers when it can (`"10"` equals `10`). If either side is not a number, it compares both as text.
- `int` and `float` convert both sides first; `int` drops decimals (3.7 becomes 3). If a value cannot be converted, `result` is `False` and the console explains why.
- `string` always compares text, character by character.
- `in` is plain membership. If `b` is a list, tuple or set, `a` must be one of its items. If `b` is a dictionary, `a` must be one of its keys. Otherwise `str(a)` must appear inside `str(b)`, case-sensitive. A number matches its text (`7` matches `"7"`). There is no fuzzy matching, and an empty `a` is found in any text.
- If `a` or `b` is empty (`None`), `result` is `False`.

| `a` | `op` | `b` | `type_cast` | `result` | Why |
|---|---|---|---|---|---|
| `1920` | `>` | `1024` | `auto` | `True` | Numbers. |
| `"10"` | `>` | `"9"` | `auto` | `True` | Both read as numbers. |
| `"10"` | `>` | `"9"` | `string` | `False` | As text, `"1"` sorts before `"9"`. |
| `3.7` | `is` | `3` | `int` | `True` | `int` drops the decimals. |
| `3.7` | `is` | `3` | `auto` | `False` | 3.7 is not 3. |
| `"portrait"` | `in` | `"portrait_01.png"` | — | `True` | Substring. |
| `"PORTRAIT"` | `in` | `"portrait_01.png"` | — | `False` | Case-sensitive. |
| `13` | `in` | `"a13b"` | — | `True` | Substring. |
| `13` | `in` | `"1024x768 3"` | — | `False` | `13` is not written anywhere in it. |
| `7` | `in` | list `[1, 7]` | — | `True` | List item. |

**Example — run a step only for wide images**

```text
Get Image Size.width → Compare (NH).a        (op: >)
Int (value 1024)     → Compare (NH).b
Compare (NH).result  → If/Else (NH).condition
```

> **Tip:** To compare with a fixed value, connect one of ComfyUI's `Int`, `Float` or `Text` primitive nodes to `a` or `b`.

---

## Logic Gate (NH)

<p align="center"><img src="../assets/nodes/NH_LogicGate.svg" alt="Logic Gate (NH) node" width="360"></p>

Combines two true/false values with AND, OR, NOT, XOR, NAND or NOR and outputs one true/false result.

**Use it when** several conditions decide together, such as "the image is wide enough AND quality mode is on".

| Input | Type | Default | Description |
|---|---|---|---|
| `a` | BOOLEAN | `False` | First value. |
| `op` | COMBO | `AND` | `AND`, `OR`, `NOT`, `XOR`, `NAND` or `NOR`. |
| `b` *(optional)* | BOOLEAN | `False` | Second value. `NOT` ignores it. |

| Output | Type | Description |
|---|---|---|
| `result` | BOOLEAN | The result of `op`. |

| `op` | `True` when… |
|---|---|
| `AND` | both are `True` |
| `OR` | at least one is `True` |
| `NOT` | `a` is `False` (`b` is ignored) |
| `XOR` | exactly one is `True` |
| `NAND` | not both are `True` (the opposite of AND) |
| `NOR` | both are `False` (the opposite of OR) |

**Truth table** (`T` = `True`, `F` = `False`)

| `a` | `b` | AND | OR | NOT a | XOR | NAND | NOR |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| F | F | F | F | T | F | T | T |
| F | T | F | T | T | T | T | F |
| T | F | F | T | F | T | T | F |
| T | T | T | T | F | F | F | F |

**Example — both conditions must pass**

```text
Compare (NH) "width > 1024" → Logic Gate (NH).a     (op: AND)
Boolean Switch (NH)         → Logic Gate (NH).b
Logic Gate (NH).result      → Gate Switch (NH).enabled
```

- `OR`: run a face fix when a detector finds a face **or** you force it on with a `Boolean Switch (NH)`.
- `NOT`: send a `Boolean Switch (NH)` named `skip_upscale` through `NOT` into `enabled`, so turning it on switches the upscale branch off.

> **Tip:** `docs/logic_gate_diagram.html` is an illustrated version of these recipes. Download it and open it in a browser.

---

## If/Else (NH)

<p align="center"><img src="../assets/nodes/NH_IfElse.svg" alt="If/Else (NH) node" width="360"></p>

Outputs `if_true` when `condition` is `True` and `if_false` otherwise. Only the chosen branch is executed.

**Use it when** you want two versions of a result, for example with or without a refine pass, without paying for both.

| Input | Type | Default | Description |
|---|---|---|---|
| `condition` | BOOLEAN | `True` | Chooses the branch. |
| `if_true` | `*` | — | Lazy. Used when `condition` is `True`. |
| `if_false` | `*` | — | Lazy. Used when `condition` is `False`. |

| Output | Type | Description |
|---|---|---|
| `result` | `*` | The chosen input, unchanged. |
| `condition_out` | BOOLEAN | `condition`, repeated for chaining. |

**How it works**

- ComfyUI evaluates `condition` first, then asks only for the chosen branch. Nodes that feed the other branch never run, so a sampler there is skipped.
- Both inputs must be connected, but only one of them is executed.

**Example — refine only when needed**

```text
Compare (NH).result            → If/Else (NH).condition
VAE Decode                     → If/Else (NH).if_false   (original image)
second KSampler → VAE Decode   → If/Else (NH).if_true    (refined image; runs only when True)
If/Else (NH).result            → Save Image
```

> **Tip:** Unlike Gate Switch, If/Else always produces an output, so `Save Image` still runs when the condition is `False`.

---

## Switch N (NH)

<p align="center"><img src="../assets/nodes/NH_SwitchN.svg" alt="Switch N (NH) node" width="360"></p>

Outputs the input whose slot number equals `index` (`input_0` to `input_9`). Only the chosen input is executed.

**Use it when** you cycle through several prompts, models or images, or when another node picks the branch by number.

| Input | Type | Default | Description |
|---|---|---|---|
| `index` | INT | `0` | Slot to use, 0 to 9. |
| `input_0` … `input_9` *(optional)* | `*` | — | Lazy. Connect only the slots you need; gaps are fine. |

| Output | Type | Description |
|---|---|---|
| `result` | `*` | The chosen input. |
| `count` | INT | Number of connected slots. |

**How it works**

- Only the chosen slot is executed. The nodes behind the other slots never run.
- If the chosen slot is not connected, or `index` is out of range (for example `-1` arriving from another node), the lowest connected slot is used and the console prints a note.
- With no connected slot, `result` is empty and the console prints a warning; `count` is `0`.
- `count` counts connected slots, not slot numbers: `input_0` and `input_5` give `2`. A slot whose upstream node returns nothing still counts as connected.

**Example — Counter → Switch N → KSampler**

```text
Counter (NH).current → Switch N (NH).index        (Counter: start 0, step 1, max_value 2)
CLIP Text Encode A / B / C → Switch N (NH).input_0 / input_1 / input_2
Switch N (NH).result → KSampler.positive
```

Each queue advances the counter, so the next prompt is used, and only its CLIP Text Encode node runs. See [Counter (NH)](./batch-lists.md#counter-nh).

> **Tip:** Value Match Index returns 1-based numbers, so `value_1` selects `input_1`. Connect your default branch to `input_0`: a "no match" index of `-1` falls back to the lowest connected slot.

---

## Any Switch (NH)

<p align="center"><img src="../assets/nodes/NH_AnySwitchBoolean.svg" alt="Any Switch (NH) node" width="360"></p>

A true/false switch for any data type. It outputs `input_true` or `input_false` and runs only the chosen one.

**Use it when** you prefer these socket names to If/Else. The logic is identical.

| Input | Type | Default | Description |
|---|---|---|---|
| `condition` | BOOLEAN | `True` | Chooses the input. |
| `input_true` | `*` | — | Lazy. Used when `condition` is `True`. |
| `input_false` | `*` | — | Lazy. Used when `condition` is `False`. |

| Output | Type | Description |
|---|---|---|
| `result` | `*` | The chosen input, unchanged. |
| `condition` | BOOLEAN | `condition`, repeated. |

**Example — choose between two source images**

```text
Boolean Switch (NH).boolean → Any Switch (NH).condition
Load Image A / Load Image B → Any Switch (NH).input_true / input_false
```

Only the chosen `Load Image` node runs.

---

## Any Branch Switch (NH)

<p align="center"><img src="../assets/nodes/NH_AnySwitch.svg" alt="Any Branch Switch (NH) node" width="360"></p>

Sends one input to one of five outputs, `output 0` to `output 4`. Nodes connected to the other outputs are skipped.

**Use it when** one image or value must go through exactly one of up to five pipelines, such as five different styles or upscalers.

| Input | Type | Default | Description |
|---|---|---|---|
| `input` | `*` | — | The data to route. Always evaluated. |
| `index` | INT | `0` | Output that receives the data, 0 to 4. Other values are clamped to that range. |

| Output | Type | Description |
|---|---|---|
| `output 0` … `output 4` | `*` | `input` on the chosen output; blocked on the others. |
| `index` | INT | The index actually used, after clamping. |

**How it works**

- Unchosen outputs are blocked. A node that receives a blocked value is skipped silently, and so is everything after it.
- `input` is not lazy: the node feeding it always runs. Only what comes after the switch is skipped.
- A node with several inputs is skipped when any one of them is blocked.

**Example — route by file name**

```text
Value Match Index (NH).index → Any Branch Switch (NH).index
VAE Decode                   → Any Branch Switch (NH).input
output 0 / output 1 / …      → one processing chain per output, each ending in its own Save Image
```

> **Tip:** `-1` (no match) is clamped to `0`, so `output 0` works as the default path.

---

## Gate Switch (NH)

<p align="center"><img src="../assets/nodes/NH_GateSwitch.svg" alt="Gate Switch (NH) node" width="360"></p>

Passes `input` through when `enabled` is `True`. When it is `False`, it stops the data and every node after it.

**Use it when** an optional stage, such as an extra upscale or an extra save, should run only when a condition is met.

| Input | Type | Default | Description |
|---|---|---|---|
| `input` | `*` | — | Lazy. The data to pass on. |
| `enabled` | BOOLEAN | `True` | `True` lets the data through; `False` blocks it. |

| Output | Type | Description |
|---|---|---|
| `output` | `*` | `input` when enabled; blocked when disabled. |

**How it works**

- Enabled: `input` is evaluated and passed on unchanged.
- Disabled: `input` is never requested, so the nodes that feed it do not run. The output is blocked, so the nodes after the gate are skipped without an error.

**Example — an optional extra output**

```text
VAE Decode                 → Save Image               (always runs)
VAE Decode → Upscale Image → Gate Switch (NH).input
Boolean Switch (NH)        → Gate Switch (NH).enabled
Gate Switch (NH).output    → second Save Image        (runs only when enabled)
```

> **Tip:** To keep saving the original image when the optional step is off, use If/Else instead: `if_false` carries the original.

---

## Value Match Index (NH)

<p align="center"><img src="../assets/nodes/NH_ValueMatchIndex.svg" alt="Value Match Index (NH) node" width="360"></p>

Looks for your configured values in the input and returns the 1-based position of the single value that matches.

**Use it when** a file name, label or prompt should decide which branch to take, for example "name contains `red`" selects branch 2.

| Input | Type | Default | Description |
|---|---|---|---|
| `input` | `*` | — | The value to test. It is converted to text. |
| `compare_mode` | COMBO | `in` | `in`: the configured value must appear inside the input. `is`: it must equal the input exactly. |
| `value_count` | INT | `5` | How many `value_N` fields are used, 1 to 64. |
| `no_match_index` | INT | `-1` | Returned when nothing matches (-9999 to 9999). |
| `multi_match_index` | INT | `-1` | Returned when more than one value matches (-9999 to 9999). |
| `value_1` … `value_64` *(optional)* | STRING | `""` | The values to look for. Empty fields are ignored. |

| Output | Type | Description |
|---|---|---|
| `index` | INT | Number of the matching value (1 for `value_1`), or `no_match_index` / `multi_match_index`. |

**How it works**

- Matching is on text and is case-sensitive. Only the first `value_count` fields count.
- Exactly one match returns its number. No match returns `no_match_index`. Two or more matches return `multi_match_index`.
- The node shows only `value_count` fields. If the layout looks stale after loading a workflow, press `update values`.

With `value_count` 3, `value_1` `blue`, `value_2` `red`, `value_3` `green`, `no_match_index` `-1` and `multi_match_index` `-2`:

| `input` | `compare_mode` | `index` | Why |
|---|---|---|---|
| `red_dress_03.png` | `in` | `2` | Only `red` is inside the name. |
| `blue_red_mix.png` | `in` | `-2` | `blue` and `red` both match. |
| `black.png` | `in` | `-1` | Nothing matches. |
| `red` | `is` | `2` | Exact match. |
| `red_dress` | `is` | `-1` | `is` needs the whole text to match. |
| `Red` | `is` | `-1` | Case-sensitive. |

**Example — pick a branch from a file name**

```text
Load Image Info (NH).file_name → Value Match Index (NH).input
Value Match Index (NH).index   → Switch N (NH).index
```

> **Tip:** Numbers are matched as text, so with `in` the value `1` also matches `12` and `21`. Use `is` for exact numbers.

---

## Math Eval (NH)

<p align="center"><img src="../assets/nodes/NH_MathEval.svg" alt="Math Eval (NH) node" width="360"></p>

Calculates a math expression that can use up to four variables, `a` to `d`, and returns the result as FLOAT, INT and STRING.

**Use it when** you need to snap a size to a multiple of 64, scale a value, or build a number for another node.

| Input | Type | Default | Description |
|---|---|---|---|
| `expression` | STRING | `a + b` | The formula, on one line. |
| `a`, `b`, `c`, `d` *(optional)* | FLOAT | `0.0` | Variables for the formula (step 0.01). |
| `round_to` *(optional)* | INT | `-1` | Decimal places to round to, from `-1` to `10`. `-1` leaves the result unrounded. |

| Output | Type | Description |
|---|---|---|
| `result_float` | FLOAT | The result. |
| `result_int` | INT | The result with decimals cut off (toward zero). |
| `result_string` | STRING | The result as text. |

| Part | Allowed |
|---|---|
| Numbers | `3`, `0.5`, `1e3` |
| Variables | `a`, `b`, `c`, `d` (lowercase only) |
| Operators | `+` `-` `*` `/` `//` `%` `**`, unary `+` and `-`, parentheses |
| Functions | `min(a, b)`, `max(a, b)` (two or more values), `abs(x)`, `round(x)` or `round(x, digits)`, `sqrt(x)`, `floor(x)`, `ceil(x)`, `clamp(value, low, high)` |

**How it works**

- `/` is true division (10 / 4 = 2.5), `//` rounds down (10 // 4 = 2), `%` is the remainder and `**` is the power.
- Division by zero with `/`, `//` or `%` gives 0 for that part of the formula and prints a warning in the console. The rest still evaluates: `a + b / 0` gives `a`.
- An invalid expression stops the run with an error that names it, for example `Math Eval: Unknown variable 'e' (expression: 'e + 1')`. It never returns 0 silently.
- `result_int` is taken after `round_to` is applied. Rounding ties go to the even digit (2.5 becomes 2.0, 3.5 becomes 4.0).

| `expression` | Variables | `round_to` | `result_float` | `result_int` | `result_string` |
|---|---|---|---|---|---|
| `a + b * 2` | `a` 1, `b` 3 | -1 | `7.0` | `7` | `7.0` |
| `round(a / 64) * 64` | `a` 1000 | -1 | `1024.0` | `1024` | `1024.0` |
| `ceil(a / 8) * 8` | `a` 1001 | -1 | `1008.0` | `1008` | `1008.0` |
| `clamp(a, 512, 2048)` | `a` 100 | -1 | `512.0` | `512` | `512.0` |
| `a * 9 / 16` | `a` 1920 | -1 | `1080.0` | `1080` | `1080.0` |
| `a * b / 1000000` | `a` 1024, `b` 1536 | -1 | `1.572864` | `1` | `1.572864` |
| `a / 3` | `a` 10 | 2 | `3.33` | `3` | `3.33` |
| `a` | `a` -2.7 (linked) | -1 | `-2.7` | `-2` | `-2.7` |
| `a / b` | `a` 5, `b` 0 (warning in console) | -1 | `0.0` | `0` | `0.0` |

**Example — snap a width to a multiple of 64**

```text
Float (value 1000)        → Math Eval (NH).a        (expression: round(a / 64) * 64)
Math Eval (NH).result_int → Empty Latent Image.width
```

> **Note:** A number widget without a range of its own accepts typed values from 0 to 2048 in ComfyUI. For a negative or larger number, write it in the expression or connect a FLOAT node. ComfyUI does not connect an INT output to a FLOAT input such as `a`.

---

## Random Choice (NH)

<p align="center"><img src="../assets/nodes/NH_RandomChoice.svg" alt="Random Choice (NH) node" width="360"></p>

Picks one of up to six connected inputs at random. `weights` make some inputs more likely. Only the picked input is executed.

**Use it when** each queue should use a random style, prompt or image, with chosen probabilities, and the unpicked chains should cost nothing.

| Input | Type | Default | Description |
|---|---|---|---|
| `seed` | INT | `0` | 0 to 4294967295. The same seed gives the same pick. |
| `weights` | STRING | `1,1,1,1,1,1` | Comma-separated weights. Weight *N* belongs to `input_N`. |
| `input_0` … `input_5` *(optional)* | `*` | — | Lazy. Connect the ones you want in the draw. |

| Output | Type | Description |
|---|---|---|
| `result` | `*` | The picked input. |
| `picked_index` | INT | Slot number of the pick (0 to 5), or `-1` when nothing is connected. |
| `probabilities` | STRING | Chance of each connected input in slot order, for example `75.0%, 25.0%`. |

**How it works**

- Weights are positional: the first belongs to `input_0`, the second to `input_1`, and so on. Weights of unconnected inputs are ignored and the rest are rescaled to 100%.
- A missing or non-numeric weight counts as 1. A negative weight counts as 0. An input with weight 0 is never picked. Only the first six weights are read.
- If every connected input has weight 0, the connected input with the highest slot number is picked.
- An input counts as connected as soon as a link is attached, even when the node behind it returns nothing. It keeps its slot and its share of the weights.
- The seed is used as given, 0 included. The same seed, weights and connections always give the same pick.

| `weights` | Connected inputs | `probabilities` |
|---|---|---|
| `1,1,1,1,1,1` | all six | `16.7%, 16.7%, 16.7%, 16.7%, 16.7%, 16.7%` |
| `3,1` | `input_0`, `input_1` | `75.0%, 25.0%` |
| `1,1,10` | `input_1`, `input_2` | `9.1%, 90.9%` |
| `0,1,1` | `input_0`, `input_1`, `input_2` | `0.0%, 50.0%, 50.0%` |
| `2` | `input_0`, `input_1`, `input_2` | `50.0%, 25.0%, 25.0%` |

**Example — a random style with chosen odds**

```text
three LoRA chains → Random Choice (NH).input_0 / input_1 / input_2   (weights: 6,3,1)
Random Choice (NH).result → KSampler
```

> **Tip:** The `seed` widget has ComfyUI's `control after generate` setting. `randomize` gives a new pick each time you queue; `fixed` repeats the same pick.

---

**See also:** [Batch & Lists](./batch-lists.md) for `Counter (NH)`, which drives a switch one step per queue; [Loaders](./loaders.md) for the indexed loaders that Value Match Index can drive; [Utils](./utils.md) for `Boolean Switch (NH)`; and [Text](./text.md) for `String Operations (NH)`, which turns text tests into true/false values.

[← Back to guide](./README.md)
