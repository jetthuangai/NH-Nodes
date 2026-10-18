"""Wave 3 tests: lazy branch selection (If/Else, Switch N, Random Choice) and the torch Mask Morphology.

The lazy tests run every node through a small emulation of ComfyUI's executor (execution.py, execute()):
connected-but-unevaluated lazy inputs arrive as None, unconnected optional inputs are absent from the kwargs, a
name returned by check_lazy_status is only honoured while that input is still unevaluated, and asking for an input
that is not connected is an error (NodeInputError). The ``old_*`` functions are verbatim copies of the node code
before the lazy change, so the tests prove the choice itself did not move.
"""

import contextlib
import importlib
import io
import itertools
import random

import numpy as np
import pytest
import torch
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    gaussian_filter,
    maximum_filter1d,
    minimum_filter1d,
)

logic_nodes = importlib.import_module("NH-Nodes.nodes.logic.logic_nodes")
math_nodes = importlib.import_module("NH-Nodes.nodes.logic.math_nodes")
morphology = importlib.import_module("NH-Nodes.nodes.mask.mask_morphology")


# --- executor emulation -------------------------------------------------------------------------------------

def run_lazy(node, function, widgets, connected):
    """Run `node` like ComfyUI does.

    widgets: plain (non-lazy) inputs, always available. connected: {input name: upstream value} of every connected
    lazy input; only the ones check_lazy_status asks for are delivered, the rest arrive as None.
    Returns (outputs, requested) with `requested` = the lazy inputs that had to be evaluated upstream, in order.
    """
    evaluated = {}
    requested = []
    while True:
        kwargs = {**widgets, **{name: evaluated.get(name) for name in connected}}
        wanted = [name for name in node.check_lazy_status(**kwargs) if name not in evaluated]
        if not wanted:
            return getattr(node, function)(**kwargs), requested
        for name in wanted:
            assert name in connected, f"asked for {name}, which is not connected (ComfyUI raises NodeInputError)"
            evaluated[name] = connected[name]
            requested.append(name)


# --- the node code before the lazy change (reference) -------------------------------------------------------

def old_switch_n(index, **kwargs):
    connected = {}
    for i in range(10):
        key = f"input_{i}"
        if key in kwargs and kwargs[key] is not None:
            connected[i] = kwargs[key]

    count = len(connected)
    if count == 0:
        return (None, 0)
    if index in connected:
        return (connected[index], count)
    first_key = sorted(connected.keys())[0]
    return (connected[first_key], count)


def old_random_choice(seed, weights, **kwargs):
    slots = 6
    connected = []
    for i in range(slots):
        key = f"input_{i}"
        if key in kwargs and kwargs[key] is not None:
            connected.append((i, kwargs[key]))

    if not connected:
        return (None, -1, "")

    slot_weights = []
    for s in weights.split(",")[:slots]:
        try:
            slot_weights.append(max(0.0, float(s.strip())))
        except (ValueError, TypeError):
            slot_weights.append(1.0)
    slot_weights += [1.0] * (slots - len(slot_weights))
    w = [slot_weights[i] for i, _ in connected]

    total = sum(w) or 1.0
    probs = [x / total for x in w]

    rng = random.Random(seed)
    r = rng.random()
    cumsum = 0.0
    for idx, (orig_i, item) in enumerate(connected):
        cumsum += probs[idx]
        if r <= cumsum:
            prob_str = ", ".join(f"{p:.1%}" for p in probs)
            return (item, orig_i, prob_str)

    orig_i, item = connected[-1]
    prob_str = ", ".join(f"{p:.1%}" for p in probs)
    return (item, orig_i, prob_str)


# --- If/Else ------------------------------------------------------------------------------------------------

def test_if_else_branches_are_lazy_and_the_interface_is_unchanged():
    spec = logic_nodes.NH_IfElse.INPUT_TYPES()
    assert list(spec["required"]) == ["condition", "if_true", "if_false"]
    assert "optional" not in spec
    assert spec["required"]["if_true"] == ("*", {"lazy": True})
    assert spec["required"]["if_false"] == ("*", {"lazy": True})
    assert spec["required"]["condition"] == ("BOOLEAN", {"default": True})  # the condition must run first
    assert logic_nodes.NH_IfElse.RETURN_TYPES == ("*", "BOOLEAN")
    assert logic_nodes.NH_IfElse.RETURN_NAMES == ("result", "condition_out")


@pytest.mark.parametrize(
    "condition,true_value,false_value,expected",
    [
        (True, None, None, ["if_true"]),
        (False, None, None, ["if_false"]),
        (True, None, "B", ["if_true"]),  # the branch that is not chosen being ready does not help
        (False, "A", None, ["if_false"]),
        (True, "A", None, []),
        (False, None, "B", []),
        (True, "A", "B", []),
        (False, "A", "B", []),
    ],
)
def test_if_else_check_lazy_status_asks_only_for_the_chosen_branch(condition, true_value, false_value, expected):
    node = logic_nodes.NH_IfElse()
    assert node.check_lazy_status(condition, if_true=true_value, if_false=false_value) == expected
    assert node.check_lazy_status(condition) == (["if_true"] if condition else ["if_false"])  # defaults: nothing ready


@pytest.mark.parametrize("value", [0, 0.0, "", False, [], torch.zeros(2)])
def test_if_else_an_evaluated_falsy_value_is_not_asked_for_again(value):
    node = logic_nodes.NH_IfElse()
    assert node.check_lazy_status(True, if_true=value) == []
    assert node.check_lazy_status(False, if_false=value) == []


@pytest.mark.parametrize("condition", [True, False])
def test_if_else_runs_only_the_chosen_branch_and_returns_it(condition):
    node = logic_nodes.NH_IfElse()
    connected = {"if_true": "TRUE-VALUE", "if_false": "FALSE-VALUE"}
    outputs, requested = run_lazy(node, "switch", {"condition": condition}, connected)
    assert outputs == ("TRUE-VALUE" if condition else "FALSE-VALUE", condition)
    assert requested == (["if_true"] if condition else ["if_false"])


def test_if_else_keeps_the_exact_object_and_handles_an_empty_chosen_value():
    node = logic_nodes.NH_IfElse()
    tensor = torch.rand(1, 4, 4, 3)
    outputs, requested = run_lazy(node, "switch", {"condition": False}, {"if_true": "x", "if_false": tensor})
    assert outputs[0] is tensor and requested == ["if_false"]
    # an upstream node that returned None: asked once (the executor drops the repeated ask), result is None
    outputs, requested = run_lazy(node, "switch", {"condition": True}, {"if_true": None, "if_false": "x"})
    assert outputs == (None, True) and requested == ["if_true"]


# --- Switch N -----------------------------------------------------------------------------------------------

def test_switch_n_inputs_are_lazy_and_the_interface_is_unchanged():
    spec = logic_nodes.NH_SwitchN.INPUT_TYPES()
    assert list(spec["required"]) == ["index"]
    assert spec["required"]["index"] == ("INT", {"default": 0, "min": 0, "max": 9})
    assert list(spec["optional"]) == [f"input_{i}" for i in range(10)]
    assert all(spec["optional"][f"input_{i}"] == ("*", {"lazy": True}) for i in range(10))
    assert logic_nodes.NH_SwitchN.RETURN_TYPES == ("*", "INT")
    assert logic_nodes.NH_SwitchN.RETURN_NAMES == ("result", "count")


def test_switch_n_description_says_that_connected_inputs_count_even_without_a_value():
    # With lazy inputs the node cannot see a value before it asks for it, so a connected slot whose upstream node
    # returns None is picked (the old code skipped None values). The user is told.
    description = logic_nodes.NH_SwitchN.DESCRIPTION
    assert "Connected inputs count even if they produce no value" in description
    assert "Only the chosen input is executed" in description


def test_switch_n_check_lazy_status_values():
    node = logic_nodes.NH_SwitchN()
    unevaluated = {"input_0": None, "input_2": None, "input_5": None}  # connected, nothing run yet
    assert node.check_lazy_status(2, **unevaluated) == ["input_2"]
    assert node.check_lazy_status(5, **unevaluated) == ["input_5"]
    assert node.check_lazy_status(1, **unevaluated) == ["input_0"]  # slot 1 is not connected: lowest connected slot
    assert node.check_lazy_status(9, **unevaluated) == ["input_0"]
    assert node.check_lazy_status(2, input_0=None, input_2="two", input_5=None) == []  # chosen one is ready
    assert node.check_lazy_status(2, input_0="zero", input_2=None, input_5="five") == ["input_2"]
    assert node.check_lazy_status(3) == []  # nothing connected: nothing to ask for
    assert node.check_lazy_status(0, input_4=0) == []  # a falsy value is an evaluated value


def test_switch_n_runs_only_the_selected_input():
    node = logic_nodes.NH_SwitchN()
    connected = {f"input_{i}": f"value-{i}" for i in range(4)}
    outputs, requested = run_lazy(node, "switch", {"index": 2}, connected)
    assert outputs == ("value-2", 4)
    assert requested == ["input_2"]


def test_switch_n_unconnected_selected_slot_falls_back_to_the_lowest_connected_one(capsys):
    node = logic_nodes.NH_SwitchN()
    connected = {"input_2": "two", "input_5": "five", "input_7": "seven"}
    outputs, requested = run_lazy(node, "switch", {"index": 3}, connected)
    assert outputs == ("two", 3)
    assert requested == ["input_2"]  # the fallback slot, not every connected one
    assert capsys.readouterr().out.count("[NH-Nodes] SwitchN: index 3 not connected, falling back to input_2") == 1
    outputs, requested = run_lazy(node, "switch", {"index": 5}, connected)
    assert (outputs, requested) == (("five", 3), ["input_5"])
    assert capsys.readouterr().out == ""  # no fallback message when the selected slot exists


def test_switch_n_with_nothing_connected_warns_and_returns_none(capsys):
    node = logic_nodes.NH_SwitchN()
    outputs, requested = run_lazy(node, "switch", {"index": 4}, {})
    assert outputs == (None, 0) and requested == []
    assert capsys.readouterr().out.count("[NH-Nodes] Warning: SwitchN has no connected inputs") == 1


def test_switch_n_count_is_the_number_of_connected_slots_not_of_evaluated_ones():
    node = logic_nodes.NH_SwitchN()
    outputs, requested = run_lazy(node, "switch", {"index": 1}, {"input_0": "a", "input_1": "b", "input_9": "c"})
    assert outputs == ("b", 3) and requested == ["input_1"]
    # A connected slot whose upstream produced None still counts as connected (it used to be skipped).
    outputs, requested = run_lazy(node, "switch", {"index": 1}, {"input_0": "a", "input_1": None})
    assert outputs == (None, 2) and requested == ["input_1"]


def test_switch_n_makes_the_same_choice_as_before_for_every_combination():
    node = logic_nodes.NH_SwitchN()
    checked = 0
    with contextlib.redirect_stdout(io.StringIO()):
        for subset in range(1 << 10):
            slots = [i for i in range(10) if subset >> i & 1]
            values = {f"input_{i}": f"value-{i}" for i in slots}
            for index in range(10):
                outputs, requested = run_lazy(node, "switch", {"index": index}, values)
                assert outputs == old_switch_n(index, **values), (slots, index)
                if slots:
                    chosen = index if index in slots else slots[0]
                    assert requested == [f"input_{chosen}"], (slots, index)
                else:
                    assert requested == []
                checked += 1
    assert checked == 10 * 1024


# --- Random Choice ------------------------------------------------------------------------------------------

def test_random_choice_inputs_are_lazy_and_the_interface_is_unchanged():
    spec = math_nodes.NH_RandomChoice.INPUT_TYPES()
    assert list(spec["required"]) == ["seed", "weights"]
    assert spec["required"]["seed"] == ("INT", {"default": 0, "min": 0, "max": 2**32 - 1})
    assert spec["required"]["weights"] == ("STRING", {"default": "1,1,1,1,1,1"})
    assert list(spec["optional"]) == [f"input_{i}" for i in range(6)]
    assert all(spec["optional"][f"input_{i}"] == ("*", {"lazy": True}) for i in range(6))
    assert math_nodes.NH_RandomChoice.RETURN_TYPES == ("*", "INT", "STRING")
    assert math_nodes.NH_RandomChoice.RETURN_NAMES == ("result", "picked_index", "probabilities")


def test_random_choice_description_says_that_connected_inputs_count_even_without_a_value():
    description = math_nodes.NH_RandomChoice.DESCRIPTION
    assert "Connected inputs count even if they produce no value" in description
    assert "Only the picked input is executed" in description


def test_random_choice_check_lazy_status_values():
    node = math_nodes.NH_RandomChoice()
    unevaluated = {f"input_{i}": None for i in range(6)}
    assert node.check_lazy_status(1, "0,0,5,0,0,0", **unevaluated) == ["input_2"]  # only slot 2 has any weight
    assert node.check_lazy_status(1, "0,0,5,0,0,0", **{**unevaluated, "input_2": "ready"}) == []
    assert node.check_lazy_status(1, "1,1,1,1,1,1") == []  # nothing connected
    only_two = {"input_1": None, "input_4": None}
    for seed in range(20):
        picked = old_random_choice(seed, "1,1,1,1,1,1", input_1="a", input_4="b")[1]
        assert node.check_lazy_status(seed, "1,1,1,1,1,1", **only_two) == [f"input_{picked}"]


def test_random_choice_runs_only_the_picked_input():
    node = math_nodes.NH_RandomChoice()
    for slot in range(6):
        weights = ",".join("5" if i == slot else "0" for i in range(6))
        connected = {f"input_{i}": f"value-{i}" for i in range(6)}
        outputs, requested = run_lazy(node, "choose", {"seed": 11, "weights": weights}, connected)
        assert outputs[:2] == (f"value-{slot}", slot)
        assert requested == [f"input_{slot}"]


def test_random_choice_with_nothing_connected_returns_none():
    outputs, requested = run_lazy(math_nodes.NH_RandomChoice(), "choose", {"seed": 3, "weights": "1,1"}, {})
    assert outputs == (None, -1, "") and requested == []


def test_random_choice_makes_the_same_pick_as_before_for_every_seed_weights_and_connection():
    node = math_nodes.NH_RandomChoice()
    weight_sets = ["1,1,1,1,1,1", "3,0,1,2,5,0", "0,0,0,0,0,0", "1,1,10", "x,2,,4", "", "2", "0.5, 0.25 ,0.25,9,9,9,9,9"]
    seeds = [0, 1, 2, 3, 7, 42, 99, 12345, 2**32 - 1] + list(range(10, 40))
    checked = 0
    for subset in range(1, 1 << 6):
        slots = [i for i in range(6) if subset >> i & 1]
        values = {f"input_{i}": f"value-{i}" for i in slots}
        for weights, seed in itertools.product(weight_sets, seeds):
            outputs, requested = run_lazy(node, "choose", {"seed": seed, "weights": weights}, values)
            expected = old_random_choice(seed, weights, **values)
            assert outputs == expected, (slots, weights, seed)
            assert requested == [f"input_{expected[1]}"], (slots, weights, seed)
            checked += 1
    assert checked == 63 * len(weight_sets) * len(seeds)


def test_random_choice_empty_picked_value_is_returned_as_is():
    node = math_nodes.NH_RandomChoice()
    outputs, requested = run_lazy(node, "choose", {"seed": 1, "weights": "0,1"}, {"input_0": "a", "input_1": None})
    assert outputs == (None, 1, "0.0%, 100.0%") and requested == ["input_1"]


# --- Mask Morphology ----------------------------------------------------------------------------------------

def process(mask, horizontal=0, vertical=0, fill_holes=False, blur=0.0):
    return morphology.NH_MaskMorphology().process(mask, horizontal, vertical, fill_holes, blur)[0]


def blobs(shape, seed, sigma=3.0):
    """Binary mask made of blobs a few pixels wide, so dilation and erosion both have something to do."""
    rng = np.random.default_rng(seed)
    noise = gaussian_filter(rng.random(shape), sigma=sigma)
    return (noise > np.median(noise)).astype(np.float32)


def old_expand(mask_np, horizontal_expand, vertical_expand):
    """The scipy code of Mask Morphology before the torch rewrite (expand steps only)."""
    m = mask_np
    if horizontal_expand != 0:
        op = binary_dilation if horizontal_expand > 0 else binary_erosion
        m = op(m, structure=np.ones((1, abs(horizontal_expand)))).astype(m.dtype)
    if vertical_expand != 0:
        op = binary_dilation if vertical_expand > 0 else binary_erosion
        m = op(m, structure=np.ones((abs(vertical_expand), 1))).astype(m.dtype)
    return m


def test_mask_morphology_interface_is_unchanged():
    spec = morphology.NH_MaskMorphology.INPUT_TYPES()
    assert list(spec["required"]) == ["mask", "horizontal_expand", "vertical_expand", "fill_holes", "blur_radius"]
    assert spec["required"]["horizontal_expand"] == ("INT", {"default": 0, "min": -512, "max": 512, "step": 1})
    assert spec["required"]["vertical_expand"] == ("INT", {"default": 0, "min": -512, "max": 512, "step": 1})
    assert spec["required"]["blur_radius"] == ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1})
    assert morphology.NH_MaskMorphology.RETURN_TYPES == ("MASK",)
    assert morphology.NODE_CLASS_MAPPINGS == {"NH_MaskAdvancedMorphology": morphology.NH_MaskMorphology}
    assert morphology.NODE_DISPLAY_NAME_MAPPINGS == {"NH_MaskAdvancedMorphology": "Mask Morphology (NH)"}


@pytest.mark.parametrize("horizontal,vertical", list(itertools.product([-9, -5, -3, 0, 3, 5, 9], repeat=2)))
def test_binary_mask_with_odd_kernels_equals_the_old_result_away_from_the_borders(horizontal, vertical):
    for seed in range(3):
        mask_np = blobs((72, 80), seed)  # not square: catches a swapped axis
        expected = old_expand(mask_np, horizontal, vertical)
        got = process(torch.from_numpy(mask_np)[None], horizontal, vertical)[0].numpy()
        assert got.dtype == np.float32 and got.shape == expected.shape
        rh, rv = abs(horizontal) // 2, abs(vertical) // 2
        interior = (slice(rv, 72 - rv), slice(rh, 80 - rh))
        assert np.array_equal(got[interior], expected[interior]), (horizontal, vertical, seed)
        # The old code treated the outside as empty, so it could only eat more. Growing is identical everywhere.
        assert np.all(got >= expected)
        if horizontal >= 0 and vertical >= 0:
            assert np.array_equal(got, expected)


def test_soft_masks_keep_their_gradient_instead_of_being_binarised():
    soft = torch.zeros(1, 9, 9)
    soft[0, 4, 4] = 0.3
    grown = process(soft, 5, 0)
    assert torch.equal(grown[0, 4], torch.tensor([0, 0, 0.3, 0.3, 0.3, 0.3, 0.3, 0, 0]))  # old code gave 1.0 here
    assert float(grown.sum()) == pytest.approx(5 * 0.3)

    ramp = torch.linspace(0, 1, 11).repeat(5, 1)[None]  # values rise from 0 to 1 along the width
    assert torch.allclose(process(ramp, 3, 0)[0, 0, :-1], ramp[0, 0, 1:])  # max over 3 = the next pixel
    assert torch.allclose(process(ramp, -3, 0)[0, 0, 1:], ramp[0, 0, :-1])  # min over 3 = the previous pixel

    plateau = torch.full((1, 7, 7), 0.8)
    plateau[0, 3, 3] = 0.2
    eroded = process(plateau, -3, -3)
    assert float(eroded[0, 3, 3]) == pytest.approx(0.2) and float(eroded[0, 0, 0]) == pytest.approx(0.8)
    assert bool(((eroded > 0.19) & (eroded < 0.81)).all())  # nothing was pushed to 0 or 1


def test_erosion_does_not_eat_a_mask_from_the_image_border():
    full = torch.ones(1, 12, 12)
    assert torch.equal(process(full, -9, -9), full)  # the old code emptied the border band of 4 px

    left = torch.zeros(1, 6, 14)
    left[0, :, :8] = 1.0  # the mask touches the left edge, the empty part starts at column 8
    eroded = process(left, -5, 0)[0]
    assert torch.equal(eroded[:, :6], torch.ones(6, 6))  # columns 0..5 stay (the old code lost columns 0 and 1)
    assert float(eroded[:, 6:].sum()) == 0.0  # the side facing the empty area still shrinks by 2

    top = torch.zeros(1, 14, 6)
    top[0, :8, :] = 1.0
    assert torch.equal(process(top, 0, -5)[0, :6], torch.ones(6, 6))
    assert float(process(top, 0, -5)[0, 6:].sum()) == 0.0


def test_dilation_stops_at_the_image_border_and_does_not_wrap_around():
    mask = torch.zeros(1, 5, 10)
    mask[0, 2, 0] = 1.0
    grown = process(mask, 5, 5)[0]
    assert torch.equal(grown[:, :3], torch.ones(5, 3))  # radius 2 around (2, 0): rows 0..4, columns 0..2
    assert float(grown[:, 3:].sum()) == 0.0  # nothing wrapped to the far side


@pytest.mark.parametrize(
    "expand,columns",
    [
        (1, (5, 5)),  # kernel length 1: nothing changes
        (2, (4, 6)),  # even sizes are rounded up to the next odd length: 2 -> 3
        (3, (4, 6)),
        (4, (3, 7)),  # 4 -> 5
        (5, (3, 7)),
        (10, (0, 10)),  # 10 -> 11 (radius 5), clipped by the 11 px wide image
    ],
)
def test_even_kernel_lengths_are_rounded_up_so_growing_stays_centred(expand, columns):
    mask = torch.zeros(1, 3, 11)
    mask[0, :, 5] = 1.0
    grown = process(mask, expand, 0)[0]
    first, last = columns
    assert torch.equal(grown[1], torch.tensor([1.0 if first <= x <= last else 0.0 for x in range(11)]))


@pytest.mark.parametrize("expand,columns", [(-2, (2, 8)), (-3, (2, 8)), (-4, (3, 7)), (-5, (3, 7)), (-6, (4, 6))])
def test_even_kernel_lengths_are_rounded_up_so_shrinking_stays_centred(expand, columns):
    mask = torch.zeros(1, 3, 11)
    mask[0, :, 1:10] = 1.0  # run of 9 ones, columns 1..9
    shrunk = process(mask, expand, 0)[0]
    first, last = columns
    assert torch.equal(shrunk[1], torch.tensor([1.0 if first <= x <= last else 0.0 for x in range(11)]))


def test_horizontal_expand_acts_on_columns_and_vertical_expand_on_rows_in_that_order():
    mask = torch.zeros(1, 9, 9)
    mask[0, 4, 4] = 1.0
    wide = process(mask, 5, 0)[0]
    assert torch.equal(wide.nonzero()[:, 0].unique(), torch.tensor([4])) and int(wide.sum()) == 5
    tall = process(mask, 0, 5)[0]
    assert torch.equal(tall.nonzero()[:, 1].unique(), torch.tensor([4])) and int(tall.sum()) == 5
    assert int(process(mask, 5, 5).sum()) == 25


def test_horizontal_step_runs_before_the_vertical_step_like_before():
    diagonal = torch.zeros(1, 9, 9)
    for step in (3, 4, 5):
        diagonal[0, step, step] = 1.0
    # Grow sideways first: the three pixels overlap in column 4 only, so one pixel survives the vertical erosion.
    # (Eroding first would leave nothing.)
    result = process(diagonal, 3, -3)[0]
    assert int(result.sum()) == 1 and result[4, 4] == 1.0
    expected = old_expand(diagonal[0].numpy(), 3, -3)
    assert np.array_equal(result.numpy(), expected)


@pytest.mark.parametrize("size", [3, 8, 31, 64, 255, 513])
def test_any_kernel_size_matches_an_independent_window_filter(size):
    rng = np.random.default_rng(size)
    soft = rng.random((2, 37, 41), dtype=np.float32)
    radius_size = 2 * (size // 2) + 1  # odd length actually used (even lengths round up)
    for sign, window_filter in ((1, maximum_filter1d), (-1, minimum_filter1d)):
        horizontal_only = process(torch.from_numpy(soft), sign * size, 0).numpy()
        expected = np.stack([window_filter(m, size=radius_size, axis=1, mode="nearest") for m in soft])
        assert np.array_equal(horizontal_only, expected), (size, sign)

        vertical_only = process(torch.from_numpy(soft), 0, sign * size).numpy()
        expected = np.stack([window_filter(m, size=radius_size, axis=0, mode="nearest") for m in soft])
        assert np.array_equal(vertical_only, expected), (size, sign)

        both = process(torch.from_numpy(soft), sign * size, sign * size).numpy()
        expected = np.stack([window_filter(window_filter(m, size=radius_size, axis=1, mode="nearest"),
                                           size=radius_size, axis=0, mode="nearest") for m in soft])
        assert np.array_equal(both, expected), (size, sign)


def test_a_kernel_wider_than_the_image_gives_the_global_maximum_or_minimum():
    soft = torch.rand(1, 6, 9)
    assert torch.equal(process(soft, 512, 0)[0], soft[0].amax(dim=1, keepdim=True).expand(6, 9))
    assert torch.equal(process(soft, 0, -512)[0], soft[0].amin(dim=0, keepdim=True).expand(6, 9))
    tiny = torch.rand(1, 1, 1)
    assert torch.equal(process(tiny, 512, -512), tiny)


def test_batch_items_are_independent_and_the_input_is_not_modified():
    batch = torch.from_numpy(np.stack([blobs((40, 50), seed) for seed in range(4)]))
    original = batch.clone()
    together = process(batch, 5, -3)
    assert together.shape == batch.shape and together.dtype == torch.float32
    assert torch.equal(batch, original)
    for index in range(4):
        assert torch.equal(together[index], process(batch[index : index + 1], 5, -3)[0])


@pytest.mark.parametrize("budget", [1, 42, 100, 126, 10**9])
def test_long_batches_are_processed_in_chunks_with_the_same_result(monkeypatch, budget):
    batch = torch.from_numpy(np.stack([blobs((6, 7), seed) for seed in range(5)]))  # 42 pixels per mask
    expected = process(batch, 3, -3)
    monkeypatch.setattr(morphology, "_CHUNK_PIXELS", budget)  # 1 and 42: one mask per chunk; 100: 2+2+1; 126: 3+2
    assert torch.equal(process(batch, 3, -3), expected)
    assert torch.equal(process(batch, 0, 0), batch) and process(batch).data_ptr() != batch.data_ptr()
    filled = process(batch, 3, 3, True, 0.8)
    monkeypatch.setattr(morphology, "_CHUNK_PIXELS", 1 << 24)
    assert torch.equal(filled, process(batch, 3, 3, True, 0.8))


def test_unchanged_mask_comes_back_as_a_copy():
    mask = torch.rand(2, 5, 6)
    out = process(mask)
    assert torch.equal(out, mask) and out.data_ptr() != mask.data_ptr()
    out += 1.0
    assert not torch.equal(out, mask)  # a node downstream may modify its input in place
    # Kernel length 1 and an image only one pixel wide have nothing to do either, and still return a copy.
    for horizontal, vertical in ((1, -1), (-1, 1)):
        same = process(mask, horizontal, vertical)
        assert torch.equal(same, mask) and same.data_ptr() != mask.data_ptr()
    thin = torch.rand(1, 5, 1)
    assert process(thin, 9, 0).data_ptr() != thin.data_ptr() and torch.equal(process(thin, -9, 0), thin)


def test_two_dimensional_masks_are_accepted_and_other_shapes_are_rejected_clearly():
    flat = torch.zeros(7, 7)
    flat[3, 3] = 1.0
    out = process(flat, 3, 3)
    assert out.shape == (1, 7, 7) and int(out.sum()) == 9
    with pytest.raises(ValueError, match="MASK"):
        process(torch.zeros(1, 1, 4, 4), 3, 3)
    with pytest.raises(ValueError, match="MASK"):
        process(torch.zeros(5), 3, 3)


def test_empty_batch_is_returned_empty():
    out = process(torch.zeros(0, 8, 8), 5, -5, True, 1.0)
    assert out.shape == (0, 8, 8)


@pytest.mark.parametrize("shape", [(1, 0, 8), (1, 8, 0), (3, 0, 0), (0, 8)])
@pytest.mark.parametrize(
    "settings",
    [
        (0, 0),
        (5, 0),
        (0, 5),
        (-5, 0),
        (0, -5),
        (5, -5, False, 0.0),
        (9, 9, True, 0.0),
        (0, 0, False, 1.5),
        (3, 3, True, 2.0),
    ],
)
def test_a_mask_without_pixels_comes_back_unchanged_like_before(shape, settings):
    # Zero height or width: there is nothing to grow, fill or blur. The scipy version passed such a mask through
    # whatever the settings; replicate padding cannot pad it (it raises), so the node must not try.
    mask = torch.zeros(shape)
    out = process(mask, *settings)
    assert out is not mask  # a copy, like every other unchanged result
    assert out.shape == ((1, *shape) if len(shape) == 2 else shape)  # a 2-D mask is a batch of one
    assert out.dtype == torch.float32 and out.numel() == 0


@pytest.mark.parametrize(
    "dtype,expected",
    [(torch.float32, torch.float32), (torch.float64, torch.float64), (torch.float16, torch.float16), (torch.bool, torch.float32)],
)
def test_a_mask_without_pixels_follows_the_same_dtype_rule_as_any_other(dtype, expected):
    out = process(torch.zeros(2, 0, 8, dtype=dtype), 5, 5, True, 1.0)
    assert out.shape == (2, 0, 8) and out.dtype == expected


def test_non_float_and_float64_masks_are_handled():
    as_bool = torch.zeros(1, 5, 5, dtype=torch.bool)
    as_bool[0, 2, 2] = True
    out = process(as_bool, 3, 3)
    assert out.dtype == torch.float32 and int(out.sum()) == 9
    as_double = torch.zeros(1, 5, 5, dtype=torch.float64)
    as_double[0, 2, 2] = 1.0
    out = process(as_double, 3, 3)
    assert out.dtype == torch.float64 and int(out.sum()) == 9


def test_fill_holes_and_blur_still_use_the_same_scipy_calls():
    ring = np.zeros((30, 30), np.float32)
    yy, xx = np.mgrid[:30, :30]
    ring[(np.hypot(yy - 15, xx - 15) < 10) & (np.hypot(yy - 15, xx - 15) > 4)] = 1.0
    ring_t = torch.from_numpy(ring)[None]

    filled = process(ring_t, fill_holes=True)[0].numpy()
    assert np.array_equal(filled, binary_fill_holes(ring > 0.5).astype(np.float32))
    assert filled[15, 15] == 1.0 and ring[15, 15] == 0.0

    blurred = process(ring_t, blur=2.5)[0].numpy()
    assert np.array_equal(blurred, gaussian_filter(ring, sigma=2.5))

    # expand -> fill holes -> blur, in that order
    both = process(ring_t, 3, 3, True, 1.5)[0].numpy()
    grown = process(ring_t, 3, 3)[0].numpy()
    assert np.array_equal(both, gaussian_filter(binary_fill_holes(grown > 0.5).astype(np.float32), sigma=1.5))


def test_fill_holes_binarises_a_soft_mask_at_one_half_like_before():
    soft = torch.zeros(1, 9, 9)
    soft[0, 2:7, 2:7] = 0.9
    soft[0, 4, 4] = 0.0  # hole
    soft[0, 2, 2] = 0.4  # below the 0.5 threshold: not part of the mask
    out = process(soft, fill_holes=True)[0]
    assert out[4, 4] == 1.0 and out[2, 2] == 0.0 and out[3, 3] == 1.0
    assert set(out.unique().tolist()) == {0.0, 1.0}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_morphology_runs_on_the_tensor_device_and_matches_the_cpu_result():
    mask = torch.from_numpy(np.stack([blobs((64, 80), seed) for seed in range(2)]))
    mask[0, 10:20, 10:20] *= 0.5  # keep some soft values
    cpu = process(mask, 15, -9)
    gpu = process(mask.cuda(), 15, -9)
    assert gpu.device.type == "cuda" and gpu.dtype == torch.float32
    assert torch.equal(gpu.cpu(), cpu)
    gpu_filled = process(mask.cuda(), 5, 5, True, 1.0)
    assert gpu_filled.device.type == "cuda"
    assert torch.allclose(gpu_filled.cpu(), process(mask, 5, 5, True, 1.0))
