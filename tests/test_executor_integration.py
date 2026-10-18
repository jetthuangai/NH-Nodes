"""Executor-level tests: ComfyUI's real PromptExecutor runs small graphs of NH nodes in-process.

The unit tests in the other files emulate execution.py. These run the real thing (prompt validation, lazy input
handling, ExecutionBlocker propagation, the output cache) with a stub in place of the web server, on the CPU, with no
GPU, model or network. ``Stub...`` nodes stand in for whatever feeds and consumes the NH nodes: they log what ran and
what reached the output nodes.
"""

import asyncio
import copy
import gc
import importlib
import os
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import comfy.cli_args
import comfy.sd
import execution
import nodes as comfy_nodes

logic_nodes = importlib.import_module("NH-Nodes.nodes.logic.logic_nodes")
math_nodes = importlib.import_module("NH-Nodes.nodes.logic.math_nodes")
folder_loaders = importlib.import_module("NH-Nodes.nodes.image.folder_loaders")
loader_nodes = importlib.import_module("NH-Nodes.nodes.loaders.loader_index_nodes")

# The node only reads the names in the source path, never the folder, so the tests use one that does not exist.
SOURCES = Path(os.path.abspath(os.sep)) / "nh_nodes_test_source_photos"

LOG = SimpleNamespace(ran=[], sunk={})  # reset for every test by the registered_nodes fixture


def _describe(value):
    """What the sink keeps of a value: never the value itself, so a sink cannot keep a model alive."""
    if isinstance(value, torch.Tensor):
        return (tuple(value.shape), round(float(value.mean()), 3))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return type(value).__name__


# --- stand-ins for the nodes around the NH nodes ----------------------------------------------------------------

class StubImage:
    """Makes a small constant image and logs its tag when it runs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"tag": ("STRING", {"default": "image"}), "value": ("FLOAT", {"default": 0.5})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    def run(self, tag, value):
        LOG.ran.append(tag)
        return (torch.full((1, 4, 4, 3), value),)


class StubNothing:
    """An upstream node that has no value to give (it returns None)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"tag": ("STRING", {"default": "nothing"})}}

    RETURN_TYPES = ("*",)
    FUNCTION = "run"

    def run(self, tag):
        LOG.ran.append(tag)
        return (None,)


class StubEcho:
    """Hands an image on: any node that needs the image of another one."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    def run(self, image):
        LOG.ran.append("echo")
        return (image,)


class StubSink:
    """Output node: records what reached it under its name."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"name": ("STRING", {"default": "sink"}), "value": ("*",)}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"

    def run(self, name, value):
        LOG.sunk[name] = _describe(value)
        return {}


STUBS = {
    "StubImage": StubImage,
    "StubNothing": StubNothing,
    "StubEcho": StubEcho,
    "StubSink": StubSink,
    "NH_IfElse": logic_nodes.NH_IfElse,
    "NH_SwitchN": logic_nodes.NH_SwitchN,
    "NH_AnySwitchBoolean": logic_nodes.NH_AnySwitchBoolean,
    "NH_RandomChoice": math_nodes.NH_RandomChoice,
    "NH_MathEval": math_nodes.NH_MathEval,
    "NH_LoadImagesMatching": folder_loaders.NH_LoadImagesMatching,
    "NH_DiffusionModelIndex": loader_nodes.NH_DiffusionModelIndex,
}


@pytest.fixture(autouse=True)
def registered_nodes(monkeypatch):
    for name, cls in STUBS.items():
        monkeypatch.setitem(comfy_nodes.NODE_CLASS_MAPPINGS, name, cls)
    LOG.ran.clear()
    LOG.sunk.clear()


# --- running a graph ----------------------------------------------------------------------------------------------

class _Server:
    """The part of PromptServer that PromptExecutor uses: it sends events and has a client id."""

    client_id = None
    last_node_id = None

    def __init__(self):
        self.messages = []

    def send_sync(self, event, data, sid=None):
        self.messages.append((event, data))


def new_executor(cache_type=execution.CacheType.CLASSIC, lru=0, ram=0.0, ram_inactive=0.0):
    """An executor with one of ComfyUI's caches (ram and ram_inactive are the RAM-pressure headrooms, in GB)."""
    cache_args = {"lru": lru, "ram": ram, "ram_inactive": ram_inactive}
    return execution.PromptExecutor(_Server(), cache_type=cache_type, cache_args=cache_args)


def run(prompt, executor=None):
    """Validate `prompt` the way the server does, execute it, and report what happened."""
    prompt = copy.deepcopy(prompt)  # a queued prompt is new data every time; the executor writes into it
    valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt("test", prompt, None))
    assert valid and not node_errors, (error, node_errors)

    executor = executor or new_executor()
    executor.server.messages.clear()
    LOG.ran.clear()
    LOG.sunk.clear()
    executor.execute(prompt, "test", {"client_id": "test"}, outputs)

    errors = [data for event, data in executor.server.messages if event == "execution_error"]
    return SimpleNamespace(
        ran=list(LOG.ran),
        sunk=dict(LOG.sunk),
        success=executor.success,
        blocked=[data["exception_message"] for data in errors if data["exception_type"] == "ExecutionBlocked"],
        failed=[data["exception_message"] for data in errors if data["exception_type"] != "ExecutionBlocked"],
    )


def sink(name, link):
    return {"class_type": "StubSink", "inputs": {"name": name, "value": link}}


def image(tag, value=0.5):
    return {"class_type": "StubImage", "inputs": {"tag": tag, "value": value}}


def nothing(tag="nothing"):
    return {"class_type": "StubNothing", "inputs": {"tag": tag}}


def math_eval(expression, **values):
    return {"class_type": "NH_MathEval", "inputs": {"expression": expression, **values}}


GOOD = math_eval("a + b", a=1.0, b=2.0)  # 3.0
BAD = math_eval("1 +")  # a syntax error: the prompt fails if this node ever runs


# --- 1. lazy branches: the branch that is not taken never runs -------------------------------------------------

def test_the_invalid_math_node_does_fail_when_it_runs():
    result = run({"1": BAD, "2": sink("result", ["1", 0])})
    assert result.success is False
    assert len(result.failed) == 1 and "Math Eval" in result.failed[0]
    assert result.sunk == {}


@pytest.mark.parametrize("condition", [True, False])
def test_if_else_does_not_run_the_unchosen_branch(condition):
    taken, dropped = ("if_true", "if_false") if condition else ("if_false", "if_true")
    prompt = {
        "1": GOOD,
        "2": BAD,
        "3": {"class_type": "NH_IfElse", "inputs": {"condition": condition, taken: ["1", 0], dropped: ["2", 0]}},
        "4": sink("result", ["3", 0]),
        "5": sink("condition", ["3", 1]),
    }
    result = run(prompt)
    assert result.success and result.failed == [] and result.blocked == []
    assert result.sunk == {"result": 3.0, "condition": condition}


@pytest.mark.parametrize("condition", [True, False])
def test_any_switch_does_not_run_the_unchosen_input(condition):
    taken, dropped = ("input_true", "input_false") if condition else ("input_false", "input_true")
    prompt = {
        "1": GOOD,
        "2": BAD,
        "3": {"class_type": "NH_AnySwitchBoolean", "inputs": {"condition": condition, taken: ["1", 0], dropped: ["2", 0]}},
        "4": sink("result", ["3", 0]),
    }
    result = run(prompt)
    assert result.success and result.failed == []
    assert result.sunk == {"result": 3.0}


def test_switch_n_does_not_run_the_inputs_that_were_not_picked():
    prompt = {
        "1": BAD,
        "2": GOOD,
        "3": BAD,
        "4": {"class_type": "NH_SwitchN", "inputs": {"index": 1, "input_0": ["1", 0], "input_1": ["2", 0], "input_2": ["3", 0]}},
        "5": sink("result", ["4", 0]),
        "6": sink("count", ["4", 1]),
    }
    result = run(prompt)
    assert result.success and result.failed == []
    assert result.sunk == {"result": 3.0, "count": 3}


def test_random_choice_does_not_run_the_inputs_that_were_not_picked():
    prompt = {
        "1": BAD,
        "2": GOOD,
        "3": BAD,
        "4": {"class_type": "NH_RandomChoice", "inputs": {"seed": 5, "weights": "0,1,0", "input_0": ["1", 0], "input_1": ["2", 0], "input_2": ["3", 0]}},
        "5": sink("result", ["4", 0]),
        "6": sink("picked", ["4", 1]),
    }
    result = run(prompt)
    assert result.success and result.failed == []
    assert result.sunk == {"result": 3.0, "picked": 1}


def test_a_connected_input_that_produces_nothing_still_counts_as_connected():
    # What the DESCRIPTION of Switch N and Random Choice says: the picked slot is the connected one, value or not.
    switch = {
        "1": image("other"),
        "2": nothing("empty"),
        "3": {"class_type": "NH_SwitchN", "inputs": {"index": 1, "input_0": ["1", 0], "input_1": ["2", 0]}},
        "4": sink("result", ["3", 0]),
        "5": sink("count", ["3", 1]),
    }
    result = run(switch)
    assert result.success and result.ran == ["empty"]  # the other slot did not run
    assert result.sunk == {"result": None, "count": 2}

    choice = {
        "1": image("other"),
        "2": nothing("empty"),
        "3": {"class_type": "NH_RandomChoice", "inputs": {"seed": 3, "weights": "0,1", "input_0": ["1", 0], "input_1": ["2", 0]}},
        "4": sink("result", ["3", 0]),
        "5": sink("picked", ["3", 1]),
    }
    result = run(choice)
    assert result.success and result.ran == ["empty"]
    assert result.sunk == {"result": None, "picked": 1}


# --- 2. Load Images Matching: a miss, with and without fallback_images ------------------------------------------

@pytest.fixture
def matching_tree(tmp_path):
    """<root>/Nu/Anh holds one white 8x8 picture; the source folder is called .../Nu/Anh, so it is a hit."""
    root = tmp_path / "matching"
    (root / "Nu" / "Anh").mkdir(parents=True)
    Image.new("RGB", (8, 8), (255, 255, 255)).save(root / "Nu" / "Anh" / "01.png")
    return SimpleNamespace(root=root, source=SOURCES / "Nu" / "Anh")


def matching(tree, **inputs):
    settings = {
        "source_folder_path": str(tree.source), "matching_root_path": str(tree.root), "match_text": "",
        "match_mode": "contains", "recursive": True, "index": 0, "step": 1, "repeat_count": 1,
        "context_folder_names": "Nam\nNu", "source_file_names": "",
    }
    settings.update(inputs)
    return {"class_type": "NH_LoadImagesMatching", "inputs": settings}


# Without match_text the source folder's name ("Anh") is looked for, and it is there: a hit. "Zed" is not: a miss.
MISS = {"match_text": "Zed"}


def test_a_hit_loads_the_pictures_and_never_runs_the_fallback(matching_tree):
    prompt = {
        "1": matching(matching_tree, fallback_images=["2", 0]),
        "2": image("fallback", 0.25),
        "3": sink("images", ["1", 0]),
        "4": sink("found", ["1", 4]),
    }
    result = run(prompt)
    assert result.success and result.blocked == [] and result.failed == []
    assert result.sunk == {"images": ((1, 8, 8, 3), 1.0), "found": True}
    assert result.ran == []  # fallback_images is lazy and nothing asked for it


def test_a_miss_with_fallback_images_delivers_the_fallback(matching_tree):
    prompt = {
        "1": matching(matching_tree, fallback_images=["2", 0], **MISS),
        "2": image("fallback", 0.25),
        "3": sink("images", ["1", 0]),
        "4": sink("found", ["1", 4]),
        "5": sink("matched_text", ["1", 3]),
        "6": sink("filename", ["1", 1]),
    }
    result = run(prompt)
    assert result.success and result.blocked == [] and result.failed == []
    assert result.sunk == {
        "images": ((1, 4, 4, 3), 0.25),  # the fallback, as it is
        "found": False,
        "matched_text": "Zed",
        "filename": "",
    }
    assert result.ran == ["fallback"]


def test_a_miss_with_a_fallback_that_an_earlier_prompt_cached(matching_tree):
    executor = new_executor()
    prompt = {
        "1": matching(matching_tree, fallback_images=["2", 0], **MISS),
        "2": image("fallback", 0.25),
        "3": sink("images", ["1", 0]),
    }
    first = run(prompt, executor)
    assert first.ran == ["fallback"] and first.sunk == {"images": ((1, 4, 4, 3), 0.25)}
    second = run(prompt, executor)  # the miss is never remembered, the fallback comes from the cache
    assert second.success and second.ran == []
    assert second.sunk == {"images": ((1, 4, 4, 3), 0.25)}


def test_a_miss_without_fallback_images_blocks_only_its_own_consumers(matching_tree):
    prompt = {
        "1": matching(matching_tree, **MISS),
        "2": {"class_type": "StubEcho", "inputs": {"image": ["1", 0]}},
        "3": sink("echoed", ["2", 0]),
        "4": image("independent", 0.75),
        "5": sink("independent", ["4", 0]),
        "6": sink("found", ["1", 4]),
    }
    result = run(prompt)
    assert result.success and result.failed == []
    assert result.ran == ["independent"]  # the node that needed the images never ran
    assert result.sunk == {"independent": ((1, 4, 4, 3), 0.75), "found": False}  # nothing reached the echo's sink
    assert len(result.blocked) == 1  # the reason is reported once
    assert result.blocked[0].startswith("Execution Blocked: No matching folder found for 'Zed'")


def test_routing_the_held_back_images_through_if_else_does_not_reach_the_fallback(matching_tree):
    # The reason fallback_images exists: a node with a held-back input is blocked before it can choose, whichever
    # input it would have picked, so the other branch of the If/Else is never reached.
    prompt = {
        "1": matching(matching_tree, **MISS),
        "2": image("fallback", 0.25),
        "3": {"class_type": "NH_IfElse", "inputs": {"condition": ["1", 4], "if_true": ["1", 0], "if_false": ["2", 0]}},
        "4": sink("result", ["3", 0]),
    }
    result = run(prompt)
    assert result.sunk == {} and result.ran == []
    assert len(result.blocked) == 1


def test_fallback_images_does_the_job_the_if_else_cannot(matching_tree):
    prompt = {
        "1": matching(matching_tree, fallback_images=["2", 0], **MISS),
        "2": image("fallback", 0.25),
        "3": sink("result", ["1", 0]),
    }
    result = run(prompt)
    assert result.blocked == [] and result.sunk == {"result": ((1, 4, 4, 3), 0.25)}


def test_a_setting_that_cannot_work_fails_even_with_fallback_images(matching_tree):
    prompt = {
        "1": matching(matching_tree, matching_root_path=str(matching_tree.root / "missing"), fallback_images=["2", 0]),
        "2": image("fallback", 0.25),
        "3": sink("images", ["1", 0]),
    }
    result = run(prompt)
    assert result.success is False and result.ran == []
    assert len(result.failed) == 1 and "Matching root folder does not exist" in result.failed[0]


# --- 3. Indexed loaders under ComfyUI's own output caches -------------------------------------------------------

MODEL_FILES = ["a.safetensors", "b.safetensors", "c.safetensors", "d.safetensors"]


@pytest.fixture
def diffusion_models(tmp_path, monkeypatch):
    """The indexed diffusion loader with only the file reading faked: counts loads and watches the loaded objects."""
    loads = SimpleNamespace(count=0, watchers=[])

    class Loaded:
        """A plain object, like the ModelPatcher the real loader returns."""

    def load_diffusion_model(path, model_options=None):
        loads.count += 1
        model = Loaded()
        loads.watchers.append(weakref.ref(model))
        return model

    def full_path(folder, name):
        path = tmp_path / name
        if not path.exists():  # written once: the node tells a replaced file by its size and time
            path.write_bytes(b"model")
        return str(path)

    monkeypatch.setattr(comfy.sd, "load_diffusion_model", load_diffusion_model)
    monkeypatch.setattr(loader_nodes.folder_paths, "get_full_path_or_raise", full_path)
    monkeypatch.setattr(loader_nodes.folder_paths, "get_filename_list", lambda folder: list(MODEL_FILES))
    loads.alive = lambda: sum(watcher() is not None for watcher in loads.watchers)
    return loads


def edit_unselected_slot_three_times(executor):
    """Queue the same loader prompt with a slot that nothing selected changed each time: the node runs again."""
    for unselected in MODEL_FILES[1:]:
        inputs = {
            "select_mode": "index", "index": 1, "model_count": 2, "weight_dtype": "default",
            "diffusion_model_1": "a.safetensors", "diffusion_model_2": unselected,
        }
        prompt = {"1": {"class_type": "NH_DiffusionModelIndex", "inputs": inputs}, "2": sink("model", ["1", 0])}
        result = run(prompt, executor)
        assert result.success and result.sunk == {"model": "Loaded"}
        gc.collect()


def test_default_cache_the_loader_reuses_its_model_when_an_unselected_slot_changes(diffusion_models, monkeypatch):
    monkeypatch.setattr(comfy.cli_args.args, "cache_classic", False)
    executor = new_executor(execution.CacheType.RAM_PRESSURE, ram=0.001, ram_inactive=0.001)  # plenty of free RAM
    edit_unselected_slot_three_times(executor)
    assert diffusion_models.count == 1 and diffusion_models.alive() == 1


def test_default_cache_memory_pressure_can_free_the_loaded_model(diffusion_models, monkeypatch):
    monkeypatch.setattr(comfy.cli_args.args, "cache_classic", False)
    # A headroom no machine has: the RAM-pressure cache is always short of memory and evicts what it holds.
    executor = new_executor(execution.CacheType.RAM_PRESSURE, ram=1e6, ram_inactive=1e6)
    edit_unselected_slot_three_times(executor)
    assert diffusion_models.alive() == 0  # nothing of the node's kept the evicted models in memory


def test_cache_classic_the_loader_holds_the_model_the_classic_cache_dropped(diffusion_models, monkeypatch):
    monkeypatch.setattr(comfy.cli_args.args, "cache_classic", True)  # what --cache-classic sets
    executor = new_executor(execution.CacheType.CLASSIC)
    edit_unselected_slot_three_times(executor)
    assert diffusion_models.count == 1 and diffusion_models.alive() == 1


def test_lru_cache_the_loader_reuses_its_model_like_the_default_cache(diffusion_models, monkeypatch):
    monkeypatch.setattr(comfy.cli_args.args, "cache_classic", False)
    executor = new_executor(execution.CacheType.LRU, lru=100)
    edit_unselected_slot_three_times(executor)
    assert diffusion_models.count == 1 and diffusion_models.alive() == 1
