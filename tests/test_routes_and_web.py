"""HTTP routes (core/folder_routes.py, core/large_preview_routes.py) and the frontend extensions (web/*.js).

The routes are served from a throw-away aiohttp app through each module's ``register_routes``; no ComfyUI server runs.
The extensions run in Node against a fake ComfyUI app and fake nodes (two kinds of widget: the one of the current
frontend, which has ``hidden``, and a plain one from an older frontend, which has not).
"""

import asyncio
import importlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest
import torch
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import folder_paths

pack = importlib.import_module("NH-Nodes")
folder_routes = importlib.import_module("NH-Nodes.core.folder_routes")
preview_routes = importlib.import_module("NH-Nodes.core.large_preview_routes")
preview_cache = importlib.import_module("NH-Nodes.core.large_preview_cache")
resolution_data = importlib.import_module("NH-Nodes.core.resolution_data")
picker = importlib.import_module("NH-Nodes.nodes.resolution.smart_resolution_picker")

PACK_DIR = Path(__file__).resolve().parent.parent
WEB_FILES = sorted((PACK_DIR / "web").glob("*.js"))
SHARED_MODULE = "nh_widget_utils.js"

COUNT = folder_routes.COUNT_PATH
PRESETS = folder_routes.PRESETS_PATH
CLEANUP = "/nh-nodes/large-preview/cleanup"
NO_CACHE_ID = "0" * 32


def call(method, path, **kwargs):
    """One request to the pack's routes. Returns (status, parsed JSON) or (status, raw bytes) for other content."""

    async def run():
        routes = web.RouteTableDef()
        folder_routes.register_routes(routes)
        preview_routes.register_routes(routes)
        app = web.Application()
        app.add_routes(routes)
        async with TestClient(TestServer(app)) as client:
            response = await client.request(method, path, **kwargs)
            return response.status, response.content_type, await response.read()

    status, content_type, body = asyncio.run(run())
    return status, json.loads(body) if content_type == "application/json" else body


# --- registration -----------------------------------------------------------------------------------------------

def test_every_route_is_registered_and_the_package_imports_both_route_modules():
    routes = web.RouteTableDef()
    folder_routes.register_routes(routes)
    preview_routes.register_routes(routes)
    assert {(route.method, route.path) for route in routes} == {
        ("GET", "/nh-nodes/load-images-folder/count"),
        ("GET", "/nh-nodes/smart-resolution-picker/presets"),
        ("GET", "/nh-nodes/large-preview/manifest/{cache_id}"),
        ("GET", "/nh-nodes/large-preview/thumb/{cache_id}"),
        ("GET", "/nh-nodes/large-preview/tile/{cache_id}/{level}/{tile}"),
        ("GET", "/nh-nodes/large-preview/cache-info"),
        ("POST", "/nh-nodes/large-preview/cleanup"),
    }
    assert pack.folder_routes is folder_routes and pack.large_preview_routes is preview_routes


# --- /nh-nodes/load-images-folder/count --------------------------------------------------------------------------

@pytest.fixture
def image_tree(tmp_path, monkeypatch):
    """input/ holds 2 images, input/sub/ 2 more, input/sub/deeper/ 1 more; empty/ holds none."""
    root = tmp_path / "input"
    (root / "sub" / "deeper").mkdir(parents=True)
    for relative in ("top1.png", "top2.JPG", "notes.txt", "sub/deep1.png", "sub/deep2.jpg", "sub/deeper/deep3.webp"):
        (root / relative).write_bytes(b"")
    (tmp_path / "empty").mkdir()
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(root))
    return root


def test_count_answers_with_the_count_and_nothing_else(image_tree):
    status, body = call("GET", COUNT, params={"folder_path": str(image_tree)})
    assert status == 200
    assert body == {"count": 2}  # in particular no resolved "path": ComfyUI may be reachable from the network


@pytest.mark.parametrize("recursive,expected", [("true", 5), ("TRUE", 5), ("1", 5), ("yes", 5), ("on", 5),
                                                ("false", 2), ("0", 2), ("no", 2), ("off", 2), ("", 2)])
def test_count_honours_the_recursive_parameter(image_tree, recursive, expected):
    status, body = call("GET", COUNT, params={"folder_path": str(image_tree), "recursive": recursive})
    assert (status, body) == (200, {"count": expected})


def test_count_without_recursive_is_not_recursive(image_tree):
    assert call("GET", COUNT, params={"folder_path": str(image_tree)}) == (200, {"count": 2})


def test_count_resolves_like_the_loader_node(image_tree):
    assert call("GET", COUNT, params={"folder_path": "sub"}) == (200, {"count": 2})  # relative to the input folder
    assert call("GET", COUNT, params={"folder_path": "sub", "recursive": "true"}) == (200, {"count": 3})
    assert call("GET", COUNT) == (200, {"count": 2})  # no folder: the input folder
    assert call("GET", COUNT, params={"folder_path": str(image_tree / "top1.png")}) == (200, {"count": 2})  # a file: its folder
    assert call("GET", COUNT, params={"folder_path": "  sub  "}) == (200, {"count": 2})  # surrounding blanks are ignored


def test_count_of_a_folder_without_images_is_zero(image_tree):
    assert call("GET", COUNT, params={"folder_path": str(image_tree.parent / "empty")}) == (200, {"count": 0})


def test_count_of_a_missing_folder_is_404_and_does_not_echo_the_path(image_tree):
    missing = image_tree.parent / "does-not-exist"
    status, body = call("GET", COUNT, params={"folder_path": str(missing)})
    assert status == 404
    assert body == {"error": "folder not found"}
    assert str(missing) not in json.dumps(body)


@pytest.mark.parametrize("params", [
    {"folder_path": "x", "recursive": "maybe"},
    {"folder_path": "x", "recursive": "2"},
    {"folder_path": "bad\0name"},
    {"folder_path": "a" * 5000},
])
def test_count_rejects_bad_input_with_400(image_tree, params):
    status, body = call("GET", COUNT, params=params)
    assert status == 400 and list(body) == ["error"]


def test_count_errors_are_short_and_never_leak_internals(image_tree, monkeypatch):
    def denied(folder, recursive=False):
        raise PermissionError(f"[Errno 13] Permission denied: {folder!r}")

    monkeypatch.setattr(folder_routes, "collect_image_files", denied)
    status, body = call("GET", COUNT, params={"folder_path": str(image_tree)})
    assert (status, body) == (403, {"error": "folder is not readable"})

    def broken(folder, recursive=False):
        raise RuntimeError(f"secret detail about {folder}")

    monkeypatch.setattr(folder_routes, "collect_image_files", broken)
    status, body = call("GET", COUNT, params={"folder_path": str(image_tree)})
    assert status == 500
    assert body == {"error": "could not read the folder"}


def test_count_scans_in_a_worker_thread_not_on_the_event_loop(image_tree, monkeypatch):
    seen = []
    real = folder_routes.collect_image_files

    def spy(folder, recursive=False):
        seen.append(threading.get_ident())
        return real(folder, recursive=recursive)

    monkeypatch.setattr(folder_routes, "collect_image_files", spy)
    assert call("GET", COUNT, params={"folder_path": str(image_tree)})[0] == 200
    assert seen and seen[0] != threading.get_ident()


# --- /nh-nodes/smart-resolution-picker/presets --------------------------------------------------------------------

def test_presets_payload_matches_the_resolution_data():
    status, body = call("GET", PRESETS)
    assert status == 200
    assert set(body) == {"models", "default_model", "default_preset", "presets_by_model", "resolution_levels",
                         "ratios_by_model_level"}
    assert body["models"] == resolution_data.MODEL_LABELS
    assert body["default_model"] == resolution_data.DEFAULT_MODEL_LABEL
    assert body["default_preset"] == resolution_data.DEFAULT_PRESET
    assert body["presets_by_model"] == resolution_data.PRESET_LABELS_BY_MODEL
    assert body["resolution_levels"] == picker.TARGET_RESOLUTION_LEVELS
    for model in body["models"]:
        for level in body["resolution_levels"]:
            ratios = body["ratios_by_model_level"][model][level]
            assert ratios and len(ratios) == len(set(ratios))


# --- /nh-nodes/large-preview/* -----------------------------------------------------------------------------------

@pytest.fixture
def preview_temp(tmp_path, monkeypatch):
    temp = tmp_path / "temp"
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(temp))
    return temp


@pytest.mark.parametrize("body", ["[]", "[1, 2]", '"text"', "5", "null", "true"])
def test_cleanup_rejects_a_json_body_that_is_not_an_object(preview_temp, body):
    status, payload = call("POST", CLEANUP, data=body, headers={"Content-Type": "application/json"})
    assert status == 400
    assert payload == {"error": "request body must be a JSON object"}


def test_cleanup_rejects_a_body_that_is_not_json(preview_temp):
    status, payload = call("POST", CLEANUP, data="{not json", headers={"Content-Type": "application/json"})
    assert status == 400 and "JSON" in payload["error"]
    status, _ = call("POST", CLEANUP, data=b"\xff\xfe\x00", headers={"Content-Type": "application/json"})
    assert status == 400


@pytest.mark.parametrize("kwargs", [{}, {"data": ""}, {"data": "  \n"}, {"json": {}}, {"json": {"max_age_hours": 24}}])
def test_cleanup_accepts_an_empty_body_or_an_object(preview_temp, kwargs):
    status, payload = call("POST", CLEANUP, **kwargs)
    assert status == 200 and payload["removed"] == 0


@pytest.mark.parametrize("payload", [
    {"max_age_hours": [1]},
    {"max_age_hours": "abc"},
    {"max_total_mb": [1]},
    {"max_age_hours": 10 ** 400},
    {"cache_id": "../outside"},
    {"cache_id": 5},
])
def test_cleanup_answers_400_for_values_it_cannot_use(preview_temp, payload):
    status, body = call("POST", CLEANUP, json=payload)
    assert status == 400 and list(body) == ["error"]


def test_cleanup_removes_old_entries_and_keeps_new_ones(preview_temp):
    root = preview_temp / preview_cache.CACHE_SUBDIR
    old, new = root / ("a" * 32), root / ("b" * 32)
    for entry in (old, new):
        entry.mkdir(parents=True)
        (entry / "manifest.json").write_text("{}", encoding="utf-8")
    long_ago = time.time() - 48 * 3600
    os.utime(old, (long_ago, long_ago))

    status, body = call("POST", CLEANUP, json={"max_age_hours": 24})
    assert status == 200 and body["removed"] == 1
    assert not old.exists() and new.exists()


def test_large_preview_routes_serve_and_delete_a_cache_entry(preview_temp):
    manifest = preview_cache.create_preview_cache(torch.rand(48, 64, 3))
    cache_id = manifest["cache_id"]
    base = "/nh-nodes/large-preview"

    status, body = call("GET", f"{base}/manifest/{cache_id}")
    assert status == 200 and body["cache_id"] == cache_id
    for path in (f"{base}/thumb/{cache_id}", f"{base}/tile/{cache_id}/0/0_0.jpg"):
        status, data = call("GET", path)
        assert status == 200 and data[:2] == b"\xff\xd8"  # a JPEG
    status, info = call("GET", f"{base}/cache-info")
    assert status == 200 and info["cache_count"] == 1
    assert "root" not in json.dumps(info) and str(preview_temp) not in json.dumps(info)  # no absolute paths leak

    status, body = call("POST", CLEANUP, json={"cache_id": cache_id})
    assert status == 200 and body["removed"] == 1
    assert call("GET", f"{base}/manifest/{cache_id}")[0] == 404


@pytest.mark.parametrize("path,status", [
    (f"/nh-nodes/large-preview/manifest/{NO_CACHE_ID}", 404),
    ("/nh-nodes/large-preview/manifest/not-an-id", 400),
    (f"/nh-nodes/large-preview/thumb/{NO_CACHE_ID}", 404),
    ("/nh-nodes/large-preview/thumb/zz", 400),
    (f"/nh-nodes/large-preview/tile/{NO_CACHE_ID}/0/0_0.jpg", 404),
    (f"/nh-nodes/large-preview/tile/{NO_CACHE_ID}/x/0_0.jpg", 400),
    (f"/nh-nodes/large-preview/tile/{NO_CACHE_ID}/0/passwd", 400),
])
def test_large_preview_routes_answer_404_for_missing_and_400_for_malformed(preview_temp, path, status):
    assert call("GET", path)[0] == status


def test_large_preview_tile_route_does_not_leave_the_cache_folder(preview_temp):
    cache_id = preview_cache.create_preview_cache(torch.rand(8, 8, 3))["cache_id"]
    status, _ = call("GET", f"/nh-nodes/large-preview/tile/{cache_id}/0/..%2F..%2Fmanifest.json")
    assert status in (400, 404)


# --- web/*.js: syntax and structure ------------------------------------------------------------------------------

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


@needs_node
@pytest.mark.parametrize("path", WEB_FILES, ids=lambda path: path.name)
def test_every_js_file_is_valid_syntax(path):
    check = subprocess.run([shutil.which("node"), "--check", str(path)], capture_output=True, text=True,
                           encoding="utf-8", timeout=60, check=False)
    assert check.returncode == 0, check.stderr


def test_the_shared_module_imports_nothing_and_registers_no_extension():
    source = (PACK_DIR / "web" / SHARED_MODULE).read_text(encoding="utf-8")
    assert "registerExtension" not in source
    assert not re.search(r"^\s*import\b", source, re.MULTILINE)


COPIED_HELPER = re.compile(
    r"^\s*(?:async\s+)?function\s+(findWidget|toInt|clampInt|hideWidget|showWidget|chainCallback|markDirty|refreshNode)\b",
    re.MULTILINE,
)


@pytest.mark.parametrize("path", [path for path in WEB_FILES if path.name != SHARED_MODULE], ids=lambda path: path.name)
def test_extensions_import_the_shared_helpers_instead_of_copying_them(path):
    source = path.read_text(encoding="utf-8")
    assert not COPIED_HELPER.search(source)
    if re.search(r"\b(?:findWidget|toInt|clampInt|hideWidget|showWidget|chainCallback|markDirty|refreshNode)\(", source):
        assert f'from "./{SHARED_MODULE}"' in source


# --- web/*.js: behaviour in Node ----------------------------------------------------------------------------------

JS_HARNESS = r"""
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const webDir = process.argv[2];
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const toUrl = (file) => pathToFileURL(path.join(webDir, file)).href;

// --- a fake ComfyUI application: it records which extension each file registers ---
const registered = [];
let loading = null;
const appGraph = { dirty: 0, setDirtyCanvas() { this.dirty++; } };
globalThis.__nhApp = { graph: appGraph, registerExtension(extension) { registered.push({ file: loading, extension }); } };
globalThis.__nhApi = { apiURL: (url) => url };

const files = fs.readdirSync(webDir).filter((name) => name.endsWith(".js")).sort();
for (const file of files) {
  loading = file;
  await import(toUrl(file));
}
loading = null;
const utils = await import(toUrl("nh_widget_utils.js"));

// --- two kinds of widget ---
// "modern": ComfyUI frontend 1.55.17 (litegraph BaseWidget). `hidden` is an accessor on the prototype, and a type that
// contains "hidden" hides the widget as well. "legacy": a plain object of an older frontend, no `hidden` at all.
class ModernWidget {
  constructor(name, type, value, options) {
    this.name = name;
    this._type = type;
    this.value = value;
    this.options = options ?? {};
    this._hidden = false;
    this._hiddenByType = false;
  }
  get type() { return this._type; }
  set type(value) { this._type = value; this._hiddenByType = /hidden/i.test(value); }
  get hidden() { return this._hidden || this._hiddenByType; }
  set hidden(value) { this._hidden = Boolean(value); }
  computeSize() { return [0, 20]; }
}

function makeLegacyWidget(name, type, value, options) {
  return { name, type, value, options: options ?? {}, computeSize() { return [0, 20]; } };
}

// A node with the parts of LGraphNode the extensions use, plus what the frontend does with its widgets: the
// workflow keeps the value of every widget unless widget.serialize is false (litegraph serialiseWidgetValues), the
// prompt gets every widget unless options.serialize is false (graphToPrompt), and a hidden widget takes no row.
function makeNode(kind, specs) {
  const make = kind === "modern" ? (...args) => new ModernWidget(...args) : makeLegacyWidget;
  return {
    id: 1, graph: appGraph, size: [200, 100], inputs: [], outputs: [], sizeCalls: 0, dirtyCalls: 0,
    widgets: specs.map(([name, type, value, options]) => make(name, type, value, options)),
    addWidget(type, name, value, callback) {
      const widget = make(name, type, value, {});
      widget.callback = callback;
      this.widgets.push(widget);
      return widget;
    },
    addOutput(name, type) { this.outputs.push({ name, type }); },
    removeOutput(slot) { this.outputs.splice(slot, 1); },
    disconnectOutput() {},
    widget(name) { return this.widgets.find((widget) => widget.name === name); },
    isShown(widget) { return kind === "modern" ? !widget.hidden : widget.type !== "nh-hidden"; },
    shown() { return this.widgets.filter((widget) => this.isShown(widget)).map((widget) => widget.name); },
    hiddenNames() { return this.widgets.filter((widget) => !this.isShown(widget)).map((widget) => widget.name); },
    types() { return Object.fromEntries(this.widgets.map((widget) => [widget.name, widget.type])); },
    computeSize() { return [200, 40 + 24 * this.widgets.filter((widget) => this.isShown(widget)).length]; },
    setSize(size) { this.size = size; this.sizeCalls++; },
    setDirtyCanvas() { this.dirtyCalls++; },
    workflowValues() { return this.widgets.filter((widget) => widget.serialize !== false).map((widget) => widget.value); },
    promptInputs() {
      const inputs = {};
      for (const widget of this.widgets) {
        if (widget.name && widget.options?.serialize !== false) inputs[widget.name] = widget.value;
      }
      return inputs;
    },
  };
}

// What ComfyUI does for a node: beforeRegisterNodeDef hooks the class, the node is created, then configured.
async function mountNode(file, nodeTypeName, kind, specs) {
  const { extension } = registered.find((entry) => entry.file === file);
  const NodeType = function () {};
  await extension.beforeRegisterNodeDef(NodeType, { name: nodeTypeName });
  const node = makeNode(kind, specs);
  NodeType.prototype.onNodeCreated.call(node);
  NodeType.prototype.onConfigure?.call(node, {});
  await sleep(30);
  return node;
}

const results = {};
async function scenario(name, run) {
  try {
    results[name] = await run();
  } catch (error) {
    results[name] = { error: String(error?.stack ?? error) };
  }
}
const numbered = (prefix, count) => Array.from({ length: count }, (_, index) => `${prefix}_${index + 1}`);

// --- the files ---
results.registration = registered.map(({ file, extension }) => [file, extension.name]);
results.utilsExports = Object.keys(utils).sort();

// --- the shared helpers ---
await scenario("utils_basic", async () => {
  const node = { widgets: [{ name: "a", value: 1 }, { name: "b", value: 2 }] };
  return {
    found: utils.findWidget(node, "b")?.value,
    missing: utils.findWidget(node, "zzz") === undefined,
    noWidgets: utils.findWidget({}, "a") === undefined,
    nullNode: utils.findWidget(null, "a") === undefined,
    toInt: [["12", 0], [" 7", 0], [3.9, 0], ["12px", 0], ["abc", -1], [undefined, -1], [null, -1], [NaN, -1], ["", -1], [Infinity, -1]]
      .map(([value, fallback]) => utils.toInt(value, fallback)),
    clampInt: [[5, 0, 10, 1], [-3, 0, 10, 1], [99, 0, 10, 1], ["x", 2, 10, 5], ["x", 2, 10, 50], [NaN, 1, 64, 5], ["7", 1, 64, 5]]
      .map((args) => utils.clampInt(...args)),
  };
});

await scenario("hide_show_modern", async () => {
  const widget = new ModernWidget("w", "text", "keep");
  const out = {};
  utils.hideWidget(widget);
  out.hidden = widget.hidden;
  out.type = widget.type;
  out.ownComputeSize = Object.hasOwn(widget, "computeSize");
  utils.hideWidget(widget); // twice
  utils.showWidget(widget);
  out.shownAgain = widget.hidden;
  out.typeAfter = widget.type;
  utils.showWidget(widget); // twice
  out.value = widget.value;
  out.serializeUntouched = widget.serialize === undefined;
  const hiddenByDefinition = new ModernWidget("d", "text", 1);
  hiddenByDefinition.hidden = true;
  utils.showWidget(hiddenByDefinition);
  out.hiddenByDefinitionStays = hiddenByDefinition.hidden;
  utils.hideWidget(undefined);
  utils.showWidget(undefined);
  utils.showWidget(null);
  return out;
});

await scenario("hide_show_legacy", async () => {
  const widget = makeLegacyWidget("w", "text", "keep");
  const original = widget.computeSize;
  const out = {};
  utils.hideWidget(widget);
  out.hiddenType = widget.type;
  out.hiddenSize = widget.computeSize();
  out.gotHiddenProperty = "hidden" in widget;
  utils.hideWidget(widget); // a second call must not save "nh-hidden" as the type to restore
  utils.showWidget(widget);
  out.type = widget.type;
  out.sizeRestored = widget.computeSize === original;
  out.leftovers = Object.keys(widget).filter((key) => key.startsWith("__nh"));
  out.value = widget.value;
  const inherited = Object.assign(Object.create({ computeSize() { return [0, 33]; } }), { name: "i", type: "combo", value: 1, options: {} });
  utils.hideWidget(inherited);
  out.inheritedHiddenSize = inherited.computeSize();
  utils.showWidget(inherited);
  out.inheritedOwnSize = Object.hasOwn(inherited, "computeSize");
  out.inheritedSize = inherited.computeSize();
  out.inheritedType = inherited.type;
  return out;
});

await scenario("chain_callback", async () => {
  const log = [];
  const plain = { name: "plain" };
  utils.chainCallback(plain, function (...args) { log.push(["after", this === plain, args]); });
  plain.callback(1, 2);
  const withOriginal = { name: "original", callback(...args) { log.push(["original", this === withOriginal, args]); } };
  utils.chainCallback(withOriginal, function (...args) { log.push(["after", this === withOriginal, args]); });
  withOriginal.callback("x", "y");
  return log;
});

await scenario("refresh", async () => {
  const calls = [];
  const node = {
    size: [1, 1],
    computeSize() { calls.push("computeSize"); return [5, 6]; },
    setSize(size) { calls.push(["setSize", size]); },
    setDirtyCanvas(...args) { calls.push(["nodeDirty", ...args]); },
  };
  utils.refreshNode(node, { setDirtyCanvas(...args) { calls.push(["graphDirty", ...args]); } });
  const sizeless = { size: [9, 9], computeSize() { return null; }, setSize(size) { calls.push(["setSize", size]); } };
  utils.refreshNode(sizeless, null);
  utils.refreshNode({});
  utils.markDirty({}, undefined);
  return calls;
});

// --- the extensions, on both kinds of frontend ---
for (const kind of ["modern", "legacy"]) {
  await scenario(`value_match_index_${kind}`, async () => {
    const specs = [["value_count", "number", 3], ...numbered("value", 5).map((name) => [name, "text", `text ${name}`])];
    const node = await mountNode("nh_value_match_index.js", "NH_ValueMatchIndex", kind, specs);
    const out = { shown: node.shown(), hidden: node.hiddenNames(), types: node.types() };
    const count = node.widget("value_count");
    count.value = 5;
    count.callback();
    out.five = node.shown();
    out.typesAfterFive = node.types();
    count.value = 1;
    count.callback();
    out.one = node.shown();
    out.workflowValues = node.workflowValues();
    out.promptInputs = node.promptInputs();
    count.value = "abc";
    count.callback();
    out.notANumber = { count: count.value, shown: node.shown() };
    count.value = 1000;
    count.callback();
    out.tooMany = count.value;
    out.buttonSerialize = node.widget("update values").serialize;
    out.sized = node.sizeCalls > 0 && node.dirtyCalls > 0 && appGraph.dirty > 0;
    return out;
  });

  await scenario(`save_image_path_${kind}`, async () => {
    const node = await mountNode("nh_save_image_path.js", "NH_SaveImagePath", kind,
      [["create_new_folder", "toggle", false], ["new_folder_name", "text", "renders"]]);
    const out = { off: node.shown() };
    const create = node.widget("create_new_folder");
    create.value = true;
    create.callback();
    out.on = node.shown();
    create.value = false;
    create.callback();
    out.offAgain = node.shown();
    out.workflowValues = node.workflowValues();
    out.promptInputs = node.promptInputs();
    return out;
  });

  await scenario(`index_nodes_${kind}`, async () => {
    const specs = [
      ["select_mode", "combo", "index"], ["index", "number", 0], ["lora_count", "number", 2],
      ...numbered("lora", 3).map((name) => [name, "combo", name]),
      ...numbered("enable", 3).map((name) => [name, "toggle", false]),
    ];
    const node = await mountNode("nh_index_nodes.js", "NH_LoraModelIndex", kind, specs);
    const out = { indexMode: node.shown() };
    const mode = node.widget("select_mode");
    mode.value = "boolean";
    mode.callback();
    out.booleanMode = node.shown();
    const first = node.widget("enable_1");
    const second = node.widget("enable_2");
    first.value = true;
    second.value = true;
    second.callback();
    out.exclusive = [first.value, second.value];
    const count = node.widget("lora_count");
    count.value = 3;
    count.callback();
    out.threeLoras = node.shown();
    out.workflowValues = node.workflowValues();

    const concat = await mountNode("nh_index_nodes.js", "NH_TextConcatenate", kind,
      [["string_count", "number", 1], ...numbered("string", 3).map((name) => [name, "text", ""])]);
    out.concat = { count: concat.widget("string_count").value, shown: concat.shown() };
    return out;
  });
}

await scenario("any_list_split", async () => {
  const node = await mountNode("nh_any_list_split.js", "NH_AnyListSplit", "modern", [["index_count", "number", 3]]);
  const names = () => node.outputs.map((output) => output.name);
  const out = { three: names() };
  const count = node.widget("index_count");
  count.value = 1;
  count.callback();
  out.one = names();
  count.value = 100;
  count.callback();
  out.clamped = { count: count.value, names: names() };
  out.buttonSerialize = node.widget("update outputs").serialize;
  return out;
});

await scenario("load_images_folder", async () => {
  globalThis.window = { location: { origin: "http://127.0.0.1:8188" } };
  const calls = [];
  let reply = (url) => ({ ok: true, json: async () => ({ count: url.searchParams.get("recursive") === "true" ? 5 : 2 }) });
  globalThis.fetch = async (input) => {
    const url = new URL(String(input));
    calls.push({ path: url.pathname, folder: url.searchParams.get("folder_path"), recursive: url.searchParams.get("recursive") });
    return reply(url);
  };
  const specs = [["folder_path", "text", "C:/imgs"], ["index", "number", 3], ["step", "number", 1],
    ["seed_mode", "combo", "increment"], ["recursive", "toggle", true]];
  const node = await mountNode("nh_load_images_folder.js", "NH_LoadImagesFromFolder", "modern", specs);
  const index = node.widget("index");
  const recursive = node.widget("recursive");
  const out = {};
  const queue = async () => { const sent = await index.serializeValue(); await sleep(10); return [sent, index.value]; };

  out.recursiveOn = await queue();   // 5 images: 3 is kept, next is 4
  recursive.value = false;
  out.recursiveOff = await queue();  // 2 images: 4 wraps to 0, next is 1
  recursive.value = true;
  out.recursiveOnAgain = await queue();
  out.calls = calls.map((call) => call.recursive);
  out.paths = [...new Set(calls.map((call) => call.path))];
  out.folders = [...new Set(calls.map((call) => call.folder))];

  node.widget("seed_mode").value = "decrement";
  node.widget("step").value = 2;
  index.value = 0;
  out.decrement = await queue();     // 5 images: 0 stays, next is 3
  node.widget("seed_mode").value = "random";
  const random = Math.random;
  Math.random = () => 0.5;
  out.random = await queue();        // floor(0.5 * 5)
  Math.random = random;
  node.widget("seed_mode").value = "fixed";
  index.value = 7;
  out.fixed = await queue();         // 7 % 5 = 2
  reply = () => ({ ok: false, status: 404 });
  index.value = 7;
  out.notFound = await queue();      // no count: the index is left alone
  globalThis.fetch = async () => { throw new Error("offline"); };
  out.offline = await queue();

  globalThis.fetch = async (input) => { calls.push({ recursive: new URL(String(input)).searchParams.get("recursive") }); return { ok: true, json: async () => ({ count: 4 }) }; };
  const without = await mountNode("nh_load_images_folder.js", "NH_LoadImagesFromFolder", "modern", specs.slice(0, 4));
  await without.widget("index").serializeValue();
  out.withoutRecursiveWidget = calls[calls.length - 1].recursive;
  return out;
});

await scenario("smart_resolution_picker", async () => {
  const presets = {
    models: ["A", "B"], default_model: "A", default_preset: "A-1",
    presets_by_model: { A: ["A-1", "A-2"], B: ["B-1"] },
    resolution_levels: ["1 MP", "2 MP"],
    ratios_by_model_level: { A: { "1 MP": ["1:1", "3:2"], "2 MP": ["1:1"] }, B: { "1 MP": ["16:9"], "2 MP": ["16:9", "4:3"] } },
  };
  let fetches = 0;
  globalThis.fetch = async () => { fetches++; return { ok: true, status: 200, json: async () => presets }; };
  const out = {};

  const pickerNode = await mountNode("nh_smart_resolution_picker.js", "NH_SmartResolutionPicker", "modern",
    [["model_family", "combo", "A"], ["preset", "combo", "stale"]]);
  const preset = () => [pickerNode.widget("preset").options.values, pickerNode.widget("preset").value];
  out.pickerStart = preset();
  pickerNode.widget("model_family").value = "B";
  pickerNode.widget("model_family").callback();
  out.pickerModelB = preset();

  const ratioNode = await mountNode("nh_smart_resolution_picker.js", "NH_RatioPresetImageResize", "modern",
    [["model_family", "combo", "A"], ["resolution_level", "combo", "1 MP"], ["target_ratio", "combo", "9:9"]]);
  const ratio = () => [ratioNode.widget("target_ratio").options.values, ratioNode.widget("target_ratio").value];
  out.ratioStart = ratio();
  ratioNode.widget("resolution_level").value = "2 MP";
  ratioNode.widget("resolution_level").callback();
  out.ratioTwoMp = ratio();
  ratioNode.widget("model_family").value = "B";
  ratioNode.widget("model_family").callback();
  out.ratioModelB = ratio();
  out.fetches = fetches;
  return out;
});

console.log("RESULT:" + JSON.stringify(results));
"""


@pytest.fixture(scope="module")
def js(tmp_path_factory):
    """The harness results. The files are copied unchanged into a folder shaped like ComfyUI's: the extensions'
    ``../../../scripts/app.js`` finds the fake app, and ``./nh_widget_utils.js`` finds its sibling."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    work = tmp_path_factory.mktemp("web_harness")
    (work / "package.json").write_text('{"type": "module"}', encoding="utf-8")  # the copied .js files are ES modules
    (work / "scripts").mkdir()
    (work / "scripts" / "app.js").write_text("export const app = globalThis.__nhApp;\n", encoding="utf-8")
    (work / "scripts" / "api.js").write_text("export const api = globalThis.__nhApi;\n", encoding="utf-8")
    web_dir = work / "server" / "extensions" / "NH-Nodes"
    web_dir.mkdir(parents=True)
    for source in WEB_FILES:
        shutil.copy(source, web_dir / source.name)
    (work / "harness.mjs").write_text(JS_HARNESS, encoding="utf-8")
    run = subprocess.run([node, str(work / "harness.mjs"), str(web_dir)], capture_output=True, text=True,
                         encoding="utf-8", timeout=120, check=False)
    assert run.returncode == 0, run.stderr
    line = next(line for line in run.stdout.splitlines() if line.startswith("RESULT:"))
    return json.loads(line[len("RESULT:"):])


def scenario(js, name):
    result = js[name]
    assert "error" not in result, result.get("error")
    return result


def test_each_extension_file_registers_exactly_one_uniquely_named_extension(js):
    registration = js["registration"]
    assert sorted(file for file, _ in registration) == sorted(path.name for path in WEB_FILES if path.name != SHARED_MODULE)
    names = [name for _, name in registration]
    assert len(names) == len(set(names)) and all(name.startswith("NH.Nodes.") for name in names)
    assert js["utilsExports"] == sorted(
        ["chainCallback", "clampInt", "findWidget", "hideWidget", "markDirty", "refreshNode", "showWidget", "toInt"]
    )


def test_findwidget_toint_and_clampint(js):
    out = scenario(js, "utils_basic")
    assert out["found"] == 2 and out["missing"] and out["noWidgets"] and out["nullNode"]
    assert out["toInt"] == [12, 7, 3, 12, -1, -1, -1, -1, -1, -1]
    assert out["clampInt"] == [5, 0, 10, 5, 10, 5, 7]


def test_hidewidget_uses_the_hidden_flag_when_the_frontend_has_it(js):
    out = scenario(js, "hide_show_modern")
    assert out["hidden"] is True
    assert out["type"] == "text" and out["typeAfter"] == "text"  # the type is not touched
    assert out["ownComputeSize"] is False
    assert out["shownAgain"] is False
    assert out["value"] == "keep" and out["serializeUntouched"] is True  # still saved with the workflow
    assert out["hiddenByDefinitionStays"] is True  # showWidget only undoes hideWidget


def test_hidewidget_falls_back_to_the_type_swap_on_older_frontends(js):
    out = scenario(js, "hide_show_legacy")
    assert out["hiddenType"] == "nh-hidden" and out["hiddenSize"] == [0, -4] and out["gotHiddenProperty"] is False
    assert out["type"] == "text" and out["sizeRestored"] is True and out["leftovers"] == []
    assert out["value"] == "keep"
    assert out["inheritedHiddenSize"] == [0, -4]
    assert out["inheritedOwnSize"] is False and out["inheritedSize"] == [0, 33] and out["inheritedType"] == "combo"


def test_chaincallback_runs_the_existing_callback_first_with_the_same_this_and_arguments(js):
    assert scenario(js, "chain_callback") == [
        ["after", True, [1, 2]],
        ["original", True, ["x", "y"]],
        ["after", True, ["x", "y"]],
    ]


def test_refreshnode_resizes_then_repaints_and_tolerates_missing_methods(js):
    assert scenario(js, "refresh") == [
        "computeSize", ["setSize", [5, 6]], ["nodeDirty", True, True], ["graphDirty", True, True],
        ["setSize", [9, 9]],  # computeSize gave nothing: the current size is kept
    ]


@pytest.mark.parametrize("kind", ["modern", "legacy"])
def test_value_match_index_shows_only_the_counted_values_and_still_saves_the_rest(js, kind):
    out = scenario(js, f"value_match_index_{kind}")
    assert out["shown"] == ["value_count", "value_1", "value_2", "value_3", "update values"]
    assert out["hidden"] == ["value_4", "value_5"]
    assert out["five"] == ["value_count", *[f"value_{i}" for i in range(1, 6)], "update values"]
    assert out["one"] == ["value_count", "value_1", "update values"]
    hidden_type = "text" if kind == "modern" else "nh-hidden"
    assert out["types"]["value_4"] == hidden_type and out["types"]["value_1"] == "text"
    assert set(out["typesAfterFive"].values()) <= {"text", "number", "button"}  # shown again: the real type is back
    # the hidden widgets still travel with the workflow and the prompt
    assert out["workflowValues"] == [1, *[f"text value_{i}" for i in range(1, 6)]]
    assert {f"value_{i}" for i in range(1, 6)} <= set(out["promptInputs"])
    assert out["notANumber"] == {"count": 5, "shown": ["value_count", *[f"value_{i}" for i in range(1, 6)], "update values"]}
    assert out["tooMany"] == 64
    assert out["buttonSerialize"] is False and out["sized"] is True


@pytest.mark.parametrize("kind", ["modern", "legacy"])
def test_save_image_path_shows_the_folder_name_only_when_creating_a_folder(js, kind):
    out = scenario(js, f"save_image_path_{kind}")
    assert out["off"] == ["create_new_folder"]
    assert out["on"] == ["create_new_folder", "new_folder_name"]
    assert out["offAgain"] == ["create_new_folder"]
    assert out["workflowValues"] == [False, "renders"]
    assert out["promptInputs"] == {"create_new_folder": False, "new_folder_name": "renders"}


@pytest.mark.parametrize("kind", ["modern", "legacy"])
def test_index_nodes_switch_between_index_and_boolean_mode(js, kind):
    out = scenario(js, f"index_nodes_{kind}")
    assert out["indexMode"] == ["select_mode", "index", "lora_count", "lora_1", "lora_2", "update loras"]
    assert out["booleanMode"] == ["select_mode", "lora_count", "lora_1", "lora_2", "enable_1", "enable_2", "update loras"]
    assert out["exclusive"] == [False, True]  # turning enable_2 on turns enable_1 off
    assert out["threeLoras"] == ["select_mode", "lora_count", "lora_1", "lora_2", "lora_3", "enable_1", "enable_2",
                                 "enable_3", "update loras"]
    assert len(out["workflowValues"]) == 9  # all 9 widgets, shown or not (the button is not saved)
    assert out["concat"] == {"count": 2, "shown": ["string_count", "string_1", "string_2", "update strings"]}


def test_any_list_split_follows_index_count(js):
    out = scenario(js, "any_list_split")
    assert out["three"] == ["All", "index 1", "index 2", "index 3"]
    assert out["one"] == ["All", "index 1"]
    assert out["clamped"] == {"count": 10, "names": ["All", *[f"index {i}" for i in range(1, 11)]]}
    assert out["buttonSerialize"] is False


def test_load_images_folder_counts_with_the_nodes_recursive_setting(js):
    out = scenario(js, "load_images_folder")
    assert out["calls"][:3] == ["true", "false", "true"]  # follows the widget on every queue
    assert out["paths"] == ["/nh-nodes/load-images-folder/count"] and out["folders"] == ["C:/imgs"]
    assert out["recursiveOn"] == [3, 4]        # 5 images: 3, then 4
    assert out["recursiveOff"] == [0, 1]       # 2 images: 4 wraps to 0, then 1
    assert out["recursiveOnAgain"] == [1, 2]
    assert out["decrement"] == [0, 3]          # step 2 back from 0 wraps to 3 of 5
    assert out["random"] == [2, 2]
    assert out["fixed"] == [2, 7]              # 7 % 5 is sent, the widget keeps what the user typed
    assert out["notFound"] == [7, 7] and out["offline"][0] == 7  # unknown count: no wrapping
    assert out["withoutRecursiveWidget"] == "false"


def test_smart_resolution_picker_fills_the_dropdowns_from_one_request(js):
    out = scenario(js, "smart_resolution_picker")
    assert out["pickerStart"] == [["A-1", "A-2"], "A-1"]
    assert out["pickerModelB"] == [["B-1"], "B-1"]
    assert out["ratioStart"] == [["1:1", "3:2"], "1:1"]
    assert out["ratioTwoMp"] == [["1:1"], "1:1"]
    assert out["ratioModelB"] == [["16:9", "4:3"], "16:9"]
    assert out["fetches"] == 1
