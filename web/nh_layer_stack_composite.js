import { app } from "../../../scripts/app.js";

const NODE_NAME = "NH_LayerStackComposite";
const MAX_LAYERS = 64;

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function toInt(value, fallback) {
    const parsed = Number.parseInt(value, 10);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function clampLayerCount(value) {
    return Math.max(0, Math.min(MAX_LAYERS, toInt(value, 1)));
}

function findInputIndex(node, name) {
    return node.inputs?.findIndex((input) => input.name === name) ?? -1;
}

function removeLayerInputsAbove(node, count) {
    for (let index = (node.inputs?.length ?? 0) - 1; index >= 0; index--) {
        const input = node.inputs[index];
        const match = /^layer_(\d+)$/.exec(input?.name || "");
        if (match && Number.parseInt(match[1], 10) > count) {
            node.removeInput(index);
        }
    }
}

function addMissingLayerInputs(node, count) {
    for (let index = 1; index <= count; index++) {
        const name = `layer_${index}`;
        if (findInputIndex(node, name) < 0) {
            node.addInput(name, "IMAGE");
        }
    }
}

// graph.links is a Map in current frontends (a Proxy that still allows links[id]) and a plain object keyed by
// link id in older ones.
function findLink(graph, linkId) {
    if (linkId == null || !graph) {
        return null;
    }
    if (typeof graph.getLink === "function") {
        return graph.getLink(linkId) ?? null;
    }
    const links = graph.links;
    if (links instanceof Map) {
        return links.get(linkId) ?? null;
    }
    return links?.[linkId] ?? null;
}

// The link attached to the input at `index`. Current frontends derive it from the index (input.link is only a
// deprecated view of that), older ones keep the link id on the input.
function getInputLink(node, index) {
    if (!node.graph) {
        return null; // removed from the graph before the timer fired: nothing can be linked to it
    }
    if (typeof node.getInputLink === "function") {
        return node.getInputLink(index) ?? null;
    }
    return findLink(node.graph, node.inputs[index]?.link);
}

// A link remembers the index of the input it ends on (target_slot). Move every link onto its input's new index.
function retargetLinks(moves, slotCount) {
    // Park the links on free slots first: current frontends refuse to move a link onto a slot that another link
    // still holds, which is what happens when two linked inputs change places.
    moves.forEach(({ link }, offset) => {
        link.target_slot = slotCount + offset;
    });
    moves.forEach(({ link, newIndex }) => {
        link.target_slot = newIndex;
    });
}

function sortLayerInputs(node) {
    const inputs = node.inputs || [];
    const background = inputs.filter((input) => input.name === "background_image");
    const layers = inputs
        .filter((input) => /^layer_\d+$/.test(input.name))
        .sort((a, b) => toInt(a.name.slice(6), 0) - toInt(b.name.slice(6), 0));
    const others = inputs.filter(
        (input) => input.name !== "background_image" && !/^layer_\d+$/.test(input.name)
    );
    const sorted = [...background, ...layers, ...others];
    if (sorted.every((input, index) => input === inputs[index])) {
        return;
    }

    // Read the links before reordering: current frontends find an input's link through its index.
    const moves = [];
    sorted.forEach((input, newIndex) => {
        const oldIndex = inputs.indexOf(input);
        const link = oldIndex === newIndex ? null : getInputLink(node, oldIndex);
        if (link) {
            moves.push({ link, newIndex });
        }
    });

    node.inputs = sorted;
    retargetLinks(moves, sorted.length);
    node.setDirtyCanvas?.(true, true);
    (node.graph ?? app.graph)?.setDirtyCanvas?.(true, true);
}

function updateLayerInputs(node) {
    const layerCountWidget = findWidget(node, "layer_count");
    const count = clampLayerCount(layerCountWidget?.value);
    if (layerCountWidget) {
        layerCountWidget.value = count;
    }

    removeLayerInputsAbove(node, count);
    addMissingLayerInputs(node, count);
    sortLayerInputs(node);

    node.setSize?.(node.computeSize?.() || node.size);
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function installLayerStackControls(node) {
    if (node.__nhLayerStackControlsInstalled) {
        return;
    }
    node.__nhLayerStackControlsInstalled = true;

    const updateButton = node.addWidget("button", "update layers", "update", () => {
        updateLayerInputs(node);
    });
    updateButton.serialize = false;

    const layerCountWidget = findWidget(node, "layer_count");
    if (layerCountWidget) {
        const oldCallback = layerCountWidget.callback;
        layerCountWidget.callback = function () {
            oldCallback?.apply(this, arguments);
            updateLayerInputs(node);
        };
    }

    setTimeout(() => updateLayerInputs(node), 0);
}

app.registerExtension({
    name: "NH.Nodes.LayerStackComposite",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            installLayerStackControls(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            installLayerStackControls(this);
            setTimeout(() => updateLayerInputs(this), 0);
            return result;
        };
    },
});
