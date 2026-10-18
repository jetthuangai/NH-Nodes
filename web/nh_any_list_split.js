import { app } from "../../../scripts/app.js";
import { chainCallback, clampInt, findWidget, refreshNode } from "./nh_widget_utils.js";

const NODE_NAME = "NH_AnyListSplit";
const MAX_INDEX_OUTPUTS = 10;

function clampIndexCount(value) {
    return clampInt(value, 0, MAX_INDEX_OUTPUTS, 4);
}

function indexOutputNumber(output) {
    const match = /^index\s+(\d+)$/i.exec(output?.name || "");
    return match ? Number.parseInt(match[1], 10) : null;
}

function hasOutput(node, name) {
    return (node.outputs || []).some((output) => output.name === name);
}

function updateOutputs(node) {
    const countWidget = findWidget(node, "index_count");
    const count = clampIndexCount(countWidget?.value);
    if (countWidget) {
        countWidget.value = count;
    }

    if (!hasOutput(node, "All")) {
        node.addOutput("All", "*");
    }

    for (let slot = (node.outputs?.length ?? 0) - 1; slot >= 0; slot--) {
        const outputNumber = indexOutputNumber(node.outputs[slot]);
        if (outputNumber !== null && outputNumber > count) {
            node.disconnectOutput?.(slot);
            node.removeOutput(slot);
        }
    }

    for (let index = 1; index <= count; index++) {
        const name = `index ${index}`;
        if (!hasOutput(node, name)) {
            node.addOutput(name, "*");
        }
    }

    refreshNode(node, app.graph);
}

function installControls(node) {
    if (node.__nhAnyListSplitControlsInstalled) {
        return;
    }
    node.__nhAnyListSplitControlsInstalled = true;

    const updateButton = node.addWidget("button", "update outputs", "update", () => {
        updateOutputs(node);
    });
    updateButton.serialize = false;

    const countWidget = findWidget(node, "index_count");
    if (countWidget) {
        chainCallback(countWidget, () => updateOutputs(node));
    }

    setTimeout(() => updateOutputs(node), 0);
}

app.registerExtension({
    name: "NH.Nodes.AnyListSplit",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            installControls(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            installControls(this);
            setTimeout(() => updateOutputs(this), 0);
            return result;
        };
    },
});
