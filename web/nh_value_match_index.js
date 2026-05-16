import { app } from "../../../scripts/app.js";

const NODE_NAME = "NH_ValueMatchIndex";
const MAX_VALUES = 64;

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function toInt(value, fallback) {
    const parsed = Number.parseInt(value, 10);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function clampValueCount(value) {
    return Math.max(1, Math.min(MAX_VALUES, toInt(value, 5)));
}

function hideWidget(widget) {
    if (!widget || widget.__nhValueMatchHidden) {
        return;
    }

    widget.__nhValueMatchHidden = true;
    widget.__nhValueMatchType = widget.type;
    widget.__nhValueMatchComputeSize = widget.computeSize;
    widget.computeSize = () => [0, -4];
    widget.type = "nh-hidden";
}

function showWidget(widget) {
    if (!widget || !widget.__nhValueMatchHidden) {
        return;
    }

    widget.type = widget.__nhValueMatchType;
    widget.computeSize = widget.__nhValueMatchComputeSize;

    delete widget.__nhValueMatchHidden;
    delete widget.__nhValueMatchType;
    delete widget.__nhValueMatchComputeSize;
}

function updateValueWidgets(node) {
    const countWidget = findWidget(node, "value_count");
    const count = clampValueCount(countWidget?.value);

    if (countWidget) {
        countWidget.value = count;
    }

    for (let index = 1; index <= MAX_VALUES; index++) {
        const widget = findWidget(node, `value_${index}`);
        if (index <= count) {
            showWidget(widget);
        } else {
            hideWidget(widget);
        }
    }

    node.setSize?.(node.computeSize?.() || node.size);
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function installValueMatchControls(node) {
    if (node.__nhValueMatchControlsInstalled) {
        return;
    }
    node.__nhValueMatchControlsInstalled = true;

    const updateButton = node.addWidget("button", "update values", "update", () => {
        updateValueWidgets(node);
    });
    updateButton.serialize = false;

    const countWidget = findWidget(node, "value_count");
    if (countWidget) {
        const oldCallback = countWidget.callback;
        countWidget.callback = function () {
            oldCallback?.apply(this, arguments);
            updateValueWidgets(node);
        };
    }

    setTimeout(() => updateValueWidgets(node), 0);
}

app.registerExtension({
    name: "NH.Nodes.ValueMatchIndex",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            installValueMatchControls(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            installValueMatchControls(this);
            setTimeout(() => updateValueWidgets(this), 0);
            return result;
        };
    },
});
