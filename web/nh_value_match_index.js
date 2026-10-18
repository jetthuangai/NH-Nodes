import { app } from "../../../scripts/app.js";
import { chainCallback, clampInt, findWidget, hideWidget, refreshNode, showWidget } from "./nh_widget_utils.js";

const NODE_NAME = "NH_ValueMatchIndex";
const MAX_VALUES = 64;

function clampValueCount(value) {
    return clampInt(value, 1, MAX_VALUES, 5);
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

    refreshNode(node, app.graph);
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
        chainCallback(countWidget, () => updateValueWidgets(node));
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
