import { app } from "../../../scripts/app.js";

const NODE_NAME = "NH_SmartResolutionPicker";
const PRESETS_ENDPOINT = "/nh-nodes/smart-resolution-picker/presets";

let cachedPresetData = null;

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

async function getPresetData() {
    if (cachedPresetData) {
        return cachedPresetData;
    }

    const response = await fetch(PRESETS_ENDPOINT);
    if (!response.ok) {
        throw new Error(`Failed to load NH preset data: ${response.status}`);
    }
    cachedPresetData = await response.json();
    return cachedPresetData;
}

function setComboValues(widget, values) {
    widget.options = widget.options || {};
    widget.options.values = values;
    widget.values = values;
}

function updatePresetDropdown(node, presetData, keepCurrent = true) {
    const modelWidget = findWidget(node, "model_family");
    const presetWidget = findWidget(node, "preset");
    if (!modelWidget || !presetWidget) {
        return;
    }

    const model = modelWidget.value || presetData.default_model;
    const presets = presetData.presets_by_model?.[model] || presetData.presets_by_model?.[presetData.default_model] || [];
    const current = presetWidget.value;

    setComboValues(presetWidget, presets);
    if (!keepCurrent || !presets.includes(current)) {
        presetWidget.value = model === presetData.default_model ? presetData.default_preset : presets[0];
    }

    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function installSmartResolutionPicker(node) {
    if (node.__nhSmartResolutionPickerInstalled) {
        return;
    }
    node.__nhSmartResolutionPickerInstalled = true;

    getPresetData().then((presetData) => {
        updatePresetDropdown(node, presetData, true);

        const modelWidget = findWidget(node, "model_family");
        if (!modelWidget) {
            return;
        }

        const oldCallback = modelWidget.callback;
        modelWidget.callback = function () {
            oldCallback?.apply(this, arguments);
            updatePresetDropdown(node, presetData, false);
        };
    }).catch((error) => {
        console.warn("[NH-Nodes] Smart Resolution Picker preset update failed:", error);
    });
}

app.registerExtension({
    name: "NH.Nodes.SmartResolutionPicker",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            installSmartResolutionPicker(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            installSmartResolutionPicker(this);
            return result;
        };
    },
});
