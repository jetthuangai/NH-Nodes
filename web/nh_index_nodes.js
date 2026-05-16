import { app } from "../../../scripts/app.js";

const MAX_ITEMS = 64;

const NODE_CONFIGS = {
    NH_LoraModelIndex: {
        countWidget: "lora_count",
        prefix: "lora",
        buttonLabel: "update loras",
        installedFlag: "__nhLoraModelIndexInstalled",
    },
    NH_LoraClipIndex: {
        countWidget: "lora_count",
        prefix: "lora",
        buttonLabel: "update loras",
        installedFlag: "__nhLoraClipIndexInstalled",
    },
    NH_DiffusionModelIndex: {
        countWidget: "model_count",
        prefix: "diffusion_model",
        buttonLabel: "update models",
        installedFlag: "__nhDiffusionModelIndexInstalled",
    },
    NH_TextIndex: {
        countWidget: "text_count",
        prefix: "text",
        buttonLabel: "update texts",
        installedFlag: "__nhTextIndexInstalled",
    },
    NH_TextConcatenate: {
        countWidget: "string_count",
        prefix: "string",
        buttonLabel: "update strings",
        installedFlag: "__nhTextConcatenateInstalled",
        minCount: 2,
    },
};

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function toInt(value, fallback) {
    const parsed = Number.parseInt(value, 10);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function clampCount(value, minCount = 1) {
    return Math.max(minCount, Math.min(MAX_ITEMS, toInt(value, minCount)));
}

function hideWidget(widget) {
    if (!widget || widget.__nhIndexHidden) {
        return;
    }

    widget.__nhIndexHidden = true;
    widget.__nhIndexType = widget.type;
    widget.__nhIndexComputeSize = widget.computeSize;
    widget.computeSize = () => [0, -4];
    widget.type = "nh-hidden";
}

function showWidget(widget) {
    if (!widget || !widget.__nhIndexHidden) {
        return;
    }

    widget.type = widget.__nhIndexType;
    widget.computeSize = widget.__nhIndexComputeSize;

    delete widget.__nhIndexHidden;
    delete widget.__nhIndexType;
    delete widget.__nhIndexComputeSize;
}

function updateIndexedWidgets(node, config) {
    const countWidget = findWidget(node, config.countWidget);
    const count = clampCount(countWidget?.value, config.minCount || 1);

    if (countWidget) {
        countWidget.value = count;
    }

    for (let index = 1; index <= MAX_ITEMS; index++) {
        const widget = findWidget(node, `${config.prefix}_${index}`);
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

function installIndexedControls(node, config) {
    if (node[config.installedFlag]) {
        return;
    }
    node[config.installedFlag] = true;

    const updateButton = node.addWidget("button", config.buttonLabel, "update", () => {
        updateIndexedWidgets(node, config);
    });
    updateButton.serialize = false;

    const countWidget = findWidget(node, config.countWidget);
    if (countWidget) {
        const oldCallback = countWidget.callback;
        countWidget.callback = function () {
            oldCallback?.apply(this, arguments);
            updateIndexedWidgets(node, config);
        };
    }

    setTimeout(() => updateIndexedWidgets(node, config), 0);
}

app.registerExtension({
    name: "NH.Nodes.IndexNodes",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const config = NODE_CONFIGS[nodeData.name];
        if (!config) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            installIndexedControls(this, config);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            installIndexedControls(this, config);
            setTimeout(() => updateIndexedWidgets(this, config), 0);
            return result;
        };
    },
});
