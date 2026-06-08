import { app } from "../../../scripts/app.js";

const MAX_ITEMS = 64;

const NODE_CONFIGS = {
    NH_DiffusionModelIndex: {
        countWidget: "model_count",
        prefix: "diffusion_model",
        enablePrefix: "enable",
        selectModeWidget: "select_mode",
        indexWidget: "index",
        buttonLabel: "update models",
        installedFlag: "__nhDiffusionModelIndexInstalled",
    },
    NH_LoraModelIndex: {
        countWidget: "lora_count",
        prefix: "lora",
        enablePrefix: "enable",
        selectModeWidget: "select_mode",
        indexWidget: "index",
        buttonLabel: "update loras",
        installedFlag: "__nhLoraModelIndexInstalled",
    },
    NH_LoraClipIndex: {
        countWidget: "lora_count",
        prefix: "lora",
        enablePrefix: "enable",
        selectModeWidget: "select_mode",
        indexWidget: "index",
        buttonLabel: "update loras",
        installedFlag: "__nhLoraClipIndexInstalled",
    },
    NH_VAEIndex: {
        countWidget: "vae_count",
        prefix: "vae",
        enablePrefix: "enable",
        selectModeWidget: "select_mode",
        indexWidget: "index",
        buttonLabel: "update vaes",
        installedFlag: "__nhVAEIndexInstalled",
    },
    NH_ClipIndex: {
        countWidget: "clip_count",
        prefix: "clip",
        enablePrefix: "enable",
        selectModeWidget: "select_mode",
        indexWidget: "index",
        buttonLabel: "update clips",
        installedFlag: "__nhClipIndexInstalled",
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

    const selectModeWidget = config.selectModeWidget ? findWidget(node, config.selectModeWidget) : null;
    const selectMode = selectModeWidget?.value || "index";
    const indexWidget = config.indexWidget ? findWidget(node, config.indexWidget) : null;

    if (indexWidget) {
        if (selectMode === "index") showWidget(indexWidget);
        else hideWidget(indexWidget);
    }

    for (let index = 1; index <= MAX_ITEMS; index++) {
        const itemWidget = findWidget(node, `${config.prefix}_${index}`);
        const enableWidget = config.enablePrefix ? findWidget(node, `${config.enablePrefix}_${index}`) : null;

        if (index <= count) {
            showWidget(itemWidget);
            if (enableWidget) {
                if (selectMode === "boolean") {
                    showWidget(enableWidget);
                } else {
                    hideWidget(enableWidget);
                }
            }
        } else {
            hideWidget(itemWidget);
            if (enableWidget) hideWidget(enableWidget);
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

    if (config.selectModeWidget) {
        const smWidget = findWidget(node, config.selectModeWidget);
        if (smWidget) {
            const smOld = smWidget.callback;
            smWidget.callback = function () {
                smOld?.apply(this, arguments);
                updateIndexedWidgets(node, config);
            };
        }
    }

    if (config.enablePrefix) {
        for (let i = 1; i <= MAX_ITEMS; i++) {
            const enableWidget = findWidget(node, `${config.enablePrefix}_${i}`);
            if (enableWidget) {
                const enOld = enableWidget.callback;
                enableWidget.callback = function () {
                    enOld?.apply(this, arguments);
                    if (this.value) {
                        for (let j = 1; j <= MAX_ITEMS; j++) {
                            if (j !== i) {
                                const other = findWidget(node, `${config.enablePrefix}_${j}`);
                                if (other && other.value) {
                                    other.value = false;
                                }
                            }
                        }
                        updateIndexedWidgets(node, config);
                    }
                };
            }
        }
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
