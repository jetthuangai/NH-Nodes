import { app } from "../../../scripts/app.js";

const NODE_NAME = "NH_LoadImagesFromFolder";
const COUNT_ENDPOINT = "/nh-nodes/load-images-folder/count";

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function toInt(value, fallback) {
    const parsed = Number.parseInt(value, 10);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function wrapIndex(value, count) {
    const index = Math.trunc(value);
    if (count > 0) {
        return ((index % count) + count) % count;
    }
    return Math.max(0, index);
}

async function getImageCount(folderPath) {
    const url = new URL(COUNT_ENDPOINT, window.location.origin);
    url.searchParams.set("folder_path", folderPath || "");

    try {
        const response = await fetch(url);
        if (!response.ok) {
            return 0;
        }
        const data = await response.json();
        return Math.max(0, toInt(data.count, 0));
    } catch {
        return 0;
    }
}

function setWidgetValue(node, widget, value) {
    widget.value = value;
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function installIndexControl(node) {
    if (node.__nhLoadImagesFolderIndexControl) {
        return;
    }

    const folderWidget = findWidget(node, "folder_path");
    const indexWidget = findWidget(node, "index");
    const stepWidget = findWidget(node, "step");
    const modeWidget = findWidget(node, "seed_mode");

    if (!folderWidget || !indexWidget || !stepWidget || !modeWidget) {
        return;
    }

    node.__nhLoadImagesFolderIndexControl = true;

    indexWidget.serializeValue = async function () {
        const mode = modeWidget.value || "fixed";
        const step = Math.max(1, toInt(stepWidget.value, 1));
        const imageCount = await getImageCount(folderWidget.value);
        let currentIndex = wrapIndex(toInt(indexWidget.value, 0), imageCount);

        if (mode === "random") {
            currentIndex = imageCount > 0 ? Math.floor(Math.random() * imageCount) : currentIndex;
            setWidgetValue(node, indexWidget, currentIndex);
            return currentIndex;
        }

        if (mode === "increment" || mode === "decrement") {
            const delta = mode === "increment" ? step : -step;
            const nextIndex = wrapIndex(currentIndex + delta, imageCount);
            setTimeout(() => setWidgetValue(node, indexWidget, nextIndex), 0);
        }

        return currentIndex;
    };
}

app.registerExtension({
    name: "NH.Nodes.LoadImagesFolderIndexControl",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            installIndexControl(this);
            return result;
        };
    },
});
