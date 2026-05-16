import { app } from "../../../scripts/app.js";

const NODE_NAME = "NH_SaveImagePath";

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function hideWidget(widget) {
    if (!widget || widget.__nhSaveImagePathHidden) {
        return;
    }

    widget.__nhSaveImagePathHidden = true;
    widget.__nhSaveImagePathType = widget.type;
    widget.__nhSaveImagePathComputeSize = widget.computeSize;
    widget.computeSize = () => [0, -4];
    widget.type = "nh-hidden";
}

function showWidget(widget) {
    if (!widget || !widget.__nhSaveImagePathHidden) {
        return;
    }

    widget.type = widget.__nhSaveImagePathType;
    widget.computeSize = widget.__nhSaveImagePathComputeSize;

    delete widget.__nhSaveImagePathHidden;
    delete widget.__nhSaveImagePathType;
    delete widget.__nhSaveImagePathComputeSize;
}

function updateNewFolderWidget(node) {
    const createWidget = findWidget(node, "create_new_folder");
    const folderNameWidget = findWidget(node, "new_folder_name");

    if (!createWidget || !folderNameWidget) {
        return;
    }

    if (createWidget.value) {
        showWidget(folderNameWidget);
    } else {
        hideWidget(folderNameWidget);
    }

    node.setSize?.(node.computeSize?.() || node.size);
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function installSaveImagePathControls(node) {
    if (node.__nhSaveImagePathControlsInstalled) {
        return;
    }
    node.__nhSaveImagePathControlsInstalled = true;

    const createWidget = findWidget(node, "create_new_folder");
    if (createWidget) {
        const oldCallback = createWidget.callback;
        createWidget.callback = function () {
            oldCallback?.apply(this, arguments);
            updateNewFolderWidget(node);
        };
    }

    setTimeout(() => updateNewFolderWidget(node), 0);
}

app.registerExtension({
    name: "NH.Nodes.SaveImagePath",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            installSaveImagePathControls(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            installSaveImagePathControls(this);
            setTimeout(() => updateNewFolderWidget(this), 0);
            return result;
        };
    },
});
