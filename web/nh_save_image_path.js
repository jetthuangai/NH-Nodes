import { app } from "../../../scripts/app.js";
import { chainCallback, findWidget, hideWidget, refreshNode, showWidget } from "./nh_widget_utils.js";

const NODE_NAME = "NH_SaveImagePath";

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

    refreshNode(node, app.graph);
}

function installSaveImagePathControls(node) {
    if (node.__nhSaveImagePathControlsInstalled) {
        return;
    }
    node.__nhSaveImagePathControlsInstalled = true;

    const createWidget = findWidget(node, "create_new_folder");
    if (createWidget) {
        chainCallback(createWidget, () => updateNewFolderWidget(node));
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
