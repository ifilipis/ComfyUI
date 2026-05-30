import { app } from "../../scripts/app.js";

async function updateWorldStereoScenes(node, pathValue, sceneValue) {
    const pathWidget = node.widgets?.find((w) => w.name === "path");
    const sceneWidget = node.widgets?.find((w) => w.name === "scene");
    if (!pathWidget || !sceneWidget) {
        return;
    }

    const updateId = (node.worldstereoSceneUpdateId ?? 0) + 1;
    node.worldstereoSceneUpdateId = updateId;

    const response = await fetch(`/worldstereo/scenes?path=${encodeURIComponent(pathValue ?? pathWidget.value ?? "")}`);
    if (!response.ok) {
        return;
    }

    const scenes = await response.json();
    if (node.worldstereoSceneUpdateId !== updateId) {
        return;
    }

    sceneWidget.options.values = scenes;
    const selectedScene = sceneValue ?? sceneWidget.value;
    if (scenes.includes(selectedScene)) {
        sceneWidget.value = selectedScene;
    } else {
        sceneWidget.value = scenes[0] ?? "none";
    }
}

app.registerExtension({
    name: "Comfy.WorldStereoMemorySceneRefresh",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "WorldStereoMemoryLoader") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const pathWidget = this.widgets?.find((w) => w.name === "path");
            if (pathWidget) {
                const callback = pathWidget.callback;
                pathWidget.callback = (...args) => {
                    callback?.apply(pathWidget, args);
                    updateWorldStereoScenes(this);
                };
            }
            updateWorldStereoScenes(this);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            onConfigure?.apply(this, arguments);
            const pathValue = Array.isArray(info?.widgets_values) ? info.widgets_values[0] : undefined;
            const sceneValue = Array.isArray(info?.widgets_values) ? info.widgets_values[1] : undefined;
            requestAnimationFrame(() => updateWorldStereoScenes(this, pathValue, sceneValue));
        };
    },
});
