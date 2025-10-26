import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

app.registerExtension({
    name: "koolook.ai_pipeline",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "EasyAIPipeline") {
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                if (originalOnNodeCreated) {
                    originalOnNodeCreated.call(this);
                }

                // Make instruction widget read-only and dimmed
                const instructionWidget = this.widgets.find(w => w.name === "instruction");
                if (instructionWidget && instructionWidget.inputEl) {
                    instructionWidget.inputEl.readOnly = true;
                    instructionWidget.inputEl.style.opacity = 0.6;
                }

                // Add multiline display widget using ComfyWidgets.STRING
                const displayWidget = ComfyWidgets["STRING"](this, "Output Preview", ["STRING", { multiline: true }], app).widget;
                displayWidget.inputEl.readOnly = true;
                displayWidget.inputEl.style.height = "100px";

                // Helper function to get effective value (from widget or upstream if connected and simple)
                const getEffectiveValue = (node, name) => {
                    const widget = node.widgets.find(w => w.name === name);
                    const input = node.inputs?.find(i => i.name === name);
                    if (!input || input.link === null) {
                        return widget?.value;
                    }
                    const linkId = input.link;
                    const link = app.graph.links[linkId];
                    if (!link) return widget?.value;
                    const originNode = app.graph.getNodeById(link.origin_id);
                    const originWidget = originNode.widgets ? originNode.widgets[link.origin_slot] : null;
                    if (originWidget && 'value' in originWidget) {
                        return originWidget.value;
                    }
                    return null; // Cannot resolve
                };

                // Button to get and display output_directory (with cleaning for duplicate '/')
                this.addWidget("button", "Get output directory path", null, () => {
                    const values = {
                        base_directory_path: getEffectiveValue(this, "base_directory_path"),
                        shot_name: getEffectiveValue(this, "shot_name"),
                        ai_method: getEffectiveValue(this, "ai_method"),
                        version: getEffectiveValue(this, "version")
                    };
                    if (Object.values(values).some(v => v === null)) {
                        displayWidget.value = "Cannot preview: complex inputs - execute node for accurate path.";
                    } else {
                        const version_str = `v${values.version.toString().padStart(3, '0')}`;
                        let output_directory = [values.base_directory_path, values.shot_name, values.ai_method, version_str]
                            .filter(part => part.toString().trim() !== "")
                            .join("/").replace(/\\/g, "/");
                        output_directory = output_directory.replace(/\/+/g, '/');
                        displayWidget.value = output_directory;
                    }
                    this.setDirtyCanvas(true, true);
                }, { serialize: false });

                // New button to get and display full file_path (with cleaning for duplicate '/')
                this.addWidget("button", "Get output file path", null, () => {
                    const values = {
                        base_directory_path: getEffectiveValue(this, "base_directory_path"),
                        shot_name: getEffectiveValue(this, "shot_name"),
                        ai_method: getEffectiveValue(this, "ai_method"),
                        extension: getEffectiveValue(this, "extension"),
                        version: getEffectiveValue(this, "version")
                    };
                    if (Object.values(values).some(v => v === null)) {
                        displayWidget.value = "Cannot preview: complex inputs - execute node for accurate path.";
                    } else {
                        const version_str = `v${values.version.toString().padStart(3, '0')}`;
                        let output_directory = [values.base_directory_path, values.shot_name, values.ai_method, version_str]
                            .filter(part => part.toString().trim() !== "")
                            .join("/").replace(/\\/g, "/");
                        output_directory = output_directory.replace(/\/+/g, '/');
                        const name = `${values.shot_name}_${values.ai_method}_${version_str}${values.extension}`;
                        let file_path = `${output_directory}/${name}`.replace(/\\/g, "/");
                        file_path = file_path.replace(/\/+/g, '/');
                        displayWidget.value = file_path;
                    }
                    this.setDirtyCanvas(true, true);
                }, { serialize: false });
            };
        }
    },
    async nodeCreated(node) {
        if (node.comfyClass !== "EasyResize") return;

        // Function to update widgets for a mode
        const updateWidgets = (modeName, colorName, value) => {
            const colorWidgetIndex = node.widgets.findIndex(w => w.name === colorName);
            if (value === "Custom") {
                if (colorWidgetIndex === -1) {
                    node.addWidget("string", colorName, "0, 0, 0", null, { multiline: false });
                }
            } else {
                if (colorWidgetIndex !== -1) {
                    node.widgets.splice(colorWidgetIndex, 1);  // Remove
                }
            }
            node.setDirtyCanvas(true, true);
        };

        // Pad color mode
        const padModeWidget = node.widgets.find(w => w.name === "pad_color_mode");
        const origPadCallback = padModeWidget.callback;
        padModeWidget.callback = (value) => {
            if (origPadCallback) origPadCallback.call(node, value);
            updateWidgets("pad_color_mode", "pad_color", value);
        };
        updateWidgets("pad_color_mode", "pad_color", padModeWidget.value);

        // Panel color mode
        const panelModeWidget = node.widgets.find(w => w.name === "panel_color_mode");
        const origPanelCallback = panelModeWidget.callback;
        panelModeWidget.callback = (value) => {
            if (origPanelCallback) origPanelCallback.call(node, value);
            updateWidgets("panel_color_mode", "panel_color", value);
        };
        updateWidgets("panel_color_mode", "panel_color", panelModeWidget.value);
    }
});