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
                // Add multiline display widget using ComfyWidgets.STRING
                const displayWidget = ComfyWidgets["STRING"](this, "Output Preview", ["STRING", { multiline: true }], app).widget;
                displayWidget.inputEl.readOnly = true;
                displayWidget.inputEl.style.height = "100px";

                // Button to get and display output_directory (with cleaning for duplicate '/')
                this.addWidget("button", "Get output directory path", null, () => {
                    const base_directory_path = this.widgets.find(w => w.name === "base_directory_path")?.value || "";
                    const shot_name = this.widgets.find(w => w.name === "shot_name")?.value || "";
                    const ai_method = this.widgets.find(w => w.name === "ai_method")?.value || "";
                    const version = this.widgets.find(w => w.name === "version")?.value || 1;
                    const version_str = `v${version.toString().padStart(3, '0')}`;
                    let output_directory = [base_directory_path, shot_name, ai_method, version_str].filter(part => part.trim() !== "").join("/").replace(/\\/g, "/");
                    // Clean any duplicate '//' or more
                    output_directory = output_directory.replace(/\/+/g, '/');
                    displayWidget.value = output_directory;
                    this.setDirtyCanvas(true, true);
                }, { serialize: false });

                // New button to get and display full file_path (with cleaning for duplicate '/')
                this.addWidget("button", "Get output file path", null, () => {
                    const base_directory_path = this.widgets.find(w => w.name === "base_directory_path")?.value || "";
                    const shot_name = this.widgets.find(w => w.name === "shot_name")?.value || "";
                    const ai_method = this.widgets.find(w => w.name === "ai_method")?.value || "";
                    const extension = this.widgets.find(w => w.name === "extension")?.value || "";
                    const version = this.widgets.find(w => w.name === "version")?.value || 1;
                    const version_str = `v${version.toString().padStart(3, '0')}`;
                    let output_directory = [base_directory_path, shot_name, ai_method, version_str].filter(part => part.trim() !== "").join("/").replace(/\\/g, "/");
                    output_directory = output_directory.replace(/\/+/g, '/');
                    const name = `${shot_name}_${ai_method}_${version_str}${extension}`;
                    let file_path = `${output_directory}/${name}`.replace(/\\/g, "/");
                    file_path = file_path.replace(/\/+/g, '/');
                    displayWidget.value = file_path;
                    this.setDirtyCanvas(true, true);
                }, { serialize: false });
            };
        }
    },
    async nodeCreated(node) {
        if (node.comfyClass !== "EasyResize") return;

        // Add multiline display widget
        const displayWidget = ComfyWidgets["STRING"](node, "Details Preview", ["STRING", { multiline: true }], app).widget;
        displayWidget.inputEl.readOnly = true;
        displayWidget.inputEl.style.height = "100px";

        // Function to get original details
        const getOriginalDetails = (node, callback) => {
            const imageInput = node.inputs?.find(i => i.name === "image");
            if (!imageInput || !imageInput.link) {
                displayWidget.value = "No image connected";
                return;
            }
            const link = app.graph.links[imageInput.link];
            const originNode = app.graph.getNodeById(link.origin_id);
            if (!originNode || !originNode.imgs || originNode.imgs.length === 0) {
                displayWidget.value = "No image loaded in source node";
                return;
            }
            const frames = originNode.imgs.length;
            const previewImg = originNode.imgs[0];
            const img = new Image();
            img.src = previewImg.src;
            img.onload = () => {
                const w = img.naturalWidth;
                const h = img.naturalHeight;
                function gcd(a, b) {
                    return b === 0 ? a : gcd(b, a % b);
                }
                const d = gcd(w, h);
                const ar_w = w / d;
                const ar_h = h / d;
                callback(w, h, frames, ar_w, ar_h);
            };
            img.onerror = () => {
                displayWidget.value = "Error loading image";
            };
        };

        // Button to show original details
        node.addWidget("button", "Show Original Details", null, () => {
            getOriginalDetails(node, (w, h, frames, ar_w, ar_h) => {
                displayWidget.value = `Original Resolution: ${w}x${h}\nFrames: ${frames}\nAspect Ratio: ${ar_w}:${ar_h}`;
                node.setDirtyCanvas(true, true);
            });
        }, { serialize: false });

        // Callback for base_on
        const baseOnWidget = node.widgets.find(w => w.name === "base_on");
        const origBaseCallback = baseOnWidget.callback;
        baseOnWidget.callback = (value) => {
            if (origBaseCallback) origBaseCallback.call(node, value);
            if (value === "Original") {
                getOriginalDetails(node, (w, h, frames, ar_w, ar_h) => {
                    const aspectWidget = node.widgets.find(w => w.name === "aspect_ratio");
                    if (aspectWidget) {
                        aspectWidget.value = `${ar_w}:${ar_h}`;
                        node.setDirtyCanvas(true, true);
                    }
                });
            }
        };

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