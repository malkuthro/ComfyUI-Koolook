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

                // Add static label above the base_directory_path widget
                const baseDirWidget = this.widgets.find(w => w.name === "base_directory_path");
                if (baseDirWidget) {
                    const labelWidget = ComfyWidgets["STRING"](this, "base_dir_label", ["STRING", { default: "Paste your BASE directory path here:", multiline: false }], app).widget;
                    
                    // Override draw to set properties safely when inputEl is available
                    const originalLabelDraw = labelWidget.draw || function() {};
                    labelWidget.draw = function(ctx, node, width, y, height) {
                        originalLabelDraw.call(this, ctx, node, width, y, height);
                        if (this.inputEl && !this.inputEl.readOnly) {
                            this.inputEl.readOnly = true;
                            this.inputEl.style.fontWeight = "bold";
                            this.inputEl.style.backgroundColor = "#333";
                            this.inputEl.style.color = "#ccc";
                            this.inputEl.style.padding = "4px";
                            this.inputEl.style.marginBottom = "4px";  // Space it nicely above the field
                        }
                    };

                    // Insert label before the base_directory_path in the widgets array
                    const baseDirIndex = this.widgets.indexOf(baseDirWidget);
                    this.widgets.splice(baseDirIndex, 0, labelWidget);
                }

                // Existing code: Add multiline display widget using ComfyWidgets.STRING
                const displayWidget = ComfyWidgets["STRING"](this, "Output Directory Display", ["STRING", { multiline: true }], app).widget;
                
                // Override draw to set properties safely when inputEl is available (matching original intent)
                const originalDisplayDraw = displayWidget.draw || function() {};
                displayWidget.draw = function(ctx, node, width, y, height) {
                    originalDisplayDraw.call(this, ctx, node, width, y, height);
                    if (this.inputEl && !this.inputEl.readOnly) {
                        this.inputEl.readOnly = true;
                        this.inputEl.style.height = "100px";
                    }
                };

                // Existing code: Add button to compute and display output_directory
                this.addWidget("button", "Update Display", null, () => {
                    const base_directory_path = this.widgets.find(w => w.name === "base_directory_path")?.value || "";
                    const shot_name = this.widgets.find(w => w.name === "shot_name")?.value || "";
                    const ai_method = this.widgets.find(w => w.name === "ai_method")?.value || "";
                    const version = this.widgets.find(w => w.name === "version")?.value || 1;
                    const version_str = `v${version.toString().padStart(3, '0')}`;
                    let output_directory = [base_directory_path, shot_name, ai_method, version_str].filter(part => part.trim() !== "").join("/").replace(/\\/g, "/");
                    displayWidget.value = output_directory;
                    this.setDirtyCanvas(true, true);
                }, { serialize: false });

                // Recalculate node size after adding widgets
                this.computeSize();
            };
        }
    }
});