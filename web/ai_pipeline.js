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
    }
});