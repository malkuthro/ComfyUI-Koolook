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
                const displayWidget = ComfyWidgets["STRING"](this, "Output Directory Display", ["STRING", { multiline: true }], app).widget;
                displayWidget.inputEl.readOnly = true;
                displayWidget.inputEl.style.height = "100px";
                // Add button to compute and display output_directory
                this.addWidget("button", "Update Display", null, () => {
                    const job_path = this.widgets.find(w => w.name === "job_path")?.value || "";
                    const shot_name = this.widgets.find(w => w.name === "shot_name")?.value || "";
                    const ai_method = this.widgets.find(w => w.name === "ai_method")?.value || "";
                    const version = this.widgets.find(w => w.name === "version")?.value || 1;
                    const version_str = `v${version.toString().padStart(3, '0')}`;
                    let output_directory = [job_path, shot_name, ai_method, version_str].filter(part => part.trim() !== "").join("/").replace(/\\/g, "/");
                    displayWidget.value = output_directory;
                    this.setDirtyCanvas(true, true);
                }, { serialize: false });
            };
        }
    }
});