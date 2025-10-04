import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "koolook.ai_pipeline",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "EasyAIPipeline") {
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                if (originalOnNodeCreated) {
                    originalOnNodeCreated.call(this);
                }

                // Add multiline text widget for display (read-only, not serialized)
                const displayWidget = this.addWidget(
                    "text",
                    "Output Directory Display",
                    "",
                    () => {},
                    { multiline: true, serialize: false }
                );
                displayWidget.inputEl.readOnly = true;
                displayWidget.inputEl.style.height = "100px";  // Adjust for visibility

                // Add button to compute and display output_directory
                this.addWidget(
                    "button",
                    "Update Display",
                    null,
                    () => {
                        // Get current input values from widgets
                        const job_path = this.widgets.find(w => w.name === "job_path")?.value || "";
                        const shot_name = this.widgets.find(w => w.name === "shot_name")?.value || "";
                        const ai_method = this.widgets.find(w => w.name === "ai_method")?.value || "";
                        const version = this.widgets.find(w => w.name === "version")?.value || 1;

                        // Compute version_str
                        const version_str = `v${version.toString().padStart(3, '0')}`;

                        // Compute output_directory (mirror Python logic)
                        let output_directory = [job_path, shot_name, ai_method, version_str]
                            .filter(part => part.trim() !== "")
                            .join("/")
                            .replace(/\\/g, "/");

                        // Update display
                        displayWidget.value = output_directory;

                        // Redraw canvas
                        this.setDirtyCanvas(true, true);
                    },
                    { serialize: false }
                );
            };
        }
    }
});