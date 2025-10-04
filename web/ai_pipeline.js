import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "koolook.ai_pipeline",
    nodeCreated(node) {
        if (node.comfyClass !== "EasyAIPipeline") return;

        // Add multiline text widget for display
        const displayWidget = node.addWidget(
            "text",
            "Output Directory Display",
            "",
            () => {},
            { multiline: true, serialize: false }
        );
        displayWidget.inputEl.readOnly = true;
        displayWidget.inputEl.style.height = "100px";

        // Hook into onExecuted to update after execution
        const originalOnExecuted = node.onExecuted || function() {};
        node.onExecuted = function(output) {
            originalOnExecuted.call(this, output);
            const outputDirectory = output['3'] || "Not computed";
            displayWidget.value = outputDirectory;
            node.setDirtyCanvas(true, true);
        };
    }
});