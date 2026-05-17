// SPDX-FileCopyrightText: 2025-2026 Kforge Labs <https://github.com/malkuthro/ComfyUI-Koolook>
// SPDX-License-Identifier: GPL-3.0-only

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

                // Output preview: one logical string per line, never word-wrapped. A path
                // / filename should always render on a single visual line — wrap=off on the
                // textarea gives horizontal scroll instead of breaking long paths mid-string,
                // which previously made the output look like it contained newlines.
                const displayWidget = ComfyWidgets["STRING"](this, "Output Preview", ["STRING", { multiline: true }], app).widget;
                displayWidget.inputEl.readOnly = true;
                displayWidget.inputEl.style.height = "40px";
                displayWidget.inputEl.setAttribute("wrap", "off");
                displayWidget.inputEl.style.whiteSpace = "pre";
                displayWidget.inputEl.style.overflowX = "auto";
                displayWidget.inputEl.style.overflowY = "hidden";

                // Defensive strip: an upstream node might feed shot_name / ai_method with
                // embedded newlines, tabs, or carriage returns (e.g. a Text Multiline node
                // with a stray paragraph break). Those would silently break filesystem
                // writes — the path string can't legally contain them. Strip before display
                // so the preview matches what the Python node will actually write.
                const cleanLine = (s) => String(s ?? "").replace(/[\r\n\t]+/g, "");

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

                // Mirror of _normalize_base_path in k_ai_pipeline.py — must stay in sync.
                // Strips control chars (newlines/tabs from upstream text widgets), then
                // surrounding whitespace, one pair of surrounding quotes, and trailing
                // path separators. Drive roots (C:\, n:/) are preserved.
                const normalizeBasePath = (raw) => {
                    let s = cleanLine(raw).trim();
                    if (s.length >= 2 && s[0] === s[s.length - 1] && (s[0] === '"' || s[0] === "'")) {
                        s = s.slice(1, -1).trim();
                    }
                    while (s.length > 3 && (s[s.length - 1] === "/" || s[s.length - 1] === "\\")) {
                        s = s.slice(0, -1);
                    }
                    return s;
                };

                // Mirror of _sanitize_segment in k_ai_pipeline.py — strips control chars
                // first, then drive prefix and leading separators so a segment can only be
                // joined onto base_directory_path, never replace it via os.path.join's
                // "last absolute path wins" semantics. Leading seps are stripped BEFORE
                // splitdrive so multi-slash input doesn't get parsed as a UNC prefix
                // (matches Python's lstrip → splitdrive → lstrip flow).
                const sanitizeSegment = (raw) => {
                    let s = cleanLine(raw);
                    s = s.replace(/^[\/\\]+/, "");
                    if (/^[a-zA-Z]:/.test(s)) s = s.slice(2);
                    return s.replace(/^[\/\\]+/, "");
                };

                // Mirror of the Python directory build in k_ai_pipeline.py — must stay in sync.
                const buildOutputDirectory = (values, version_str) => {
                    const base = normalizeBasePath(values.base_directory_path);
                    const shotSeg = sanitizeSegment(values.shot_name);
                    const aiSeg = sanitizeSegment(values.ai_method);
                    let dir;
                    if (values.no_subfolders) {
                        dir = base;
                    } else {
                        dir = [base, shotSeg, aiSeg, version_str]
                            .filter(part => part.toString().trim() !== "")
                            .join("/");
                    }
                    dir = dir.replace(/\\/g, "/").replace(/\/+/g, '/');
                    // Strip trailing slash unless we'd be left with a bare drive root like 'n:/'.
                    if (dir.length > 3) {
                        dir = dir.replace(/\/+$/, '');
                    }
                    return dir;
                };

                this.addWidget("button", "Get output directory path", null, () => {
                    const values = {
                        base_directory_path: getEffectiveValue(this, "base_directory_path"),
                        shot_name: getEffectiveValue(this, "shot_name"),
                        ai_method: getEffectiveValue(this, "ai_method"),
                        version: getEffectiveValue(this, "version"),
                        disable_versioning: getEffectiveValue(this, "disable_versioning"),
                        enable_overwrite: getEffectiveValue(this, "enable_overwrite"),
                        no_subfolders: getEffectiveValue(this, "no_subfolders"),
                    };
                    if (Object.values(values).some(v => v === null)) {
                        displayWidget.value = "Cannot preview: complex inputs - execute node for accurate path.";
                    } else {
                        const version_str = values.disable_versioning ? "" : `v${values.version.toString().padStart(3, '0')}`;
                        displayWidget.value = buildOutputDirectory(values, version_str);
                    }
                    this.setDirtyCanvas(true, true);
                }, { serialize: false });

                this.addWidget("button", "Get output file path", null, () => {
                    const values = {
                        base_directory_path: getEffectiveValue(this, "base_directory_path"),
                        shot_name: getEffectiveValue(this, "shot_name"),
                        ai_method: getEffectiveValue(this, "ai_method"),
                        extension: getEffectiveValue(this, "extension"),
                        version: getEffectiveValue(this, "version"),
                        disable_versioning: getEffectiveValue(this, "disable_versioning"),
                        enable_overwrite: getEffectiveValue(this, "enable_overwrite"),
                        no_subfolders: getEffectiveValue(this, "no_subfolders"),
                    };
                    if (Object.values(values).some(v => v === null)) {
                        displayWidget.value = "Cannot preview: complex inputs - execute node for accurate path.";
                    } else {
                        const version_str = values.disable_versioning ? "" : `v${values.version.toString().padStart(3, '0')}`;
                        const output_directory = buildOutputDirectory(values, version_str);
                        // Filenames can't contain / or \ on any OS, so flatten any internal
                        // separators in shot_name / ai_method to `_`. Mirror of the Python
                        // filename build — directory build above keeps the separators so
                        // slash-delimited shot_name still becomes nested subfolders when
                        // no_subfolders=false.
                        const shotSegFlat = sanitizeSegment(values.shot_name).replace(/[\/\\]/g, "_");
                        const aiSegFlat = sanitizeSegment(values.ai_method).replace(/[\/\\]/g, "_");
                        const name = [shotSegFlat, aiSegFlat, version_str]
                            .filter(part => part.toString().trim() !== "")
                            .join("_") + cleanLine(values.extension);
                        // Filter empties before joining so an empty output_directory (empty base
                        // + no_subfolders=true) doesn't leak a spurious leading "/" into file_path.
                        const file_path = [output_directory, name]
                            .filter(p => p)
                            .join("/")
                            .replace(/\\/g, "/")
                            .replace(/\/+/g, '/');
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