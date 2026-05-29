// SPDX-FileCopyrightText: 2025-2026 Kforge Labs <https://github.com/malkuthro/ComfyUI-Koolook>
// SPDX-License-Identifier: GPL-3.0-only

import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

globalThis.__KOLOOK_AI_PIPELINE_PREVIEW_RESOLVER__ = "easyuse-getset-v2";

// Pysssss's presetText.js wraps every STRING widget's serializeValue with
// a callback chain that does `value.replace(...)` for {variable} substitution.
// If `widget.value` is ever undefined or null (stale pre-PR#180 widgets_values
// slot, widget-to-input conversion that voids the value, or a missing key
// in dict-based widgets_values), `.replace` throws and the entire
// graphToPrompt aborts — Run becomes a no-op for the whole workflow.
//
// Defensive coercion: trap reads/writes on widget.value so the property is
// always a string, no matter who touches it. Cheap, survives upstream
// widget patches because we wrap the property descriptor itself.
function bulletproofStringWidget(widget, fallback = "") {
    if (!widget) return;
    let stored;
    const initial = widget.value;
    if (typeof initial === "string") {
        stored = initial;
    } else if (initial == null) {
        stored = fallback;
    } else {
        stored = String(initial);
    }
    Object.defineProperty(widget, "value", {
        configurable: true,
        enumerable: true,
        get() { return stored; },
        set(v) {
            if (typeof v === "string") stored = v;
            else if (v == null) stored = fallback;
            else stored = String(v);
        },
    });
}

app.registerExtension({
    name: "koolook.ai_pipeline",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "EasyAIPipeline") {
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                if (originalOnNodeCreated) {
                    originalOnNodeCreated.call(this);
                }

                // Guard the version STRING widget. Pre-PR#180 workflows save
                // an INT here; widget-to-input conversion can void the value
                // to undefined. Either trips pysssss's .replace callback.
                bulletproofStringWidget(this.widgets.find(w => w.name === "version"), "v001");

                // Make instruction widget read-only and dimmed
                const instructionWidget = this.widgets.find(w => w.name === "instruction");
                if (instructionWidget && instructionWidget.inputEl) {
                    instructionWidget.inputEl.readOnly = true;
                    instructionWidget.inputEl.style.opacity = 0.6;
                }

                // Standard ComfyUI multi-line STRING widget. Word-wrap is the default
                // and intentional — long paths visually wrap to multiple rows so the user
                // can mouse-drag-select the whole string and copy it. The wrap is purely
                // visual; the underlying string is one logical line because cleanLine
                // strips any actual newlines from upstream inputs before the path build.
                const displayWidget = ComfyWidgets["STRING"](this, "Output Preview", ["STRING", { multiline: true }], app).widget;
                displayWidget.inputEl.readOnly = true;
                displayWidget.inputEl.style.height = "100px";

                // Defensive strip: an upstream node might feed shot_name / ai_method with
                // embedded newlines, tabs, or carriage returns (e.g. a Text Multiline node
                // with a stray paragraph break). Those would silently break filesystem
                // writes — the path string can't legally contain them. Strip before display
                // so the preview matches what the Python node will actually write.
                const cleanLine = (s) => String(s ?? "").replace(/[\r\n\t]+/g, "");

                const SENTINEL_STRINGS = new Set(["undefined", "null", "none"]);

                // Mirror of _normalize_text_input in k_ai_pipeline.py.
                const normalizeTextInput = (raw) => {
                    let s = cleanLine(raw).trim();
                    if (s.length >= 2 && s[0] === s[s.length - 1] && (s[0] === '"' || s[0] === "'")) {
                        s = s.slice(1, -1).trim();
                    }
                    if (SENTINEL_STRINGS.has(s.toLowerCase())) {
                        return "";
                    }
                    return s;
                };

                // Strict: extension must never contain whitespace at all. Even a single
                // trailing space (easy to acquire from a paste) makes downstream save
                // nodes that check via os.path.splitext fail with "filepath doesn't end
                // in .exr" — splitext returns ".exr " with the space, which doesn't match.
                const cleanExtension = (s) => String(s ?? "").replace(/\s/g, "");

                const getNodeWidgetValue = (node, slot) => {
                    const output = node.outputs?.[slot];
                    const candidateWidgets = [
                        node.widgets?.[slot],
                        output?.name ? node.widgets?.find(w => w.name === output.name) : null,
                        node.widgets?.find(w => w.name === "value"),
                        node.widgets?.find(w => w.name === "text"),
                        node.widgets?.find(w => w.name === "string"),
                    ].filter(Boolean);
                    for (const widget of candidateWidgets) {
                        if (widget && "value" in widget) {
                            return widget.value;
                        }
                    }
                    if (Array.isArray(node.widgets_values) && node.widgets_values.length > 0) {
                        return node.widgets_values[slot] ?? node.widgets_values[0];
                    }
                    return null;
                };

                const getWidgetValueByName = (node, name, fallback = null) => {
                    const widget = node?.widgets?.find(w => w.name === name);
                    if (widget && "value" in widget) return widget.value;
                    if (Array.isArray(node?.widgets_values)) {
                        const inputIndex = node.inputs?.findIndex(input => input.name === name);
                        if (inputIndex >= 0 && node.widgets_values[inputIndex] !== undefined) {
                            return node.widgets_values[inputIndex];
                        }
                    }
                    return fallback;
                };

                const getLink = (graph, linkId) => {
                    if (!graph || linkId === null || linkId === undefined) return null;
                    if (typeof graph.getLink === "function") return graph.getLink(linkId);
                    const links = graph.links ?? graph._links;
                    if (links instanceof Map) return links.get(linkId) ?? null;
                    return links?.[linkId] ?? null;
                };

                const getGraphNodeById = (graph, nodeId) => {
                    if (!graph || nodeId === null || nodeId === undefined) return null;
                    if (typeof graph.getNodeById === "function") return graph.getNodeById(nodeId);
                    return graph._nodes?.find(n => n.id === nodeId) ?? null;
                };

                const linkOriginId = (link) => link?.origin_id ?? link?.[1];
                const linkOriginSlot = (link) => link?.origin_slot ?? link?.[2];
                const graphIds = new WeakMap();
                let nextGraphId = 1;
                const graphSeenKey = (graph, linkId) => {
                    if (!graph || (typeof graph !== "object" && typeof graph !== "function")) {
                        return `graph:${linkId}`;
                    }
                    if (!graphIds.has(graph)) {
                        graphIds.set(graph, nextGraphId++);
                    }
                    return `${graphIds.get(graph)}:${linkId}`;
                };

                const isEasyUseGetNode = (node) => {
                    return node?.type === "easy getNode"
                        || node?.comfyClass === "easy getNode"
                        || (String(node?.title ?? "").startsWith("Get_") && node?.widgets?.[0]?.name === "Constant");
                };

                const findEasyUseSetter = (getNode, graph) => {
                    if (typeof getNode.findSetter === "function") {
                        const setter = getNode.findSetter(graph || getNode.graph || app.graph);
                        if (setter) return setter;
                    }
                    const key = getNode.widgets?.[0]?.value;
                    if (!key) return null;
                    return (graph || getNode.graph || app.graph)?._nodes?.find(otherNode => {
                        return (otherNode.type === "easy setNode"
                                || otherNode.comfyClass === "easy setNode"
                                || String(otherNode.title ?? "").startsWith("Set_"))
                            && otherNode.widgets?.[0]?.value === key;
                    }) ?? null;
                };

                const resolveSubgraphInputValue = (hostNode, inputSlot, seen) => {
                    if (!hostNode) return null;
                    const hostInput = hostNode.inputs?.[inputSlot];
                    if (!hostInput) return null;
                    if (hostInput.link !== null && hostInput.link !== undefined) {
                        return resolveLinkValue(hostInput.link, null, seen, hostNode.graph || app.graph, null);
                    }
                    const widgetName = hostInput.widget?.name ?? hostInput.name;
                    return getWidgetValueByName(hostNode, widgetName, getNodeWidgetValue(hostNode, inputSlot));
                };

                const resolveSubgraphOutputValue = (hostNode, outputSlot, seen) => {
                    const subgraph = hostNode?.subgraph;
                    const output = subgraph?.outputs?.[outputSlot];
                    const linkIds = output?.linkIds ?? output?.links;
                    const innerLinkId = Array.isArray(linkIds) ? linkIds[0] : linkIds;
                    if (innerLinkId === null || innerLinkId === undefined) return null;
                    return resolveLinkValue(innerLinkId, null, seen, subgraph, hostNode);
                };

                const evaluateKnownNodeOutput = (node, slot, graph, seen, subgraphHost) => {
                    if (node?.type !== "Easy_Utility" || slot !== 0) return null;
                    const mode = getWidgetValueByName(node, "mode", node.widgets_values?.[0] ?? "int_to_padded_string");
                    if (mode !== "int_to_padded_string") return null;

                    const intInput = node.inputs?.find(input => input.name === "int_value");
                    let intValue;
                    if (intInput?.link !== null && intInput?.link !== undefined) {
                        intValue = resolveLinkValue(intInput.link, null, seen, graph, subgraphHost);
                    }
                    if (intValue === null || intValue === undefined) {
                        intValue = getWidgetValueByName(node, "int_value", node.widgets_values?.[1] ?? 1);
                    }
                    const prefix = getWidgetValueByName(node, "prefix", node.widgets_values?.[2] ?? "");
                    const padWidth = getWidgetValueByName(node, "pad_width", node.widgets_values?.[3] ?? 3);
                    const n = Number.parseInt(intValue, 10);
                    const w = Math.max(0, Number.parseInt(padWidth, 10) || 0);
                    if (Number.isNaN(n)) return null;
                    return `${prefix ?? ""}${String(n).padStart(w, "0")}`;
                };

                const resolveLinkValue = (linkId, fallback, seen = new Set(), graph = app.graph, subgraphHost = null) => {
                    const seenKey = graphSeenKey(graph, linkId);
                    if (linkId === null || linkId === undefined || seen.has(seenKey)) {
                        return fallback;
                    }
                    seen.add(seenKey);
                    const link = getLink(graph, linkId);
                    if (!link) return fallback;

                    const originId = linkOriginId(link);
                    const originSlot = linkOriginSlot(link);
                    if (originId === -10) {
                        const value = resolveSubgraphInputValue(subgraphHost, originSlot, seen);
                        return value ?? fallback;
                    }

                    const originNode = getGraphNodeById(graph, originId);
                    if (!originNode) return fallback;

                    const subgraphValue = resolveSubgraphOutputValue(originNode, originSlot, seen);
                    if (subgraphValue !== null && subgraphValue !== undefined) return subgraphValue;

                    const knownValue = evaluateKnownNodeOutput(originNode, originSlot, graph, seen, subgraphHost);
                    if (knownValue !== null && knownValue !== undefined) return knownValue;

                    // EasyUse GET nodes are virtual tunnels. The visible widget is the
                    // key name ("OUT-folder"), not the value. Follow GET -> matching SET
                    // -> SET input link so preview buttons mirror render-time execution.
                    if (isEasyUseGetNode(originNode)) {
                        const setter = findEasyUseSetter(originNode, graph);
                        const setterLink = setter?.inputs?.[0]?.link;
                        if (setterLink !== null && setterLink !== undefined) {
                            const resolved = resolveLinkValue(setterLink, null, seen, graph, subgraphHost);
                            if (resolved !== null && resolved !== undefined) return resolved;
                        }
                    }

                    return getNodeWidgetValue(originNode, originSlot);
                };

                // Helper function to get effective value (from widget or upstream if connected and simple)
                const getEffectiveValue = (node, name) => {
                    const widget = node.widgets.find(w => w.name === name);
                    const input = node.inputs?.find(i => i.name === name);
                    if (!input || input.link === null) {
                        return widget?.value;
                    }
                    return resolveLinkValue(input.link, null);
                };

                // Mirror of _normalize_base_path in k_ai_pipeline.py — must stay in sync.
                // Strips control chars (newlines/tabs from upstream text widgets), then
                // surrounding whitespace, one pair of surrounding quotes, and trailing
                // path separators. Drive roots (C:\, n:/) are preserved.
                const normalizeBasePath = (raw) => {
                    let s = normalizeTextInput(raw);
                    while (s.length > 3 && (s[s.length - 1] === "/" || s[s.length - 1] === "\\")) {
                        s = s.slice(0, -1);
                    }
                    return stripSentinelComponents(s);
                };

                const stripSentinelComponents = (path) => {
                    if (!path) return path;
                    const sep = path.includes("\\") ? "\\" : "/";
                    const cleaned = path
                        .split(sep)
                        .filter(part => part === "" || !SENTINEL_STRINGS.has(part.toLowerCase()));
                    return cleaned.join(sep);
                };

                // Mirror of _sanitize_segment in k_ai_pipeline.py — strips control chars
                // and surrounding whitespace first, then drive prefix and leading
                // separators so a segment can only be joined onto base_directory_path,
                // never replace it via os.path.join's "last absolute path wins" semantics.
                // Leading seps are stripped BEFORE splitdrive so multi-slash input
                // doesn't get parsed as a UNC prefix (matches Python's flow).
                const sanitizeSegment = (raw) => {
                    let s = normalizeTextInput(raw);
                    s = s.replace(/^[\/\\]+/, "");
                    if (/^[a-zA-Z]:/.test(s)) s = s.slice(2);
                    return s.replace(/^[\/\\]+/, "");
                };

                // Mirror of the Python directory build in k_ai_pipeline.py — must stay in sync.
                // With no_subfolders=true: only [base, version_str] go into the directory;
                // shot_name and ai_method drop out (they're only in the filename below).
                // The version folder still applies when versioning is enabled. With
                // no_subfolders=false: full [base, shot, ai, version] chain.
                const buildOutputDirectory = (values, version_str) => {
                    const base = normalizeBasePath(values.base_directory_path);
                    const shotSeg = sanitizeSegment(values.shot_name);
                    const aiSeg = sanitizeSegment(values.ai_method);
                    let dir;
                    if (values.no_subfolders) {
                        dir = [base, version_str]
                            .filter(part => part.toString().trim() !== "")
                            .join("/");
                    } else {
                        dir = [base, shotSeg, aiSeg, version_str]
                            .filter(part => part.toString().trim() !== "")
                            .join("/");
                    }
                    dir = dir.replace(/\\/g, "/").replace(/\/+/g, '/');
                    dir = stripSentinelComponents(dir);
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
                            .join("_") + cleanExtension(values.extension);
                        // Filter empties before joining so an empty output_directory (empty base
                        // + no_subfolders=true) doesn't leak a spurious leading "/" into file_path.
                        const file_path = [output_directory, name]
                            .filter(p => p)
                            .join("/")
                            .replace(/\\/g, "/")
                            .replace(/\/+/g, '/');
                        displayWidget.value = stripSentinelComponents(file_path);
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
