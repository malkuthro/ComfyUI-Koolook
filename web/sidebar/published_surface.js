// SPDX-FileCopyrightText: 2025-2026 Kforge Labs <https://github.com/malkuthro/ComfyUI-Koolook>
// SPDX-License-Identifier: GPL-3.0-only

function isNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
}

function rectFromGroup(group) {
    if (!group || typeof group !== "object") return null;
    if (Array.isArray(group.bounding) && group.bounding.length >= 4) {
        if (!group.bounding.slice(0, 4).every(isNumber)) return null;
        return {
            x: group.bounding[0],
            y: group.bounding[1],
            w: group.bounding[2],
            h: group.bounding[3],
        };
    }
    if (Array.isArray(group.pos) && Array.isArray(group.size)) {
        if (!group.pos.slice(0, 2).every(isNumber) || !group.size.slice(0, 2).every(isNumber)) {
            return null;
        }
        return {
            x: group.pos[0],
            y: group.pos[1],
            w: group.size[0],
            h: group.size[1],
        };
    }
    return null;
}

function rectFromNode(node) {
    if (!node || typeof node !== "object" || !Array.isArray(node.pos)) return null;
    const size = Array.isArray(node.size) ? node.size : [200, 100];
    if (!node.pos.slice(0, 2).every(isNumber) || !size.slice(0, 2).every(isNumber)) {
        return null;
    }
    return {
        x: node.pos[0],
        y: node.pos[1],
        w: size[0],
        h: size[1],
    };
}

function rectsOverlap(a, b) {
    if (!a || !b || a.w <= 0 || a.h <= 0 || b.w <= 0 || b.h <= 0) return false;
    return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}

function nodeSummary(node) {
    const type = typeof node?.type === "string" ? node.type : "";
    const title = typeof node?.title === "string" && node.title ? node.title : (type || String(node?.id));
    return {
        id: String(node?.id),
        type,
        title,
    };
}

const PUBLISH_INPUT_CLASS = "Koolook_PublishInput";
const PUBLISH_OUTPUT_CLASS = "Koolook_PublishOutput";
const PUBLISH_RESULT_CLASS = "Koolook_PublishResult";
const WIDGET_NAMES_BY_CLASS = {
    [PUBLISH_INPUT_CLASS]: ["mode", "sequence_folder", "qt_file", "single_file", "prompt"],
    [PUBLISH_OUTPUT_CLASS]: ["folder", "name", "version", "output_mode"],
    [PUBLISH_RESULT_CLASS]: ["result"],
};
const PUBLISH_INPUT_FIELDS = [
    ["sequence_folder", "Sequence folder", true],
    ["qt_file", "QT file", true],
    ["single_file", "Single file", true],
];
// The prompt is an always-on field, independent of the EXR/QT/Img source
// switch: an external user picks a source AND can describe the shot. The
// author's prompt-widget text becomes the placeholder hint; the submitted
// default is empty so an untouched hint is never sent as the real prompt.
const PUBLISH_PROMPT_HELP =
    "Describe the shot in one simple line: subject + action + setting.";
const PUBLISH_INPUT_MODES = [
    [0, "EXR", "sequence_folder"],
    [1, "QT", "qt_file"],
    [2, "Img", "single_file"],
    [3, "Prompt", "prompt"],
];
const PUBLISH_OUTPUT_FIELDS = [
    ["folder", "Output folder", true],
    ["name", "Output name", true],
    ["version", "Version", true],
];
// Output-type modes are a subset of the input modes (no Prompt — not a real
// output format). Indices match PUBLISH_INPUT_MODES so a router slot means the
// same type on either switch. -1 is the "Same as input" sentinel default.
const PUBLISH_OUTPUT_MODES = [
    [0, "EXR"],
    [1, "QT"],
    [2, "Img"],
];
const PUBLISH_OUTPUT_SAME_AS_INPUT = -1;
const PUBLISH_RESULT_FIELDS = [
    ["result", "Result", true],
];

function groupedNodes(graph, title) {
    const groups = Array.isArray(graph?.groups) ? graph.groups : [];
    const nodes = Array.isArray(graph?.nodes) ? graph.nodes : [];
    const nodeRects = nodes
        .filter(node => node && typeof node === "object")
        .map(node => ({ node, rect: rectFromNode(node) }));

    return groups
        .filter(group => group && typeof group === "object" && group.title === title)
        .map(group => ({
            group: title,
            nodes: nodeRects
                .filter(({ rect }) => rectsOverlap(rectFromGroup(group), rect))
                .map(({ node }) => nodeSummary(node)),
        }))
        .filter(entry => entry.nodes.length > 0);
}

function nodesInGroup(graph, title) {
    const groups = Array.isArray(graph?.groups) ? graph.groups : [];
    const nodes = Array.isArray(graph?.nodes) ? graph.nodes : [];
    const nodeRects = nodes
        .filter(node => node && typeof node === "object")
        .map(node => ({ node, rect: rectFromNode(node) }));

    const out = [];
    for (const group of groups) {
        if (!group || typeof group !== "object" || group.title !== title) continue;
        const groupRect = rectFromGroup(group);
        for (const { node, rect } of nodeRects) {
            if (rectsOverlap(groupRect, rect)) out.push(node);
        }
    }
    return out;
}

function firstNodeOfType(nodes, classType) {
    return nodes.find(node => node?.type === classType) || null;
}

function widgetValue(node, key) {
    const values = node?.widgets_values;
    if (values && typeof values === "object" && !Array.isArray(values)) return values[key];
    if (!Array.isArray(values)) return null;
    const names = WIDGET_NAMES_BY_CLASS[node?.type] || [];
    const index = names.indexOf(key);
    return index >= 0 && index < values.length ? values[index] : null;
}

function fieldSpecs(node, specs) {
    if (!node) return [];
    return specs.map(([key, label, visible]) => ({
        key,
        label,
        visible,
        target: { node: String(node.id), input: key },
        default: widgetValue(node, key),
    }));
}

function inputModeIndex(value) {
    if (Number.isInteger(value)) return value;
    const text = String(value ?? "").trim().toLowerCase();
    const found = PUBLISH_INPUT_MODES.find(([_value, label]) => label.toLowerCase() === text);
    return found ? found[0] : 2;
}

function inputSwitch(node, inputs) {
    const inputsByKey = Object.fromEntries(inputs.map(item => [item.key, item]));
    return {
        key: "switch",
        label: "Input type",
        visible: true,
        target: { node: String(node.id), input: "mode" },
        default: inputModeIndex(widgetValue(node, "mode")),
        options: PUBLISH_INPUT_MODES.map(([value, label, input]) => ({
            value,
            label,
            // Standalone fields (e.g. the always-on prompt) are not source
            // modes, so they never appear as a switch option.
            visible: Boolean(inputsByKey[input]?.visible) && !inputsByKey[input]?.standalone,
            input,
        })),
    };
}

function outputModeIndex(value) {
    if (Number.isInteger(value)) return value;
    const text = String(value ?? "").trim().toLowerCase();
    const found = PUBLISH_OUTPUT_MODES.find(([_value, label]) => label.toLowerCase() === text);
    return found ? found[0] : PUBLISH_OUTPUT_SAME_AS_INPUT;
}

function outputSwitch(node) {
    return {
        key: "output_switch",
        label: "Output type",
        visible: true,
        sameAsInput: true,
        target: { node: String(node.id), input: "output_mode" },
        default: outputModeIndex(widgetValue(node, "output_mode")),
        options: PUBLISH_OUTPUT_MODES.map(([value, label]) => ({
            value,
            label,
            visible: true,
        })),
    };
}

function promptField(node) {
    if (!node) return null;
    const hint = widgetValue(node, "prompt");
    return {
        key: "prompt",
        label: "Prompt",
        visible: true,
        standalone: true,
        multiline: true,
        target: { node: String(node.id), input: "prompt" },
        default: "",
        placeholder: typeof hint === "string" ? hint : "",
        help: PUBLISH_PROMPT_HELP,
    };
}

function inferAppSurface(graph) {
    const inputNode = firstNodeOfType(nodesInGroup(graph, "Koolook Input"), PUBLISH_INPUT_CLASS);
    const outputNodes = nodesInGroup(graph, "Koolook Output");
    const outputNode = firstNodeOfType(outputNodes, PUBLISH_OUTPUT_CLASS);
    const resultNode = firstNodeOfType(outputNodes, PUBLISH_RESULT_CLASS);
    const inputs = fieldSpecs(inputNode, PUBLISH_INPUT_FIELDS);
    const prompt = promptField(inputNode);
    if (prompt) inputs.push(prompt);
    const app = {
        inputs,
        outputs: fieldSpecs(outputNode, PUBLISH_OUTPUT_FIELDS),
        results: fieldSpecs(resultNode, PUBLISH_RESULT_FIELDS),
    };
    if (inputNode) app.switch = inputSwitch(inputNode, inputs);
    if (outputNode) app.outputSwitch = outputSwitch(outputNode);
    return app;
}

export function inferSetupSurface(graph) {
    return {
        sourceInputs: groupedNodes(graph, "Koolook Input"),
        outputs: groupedNodes(graph, "Koolook Output"),
        controls: [],
        app: inferAppSurface(graph),
    };
}

// Review-preview labels for ComfyUI App-builder picks (extra.linearData).
// The server converts resolvable picks into declared param fields at publish
// time (koolook_setups._app_builder_param_fields is authoritative — it also
// checks the API prompt); this only lists what the author picked so the
// publish dialog can show them for review.
export function appBuilderParamLabels(graph) {
    const picks = graph?.extra?.linearData?.inputs;
    if (!Array.isArray(picks)) return [];
    const byId = new Map();
    for (const node of Array.isArray(graph?.nodes) ? graph.nodes : []) {
        if (node && node.id != null) byId.set(String(node.id), node);
    }
    const labels = [];
    for (const pick of picks) {
        if (!Array.isArray(pick) || pick.length < 2) continue;
        const nodeId = String(pick[0]).trim();
        const widget = String(pick[1]).trim();
        if (!nodeId || !widget) continue;
        const node = byId.get(nodeId);
        const title = (node?.title || node?.type || `Node ${nodeId}`);
        labels.push(`${title}: ${widget}`);
    }
    return labels;
}
