// SPDX-FileCopyrightText: 2025-2026 Kforge Labs <https://github.com/malkuthro/ComfyUI-Koolook>
// SPDX-License-Identifier: GPL-3.0-only

function numberOrDefault(value, fallback) {
    return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function rectFromGroup(group) {
    if (!group || typeof group !== "object") return null;
    if (Array.isArray(group.bounding) && group.bounding.length >= 4) {
        return {
            x: numberOrDefault(group.bounding[0], 0),
            y: numberOrDefault(group.bounding[1], 0),
            w: numberOrDefault(group.bounding[2], 0),
            h: numberOrDefault(group.bounding[3], 0),
        };
    }
    if (Array.isArray(group.pos) && Array.isArray(group.size)) {
        return {
            x: numberOrDefault(group.pos[0], 0),
            y: numberOrDefault(group.pos[1], 0),
            w: numberOrDefault(group.size[0], 0),
            h: numberOrDefault(group.size[1], 0),
        };
    }
    return null;
}

function rectFromNode(node) {
    if (!node || typeof node !== "object" || !Array.isArray(node.pos)) return null;
    const size = Array.isArray(node.size) ? node.size : [200, 100];
    return {
        x: numberOrDefault(node.pos[0], 0),
        y: numberOrDefault(node.pos[1], 0),
        w: numberOrDefault(size[0], 200),
        h: numberOrDefault(size[1], 100),
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

export function inferSetupSurface(graph) {
    return {
        sourceInputs: groupedNodes(graph, "Koolook Input"),
        outputs: groupedNodes(graph, "Koolook Output"),
        controls: [],
    };
}
