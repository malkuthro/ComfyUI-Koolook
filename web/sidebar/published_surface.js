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
