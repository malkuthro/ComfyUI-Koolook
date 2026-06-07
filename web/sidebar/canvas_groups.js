// SPDX-FileCopyrightText: 2025-2026 Kforge Labs <https://github.com/malkuthro/ComfyUI-Koolook>
// SPDX-License-Identifier: GPL-3.0-only

function cloneJson(value) {
    return JSON.parse(JSON.stringify(value));
}

function idKey(id) {
    return id == null ? null : String(id);
}

function rectFromGroup(group) {
    if (!group || typeof group !== "object") return null;
    if (Array.isArray(group.bounding) && group.bounding.length >= 4) {
        return {
            x: Number(group.bounding[0]) || 0,
            y: Number(group.bounding[1]) || 0,
            w: Number(group.bounding[2]) || 0,
            h: Number(group.bounding[3]) || 0,
        };
    }
    if (Array.isArray(group.pos) && Array.isArray(group.size)) {
        return {
            x: Number(group.pos[0]) || 0,
            y: Number(group.pos[1]) || 0,
            w: Number(group.size[0]) || 0,
            h: Number(group.size[1]) || 0,
        };
    }
    return null;
}

function rectFromNode(node) {
    if (!node || typeof node !== "object" || !Array.isArray(node.pos)) return null;
    const size = Array.isArray(node.size) ? node.size : [200, 100];
    return {
        x: Number(node.pos[0]) || 0,
        y: Number(node.pos[1]) || 0,
        w: Number(size[0]) || 200,
        h: Number(size[1]) || 100,
    };
}

function rectsOverlap(a, b) {
    if (!a || !b || a.w <= 0 || a.h <= 0 || b.w <= 0 || b.h <= 0) return false;
    return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}

export function groupsForSelectedNodes(fullGraph, selectedKeys) {
    const groups = Array.isArray(fullGraph?.groups) ? fullGraph.groups : [];
    if (groups.length === 0 || !(selectedKeys instanceof Set) || selectedKeys.size === 0) {
        return [];
    }
    const selectedRects = (Array.isArray(fullGraph?.nodes) ? fullGraph.nodes : [])
        .filter(node => selectedKeys.has(idKey(node?.id)))
        .map(rectFromNode)
        .filter(Boolean);
    if (selectedRects.length === 0) return [];

    return groups
        .filter(group => {
            const groupRect = rectFromGroup(group);
            return selectedRects.some(nodeRect => rectsOverlap(groupRect, nodeRect));
        })
        .map(cloneJson);
}

export function translateGroups(groups, dx, dy) {
    if (!Array.isArray(groups) || groups.length === 0) return [];
    const tx = Number(dx) || 0;
    const ty = Number(dy) || 0;
    return groups.map(group => {
        const clone = cloneJson(group);
        if (Array.isArray(clone.bounding) && clone.bounding.length >= 2) {
            clone.bounding[0] = (Number(clone.bounding[0]) || 0) + tx;
            clone.bounding[1] = (Number(clone.bounding[1]) || 0) + ty;
        }
        if (Array.isArray(clone.pos) && clone.pos.length >= 2) {
            clone.pos[0] = (Number(clone.pos[0]) || 0) + tx;
            clone.pos[1] = (Number(clone.pos[1]) || 0) + ty;
        }
        return clone;
    });
}
