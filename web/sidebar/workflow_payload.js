// SPDX-FileCopyrightText: 2025-2026 Kforge Labs <https://github.com/malkuthro/ComfyUI-Koolook>
// SPDX-License-Identifier: GPL-3.0-only

function makeWorkflowId() {
    try {
        if (crypto && typeof crypto.randomUUID === "function") {
            return crypto.randomUUID();
        }
    } catch {
        // Fall through to the local fallback below.
    }
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
        const r = Math.floor(Math.random() * 16);
        const v = c === "x" ? r : (r & 0x3) | 0x8;
        return v.toString(16);
    });
}

function hash32(input, seed) {
    let h = (0x811c9dc5 ^ seed) >>> 0;
    for (let i = 0; i < input.length; i += 1) {
        h ^= input.charCodeAt(i);
        h = Math.imul(h, 0x01000193) >>> 0;
    }
    return h.toString(16).padStart(8, "0");
}

// Deterministic UUID-shaped key, not a spec UUIDv5. Comfy treats workflow ids
// as opaque draft-cache keys; stable shape is enough to replace per-workflow
// drafts instead of accumulating one browser-storage entry per sidebar load.
function stableWorkflowId(loadKey) {
    const key = String(loadKey);
    const hex =
        hash32(key, 0) +
        hash32(key, 0x9e3779b9) +
        hash32(key, 0x85ebca6b) +
        hash32(key, 0xc2b2ae35);
    return [
        hex.slice(0, 8),
        hex.slice(8, 12),
        `5${hex.slice(13, 16)}`,
        `8${hex.slice(17, 20)}`,
        hex.slice(20, 32),
    ].join("-");
}

export function cloneWorkflowForTemporaryLoad(graphData, loadKey = null) {
    const clone = JSON.parse(JSON.stringify(graphData || {}));
    clone.id = loadKey == null ? makeWorkflowId() : stableWorkflowId(loadKey);
    return clone;
}
