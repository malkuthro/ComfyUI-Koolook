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

export function cloneWorkflowForTemporaryLoad(graphData) {
    const clone = JSON.parse(JSON.stringify(graphData || {}));
    clone.id = makeWorkflowId();
    return clone;
}
