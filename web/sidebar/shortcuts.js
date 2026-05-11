// SPDX-FileCopyrightText: 2025-2026 Kforge Labs <https://github.com/malkuthro/ComfyUI-Koolook>
// SPDX-License-Identifier: GPL-3.0-only

const SHORTCUTS_KEY = "koolook-shortcuts.v1";

export const DEFAULT_SHORTCUTS = {
    focusSearch: "Ctrl+K",
    saveSnapshot: "Ctrl+S",
    loadSnapshot: "Ctrl+L",
    quickSaveSnapshot: "Ctrl+Shift+S",
    openHelp: "Ctrl+H",
    openShortcutSettings: "Ctrl+Alt+K",
};

const ACTIONS = new Set(Object.keys(DEFAULT_SHORTCUTS));

export function getShortcutMap() {
    try {
        const raw = localStorage.getItem(SHORTCUTS_KEY);
        if (!raw) return { ...DEFAULT_SHORTCUTS };
        const parsed = JSON.parse(raw);
        const next = { ...DEFAULT_SHORTCUTS };
        for (const [k, v] of Object.entries(parsed || {})) {
            if (!ACTIONS.has(k)) continue;
            if (typeof v !== "string" || !v.trim()) continue;
            next[k] = normalizeCombo(v);
        }
        return next;
    } catch {
        return { ...DEFAULT_SHORTCUTS };
    }
}

export function saveShortcutMap(map) {
    const out = {};
    for (const k of ACTIONS) {
        const v = map && typeof map[k] === "string" ? map[k].trim() : "";
        if (!v) continue;
        out[k] = normalizeCombo(v);
    }
    localStorage.setItem(SHORTCUTS_KEY, JSON.stringify(out));
}

export function resetShortcuts() {
    try { localStorage.removeItem(SHORTCUTS_KEY); } catch { /* noop */ }
    return { ...DEFAULT_SHORTCUTS };
}

export function normalizeCombo(combo) {
    if (!combo || typeof combo !== "string") return "";
    const parts = combo.split("+").map((p) => p.trim()).filter(Boolean);
    let ctrl = false; let alt = false; let shift = false; let meta = false;
    let key = "";
    for (const raw of parts) {
        const p = raw.toLowerCase();
        if (p === "ctrl" || p === "control") ctrl = true;
        else if (p === "alt" || p === "option") alt = true;
        else if (p === "shift") shift = true;
        else if (p === "meta" || p === "cmd" || p === "command") meta = true;
        else key = humanKey(raw);
    }
    if (!key) return "";
    const out = [];
    if (ctrl) out.push("Ctrl");
    if (alt) out.push("Alt");
    if (shift) out.push("Shift");
    if (meta) out.push("Meta");
    out.push(humanKey(key));
    return out.join("+");
}

function humanKey(k) {
    if (!k) return "";
    const key = String(k).trim();
    if (key.length === 1) return key.toUpperCase();
    return key[0].toUpperCase() + key.slice(1).toLowerCase();
}

function comboFromEvent(e) {
    const key = humanKey(e.key === " " ? "Space" : e.key);
    if (!key || ["Control", "Alt", "Shift", "Meta"].includes(key)) return "";
    const out = [];
    if (e.ctrlKey) out.push("Ctrl");
    if (e.altKey) out.push("Alt");
    if (e.shiftKey) out.push("Shift");
    if (e.metaKey) out.push("Meta");
    out.push(key);
    return out.join("+");
}

export function actionFromKeyboardEvent(e, map) {
    const combo = comboFromEvent(e);
    if (!combo) return null;
    for (const [action, binding] of Object.entries(map || {})) {
        if (normalizeCombo(binding) === combo) return action;
    }
    return null;
}

export function isTypingTarget(target) {
    if (!target || !(target instanceof Element)) return false;
    const tag = (target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select") return true;
    if (target.isContentEditable) return true;
    return Boolean(target.closest("input, textarea, select, [contenteditable='true']"));
}
