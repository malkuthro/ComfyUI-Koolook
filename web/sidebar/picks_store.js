// SPDX-FileCopyrightText: 2025-2026 Kforge Labs <https://github.com/malkuthro/ComfyUI-Koolook>
// SPDX-License-Identifier: GPL-3.0-only

// =============================================================================
// User picks (localStorage, browser-local persistence)
// =============================================================================
import {
    STORAGE_KEY,
    PICKS_CHANGED_EVENT,
    AUTOPULL_HIDDEN_KEY,
} from "./constants.js";

// Read-only render-source override (issue #181, Compare mode). When set, the
// render path (`loadUserPicks`) returns this list instead of the user's live
// localStorage picks, so the Compare view can render a *second* sidebar from a
// loaded snapshot without touching live state. Set only synchronously around a
// read-only render and always cleared afterwards (see `withSnapshotSource` in
// tree.js); never written back to localStorage.
let renderSourcePicks = null;

export function setPicksRenderSource(picks) {
    renderSourcePicks = Array.isArray(picks) ? picks.filter(p => typeof p === "string") : [];
}

export function clearPicksRenderSource() {
    renderSourcePicks = null;
}

export function loadUserPicks() {
    if (renderSourcePicks) return [...renderSourcePicks];
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return [];
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? parsed.filter(x => typeof x === "string") : [];
    } catch (e) {
        // Surface corruption so the user has a chance to spot it in the console
        // before the next saveUserPicks() overwrites the bad blob with a fresh list.
        console.warn("[Koolook] failed to parse user picks; returning empty list:", e);
        return [];
    }
}

function saveUserPicks(picks) {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(picks));
        return true;
    } catch (e) {
        console.warn("[Koolook] failed to save picks:", e);
        return false;
    }
}

// Tri-state: "added" | "duplicate" | "failed". Callers need to distinguish
// "save failed" from "already in list" so the user isn't told a write
// succeeded when localStorage rejected it (quota / private mode).
export function addToMyPicks(typeName) {
    if (!typeName) return "failed";
    // The `+` action always means "make this visible in favorites." If the
    // user previously hid this type via `×` (when it was auto-pulled by a
    // REPOS{select: "all"} entry), clear that hide as a side effect — so a
    // subsequent gather pass surfaces it again either via picks or via auto-
    // pull. Without this, `×` then `+` on the same Koolook node would do
    // nothing visible — the hide would silently outlast the re-add.
    unhideAutoPullType(typeName);
    const picks = loadUserPicks();
    if (picks.includes(typeName)) return "duplicate";
    picks.push(typeName);
    return saveUserPicks(picks) ? "added" : "failed";
}

export function removeFromMyPicks(typeName) {
    saveUserPicks(loadUserPicks().filter(p => p !== typeName));
}

// =============================================================================
// Auto-pull hidden set — types the user has explicitly excluded from the
// REPOS{select: "all"} auto-pull. See AUTOPULL_HIDDEN_KEY in constants.js.
// =============================================================================
export function loadAutoPullHidden() {
    try {
        const raw = localStorage.getItem(AUTOPULL_HIDDEN_KEY);
        if (!raw) return new Set();
        const parsed = JSON.parse(raw);
        return new Set(Array.isArray(parsed) ? parsed.filter(x => typeof x === "string") : []);
    } catch (e) {
        console.warn("[Koolook] failed to parse autoPullHidden; starting empty:", e);
        return new Set();
    }
}

function saveAutoPullHidden(set) {
    try {
        localStorage.setItem(AUTOPULL_HIDDEN_KEY, JSON.stringify([...set]));
        return true;
    } catch (e) {
        console.warn("[Koolook] failed to save autoPullHidden:", e);
        return false;
    }
}

export function hideAutoPullType(typeName) {
    if (!typeName) return;
    const set = loadAutoPullHidden();
    if (set.has(typeName)) return;
    set.add(typeName);
    saveAutoPullHidden(set);
}

export function unhideAutoPullType(typeName) {
    if (!typeName) return;
    const set = loadAutoPullHidden();
    if (!set.delete(typeName)) return;
    saveAutoPullHidden(set);
}

// Bulk-replace the entire pick list. Used by the snapshot import flow —
// the caller has just deserialized a preset and wants picks to mirror it
// exactly. Filters non-strings defensively (snapshot files are user-
// editable and may have been hand-touched). Returns true on persist
// success, false on localStorage rejection (quota / private mode).
export function setAllPicks(picks) {
    const cleaned = Array.isArray(picks)
        ? picks.filter(p => typeof p === "string")
        : [];
    return saveUserPicks(cleaned);
}

export function notifyPicksChanged() {
    window.dispatchEvent(new CustomEvent(PICKS_CHANGED_EVENT));
}

