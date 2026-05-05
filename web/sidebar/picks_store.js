// =============================================================================
// User picks (localStorage, browser-local persistence)
// =============================================================================
import {
    STORAGE_KEY,
    PICKS_CHANGED_EVENT,
} from "./constants.js";

export function loadUserPicks() {
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
    const picks = loadUserPicks();
    if (picks.includes(typeName)) return "duplicate";
    picks.push(typeName);
    return saveUserPicks(picks) ? "added" : "failed";
}

export function removeFromMyPicks(typeName) {
    saveUserPicks(loadUserPicks().filter(p => p !== typeName));
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

