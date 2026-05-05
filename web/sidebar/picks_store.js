// =============================================================================
// User picks (localStorage, browser-local persistence)
// =============================================================================
import {
    STORAGE_KEY,
    SEEDED_KEY,
    PICKS_CHANGED_EVENT,
    DEFAULTS_URL,
    toast,
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

export async function seedDefaultsIfNeeded() {
    if (localStorage.getItem(SEEDED_KEY)) return;
    if (loadUserPicks().length > 0) {
        localStorage.setItem(SEEDED_KEY, "1");
        return;
    }
    try {
        const resp = await fetch(DEFAULTS_URL);
        if (!resp.ok) {
            localStorage.setItem(SEEDED_KEY, "1");
            return;
        }
        const data = await resp.json();
        const picks = (data && Array.isArray(data.picks))
            ? data.picks.filter(p => typeof p === "string")
            : [];
        if (picks.length > 0) {
            // Only mark seeded if the save actually landed — otherwise the next
            // page load retries instead of being permanently blocked by the flag.
            if (!saveUserPicks(picks)) {
                console.warn("[Koolook] seed save failed; will retry on next load");
                // Surface the failure in the UI too — the console-only path
                // is invisible to anyone who hasn't opened DevTools, and the
                // most common cause (full or quota-restricted localStorage)
                // is something the user can act on.
                toast("Could not seed default favorites — localStorage write rejected. See console.");
                return;
            }
        }
        localStorage.setItem(SEEDED_KEY, "1");
        console.log(`[Koolook] seeded ${picks.length} default pick(s)`);
    } catch (e) {
        console.warn("[Koolook] failed to load curated_defaults.json:", e);
        localStorage.setItem(SEEDED_KEY, "1");
    }
}

function picksAsDistributionJSON() {
    const picks = [...loadUserPicks()].sort();
    return JSON.stringify({ picks }, null, 2) + "\n";
}

export async function exportPicks() {
    const picks = loadUserPicks();
    if (picks.length === 0) {
        toast("No picks to export yet.");
        return;
    }
    const json = picksAsDistributionJSON();
    const noun = picks.length === 1 ? "pick" : "picks";
    try {
        await navigator.clipboard.writeText(json);
        toast(`Copied ${picks.length} ${noun} to clipboard. Paste into web/curated_defaults.json.`);
        return;
    } catch (e) {
        console.warn("[Koolook] clipboard write failed, falling back to download:", e);
    }
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "curated_defaults.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    toast(`Downloaded ${picks.length} ${noun} as curated_defaults.json. Replace the file in web/ with it.`);
}
