// =============================================================================
// Snapshot / preset library (#46 item 1).
//
// A "snapshot" is a single JSON file that captures the full Kforge Labs state
// — curated picks + the entire workflows store — under a user-chosen name.
// Multiple snapshots live side-by-side as files inside ComfyUI's userdata
// at /userdata/koolook-presets/<name>.json, so they're trivially shareable
// across machines (copy the folder, sync via Dropbox/iCloud/Drive, or use
// the per-snapshot Download/Upload actions in the Manage tab).
//
// Schema:
//   {
//     "kind": "koolook-snapshot",
//     "version": 1,
//     "name": "Wan video kit",
//     "exportedAt": "2026-05-05T08:42:13.000Z",
//     "picks": ["EasyResize_Koolook", ...],
//     "workflows": { "directories": { ... } }
//   }
//
// `kind` + `version` are dispatch fields for future migrations. Anything
// without `kind === "koolook-snapshot"` is rejected on import — keeps a
// stray unrelated JSON file from being applied as state.
// =============================================================================
import { loadUserPicks, setAllPicks, notifyPicksChanged } from "./picks_store.js";
import { replaceAllWorkflows, getAllWorkflowsForExport } from "./workflows_store.js";

export const SNAPSHOT_KIND = "koolook-snapshot";
export const SNAPSHOT_VERSION = 1;
export const PRESETS_DIR = "koolook-presets";
const USERDATA_OVERWRITE_QUERY = "?overwrite=true";

// =============================================================================
// Filename hygiene
// =============================================================================

// User-typed names need to survive filesystem + URL roundtripping. Strip
// path separators and control characters, collapse runs of whitespace, trim.
// Empty after normalization → null (caller should reject).
export function sanitizeName(raw) {
    if (typeof raw !== "string") return null;
    const cleaned = raw
        .replace(/[\\/]+/g, "-")            // path separators
        // eslint-disable-next-line no-control-regex
        .replace(/[\x00-\x1f\x7f]/g, "")     // control chars
        .replace(/\s+/g, " ")
        .trim();
    return cleaned || null;
}

// =============================================================================
// Capture / apply (the in-memory side)
// =============================================================================

export function gatherSnapshot(name) {
    return {
        kind: SNAPSHOT_KIND,
        version: SNAPSHOT_VERSION,
        name,
        exportedAt: new Date().toISOString(),
        picks: loadUserPicks(),
        workflows: getAllWorkflowsForExport(),
    };
}

// Validate a parsed object as a snapshot. Returns the snapshot on success,
// throws an Error with a user-readable message on failure. The error
// messages get surfaced verbatim in toasts.
export function validateSnapshot(obj) {
    if (!obj || typeof obj !== "object") {
        throw new Error("Not a valid snapshot file (empty or non-object).");
    }
    if (obj.kind !== SNAPSHOT_KIND) {
        throw new Error(`Not a Kforge Labs snapshot — expected kind="${SNAPSHOT_KIND}".`);
    }
    if (typeof obj.version !== "number" || obj.version < 1) {
        throw new Error(`Snapshot version "${obj.version}" not recognized.`);
    }
    if (obj.version > SNAPSHOT_VERSION) {
        throw new Error(
            `Snapshot version ${obj.version} is newer than this Kforge Labs ` +
            `understands (max v${SNAPSHOT_VERSION}). Update the plugin to load it.`
        );
    }
    if (!Array.isArray(obj.picks)) {
        throw new Error("Snapshot is missing a `picks` array.");
    }
    if (!obj.workflows || typeof obj.workflows !== "object") {
        throw new Error("Snapshot is missing a `workflows` object.");
    }
    return obj;
}

// Apply a validated snapshot to the in-memory + persisted stores. Returns
// `{picksOk, workflowsOk}` so the caller can surface a partial-failure
// toast if one of the two backends rejected the write.
export async function applySnapshot(snapshot) {
    const picksOk = setAllPicks(snapshot.picks);
    if (picksOk) notifyPicksChanged();
    const workflowsOk = await replaceAllWorkflows(snapshot.workflows);
    return { picksOk, workflowsOk };
}

// =============================================================================
// Server I/O — list / read / write / delete presets in /userdata/koolook-presets/
//
// ComfyUI's userdata API:
//   GET    /userdata/<file>                   → file content
//   POST   /userdata/<file>?overwrite=...    → write file
//   DELETE /userdata/<file>                   → delete file (assumed; if
//                                              the deployment doesn't
//                                              support it the response
//                                              just reports an error
//                                              cleanly)
//   GET    /userdata?dir=<dir>&recurse=0&split=1
//                                            → list directory; with split=1,
//                                              entries are
//                                              [path, modified_at, size]
// =============================================================================

function presetFilePath(name) {
    return `${PRESETS_DIR}/${encodeURIComponent(name)}.json`;
}

// Returns an array of `{name, exportedAt, workflowCount, pickCount}` for
// each preset in /userdata/koolook-presets/, sorted alphabetically by
// name (case-insensitive). Returns `[]` on a missing directory or
// listing error — the modal shows "no presets yet" rather than crashing.
export async function listPresets() {
    let resp;
    try {
        resp = await fetch(
            `/userdata?dir=${encodeURIComponent(PRESETS_DIR)}&recurse=0&split=1&full_info=1`
        );
    } catch (e) {
        console.warn("[Koolook] preset listing failed (network):", e);
        return [];
    }
    // 404 = directory doesn't exist yet (no presets saved). Treat as empty.
    if (resp.status === 404) return [];
    if (!resp.ok) {
        console.warn(`[Koolook] preset listing returned HTTP ${resp.status}`);
        return [];
    }
    let data;
    try {
        data = await resp.json();
    } catch (e) {
        console.warn("[Koolook] preset listing JSON parse failed:", e);
        return [];
    }
    if (!Array.isArray(data)) return [];

    // Each entry under `split=1` is `[path, mtimeSeconds, sizeBytes]`.
    // Path is relative to the userdata root (so includes the
    // `koolook-presets/` prefix). Older ComfyUI servers return just the
    // path string — handle both shapes defensively.
    const entries = data
        .map(row => Array.isArray(row) ? row[0] : row)
        .filter(p => typeof p === "string" && p.toLowerCase().endsWith(".json"));

    // Read each preset's metadata in parallel. Failed reads (corrupt file,
    // wrong shape, etc.) are filtered out so the list shows only valid
    // entries — corrupt files surface in the console for debugging.
    const previews = await Promise.all(entries.map(async (relPath) => {
        try {
            const content = await fetch(`/userdata/${relPath}`);
            if (!content.ok) return null;
            const text = await content.text();
            const obj = JSON.parse(text);
            if (obj.kind !== SNAPSHOT_KIND) return null;
            const fileName = relPath.split("/").pop().replace(/\.json$/i, "");
            const decodedName = decodeURIComponent(fileName);
            return {
                // Display name comes from inside the file; the filename is
                // the storage key. They should always match, but a
                // hand-edited file might disagree — trust the file's name
                // for display and the filename for ops.
                displayName: typeof obj.name === "string" && obj.name ? obj.name : decodedName,
                fileName: decodedName,
                exportedAt: typeof obj.exportedAt === "string" ? obj.exportedAt : null,
                workflowCount: countWorkflowsInStore(obj.workflows),
                pickCount: Array.isArray(obj.picks) ? obj.picks.length : 0,
            };
        } catch (e) {
            console.warn(`[Koolook] failed to read preset "${relPath}":`, e);
            return null;
        }
    }));
    return previews
        .filter(p => p !== null)
        .sort((a, b) => a.displayName.localeCompare(b.displayName, undefined, { sensitivity: "base" }));
}

// Read one preset by its filename (no `.json` suffix). Returns the parsed
// + validated snapshot, or throws.
export async function readPreset(fileName) {
    const path = presetFilePath(fileName);
    const resp = await fetch(`/userdata/${path}`);
    if (!resp.ok) throw new Error(`Could not read preset "${fileName}" (HTTP ${resp.status}).`);
    const text = await resp.text();
    let parsed;
    try {
        parsed = JSON.parse(text);
    } catch (e) {
        throw new Error(`Preset file "${fileName}" is not valid JSON.`);
    }
    return validateSnapshot(parsed);
}

// Write a snapshot under `fileName`. Returns true on success, throws on
// failure. The caller is responsible for sanitizing `fileName` upstream.
export async function writePreset(fileName, snapshot) {
    const path = presetFilePath(fileName);
    const resp = await fetch(`/userdata/${path}${USERDATA_OVERWRITE_QUERY}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(snapshot, null, 2),
    });
    if (!resp.ok) throw new Error(`Could not save preset "${fileName}" (HTTP ${resp.status}).`);
    return true;
}

// Returns true if a preset with `fileName` exists. Used for save-time
// "overwrite?" prompts.
export async function presetExists(fileName) {
    try {
        const resp = await fetch(
            `/userdata/${presetFilePath(fileName)}`,
            { method: "HEAD" }
        );
        return resp.ok;
    } catch (e) {
        return false;
    }
}

// Delete a preset. ComfyUI's userdata API supports DELETE; if the
// deployment doesn't, the caller surfaces the error. No silent fallback —
// hidden-but-not-deleted files would mislead the user.
export async function deletePreset(fileName) {
    const path = presetFilePath(fileName);
    const resp = await fetch(`/userdata/${path}`, { method: "DELETE" });
    if (!resp.ok) {
        throw new Error(`Could not delete preset "${fileName}" (HTTP ${resp.status}).`);
    }
    return true;
}

// =============================================================================
// Local-disk download / upload (for cross-server transfer)
// =============================================================================

// Trigger a browser download of the snapshot JSON. Filename is the snapshot's
// own `name` (sanitized) + `.json`.
export function downloadSnapshotAsFile(snapshot) {
    const fileName = sanitizeName(snapshot.name) || "koolook-snapshot";
    const blob = new Blob([JSON.stringify(snapshot, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${fileName}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}

// Read a File (from <input type="file">) as JSON and validate as a snapshot.
// Returns the validated snapshot, or throws.
export async function readSnapshotFile(file) {
    const text = await file.text();
    let parsed;
    try {
        parsed = JSON.parse(text);
    } catch (e) {
        throw new Error(`File "${file.name}" is not valid JSON.`);
    }
    return validateSnapshot(parsed);
}

// =============================================================================
// Internal helpers
// =============================================================================

// Recursive workflow count over a snapshot's `workflows.directories` tree.
// Counts active + archived. Used for the preview metadata in the Load tab.
function countWorkflowsInStore(store) {
    if (!store || typeof store !== "object") return 0;
    let total = 0;
    const walkDirs = (dirs) => {
        if (!dirs || typeof dirs !== "object") return;
        for (const dir of Object.values(dirs)) {
            if (!dir || typeof dir !== "object") continue;
            if (dir.workflows && typeof dir.workflows === "object") {
                total += Object.keys(dir.workflows).length;
            }
            walkDirs(dir.directories);
        }
    };
    walkDirs(store.directories);
    return total;
}
