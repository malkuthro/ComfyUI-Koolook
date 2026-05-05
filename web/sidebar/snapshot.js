// =============================================================================
// Snapshot / preset library (#46 item 1).
//
// A "snapshot" is a single JSON file that captures the full Kforge Labs state
// — curated picks + the entire workflows store — under a user-chosen name.
// Multiple snapshots live side-by-side as files in a configurable filesystem
// directory served by Koolook's `/koolook/presets/*` endpoints (see
// `koolook_routes.py`).
//
// Storage location:
//   - Default: `<comfyui-userdata>/koolook-presets/`
//   - Configurable: set the `KFORGELABS_PRESETS` env var on the ComfyUI
//     server before launch to point at any absolute filesystem path
//     (NFS / SMB mount for facility-shared libraries, Dropbox folder for
//     cross-machine personal sync, etc.).
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
//
// Server-side filename validation: filenames are restricted to a whitelist
// at the route handler. Anything outside that whitelist returns 400 from
// the server.
// =============================================================================
import { loadUserPicks, setAllPicks, notifyPicksChanged } from "./picks_store.js";
import { replaceAllWorkflows, getAllWorkflowsForExport } from "./workflows_store.js";

export const SNAPSHOT_KIND = "koolook-snapshot";
export const SNAPSHOT_VERSION = 1;

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
// Server I/O — calls Koolook's `/koolook/presets/*` routes, defined in
// `koolook_routes.py`. The server reads the `KFORGELABS_PRESETS` env var
// to decide where on disk the library lives (with a default fallback to
// `<comfyui-userdata>/koolook-presets/`), so the same client code works
// for personal-only, NFS/SMB facility-shared, and Dropbox-synced setups.
//
// Routes:
//   GET    /koolook/presets/info         → {path, isDefault, envVar, exists, writable}
//   GET    /koolook/presets/list         → array of {name, mtime, size}
//   GET    /koolook/presets/file?name=…  → file content
//   POST   /koolook/presets/file?name=…  → write file body
//   DELETE /koolook/presets/file?name=…  → delete file
//
// `name` is always the FULL filename including `.json` extension —
// matches the server-side validation regex exactly.
// =============================================================================

const ROUTE_INFO = "/koolook/presets/info";
const ROUTE_LIST = "/koolook/presets/list";
const ROUTE_FILE = "/koolook/presets/file";

function fileQuery(fileName) {
    return `?name=${encodeURIComponent(`${fileName}.json`)}`;
}

// Filename → preview metadata. Returns
// `{displayName, fileName, exportedAt, workflowCount, pickCount}` on
// success, `null` if the file isn't a valid snapshot. Failed reads are
// filtered out by `listPresets` so the list shows only valid entries;
// corrupt files surface in the console for debugging.
async function loadPreview(fullName) {
    try {
        const bareName = fullName.replace(/\.json$/i, "");
        const resp = await fetch(`${ROUTE_FILE}?name=${encodeURIComponent(fullName)}`);
        if (!resp.ok) return null;
        const text = await resp.text();
        const obj = JSON.parse(text);
        if (obj.kind !== SNAPSHOT_KIND) return null;
        return {
            // Display name comes from inside the file; the filename is the
            // storage key. They should match, but a hand-edited file might
            // disagree — trust the file's `name` for display, the filename
            // for operations.
            displayName: typeof obj.name === "string" && obj.name ? obj.name : bareName,
            fileName: bareName,
            exportedAt: typeof obj.exportedAt === "string" ? obj.exportedAt : null,
            workflowCount: countWorkflowsInStore(obj.workflows),
            pickCount: Array.isArray(obj.picks) ? obj.picks.length : 0,
        };
    } catch (e) {
        console.warn(`[Koolook] failed to read preset "${fullName}":`, e);
        return null;
    }
}

// Returns the library's storage path + writability for the Manage tab's
// info icon tooltip. Returns null if the endpoint isn't reachable (e.g.
// the server-side routes weren't registered) — caller falls back to a
// generic message.
export async function getLibraryInfo() {
    try {
        const resp = await fetch(ROUTE_INFO);
        if (!resp.ok) return null;
        return await resp.json();
    } catch (e) {
        console.warn("[Koolook] preset library info unavailable:", e);
        return null;
    }
}

// Returns an array of preview objects, sorted by display name
// (case-insensitive). Empty array on a missing directory or unreachable
// server — modal shows an empty state rather than throwing.
export async function listPresets() {
    let resp;
    try {
        resp = await fetch(ROUTE_LIST);
    } catch (e) {
        console.warn("[Koolook] preset listing failed (network):", e);
        return [];
    }
    if (!resp.ok) {
        console.warn(`[Koolook] preset listing returned HTTP ${resp.status}`);
        return [];
    }
    let entries;
    try {
        entries = await resp.json();
    } catch (e) {
        console.warn("[Koolook] preset listing JSON parse failed:", e);
        return [];
    }
    if (!Array.isArray(entries)) return [];

    const previews = await Promise.all(entries.map((row) => loadPreview(row.name)));
    return previews
        .filter(p => p !== null)
        .sort((a, b) =>
            a.displayName.localeCompare(b.displayName, undefined, { sensitivity: "base" })
        );
}

// Read one preset by its filename (no `.json` suffix). Returns the parsed
// + validated snapshot, throws on read or parse failure.
export async function readPreset(fileName) {
    const resp = await fetch(`${ROUTE_FILE}${fileQuery(fileName)}`);
    if (!resp.ok) {
        throw new Error(`Could not read preset "${fileName}" (HTTP ${resp.status}).`);
    }
    const text = await resp.text();
    let parsed;
    try {
        parsed = JSON.parse(text);
    } catch (e) {
        throw new Error(`Preset file "${fileName}" is not valid JSON.`);
    }
    return validateSnapshot(parsed);
}

// Write a snapshot under `fileName` (no `.json` suffix). Throws on server
// failure with the server-provided reason — surfaces "read-only mount",
// "parent missing", "invalid filename" in the user-facing toast.
export async function writePreset(fileName, snapshot) {
    const resp = await fetch(`${ROUTE_FILE}${fileQuery(fileName)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(snapshot, null, 2),
    });
    if (!resp.ok) {
        const reason = resp.statusText || `HTTP ${resp.status}`;
        throw new Error(`Could not save preset "${fileName}": ${reason}.`);
    }
    return true;
}

// Returns true if a preset with `fileName` exists. Implemented as HEAD
// — server returns 404 cleanly when missing.
export async function presetExists(fileName) {
    try {
        const resp = await fetch(`${ROUTE_FILE}${fileQuery(fileName)}`, { method: "HEAD" });
        return resp.ok;
    } catch (e) {
        return false;
    }
}

// Delete a preset. Throws on server failure (read-only mount, missing
// file, etc.) — no silent fallback.
export async function deletePreset(fileName) {
    const resp = await fetch(`${ROUTE_FILE}${fileQuery(fileName)}`, { method: "DELETE" });
    if (!resp.ok) {
        const reason = resp.statusText || `HTTP ${resp.status}`;
        throw new Error(`Could not delete preset "${fileName}": ${reason}.`);
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
