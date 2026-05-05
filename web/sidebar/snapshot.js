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
import {
    STARTER_SEEDED_KEY,
    STARTER_URL,
    STARTER_PRESET_FILENAME,
    toast,
} from "./constants.js";

export const SNAPSHOT_KIND = "koolook-snapshot";
export const SNAPSHOT_VERSION = 1;

// =============================================================================
// Filename hygiene
// =============================================================================

// User-typed names need to survive filesystem + URL roundtripping AND
// match the server's filename whitelist (see `_FILENAME_RE` in
// `koolook_routes.py`: `^[A-Za-z0-9 _.()\-]+\.json$`). The server
// validates at the route boundary; we mirror the same rules here so a
// reasonable-looking name doesn't sail through the form only to fail
// with an opaque HTTP 400 toast at save time.
//
// Strategy: replace anything outside the server whitelist with `_`,
// collapse whitespace runs, trim. Empty after normalization → null
// (caller should reject before submit).
export function sanitizeName(raw) {
    if (typeof raw !== "string") return null;
    const cleaned = raw
        // Replace any character outside the server's whitelist with `_`
        // — `+`, `&`, `'`, `,`, `;`, `[]`, accented letters, emoji, etc.
        // all collapse to underscores rather than triggering a 400.
        .replace(/[^A-Za-z0-9 _.()\-]+/g, "_")
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
// Starter preset distribution — replaces the legacy curated_defaults.json
// pick-only seed with a full snapshot file shipped at web/starter_preset.json.
// On a fresh install the seeder copies it into the user's snapshot library
// (via the same /koolook/presets/file endpoint the Save dialog uses), so it
// shows up as one entry called "starter" in the Load dialog. The user gets
// one click to populate picks + workflows + tags + archive — and full agency
// to delete or edit it later, since it's just a regular preset on disk.
// =============================================================================

// Seed the bundled starter preset into the user's snapshot library iff:
//   • we haven't already attempted on this browser (`STARTER_SEEDED_KEY`),
//   • the user is fresh (no existing picks — existing users keep their state),
//   • the library is reachable AND has no preset called "starter" yet.
//
// If the seed succeeds, toast the user so they know the Load dialog now has
// something to apply. On any reachability failure (server down, fetch error)
// we deliberately DON'T set the seeded flag — the next page load retries.
export async function seedStarterPresetIfNeeded() {
    if (localStorage.getItem(STARTER_SEEDED_KEY)) return;

    // Existing user with non-empty picks → already seeded by the legacy
    // curated_defaults.json flow. Mark and skip; their state is theirs.
    if (loadUserPicks().length > 0) {
        localStorage.setItem(STARTER_SEEDED_KEY, "1");
        return;
    }

    // Existence probe — `presetExists` returns true/false/null where null
    // means the library is unreachable. We bail without setting the flag on
    // null so a server hiccup doesn't permanently skip the seed.
    let exists;
    try {
        exists = await presetExists(STARTER_PRESET_FILENAME);
    } catch (e) {
        console.warn("[Koolook] starter seed: presetExists threw:", e);
        return;
    }
    if (exists === null) {
        console.warn("[Koolook] starter seed: library unreachable, retrying next load");
        return;
    }
    if (exists === true) {
        // Already there — facility-shared library, or a re-install on a
        // machine that previously had it. Mark and move on.
        localStorage.setItem(STARTER_SEEDED_KEY, "1");
        return;
    }

    // Fetch the bundled preset from the package's web/ folder. A 4xx/5xx or
    // a parse error means the package is broken; mark seeded so we don't
    // retry forever, and log so the maintainer notices.
    let preset;
    try {
        const resp = await fetch(STARTER_URL);
        if (!resp.ok) {
            console.warn(`[Koolook] starter_preset.json HTTP ${resp.status}; skipping seed`);
            localStorage.setItem(STARTER_SEEDED_KEY, "1");
            return;
        }
        preset = await resp.json();
    } catch (e) {
        console.warn("[Koolook] failed to fetch starter_preset.json:", e);
        localStorage.setItem(STARTER_SEEDED_KEY, "1");
        return;
    }

    // Validate before writing — we don't want to seed a malformed file into
    // every fresh install. validateSnapshot throws with a user-readable
    // message; we just log it (no toast — fresh users shouldn't see scary
    // errors about a feature they haven't discovered yet).
    try {
        validateSnapshot(preset);
    } catch (e) {
        console.warn("[Koolook] starter_preset.json failed validation:", e.message);
        localStorage.setItem(STARTER_SEEDED_KEY, "1");
        return;
    }

    try {
        await writePreset(STARTER_PRESET_FILENAME, preset);
    } catch (e) {
        console.warn("[Koolook] starter preset write failed (retrying next load):", e.message);
        // Don't set the flag — a read-only mount or network blip should
        // self-heal on the user's next launch.
        return;
    }

    localStorage.setItem(STARTER_SEEDED_KEY, "1");
    console.log(`[Koolook] seeded starter preset into snapshot library as "${STARTER_PRESET_FILENAME}.json"`);
    toast(`Starter preset "${STARTER_PRESET_FILENAME}" added — open Snapshot → Load to populate picks & workflows.`);
}

// Maintainer flow: capture the current full state as the bundled starter
// preset for the next release. Copies a JSON-formatted snapshot to clipboard
// (with download fallback) so the maintainer can paste it over
// `web/starter_preset.json` in the repo.
//
// Distinct from the regular Snapshot Save flow — that writes to the user's
// snapshot library directory (per-user state). This writes nothing to disk;
// it just hands the maintainer the JSON for a repo-side commit.
export async function exportStarterPreset() {
    const snapshot = gatherSnapshot(STARTER_PRESET_FILENAME);
    const json = JSON.stringify(snapshot, null, 2) + "\n";
    const pickN = snapshot.picks.length;
    const wfN = countWorkflowsInStore(snapshot.workflows);
    const summary = `${pickN} pick${pickN === 1 ? "" : "s"} · ${wfN} workflow${wfN === 1 ? "" : "s"}`;
    try {
        await navigator.clipboard.writeText(json);
        toast(`Copied starter preset to clipboard (${summary}). Paste into web/starter_preset.json.`);
        return;
    } catch (e) {
        console.warn("[Koolook] clipboard write failed, falling back to download:", e);
    }
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "starter_preset.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    toast(`Downloaded starter_preset.json (${summary}). Replace the file in web/ with it.`);
}

// =============================================================================
// Current-preset tracking — remembers which preset the user last loaded or
// saved, so a Save button can default to "save over the current one" instead
// of forcing the user to retype the name every session. Persists in
// localStorage so it survives reloads.
// =============================================================================

const CURRENT_PRESET_KEY = "koolook.snapshot.currentPresetName.v1";

export function getCurrentPresetName() {
    try {
        const v = localStorage.getItem(CURRENT_PRESET_KEY);
        return typeof v === "string" && v ? v : null;
    } catch (e) {
        return null;
    }
}

export function setCurrentPresetName(name) {
    try {
        if (name) localStorage.setItem(CURRENT_PRESET_KEY, name);
        else localStorage.removeItem(CURRENT_PRESET_KEY);
    } catch (e) {
        // localStorage rejected (quota, private mode) — non-fatal,
        // the next save just won't pre-fill. Log so power users notice.
        console.warn("[Koolook] could not persist current preset name:", e);
    }
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
const ROUTE_SETTINGS = "/koolook/presets/settings";

// Read a non-OK response's reason for a user-facing toast. Prefers the
// response body (always available, carries aiohttp's `reason=` text),
// then `resp.statusText` (HTTP/1.1 only — HTTP/2 strips reason phrases
// per RFC 7540, so behind any HTTP/2-terminating proxy this is empty),
// then `HTTP <status>` as a last resort. Async because we have to await
// the body read.
async function readErrorReason(resp) {
    try {
        const body = (await resp.text()).trim();
        if (body) return body;
    } catch (e) { /* fall through */ }
    return resp.statusText || `HTTP ${resp.status}`;
}

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
        if (obj.kind !== SNAPSHOT_KIND) {
            // Non-snapshot JSON in the library directory (someone dropped a
            // workflow export, a settings file, etc.). Log so a "library
            // appears empty" symptom is debuggable from the console.
            console.warn(
                `[Koolook] preset "${fullName}" skipped — kind="${obj.kind}", ` +
                `expected "${SNAPSHOT_KIND}".`
            );
            return null;
        }
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

// Returns the library's resolved storage path + writability + source
// (settings / env / default). Returns null if the endpoint isn't
// reachable — caller falls back to a generic message.
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

// Read the saved-in-UI library path (settings file's `libraryPath` field)
// alongside the currently-resolved path and source. The Settings dialog
// uses `savedLibraryPath` as the editable field, `resolvedPath` as the
// "currently in effect" readout, and `source` to label the chain.
export async function getSettings() {
    try {
        const resp = await fetch(ROUTE_SETTINGS);
        if (!resp.ok) return null;
        return await resp.json();
    } catch (e) {
        console.warn("[Koolook] preset settings unavailable:", e);
        return null;
    }
}

// Save (or clear, if `libraryPath` is empty) the in-UI library-path
// override. Returns the server's echoed state ({savedLibraryPath,
// resolvedPath, source}) on success. Throws on server failure with the
// server-provided reason.
export async function saveSettings(libraryPath) {
    const resp = await fetch(ROUTE_SETTINGS, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ libraryPath: libraryPath || "" }),
    });
    if (!resp.ok) {
        throw new Error(`Could not save settings: ${await readErrorReason(resp)}.`);
    }
    return await resp.json();
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
        throw new Error(`Could not read preset "${fileName}": ${await readErrorReason(resp)}.`);
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
        throw new Error(`Could not save preset "${fileName}": ${await readErrorReason(resp)}.`);
    }
    return true;
}

// Returns ``true`` / ``false`` for present/absent, ``null`` when the
// existence check itself couldn't run (network down, server error).
// Caller distinguishes:
//   - `false` → safe to write without the overwrite-confirm prompt
//   - `null`  → don't trust either branch; surface a "library
//                unreachable" warning rather than silently writing
//   - `true`  → ask for overwrite confirmation before writing
//
// Server has a dedicated HEAD handler that does just `is_file()` + status,
// so this is a cheap probe even on multi-MB snapshots over Dropbox/NFS.
export async function presetExists(fileName) {
    try {
        const resp = await fetch(`${ROUTE_FILE}${fileQuery(fileName)}`, { method: "HEAD" });
        if (resp.ok) return true;
        if (resp.status === 404) return false;
        // 5xx, 4xx other than 404, opaque proxy errors → unreachable.
        console.warn(`[Koolook] presetExists: server returned HTTP ${resp.status}`);
        return null;
    } catch (e) {
        console.warn("[Koolook] presetExists: network failure:", e);
        return null;
    }
}

// Delete a preset. Throws on server failure (read-only mount, missing
// file, etc.) — no silent fallback.
export async function deletePreset(fileName) {
    const resp = await fetch(`${ROUTE_FILE}${fileQuery(fileName)}`, { method: "DELETE" });
    if (!resp.ok) {
        throw new Error(`Could not delete preset "${fileName}": ${await readErrorReason(resp)}.`);
    }
    return true;
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
