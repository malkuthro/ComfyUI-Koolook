// SPDX-FileCopyrightText: 2025-2026 Kforge Labs <https://github.com/malkuthro/ComfyUI-Koolook>
// SPDX-License-Identifier: GPL-3.0-only

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
    SNAPSHOT_STATUS_CHANGED_EVENT,
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
// Auto-save (defensive snapshots).
//
// All autosaves live in a per-preset SUBFOLDER under the library directory,
// invisible to the user-facing Load list:
//
//   <library>/
//     Wan video kit.json                  ← user-named save (in Load list)
//     Wan video kit_autosave/             ← shadow folder (HIDDEN from Load)
//       periodic.json                     ← single rolling file, overwritten
//       pre_load_<iso>.json × ≤5          ← rotated; permanent per Load
//     another preset.json
//     _unsaved_autosave/                  ← used when no preset is tracked
//       periodic.json
//       pre_load_<iso>.json × ≤5
//
// The server (`koolook_routes.py`) hides subfolders + legacy `_autosave_*`
// flat files from the root-level list endpoint, so the regular Load dialog
// never enumerates these files. Recovery is via direct filesystem access.
//
// Why a subfolder per preset, not flat? The user's library was getting
// cluttered with `_autosave_*` files at root (visible in Total Commander,
// Finder, etc.). Tying autosaves to a per-preset folder makes the on-disk
// structure self-documenting: "the shadow recovery for X lives at X_autosave/".
//
// Why one rolling `periodic.json` (not timestamped)? The latest periodic
// auto-save IS the most useful periodic state. Earlier periodic snapshots
// are worse copies of approximately the same work. Keeping just one trades
// disk diversity for cleanliness — and pre-load files cover the "I want a
// recovery point from before a deliberate destructive action" case.
//
// Why timestamped pre-load files (not rolling)? Each Load is a deliberate
// destructive action and deserves a permanent recovery point. Rotation
// keeps last 5 to bound disk; rare "5 Loads back" recovery loss is the
// trade-off for not letting the folder grow unbounded.
//
// `setCurrentPresetName` is intentionally NOT called by any auto-save —
// these aren't user-initiated saves and shouldn't hijack the "Save over X"
// default the way a manual Save does.
// =============================================================================

const AUTOSAVE_PRE_LOAD_PREFIX = "pre_load_";
const AUTOSAVE_PERIODIC_FILENAME = "periodic";
const AUTOSAVE_UNSAVED_SUBDIR = "_unsaved_autosave";
const AUTOSAVE_SUBDIR_SUFFIX = "_autosave";
const AUTOSAVE_PRE_LOAD_KEEP = 5;

function isoToFsToken(iso) {
    // 2026-05-06T11:45:33.123Z → 2026-05-06T11-45-33-123Z
    return iso.replace(/[:.]/g, "-");
}

// Resolves the subfolder name for the current session's autosaves.
// Tied to `getCurrentPresetName()` — when the user loads or saves "Foo"
// the autosaves redirect to `Foo_autosave/`. When no preset is tracked
// (fresh session, or user explicitly cleared the tracker) autosaves go
// to `_unsaved_autosave/`. Sanitizes the preset name through the same
// filename whitelist so the dirname is server-validation-safe.
function _autosaveSubdir() {
    const name = getCurrentPresetName();
    if (!name) return AUTOSAVE_UNSAVED_SUBDIR;
    const sanitized = sanitizeName(name);
    if (!sanitized) return AUTOSAVE_UNSAVED_SUBDIR;
    return sanitized + AUTOSAVE_SUBDIR_SUFFIX;
}

// =============================================================================
// Snapshot status — tracks whether the live in-memory state matches the last
// named save, the latest periodic auto-save, both, or neither. Surfaces in
// the sidebar via `getSnapshotStatus()`.
//
// Two reference fingerprints:
//   • `_lastNamedSaveFingerprint` — set by markStateSaved() on Save / Load
//     success. Persisted to localStorage so the baseline survives reloads
//     (otherwise every fresh page-load would show "saved" briefly even
//     after the user mutated the state in a previous session).
//   • `_lastPeriodicFingerprint` — set when a periodic auto-save lands.
//     Lives in memory only; periodic auto-saves are session-scoped recovery
//     points, not user-visible save history.
//
// `_computeFingerprint()` strips fields that change every call (`exportedAt`,
// the wrapping `name`/`kind`/`version` envelope) so the comparison is
// "did the actual data change" rather than "is this a different export."
// =============================================================================

const SAVED_FINGERPRINT_KEY = "koolook.snapshot.savedFingerprint.v1";
const SAVED_AT_KEY = "koolook.snapshot.savedAt.v1";

let _lastNamedSaveFingerprint = null;
let _lastNamedSaveAt = null;
let _lastPeriodicFingerprint = null;
let _lastPeriodicAt = null;

(function _initSavedFingerprintFromStorage() {
    try {
        const v = localStorage.getItem(SAVED_FINGERPRINT_KEY);
        if (typeof v === "string" && v) _lastNamedSaveFingerprint = v;
        const at = localStorage.getItem(SAVED_AT_KEY);
        if (typeof at === "string" && at) _lastNamedSaveAt = at;
    } catch (e) {
        // localStorage unreadable (private mode + permission edge cases).
        // Status will be slightly less accurate this session — fine.
    }
})();

function _persistSavedFingerprint() {
    try {
        if (_lastNamedSaveFingerprint) {
            localStorage.setItem(SAVED_FINGERPRINT_KEY, _lastNamedSaveFingerprint);
        } else {
            localStorage.removeItem(SAVED_FINGERPRINT_KEY);
        }
        if (_lastNamedSaveAt) {
            localStorage.setItem(SAVED_AT_KEY, _lastNamedSaveAt);
        } else {
            localStorage.removeItem(SAVED_AT_KEY);
        }
    } catch (e) {
        // Quota / private mode — the in-memory fingerprint still works for
        // this session; only cross-session accuracy degrades.
    }
}

function _computeFingerprint() {
    return JSON.stringify({
        picks: loadUserPicks(),
        workflows: getAllWorkflowsForExport(),
    });
}

function _emitStatusChanged() {
    try {
        window.dispatchEvent(new CustomEvent(SNAPSHOT_STATUS_CHANGED_EVENT));
    } catch (e) {
        // Some test environments don't have window — defensive only.
    }
}

// Mark the current in-memory state as "saved" — call after any successful
// named save (writePreset for a user-named preset) or successful Load
// (applySnapshot baseline). Auto-saves do NOT call this; they're not
// user-initiated saves and shouldn't reset the dirty indicator.
export function markStateSaved() {
    _lastNamedSaveFingerprint = _computeFingerprint();
    _lastNamedSaveAt = new Date().toISOString();
    _persistSavedFingerprint();
    _emitStatusChanged();
}

// Returns one of:
//   { name: <string|null>, state: "saved" | "autosaved" | "unsaved" | "none",
//     lastNamedSaveAt: <ISO|null>, lastAutosaveAt: <ISO|null> }
//
// Status precedence (highest first):
//   • "saved"     — current fingerprint matches the last named save
//   • "autosaved" — current matches the latest periodic auto-save (only)
//   • "unsaved"   — there IS a tracked preset but state diverged
//   • "none"      — no preset tracked, no autosave match
export function getSnapshotStatus() {
    const name = getCurrentPresetName();
    const fp = _computeFingerprint();
    if (_lastNamedSaveFingerprint && fp === _lastNamedSaveFingerprint) {
        return { name, state: "saved", lastNamedSaveAt: _lastNamedSaveAt, lastAutosaveAt: _lastPeriodicAt };
    }
    if (_lastPeriodicFingerprint && fp === _lastPeriodicFingerprint) {
        return { name, state: "autosaved", lastNamedSaveAt: _lastNamedSaveAt, lastAutosaveAt: _lastPeriodicAt };
    }
    if (name) {
        return { name, state: "unsaved", lastNamedSaveAt: _lastNamedSaveAt, lastAutosaveAt: _lastPeriodicAt };
    }
    return { name: null, state: "none", lastNamedSaveAt: _lastNamedSaveAt, lastAutosaveAt: _lastPeriodicAt };
}

// Write a pre-load auto-save into the current session's autosave subfolder
// and return a `<subdir>/<filename>` token on success. Throws on persist
// failure — callers should treat that as a hard abort signal: the
// destructive Load that motivated this auto-save MUST NOT proceed without
// a recovery point, otherwise the entire premise of the feature collapses.
//
// IMPORTANT: the subfolder is captured at write time, NOT load time. Pre-
// load autosaves preserve the OLD state (which is named after the OLD
// preset). If the user has "Foo" loaded and Loads "Bar", the pre-load
// recovery file lands in `Foo_autosave/` — that's where the user looks
// when they realize they wanted to undo back to the Foo state.
export async function writePreLoadAutosave(label) {
    const isoNow = new Date().toISOString();
    const subdir = _autosaveSubdir();
    const fileName = AUTOSAVE_PRE_LOAD_PREFIX + isoToFsToken(isoNow);
    const displayName = label
        ? `Pre-load auto-save · ${subdir} · ${label} · ${isoNow}`
        : `Pre-load auto-save · ${subdir} · ${isoNow}`;
    const snap = gatherSnapshot(displayName);
    await writePreset(fileName, snap, { dir: subdir });
    // Prune older pre-load files in the same subdir to bound disk.
    await _prunePreLoadAutosaves(subdir, AUTOSAVE_PRE_LOAD_KEEP);
    return `${subdir}/${fileName}`;
}

async function _prunePreLoadAutosaves(subdir, keep) {
    try {
        const all = await listPresets({ dir: subdir });
        // Lex sort of ISO-ish filenames is chronological order. We strip
        // to pre-load files only (keeping `periodic.json` untouched),
        // sort newest-first, delete everything past `keep`.
        const preLoads = all
            .filter((p) => typeof p.fileName === "string" &&
                          p.fileName.startsWith(AUTOSAVE_PRE_LOAD_PREFIX))
            .sort((a, b) => b.fileName.localeCompare(a.fileName));
        const toDelete = preLoads.slice(keep);
        for (const p of toDelete) {
            try {
                await deletePreset(p.fileName, { dir: subdir });
            } catch (e) {
                console.warn(`[Koolook] failed to prune ${subdir}/${p.fileName}:`, e);
            }
        }
    } catch (e) {
        console.warn(`[Koolook] pre-load prune failed for ${subdir}:`, e);
    }
}

// =============================================================================
// Periodic auto-save — module-private timer + change-detection state.
// Started by `startPeriodicAutosave` (called from the entry `setup()`).
// Shares `_lastPeriodicFingerprint` with the snapshot-status block above —
// when a tick lands successfully, that's also the new "matches latest
// auto-save" reference for the status indicator. `null` initial value
// guarantees the first tick saves unconditionally, capturing a baseline
// recovery point ~30s after sidebar open.
// =============================================================================
let _periodicTimerId = null;
let _periodicFirstTickId = null;

async function _periodicTick() {
    // Skip while the tab is in the background — saving while the user can't
    // see toasts / errors is fine, but saving WORK they're not actively
    // doing isn't valuable, and quietly hammering /userdata in a backgrounded
    // tab wastes server cycles.
    if (typeof document !== "undefined" && document.hidden) return;
    try {
        const isoNow = new Date().toISOString();
        const subdir = _autosaveSubdir();
        const displayName = `Periodic auto-save · ${subdir} · ${isoNow}`;
        const snap = gatherSnapshot(displayName);
        // Change detection — uses the same picks-and-workflows fingerprint
        // as the status indicator so post-tick the "auto-saved" status is
        // accurate without recomputing.
        const fingerprint = _computeFingerprint();
        if (fingerprint === _lastPeriodicFingerprint) return;
        // Single rolling file inside the subdir — overwriting the previous
        // periodic snapshot. No rotation / no list-and-delete dance; the
        // latest periodic IS the most useful, and pre-load files cover the
        // "deliberate destructive-action recovery" case independently.
        await writePreset(AUTOSAVE_PERIODIC_FILENAME, snap, { dir: subdir });
        _lastPeriodicFingerprint = fingerprint;
        _lastPeriodicAt = isoNow;
        // Tell the sidebar status indicator to refresh — state didn't change
        // (mutation events would have covered that) but the *match* against
        // the latest auto-save did, so "unsaved" can flip to "auto-saved".
        _emitStatusChanged();
        console.log(`[Koolook] periodic auto-save: ${subdir}/${AUTOSAVE_PERIODIC_FILENAME}.json`);
    } catch (e) {
        // Library unreachable, read-only mount, etc. — log and try again
        // next interval. Periodic auto-save failures should NOT surface
        // as toasts; they're a background defensive layer, not a primary
        // action the user is waiting on.
        console.warn("[Koolook] periodic auto-save failed:", e);
    }
}

// Start the periodic auto-save loop. Idempotent — calling twice is a no-op
// (the second call returns without registering another timer). Default
// interval 5 minutes. The first tick fires after a short grace period
// (30s) rather than immediately so the load / seed flows have time to
// settle before we capture the first baseline.
export function startPeriodicAutosave({ intervalMs = 5 * 60 * 1000, firstTickDelayMs = 30 * 1000 } = {}) {
    if (_periodicTimerId !== null) return;
    const tick = () => _periodicTick();
    _periodicFirstTickId = setTimeout(tick, firstTickDelayMs);
    _periodicTimerId = setInterval(tick, intervalMs);
    console.log(
        `[Koolook] periodic auto-save started: every ${Math.round(intervalMs / 1000)}s, ` +
        `first tick in ${Math.round(firstTickDelayMs / 1000)}s`
    );
}

// Stop the periodic auto-save loop. Idempotent. Useful for tests and for
// hot-reload paths during development; in normal use the timer outlives
// the page anyway.
export function stopPeriodicAutosave() {
    if (_periodicFirstTickId !== null) {
        clearTimeout(_periodicFirstTickId);
        _periodicFirstTickId = null;
    }
    if (_periodicTimerId !== null) {
        clearInterval(_periodicTimerId);
        _periodicTimerId = null;
    }
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
    // Status display shows the tracked name — refresh on every change.
    _emitStatusChanged();
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
const ROUTE_BROWSE = "/koolook/presets/browse";
const ROUTE_BROWSE_NEW_FOLDER = "/koolook/presets/browse/new-folder";
const ROUTE_AUTOSAVES_LIST = "/koolook/presets/autosaves/list";
const ROUTE_REVEAL = "/koolook/presets/reveal";

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

function fileQuery(fileName, dir) {
    let q = `?name=${encodeURIComponent(`${fileName}.json`)}`;
    if (dir) q += `&dir=${encodeURIComponent(dir)}`;
    return q;
}

// Filename → preview metadata. Returns
// `{displayName, fileName, exportedAt, workflowCount, pickCount}` on
// success, `null` if the file isn't a valid snapshot. Failed reads are
// filtered out by `listPresets` so the list shows only valid entries;
// corrupt files surface in the console for debugging.
async function loadPreview(fullName, dir) {
    try {
        const bareName = fullName.replace(/\.json$/i, "");
        const dirParam = dir ? `&dir=${encodeURIComponent(dir)}` : "";
        const resp = await fetch(`${ROUTE_FILE}?name=${encodeURIComponent(fullName)}${dirParam}`);
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

// Directory browser for the Settings dialog. The server returns directory
// entries only; the selected path is still persisted via saveSettings().
export async function browseDirectories(path) {
    const q = path ? `?path=${encodeURIComponent(path)}` : "";
    const resp = await fetch(`${ROUTE_BROWSE}${q}`);
    if (!resp.ok) {
        throw new Error(`Could not browse directories: ${await readErrorReason(resp)}.`);
    }
    return await resp.json();
}

export async function createBrowseDirectory(parentPath, name) {
    const resp = await fetch(ROUTE_BROWSE_NEW_FOLDER, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ parentPath, name }),
    });
    if (!resp.ok) {
        throw new Error(`Could not create folder: ${await readErrorReason(resp)}.`);
    }
    return await resp.json();
}

// Returns an array of preview objects, sorted by display name
// (case-insensitive). Empty array on a missing directory or unreachable
// server — modal shows an empty state rather than throwing.
//
// Optional `dir` scopes the listing to a subfolder (used for autosave
// pruning). The user-facing Load list calls this without `dir` and gets
// only root-level user-named presets — autosave subfolders are NEVER
// enumerated as part of the regular Load flow.
export async function listPresets({ dir } = {}) {
    let resp;
    const url = dir ? `${ROUTE_LIST}?dir=${encodeURIComponent(dir)}` : ROUTE_LIST;
    try {
        resp = await fetch(url);
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

    const previews = await Promise.all(entries.map((row) => loadPreview(row.name, dir)));
    return previews
        .filter(p => p !== null)
        .sort((a, b) =>
            a.displayName.localeCompare(b.displayName, undefined, { sensitivity: "base" })
        );
}

// Read one preset by its filename (no `.json` suffix). Optional `dir`
// scopes the read to an autosave subfolder. Returns the parsed +
// validated snapshot, throws on read or parse failure.
export async function readPreset(fileName, { dir } = {}) {
    const resp = await fetch(`${ROUTE_FILE}${fileQuery(fileName, dir)}`);
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

// Write a snapshot under `fileName` (no `.json` suffix). Optional `dir`
// scopes the write to an autosave subfolder; the server auto-creates the
// subfolder on POST. Throws on server failure with the server-provided
// reason — surfaces "read-only mount", "parent missing", "invalid
// filename" in the user-facing toast.
export async function writePreset(fileName, snapshot, { dir } = {}) {
    const resp = await fetch(`${ROUTE_FILE}${fileQuery(fileName, dir)}`, {
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
export async function presetExists(fileName, { dir } = {}) {
    try {
        const resp = await fetch(`${ROUTE_FILE}${fileQuery(fileName, dir)}`, { method: "HEAD" });
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
export async function deletePreset(fileName, { dir } = {}) {
    const resp = await fetch(`${ROUTE_FILE}${fileQuery(fileName, dir)}`, { method: "DELETE" });
    if (!resp.ok) {
        throw new Error(`Could not delete preset "${fileName}": ${await readErrorReason(resp)}.`);
    }
    return true;
}

// List every recovery autosave across all `*_autosave/` subfolders. Returns
// preview objects enriched with `dir`, `mtime`, and a `kind` discriminator
// ("pre_load" | "periodic" | "other"). Empty array on a missing library or
// unreachable server. Used by the Load dialog's collapsible "Recovery
// auto-saves" section — without this endpoint the user's pre-load /
// periodic auto-saves are written but unreachable from the UI.
//
// N+1 cost is real here: one HEAD-or-similar metadata call to list, then
// one GET per file to validate + extract displayName. We accept it because
// the recovery section is collapsed by default — the cost is paid only
// when the user expands it. For very large libraries (many presets ×
// many autosaves) this can be slow; we'd add server-side preview-flatten
// before that becomes a real complaint.
export async function listAutosaves() {
    let resp;
    try {
        resp = await fetch(ROUTE_AUTOSAVES_LIST);
    } catch (e) {
        console.warn("[Koolook] autosave listing failed (network):", e);
        return [];
    }
    if (!resp.ok) {
        console.warn(`[Koolook] autosave listing returned HTTP ${resp.status}`);
        return [];
    }
    let entries;
    try {
        entries = await resp.json();
    } catch (e) {
        console.warn("[Koolook] autosave listing JSON parse failed:", e);
        return [];
    }
    if (!Array.isArray(entries)) return [];

    const previews = await Promise.all(entries.map(async (row) => {
        try {
            const bareName = row.name.replace(/\.json$/i, "");
            const dirParam = `&dir=${encodeURIComponent(row.dir)}`;
            const r = await fetch(
                `${ROUTE_FILE}?name=${encodeURIComponent(row.name)}${dirParam}`
            );
            if (!r.ok) return null;
            const obj = JSON.parse(await r.text());
            if (obj.kind !== SNAPSHOT_KIND) return null;
            // Classify by filename: `pre_load_*.json` are timestamped
            // recovery points (one per Load), `periodic.json` is the
            // single rolling auto-save. Anything else (legacy / hand-
            // dropped) is "other".
            let kind = "other";
            if (bareName === "periodic") kind = "periodic";
            else if (bareName.startsWith("pre_load_")) kind = "pre_load";
            return {
                dir: row.dir,
                fileName: bareName,
                kind,
                displayName: typeof obj.name === "string" && obj.name ? obj.name : bareName,
                exportedAt: typeof obj.exportedAt === "string" ? obj.exportedAt : null,
                mtime: row.mtime,
                workflowCount: countWorkflowsInStore(obj.workflows),
                pickCount: Array.isArray(obj.picks) ? obj.picks.length : 0,
            };
        } catch (e) {
            console.warn(`[Koolook] failed to read autosave ${row.dir}/${row.name}:`, e);
            return null;
        }
    }));
    return previews.filter((p) => p !== null);
}

// Open the preset library (or an autosave subfolder) in the OS file
// manager. Returns the resolved server-side path on success — the caller
// can surface it in a toast so the user has the literal path even when
// the OS open call silently fails (rare, but the path text is enough to
// paste into Finder / Explorer manually).
export async function revealPresetFolder({ dir = "" } = {}) {
    const dirParam = dir ? `?dir=${encodeURIComponent(dir)}` : "";
    const r = await fetch(`${ROUTE_REVEAL}${dirParam}`, { method: "POST" });
    if (!r.ok) throw new Error(await readErrorReason(r));
    return r.json();
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
