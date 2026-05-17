// SPDX-FileCopyrightText: 2025-2026 Kforge Labs <https://github.com/malkuthro/ComfyUI-Koolook>
// SPDX-License-Identifier: GPL-3.0-only

// =============================================================================
// Workflows storage (ComfyUI /userdata API with localStorage fallback) +
// transaction layer + in-memory mutation operations.
//
// Schema (recursive, since v0.3 — back-compatible with the v0.2 flat shape):
//   workflowsCache = { directories: { [name]: DirNode } }
//   DirNode = { workflows: { [wfName]: WorkflowEntry }, directories: { [name]: DirNode } }
//
// Every directory can host workflows AND nested subdirectories at any depth.
// The root has only `directories` — workflows live exclusively inside named
// directories. Old flat data without the nested `directories` field still
// loads fine; `normalizeWorkflowsStore` adds an empty `directories: {}` per
// node during migration, so a user's existing /userdata file is preserved.
//
// All mutator/lookup ops take a `path: string[]` of segment names. The empty
// path `[]` is the root. Path segments must be non-empty strings.
//
// `workflowsCache` is module-private. Outside callers go through the exported
// helpers — direct access would otherwise hold a stale reference after the
// cache rebinds during seed/recovery.
// =============================================================================
import {
    WORKFLOWS_USERDATA_PATH,
    WORKFLOWS_FALLBACK_KEY,
    WORKFLOWS_SEEDED_KEY,
    WORKFLOWS_CHANGED_EVENT,
    WORKFLOWS_DEFAULTS_URL,
    MODULE_TAG,
    compareNames,
    toast,
    criticalToast,
} from "./constants.js";

let workflowsCache = { directories: {} };

function notifyWorkflowsChanged() {
    window.dispatchEvent(new CustomEvent(WORKFLOWS_CHANGED_EVENT));
}

// =============================================================================
// Normalization (with back-compat migration for the pre-v0.3 flat shape)
// =============================================================================
function normalizeWorkflowsStore(data) {
    if (!data || typeof data !== "object") return { directories: {} };
    const dirs = data.directories;
    if (!dirs || typeof dirs !== "object") return { directories: {} };
    const stats = { dropped: 0 };
    const out = {};
    for (const [name, dir] of Object.entries(dirs)) {
        const cleaned = normalizeDirNode(dir, stats);
        if (cleaned) out[name] = cleaned;
    }
    if (stats.dropped > 0) {
        console.warn(`[Koolook] dropped ${stats.dropped} malformed workflow entr(y/ies) during normalize`);
    }
    return { directories: out };
}

function normalizeDirNode(node, stats) {
    if (!node || typeof node !== "object") return null;
    const wfs = node.workflows && typeof node.workflows === "object" ? node.workflows : {};
    const cleanedWfs = {};
    for (const [wfName, wf] of Object.entries(wfs)) {
        // Drop entries that can't be loaded (missing/non-object graph).
        // Coerce `archived` to a strict boolean so a stray "false" string
        // can't accidentally flag an entry as archived. Coerce `tags` to a
        // clean string[] (trim, drop empties, dedupe) so old entries without
        // a tags field load with `tags: []` and the rest of the code can
        // assume the field always exists.
        if (!wf || typeof wf !== "object" || !wf.graph || typeof wf.graph !== "object") {
            stats.dropped += 1;
            continue;
        }
        const tags = [];
        if (Array.isArray(wf.tags)) {
            const seen = new Set();
            for (const raw of wf.tags) {
                // Reject non-strings outright instead of coercing — `String(null)`
                // becomes the literal string `"null"`, which would silently
                // surface as a garbage tag in the Tags section. The schema
                // contract is `string[]`; anything else is corruption from a
                // hand-edited /userdata file or a bug elsewhere.
                if (typeof raw !== "string") {
                    stats.dropped += 1;
                    continue;
                }
                const t = raw.trim();
                if (!t || seen.has(t)) continue;
                seen.add(t);
                tags.push(t);
            }
        }
        const module = wf.module === true || tags.includes(MODULE_TAG);
        cleanedWfs[wfName] = { ...wf, archived: wf.archived === true, module, tags };
    }
    // Recurse into subdirectories. Pre-v0.3 nodes don't have a `directories`
    // field — give them an empty one so the rest of the code can assume it
    // always exists.
    const subs = node.directories && typeof node.directories === "object" ? node.directories : {};
    const cleanedSubs = {};
    for (const [subName, subDir] of Object.entries(subs)) {
        const cleaned = normalizeDirNode(subDir, stats);
        if (cleaned) cleanedSubs[subName] = cleaned;
    }
    return { workflows: cleanedWfs, directories: cleanedSubs };
}

function cloneJson(value) {
    return JSON.parse(JSON.stringify(value));
}

function workflowSavedAtMs(wf) {
    if (!wf || typeof wf !== "object" || typeof wf.savedAt !== "string") return 0;
    const ms = Date.parse(wf.savedAt);
    return Number.isFinite(ms) ? ms : 0;
}

function mergeNewerFallbackDir(serverDir, fallbackDir) {
    let changed = false;
    if (!serverDir.workflows || typeof serverDir.workflows !== "object") serverDir.workflows = {};
    if (!serverDir.directories || typeof serverDir.directories !== "object") serverDir.directories = {};

    const fallbackWorkflows =
        fallbackDir && fallbackDir.workflows && typeof fallbackDir.workflows === "object"
            ? fallbackDir.workflows
            : {};
    for (const [wfName, fallbackWorkflow] of Object.entries(fallbackWorkflows)) {
        const serverWorkflow = serverDir.workflows[wfName];
        if (!serverWorkflow || workflowSavedAtMs(fallbackWorkflow) > workflowSavedAtMs(serverWorkflow)) {
            serverDir.workflows[wfName] = cloneJson(fallbackWorkflow);
            changed = true;
        }
    }

    const fallbackDirs =
        fallbackDir && fallbackDir.directories && typeof fallbackDir.directories === "object"
            ? fallbackDir.directories
            : {};
    for (const [dirName, fallbackSubdir] of Object.entries(fallbackDirs)) {
        if (!serverDir.directories[dirName]) {
            serverDir.directories[dirName] = cloneJson(fallbackSubdir);
            changed = true;
        } else if (mergeNewerFallbackDir(serverDir.directories[dirName], fallbackSubdir)) {
            changed = true;
        }
    }
    return changed;
}

function mergeNewerFallbackStore(serverStore, fallbackStore) {
    const merged = cloneJson(serverStore || { directories: {} });
    if (!merged.directories || typeof merged.directories !== "object") merged.directories = {};
    const fallbackDirs =
        fallbackStore && fallbackStore.directories && typeof fallbackStore.directories === "object"
            ? fallbackStore.directories
            : {};
    let changed = false;
    for (const [dirName, fallbackDir] of Object.entries(fallbackDirs)) {
        if (!merged.directories[dirName]) {
            merged.directories[dirName] = cloneJson(fallbackDir);
            changed = true;
        } else if (mergeNewerFallbackDir(merged.directories[dirName], fallbackDir)) {
            changed = true;
        }
    }
    return changed ? merged : null;
}

// =============================================================================
// Server I/O
// =============================================================================

// Sentinel for "server reachable but file content is unparseable".
// Distinct from `undefined` (server unreachable) and `null` (file missing)
// so callers can refuse to auto-seed on top of a corrupt-but-existing file.
const SERVER_FILE_CORRUPT = Symbol("workflows-server-corrupt");

async function fetchWorkflowsFromServer() {
    let resp;
    try {
        resp = await fetch(`/userdata/${WORKFLOWS_USERDATA_PATH}`);
    } catch (e) {
        console.warn("[Koolook] /userdata read failed (network):", e);
        return undefined;
    }
    if (resp.status === 404) return null;
    if (!resp.ok) {
        console.warn(`[Koolook] /userdata read returned HTTP ${resp.status}`);
        return undefined;
    }
    const text = await resp.text();
    if (!text || !text.trim()) return null;
    try {
        return JSON.parse(text);
    } catch (e) {
        console.error("[Koolook] /userdata workflow file is unreadable; refusing to auto-recover:", e);
        return SERVER_FILE_CORRUPT;
    }
}

// ComfyUI's /userdata endpoint requires `?overwrite=true` to allow POST over
// an existing file; without it the server 409s on every save after the
// first. The flag is part of the endpoint contract — pinned here as a named
// constant so the contract is visible at a glance.
const USERDATA_OVERWRITE_QUERY = "?overwrite=true";

// Returns "server" | "fallback" | false. Callers that care about durability
// (specifically the seeders) should only treat "server" as a success — a
// "fallback"-only write is per-browser and won't reach future page loads if
// /userdata becomes reachable again.
async function persistWorkflowsToServer(store) {
    const json = JSON.stringify(store, null, 2);
    try {
        const resp = await fetch(`/userdata/${WORKFLOWS_USERDATA_PATH}${USERDATA_OVERWRITE_QUERY}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: json,
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return "server";
    } catch (e) {
        console.warn("[Koolook] /userdata write failed, using localStorage fallback:", e);
        try {
            localStorage.setItem(WORKFLOWS_FALLBACK_KEY, json);
            return "fallback";
        } catch (e2) {
            console.error("[Koolook] both /userdata and localStorage write failed:", e2);
            return false;
        }
    }
}

// Returns { corrupt: false } on normal load (cache populated), or
// { corrupt: true } when /userdata reachable but file content is unparseable —
// the caller (setup) must skip seeding so we don't clobber the corrupt-but-
// recoverable file with stock defaults.
export async function loadWorkflowsStore() {
    const fromServer = await fetchWorkflowsFromServer();
    if (fromServer === SERVER_FILE_CORRUPT) {
        // Corrupt /userdata file is a recovery situation — the user almost
        // certainly thinks their data is gone. Sticky toast forces explicit
        // acknowledgment and tells them where to look.
        criticalToast(
            "Workflow file on /userdata is unreadable (parse error). The " +
            "file is preserved on disk — refusing to auto-recover so you " +
            "can manually inspect / repair it before any save overwrites " +
            "the bad blob. Check the browser console for the parse error " +
            "and the file at /userdata/" + WORKFLOWS_USERDATA_PATH + "."
        );
        workflowsCache = { directories: {} };
        return { corrupt: true };
    }
    if (fromServer === null) {
        // Server reachable, file just doesn't exist yet — return empty.
        workflowsCache = { directories: {} };
        return { corrupt: false };
    }
    if (fromServer === undefined) {
        // Server unreachable — try localStorage fallback.
        try {
            const raw = localStorage.getItem(WORKFLOWS_FALLBACK_KEY);
            if (raw) {
                workflowsCache = normalizeWorkflowsStore(JSON.parse(raw));
                return { corrupt: false };
            }
        } catch (e) {
            console.warn("[Koolook] failed to parse localStorage workflows fallback:", e);
        }
        workflowsCache = { directories: {} };
        return { corrupt: false };
    }
    workflowsCache = normalizeWorkflowsStore(fromServer);
    // Reconciliation surface: /userdata loaded fine but a localStorage
    // fallback from an earlier outage may still exist. Missing or newer entries are
    // merged below, then persisted back to /userdata when possible.
    // Older/ambiguous fallback data still uses the manual recovery surface
    // described here.
    // Server-only entries are preserved during automatic recovery.
    // Same-name conflicts only auto-recover when the fallback timestamp wins.
    // We don't fire the recovery toast from here either. The toast is
    // wired by the entry point (`koolook_sidebar.js`) where the
    // recovery handlers — Restore-as-snapshot and Discard — have access to
    // `writePreset` / `loadUserPicks` / `showConfirmModal` without creating
    // a circular import. We just return the blob so the caller can decide.
    const fallbackBlob = localStorage.getItem(WORKFLOWS_FALLBACK_KEY);
    if (fallbackBlob) {
        let fallbackStore = null;
        try {
            fallbackStore = normalizeWorkflowsStore(JSON.parse(fallbackBlob));
        } catch (e) {
            console.warn("[Koolook] failed to parse localStorage workflows fallback for reconciliation:", e);
        }
        const mergedStore = fallbackStore ? mergeNewerFallbackStore(workflowsCache, fallbackStore) : null;
        if (mergedStore) {
            workflowsCache = mergedStore;
            const reconcileResult = await persistWorkflowsToServer(workflowsCache);
            if (reconcileResult === "server") {
                localStorage.removeItem(WORKFLOWS_FALLBACK_KEY);
                notifyWorkflowsChanged();
                console.warn(
                    "[Koolook] merged browser-local workflow fallback and wrote it back to /userdata."
                );
                return { corrupt: false, fallbackRecovered: true };
            }
            console.warn(
                `[Koolook] merged browser-local workflow fallback is live, but re-persist landed in ` +
                `${reconcileResult || "neither"}; keeping recovery banner available.`
            );
            return { corrupt: false, fallbackBlob };
        }
        console.warn(
            `[Koolook] /userdata loaded, but a stale localStorage fallback exists ` +
            `at "${WORKFLOWS_FALLBACK_KEY}". If workflows you saved during a previous ` +
            `outage are missing, recover from there before clearing.`
        );
    }
    return { corrupt: false, fallbackBlob: fallbackBlob || null };
}

// Removes the localStorage fallback blob written during an earlier
// `/userdata` outage. Called by the recovery toast's Restore / Discard
// actions once the blob has been either persisted as a snapshot or
// explicitly discarded — clears the underlying condition so the toast
// stops re-firing on subsequent page loads. Best-effort: a quota-disabled
// browser would have failed the write anyway.
export function clearOfflineFallback() {
    try {
        localStorage.removeItem(WORKFLOWS_FALLBACK_KEY);
    } catch (e) {
        console.warn("[Koolook] failed to clear offline fallback:", e);
    }
}

// =============================================================================
// Transaction layer (commit + rollback + persistMutation)
// =============================================================================
// Track whether we've already raised the "fallback-only" sticky toast for
// the current outage. Without this, every mutation during a /userdata outage
// would stack another red banner on screen — quickly unusable. We re-arm
// the warning only after a successful "server" write proves /userdata is
// back, so a SECOND outage in the same session still alerts the user.
let _fallbackWarnedThisOutage = false;

async function commit() {
    // Both "server" and "fallback" are user-visible successes — the panel
    // shows the change either way. Only `false` (both backends rejected)
    // indicates a real loss requiring rollback at the call site.
    const result = await persistWorkflowsToServer(workflowsCache);
    if (result === "fallback" && !_fallbackWarnedThisOutage) {
        _fallbackWarnedThisOutage = true;
        criticalToast(
            "Workflow saved to browser-local fallback only — /userdata server " +
            "unreachable. Data persists per-browser until the server is back. " +
            "DO NOT clear browser data until you've confirmed a server save."
        );
    } else if (result === "server" && _fallbackWarnedThisOutage) {
        // Server is back — re-arm the warning so a future outage will surface.
        _fallbackWarnedThisOutage = false;
    }
    if (result) notifyWorkflowsChanged();
    return result !== false;
}

// Snapshot the cache so callers can roll back a mutation when commit() fails.
// Pairs with a `restore()` function that puts the cache back and re-renders.
function snapshotCache() {
    const snap = JSON.stringify(workflowsCache);
    return () => {
        try {
            workflowsCache = JSON.parse(snap);
            notifyWorkflowsChanged();
        } catch (e) {
            console.error("[Koolook] cache rollback failed:", e);
        }
    };
}

// Mutate-then-commit with automatic rollback on persist failure. Mutation
// returning `false` is treated as a no-op (e.g. name collision) and bypasses
// commit. `onSuccess(result)` and `onNoOp()` callbacks run their own toasts.
export async function persistMutation({ mutate, onSuccess, onNoOp, persistFailedMessage }) {
    const restore = snapshotCache();
    const result = mutate();
    if (result === false) {
        if (onNoOp) onNoOp();
        return false;
    }
    if (await commit()) {
        if (onSuccess) onSuccess(result);
        return true;
    }
    // Capture the unsaved state BEFORE rollback so a recovery copy ends up
    // in the user's clipboard via the critical toast — full backend failure
    // is rare but devastating, and the brief moment between "user clicked
    // save" and "rollback applied" is the only chance to surface what was
    // lost. Pretty-printed for human readability since the user might paste
    // it into a snapshot file or text editor.
    const unsavedJson = JSON.stringify(workflowsCache, null, 2);
    restore();
    criticalToast(
        persistFailedMessage ||
            "Workflow save failed — both /userdata server AND browser " +
            "localStorage rejected the write. Your last change has been " +
            "reverted. Click Copy details to save a recovery JSON to your " +
            "clipboard before retrying.",
        { copyText: unsavedJson }
    );
    return false;
}

// =============================================================================
// Seeding
// =============================================================================
export async function seedWorkflowDefaultsIfNeeded() {
    if (localStorage.getItem(WORKFLOWS_SEEDED_KEY)) return;
    // If existing data is non-empty, respect it and mark seeded.
    const dirNames = Object.keys(workflowsCache.directories || {});
    if (dirNames.length > 0) {
        localStorage.setItem(WORKFLOWS_SEEDED_KEY, "1");
        return;
    }
    try {
        const resp = await fetch(WORKFLOWS_DEFAULTS_URL);
        if (!resp.ok) {
            localStorage.setItem(WORKFLOWS_SEEDED_KEY, "1");
            return;
        }
        const data = await resp.json();
        const normalized = normalizeWorkflowsStore(data);
        const seedDirCount = Object.keys(normalized.directories).length;
        if (seedDirCount > 0) {
            workflowsCache = normalized;
            // Only mark seeded if the persist *reached the server* — a
            // "fallback"-only write is per-browser, and once /userdata
            // becomes reachable on a later load the empty server file
            // would otherwise trump the locally-seeded data.
            const persistResult = await persistWorkflowsToServer(workflowsCache);
            if (persistResult !== "server") {
                console.warn(
                    `[Koolook] seed persist landed in ${persistResult || "neither"}; ` +
                    `not marking seeded so we retry against /userdata next load.`
                );
                return;
            }
        }
        localStorage.setItem(WORKFLOWS_SEEDED_KEY, "1");
        console.log(`[Koolook] seeded ${seedDirCount} default workflow director(y/ies)`);
    } catch (e) {
        console.warn("[Koolook] failed to load workflow_defaults.json:", e);
        localStorage.setItem(WORKFLOWS_SEEDED_KEY, "1");
    }
}

// =============================================================================
// Path-based directory operations (the public mutator API)
//
// `path` is a `string[]` of segment names. `[]` is the root (which has no
// workflows; only directories live there). Mutators that take a `parentPath`
// position the operation against that parent directory; e.g.
// `addDirectory(["UP-scale"], "Type-A")` creates `UP-scale/Type-A`.
//
// Subdirectories cannot be named "Archive" (case-insensitive) — that name is
// reserved by the synthetic Archive folder rendered for archived workflows.
//
// === Mutator invariants — read this before adding a new mutator ===
//
//   1. **Mutate-then-commit, never one without the other.** Every mutator
//      below mutates `workflowsCache` in place and returns immediately. The
//      cache change is invisible on disk until a caller pairs the mutator
//      with `await commit()` (or wraps both in `persistMutation`, which is
//      the strongly-preferred path because it also handles snapshot/rollback
//      and the "no-op" return-false convention). A mutator that "succeeds"
//      without a paired commit silently drops the change on the next
//      reload — a class of bug we've already shipped + reverted once.
//
//   2. **Return `false` for no-op (collisions, missing source, validation
//      failures), truthy for success.** `persistMutation` treats a `false`
//      return as a no-op and skips commit. Truthy returns are passed to
//      `onSuccess(result)` as-is, which is why `saveWorkflowEntry` returns
//      `{archivedAs}` — the success callback wants to surface that detail.
//
//   3. **Mutate in place; don't return new structures.** Callers (and the
//      transaction layer's snapshot) expect `workflowsCache` to be the
//      single source of truth. Returning a new structure would silently
//      detach future reads from the change.
//
//   4. **Never replace `workflowsCache` itself except in seed / load /
//      rollback / snapshot-apply paths.** The four legitimate rebind sites
//      are:
//        - module init (this file's top-level `let workflowsCache = …`)
//        - `loadWorkflowsStore` and `seedWorkflowDefaultsIfNeeded`
//          (re-binding the cache to a freshly-normalized server payload
//          or a default seed)
//        - the closure returned by `snapshotCache()` running rollback
//          inside `persistMutation` — this rebind is what makes commit
//          failure recoverable
//        - `replaceAllWorkflows` (snapshot-apply) — bulk-replaces the
//          entire cache from a deserialized snapshot file, with the same
//          `snapshotCache()` rollback wrapper as `persistMutation` so the
//          atomic "swap on persist success, restore on failure" contract
//          is preserved
//      Outside callers hold no reference to the cache (it's
//      module-private), so external rebinds are impossible. The constraint
//      is internal: a fifth rebind site added by a future contributor for
//      any other reason would break the snapshot/restore semantics that
//      `persistMutation` and `replaceAllWorkflows` depend on.
// =============================================================================

const ARCHIVE_RESERVED_NAME = "archive";

// Resolves to the DirNode at `path`, or `undefined` if any segment is missing.
// `[]` returns a synthetic wrapper around the root (with `directories`
// pointing at `workflowsCache.directories`) so callers don't need a special
// case for root vs. nested.
export function dirOf(path) {
    if (!Array.isArray(path)) return undefined;
    if (path.length === 0) {
        return { workflows: {}, directories: workflowsCache.directories || {} };
    }
    let node = { directories: workflowsCache.directories || {} };
    for (const seg of path) {
        if (typeof seg !== "string" || !seg) return undefined;
        if (!node.directories || !node.directories[seg]) return undefined;
        node = node.directories[seg];
    }
    return node;
}

// Direct child directory names at `parentPath`, sorted A→Z.
export function listDirectoryNames(parentPath = []) {
    const node = dirOf(parentPath);
    if (!node || !node.directories) return [];
    return Object.keys(node.directories).sort(compareNames);
}

// Internal: walk to `parentPath` and create intermediate directories along
// the way. Used only by `saveWorkflowEntry` so a save into a freshly-typed
// new top-level directory always succeeds. Subdirectory creation is explicit
// (right-click → Create subdirectory…) so we don't auto-create nested paths.
function ensureDirectoryAtPath(path) {
    if (!Array.isArray(path) || path.length === 0) return null;
    let node = { directories: workflowsCache.directories };
    for (let i = 0; i < path.length; i += 1) {
        const seg = path[i];
        if (typeof seg !== "string" || !seg) return null;
        if (!node.directories) node.directories = {};
        if (!node.directories[seg]) {
            node.directories[seg] = { workflows: {}, directories: {} };
        }
        node = node.directories[seg];
    }
    return node;
}

// Add a directory at `parentPath` with `name`. Returns false on:
//   - empty trimmed name
//   - reserved name "Archive" (case-insensitive) when nested under another
//     directory (collides with the synthetic Archive folder rendering)
//   - parent path doesn't exist
//   - sibling with the same name already exists
export function addDirectory(parentPath, name) {
    name = (name || "").trim();
    if (!name) return false;
    // Type guard mirroring `moveDirectory` — without this, a future caller
    // passing `undefined` would crash on `parentPath.length`.
    if (!Array.isArray(parentPath)) return false;
    // Reserved-name check: "Archive" at root is fine (no synthetic Archive
    // collides at root because root has no archived workflows of its own),
    // but inside a directory it would shadow the archived-workflows folder.
    if (parentPath.length > 0 && name.toLowerCase() === ARCHIVE_RESERVED_NAME) return false;
    const parent = dirOf(parentPath);
    if (!parent) return false;
    if (!parent.directories) parent.directories = {};
    if (parent.directories[name]) return false;
    parent.directories[name] = { workflows: {}, directories: {} };
    return true;
}

export function renameDirectory(parentPath, oldName, newName) {
    newName = (newName || "").trim();
    if (!newName || newName === oldName) return false;
    if (parentPath.length > 0 && newName.toLowerCase() === ARCHIVE_RESERVED_NAME) return false;
    const parent = dirOf(parentPath);
    if (!parent || !parent.directories || !parent.directories[oldName]) return false;
    if (parent.directories[newName]) return false;
    parent.directories[newName] = parent.directories[oldName];
    delete parent.directories[oldName];
    return true;
}

export function deleteDirectory(parentPath, name) {
    const parent = dirOf(parentPath);
    if (!parent || !parent.directories || !parent.directories[name]) return false;
    delete parent.directories[name];
    return true;
}

// =============================================================================
// Workflow operations (path-addressed)
// =============================================================================
export function saveWorkflowEntry(path, wfName, graphData, options = {}) {
    const dir = ensureDirectoryAtPath(path);
    if (!dir) return false;
    let archivedAs = null;
    const existing = dir.workflows[wfName];
    if (existing) {
        // Same-name save: move the existing version into the directory's
        // Archive (timestamp-suffixed) before overwriting.
        const ts = new Date().toISOString().slice(0, 19).replace("T", " ");
        let archiveName = `${wfName} (archived ${ts})`;
        let n = 1;
        while (dir.workflows[archiveName]) {
            n += 1;
            archiveName = `${wfName} (archived ${ts}) #${n}`;
        }
        dir.workflows[archiveName] = { ...existing, archived: true };
        archivedAs = archiveName;
    }
    dir.workflows[wfName] = {
        savedAt: new Date().toISOString(),
        graph: graphData,
        module: options.module === true,
    };
    return { archivedAs };
}

export function archiveWorkflow(path, wfName) {
    const dir = dirOf(path);
    if (!dir || !dir.workflows[wfName]) return false;
    dir.workflows[wfName].archived = true;
    return true;
}

export function unarchiveWorkflow(path, wfName) {
    const dir = dirOf(path);
    if (!dir || !dir.workflows[wfName]) return false;
    delete dir.workflows[wfName].archived;
    return true;
}

export function renameWorkflow(path, oldWfName, newWfName) {
    newWfName = (newWfName || "").trim();
    if (!newWfName || newWfName === oldWfName) return false;
    const dir = dirOf(path);
    if (!dir || !dir.workflows[oldWfName]) return false;
    if (dir.workflows[newWfName]) return false;
    dir.workflows[newWfName] = dir.workflows[oldWfName];
    delete dir.workflows[oldWfName];
    return true;
}

export function deleteWorkflow(path, wfName) {
    const dir = dirOf(path);
    if (!dir || !dir.workflows[wfName]) return false;
    delete dir.workflows[wfName];
    return true;
}

// `srcPath` and `dstPath` must both reference existing directories. Returns
// false on identical paths, missing source workflow, missing destination,
// or a name collision in the destination.
export function moveWorkflow(srcPath, wfName, dstPath) {
    if (pathsEqual(srcPath, dstPath)) return false;
    const src = dirOf(srcPath);
    if (!src || !src.workflows[wfName]) return false;
    const dst = dirOf(dstPath);
    if (!dst) return false;
    if (dst.workflows[wfName]) return false;
    dst.workflows[wfName] = src.workflows[wfName];
    delete src.workflows[wfName];
    return true;
}

export function getWorkflowGraph(path, wfName) {
    const dir = dirOf(path);
    if (!dir || !dir.workflows[wfName]) return null;
    return dir.workflows[wfName].graph || null;
}

export function isWorkflowModule(path, wfName) {
    const dir = dirOf(path);
    if (!dir || !dir.workflows[wfName]) return false;
    const wf = dir.workflows[wfName];
    const tags = Array.isArray(wf.tags) ? wf.tags : [];
    return wf.module === true || tags.includes(MODULE_TAG);
}

// =============================================================================
// Per-workflow tag operations. Tags are insertion-ordered string[] on each
// workflow entry. Comparison is case-sensitive: "AI" and "ai" are distinct.
// `normalizeDirNode` guarantees `tags` is always an array on cached entries,
// but the mutators still defensively coerce so a freshly-saved workflow that
// hasn't gone through normalize still gets sane behavior.
// =============================================================================
export function getWorkflowTags(path, wfName) {
    const dir = dirOf(path);
    if (!dir || !dir.workflows[wfName]) return null;
    const tags = dir.workflows[wfName].tags;
    return Array.isArray(tags) ? [...tags] : [];
}

export function addTag(path, wfName, tag) {
    tag = (tag || "").trim();
    if (!tag) return false;
    const dir = dirOf(path);
    if (!dir || !dir.workflows[wfName]) return false;
    const wf = dir.workflows[wfName];
    if (!Array.isArray(wf.tags)) wf.tags = [];
    if (wf.tags.includes(tag)) return false;
    wf.tags.push(tag);
    if (tag === MODULE_TAG) wf.module = true;
    return true;
}

export function removeTag(path, wfName, tag) {
    const dir = dirOf(path);
    if (!dir || !dir.workflows[wfName]) return false;
    const wf = dir.workflows[wfName];
    if (!Array.isArray(wf.tags)) return false;
    const idx = wf.tags.indexOf(tag);
    if (idx < 0) return false;
    wf.tags.splice(idx, 1);
    if (tag === MODULE_TAG) wf.module = false;
    return true;
}

// =============================================================================
// Snapshot / preset support — bulk replace + read-only export of the cache.
// Used by the snapshot library flow (#46 item 1).
// =============================================================================

// Replace the entire `workflowsCache` with a freshly-normalized version of
// `rawStore` and persist the result atomically. Returns `true` on
// successful persist (server or localStorage fallback), `false` on full
// persist rejection — in which case the cache is rolled back to its
// pre-call state so the in-memory state stays consistent with disk.
//
// Atomicity matters because a snapshot apply replaces both pieces of
// user state (picks + workflows) and the caller's UI presents the
// "Replace current state?" confirm as a single act. A partially-applied
// load (cache rebound but persist rejected) would diverge in-memory
// from disk; the next reload would revert silently and the user would
// have no signal that the load didn't stick.
//
// Bypasses `persistMutation` because the entire cache is being replaced
// (rather than mutated through a single mutator). Uses `snapshotCache`
// directly for the same atomic semantics — it's the same primitive
// `persistMutation` uses internally.
export async function replaceAllWorkflows(rawStore) {
    const restore = snapshotCache();
    workflowsCache = normalizeWorkflowsStore(rawStore);
    const result = await persistWorkflowsToServer(workflowsCache);
    if (result === false) {
        restore();
        return false;
    }
    notifyWorkflowsChanged();
    return true;
}

// Deep-cloned read-only view of `workflowsCache` for the snapshot export
// flow. Returning a clone (rather than the live ref) keeps callers from
// accidentally mutating cache state through the export pipeline.
export function getAllWorkflowsForExport() {
    return JSON.parse(JSON.stringify(workflowsCache));
}

// Delete every archived workflow under the directory at `path`. Returns
// `{ count }` (number of entries removed) on success, `false` if the
// directory is missing or has no archived entries. Active (non-archived)
// workflows in the same directory are untouched. Used by the Archive
// folder's right-click "Delete archive" action.
export function clearArchive(path) {
    const dir = dirOf(path);
    if (!dir || !dir.workflows) return false;
    const archivedNames = Object.entries(dir.workflows)
        .filter(([, wf]) => wf && wf.archived === true)
        .map(([n]) => n);
    if (archivedNames.length === 0) return false;
    for (const name of archivedNames) delete dir.workflows[name];
    return { count: archivedNames.length };
}

// Move the directory at `srcParentPath/name` to live under `dstParentPath`
// (its name is preserved). Returns false when:
//   - the source doesn't exist
//   - the destination parent doesn't exist
//   - the destination already has a sibling with the same name
//   - the move would create a cycle (dst is the source itself or any
//     descendant of the source — you can't drop a folder into itself)
//   - the source and destination parent are identical (no-op)
//   - the new location at root level would use the reserved name "Archive"
//     in a non-root parent (already enforced by addDirectory's check would
//     not apply here; we re-check explicitly)
export function moveDirectory(srcParentPath, name, dstParentPath) {
    name = (name || "").trim();
    if (!name) return false;
    if (!Array.isArray(srcParentPath) || !Array.isArray(dstParentPath)) return false;
    // Same parent → no-op (identical location).
    if (pathsEqual(srcParentPath, dstParentPath)) return false;
    // Reserved-name check at the new (non-root) parent.
    if (dstParentPath.length > 0 && name.toLowerCase() === ARCHIVE_RESERVED_NAME) return false;
    const src = dirOf(srcParentPath);
    if (!src || !src.directories || !src.directories[name]) return false;
    const dst = dirOf(dstParentPath);
    if (!dst) return false;
    if (!dst.directories) dst.directories = {};
    if (dst.directories[name]) return false; // collision in destination
    // Cycle prevention: dstParentPath must not be the source itself or a
    // descendant. Source path is srcParentPath + [name]; reject any dst
    // path that begins with that prefix.
    const srcFullPath = [...srcParentPath, name];
    if (isPathDescendantOrSame(dstParentPath, srcFullPath)) return false;
    dst.directories[name] = src.directories[name];
    delete src.directories[name];
    return true;
}

// =============================================================================
// Path utilities
// =============================================================================
export function pathsEqual(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i += 1) if (a[i] !== b[i]) return false;
    return true;
}

// True when `testPath` equals or is a descendant of `ancestorPath`.
// e.g. isPathDescendantOrSame(["A","B"], ["A"]) → true (B is under A)
//      isPathDescendantOrSame(["A"],     ["A","B"]) → false (A is above)
//      isPathDescendantOrSame(["A","B"], ["A","B"]) → true (same)
function isPathDescendantOrSame(testPath, ancestorPath) {
    if (!Array.isArray(testPath) || !Array.isArray(ancestorPath)) return false;
    if (testPath.length < ancestorPath.length) return false;
    for (let i = 0; i < ancestorPath.length; i += 1) {
        if (testPath[i] !== ancestorPath[i]) return false;
    }
    return true;
}

// `isArchiveReservedName` was previously exported but never imported — the
// reserved-name check is enforced inline by every mutator that creates or
// renames a directory (`addDirectory`, `renameDirectory`, `moveDirectory`).
// Removed during a dead-export sweep; revive as an export only if a future
// caller needs to gate UI before submit (e.g. live-validate the new-dir
// input in the save modal).
