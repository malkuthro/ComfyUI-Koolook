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
    compareNames,
    toast,
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
        cleanedWfs[wfName] = { ...wf, archived: wf.archived === true, tags };
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

// Returns "server" | "fallback" | false. Callers that care about durability
// (specifically the seeders) should only treat "server" as a success — a
// "fallback"-only write is per-browser and won't reach future page loads if
// /userdata becomes reachable again.
async function persistWorkflowsToServer(store) {
    const json = JSON.stringify(store, null, 2);
    try {
        const resp = await fetch(`/userdata/${WORKFLOWS_USERDATA_PATH}?overwrite=true`, {
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
        toast("Workflow file on /userdata is unreadable. Refusing to auto-recover; check console.");
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
    // Reconciliation: /userdata loaded successfully but a stale fallback blob
    // from an earlier outage still exists. We don't auto-merge (risk of
    // clobbering) but we surface it visibly so the user knows where to recover.
    if (localStorage.getItem(WORKFLOWS_FALLBACK_KEY)) {
        console.warn(
            `[Koolook] /userdata loaded, but a stale localStorage fallback exists ` +
            `at "${WORKFLOWS_FALLBACK_KEY}". If workflows you saved during a previous ` +
            `outage are missing, recover from there before clearing.`
        );
        toast(`Old offline workflow data found in localStorage["${WORKFLOWS_FALLBACK_KEY}"]. See console to recover.`, 4500);
    }
    return { corrupt: false };
}

// =============================================================================
// Transaction layer (commit + rollback + persistMutation)
// =============================================================================
async function commit() {
    // Both "server" and "fallback" are user-visible successes — the panel
    // shows the change either way. Only `false` (both backends rejected)
    // indicates a real loss requiring rollback at the call site.
    const result = await persistWorkflowsToServer(workflowsCache);
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
    restore();
    toast(persistFailedMessage || "Save failed — change reverted. See console.");
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

// All directory paths in DFS, sorted A→Z at each level. Used by the save
// modal to populate the directory dropdown as a flat path picker.
export function listAllDirectoryPaths() {
    const out = [];
    const walk = (parentPath) => {
        for (const name of listDirectoryNames(parentPath)) {
            const next = [...parentPath, name];
            out.push(next);
            walk(next);
        }
    };
    walk([]);
    return out;
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
export function saveWorkflowEntry(path, wfName, graphData) {
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
    return true;
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
