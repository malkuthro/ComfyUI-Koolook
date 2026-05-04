// =============================================================================
// Workflows storage (ComfyUI /userdata API with localStorage fallback) +
// transaction layer + in-memory mutation operations.
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

function normalizeWorkflowsStore(data) {
    if (!data || typeof data !== "object") return { directories: {} };
    const dirs = data.directories;
    if (!dirs || typeof dirs !== "object") return { directories: {} };
    const out = {};
    let dropped = 0;
    for (const [name, dir] of Object.entries(dirs)) {
        if (!dir || typeof dir !== "object") continue;
        const wfs = dir.workflows && typeof dir.workflows === "object" ? dir.workflows : {};
        const cleanedWfs = {};
        for (const [wfName, wf] of Object.entries(wfs)) {
            // Drop entries that can't be loaded (missing/non-object graph).
            // Coerce `archived` to a strict boolean so a stray "false" string
            // can't accidentally flag an entry as archived.
            if (!wf || typeof wf !== "object" || !wf.graph || typeof wf.graph !== "object") {
                dropped += 1;
                continue;
            }
            cleanedWfs[wfName] = { ...wf, archived: wf.archived === true };
        }
        out[name] = { workflows: cleanedWfs };
    }
    if (dropped > 0) {
        console.warn(`[Koolook] dropped ${dropped} malformed workflow entr(y/ies) during normalize`);
    }
    return { directories: out };
}

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
// Workflow operations (in-memory; pair every mutation with persistMutation())
// =============================================================================
export function listDirectoryNames() {
    return Object.keys(workflowsCache.directories || {}).sort(compareNames);
}

export function dirOf(name) {
    return workflowsCache.directories[name];
}

function ensureDirectory(name) {
    if (!workflowsCache.directories[name]) {
        workflowsCache.directories[name] = { workflows: {} };
    }
    return workflowsCache.directories[name];
}

export function addDirectory(name) {
    name = (name || "").trim();
    if (!name) return false;
    if (workflowsCache.directories[name]) return false; // already exists
    workflowsCache.directories[name] = { workflows: {} };
    return true;
}

export function renameDirectory(oldName, newName) {
    newName = (newName || "").trim();
    if (!newName || newName === oldName) return false;
    if (!workflowsCache.directories[oldName]) return false;
    if (workflowsCache.directories[newName]) return false;
    workflowsCache.directories[newName] = workflowsCache.directories[oldName];
    delete workflowsCache.directories[oldName];
    return true;
}

export function deleteDirectory(name) {
    if (!workflowsCache.directories[name]) return false;
    delete workflowsCache.directories[name];
    return true;
}

export function saveWorkflowEntry(dirName, wfName, graphData) {
    const dir = ensureDirectory(dirName);
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

export function archiveWorkflow(dirName, wfName) {
    const dir = dirOf(dirName);
    if (!dir || !dir.workflows[wfName]) return false;
    dir.workflows[wfName].archived = true;
    return true;
}

export function unarchiveWorkflow(dirName, wfName) {
    const dir = dirOf(dirName);
    if (!dir || !dir.workflows[wfName]) return false;
    delete dir.workflows[wfName].archived;
    return true;
}

export function renameWorkflow(dirName, oldWfName, newWfName) {
    newWfName = (newWfName || "").trim();
    if (!newWfName || newWfName === oldWfName) return false;
    const dir = dirOf(dirName);
    if (!dir || !dir.workflows[oldWfName]) return false;
    if (dir.workflows[newWfName]) return false;
    dir.workflows[newWfName] = dir.workflows[oldWfName];
    delete dir.workflows[oldWfName];
    return true;
}

export function deleteWorkflow(dirName, wfName) {
    const dir = dirOf(dirName);
    if (!dir || !dir.workflows[wfName]) return false;
    delete dir.workflows[wfName];
    return true;
}

export function moveWorkflow(srcDir, wfName, dstDir) {
    if (srcDir === dstDir) return false;
    const src = dirOf(srcDir);
    if (!src || !src.workflows[wfName]) return false;
    const dst = ensureDirectory(dstDir);
    if (dst.workflows[wfName]) return false; // name collision in destination
    dst.workflows[wfName] = src.workflows[wfName];
    delete src.workflows[wfName];
    return true;
}

export function getWorkflowGraph(dirName, wfName) {
    const dir = dirOf(dirName);
    if (!dir || !dir.workflows[wfName]) return null;
    return dir.workflows[wfName].graph || null;
}
