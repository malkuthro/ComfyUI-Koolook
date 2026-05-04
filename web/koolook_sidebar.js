import { app } from "../../../scripts/app.js";

// =============================================================================
// Configuration — static "always shown" entries.
//
// You normally don't need to edit this. To grow the panel from inside
// ComfyUI, use either:
//   • the "+" button at the top of the sidebar (with a canvas node selected), or
//   • right-click a canvas node → "Add to Curated Sidebar".
//
// Those additions are stored in the browser's localStorage and surface as
// the "My Picks" group at the top of the tree.
// =============================================================================
const REPOS = [
    {
        label: "Koolook",
        categoryRoot: "Koolook",
        select: "all",
        excludePatterns: [/\bdeprecated\b/i],
    },
];

const TAB_ID = "koolook.curatedNodes";
const TAB_TITLE = "Curated Nodes";
const TAB_TOOLTIP = "Curated selection of ComfyUI nodes";
const TAB_ICON = "pi pi-star";
const ROOT_GROUP_LABEL = "Nodes (favorites)";
const WORKFLOWS_GROUP_LABEL = "Workflows";

const STORAGE_KEY = "koolook.curated.userPicks.v1";
const SEEDED_KEY = "koolook.curated.seeded.v1";
const PICKS_CHANGED_EVENT = "koolook-picks-changed";
const DEFAULTS_URL = new URL("./curated_defaults.json", import.meta.url).href;

const WORKFLOWS_USERDATA_PATH = "koolook_workflows.json";
const WORKFLOWS_FALLBACK_KEY = "koolook.workflows.fallback.v1";
const WORKFLOWS_SEEDED_KEY = "koolook.workflows.seeded.v1";
const WORKFLOWS_CHANGED_EVENT = "koolook-workflows-changed";
const WORKFLOWS_DEFAULTS_URL = new URL("./workflow_defaults.json", import.meta.url).href;

// =============================================================================
// Styles
// =============================================================================
const STYLE_ID = "koolook-sidebar-style";
function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
.koolook-sidebar { display: flex; flex-direction: column; height: 100%; font-size: 13px; user-select: none; }
.koolook-search-row { margin: 6px; flex-shrink: 0; }
.koolook-actions-row { display: flex; align-items: center; gap: 4px; padding: 2px 6px; flex-shrink: 0; }
.koolook-actions-label { font-size: 10px; opacity: 0.55; text-transform: uppercase; letter-spacing: 0.08em; flex: 1; font-weight: 600; }
.koolook-tree-divider { margin: 8px 8px; border-top: 1px solid rgba(255,255,255,0.08); flex-shrink: 0; }
.koolook-search-wrap { position: relative; width: 100%; }
.koolook-search-icon { position: absolute; left: 8px; top: 50%; transform: translateY(-50%); opacity: 0.55; font-size: 11px; pointer-events: none; }
.koolook-search { width: 100%; padding: 5px 8px 5px 26px; box-sizing: border-box; background: var(--comfy-input-bg, rgba(0,0,0,0.25)); border: 1px solid var(--border-color, rgba(255,255,255,0.1)); border-radius: 4px; color: var(--input-text, inherit); font-size: 12px; outline: none; }
.koolook-search:focus { border-color: var(--p-primary-color, rgba(100,150,255,0.5)); }
.koolook-add-btn { padding: 0 12px; background: var(--comfy-input-bg, rgba(0,0,0,0.25)); border: 1px solid var(--border-color, rgba(255,255,255,0.1)); border-radius: 4px; cursor: pointer; color: var(--input-text, inherit); font-size: 16px; line-height: 1; flex-shrink: 0; }
.koolook-add-btn:hover { background: rgba(255,255,255,0.1); }
.koolook-add-btn:active { background: rgba(255,255,255,0.15); }
.koolook-export-btn { padding: 0 10px; font-size: 12px; }
.koolook-icon-btn { padding: 0 10px; font-size: 12px; }
.koolook-icon-btn:disabled { opacity: 0.35; cursor: not-allowed; }
.koolook-tree { flex: 1; overflow-y: auto; padding: 0 4px 8px; }
.koolook-row { display: flex; align-items: center; padding: 3px 6px; cursor: pointer; gap: 6px; border-radius: 3px; line-height: 1.3; }
.koolook-row:hover { background: var(--comfy-input-bg, rgba(255,255,255,0.06)); }
.koolook-chevron { width: 10px; display: inline-block; opacity: 0.7; text-align: center; font-size: 10px; flex-shrink: 0; }
.koolook-folder-icon { opacity: 0.85; flex-shrink: 0; }
.koolook-pin-icon { color: #ffb84d; opacity: 0.95; flex-shrink: 0; }
.koolook-workflows-icon { color: #6db4ff; opacity: 0.95; flex-shrink: 0; }
.koolook-archive-icon { opacity: 0.55; flex-shrink: 0; }
.koolook-leaf-icon { opacity: 0.7; flex-shrink: 0; font-size: 11px; }
.koolook-leaf-dot { width: 6px; height: 6px; margin: 0 2px; border-radius: 50%; background: rgba(255,255,255,0.45); flex-shrink: 0; }
.koolook-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.koolook-count { background: rgba(255,255,255,0.08); padding: 1px 7px; border-radius: 8px; font-size: 11px; opacity: 0.75; }
.koolook-children { padding-left: 14px; }
.koolook-empty { padding: 14px 8px; opacity: 0.65; font-size: 12px; line-height: 1.4; }
.koolook-leaf .koolook-chevron { visibility: hidden; }
.koolook-remove { padding: 0 4px; margin-left: 2px; opacity: 0; font-size: 14px; line-height: 1; cursor: pointer; flex-shrink: 0; }
.koolook-row:hover .koolook-remove { opacity: 0.5; }
.koolook-remove:hover { opacity: 1 !important; color: #ff7777; }
.koolook-section-divider { border-top: 1px solid var(--border-color, rgba(255,255,255,0.08)); margin: 8px 4px 0; }
.koolook-toast { position: fixed; bottom: 30px; right: 30px; background: rgba(40,40,40,0.95); color: #fff; padding: 8px 14px; border-radius: 4px; font-size: 12px; z-index: 9999; transition: opacity 0.3s; box-shadow: 0 2px 8px rgba(0,0,0,0.4); pointer-events: none; max-width: 360px; }
.koolook-modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.55); z-index: 9998; display: flex; align-items: center; justify-content: center; }
.koolook-modal { background: var(--comfy-menu-bg, #2a2a2a); border: 1px solid var(--border-color, rgba(255,255,255,0.15)); border-radius: 6px; padding: 16px 18px; min-width: 320px; max-width: 440px; box-shadow: 0 6px 24px rgba(0,0,0,0.55); color: var(--input-text, inherit); }
.koolook-modal-title { font-size: 14px; font-weight: 600; margin-bottom: 12px; }
.koolook-modal-message { font-size: 12px; opacity: 0.85; margin-bottom: 14px; line-height: 1.45; }
.koolook-modal-label { font-size: 11px; opacity: 0.7; margin: 6px 0 4px; display: block; text-transform: uppercase; letter-spacing: 0.04em; }
.koolook-modal-input, .koolook-modal-select { width: 100%; padding: 6px 8px; background: var(--comfy-input-bg, rgba(0,0,0,0.3)); border: 1px solid var(--border-color, rgba(255,255,255,0.1)); border-radius: 4px; color: inherit; font-size: 13px; box-sizing: border-box; outline: none; }
.koolook-modal-input:focus, .koolook-modal-select:focus { border-color: var(--p-primary-color, rgba(100,150,255,0.5)); }
.koolook-modal-actions { display: flex; gap: 8px; justify-content: flex-end; margin-top: 16px; }
.koolook-modal-btn { padding: 6px 14px; background: var(--comfy-input-bg, rgba(0,0,0,0.3)); border: 1px solid var(--border-color, rgba(255,255,255,0.1)); border-radius: 4px; color: inherit; cursor: pointer; font-size: 12px; }
.koolook-modal-btn:hover { background: rgba(255,255,255,0.1); }
.koolook-modal-btn-primary { background: rgba(80,140,235,0.35); border-color: rgba(80,140,235,0.55); }
.koolook-modal-btn-primary:hover { background: rgba(80,140,235,0.5); }
.koolook-modal-btn-danger { background: rgba(220,80,80,0.25); border-color: rgba(220,80,80,0.5); }
.koolook-modal-btn-danger:hover { background: rgba(220,80,80,0.4); }
.koolook-context-menu { position: fixed; background: var(--comfy-menu-bg, #2a2a2a); border: 1px solid var(--border-color, rgba(255,255,255,0.15)); border-radius: 4px; padding: 4px 0; min-width: 160px; z-index: 9999; box-shadow: 0 4px 12px rgba(0,0,0,0.4); font-size: 12px; }
.koolook-context-item { padding: 5px 12px; cursor: pointer; }
.koolook-context-item:hover { background: rgba(255,255,255,0.08); }
.koolook-context-danger { color: #ff8888; }
.koolook-context-sep { height: 1px; background: rgba(255,255,255,0.1); margin: 4px 0; }
.koolook-context-submenu-arrow { float: right; opacity: 0.5; }
`;
    document.head.appendChild(s);
}

// =============================================================================
// Toast helper
// =============================================================================
function toast(msg, duration = 2200) {
    const t = document.createElement("div");
    t.className = "koolook-toast";
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => {
        t.style.opacity = "0";
        setTimeout(() => t.remove(), 300);
    }, duration);
}

// =============================================================================
// Sort helper — human-alphabetical (case-insensitive). Use everywhere we sort
// names, labels, or directory entries so the order in the sidebar matches what
// a user would expect from an A→Z list.
// =============================================================================
function compareNames(a, b) {
    return String(a).localeCompare(String(b), undefined, { sensitivity: "base" });
}

// =============================================================================
// User picks (localStorage, browser-local persistence)
// =============================================================================
function loadUserPicks() {
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

function addToMyPicks(typeName) {
    if (!typeName) return false;
    const picks = loadUserPicks();
    if (picks.includes(typeName)) return false;
    picks.push(typeName);
    saveUserPicks(picks);
    return true;
}

function removeFromMyPicks(typeName) {
    saveUserPicks(loadUserPicks().filter(p => p !== typeName));
}

function notifyPicksChanged() {
    window.dispatchEvent(new CustomEvent(PICKS_CHANGED_EVENT));
}

async function seedDefaultsIfNeeded() {
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

async function exportPicks() {
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

// =============================================================================
// Folder expansion state — persisted across re-renders so saving (or any
// other action that triggers a tree rebuild) doesn't collapse what the user
// was looking at. Map<path, boolean>; truthy = expanded.
// =============================================================================
const pathStates = new Map();

// =============================================================================
// Workflows storage (ComfyUI /userdata API with localStorage fallback)
// =============================================================================
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

async function fetchWorkflowsFromServer() {
    try {
        const resp = await fetch(`/userdata/${WORKFLOWS_USERDATA_PATH}`);
        if (resp.status === 404) return null;
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const text = await resp.text();
        return text ? JSON.parse(text) : null;
    } catch (e) {
        console.warn("[Koolook] /userdata read failed, using localStorage fallback:", e);
        return undefined; // sentinel: server unreachable
    }
}

async function persistWorkflowsToServer(store) {
    const json = JSON.stringify(store, null, 2);
    try {
        const resp = await fetch(`/userdata/${WORKFLOWS_USERDATA_PATH}?overwrite=true`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: json,
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return true;
    } catch (e) {
        console.warn("[Koolook] /userdata write failed, using localStorage fallback:", e);
        try {
            localStorage.setItem(WORKFLOWS_FALLBACK_KEY, json);
            return true;
        } catch (e2) {
            console.error("[Koolook] both /userdata and localStorage write failed:", e2);
            return false;
        }
    }
}

async function loadWorkflowsStore() {
    const fromServer = await fetchWorkflowsFromServer();
    if (fromServer === null) {
        // Server reachable, file just doesn't exist yet — return empty.
        workflowsCache = { directories: {} };
        return workflowsCache;
    }
    if (fromServer === undefined) {
        // Server unreachable — try localStorage fallback.
        try {
            const raw = localStorage.getItem(WORKFLOWS_FALLBACK_KEY);
            if (raw) {
                workflowsCache = normalizeWorkflowsStore(JSON.parse(raw));
                return workflowsCache;
            }
        } catch (e) {
            console.warn("[Koolook] failed to parse localStorage workflows fallback:", e);
        }
        workflowsCache = { directories: {} };
        return workflowsCache;
    }
    workflowsCache = normalizeWorkflowsStore(fromServer);
    // Reconciliation hint: if /userdata loaded successfully but an old fallback
    // blob is still present, the user may have unsaved work from a prior outage.
    // We don't auto-merge (risk of clobbering) — surface it so they can recover
    // manually from DevTools if needed.
    if (localStorage.getItem(WORKFLOWS_FALLBACK_KEY)) {
        console.warn(
            `[Koolook] /userdata loaded, but a stale localStorage fallback exists ` +
            `at "${WORKFLOWS_FALLBACK_KEY}". If workflows you saved during a previous ` +
            `outage are missing, recover from there before clearing.`
        );
    }
    return workflowsCache;
}

async function commit() {
    const ok = await persistWorkflowsToServer(workflowsCache);
    if (ok) notifyWorkflowsChanged();
    return ok;
}

async function seedWorkflowDefaultsIfNeeded() {
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
            // Only mark seeded if the persist actually landed — otherwise the
            // next page load retries instead of being permanently blocked.
            if (!(await persistWorkflowsToServer(workflowsCache))) {
                console.warn("[Koolook] seed persist failed; will retry on next load");
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
// Workflow operations (in-memory, then commit())
// =============================================================================
function listDirectoryNames() {
    return Object.keys(workflowsCache.directories || {}).sort(compareNames);
}

function dirOf(name) {
    return workflowsCache.directories[name];
}

function ensureDirectory(name) {
    if (!workflowsCache.directories[name]) {
        workflowsCache.directories[name] = { workflows: {} };
    }
    return workflowsCache.directories[name];
}

function addDirectory(name) {
    name = (name || "").trim();
    if (!name) return false;
    if (workflowsCache.directories[name]) return false; // already exists
    workflowsCache.directories[name] = { workflows: {} };
    return true;
}

function renameDirectory(oldName, newName) {
    newName = (newName || "").trim();
    if (!newName || newName === oldName) return false;
    if (!workflowsCache.directories[oldName]) return false;
    if (workflowsCache.directories[newName]) return false;
    workflowsCache.directories[newName] = workflowsCache.directories[oldName];
    delete workflowsCache.directories[oldName];
    return true;
}

function deleteDirectory(name) {
    if (!workflowsCache.directories[name]) return false;
    delete workflowsCache.directories[name];
    return true;
}

function saveWorkflowEntry(dirName, wfName, graphData) {
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

function archiveWorkflow(dirName, wfName) {
    const dir = dirOf(dirName);
    if (!dir || !dir.workflows[wfName]) return false;
    dir.workflows[wfName].archived = true;
    return true;
}

function unarchiveWorkflow(dirName, wfName) {
    const dir = dirOf(dirName);
    if (!dir || !dir.workflows[wfName]) return false;
    delete dir.workflows[wfName].archived;
    return true;
}

function renameWorkflow(dirName, oldWfName, newWfName) {
    newWfName = (newWfName || "").trim();
    if (!newWfName || newWfName === oldWfName) return false;
    const dir = dirOf(dirName);
    if (!dir || !dir.workflows[oldWfName]) return false;
    if (dir.workflows[newWfName]) return false;
    dir.workflows[newWfName] = dir.workflows[oldWfName];
    delete dir.workflows[oldWfName];
    return true;
}

function deleteWorkflow(dirName, wfName) {
    const dir = dirOf(dirName);
    if (!dir || !dir.workflows[wfName]) return false;
    delete dir.workflows[wfName];
    return true;
}

function moveWorkflow(srcDir, wfName, dstDir) {
    if (srcDir === dstDir) return false;
    const src = dirOf(srcDir);
    if (!src || !src.workflows[wfName]) return false;
    const dst = ensureDirectory(dstDir);
    if (dst.workflows[wfName]) return false; // name collision in destination
    dst.workflows[wfName] = src.workflows[wfName];
    delete src.workflows[wfName];
    return true;
}

function getWorkflowGraph(dirName, wfName) {
    const dir = dirOf(dirName);
    if (!dir || !dir.workflows[wfName]) return null;
    return dir.workflows[wfName].graph || null;
}

// =============================================================================
// Selection / canvas serialization
// =============================================================================
function serializeFullCanvas() {
    try {
        return app.graph.serialize();
    } catch (e) {
        console.warn("[Koolook] graph.serialize() failed:", e);
        return null;
    }
}

function getSelectedNodeIds() {
    try {
        const sel = (app.canvas && app.canvas.selected_nodes) || {};
        return new Set(
            Object.values(sel)
                .filter(n => n && n.id != null)
                .map(n => n.id)
        );
    } catch (e) {
        return new Set();
    }
}

function serializeSelection() {
    const selectedIds = getSelectedNodeIds();
    if (selectedIds.size === 0) return null;
    const full = serializeFullCanvas();
    if (!full) return null;

    const internalLinks = (full.links || []).filter(link =>
        Array.isArray(link) && selectedIds.has(link[1]) && selectedIds.has(link[3])
    );
    const internalLinkIds = new Set(internalLinks.map(l => l[0]));

    const nodes = (full.nodes || [])
        .filter(n => selectedIds.has(n.id))
        .map(n => {
            const clone = JSON.parse(JSON.stringify(n));
            if (Array.isArray(clone.inputs)) {
                for (const inp of clone.inputs) {
                    if (inp && inp.link != null && !internalLinkIds.has(inp.link)) {
                        inp.link = null;
                    }
                }
            }
            if (Array.isArray(clone.outputs)) {
                for (const out of clone.outputs) {
                    if (out && Array.isArray(out.links)) {
                        out.links = out.links.filter(l => internalLinkIds.has(l));
                        if (out.links.length === 0) out.links = null;
                    }
                }
            }
            return clone;
        });

    if (nodes.length === 0) return null;

    return {
        last_node_id: full.last_node_id,
        last_link_id: full.last_link_id,
        nodes,
        links: internalLinks,
        groups: [],
        config: full.config || {},
        extra: full.extra || {},
        version: full.version || 0.4,
    };
}

function canvasIsNonEmpty() {
    try {
        return app.graph && app.graph._nodes && app.graph._nodes.length > 0;
    } catch (e) {
        return false;
    }
}

async function loadWorkflowOntoCanvas(dirName, wfName) {
    const graph = getWorkflowGraph(dirName, wfName);
    if (!graph) {
        toast(`Workflow not found: ${wfName}`);
        return;
    }
    const apply = async () => {
        try {
            // Passing `wfName` as the 4th arg makes loadGraphData create a
            // temporary workflow tab titled with that name (instead of
            // "Unsaved Workflow (N)"). Frontend reference: ComfyUI_frontend
            // src/scripts/app.ts (loadGraphData) →
            // workflowService.afterLoadNewGraph → createNewTemporary, which
            // builds path `workflows/<wfName>.json` and binds it to the tab.
            await app.loadGraphData(graph, true, true, wfName, {});

            // Defensive fallback for frontends that didn't honor the 4th arg.
            try {
                const wf = app.extensionManager?.workflow?.activeWorkflow;
                if (wf && wf.isTemporary && typeof wf.rename === "function" && wf.filename !== wfName) {
                    await wf.rename(`workflows/${wfName}.json`);
                }
            } catch (e) {
                console.warn("[Koolook] workflow rename fallback failed:", e);
            }

            toast(`Loaded "${wfName}".`);
        } catch (e) {
            console.error("[Koolook] loadGraphData failed:", e);
            toast(`Failed to load "${wfName}". See console.`);
        }
    };
    if (canvasIsNonEmpty()) {
        showConfirmModal({
            title: "Replace current workflow?",
            message: `Loading "${wfName}" will replace what's currently on the canvas. This cannot be undone with Ctrl+Z in some cases.`,
            confirmLabel: "Load anyway",
            danger: true,
            onConfirm: apply,
        });
    } else {
        apply();
    }
}

// =============================================================================
// Modals
// =============================================================================
function makeModalShell({ title, body, actions }) {
    const overlay = document.createElement("div");
    overlay.className = "koolook-modal-overlay";

    const modal = document.createElement("div");
    modal.className = "koolook-modal";

    const titleEl = document.createElement("div");
    titleEl.className = "koolook-modal-title";
    titleEl.textContent = title;
    modal.appendChild(titleEl);

    if (body) modal.appendChild(body);

    const actionsEl = document.createElement("div");
    actionsEl.className = "koolook-modal-actions";
    for (const action of actions) actionsEl.appendChild(action);
    modal.appendChild(actionsEl);

    overlay.appendChild(modal);
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) overlay.remove();
    });
    document.addEventListener("keydown", function escHandler(e) {
        if (e.key === "Escape") {
            overlay.remove();
            document.removeEventListener("keydown", escHandler);
        }
    });
    document.body.appendChild(overlay);
    return { overlay, modal };
}

function makeModalButton({ label, primary, danger, onClick }) {
    const btn = document.createElement("button");
    btn.className = "koolook-modal-btn";
    if (primary) btn.classList.add("koolook-modal-btn-primary");
    if (danger) btn.classList.add("koolook-modal-btn-danger");
    btn.textContent = label;
    btn.addEventListener("click", onClick);
    return btn;
}

function showInputModal({ title, label, defaultValue, placeholder, confirmLabel, onSubmit }) {
    const body = document.createElement("div");

    const lbl = document.createElement("label");
    lbl.className = "koolook-modal-label";
    lbl.textContent = label || "Name";
    body.appendChild(lbl);

    const input = document.createElement("input");
    input.className = "koolook-modal-input";
    input.value = defaultValue || "";
    input.placeholder = placeholder || "";
    body.appendChild(input);

    let overlay;
    const submit = () => {
        const v = input.value.trim();
        if (!v) {
            input.focus();
            return;
        }
        overlay.remove();
        onSubmit(v);
    };

    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") submit();
    });

    const cancel = makeModalButton({ label: "Cancel", onClick: () => overlay.remove() });
    const ok = makeModalButton({ label: confirmLabel || "OK", primary: true, onClick: submit });

    ({ overlay } = makeModalShell({ title, body, actions: [cancel, ok] }));
    setTimeout(() => { input.focus(); input.select(); }, 0);
}

function showConfirmModal({ title, message, confirmLabel, cancelLabel, danger, onConfirm }) {
    const body = document.createElement("div");
    const msg = document.createElement("div");
    msg.className = "koolook-modal-message";
    msg.textContent = message;
    body.appendChild(msg);

    let overlay;
    const cancel = makeModalButton({ label: cancelLabel || "Cancel", onClick: () => overlay.remove() });
    const ok = makeModalButton({
        label: confirmLabel || "OK",
        primary: !danger,
        danger,
        onClick: () => { overlay.remove(); onConfirm(); },
    });
    ({ overlay } = makeModalShell({ title, body, actions: [cancel, ok] }));
}

function showSaveWorkflowModal({ titleSuffix, defaultName, defaultDir, onSave }) {
    const body = document.createElement("div");

    // ---- Directory ----
    const dirLbl = document.createElement("label");
    dirLbl.className = "koolook-modal-label";
    dirLbl.textContent = "Directory";
    body.appendChild(dirLbl);

    const dirNames = listDirectoryNames();
    const dirSelect = document.createElement("select");
    dirSelect.className = "koolook-modal-select";
    if (dirNames.length === 0) {
        const opt = document.createElement("option");
        opt.value = "__new__";
        opt.textContent = "+ New directory…";
        dirSelect.appendChild(opt);
    } else {
        for (const d of dirNames) {
            const opt = document.createElement("option");
            opt.value = d;
            opt.textContent = d;
            if (d === defaultDir) opt.selected = true;
            dirSelect.appendChild(opt);
        }
        const newOpt = document.createElement("option");
        newOpt.value = "__new__";
        newOpt.textContent = "+ New directory…";
        dirSelect.appendChild(newOpt);
    }
    body.appendChild(dirSelect);

    const newDirInput = document.createElement("input");
    newDirInput.className = "koolook-modal-input";
    newDirInput.placeholder = "New directory name";
    newDirInput.style.marginTop = "6px";
    newDirInput.style.display = dirNames.length === 0 ? "" : "none";
    body.appendChild(newDirInput);

    // ---- Base on existing (only shown when the chosen directory has workflows) ----
    const baseLbl = document.createElement("label");
    baseLbl.className = "koolook-modal-label";
    baseLbl.textContent = "Base on existing";
    body.appendChild(baseLbl);

    const baseSelect = document.createElement("select");
    baseSelect.className = "koolook-modal-select";
    body.appendChild(baseSelect);

    // ---- Action ----
    const actionLbl = document.createElement("label");
    actionLbl.className = "koolook-modal-label";
    actionLbl.textContent = "Action";
    body.appendChild(actionLbl);

    const actionSelect = document.createElement("select");
    actionSelect.className = "koolook-modal-select";
    [
        { value: "new", label: "New name" },
        { value: "use_existing", label: "Use existing name (archive previous)" },
        { value: "modify_existing", label: "Modify existing name" },
    ].forEach(a => {
        const opt = document.createElement("option");
        opt.value = a.value;
        opt.textContent = a.label;
        actionSelect.appendChild(opt);
    });
    body.appendChild(actionSelect);

    // ---- Workflow name (only shown when needed) ----
    const nameLbl = document.createElement("label");
    nameLbl.className = "koolook-modal-label";
    nameLbl.textContent = "Workflow name";
    body.appendChild(nameLbl);

    const nameInput = document.createElement("input");
    nameInput.className = "koolook-modal-input";
    nameInput.value = defaultName || "";
    nameInput.placeholder = "My workflow";
    body.appendChild(nameInput);

    // ---- Wiring ----
    function getActiveWorkflowsInCurrentDir() {
        const v = dirSelect.value;
        if (v === "__new__") return [];
        const dir = dirOf(v);
        if (!dir) return [];
        return Object.keys(dir.workflows)
            .filter(n => !dir.workflows[n].archived)
            .sort(compareNames);
    }

    function rebuildBaseOptions(names) {
        const previous = baseSelect.value;
        baseSelect.innerHTML = "";
        for (const n of names) {
            const opt = document.createElement("option");
            opt.value = n;
            opt.textContent = n;
            baseSelect.appendChild(opt);
        }
        if (names.includes(previous)) baseSelect.value = previous;
    }

    function applyState({ refocusName = false } = {}) {
        const dirIsNew = dirSelect.value === "__new__";
        newDirInput.style.display = dirIsNew ? "" : "none";

        const activeNames = dirIsNew ? [] : getActiveWorkflowsInCurrentDir();
        const hasBase = activeNames.length > 0;

        // Base on existing — visible only when the chosen directory has active workflows.
        if (hasBase) {
            baseLbl.style.display = "";
            baseSelect.style.display = "";
            rebuildBaseOptions(activeNames);
        } else {
            baseLbl.style.display = "none";
            baseSelect.style.display = "none";
        }

        // Action options — disable "Use existing" / "Modify existing" when no base is available.
        for (const opt of actionSelect.options) {
            if (opt.value === "use_existing" || opt.value === "modify_existing") {
                opt.disabled = !hasBase;
            }
        }
        if (!hasBase && actionSelect.value !== "new") {
            actionSelect.value = "new";
        }

        // Workflow name — visible only when needed.
        const action = actionSelect.value;
        if (action === "use_existing") {
            // Name comes from the base; field is irrelevant.
            nameLbl.style.display = "none";
            nameInput.style.display = "none";
        } else {
            nameLbl.style.display = "";
            nameInput.style.display = "";
            if (action === "modify_existing" && baseSelect.value) {
                nameInput.value = baseSelect.value;
                nameInput.readOnly = false;
                if (refocusName) {
                    setTimeout(() => {
                        nameInput.focus();
                        const len = nameInput.value.length;
                        nameInput.setSelectionRange(len, len);
                    }, 0);
                }
            } else {
                // "new" — leave the value alone so the user's typing isn't clobbered.
                nameInput.readOnly = false;
            }
        }
    }

    dirSelect.addEventListener("change", () => {
        if (dirSelect.value === "__new__") newDirInput.focus();
        applyState();
    });
    actionSelect.addEventListener("change", () => applyState({ refocusName: true }));
    baseSelect.addEventListener("change", () => {
        if (actionSelect.value === "modify_existing") {
            nameInput.value = baseSelect.value;
        }
    });

    applyState();

    let overlay;
    const submit = async () => {
        let dir = dirSelect.value;
        if (dir === "__new__") {
            dir = newDirInput.value.trim();
            if (!dir) { newDirInput.focus(); return; }
        }
        const action = actionSelect.value;
        let name;
        if (action === "use_existing") {
            name = (baseSelect.value || "").trim();
            if (!name) { actionSelect.value = "new"; applyState(); nameInput.focus(); return; }
        } else {
            name = nameInput.value.trim();
            if (!name) { nameInput.focus(); return; }
        }
        overlay.remove();
        await onSave({ name, dir });
    };

    nameInput.addEventListener("keydown", (e) => { if (e.key === "Enter") submit(); });
    newDirInput.addEventListener("keydown", (e) => { if (e.key === "Enter") submit(); });

    const cancel = makeModalButton({ label: "Cancel", onClick: () => overlay.remove() });
    const ok = makeModalButton({ label: "Save", primary: true, onClick: submit });
    ({ overlay } = makeModalShell({
        title: titleSuffix ? `Save workflow — ${titleSuffix}` : "Save workflow",
        body,
        actions: [cancel, ok],
    }));
    setTimeout(() => {
        if (nameInput.style.display !== "none") {
            nameInput.focus();
            nameInput.select();
        }
    }, 0);
}

// =============================================================================
// Context menu helper
// =============================================================================
function showContextMenu(event, items) {
    event.preventDefault();
    event.stopPropagation();

    const menu = document.createElement("div");
    menu.className = "koolook-context-menu";

    for (const item of items) {
        if (!item) {
            const sep = document.createElement("div");
            sep.className = "koolook-context-sep";
            menu.appendChild(sep);
            continue;
        }
        const m = document.createElement("div");
        m.className = "koolook-context-item";
        if (item.danger) m.classList.add("koolook-context-danger");
        m.textContent = item.label;
        if (item.disabled) {
            m.style.opacity = "0.4";
            m.style.cursor = "not-allowed";
        } else {
            m.addEventListener("click", () => {
                menu.remove();
                item.action();
            });
        }
        menu.appendChild(m);
    }

    menu.style.left = `${event.clientX}px`;
    menu.style.top = `${event.clientY}px`;
    document.body.appendChild(menu);

    const rect = menu.getBoundingClientRect();
    if (rect.right > window.innerWidth) menu.style.left = `${event.clientX - rect.width}px`;
    if (rect.bottom > window.innerHeight) menu.style.top = `${event.clientY - rect.height}px`;

    setTimeout(() => {
        const closeOnClick = (ev) => {
            if (!menu.contains(ev.target)) {
                menu.remove();
                document.removeEventListener("click", closeOnClick);
                document.removeEventListener("contextmenu", closeOnClick);
            }
        };
        document.addEventListener("click", closeOnClick);
        document.addEventListener("contextmenu", closeOnClick);
    }, 0);
}

// =============================================================================
// Data gathering — nodes
// =============================================================================
function subcategoryFor(category, categoryRoot) {
    if (!category) return "(uncategorized)";
    if (category === categoryRoot) return "(root)";
    if (categoryRoot && category.startsWith(categoryRoot + "/")) {
        return category.slice(categoryRoot.length + 1);
    }
    const parts = category.split("/");
    return parts.length > 1 ? parts.slice(1).join("/") : category;
}

function matchesQuery(display, type, q) {
    if (!q) return true;
    return display.toLowerCase().includes(q) || type.toLowerCase().includes(q);
}

function gatherNodesByRepo(query) {
    const q = (query || "").trim().toLowerCase();
    const registry = (typeof LiteGraph !== "undefined" && LiteGraph.registered_node_types) || {};
    const out = [];

    for (const repo of REPOS) {
        let candidateIds;
        if (repo.select === "all") {
            const root = repo.categoryRoot || "";
            candidateIds = Object.entries(registry)
                .filter(([, nc]) => {
                    const cat = (nc && nc.category) || "";
                    return root && (cat === root || cat.startsWith(root + "/"));
                })
                .map(([type]) => type);
        } else if (Array.isArray(repo.select)) {
            candidateIds = repo.select.filter(t => registry[t] !== undefined);
        } else {
            candidateIds = [];
        }

        const subcats = new Map();
        let total = 0;
        for (const type of candidateIds) {
            const nc = registry[type];
            const display = (nc && nc.title) || type;
            if (repo.excludePatterns && repo.excludePatterns.some(re => re.test(display))) continue;
            if (!matchesQuery(display, type, q)) continue;

            const sub = subcategoryFor((nc && nc.category) || "", repo.categoryRoot);
            if (!subcats.has(sub)) subcats.set(sub, []);
            subcats.get(sub).push({ type, display });
            total += 1;
        }

        const categories = [...subcats.entries()]
            .map(([name, nodes]) => ({
                name,
                nodes: nodes.sort((a, b) => compareNames(a.display, b.display)),
            }))
            .sort((a, b) => compareNames(a.name, b.name));

        out.push({ label: repo.label, categories, total });
    }
    return out;
}

function gatherUserPickPacks(query) {
    const q = (query || "").trim().toLowerCase();
    const registry = (typeof LiteGraph !== "undefined" && LiteGraph.registered_node_types) || {};
    const picks = loadUserPicks();

    const autoCategoryRoots = new Set(
        REPOS.filter(r => r.select === "all" && r.categoryRoot).map(r => r.categoryRoot)
    );

    const byPack = new Map();
    for (const type of picks) {
        const nc = registry[type];
        if (!nc) continue;
        const cat = nc.category || "";
        const packLabel = cat.split("/")[0] || "(uncategorized)";
        if (autoCategoryRoots.has(packLabel)) continue;

        const display = nc.title || type;
        if (!matchesQuery(display, type, q)) continue;

        const sub = subcategoryFor(cat, packLabel);
        if (!byPack.has(packLabel)) byPack.set(packLabel, new Map());
        const subcatMap = byPack.get(packLabel);
        if (!subcatMap.has(sub)) subcatMap.set(sub, []);
        subcatMap.get(sub).push({ type, display });
    }

    const result = [];
    for (const [packLabel, subcatMap] of byPack.entries()) {
        const categories = [...subcatMap.entries()]
            .map(([name, nodes]) => ({
                name,
                nodes: nodes.sort((a, b) => compareNames(a.display, b.display)),
            }))
            .sort((a, b) => compareNames(a.name, b.name));
        const total = categories.reduce((acc, c) => acc + c.nodes.length, 0);
        result.push({ label: packLabel, categories, total, isUserPicks: true });
    }
    result.sort((a, b) => compareNames(a.label, b.label));
    return result;
}

// =============================================================================
// Data gathering — workflows
// =============================================================================
function gatherWorkflows(query) {
    const q = (query || "").trim().toLowerCase();
    const directories = workflowsCache.directories || {};
    const out = [];
    let total = 0;

    for (const dirName of Object.keys(directories).sort(compareNames)) {
        const dir = directories[dirName];
        const allNames = Object.keys(dir.workflows || {});
        const matches = (n) => !q || n.toLowerCase().includes(q) || dirName.toLowerCase().includes(q);

        const active = [];
        const archived = [];
        for (const n of allNames) {
            if (!matches(n)) continue;
            if (dir.workflows[n] && dir.workflows[n].archived) archived.push(n);
            else active.push(n);
        }
        active.sort(compareNames);
        archived.sort(compareNames);
        if (active.length === 0 && archived.length === 0) continue;
        out.push({ name: dirName, active, archived });
        total += active.length + archived.length;
    }

    return { directories: out, total };
}

// =============================================================================
// Node insertion
// =============================================================================
function placeAtCanvasCenter(node) {
    try {
        const canvas = app.canvas;
        const ds = canvas.ds;
        const cx = -ds.offset[0] + canvas.canvas.width / (2 * ds.scale);
        const cy = -ds.offset[1] + canvas.canvas.height / (2 * ds.scale);
        node.pos = [cx - node.size[0] / 2, cy - node.size[1] / 2];
    } catch (e) {
        // Default position is fine if the canvas isn't ready yet.
    }
}

function insertNode(typeName) {
    if (typeof LiteGraph === "undefined") return;
    const node = LiteGraph.createNode(typeName);
    if (!node) {
        console.warn(`[Koolook] could not create node: ${typeName}`);
        return;
    }
    app.graph.add(node);
    placeAtCanvasCenter(node);
    app.canvas.setDirty(true, true);
}

function getSelectedNodeTypes() {
    try {
        const sel = (app.canvas && app.canvas.selected_nodes) || {};
        return Object.values(sel)
            .filter(n => n && n.type)
            .map(n => n.type);
    } catch (e) {
        return [];
    }
}

// =============================================================================
// DOM helpers
// =============================================================================
function makeFolderRow({ name, count, iconKind, onToggle, onContextMenu }) {
    const row = document.createElement("div");
    row.className = "koolook-row";

    const chevron = document.createElement("span");
    chevron.className = "koolook-chevron";
    chevron.textContent = "▾";
    row.appendChild(chevron);

    const icon = document.createElement("span");
    if (iconKind === "favorites") {
        icon.className = "pi pi-star koolook-pin-icon";
    } else if (iconKind === "workflows") {
        icon.className = "pi pi-th-large koolook-workflows-icon";
    } else if (iconKind === "archive") {
        icon.className = "pi pi-box koolook-archive-icon";
    } else {
        icon.className = "pi pi-folder koolook-folder-icon";
    }
    row.appendChild(icon);

    const nameEl = document.createElement("span");
    nameEl.className = "koolook-name";
    nameEl.textContent = name;
    row.appendChild(nameEl);

    if (count != null) {
        const cnt = document.createElement("span");
        cnt.className = "koolook-count";
        cnt.textContent = String(count);
        row.appendChild(cnt);
    }

    row.addEventListener("click", onToggle);
    if (onContextMenu) row.addEventListener("contextmenu", onContextMenu);
    return { row, chevron };
}

function makeNodeLeafRow({ display, type, removable, onClick }) {
    const row = document.createElement("div");
    row.className = "koolook-row koolook-leaf";
    row.title = type;

    const chevron = document.createElement("span");
    chevron.className = "koolook-chevron";
    row.appendChild(chevron);

    const dot = document.createElement("span");
    dot.className = "koolook-leaf-dot";
    row.appendChild(dot);

    const nameEl = document.createElement("span");
    nameEl.className = "koolook-name";
    nameEl.textContent = display;
    row.appendChild(nameEl);

    if (removable) {
        const rm = document.createElement("span");
        rm.className = "koolook-remove";
        rm.textContent = "×";
        rm.title = "Remove from favorites";
        rm.addEventListener("click", (e) => {
            e.stopPropagation();
            removeFromMyPicks(type);
            notifyPicksChanged();
        });
        row.appendChild(rm);
    }

    row.addEventListener("click", onClick);
    return row;
}

function makeWorkflowLeafRow({ name, dirName, onClick, onContextMenu }) {
    const row = document.createElement("div");
    row.className = "koolook-row koolook-leaf";
    row.title = `${dirName} / ${name}`;

    const chevron = document.createElement("span");
    chevron.className = "koolook-chevron";
    row.appendChild(chevron);

    const icon = document.createElement("span");
    icon.className = "pi pi-file koolook-leaf-icon";
    row.appendChild(icon);

    const nameEl = document.createElement("span");
    nameEl.className = "koolook-name";
    nameEl.textContent = name;
    row.appendChild(nameEl);

    row.addEventListener("click", onClick);
    if (onContextMenu) row.addEventListener("contextmenu", onContextMenu);
    return row;
}

function buildFolder({ name, count, iconKind, childrenBuilder, onContextMenu, startExpanded = true, path = null, forceExpanded = false }) {
    // Resolution order:
    //   1. forceExpanded (e.g. search active) — overrides everything
    //   2. pathStates (user has previously toggled this folder)
    //   3. startExpanded (the natural default for this folder)
    let initiallyExpanded;
    if (forceExpanded) {
        initiallyExpanded = true;
    } else if (path && pathStates.has(path)) {
        initiallyExpanded = pathStates.get(path);
    } else {
        initiallyExpanded = startExpanded;
    }

    const wrapper = document.createElement("div");
    wrapper.dataset.expanded = initiallyExpanded ? "true" : "false";

    const children = document.createElement("div");
    children.className = "koolook-children";
    if (!initiallyExpanded) children.style.display = "none";

    const { row, chevron } = makeFolderRow({
        name,
        count,
        iconKind,
        onContextMenu,
        onToggle: () => {
            const expanded = wrapper.dataset.expanded !== "false";
            const next = !expanded;
            wrapper.dataset.expanded = next ? "true" : "false";
            chevron.textContent = next ? "▾" : "▸";
            children.style.display = next ? "" : "none";
            if (path) pathStates.set(path, next);
        },
    });
    if (!initiallyExpanded) chevron.textContent = "▸";

    wrapper.appendChild(row);
    wrapper.appendChild(children);
    childrenBuilder(children);
    return wrapper;
}

// =============================================================================
// Context-menu wiring for workflow rows
// =============================================================================
function workflowRowContextMenu(event, dirName, wfName, isArchived = false) {
    const dirNames = listDirectoryNames();
    const moveItems = dirNames
        .filter(d => d !== dirName)
        .map(d => ({
            label: `→ ${d}`,
            action: async () => {
                if (moveWorkflow(dirName, wfName, d)) {
                    await commit();
                    toast(`Moved "${wfName}" to ${d}.`);
                } else {
                    toast(`Could not move (name conflict?).`);
                }
            },
        }));

    const archiveItem = isArchived
        ? {
            label: "Restore from archive",
            action: async () => {
                if (unarchiveWorkflow(dirName, wfName)) {
                    await commit();
                    toast(`Restored "${wfName}".`);
                }
            },
        }
        : {
            label: "Move to archive",
            action: async () => {
                if (archiveWorkflow(dirName, wfName)) {
                    await commit();
                    toast(`Archived "${wfName}".`);
                }
            },
        };

    showContextMenu(event, [
        {
            label: "Load",
            action: () => loadWorkflowOntoCanvas(dirName, wfName),
        },
        {
            label: "Rename…",
            action: () => {
                showInputModal({
                    title: "Rename workflow",
                    label: "New name",
                    defaultValue: wfName,
                    confirmLabel: "Rename",
                    onSubmit: async (newName) => {
                        if (renameWorkflow(dirName, wfName, newName)) {
                            await commit();
                            toast(`Renamed to "${newName}".`);
                        } else {
                            toast(`Rename failed (name in use?).`);
                        }
                    },
                });
            },
        },
        archiveItem,
        ...(moveItems.length > 0 ? [null, ...moveItems] : []),
        null,
        {
            label: "Delete",
            danger: true,
            action: () => {
                showConfirmModal({
                    title: "Delete workflow?",
                    message: `"${wfName}" will be removed. This cannot be undone.`,
                    confirmLabel: "Delete",
                    danger: true,
                    onConfirm: async () => {
                        if (deleteWorkflow(dirName, wfName)) {
                            await commit();
                            toast(`Deleted "${wfName}".`);
                        }
                    },
                });
            },
        },
    ]);
}

function directoryRowContextMenu(event, dirName) {
    const dir = dirOf(dirName);
    const isEmpty = !dir || Object.keys(dir.workflows || {}).length === 0;

    showContextMenu(event, [
        {
            label: "Rename directory…",
            action: () => {
                showInputModal({
                    title: "Rename directory",
                    label: "New name",
                    defaultValue: dirName,
                    confirmLabel: "Rename",
                    onSubmit: async (newName) => {
                        if (renameDirectory(dirName, newName)) {
                            await commit();
                            toast(`Renamed to "${newName}".`);
                        } else {
                            toast(`Rename failed (name in use?).`);
                        }
                    },
                });
            },
        },
        null,
        {
            label: "Delete directory",
            danger: true,
            action: () => {
                if (isEmpty) {
                    showConfirmModal({
                        title: "Delete empty directory?",
                        message: `"${dirName}" will be removed.`,
                        confirmLabel: "Delete",
                        danger: true,
                        onConfirm: async () => {
                            if (deleteDirectory(dirName)) {
                                await commit();
                                toast(`Deleted directory "${dirName}".`);
                            }
                        },
                    });
                } else {
                    const count = Object.keys(dir.workflows).length;
                    showConfirmModal({
                        title: "Delete non-empty directory?",
                        message: `"${dirName}" contains ${count} workflow${count === 1 ? "" : "s"}. They will be permanently deleted.`,
                        confirmLabel: "Delete all",
                        danger: true,
                        onConfirm: async () => {
                            if (deleteDirectory(dirName)) {
                                await commit();
                                toast(`Deleted "${dirName}" and its ${count} workflow${count === 1 ? "" : "s"}.`);
                            }
                        },
                    });
                }
            },
        },
    ]);
}

// =============================================================================
// Tree rendering
// =============================================================================
function rebuildTree(treeEl, query) {
    treeEl.innerHTML = "";
    const isFiltered = (query || "").trim().length > 0;

    // ------ Gather both sections first so we know whether to render a divider ------
    const autoPacks = gatherNodesByRepo(query);
    const userPickPacks = gatherUserPickPacks(query);
    const allPacks = [...autoPacks, ...userPickPacks]
        .filter(p => p.total > 0)
        .sort((a, b) => compareNames(a.label, b.label));
    const nodesTotal = allPacks.reduce((acc, p) => acc + p.total, 0);
    const showNodes = nodesTotal > 0;

    const wf = gatherWorkflows(query);
    const showWorkflows = wf.directories.length > 0;

    // Search returned nothing across both sections.
    if (!showNodes && !showWorkflows && isFiltered) {
        const empty = document.createElement("div");
        empty.className = "koolook-empty";
        empty.textContent = "No nodes or workflows match your search.";
        treeEl.appendChild(empty);
        return;
    }

    // ------ Nodes section ------
    if (showNodes) {
        const nodesGroup = buildFolder({
            name: ROOT_GROUP_LABEL,
            count: nodesTotal,
            iconKind: "favorites",
            startExpanded: true,
            path: "nodes",
            forceExpanded: isFiltered,
            childrenBuilder: (rootChildren) => {
                for (const pack of allPacks) {
                    const packFolder = buildFolder({
                        name: pack.label,
                        count: pack.total,
                        startExpanded: false,
                        path: `nodes/${pack.label}`,
                        forceExpanded: isFiltered,
                        childrenBuilder: (packChildren) => {
                            for (const cat of pack.categories) {
                                const catFolder = buildFolder({
                                    name: cat.name,
                                    count: cat.nodes.length,
                                    startExpanded: false,
                                    path: `nodes/${pack.label}/${cat.name}`,
                                    forceExpanded: isFiltered,
                                    childrenBuilder: (catChildren) => {
                                        for (const n of cat.nodes) {
                                            catChildren.appendChild(makeNodeLeafRow({
                                                display: n.display,
                                                type: n.type,
                                                removable: !!pack.isUserPicks,
                                                onClick: () => insertNode(n.type),
                                            }));
                                        }
                                    },
                                });
                                packChildren.appendChild(catFolder);
                            }
                        },
                    });
                    rootChildren.appendChild(packFolder);
                }
            },
        });
        treeEl.appendChild(nodesGroup);
    } else if (!isFiltered) {
        const empty = document.createElement("div");
        empty.className = "koolook-empty";
        empty.textContent = "No curated nodes yet. Click + above (with a canvas node selected) or right-click a node on the canvas → Add to Curated Sidebar.";
        treeEl.appendChild(empty);
    }

    // ------ Divider between sections ------
    if (showNodes && showWorkflows) {
        const divider = document.createElement("div");
        divider.className = "koolook-tree-divider";
        treeEl.appendChild(divider);
    }

    // ------ Workflows section ------
    if (showWorkflows) {
        const workflowsGroup = buildFolder({
            name: WORKFLOWS_GROUP_LABEL,
            count: wf.total,
            iconKind: "workflows",
            startExpanded: true,
            path: "workflows",
            forceExpanded: isFiltered,
            childrenBuilder: (wfChildren) => {
                for (const dir of wf.directories) {
                    const totalInDir = dir.active.length + dir.archived.length;
                    const dirFolder = buildFolder({
                        name: dir.name,
                        count: totalInDir,
                        startExpanded: false,
                        path: `workflows/${dir.name}`,
                        forceExpanded: isFiltered,
                        onContextMenu: (e) => directoryRowContextMenu(e, dir.name),
                        childrenBuilder: (dirChildren) => {
                            // Archive first, then active — so the latest (active)
                            // workflow is closest to the bottom of the directory.
                            if (dir.archived.length > 0) {
                                const archiveFolder = buildFolder({
                                    name: "Archive",
                                    count: dir.archived.length,
                                    iconKind: "archive",
                                    startExpanded: false,
                                    path: `workflows/${dir.name}/Archive`,
                                    forceExpanded: isFiltered,
                                    childrenBuilder: (archiveChildren) => {
                                        for (const wfName of dir.archived) {
                                            archiveChildren.appendChild(makeWorkflowLeafRow({
                                                name: wfName,
                                                dirName: dir.name,
                                                onClick: () => loadWorkflowOntoCanvas(dir.name, wfName),
                                                onContextMenu: (e) => workflowRowContextMenu(e, dir.name, wfName, true),
                                            }));
                                        }
                                    },
                                });
                                dirChildren.appendChild(archiveFolder);
                            }
                            for (const wfName of dir.active) {
                                dirChildren.appendChild(makeWorkflowLeafRow({
                                    name: wfName,
                                    dirName: dir.name,
                                    onClick: () => loadWorkflowOntoCanvas(dir.name, wfName),
                                    onContextMenu: (e) => workflowRowContextMenu(e, dir.name, wfName, false),
                                }));
                            }
                        },
                    });
                    wfChildren.appendChild(dirFolder);
                }
            },
        });
        treeEl.appendChild(workflowsGroup);
    }
    // (When workflows is empty and not filtered, we silently hide the section;
    // the toolbar above still lets the user create a directory or save a workflow.)
}

// =============================================================================
// Panel rendering
// =============================================================================
function renderPanel(container) {
    ensureStyle();
    container.innerHTML = "";
    container.classList.add("koolook-sidebar");

    // ---- Row 1: Search ----
    const searchRow = document.createElement("div");
    searchRow.className = "koolook-search-row";

    const searchWrap = document.createElement("div");
    searchWrap.className = "koolook-search-wrap";

    const searchIcon = document.createElement("span");
    searchIcon.className = "pi pi-search koolook-search-icon";
    searchWrap.appendChild(searchIcon);

    const search = document.createElement("input");
    search.type = "search";
    search.className = "koolook-search";
    search.placeholder = "Search nodes & workflows...";
    searchWrap.appendChild(search);

    searchRow.appendChild(searchWrap);
    container.appendChild(searchRow);

    // ---- Row 2: Nodes action bar ----
    const nodesRow = document.createElement("div");
    nodesRow.className = "koolook-actions-row";

    const nodesLabel = document.createElement("span");
    nodesLabel.className = "koolook-actions-label";
    nodesLabel.textContent = "Nodes";
    nodesRow.appendChild(nodesLabel);

    const addBtn = document.createElement("button");
    addBtn.className = "koolook-add-btn";
    addBtn.textContent = "+";
    addBtn.title = "Add the selected canvas node(s) to favorites";
    addBtn.addEventListener("click", () => {
        const types = getSelectedNodeTypes();
        if (types.length === 0) {
            toast("Select a node on the canvas first.");
            return;
        }
        let added = 0;
        for (const t of types) {
            if (addToMyPicks(t)) added += 1;
        }
        if (added > 0) {
            const noun = added === 1 ? "node" : "nodes";
            toast(`Added ${added} ${noun} to favorites.`);
            notifyPicksChanged();
        } else {
            toast("Already in favorites.");
        }
    });
    nodesRow.appendChild(addBtn);

    const exportBtn = document.createElement("button");
    exportBtn.className = "koolook-add-btn koolook-export-btn";
    exportBtn.innerHTML = '<span class="pi pi-download"></span>';
    exportBtn.title = "Export current picks as curated_defaults.json (copies JSON to clipboard)";
    exportBtn.addEventListener("click", exportPicks);
    nodesRow.appendChild(exportBtn);

    container.appendChild(nodesRow);

    // Divider between the two action bars
    const dividerBetweenBars = document.createElement("div");
    dividerBetweenBars.className = "koolook-tree-divider";
    container.appendChild(dividerBetweenBars);

    // ---- Row 3: Workflows action bar ----
    const wfRow = document.createElement("div");
    wfRow.className = "koolook-actions-row";

    const wfLabel = document.createElement("span");
    wfLabel.className = "koolook-actions-label";
    wfLabel.textContent = "Workflows";
    wfRow.appendChild(wfLabel);

    const newDirBtn = document.createElement("button");
    newDirBtn.className = "koolook-add-btn koolook-icon-btn";
    newDirBtn.innerHTML = '<span class="pi pi-folder-open"></span>';
    newDirBtn.title = "Create new workflow directory";
    newDirBtn.addEventListener("click", () => {
        showInputModal({
            title: "Create directory",
            label: "Directory name",
            placeholder: "e.g. Inpainting",
            confirmLabel: "Create",
            onSubmit: async (name) => {
                if (addDirectory(name)) {
                    await commit();
                    toast(`Directory "${name}" created.`);
                } else {
                    toast(`Directory "${name}" already exists.`);
                }
            },
        });
    });
    wfRow.appendChild(newDirBtn);

    // Shared save handler for both buttons. Keeps "save → mark visible → commit
    // → toast" flow in one place; toasts a failure if commit() returns false so
    // the user isn't told a write succeeded when both /userdata and the
    // localStorage fallback failed.
    const saveAndToast = async (graph, name, dir) => {
        const result = saveWorkflowEntry(dir, name, graph);
        pathStates.set("workflows", true);
        pathStates.set(`workflows/${dir}`, true);
        const ok = await commit();
        if (!ok) {
            toast(`Save failed — could not write "${name}". See console.`);
            return;
        }
        if (result.archivedAs) {
            toast(`Saved "${name}" in ${dir}. Previous version moved to Archive.`);
        } else {
            toast(`Saved "${name}" in ${dir}.`);
        }
    };

    const saveCanvasBtn = document.createElement("button");
    saveCanvasBtn.className = "koolook-add-btn koolook-icon-btn";
    saveCanvasBtn.innerHTML = '<span class="pi pi-save"></span>';
    saveCanvasBtn.title = "Save entire canvas as a workflow";
    saveCanvasBtn.addEventListener("click", () => {
        // Check emptiness first so a serialize() throw doesn't get misreported
        // as "Canvas is empty" — distinct toasts for distinct failure modes.
        if (!canvasIsNonEmpty()) {
            toast("Canvas is empty.");
            return;
        }
        const graph = serializeFullCanvas();
        if (!graph || !graph.nodes || graph.nodes.length === 0) {
            toast("Failed to serialize canvas. See console.");
            return;
        }
        showSaveWorkflowModal({
            titleSuffix: "entire canvas",
            onSave: ({ name, dir }) => saveAndToast(graph, name, dir),
        });
    });
    wfRow.appendChild(saveCanvasBtn);

    const saveSelectionBtn = document.createElement("button");
    saveSelectionBtn.className = "koolook-add-btn koolook-icon-btn";
    saveSelectionBtn.innerHTML = '<span class="pi pi-objects-column"></span>';
    saveSelectionBtn.title = "Save current selection as a workflow";
    saveSelectionBtn.addEventListener("click", () => {
        const graph = serializeSelection();
        if (!graph) {
            toast("Select one or more nodes on the canvas first.");
            return;
        }
        showSaveWorkflowModal({
            titleSuffix: `${graph.nodes.length} selected node${graph.nodes.length === 1 ? "" : "s"}`,
            onSave: ({ name, dir }) => saveAndToast(graph, name, dir),
        });
    });
    wfRow.appendChild(saveSelectionBtn);

    container.appendChild(wfRow);

    // Divider between the workflows action bar and the list below
    const dividerBeforeTree = document.createElement("div");
    dividerBeforeTree.className = "koolook-tree-divider";
    container.appendChild(dividerBeforeTree);

    // ---- Tree (scrollable) ----
    const tree = document.createElement("div");
    tree.className = "koolook-tree";
    container.appendChild(tree);

    rebuildTree(tree, "");

    // ---- Search wiring ----
    let debounce = null;
    search.addEventListener("input", (e) => {
        const q = e.target.value;
        if (debounce) clearTimeout(debounce);
        debounce = setTimeout(() => rebuildTree(tree, q), 60);
    });

    // ---- Event subscriptions for live re-rendering ----
    window.addEventListener(PICKS_CHANGED_EVENT, () => rebuildTree(tree, search.value));
    window.addEventListener(WORKFLOWS_CHANGED_EVENT, () => rebuildTree(tree, search.value));
    window.addEventListener("storage", (e) => {
        if (e.key === STORAGE_KEY || e.key === WORKFLOWS_FALLBACK_KEY) {
            rebuildTree(tree, search.value);
        }
    });
}

// =============================================================================
// Right-click context menu on canvas nodes — adds to favorites
// =============================================================================
function patchCanvasMenu() {
    const C = (typeof LGraphCanvas !== "undefined") ? LGraphCanvas : null;
    if (!C || !C.prototype || !C.prototype.getNodeMenuOptions) {
        console.warn("[Koolook] LGraphCanvas.getNodeMenuOptions not available; right-click menu skipped.");
        return;
    }
    const orig = C.prototype.getNodeMenuOptions;
    C.prototype.getNodeMenuOptions = function (node) {
        const options = orig.apply(this, arguments);
        options.push(null);
        options.push({
            content: "Add to Curated Sidebar",
            callback: () => {
                if (!node || !node.type) return;
                if (addToMyPicks(node.type)) {
                    toast(`Added "${node.title || node.type}" to favorites.`);
                    notifyPicksChanged();
                } else {
                    toast("Already in favorites.");
                }
            },
        });
        return options;
    };
}

// =============================================================================
// Tab registration
// =============================================================================
app.registerExtension({
    name: "koolook.curated_sidebar",
    async setup() {
        if (!app.extensionManager || !app.extensionManager.registerSidebarTab) {
            console.warn("[Koolook] extensionManager.registerSidebarTab not available; sidebar not registered.");
            return;
        }
        await seedDefaultsIfNeeded();
        await loadWorkflowsStore();
        await seedWorkflowDefaultsIfNeeded();
        app.extensionManager.registerSidebarTab({
            id: TAB_ID,
            title: TAB_TITLE,
            tooltip: TAB_TOOLTIP,
            icon: TAB_ICON,
            type: "custom",
            render: (el) => renderPanel(el),
        });
        patchCanvasMenu();
    },
});
