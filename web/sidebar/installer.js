// =============================================================================
// ComfyUI-Manager API client + pick → install resolver.
//
// Detects whether Manager is reachable, walks the user's curated picks
// against /customnode/getmappings to find which packs need installing,
// queues git-URL installs, polls the install queue for progress, and
// reboots. UI-free — modals.js drives the user-facing flow on top of the
// functions exported here.
//
// Routes referenced (Manager source pins):
//   GET  /manager/queue/status         → reachability probe + progress poll
//   GET  /customnode/getmappings?mode  → unified node-id ↔ git-url mapping
//   POST /customnode/install/git_url   → queue one repo for install (text body)
//   POST /manager/queue/start          → kick off the worker thread
//   POST /manager/reboot               → restart ComfyUI to load installed code
//
// `?mode=local` is REQUIRED on getmappings — Manager errors with KeyError if
// the query param is absent (manager_server.py: `request.rel_url.query["mode"]`).
// =============================================================================

const ENDPOINT_QUEUE_STATUS = "/manager/queue/status";
const ENDPOINT_QUEUE_START = "/manager/queue/start";
const ENDPOINT_GETMAPPINGS = "/customnode/getmappings?mode=local";
const ENDPOINT_INSTALL_GIT_URL = "/customnode/install/git_url";
const ENDPOINT_REBOOT = "/manager/reboot";

export const POLL_INTERVAL_MS = 1000;

// =============================================================================
// Detection
// =============================================================================

// Reachability probe. The status endpoint is unauthenticated and side-effect-
// free, so a 200 here is a strong signal Manager is loaded and serving routes.
// Anything else (network error, 404 — e.g. user disabled Manager) routes the
// caller into the copy-URL fallback path.
export async function detectManager() {
    try {
        const r = await fetch(ENDPOINT_QUEUE_STATUS, { method: "GET" });
        return r.ok;
    } catch (e) {
        console.warn("[Koolook] Manager reachability probe failed:", e);
        return false;
    }
}

// =============================================================================
// Mapping load + pick resolution
// =============================================================================

// /customnode/getmappings response shape (mirrors extension-node-map.json):
//   { "<git_url>": [[node_id1, node_id2, ...], { title, ... }] }
// We invert it to a node-id → git-url Map so a list of picks can be resolved
// in O(picks). First-write-wins on duplicate node IDs across repos — rare,
// but if it happens the user can override by editing picks; either repo is a
// valid install target.
export async function loadMappings() {
    const r = await fetch(ENDPOINT_GETMAPPINGS);
    if (!r.ok) throw new Error(`getmappings ${r.status}`);
    const data = await r.json();
    const urlByNodeId = new Map();
    for (const [gitUrl, val] of Object.entries(data || {})) {
        if (!Array.isArray(val) || !Array.isArray(val[0])) continue;
        for (const id of val[0]) {
            if (typeof id !== "string") continue;
            if (!urlByNodeId.has(id)) urlByNodeId.set(id, gitUrl);
        }
    }
    return { urlByNodeId };
}

// Bucket the picks by what we need to do with them. The `byUrl` Map dedupes
// repos when several picks live in the same pack — we install once per repo,
// not once per node ID.
//
// `unresolved` collects picks the mapping doesn't know about. Common causes:
//   • a private/local node pack (never published to ComfyUI-Manager's index)
//   • a pack so new it isn't in the cached extension-node-map.json yet
//   • a renamed node that the upstream mapping hasn't caught up with
// We surface those by ID so the user can act manually instead of silently
// dropping them.
export function resolvePicksToInstall(picks, urlByNodeId) {
    const registry = (typeof LiteGraph !== "undefined" && LiteGraph.registered_node_types) || {};
    const alreadyInstalled = [];
    const unresolved = [];
    const byUrl = new Map();
    for (const id of picks) {
        if (registry[id]) {
            alreadyInstalled.push(id);
            continue;
        }
        const url = urlByNodeId.get(id);
        if (!url) {
            unresolved.push(id);
            continue;
        }
        if (!byUrl.has(url)) byUrl.set(url, []);
        byUrl.get(url).push(id);
    }
    return { alreadyInstalled, willInstall: { byUrl }, unresolved };
}

// =============================================================================
// UI-free discovery helper
// =============================================================================

// Wraps the detect → loadMappings → resolvePicksToInstall sequence into a
// single call so non-modal callers (e.g. the "drop placeholders onto canvas"
// button) can run discovery without paying for the Install-Missing modal.
//
// Discriminated result. `ok: true` means we have a usable bucket breakdown;
// `ok: false` means we hit a precondition failure that should surface as a
// toast or an in-place message rather than as an empty bucket list.
//
//   { ok: true,  result: { alreadyInstalled, willInstall: { byUrl }, unresolved } }
//   { ok: false, reason: "manager-unreachable" }
//   { ok: false, reason: "mapping-load-failed", error: Error }
export async function discoverMissingPacks(picks) {
    const managerOk = await detectManager();
    if (!managerOk) return { ok: false, reason: "manager-unreachable" };
    let mappings;
    try {
        mappings = await loadMappings();
    } catch (e) {
        return { ok: false, reason: "mapping-load-failed", error: e };
    }
    return { ok: true, result: resolvePicksToInstall(picks, mappings.urlByNodeId) };
}

// =============================================================================
// Install queue dispatch
// =============================================================================

// One git URL → one queue entry. Body is plain text (per the route's
// `await request.text()` on the server); fetch defaults to a text/plain
// content-type for a string body, which Manager accepts.
//
// 403 is the structured "your security level forbids this" response — we map
// it to user-facing language because raw HTTP codes don't help the user act.
// Other non-2xx are surfaced with their status so the result modal can still
// show what failed.
export async function queueInstallGitUrl(gitUrl) {
    try {
        const r = await fetch(ENDPOINT_INSTALL_GIT_URL, {
            method: "POST",
            headers: { "Content-Type": "text/plain" },
            body: gitUrl,
        });
        if (r.ok) return { ok: true };
        const message = r.status === 403
            ? "Manager rejected the install — your security level forbids git-URL installs. Install via Manager UI, or relax the security_level setting."
            : `Install request failed (HTTP ${r.status}).`;
        return { ok: false, status: r.status, message };
    } catch (e) {
        return { ok: false, status: 0, message: `Network error: ${e.message}` };
    }
}

// Start the worker thread that drains the queue. 200 = started, 201 = already
// processing (e.g. the user kicked off another install moments earlier).
// Both are success from our perspective — the queue is being worked.
export async function startQueue() {
    try {
        const r = await fetch(ENDPOINT_QUEUE_START, { method: "POST" });
        return r.ok || r.status === 201;
    } catch (e) {
        console.warn("[Koolook] startQueue failed:", e);
        return false;
    }
}

export async function getQueueStatus() {
    const r = await fetch(ENDPOINT_QUEUE_STATUS);
    if (!r.ok) throw new Error(`queue/status ${r.status}`);
    return await r.json();
}

// Poll status until `is_processing` flips false. `signal.aborted = true` lets
// the caller bail without canceling installs already in flight (we can't —
// Manager has no per-task cancel) — it just disengages the UI loop.
//
// Returns the final status snapshot, or null if aborted / a poll fetch threw.
export async function pollUntilDone({ intervalMs = POLL_INTERVAL_MS, onTick, signal } = {}) {
    while (true) {
        if (signal && signal.aborted) return null;
        let status;
        try {
            status = await getQueueStatus();
        } catch (e) {
            console.warn("[Koolook] queue status poll failed:", e);
            return null;
        }
        if (typeof onTick === "function") onTick(status);
        if (!status.is_processing) return status;
        await new Promise(resolve => setTimeout(resolve, intervalMs));
    }
}

// =============================================================================
// Restart
// =============================================================================

// Manager's reboot endpoint signals the launcher (via a sentinel file) that
// ComfyUI should restart. The HTTP request itself usually fails partway —
// the server is going down — so we treat a network error here as "request
// likely accepted" rather than a hard failure.
export async function reboot() {
    try {
        const r = await fetch(ENDPOINT_REBOOT, { method: "POST" });
        return r.ok;
    } catch (e) {
        return true;
    }
}
