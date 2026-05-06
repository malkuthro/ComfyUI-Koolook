// SPDX-FileCopyrightText: 2025-2026 Kforge Labs <https://github.com/malkuthro/ComfyUI-Koolook>
// SPDX-License-Identifier: GPL-3.0-only

// =============================================================================
// Canvas IO — anything that touches `app.graph` or `app.canvas` directly.
// Selection serialization prunes cross-boundary links so a saved selection
// reloads cleanly. `loadWorkflowOntoCanvas` snapshots the prior canvas and
// restores it if the load throws partway, so a corrupt workflow can't strand
// the user with a half-loaded graph.
// =============================================================================
import { app } from "../../../../scripts/app.js";

import { toast } from "./constants.js";
import { getWorkflowGraph } from "./workflows_store.js";
import { showConfirmModal } from "./modals.js";

export function serializeFullCanvas() {
    try {
        return app.graph.serialize();
    } catch (e) {
        console.warn("[Koolook] graph.serialize() failed:", e);
        return null;
    }
}

// True when the graph has at least one node whose `type` references a defined
// subgraph in the same graph. Used to bias the "Save as module" default —
// a canvas containing a subgraph wrapper is almost certainly a reusable kit.
// We require both the definition AND a referencing node so a stale definition
// (subgraph deleted, definition not yet GC'd) doesn't false-positive.
export function graphHasSubgraphInstance(graph) {
    if (!graph) return false;
    const defs = graph.definitions && graph.definitions.subgraphs;
    if (!Array.isArray(defs) || defs.length === 0) return false;
    const ids = new Set(defs.map(sg => sg && sg.id).filter(Boolean));
    if (ids.size === 0) return false;
    const nodes = Array.isArray(graph.nodes) ? graph.nodes : [];
    return nodes.some(n => n && ids.has(n.type));
}

// Subgraph UUIDs follow LiteGraph's serialized format. Used by the insert
// pre-flight to surface a more actionable error when a UUID-typed wrapper
// can't be created (the cause is almost always a missing subgraph
// definition, not a missing custom-node pack).
const SUBGRAPH_UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

// Walk a saved-graph's `definitions.subgraphs` to collect every UUID that
// `rootIds` transitively reference (a saved subgraph's interior nodes can
// themselves be wrappers around other subgraphs). The visited set is
// returned so callers can copy exactly the needed definitions and nothing
// more — saving every definition the live canvas knows about would bloat
// every selection-save with unrelated subgraphs.
function collectReferencedSubgraphIds(rootIds, allDefs) {
    const visited = new Set();
    const work = [...rootIds];
    while (work.length > 0) {
        const id = work.pop();
        if (!id || visited.has(id)) continue;
        visited.add(id);
        const def = allDefs.find(d => d && d.id === id);
        if (!def) continue;
        const interiorNodes = Array.isArray(def.nodes) ? def.nodes : [];
        for (const n of interiorNodes) {
            if (n && typeof n.type === "string" && allDefs.some(d => d && d.id === n.type)) {
                work.push(n.type);
            }
        }
    }
    return visited;
}

// Best-effort registration of incoming subgraph definitions before insert
// pre-flight. Without this, a saved selection containing a subgraph wrapper
// can be inserted only into a Comfy session that already has the matching
// definition cached — across sessions the pre-flight rejects the wrapper
// on `missing node type: <UUID>`.
//
// Strategy: try the public service path, then fall back to direct mutation
// of `app.graph.definitions.subgraphs`. The fallback ensures the definition
// is at least present in the graph data (so a save/reload round-trip
// consolidates it); LiteGraph's per-UUID type registration may still be
// missing in that case, which the improved pre-flight error makes
// actionable for the user.
function mergeIncomingSubgraphDefinitions(clone) {
    const incoming = clone && clone.definitions && Array.isArray(clone.definitions.subgraphs)
        ? clone.definitions.subgraphs.filter(sg => sg && sg.id)
        : [];
    if (incoming.length === 0) return;

    // Preferred: Comfy's subgraph service handles registration end-to-end
    // (definition push + LiteGraph node-type registration). Exposed on the
    // app singleton in current frontends; older versions may not expose it.
    try {
        const svc = app.subgraphService;
        if (svc && typeof svc.loadSubgraphs === "function") {
            svc.loadSubgraphs({ definitions: { subgraphs: incoming } });
            return;
        }
    } catch (e) {
        console.warn("[Koolook] subgraphService.loadSubgraphs failed; falling back to direct merge:", e);
    }

    // Fallback: append to app.graph.definitions.subgraphs idempotently.
    try {
        if (!app.graph.definitions) app.graph.definitions = {};
        if (!Array.isArray(app.graph.definitions.subgraphs)) app.graph.definitions.subgraphs = [];
        const existing = new Set(app.graph.definitions.subgraphs.map(d => d && d.id).filter(Boolean));
        for (const sg of incoming) {
            if (!existing.has(sg.id)) {
                app.graph.definitions.subgraphs.push(JSON.parse(JSON.stringify(sg)));
                existing.add(sg.id);
            }
        }
    } catch (e) {
        console.warn("[Koolook] direct subgraph definition merge failed:", e);
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

// Returns a discriminated result so callers can distinguish "user has not
// selected anything yet" from "user has a selection but every node in it is
// gone" — those produce different toasts in the UI.
//
// Shapes:
//   { kind: "empty" }   — `selectedIds.size === 0` OR `serializeFullCanvas`
//                          failed (canvas not ready / threw); caller should
//                          prompt the user to select something.
//   { kind: "stale" }   — `selectedIds.size > 0` but none of the IDs match
//                          a live node on the graph (selection points at
//                          deleted nodes — common after undo/redo). Caller
//                          should prompt the user to re-select live nodes.
//   { kind: "ok", graph } — success.
export function serializeSelection() {
    const selectedIds = getSelectedNodeIds();
    if (selectedIds.size === 0) return { kind: "empty" };
    const full = serializeFullCanvas();
    if (!full) return { kind: "empty" };

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

    if (nodes.length === 0) return { kind: "stale" };

    // Carry along any subgraph definitions this selection transitively
    // references. Without this, a saved selection containing a subgraph
    // wrapper would point at a UUID type whose definition only lives in
    // the source canvas's session — re-inserting it later (or in a fresh
    // Comfy session) would render against a stale cached definition or
    // fail pre-flight outright.
    const definitions = collectDefinitionsForNodes(nodes, full);

    return {
        kind: "ok",
        graph: {
            last_node_id: full.last_node_id,
            last_link_id: full.last_link_id,
            nodes,
            links: internalLinks,
            groups: [],
            ...(definitions ? { definitions } : {}),
            config: full.config || {},
            extra: full.extra || {},
            version: full.version || 0.4,
        },
    };
}

// Returns `{ subgraphs: [...] }` containing every subgraph definition the
// given nodes transitively reference, or `null` if none. Definitions are
// deep-cloned so callers can mutate the saved graph without disturbing
// `app.graph`.
function collectDefinitionsForNodes(nodes, fullGraph) {
    const liveDefs = (fullGraph && fullGraph.definitions && Array.isArray(fullGraph.definitions.subgraphs))
        ? fullGraph.definitions.subgraphs
        : [];
    if (liveDefs.length === 0) return null;
    const liveIds = new Set(liveDefs.map(d => d && d.id).filter(Boolean));
    const rootIds = nodes
        .map(n => n && n.type)
        .filter(t => typeof t === "string" && liveIds.has(t));
    if (rootIds.length === 0) return null;
    const required = collectReferencedSubgraphIds(rootIds, liveDefs);
    const subgraphs = liveDefs
        .filter(d => d && required.has(d.id))
        .map(d => JSON.parse(JSON.stringify(d)));
    if (subgraphs.length === 0) return null;
    return { subgraphs };
}

export function canvasIsNonEmpty() {
    try {
        return app.graph && app.graph._nodes && app.graph._nodes.length > 0;
    } catch (e) {
        return false;
    }
}

export async function loadWorkflowOntoCanvas(dirPath, wfName) {
    const graph = getWorkflowGraph(dirPath, wfName);
    if (!graph) {
        toast(`Workflow not found: ${wfName}`);
        return;
    }
    const apply = async () => {
        // Snapshot the current canvas before loading. If loadGraphData throws
        // partway (missing node type, malformed payload), we restore the
        // user's prior canvas instead of leaving them with neither workflow.
        let previousGraph = null;
        try {
            if (canvasIsNonEmpty()) previousGraph = app.graph.serialize();
        } catch (e) {
            previousGraph = null;
        }
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
            if (previousGraph) {
                try {
                    await app.loadGraphData(previousGraph);
                    toast(`Failed to load "${wfName}". Canvas restored. See console.`);
                    return;
                } catch (restoreErr) {
                    console.error("[Koolook] canvas restore failed:", restoreErr);
                    toast(`Failed to load "${wfName}". Canvas may be partial. See console.`);
                    return;
                }
            }
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
// Placeholder workflow for "Install missing via Manager UI" handoff
// =============================================================================

// Build a fresh workflow that contains one node per missing pack, using a
// representative node ID from each pack as the node `type`. The pack isn't
// installed, so ComfyUI renders these as red "missing node type" placeholders
// — which is exactly what triggers Manager's built-in "Install Missing Custom
// Nodes" detection (Manager scans the active graph for unknown types and
// looks them up in its mapping). The user then drives Manager's own UI to
// install, which goes through Manager's UI-driven install path rather than
// our programmatic /customnode/install/git_url call — so it works at
// `security_level=normal` without us monkey-patching anyone's gate.
//
// Layout: 4-column grid, 220×100 px cells, starting at (50, 50). Each pack
// becomes one node, captioned by its representative node ID. Loaded into a
// fresh tab named "missing-nodes-for-picks" so the user can close it without
// touching their real workflow.
//
// `packsByUrl` is the same Map shape `resolvePicksToInstall().willInstall`
// returns: gitUrl → array of node IDs from that pack that are missing. We
// take the first node ID per pack as the placeholder's type. One per pack
// is enough — Manager dedupes by pack, so dropping 27 nodes from one pack
// just slows the canvas without telling Manager anything new.
//
// Confirms canvas replacement before loading (preserves the user's current
// graph as a rollback target if `loadGraphData` throws). Returns a
// discriminated result so the caller can decide how to surface failures.
export async function dropPlaceholdersForPacks(packsByUrl) {
    if (!packsByUrl || packsByUrl.size === 0) {
        return { ok: false, reason: "no-packs" };
    }

    const cellW = 220, cellH = 100, cols = 4, marginX = 50, marginY = 50;
    const nodes = [];
    let nextId = 1;
    let idx = 0;
    for (const nodeIds of packsByUrl.values()) {
        if (!Array.isArray(nodeIds) || nodeIds.length === 0) continue;
        const repName = nodeIds[0];
        const col = idx % cols;
        const row = Math.floor(idx / cols);
        nodes.push({
            id: nextId++,
            type: repName,
            pos: [marginX + col * cellW, marginY + row * cellH],
            size: [cellW - 20, cellH - 20],
            flags: {},
            order: idx,
            mode: 0,
            inputs: [],
            outputs: [],
            properties: { "Node name for S&R": repName },
            widgets_values: [],
        });
        idx++;
    }

    if (nodes.length === 0) return { ok: false, reason: "no-nodes" };

    const graph = {
        last_node_id: nextId - 1,
        last_link_id: 0,
        nodes,
        links: [],
        groups: [],
        config: {},
        extra: {},
        version: 0.4,
    };

    const apply = async () => {
        let previousGraph = null;
        try {
            if (canvasIsNonEmpty()) previousGraph = app.graph.serialize();
        } catch (e) {
            previousGraph = null;
        }
        try {
            await app.loadGraphData(graph, true, true, "missing-nodes-for-picks", {});
            toast(
                `Dropped ${nodes.length} placeholder${nodes.length === 1 ? "" : "s"}. ` +
                `Open Manager → "Install Missing Custom Nodes" to install.`
            );
            return { ok: true, count: nodes.length };
        } catch (e) {
            console.error("[Koolook] dropPlaceholders loadGraphData failed:", e);
            if (previousGraph) {
                try {
                    await app.loadGraphData(previousGraph);
                    toast("Failed to drop placeholders. Canvas restored. See console.");
                    return { ok: false, reason: "load-failed", error: e };
                } catch (restoreErr) {
                    console.error("[Koolook] canvas restore failed:", restoreErr);
                }
            }
            toast("Failed to drop placeholders. See console.");
            return { ok: false, reason: "load-failed", error: e };
        }
    };

    if (canvasIsNonEmpty()) {
        return new Promise(resolve => {
            showConfirmModal({
                title: "Replace current workflow?",
                message:
                    `This will replace your canvas with ${nodes.length} placeholder ` +
                    `node${nodes.length === 1 ? "" : "s"} (one per missing pack). ` +
                    `Manager's "Install Missing Custom Nodes" will then offer to install ` +
                    `them. Save your current work first if you have anything unsaved.`,
                confirmLabel: "Replace canvas",
                danger: true,
                onConfirm: async () => resolve(await apply()),
            });
        });
    }
    return await apply();
}

// =============================================================================
// Node insertion
// =============================================================================
function placeAtCanvasCenter(node) {
    try {
        const canvas = app.canvas;
        const ds = canvas.ds;
        // `canvas.canvas.width` / `.height` are the BACKING-STORE pixel buffer
        // dimensions, which on a HiDPI display (Retina/Mac, Windows scaling)
        // are devicePixelRatio× the visible CSS pixels. LiteGraph's pan/zoom
        // math is in CSS pixels, so dividing the buffer width by 2 puts the
        // node at devicePixelRatio× the visible-center distance from the
        // origin — i.e. way off the right side of the viewport on a 2× Mac
        // display. Use clientWidth/clientHeight (CSS pixels) instead;
        // fall back to the buffer dims if the canvas isn't laid out yet.
        const cssWidth = canvas.canvas.clientWidth || canvas.canvas.width;
        const cssHeight = canvas.canvas.clientHeight || canvas.canvas.height;
        const cx = -ds.offset[0] + cssWidth / (2 * ds.scale);
        const cy = -ds.offset[1] + cssHeight / (2 * ds.scale);
        node.pos = [cx - node.size[0] / 2, cy - node.size[1] / 2];
    } catch (e) {
        // Default position is fine if the canvas isn't ready yet.
    }
}

export function insertNode(typeName) {
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

// =============================================================================
// Workflow insertion (non-destructive — ADDS nodes from a saved workflow into
// the live canvas instead of replacing it). Sibling to `loadWorkflowOntoCanvas`
// above; the existing function rebuilds the entire graph via
// `app.loadGraphData`, which is the right call for "load this workflow as my
// session", but the wrong one for "drop this preset cluster next to my
// existing nodes". This primitive is the basis for the Modules feature: a
// saved selection (typically tagged `module`) that the user wants to splice
// into whatever they're already building.
//
// Mechanics — kept tight because the failure modes are subtle:
//
//   1. **Pre-flight missing types.** A saved cluster can reference custom
//      nodes the user hasn't installed. `app.loadGraphData` masks this by
//      rendering red error nodes; for an insert that would silently splice
//      broken stubs into the user's working graph. We check
//      `LiteGraph.registered_node_types` up-front and abort with a toast if
//      anything is missing — partial inserts are worse than no insert.
//
//   2. **Let the live graph allocate ids.** The saved graph's node ids and
//      link ids almost certainly collide with what's already on canvas (both
//      counters reset per saved workflow). Easiest correct fix: deep-clone
//      the saved graph, then for each cloned node call `node.configure(...)`
//      (which restores widget values + properties from the JSON) but force
//      `node.id = -1` before `app.graph.add(node)` so LiteGraph hands out a
//      fresh id from `last_node_id + 1`. Build an `oldId → newId` map as we
//      go so links can be reconstructed afterwards.
//
//   3. **Recreate links via `node.connect`.** Walking `clone.links` and
//      calling `originNode.connect(originSlot, targetNode, targetSlot)`
//      sidesteps link-id remapping entirely — `connect` allocates a fresh
//      link id from `app.graph.last_link_id + 1`. Cross-boundary references
//      (links pointing at nodes that weren't part of the saved selection)
//      are already nulled by `serializeSelection`, so the typical case is a
//      clean closed sub-graph.
//
//   4. **Place the cluster at the viewport center.** Compute the bbox of
//      the cloned nodes (in their saved coords), then translate every node
//      by `(viewportCenter - bboxCenter)`. Reuses the same HiDPI-correct
//      math as `placeAtCanvasCenter` (CSS pixels via `clientWidth`, not the
//      backing-store buffer). If the canvas isn't laid out yet, we fall
//      through and place at original coords — same defensive posture as
//      `placeAtCanvasCenter`.
//
//   5. **Select the inserted cluster** so the user can immediately drag it
//      somewhere else, delete it, or hit Ctrl+C to duplicate. Selection API
//      varies across LiteGraph forks (`selectNodes` vs. `selectNode(_, true)`
//      for additive); we try both and swallow failures because selection is
//      a nice-to-have, not load-critical.
//
// Returns `{ ok: true, count }` on success, `{ ok: false, reason }` on any
// abort path so callers can branch (the right-click menu just toasts; a
// future drag-onto-canvas path might want to surface a different message).
// =============================================================================

function placeBboxAtCanvasCenter(nodes) {
    if (!nodes || nodes.length === 0) return;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const n of nodes) {
        const w = (n.size && n.size[0]) || 200;
        const h = (n.size && n.size[1]) || 100;
        const x = (n.pos && n.pos[0]) || 0;
        const y = (n.pos && n.pos[1]) || 0;
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x + w > maxX) maxX = x + w;
        if (y + h > maxY) maxY = y + h;
    }
    if (!isFinite(minX) || !isFinite(minY)) return;
    const bboxCx = (minX + maxX) / 2;
    const bboxCy = (minY + maxY) / 2;

    let dx = 0, dy = 0;
    try {
        const canvas = app.canvas;
        const ds = canvas.ds;
        const cssWidth = canvas.canvas.clientWidth || canvas.canvas.width;
        const cssHeight = canvas.canvas.clientHeight || canvas.canvas.height;
        const cx = -ds.offset[0] + cssWidth / (2 * ds.scale);
        const cy = -ds.offset[1] + cssHeight / (2 * ds.scale);
        dx = cx - bboxCx;
        dy = cy - bboxCy;
    } catch (e) {
        // Canvas not ready — leave nodes at their saved coords. This matches
        // `placeAtCanvasCenter`'s posture: don't throw on a layout race.
    }

    if (dx === 0 && dy === 0) return;
    for (const n of nodes) {
        if (!n.pos) n.pos = [0, 0];
        n.pos[0] += dx;
        n.pos[1] += dy;
    }
}

export async function insertWorkflowOntoCanvas(dirPath, wfName) {
    const sourceGraph = getWorkflowGraph(dirPath, wfName);
    if (!sourceGraph) {
        toast(`Workflow not found: ${wfName}`);
        return { ok: false, reason: "not-found" };
    }
    if (typeof LiteGraph === "undefined") {
        toast(`Cannot insert "${wfName}" — LiteGraph not available.`);
        return { ok: false, reason: "no-litegraph" };
    }
    // Deep clone — `node.configure` and `placeBboxAtCanvasCenter` both
    // mutate. We never want to touch the in-memory store copy.
    const clone = JSON.parse(JSON.stringify(sourceGraph));
    const nodesRaw = Array.isArray(clone.nodes) ? clone.nodes : [];
    if (nodesRaw.length === 0) {
        toast(`"${wfName}" has no nodes to insert.`);
        return { ok: false, reason: "empty" };
    }

    // Register any subgraph definitions the saved file carries before
    // pre-flight runs — otherwise wrappers around subgraphs that aren't
    // already cached in this Comfy session would fail with `missing node
    // type: <UUID>`.
    mergeIncomingSubgraphDefinitions(clone);

    // Pre-flight: every node type must be registered. Aborting here is the
    // entire reason this is a separate primitive from `loadWorkflowOntoCanvas`
    // — a partial insert with stub nodes silently corrupts the user's graph.
    const registered = LiteGraph.registered_node_types || {};
    const missingTypes = [];
    const seenMissing = new Set();
    for (const n of nodesRaw) {
        const t = n && n.type;
        if (!t) continue;
        if (!registered[t] && !seenMissing.has(t)) {
            seenMissing.add(t);
            missingTypes.push(t);
        }
    }
    if (missingTypes.length > 0) {
        const subgraphMisses = missingTypes.filter(t => SUBGRAPH_UUID_RE.test(t));
        const packMisses = missingTypes.filter(t => !SUBGRAPH_UUID_RE.test(t));
        let msg;
        if (subgraphMisses.length > 0 && packMisses.length === 0) {
            // All misses are UUIDs — almost certainly an unregistered subgraph
            // definition rather than a missing custom-node pack. Tell the user
            // what to actually do about it.
            msg = `Cannot insert "${wfName}" — subgraph definition not registered. ` +
                  `Native-Load the workflow once in this session to register it, then retry Insert.`;
        } else {
            const sample = packMisses.slice(0, 3).join(", ");
            const more = packMisses.length > 3 ? ` (+${packMisses.length - 3} more)` : "";
            msg = `Cannot insert "${wfName}" — missing node type${packMisses.length === 1 ? "" : "s"}: ` +
                  `${sample}${more}. Install the required pack(s) first.`;
        }
        toast(msg, 4500);
        console.warn(`[Koolook] insert "${wfName}" aborted; missing types:`, missingTypes);
        return { ok: false, reason: "missing-types", missingTypes };
    }

    // Translate the cluster to the viewport center BEFORE creating live
    // nodes. `placeBboxAtCanvasCenter` writes into `n.pos` on the cloned
    // entries, which `node.configure(...)` then reads.
    placeBboxAtCanvasCenter(nodesRaw);

    // Insert nodes; let the live graph allocate ids so nothing collides
    // with `app.graph.last_node_id`.
    const oldToNew = new Map();
    const inserted = [];
    for (const oldNode of nodesRaw) {
        const node = LiteGraph.createNode(oldNode.type);
        if (!node) {
            // Pre-flight should have caught this; log if it happens anyway
            // (a registered type whose factory returned null is a LiteGraph
            // bug or a misregistered pack — we'd rather skip than crash the
            // whole insert at this point).
            console.warn(`[Koolook] LiteGraph.createNode("${oldNode.type}") returned null after pre-flight passed`);
            continue;
        }
        try {
            node.configure(oldNode);
        } catch (e) {
            console.warn(`[Koolook] node.configure failed for "${oldNode.type}":`, e);
        }
        // configure() restored `node.id` from the saved data — that's the
        // OLD id. Force re-allocation so we don't collide with the live
        // graph (or with another inserted node from the same cluster).
        const oldId = oldNode.id;
        node.id = -1;
        app.graph.add(node);
        if (oldId != null) oldToNew.set(oldId, node.id);
        inserted.push(node);
    }

    // Recreate internal links. `serializeSelection` already nulled out
    // cross-boundary references, but defensively skip any link whose
    // endpoint we don't have in the remap (covers hand-edited /userdata
    // files and old saves from before that pruning landed).
    const linksRaw = Array.isArray(clone.links) ? clone.links : [];
    let connectedLinks = 0;
    for (const link of linksRaw) {
        if (!Array.isArray(link) || link.length < 5) continue;
        const oldOrigin = link[1];
        const originSlot = link[2];
        const oldTarget = link[3];
        const targetSlot = link[4];
        const newOrigin = oldToNew.get(oldOrigin);
        const newTarget = oldToNew.get(oldTarget);
        if (newOrigin == null || newTarget == null) continue;
        const originNode = app.graph.getNodeById(newOrigin);
        const targetNode = app.graph.getNodeById(newTarget);
        if (!originNode || !targetNode) continue;
        try {
            const result = originNode.connect(originSlot, targetNode, targetSlot);
            if (result) connectedLinks += 1;
        } catch (e) {
            console.warn(
                `[Koolook] link reconnect failed: ${originNode.type}[${originSlot}] → ` +
                `${targetNode.type}[${targetSlot}]:`, e
            );
        }
    }

    // Selection — best-effort, never block on it.
    try {
        if (typeof app.canvas.selectNodes === "function") {
            app.canvas.selectNodes(inserted);
        } else if (typeof app.canvas.selectNode === "function") {
            // Older LiteGraph: additive single-select per node.
            for (let i = 0; i < inserted.length; i += 1) {
                app.canvas.selectNode(inserted[i], i > 0);
            }
        }
    } catch (e) {
        // Don't surface — the nodes are on canvas, selection is just polish.
    }

    app.canvas.setDirty(true, true);
    const noun = inserted.length === 1 ? "node" : "nodes";
    const linkSuffix = connectedLinks > 0
        ? ` (${connectedLinks} link${connectedLinks === 1 ? "" : "s"})`
        : "";
    toast(`Inserted "${wfName}" — ${inserted.length} ${noun}${linkSuffix}.`);
    return { ok: true, count: inserted.length, links: connectedLinks };
}

export function getSelectedNodeTypes() {
    try {
        const sel = (app.canvas && app.canvas.selected_nodes) || {};
        return Object.values(sel)
            .filter(n => n && n.type)
            .map(n => n.type);
    } catch (e) {
        return [];
    }
}
