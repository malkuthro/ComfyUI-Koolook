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
import { cloneWorkflowForTemporaryLoad } from "./workflow_payload.js";
import { groupsForSelectedNodes, translateGroups } from "./canvas_groups.js";

export function serializeFullCanvas() {
    try {
        return app.graph.serialize();
    } catch (e) {
        console.warn("[Koolook] graph.serialize() failed:", e);
        return null;
    }
}

const SUBGRAPH_UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function collectReferencedSubgraphIds(rootIds, allDefs) {
    const visited = new Set();
    const defsById = new Map(
        allDefs
            .filter(d => d && d.id)
            .map(d => [d.id, d])
    );
    const work = [...rootIds];
    while (work.length > 0) {
        const id = work.pop();
        if (!id || visited.has(id)) continue;
        visited.add(id);
        const def = defsById.get(id);
        if (!def) continue;
        const interiorNodes = Array.isArray(def.nodes) ? def.nodes : [];
        for (const n of interiorNodes) {
            if (n && typeof n.type === "string" && defsById.has(n.type)) {
                work.push(n.type);
            } else if (n && typeof n.type === "string" && SUBGRAPH_UUID_RE.test(n.type)) {
                console.warn("[Koolook] subgraph save: unresolved transitive ref", n.type);
            }
        }
    }
    return visited;
}

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
    return { subgraphs };
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

export function getSelectedNodeCount() {
    return getSelectedNodeIds().size;
}

export function getCanvasNodeCount() {
    try {
        return (app.graph && app.graph._nodes && app.graph._nodes.length) || 0;
    } catch (e) {
        return 0;
    }
}

function idKey(id) {
    return id == null ? null : String(id);
}

function graphLinksAsList(links) {
    if (Array.isArray(links)) return links;
    if (links && typeof links === "object") return Object.values(links);
    return [];
}

function getLinkId(link) {
    if (Array.isArray(link)) return link[0];
    if (link && typeof link === "object") return link.id;
    return null;
}

function getLinkOriginId(link) {
    if (Array.isArray(link)) return link[1];
    if (link && typeof link === "object") return link.origin_id ?? link.originId;
    return null;
}

function getLinkOriginSlot(link) {
    if (Array.isArray(link)) return link[2];
    if (link && typeof link === "object") return link.origin_slot ?? link.originSlot;
    return null;
}

function getLinkTargetId(link) {
    if (Array.isArray(link)) return link[3];
    if (link && typeof link === "object") return link.target_id ?? link.targetId;
    return null;
}

function getLinkTargetSlot(link) {
    if (Array.isArray(link)) return link[4];
    if (link && typeof link === "object") return link.target_slot ?? link.targetSlot;
    return null;
}

function cloneNodeWithoutLinkState(node) {
    const clone = JSON.parse(JSON.stringify(node));
    if (Array.isArray(clone.inputs)) {
        for (const inp of clone.inputs) {
            if (inp) inp.link = null;
        }
    }
    if (Array.isArray(clone.outputs)) {
        for (const out of clone.outputs) {
            if (out) out.links = null;
        }
    }
    return clone;
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
    const selectedKeys = new Set([...selectedIds].map(idKey));
    const full = serializeFullCanvas();
    if (!full) return { kind: "empty" };

    const internalLinks = graphLinksAsList(full.links).filter(link =>
        selectedKeys.has(idKey(getLinkOriginId(link))) &&
        selectedKeys.has(idKey(getLinkTargetId(link)))
    );
    const internalLinkIds = new Set(internalLinks.map(l => idKey(getLinkId(l))));

    const nodes = (full.nodes || [])
        .filter(n => selectedKeys.has(idKey(n.id)))
        .map(n => {
            const clone = JSON.parse(JSON.stringify(n));
            if (Array.isArray(clone.inputs)) {
                for (const inp of clone.inputs) {
                    if (inp && inp.link != null && !internalLinkIds.has(idKey(inp.link))) {
                        inp.link = null;
                    }
                }
            }
            if (Array.isArray(clone.outputs)) {
                for (const out of clone.outputs) {
                    if (out && Array.isArray(out.links)) {
                        out.links = out.links.filter(l => internalLinkIds.has(idKey(l)));
                        if (out.links.length === 0) out.links = null;
                    }
                }
            }
            return clone;
        });

    if (nodes.length === 0) return { kind: "stale" };
    const definitions = collectDefinitionsForNodes(nodes, full);

    return {
        kind: "ok",
        graph: {
            last_node_id: full.last_node_id,
            last_link_id: full.last_link_id,
            nodes,
            links: internalLinks,
            groups: groupsForSelectedNodes(full, selectedKeys),
            ...(definitions ? { definitions } : {}),
            config: full.config || {},
            extra: full.extra || {},
            version: full.version || 0.4,
        },
    };
}

export function canvasIsNonEmpty() {
    try {
        return app.graph && app.graph._nodes && app.graph._nodes.length > 0;
    } catch (e) {
        return false;
    }
}

export async function loadWorkflowOntoCanvas(dirPath, wfName) {
    const sourceGraph = getWorkflowGraph(dirPath, wfName);
    if (!sourceGraph) {
        toast(`Workflow not found: ${wfName}`);
        return;
    }
    const graph = cloneWorkflowForTemporaryLoad(sourceGraph, [...dirPath, wfName].join("\u0000"));
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
            // Keep sidebar workflow loads as Comfy-owned temporary drafts.
            // Passing `wfName` here, or renaming the active Comfy workflow to
            // `workflows/<wfName>.json`, binds the tab to Comfy's native
            // workflow-file namespace. If that file already exists, Comfy's
            // autosave/draft writer can hit a 409 Conflict even though the
            // Koolook sidebar workflow with the same display name is valid.
            await app.loadGraphData(graph, true, true);

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
                // Without this, Cancel / Escape / click-outside left the
                // Promise pending forever — caller's await never returned.
                onCancel: () => resolve({ ok: false, reason: "cancelled" }),
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
    if (!nodes || nodes.length === 0) return { dx: 0, dy: 0 };
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
    if (!isFinite(minX) || !isFinite(minY)) return { dx: 0, dy: 0 };
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

    if (dx === 0 && dy === 0) return { dx, dy };
    for (const n of nodes) {
        if (!n.pos) n.pos = [0, 0];
        n.pos[0] += dx;
        n.pos[1] += dy;
    }
    return { dx, dy };
}

function addWorkflowGroupsToCanvas(groups, dx, dy) {
    const translated = translateGroups(groups, dx, dy);
    if (translated.length === 0 || !app.graph) return;
    let liveGroups = null;
    if (Array.isArray(app.graph._groups)) {
        liveGroups = app.graph._groups;
    } else {
        if (!Array.isArray(app.graph.groups)) app.graph.groups = [];
        liveGroups = app.graph.groups;
    }

    for (const groupData of translated) {
        let group = null;
        try {
            const GroupClass = LiteGraph?.LGraphGroup || globalThis.LGraphGroup;
            if (typeof GroupClass === "function") {
                group = new GroupClass(groupData.title || "");
                if (typeof group.configure === "function") {
                    group.configure(groupData);
                } else {
                    Object.assign(group, groupData);
                }
            }
        } catch (e) {
            group = null;
        }
        liveGroups.push(group || groupData);
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
        // Two distinct failure modes hide behind "missing type":
        //   • pack miss — registered_node_types has no entry for a custom-node
        //     type. Fix: install the pack(s).
        //   • subgraph miss — the type IS a subgraph UUID, but its definition
        //     hasn't been registered into registered_node_types yet (subgraphs
        //     are registered as a side effect of native Load, not at startup).
        //     Fix: right-click the source workflow → Load once in the current
        //     session, which routes through `app.loadGraphData` and registers
        //     each `definitions.subgraphs[]` entry by UUID. Subsequent Inserts
        //     then resolve.
        // Pre-fix, the mixed case (both classes present) silently fell into the
        // pack-miss branch and listed UUIDs alongside pack types under "install
        // the pack(s)" — the user installed the pack, retried, and hit an
        // unexplained second "still missing" failure for the unregistered
        // subgraph. Surface BOTH classes when both are present so one toast
        // covers both fixes in order.
        // Toast wording goes plain English on purpose — "subgraph definition
        // not registered" was technically accurate but confused the user, who
        // just wants to know "what do I do to make Insert work?" Engineer-speak
        // (subgraph / registered / definition) lives in code comments and
        // console.warn for devs; user-facing copy talks in gestures and
        // outcomes ("right-click → Load it once, then left-click Insert will
        // work"). The "Load" verb mirrors the actual menu item label in
        // workflowRowContextMenu so the user doesn't hunt for a "paste" or
        // "set up" item that doesn't exist.
        const subgraphMisses = missingTypes.filter(t => SUBGRAPH_UUID_RE.test(t));
        const packMisses = missingTypes.filter(t => !SUBGRAPH_UUID_RE.test(t));
        let message;
        if (packMisses.length === 0) {
            // Pure subgraph case — only definitions are missing. Skip the
            // "why" (subgraphs need registration after a fresh session); jump
            // straight to the gesture. If it wears off after a restart the
            // toast just re-fires and the user does the same gesture again —
            // no pre-emptive edge-case warning needed in the toast itself.
            message =
                `Can't Insert "${wfName}" yet — Right-click → Load it once, ` +
                `then left-click Insert will work.`;
        } else if (subgraphMisses.length === 0) {
            // Pure pack case — only registered-pack types are missing.
            const sample = packMisses.slice(0, 3).join(", ");
            const more = packMisses.length > 3 ? ` (+${packMisses.length - 3} more)` : "";
            message =
                `Can't Insert "${wfName}" — missing node${packMisses.length === 1 ? "" : "s"}: ` +
                `${sample}${more}. Install the pack(s) first, then retry.`;
        } else {
            // Mixed case — both classes missing. Numbered two-step instruction
            // so the user knows the order; mentioning both upfront preempts a
            // second "still missing" toast after the user installs the pack
            // and retries Insert without the Load step.
            const sample = packMisses.slice(0, 3).join(", ");
            const more = packMisses.length > 3 ? ` (+${packMisses.length - 3} more)` : "";
            message =
                `Can't Insert "${wfName}" yet — ` +
                `(1) install pack(s) for ${sample}${more}, ` +
                `(2) Right-click → Load it once. Then left-click Insert will work.`;
        }
        toast(message, 6500);
        console.warn(
            `[Koolook] insert "${wfName}" aborted;`,
            { packMisses, subgraphMisses }
        );
        return { ok: false, reason: "missing-types", missingTypes };
    }

    // Translate the cluster to the viewport center BEFORE creating live
    // nodes. `placeBboxAtCanvasCenter` writes into `n.pos` on the cloned
    // entries, which `node.configure(...)` then reads.
    const placement = placeBboxAtCanvasCenter(nodesRaw);

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
        const nodeConfig = cloneNodeWithoutLinkState(oldNode);
        try {
            node.configure(nodeConfig);
        } catch (e) {
            console.warn(`[Koolook] node.configure failed for "${oldNode.type}":`, e);
        }
        // configure() restored `node.id` from the saved data — that's the
        // OLD id. Force re-allocation so we don't collide with the live
        // graph (or with another inserted node from the same cluster).
        const oldId = oldNode.id;
        node.id = -1;
        app.graph.add(node);
        if (oldId != null) oldToNew.set(idKey(oldId), node.id);
        inserted.push(node);
    }

    // Recreate internal links. `serializeSelection` already nulled out
    // cross-boundary references, but defensively skip any link whose
    // endpoint we don't have in the remap (covers hand-edited /userdata
    // files and old saves from before that pruning landed).
    const linksRaw = graphLinksAsList(clone.links);
    let connectedLinks = 0;
    for (const link of linksRaw) {
        const oldOrigin = getLinkOriginId(link);
        const originSlot = getLinkOriginSlot(link);
        const oldTarget = getLinkTargetId(link);
        const targetSlot = getLinkTargetSlot(link);
        if (oldOrigin == null || oldTarget == null || originSlot == null || targetSlot == null) continue;
        const newOrigin = oldToNew.get(idKey(oldOrigin));
        const newTarget = oldToNew.get(idKey(oldTarget));
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

    addWorkflowGroupsToCanvas(clone.groups, placement.dx, placement.dy);

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
