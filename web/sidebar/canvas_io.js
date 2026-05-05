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

    return {
        kind: "ok",
        graph: {
            last_node_id: full.last_node_id,
            last_link_id: full.last_link_id,
            nodes,
            links: internalLinks,
            groups: [],
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
