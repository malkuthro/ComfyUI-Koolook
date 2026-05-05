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
