// =============================================================================
// Drag-link-release menu — curated picks priority.
//
// When the user drags a connection out of a node socket and releases on empty
// canvas, LiteGraph shows a context menu of operations + compatible nodes
// (filtered by socket type). Default order is whatever LiteGraph emits,
// which is essentially "all installed nodes that match the type, in
// registration order." On a busy ComfyUI install this menu can be 20+ items
// of nodes the user has never used.
//
// We hoist the user's curated picks (those that ALSO match the dragged
// socket's type) to the top of the suggestion list, immediately below
// LiteGraph's operation entries (Add Node / Add Reroute / Search / Reroute).
// Other compatible nodes still render below — purely additive ordering, no
// information loss.
//
// Strategy: monkey-patch `LiteGraph.ContextMenu` globally and detect the
// connection-release menu by sentinel items in its options array. Detection
// requires BOTH "Add Node" AND "Add Reroute" to be present — narrow enough
// that other ContextMenu uses (right-click on a node, on the canvas, etc.)
// are untouched, while broad enough to survive minor LiteGraph variations.
//
// LiteGraph fork variants used by ComfyUI may shape the suggestion items
// either with `content: <typeId>` or `content: <displayName>` and a parallel
// `value: <typeId>`. We check both, so the picks lookup matches whichever
// shape is in flight.
// =============================================================================
import { loadUserPicks } from "./picks_store.js";

const SEPARATOR_LABEL = "── from your picks ──";

let installed = false;

export function patchConnectionMenu() {
    const LG = (typeof LiteGraph !== "undefined") ? LiteGraph : null;
    if (!LG || typeof LG.ContextMenu !== "function") {
        console.warn("[Koolook] LiteGraph.ContextMenu not available; connection-menu priority skipped.");
        return;
    }
    // Idempotency: this is a global constructor swap. A re-load (HMR /
    // multiple registerExtension calls) would otherwise stack wrappers,
    // running the priority logic N times per ContextMenu construction —
    // performance + duplicate "── from your picks ──" separators.
    if (LG.__koolookContextMenuPatched) return;
    LG.__koolookContextMenuPatched = true;

    const origContextMenu = LG.ContextMenu;

    function PatchedContextMenu(options, parentMenuOptions) {
        let finalOptions = options;
        try {
            if (looksLikeConnectionMenu(options)) {
                finalOptions = reorderByPicks(options);
            }
        } catch (e) {
            // Never let a reorder bug break the menu — fall back to the
            // original options array so the user always gets SOME menu.
            console.warn("[Koolook] connection-menu reorder failed, falling back:", e);
            finalOptions = options;
        }
        return new origContextMenu(finalOptions, parentMenuOptions);
    }

    // Preserve the prototype chain so `instanceof LiteGraph.ContextMenu`
    // checks elsewhere in LiteGraph / ComfyUI still pass, and any static
    // properties on the original constructor remain reachable through ours.
    PatchedContextMenu.prototype = origContextMenu.prototype;
    Object.setPrototypeOf(PatchedContextMenu, origContextMenu);
    LG.ContextMenu = PatchedContextMenu;
}

// =============================================================================
// Detection + reorder
// =============================================================================

// Detects the drag-link-release menu by looking for the operation sentinels
// LiteGraph emits at the top: BOTH "Add Node" AND "Add Reroute" must be
// present. The right-click canvas menu has "Add Node" but not "Add Reroute",
// so this filter doesn't accidentally re-shape unrelated menus.
function looksLikeConnectionMenu(options) {
    if (!Array.isArray(options) || options.length === 0) return false;
    let hasAddNode = false, hasAddReroute = false;
    for (const opt of options) {
        if (!opt || typeof opt.content !== "string") continue;
        if (opt.content === "Add Node") hasAddNode = true;
        else if (opt.content === "Add Reroute") hasAddReroute = true;
        if (hasAddNode && hasAddReroute) return true;
    }
    return false;
}

// LiteGraph sets `content` to either the node type ID directly or to the
// node's display title (which can include spaces, emoji, decorations).
// Our picks list stores type IDs, so we check `value` first (when present
// it's typically the type ID) and fall back to `content`.
function pickIdOf(opt) {
    if (!opt) return null;
    if (typeof opt.value === "string") return opt.value;
    if (typeof opt.content === "string") return opt.content;
    return null;
}

// Operation entries that always sit at the top of LiteGraph's connection
// menu — preserve their position regardless of pick membership.
const OPERATIONS = new Set(["Add Node", "Add Reroute", "Search", "Reroute"]);

function reorderByPicks(options) {
    const picks = new Set(loadUserPicks());
    if (picks.size === 0) return options;

    // Walk the options array linearly. The operation block is typically
    // contiguous at the top (Add Node, Add Reroute, Search, plus the two
    // Reroute insertion entries); everything after is a node suggestion.
    // Stop the operation slice at the first non-operation, non-null item —
    // null entries are LiteGraph separators which we want to leave wherever
    // LiteGraph put them.
    let operationsEnd = 0;
    while (operationsEnd < options.length) {
        const opt = options[operationsEnd];
        if (opt === null) {
            operationsEnd += 1;
            continue;
        }
        if (opt && typeof opt.content === "string" && OPERATIONS.has(opt.content)) {
            operationsEnd += 1;
            continue;
        }
        break;
    }

    const operations = options.slice(0, operationsEnd);
    const suggestions = options.slice(operationsEnd);

    // Stable partition: picks first, others second. Skipping non-pick-able
    // entries (separators, weird shapes) puts them in `others` to preserve
    // their relative order with the rest of the suggestion block.
    const picksHits = [];
    const others = [];
    for (const opt of suggestions) {
        const id = pickIdOf(opt);
        if (id != null && picks.has(id)) picksHits.push(opt);
        else others.push(opt);
    }

    // No pick hits → no reordering needed; return the original array reference
    // so we don't pay the array-allocation cost in the common no-match path.
    if (picksHits.length === 0) return options;

    const reordered = [...operations, ...picksHits];
    if (others.length > 0) {
        // Inert separator label so the user sees where their picks end and
        // ComfyUI's default suggestions begin. `disabled: true` prevents
        // hover/click activation; LiteGraph still renders it as a row.
        reordered.push({ content: SEPARATOR_LABEL, disabled: true });
        reordered.push(...others);
    }
    return reordered;
}
