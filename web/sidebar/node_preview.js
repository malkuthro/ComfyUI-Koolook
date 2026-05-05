// =============================================================================
// Node preview — hover card for sidebar leaf rows.
//
// Mirrors ComfyUI's official Node Library preview (NodePreview.vue):
// plain HTML/CSS card with a colored title bar, two columns of type-colored
// slot dots (inputs / outputs), a widgets list, and an optional description.
// No real LiteGraph node is instantiated — the card is a styled mock built
// from the node class's static metadata, so opening a preview costs ~one
// fixed-position div, not a hidden canvas + simulated graph state.
//
// Wiring: `attachHoverPreview(rowEl, type)` registers mouseenter/mouseleave
// on a leaf row. mouseenter starts a 250ms debounce; on fire we look up
// `LiteGraph.registered_node_types[type]`, build the card, position it to
// the right of the row (flipping leftward if it'd overflow the viewport),
// and append to `<body>`. mouseleave clears the timer and removes any
// active card. Only one card is visible at a time — the module owns a
// singleton `activeCard` that's torn down before every fresh build.
//
// The card has `pointer-events: none` so it never intercepts mouse events
// from the rows it floats over — moving the cursor between rows fires
// the next row's mouseenter as expected, even when the card visually
// covers part of the sidebar.
//
// `teardownPreview()` is exported because `tree.js`'s `renderTree` clears
// the tree by `innerHTML = ""`, which detaches every row without firing
// `mouseleave`. Without an explicit teardown call from the renderer, an
// active card could survive a re-render — pointer-events:none keeps the
// user from dismissing it manually, so it'd hang there until the next
// successful hover. Calling `teardownPreview` before each render ensures
// any stray card is removed in lockstep with its anchor.
// =============================================================================

import { app } from "../../../../scripts/app.js";

const HOVER_DELAY_MS = 250;
const ROW_OFFSET_PX = 12;
const VIEWPORT_PADDING_PX = 8;

// Module-level singletons. We never need more than one card or one timer
// at a time; clearing both before each fresh build keeps the DOM and
// scheduling state in lockstep.
let activeCard = null;
let hoverTimer = null;

export function teardownPreview() {
    if (hoverTimer) {
        clearTimeout(hoverTimer);
        hoverTimer = null;
    }
    if (activeCard) {
        activeCard.remove();
        activeCard = null;
    }
}

// Tab-switch / window-blur teardown. The card's `pointer-events: none`
// makes it undismissable by mouse if the user navigates away mid-hover —
// keyboard-switching ComfyUI sidebar tabs, an overlay/modal stealing
// focus, or the page going hidden never crosses a row's mouseleave
// region, so the reactive teardown path doesn't fire. These global
// listeners cover those gaps. Wired exactly once per module load
// (top-level), and kept module-private so callers don't rewire them.
if (typeof document !== "undefined") {
    document.addEventListener("visibilitychange", () => {
        if (document.hidden) teardownPreview();
    });
}
if (typeof window !== "undefined") {
    window.addEventListener("blur", teardownPreview);
}

export function attachHoverPreview(rowEl, type) {
    rowEl.addEventListener("mouseenter", () => {
        teardownPreview();
        hoverTimer = setTimeout(() => {
            hoverTimer = null;
            const nodeClass = readNodeClass(type);
            const card = renderPreviewCard(nodeClass, type);
            if (!card) return;
            // Two-phase position: append off-screen so we can measure the
            // card's actual rendered size, then move it to the final
            // location based on row + viewport bounds.
            card.style.position = "fixed";
            card.style.visibility = "hidden";
            card.style.left = "-9999px";
            card.style.top = "0";
            document.body.appendChild(card);
            const cardRect = card.getBoundingClientRect();
            const rowRect = rowEl.getBoundingClientRect();
            const vw = window.innerWidth;
            const vh = window.innerHeight;
            // Right-side preferred (the sidebar is on the left in the
            // common Kforge Labs layout). If the card would clip the
            // right edge of the viewport, flip to the left of the row.
            // The flipped position is also clamped so a wide card doesn't
            // negative-overflow off the left edge of the screen.
            const wouldOverflowRight = rowRect.right + ROW_OFFSET_PX + cardRect.width > vw;
            const left = wouldOverflowRight
                ? Math.max(VIEWPORT_PADDING_PX, rowRect.left - ROW_OFFSET_PX - cardRect.width)
                : rowRect.right + ROW_OFFSET_PX;
            const top = Math.max(
                VIEWPORT_PADDING_PX,
                Math.min(vh - cardRect.height - VIEWPORT_PADDING_PX, rowRect.top),
            );
            card.style.left = `${left}px`;
            card.style.top = `${top}px`;
            card.style.visibility = "visible";
            activeCard = card;
        }, HOVER_DELAY_MS);
    });
    rowEl.addEventListener("mouseleave", teardownPreview);
}

// =============================================================================
// Card rendering
// =============================================================================

function readNodeClass(type) {
    if (typeof LiteGraph === "undefined") return null;
    const reg = LiteGraph.registered_node_types || {};
    return reg[type] || null;
}

export function renderPreviewCard(nodeClass, type) {
    const card = document.createElement("div");
    card.className = "koolook-preview-card";

    if (!nodeClass) {
        // Pack not currently loaded. Same source-of-truth as the
        // (unresolved) bucket in category-mode gather. The stub keeps the
        // hover-preview affordance consistent across resolved/unresolved
        // picks rather than silently no-oping for missing packs.
        const title = document.createElement("div");
        title.className = "koolook-preview-title koolook-preview-title-unresolved";
        title.textContent = type;
        card.appendChild(title);
        const stub = document.createElement("div");
        stub.className = "koolook-preview-stub";
        stub.textContent = "Pack not loaded — use the ↓ Install missing button in the Tools row.";
        card.appendChild(stub);
        return card;
    }

    const cat = (nodeClass.category && String(nodeClass.category)) || "";
    const titleText = (nodeClass.title && String(nodeClass.title)) || type;

    const title = document.createElement("div");
    title.className = "koolook-preview-title";
    title.textContent = titleText;
    title.style.background = categoryColor(cat);
    card.appendChild(title);

    if (cat) {
        const catEl = document.createElement("div");
        catEl.className = "koolook-preview-cat";
        catEl.textContent = cat;
        card.appendChild(catEl);
    }

    const slots = readSlots(nodeClass);
    if (slots.inputs.length > 0 || slots.outputs.length > 0) {
        const ioGrid = document.createElement("div");
        ioGrid.className = "koolook-preview-io";
        ioGrid.appendChild(renderSlotsColumn("Inputs", slots.inputs));
        ioGrid.appendChild(renderSlotsColumn("Outputs", slots.outputs));
        card.appendChild(ioGrid);
    }

    if (slots.widgets.length > 0) {
        const wSec = document.createElement("div");
        wSec.className = "koolook-preview-widgets";
        const wTitle = document.createElement("div");
        wTitle.className = "koolook-preview-section-title";
        wTitle.textContent = "Widgets";
        wSec.appendChild(wTitle);
        for (const w of slots.widgets) {
            const wRow = document.createElement("div");
            wRow.className = "koolook-preview-widget-row";
            const name = document.createElement("span");
            name.className = "koolook-preview-widget-name";
            name.textContent = w.name;
            wRow.appendChild(name);
            const def = document.createElement("span");
            def.className = "koolook-preview-widget-default";
            const dStr = w.default !== undefined && w.default !== null
                ? ` = ${truncate(String(w.default), 32)}`
                : "";
            def.textContent = `${w.type}${dStr}`;
            wRow.appendChild(def);
            wSec.appendChild(wRow);
        }
        card.appendChild(wSec);
    }

    const description = nodeClass.description ? String(nodeClass.description).trim() : "";
    if (description) {
        const desc = document.createElement("div");
        desc.className = "koolook-preview-desc";
        desc.textContent = description;
        card.appendChild(desc);
    }

    return card;
}

function renderSlotsColumn(label, slots) {
    const col = document.createElement("div");
    col.className = "koolook-preview-col";

    const t = document.createElement("div");
    t.className = "koolook-preview-section-title";
    t.textContent = label;
    col.appendChild(t);

    if (slots.length === 0) {
        const empty = document.createElement("div");
        empty.className = "koolook-preview-empty";
        empty.textContent = "—";
        col.appendChild(empty);
        return col;
    }
    for (const slot of slots) {
        const row = document.createElement("div");
        row.className = "koolook-preview-slot-row";
        const dot = document.createElement("span");
        dot.className = "koolook-preview-slot-dot";
        dot.style.background = slotColor(slot.type);
        row.appendChild(dot);
        const nameEl = document.createElement("span");
        nameEl.className = "koolook-preview-slot-name";
        if (slot.optional) nameEl.classList.add("koolook-preview-slot-optional");
        nameEl.textContent = slot.name;
        row.appendChild(nameEl);
        const tEl = document.createElement("span");
        tEl.className = "koolook-preview-slot-type";
        tEl.textContent = slot.type;
        row.appendChild(tEl);
        col.appendChild(row);
    }
    return col;
}

// =============================================================================
// Metadata extraction
//
// ComfyUI's INPUT_TYPES is convention-driven: each input is `[type, config?]`
// where `type` is either a string (slot type like "IMAGE", "MODEL", or a
// scalar widget type like "INT") or an array of choices (a COMBO widget).
// The same convention is mirrored on the JS side because LiteGraph reads
// the Python-derived definitions verbatim.
//
// "Widget" classification matches ComfyUI's frontend rule: scalar types
// (INT, FLOAT, STRING, BOOLEAN) and arrays-of-choices (COMBO). Anything
// else is a connection slot.
// =============================================================================

const SCALAR_WIDGET_TYPES = new Set(["INT", "FLOAT", "STRING", "BOOLEAN"]);

function readSlots(nodeClass) {
    const inputs = [];
    const widgets = [];
    const outputs = [];

    let inputTypes = null;
    try {
        if (typeof nodeClass.INPUT_TYPES === "function") {
            inputTypes = nodeClass.INPUT_TYPES();
        } else if (nodeClass.INPUT_TYPES && typeof nodeClass.INPUT_TYPES === "object") {
            inputTypes = nodeClass.INPUT_TYPES;
        }
    } catch (e) {
        // Some custom-node packs throw from INPUT_TYPES under unusual
        // conditions (missing sibling deps, lazy imports). Treat as
        // "no input metadata available" and continue — the card still
        // renders the title + outputs + description. Log so the failure
        // is visible during pack debugging instead of silently rendering
        // an empty-looking node. Matches the project-wide convention
        // (see picks_store.js, canvas_io.js, installer.js).
        console.warn("[Kforge Labs preview] INPUT_TYPES() threw for", nodeClass.title || "<no title>", e);
        inputTypes = null;
    }

    if (inputTypes && typeof inputTypes === "object") {
        for (const section of ["required", "optional"]) {
            const block = inputTypes[section];
            if (!block || typeof block !== "object") continue;
            for (const [name, spec] of Object.entries(block)) {
                if (!Array.isArray(spec) || spec.length === 0) continue;
                const t = spec[0];
                const config = (spec[1] && typeof spec[1] === "object") ? spec[1] : {};
                if (Array.isArray(t)) {
                    // COMBO — first element is the array of choices, default
                    // is either explicit or the first choice.
                    const fallback = t.length > 0 ? t[0] : undefined;
                    widgets.push({
                        name,
                        type: "COMBO",
                        default: config.default !== undefined ? config.default : fallback,
                    });
                } else if (typeof t === "string" && SCALAR_WIDGET_TYPES.has(t)) {
                    widgets.push({ name, type: t, default: config.default });
                } else {
                    inputs.push({
                        name,
                        type: typeof t === "string" ? t : String(t),
                        optional: section === "optional",
                    });
                }
            }
        }
    }

    const returnTypes = Array.isArray(nodeClass.RETURN_TYPES) ? nodeClass.RETURN_TYPES : [];
    const returnNames = Array.isArray(nodeClass.RETURN_NAMES) ? nodeClass.RETURN_NAMES : [];
    for (let i = 0; i < returnTypes.length; i++) {
        const t = returnTypes[i];
        const tStr = typeof t === "string" ? t : String(t);
        const name = returnNames[i] != null ? String(returnNames[i]) : tStr;
        outputs.push({ name, type: tStr });
    }
    return { inputs, widgets, outputs };
}

// =============================================================================
// Color helpers
// =============================================================================

// Hash a category path into a stable HSL hue. Keeps related-category cards
// visually distinct without maintaining a hand-mapped palette. Saturation
// + lightness are dialed for legibility on the sidebar's dark background.
function categoryColor(category) {
    if (!category) return "rgba(120,120,120,0.6)";
    let hash = 0;
    for (let i = 0; i < category.length; i++) {
        hash = ((hash << 5) - hash + category.charCodeAt(i)) | 0;
    }
    const hue = Math.abs(hash) % 360;
    return `hsl(${hue}, 38%, 32%)`;
}

// Slot dot color — match the canvas's wire colors so a preview and the
// on-canvas wires read as the same palette. ComfyUI populates the rich
// per-type palette at runtime on the canvas instance
// (`default_connection_color_byType`), while LiteGraph's static
// `link_type_colors` only carries a stub set (`-1`, `number`, `node`).
// Read the runtime palette first; fall back through the static map; fall
// back to neutral grey for unknown slot types so the card still renders
// rather than emitting transparent-dot rows.
function slotColor(type) {
    const key = String(type);
    // Both palettes are normally string-keyed maps of CSS color strings,
    // but a misbehaving custom-node pack could mutate the runtime palette
    // to hold a non-string (object, function, etc.). Assigning a
    // non-string to `dot.style.background` silently no-ops, leaving a
    // transparent dot. Guarding both branches with a `typeof === "string"`
    // check makes the fallback chain robust against that, so the worst
    // case is still a visible neutral-grey dot.
    const runtime = (app && app.canvas && app.canvas.default_connection_color_byType) || null;
    if (runtime && typeof runtime[key] === "string") return runtime[key];
    const staticMap = (typeof LiteGraph !== "undefined"
        && LiteGraph.LGraphCanvas
        && LiteGraph.LGraphCanvas.link_type_colors) || {};
    if (typeof staticMap[key] === "string") return staticMap[key];
    return "rgba(180,180,180,0.5)";
}

function truncate(s, n) {
    return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}
