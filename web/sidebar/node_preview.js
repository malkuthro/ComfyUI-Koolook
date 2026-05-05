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
        // picks rather than silently no-oping for missing packs. Renders
        // header + stub only, no slot grid — there's nothing to mock.
        appendHeader(card, type, /* unresolved */ true);
        const stub = document.createElement("div");
        stub.className = "koolook-preview-stub";
        stub.textContent = "Pack not loaded — use the ↓ Install missing button in the Tools row.";
        card.appendChild(stub);
        return card;
    }

    // Category / title / description — try the constructor-level properties
    // first (set by some non-ComfyUI registered nodes), then fall through to
    // `nodeData` (where ComfyUI's `registerNodeDef` parks the original def).
    const def = nodeClass.nodeData || null;
    const cat = (nodeClass.category && String(nodeClass.category))
        || (def && def.category && String(def.category))
        || "";
    const titleText = (nodeClass.title && String(nodeClass.title))
        || (def && def.display_name && String(def.display_name))
        || type;

    appendHeader(card, titleText, /* unresolved */ false, cat);

    // "PREVIEW" badge — purely a visual marker that this is a static mock,
    // not a live node. Mirrors upstream NodePreview.vue's `_sb_preview_badge`.
    const badge = document.createElement("div");
    badge.className = "koolook-preview-badge";
    badge.textContent = "Preview";
    card.appendChild(badge);

    const slots = readSlots(nodeClass);
    // Slot rows pair inputs and outputs horizontally — row i shows
    // `[dot] [input[i].name] [spacer] [output[i].name] [dot]`. When one
    // side is shorter, the other side's later cells are blank. This is
    // what makes the card read as a node mock: the sockets line up the
    // same way they do on the real node.
    const pairCount = Math.max(slots.inputs.length, slots.outputs.length);
    for (let i = 0; i < pairCount; i++) {
        appendSlotPairRow(card, slots.inputs[i] || null, slots.outputs[i] || null);
    }

    // Widget rows — each rendered as a rounded "pill" with ◄ ► chrome to
    // mimic LiteGraph widget UI. The triangles are static glyphs, no
    // interaction; the value column shows the widget's default (or its
    // first choice for COMBOs).
    for (const w of slots.widgets) {
        appendWidgetRow(card, w);
    }

    const description = (nodeClass.description && String(nodeClass.description).trim())
        || (def && def.description && String(def.description).trim())
        || "";
    if (description) {
        const desc = document.createElement("div");
        desc.className = "koolook-preview-desc";
        desc.textContent = description;
        card.appendChild(desc);
    }

    return card;
}

// Header — colored dot + display name. The dot color is derived from the
// node's category via `categoryColor` so cards from the same category share
// a hue, making them visually groupable at a glance. Unresolved cards get
// the same shape with a muted dot and an "italic" title to read as a stub.
function appendHeader(card, title, unresolved, category) {
    const header = document.createElement("div");
    header.className = "koolook-preview-header";
    if (unresolved) header.classList.add("koolook-preview-header-unresolved");
    const dot = document.createElement("span");
    dot.className = "koolook-preview-headdot";
    dot.style.background = unresolved ? "rgba(180,120,120,0.7)" : categoryColor(category || "");
    header.appendChild(dot);
    const titleEl = document.createElement("span");
    titleEl.className = "koolook-preview-headtitle";
    titleEl.textContent = title;
    header.appendChild(titleEl);
    card.appendChild(header);
}

// One row in the 5-column slot grid: `[dot] [in-name] [middle] [out-name] [dot]`.
// Either slot may be null when the input/output counts are unequal — the
// corresponding cells render as empty placeholders so the grid alignment
// holds.
function appendSlotPairRow(card, inputSlot, outputSlot) {
    const row = document.createElement("div");
    row.className = "koolook-preview-row koolook-preview-row-slot";

    // Column 1: input dot
    const inDot = document.createElement("div");
    inDot.className = "koolook-preview-col";
    if (inputSlot) {
        const dot = document.createElement("span");
        dot.className = "koolook-preview-slot-dot";
        dot.style.background = slotColor(inputSlot.type);
        inDot.appendChild(dot);
    }
    row.appendChild(inDot);

    // Column 2: input name (left-aligned)
    const inName = document.createElement("div");
    inName.className = "koolook-preview-col koolook-preview-col-input";
    if (inputSlot) {
        inName.textContent = inputSlot.name;
        if (inputSlot.optional) inName.classList.add("koolook-preview-slot-optional");
        inName.title = `${inputSlot.name}: ${inputSlot.type}`;
    }
    row.appendChild(inName);

    // Column 3: middle spacer (empty by design)
    const mid = document.createElement("div");
    mid.className = "koolook-preview-col koolook-preview-col-middle";
    row.appendChild(mid);

    // Column 4: output name (right-aligned)
    const outName = document.createElement("div");
    outName.className = "koolook-preview-col koolook-preview-col-output";
    if (outputSlot) {
        outName.textContent = outputSlot.name;
        outName.title = `${outputSlot.name}: ${outputSlot.type}`;
    }
    row.appendChild(outName);

    // Column 5: output dot
    const outDot = document.createElement("div");
    outDot.className = "koolook-preview-col";
    if (outputSlot) {
        const dot = document.createElement("span");
        dot.className = "koolook-preview-slot-dot";
        dot.style.background = slotColor(outputSlot.type);
        outDot.appendChild(dot);
    }
    row.appendChild(outDot);

    card.appendChild(row);
}

// Widget pill — same 5-column grid shape as slot rows but with `◄` / `►`
// triangle glyphs in the dot columns and a rounded-bordered background to
// read as a LiteGraph widget. Value column truncates long defaults.
function appendWidgetRow(card, widget) {
    const row = document.createElement("div");
    row.className = "koolook-preview-row koolook-preview-row-widget";

    const arrowL = document.createElement("div");
    arrowL.className = "koolook-preview-col koolook-preview-arrow";
    arrowL.textContent = "◀";
    row.appendChild(arrowL);

    const name = document.createElement("div");
    name.className = "koolook-preview-col koolook-preview-widget-name";
    name.textContent = widget.name;
    name.title = `${widget.name} (${widget.type})`;
    row.appendChild(name);

    const mid = document.createElement("div");
    mid.className = "koolook-preview-col koolook-preview-col-middle";
    row.appendChild(mid);

    const value = document.createElement("div");
    value.className = "koolook-preview-col koolook-preview-widget-value";
    if (widget.default !== undefined && widget.default !== null) {
        value.textContent = truncate(String(widget.default), 24);
    } else {
        value.textContent = widget.type;
    }
    row.appendChild(value);

    const arrowR = document.createElement("div");
    arrowR.className = "koolook-preview-col koolook-preview-arrow";
    arrowR.textContent = "▶";
    row.appendChild(arrowR);

    card.appendChild(row);
}


// =============================================================================
// Metadata extraction
//
// ComfyUI doesn't keep `INPUT_TYPES` / `RETURN_TYPES` as static properties on
// the registered constructor. Instead, the original Python-side node def
// (the V1 shape returned by `/object_info`) is stashed at `nodeClass.nodeData`
// during registration (see Comfy-Org/ComfyUI_frontend `litegraphService.ts`
// `registerNodeDef` — `node.nodeData = nodeDef` right before
// `LiteGraph.registerNodeType`). `nodeData` exposes:
//
//   nodeData.input.required   { name: ["TYPE", config?], ... }   V1 shape
//   nodeData.input.optional   { name: ["TYPE", config?], ... }   V1 shape
//   nodeData.output           ["TYPE1", "TYPE2", ...]            V1 shape
//   nodeData.output_name      ["name1", "name2", ...]            V1 shape
//   nodeData.description      string
//   nodeData.category         string
//   nodeData.display_name     string
//
// Plus a parallel V2 transform (`inputs` record / `outputs` array) — we read
// V1 because it's a 1-to-1 with the Python convention and matches the
// upstream `INPUT_TYPES` shape exactly. Fall back to direct
// `nodeClass.INPUT_TYPES` / `RETURN_TYPES` for any future or non-ComfyUI
// LiteGraph nodes that follow the old convention.
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

    const def = nodeClass.nodeData || null;

    // Inputs: prefer `nodeData.input` (V1 shape ComfyUI preserves verbatim);
    // fall back to direct `INPUT_TYPES` for non-ComfyUI registered nodes.
    let inputBlock = null;
    if (def && def.input && typeof def.input === "object") {
        inputBlock = def.input;
    } else {
        try {
            if (typeof nodeClass.INPUT_TYPES === "function") {
                inputBlock = nodeClass.INPUT_TYPES();
            } else if (nodeClass.INPUT_TYPES && typeof nodeClass.INPUT_TYPES === "object") {
                inputBlock = nodeClass.INPUT_TYPES;
            }
        } catch (e) {
            // Some legacy packs throw from INPUT_TYPES under unusual
            // conditions (missing sibling deps, lazy imports). Log so the
            // failure is visible during pack debugging instead of silently
            // rendering an empty-looking node. Matches the project-wide
            // convention (see picks_store.js, canvas_io.js, installer.js).
            console.warn("[Kforge Labs preview] INPUT_TYPES() threw for", nodeClass.title || "<no title>", e);
            inputBlock = null;
        }
    }

    if (inputBlock && typeof inputBlock === "object") {
        for (const section of ["required", "optional"]) {
            const block = inputBlock[section];
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

    // Outputs: prefer `nodeData.output` (parallel `output_name` for labels);
    // fall back to direct `RETURN_TYPES` / `RETURN_NAMES` on the constructor.
    let returnTypes = [];
    let returnNames = [];
    if (def && Array.isArray(def.output)) {
        returnTypes = def.output;
        returnNames = Array.isArray(def.output_name) ? def.output_name : [];
    } else if (Array.isArray(nodeClass.RETURN_TYPES)) {
        returnTypes = nodeClass.RETURN_TYPES;
        returnNames = Array.isArray(nodeClass.RETURN_NAMES) ? nodeClass.RETURN_NAMES : [];
    }
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
