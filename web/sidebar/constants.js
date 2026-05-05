// =============================================================================
// Configuration — static "always shown" entries.
//
// You normally don't need to edit this. To grow the panel from inside
// ComfyUI, use either:
//   • the "+" button at the top of the sidebar (with a canvas node selected), or
//   • right-click a canvas node → "Add to Kforge Labs".
//
// Those additions are stored in the browser's localStorage and surface as
// the "My Picks" group at the top of the tree.
// =============================================================================
export const REPOS = [
    {
        label: "Koolook",
        categoryRoot: "Koolook",
        select: "all",
        excludePatterns: [/\bdeprecated\b/i],
    },
];

// `TAB_ID` is the stable identifier ComfyUI persists per-user — never rename
// or existing tab state (pinning, tab order) gets orphaned. The visible
// title and tooltip can change freely.
export const TAB_ID = "koolook.curatedNodes";
export const TAB_TITLE = "Kforge Labs";
export const TAB_TOOLTIP = "Kforge Labs — curated nodes & workflows";
export const TAB_ICON = "pi pi-star";
export const ROOT_GROUP_LABEL = "Nodes (favorites)";
export const WORKFLOWS_GROUP_LABEL = "Workflows";

// Defaults JSON files live in web/ alongside the entry; this module is one
// level deeper at web/sidebar/, so the relative URL needs the extra `..`.
export const STORAGE_KEY = "koolook.curated.userPicks.v1";
export const SEEDED_KEY = "koolook.curated.seeded.v1";
export const PICKS_CHANGED_EVENT = "koolook-picks-changed";
export const DEFAULTS_URL = new URL("../curated_defaults.json", import.meta.url).href;

export const WORKFLOWS_USERDATA_PATH = "koolook_workflows.json";
export const WORKFLOWS_FALLBACK_KEY = "koolook.workflows.fallback.v1";
export const WORKFLOWS_SEEDED_KEY = "koolook.workflows.seeded.v1";
export const WORKFLOWS_CHANGED_EVENT = "koolook-workflows-changed";
export const WORKFLOWS_DEFAULTS_URL = new URL("../workflow_defaults.json", import.meta.url).href;

// =============================================================================
// Styles
// =============================================================================
const STYLE_ID = "koolook-sidebar-style";
export function ensureStyle() {
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
.koolook-row[draggable="true"] { cursor: grab; }
.koolook-row[draggable="true"]:active { cursor: grabbing; }
.koolook-drop-target { background: rgba(80,140,235,0.18) !important; outline: 2px solid rgba(80,140,235,0.6); outline-offset: -2px; }
`;
    document.head.appendChild(s);
}

// =============================================================================
// Toast helper
// =============================================================================
export function toast(msg, duration = 2200) {
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
export function compareNames(a, b) {
    return String(a).localeCompare(String(b), undefined, { sensitivity: "base" });
}
