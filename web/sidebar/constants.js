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
export const PICKS_CHANGED_EVENT = "koolook-picks-changed";

// Group mode for the Nodes section — "repo" (default; group by pack/repo,
// matching pre-#73 behavior) or "category" (group by node-class CATEGORY
// path, with case-insensitive merging). Persisted in localStorage so the
// chosen mode survives reloads.
export const GROUP_MODE_KEY = "koolook.groupMode.v1";
export const GROUP_MODE_DEFAULT = "repo";

// Starter preset — a single shipped snapshot file the seeder copies into the
// user's snapshot library directory on first run (instead of seeding picks
// directly into localStorage). Carries the full snapshot schema (picks +
// workflows + tags + archive), so the user gets one Load click for the whole
// starter state. `STARTER_SEEDED_KEY` is the localStorage flag that says
// "we've already attempted the seed on this browser, don't retry."
export const STARTER_SEEDED_KEY = "koolook.starter.seeded.v1";
export const STARTER_URL = new URL("../starter_preset.json", import.meta.url).href;
// Filename inside the snapshot library — without `.json`, matches the
// shape `writePreset(fileName, ...)` expects.
export const STARTER_PRESET_FILENAME = "starter";

export const WORKFLOWS_USERDATA_PATH = "koolook_workflows.json";
export const WORKFLOWS_FALLBACK_KEY = "koolook.workflows.fallback.v1";
export const WORKFLOWS_SEEDED_KEY = "koolook.workflows.seeded.v1";
export const WORKFLOWS_CHANGED_EVENT = "koolook-workflows-changed";
export const WORKFLOWS_DEFAULTS_URL = new URL("../workflow_defaults.json", import.meta.url).href;

// Convention-driven "module" classification — a saved workflow tagged with
// this literal string is treated as a reusable building block (insert into
// the existing canvas) instead of a full session (replace the canvas).
//
// Why a tag instead of a dedicated `isModule` field on the entry? Three
// reasons: (1) zero schema migration; (2) the Tags section already groups
// `module`-tagged entries for free; (3) re-tagging an existing saved entry
// turns it into a module without any data shape change.
//
// Comparison is case-sensitive to match `addTag` / `removeTag` semantics in
// `workflows_store.js`. The literal lives here so future renames (or a
// "modules also includes `preset`" rule) only need to touch one constant.
export const MODULE_TAG = "module";

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
.koolook-module-icon { color: #7be08a; opacity: 0.95; flex-shrink: 0; font-size: 11px; }
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
.koolook-modal-checkbox-row { display: flex; align-items: center; gap: 8px; margin-top: 10px; font-size: 12px; opacity: 0.85; cursor: pointer; user-select: none; }
.koolook-modal-checkbox-row input[type="checkbox"] { cursor: pointer; }
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
.koolook-tags-chips { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 4px; min-height: 22px; align-items: center; }
.koolook-tags-empty { opacity: 0.55; font-size: 12px; }
.koolook-tag-chip { display: inline-flex; align-items: center; gap: 4px; background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); border-radius: 10px; padding: 2px 4px 2px 9px; font-size: 11px; line-height: 1.4; }
.koolook-tag-chip-x { cursor: pointer; opacity: 0.6; padding: 0 4px; font-size: 13px; line-height: 1; border-radius: 50%; }
.koolook-tag-chip-x:hover { opacity: 1; color: #ff7777; background: rgba(255,255,255,0.06); }
.koolook-tag-add-row { display: flex; gap: 6px; align-items: center; }
.koolook-tag-add-row .koolook-modal-input { flex: 1; }
.koolook-install-stat-line { font-size: 12px; line-height: 1.5; padding: 1px 0; }
.koolook-install-stat-line.koolook-install-stat-fail { color: #ff8888; }
.koolook-install-unresolved { font-size: 11px; opacity: 0.75; margin-top: 4px; padding: 6px 8px; background: rgba(255,255,255,0.04); border-radius: 4px; word-break: break-all; max-height: 110px; overflow-y: auto; }
.koolook-install-unresolved-summary { cursor: pointer; font-size: 11px; opacity: 0.75; margin-top: 4px; }
.koolook-install-progress { width: 100%; height: 6px; background: rgba(255,255,255,0.08); border-radius: 3px; margin: 10px 0 4px; overflow: hidden; }
.koolook-install-progress-bar { height: 100%; background: rgba(80,140,235,0.6); width: 0%; transition: width 200ms ease; }
.koolook-snapshot-list { max-height: 280px; overflow-y: auto; border: 1px solid rgba(255,255,255,0.08); border-radius: 4px; }
.koolook-snapshot-row { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,0.05); }
.koolook-snapshot-row:last-child { border-bottom: none; }
.koolook-snapshot-row-info { flex: 1; min-width: 0; cursor: pointer; }
.koolook-snapshot-row-info:hover .koolook-snapshot-row-name { color: var(--p-primary-color, rgba(120,170,255,1)); }
.koolook-snapshot-row-name { font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.koolook-snapshot-row-meta { font-size: 11px; opacity: 0.55; margin-top: 2px; }
.koolook-snapshot-row-actions { display: flex; gap: 4px; flex-shrink: 0; }
.koolook-snapshot-row-btn { padding: 3px 8px; background: var(--comfy-input-bg, rgba(0,0,0,0.3)); border: 1px solid rgba(255,255,255,0.08); border-radius: 3px; cursor: pointer; color: inherit; font-size: 11px; }
.koolook-snapshot-row-btn:hover { background: rgba(255,255,255,0.08); }
.koolook-snapshot-row-btn-danger:hover { background: rgba(220,80,80,0.25); border-color: rgba(220,80,80,0.4); }
.koolook-snapshot-empty { padding: 20px; opacity: 0.5; font-size: 12px; text-align: center; }
.koolook-mode-toggle { display: inline-flex; gap: 0; margin-right: 4px; border: 1px solid var(--border-color, rgba(255,255,255,0.1)); border-radius: 4px; overflow: hidden; }
.koolook-mode-toggle-btn { padding: 0 8px; height: 22px; background: transparent; border: none; cursor: pointer; color: inherit; font-size: 11px; opacity: 0.5; line-height: 1; }
.koolook-mode-toggle-btn:hover { background: rgba(255,255,255,0.06); opacity: 0.85; }
.koolook-mode-toggle-btn.koolook-mode-active { background: rgba(80,140,235,0.25); opacity: 1; }
.koolook-pack-badge { font-size: 10px; padding: 1px 6px; border-radius: 8px; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08); opacity: 0.7; flex-shrink: 0; max-width: 110px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.koolook-leaf-unresolved { opacity: 0.55; font-style: italic; }
.koolook-leaf-crumb { opacity: 0.5; font-size: 11px; margin-right: 1px; }
.koolook-preview-card { position: fixed; z-index: 10000; background: var(--comfy-menu-bg, #2a2a2a); border: 1px solid var(--border-color, rgba(255,255,255,0.18)); border-radius: 6px; box-shadow: 0 6px 24px rgba(0,0,0,0.55); color: var(--input-text, #ddd); font-size: 12px; width: 320px; max-width: 90vw; max-height: calc(100vh - 16px); pointer-events: none; overflow: hidden auto; }
.koolook-preview-title { padding: 6px 10px; font-size: 13px; font-weight: 600; color: #fff; }
.koolook-preview-title-unresolved { background: rgba(180,80,80,0.4); }
.koolook-preview-cat { padding: 3px 10px 5px; font-size: 10px; opacity: 0.6; border-bottom: 1px solid rgba(255,255,255,0.05); font-family: monospace; }
.koolook-preview-stub { padding: 10px; font-style: italic; opacity: 0.7; }
.koolook-preview-io { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: rgba(255,255,255,0.06); }
.koolook-preview-col { padding: 7px 9px; background: var(--comfy-menu-bg, #2a2a2a); }
.koolook-preview-section-title { font-size: 10px; opacity: 0.5; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 4px; }
.koolook-preview-slot-row { display: flex; align-items: center; gap: 5px; padding: 1px 0; line-height: 1.35; }
.koolook-preview-slot-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; box-shadow: 0 0 0 1px rgba(0,0,0,0.4); }
.koolook-preview-slot-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.koolook-preview-slot-name.koolook-preview-slot-optional { opacity: 0.7; font-style: italic; }
.koolook-preview-slot-type { font-size: 10px; opacity: 0.5; flex-shrink: 0; font-family: monospace; max-width: 50%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.koolook-preview-empty { opacity: 0.4; font-size: 11px; padding: 1px 0; }
.koolook-preview-widgets { padding: 6px 9px; border-top: 1px solid rgba(255,255,255,0.06); }
.koolook-preview-widget-row { display: flex; gap: 6px; padding: 1px 0; line-height: 1.35; }
.koolook-preview-widget-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.koolook-preview-widget-default { opacity: 0.5; font-size: 10px; font-family: monospace; max-width: 60%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex-shrink: 0; }
.koolook-preview-desc { padding: 6px 9px; border-top: 1px solid rgba(255,255,255,0.06); font-size: 11px; opacity: 0.78; line-height: 1.4; max-height: 100px; overflow: hidden; }
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
