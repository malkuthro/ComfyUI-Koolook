// SPDX-FileCopyrightText: 2025-2026 Kforge Labs <https://github.com/malkuthro/ComfyUI-Koolook>
// SPDX-License-Identifier: GPL-3.0-only

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

// Set of node types the user has explicitly removed via the `×` button on
// rows that would otherwise be auto-pulled by `REPOS{select: "all"}`. Lets
// `×` work uniformly for any visible favorite — without this, auto-pulled
// nodes (the whole Koolook pack on a stock install) would silently re-
// appear on every render. Cleared per-type when the user re-adds via `+`.
export const AUTOPULL_HIDDEN_KEY = "koolook.autoPullHidden.v1";

// Fired when the snapshot status changes for reasons other than picks /
// workflows mutations (which already fire their own events). Specifically:
//   • markStateSaved() — after a successful named save or applySnapshot
//   • periodic auto-save success — status flips "unsaved" → "auto-saved"
//   • setCurrentPresetName clears — status name flips back to "(no snapshot)"
// The status indicator in the sidebar listens to PICKS_CHANGED_EVENT,
// WORKFLOWS_CHANGED_EVENT, AND this one to cover all status-change paths.
export const SNAPSHOT_STATUS_CHANGED_EVENT = "koolook-snapshot-status-changed";

// Group mode for the Nodes section — "repo" (default; group by pack/repo,
// matching pre-#73 behavior) or "category" (group by node-class CATEGORY
// path, with case-insensitive merging). Persisted in localStorage so the
// chosen mode survives reloads.
export const GROUP_MODE_KEY = "koolook.groupMode.v1";
// Theme mode is the better default for newcomers — they're more likely to
// think "where are my image nodes?" than "which pack is this from?". Existing
// users with localStorage already set keep whatever mode they chose.
export const GROUP_MODE_DEFAULT = "category";

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
export const GUIDE_URL = new URL("../guide/index.html", import.meta.url).href;

export function noStoreUrl(url) {
    const sep = url.includes("?") ? "&" : "?";
    return `${url}${sep}_=${Date.now()}`;
}

// Convention-driven "module" classification — a saved workflow tagged with
// this literal string is treated as a reusable building block (insert into
// the existing canvas) instead of a full session (replace the canvas).
//
// Module-ness is stored in both places: `wf.module === true` for fast row
// rendering and this literal tag for Tags-section grouping / manual
// right-click toggling. `addTag` / `removeTag` keep them in sync; callers
// should read via `isWorkflowModule()` instead of inspecting either field.
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
.koolook-sidebar { display: flex; flex-direction: column; height: 100%; font-size: 13px; user-select: none; padding-top: 10px; }
.koolook-search-row { margin: 6px; flex-shrink: 0; }
.koolook-actions-row { display: flex; align-items: center; gap: 3px; padding: 2px 6px; flex-shrink: 0; }
.koolook-actions-label { font-size: 10px; opacity: 0.55; text-transform: uppercase; letter-spacing: 0.08em; flex: 1; font-weight: 600; }
/* Snapshot status indicator — replaces the static "Snapshot" label. Shows
   the currently-tracked preset name + a colored dot indicating whether the
   live state matches the last named save (green), the latest periodic
   auto-save (blue), neither (orange = unsaved), or there's no tracked
   preset at all (grey). Tooltip carries the full breakdown. */
.koolook-snap-status { flex: 1; display: flex; align-items: center; font-size: 11px; min-width: 0; cursor: default; gap: 0; }
.koolook-snap-status-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 7px; flex-shrink: 0; box-shadow: 0 0 0 1px rgba(0,0,0,0.35); }
.koolook-snap-status-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; opacity: 0.9; }
.koolook-snap-status-name-empty { font-style: italic; opacity: 0.55; }
.koolook-snap-status-state { opacity: 0.55; flex-shrink: 0; margin-left: 4px; font-size: 10px; }
.koolook-snap-status-saved { background: #6db86d; }
.koolook-snap-status-autosaved { background: #6db4ff; }
.koolook-snap-status-unsaved { background: #e0a64a; }
/* Drifted (#161) — named-snapshot file diverged from live state at boot.
   More intense red-orange than "unsaved" because this is a recovery-
   actionable warning, not just an "edited since last save" indicator. */
.koolook-snap-status-drifted { background: #d97546; box-shadow: 0 0 0 1px rgba(0,0,0,0.45), 0 0 4px rgba(217,117,70,0.55); }
.koolook-snap-status-none { background: rgba(255,255,255,0.28); }
.koolook-snap-status-comparing { background: #6db4ff; }
.koolook-tree-divider { margin: 8px 8px; border-top: 1px solid rgba(255,255,255,0.08); flex-shrink: 0; }
.koolook-search-wrap { position: relative; width: 100%; }
.koolook-search-icon { position: absolute; left: 8px; top: 50%; transform: translateY(-50%); opacity: 0.55; font-size: 11px; pointer-events: none; }
.koolook-search { width: 100%; padding: 5px 8px 5px 26px; box-sizing: border-box; background: var(--comfy-input-bg, rgba(0,0,0,0.25)); border: 1px solid var(--border-color, rgba(255,255,255,0.1)); border-radius: 4px; color: var(--input-text, inherit); font-size: 12px; outline: none; }
.koolook-search:focus { border-color: var(--p-primary-color, rgba(100,150,255,0.5)); }
.koolook-add-btn { padding: 0; width: 28px; height: 17px; box-sizing: border-box; display: inline-flex; align-items: center; justify-content: center; background: var(--comfy-input-bg, rgba(0,0,0,0.25)); border: 1px solid var(--border-color, rgba(255,255,255,0.1)); border-radius: 3px; cursor: pointer; color: rgba(218,221,226,0.78); font-size: 12px; line-height: 1; flex-shrink: 0; }
.koolook-add-btn:hover { background: rgba(255,255,255,0.1); color: rgba(248,249,250,0.96); }
.koolook-add-btn:active { background: rgba(255,255,255,0.15); }
/* Dusty-green accent on the "+" Add-to-favorites button — distinguishes the
   primary action in the Nodes row from the neutral toolbar buttons it sits
   alongside. Same dimensions as koolook-icon-btn; only the colors differ. */
.koolook-add-btn-green { background: rgba(120, 165, 100, 0.25); border-color: rgba(120, 165, 100, 0.5); }
.koolook-add-btn-green:hover { background: rgba(120, 165, 100, 0.45); }
.koolook-add-btn-green:active { background: rgba(120, 165, 100, 0.6); }
.koolook-icon-btn { padding: 0; width: 28px; height: 17px; font-size: 12px; }
.koolook-icon-btn:disabled { opacity: 0.35; cursor: not-allowed; }
.koolook-icon-btn > .pi { font-size: 11px; }
.koolook-letter-icon { font-size: 11px; font-weight: 800; line-height: 1; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.koolook-inline-svg-icon { width: 11px; height: 11px; stroke: currentColor; fill: none; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }
.koolook-filled-square-icon { display: inline-block; width: 10px; height: 10px; border-radius: 2px; background: currentColor; opacity: 0.9; }
.koolook-stair-icon, .koolook-list-icon { display: inline-grid; grid-template-rows: repeat(3, 1.75px); gap: 2px; width: 15px; height: 10px; align-content: center; }
.koolook-stair-icon span, .koolook-list-icon span { display: block; width: 11px; height: 1.75px; background: currentColor; border-radius: 0; }
.koolook-stair-icon span:nth-child(2) { margin-left: 2px; }
.koolook-stair-icon span:nth-child(3) { margin-left: 4px; }
.koolook-add-btn-green > .pi { font-size: 13px; -webkit-text-stroke: 0.5px currentColor; }
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
.koolook-toast-critical { pointer-events: auto; background: rgba(180, 60, 60, 0.95); max-width: 440px; padding: 10px 14px; }
.koolook-toast-critical-stack { bottom: auto; top: 30px; }
.koolook-toast-msg { margin-bottom: 8px; line-height: 1.4; font-size: 12px; word-break: break-word; }
.koolook-toast-actions { display: flex; gap: 8px; justify-content: flex-end; }
.koolook-toast-btn { padding: 4px 10px; background: rgba(0,0,0,0.25); border: 1px solid rgba(255,255,255,0.25); border-radius: 3px; color: #fff; cursor: pointer; font-size: 11px; font-family: inherit; }
.koolook-toast-btn:hover { background: rgba(0,0,0,0.4); }
.koolook-modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.55); z-index: 9998; display: flex; align-items: center; justify-content: center; }
.koolook-modal { background: var(--comfy-menu-bg, #151515); border: 1px solid var(--border-color, #4f4f54); border-radius: 8px; padding: 16px 18px; min-width: 320px; max-width: 520px; box-shadow: 0 10px 40px rgba(0,0,0,0.6); color: var(--input-text, #f9fafb); overflow: hidden; }
.koolook-modal-title { font-size: 15px; font-weight: 700; margin: -16px -18px 14px; padding: 14px 18px; border-bottom: 1px solid var(--border-color, #302f2f); background: var(--comfy-menu-bg, #181818); }
.koolook-modal-pathline { font-size: 11px; opacity: 0.55; margin: -6px 0 12px; white-space: normal; overflow-wrap: anywhere; cursor: default; }
.koolook-modal-message { font-size: 12px; opacity: 0.85; margin-bottom: 14px; line-height: 1.45; }
.koolook-modal-label { font-size: 11px; opacity: 0.7; margin: 6px 0 4px; display: block; text-transform: uppercase; letter-spacing: 0.04em; }
.koolook-modal-input, .koolook-modal-select { width: 100%; padding: 6px 8px; background: var(--comfy-input-bg, #111111); border: 1px solid var(--border-color, #302f2f); border-radius: 4px; color: var(--input-text, #e6e8ec); font-size: 13px; box-sizing: border-box; outline: none; }
.koolook-modal-input:focus, .koolook-modal-select:focus { border-color: var(--p-primary-color, rgba(100,150,255,0.5)); }
/* Library-folder name + path pair used by the redesigned Save and Load
   dialogs to render the inline 'Saved to' / 'Loaded from' info row.
   Naming kept from the (deleted) Settings dialog for git-blame continuity;
   the two classes have always rendered the same shape (leaf folder name
   on the first line, full absolute path on the second). */
.koolook-settings-folder-name { opacity: 1; font-size: 13px; font-weight: 700; color: var(--input-text, #e6e8ec); margin-top: 2px; }
.koolook-settings-folder-path { color: var(--input-text, #8f959c); opacity: 0.68; font: 11px/1.35 ui-monospace, "Cascadia Mono", Menlo, monospace; margin-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.koolook-modal-actions { display: flex; gap: 8px; justify-content: flex-end; margin: 16px -18px -16px; padding: 12px 18px; border-top: 1px solid var(--border-color, #302f2f); background: var(--comfy-menu-bg, #131313); }
.koolook-modal-checkbox-row { display: flex; align-items: center; gap: 8px; margin-top: 10px; font-size: 12px; opacity: 0.85; cursor: pointer; user-select: none; }
.koolook-modal-checkbox-row input[type="checkbox"] { cursor: pointer; }
.koolook-modal-btn { padding: 6px 14px; background: var(--comfy-input-bg, #1a1a1f); border: 1px solid var(--border-color, #4f4f54); border-radius: 4px; color: var(--input-text, #e6e8ec); cursor: pointer; font: inherit; font-size: 12px; font-weight: 600; }
.koolook-modal-btn:hover { background: rgba(255,255,255,0.1); }
.koolook-modal-btn-primary { background: rgba(80,140,235,0.35); border-color: rgba(80,140,235,0.55); }
.koolook-modal-btn-primary:hover { background: rgba(80,140,235,0.5); }
.koolook-modal-btn-danger { background: rgba(220,80,80,0.25); border-color: rgba(220,80,80,0.5); }
.koolook-modal-btn-danger:hover { background: rgba(220,80,80,0.4); }
.koolook-modal-btn:disabled { opacity: 0.45; cursor: default; background: var(--comfy-input-bg, rgba(0,0,0,0.3)); border-color: rgba(255,255,255,0.1); }
.koolook-modal-btn:disabled:hover { background: var(--comfy-input-bg, rgba(0,0,0,0.3)); }
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
.koolook-snapshot-list { max-height: 280px; overflow-y: auto; border: 1px solid var(--border-color, #302f2f); border-radius: 4px; background: var(--comfy-input-bg, #111111); overflow: hidden auto; }
/* Recovery auto-saves section in the Load dialog. The collapsed summary is
   always visible at the bottom of the body; clicking a preset with a newer
   autosave opens one scoped group inside this container. */
.koolook-recovery-section { margin-top: 14px; border: 1px solid var(--border-color, #302f2f); border-radius: 4px; background: rgba(255,255,255,0.02); background: color-mix(in srgb, var(--comfy-menu-bg, #151515) 92%, var(--comfy-input-bg, #111111)); overflow: hidden; }
.koolook-recovery-summary { cursor: pointer; font-size: 12px; color: var(--input-text, #d8dadd); padding: 10px 12px; user-select: none; outline: none; }
.koolook-recovery-summary-passive { cursor: default; }
.koolook-recovery-summary:hover { opacity: 1; }
/* No outer border on the recovery list — each group carries its own
   bordered rows-list, mirroring the library section's structure. We
   keep the scroll container so a library with many autosave subdirs
   does not blow out the modal.
   IMPORTANT: this whole CSS string lives inside a JS template literal
   (backtick-delimited), so do NOT use literal backticks in comments
   here — they would prematurely close the template literal in ES
   module mode and break the entire constants.js module load (and with
   it the whole sidebar). Use plain quotes or no quotes at all. */
.koolook-recovery-list { margin-top: 0; max-height: 240px; overflow-y: auto; border-top: 1px solid rgba(255,255,255,0.04); }
.koolook-recovery-group { padding: 10px 12px; }
.koolook-recovery-group-head { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; margin-bottom: 6px; }
.koolook-recovery-group-title { color: var(--input-text, #e6e8ec); font-size: 12px; font-weight: 700; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.koolook-recovery-group-path { color: var(--input-text, #8f959c); opacity: 0.68; font: 11px/1.35 ui-monospace, "Cascadia Mono", Menlo, monospace; margin-bottom: 6px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.koolook-recovery-group + .koolook-recovery-group { margin-top: 10px; }
.koolook-recovery-kind { display: inline-flex; align-items: center; width: fit-content; font-size: 10px; line-height: 1.2; text-transform: uppercase; letter-spacing: 0.06em; padding: 2px 6px; border-radius: 3px; font-weight: 700; margin-bottom: 6px; }
.koolook-recovery-kind-pre_load { background: rgba(255,184,77,0.16); color: #f5d3a0; border: 1px solid rgba(255,184,77,0.35); }
.koolook-recovery-kind-periodic { background: rgba(109,180,255,0.16); color: #b3d4f5; border: 1px solid rgba(109,180,255,0.35); }
.koolook-recovery-row { display: flex; align-items: center; gap: 8px; padding: 10px 12px; border: 1px solid var(--border-color, #302f2f); border-radius: 3px; background: var(--comfy-input-bg, #111111); }
.koolook-recovery-row-selected { background: rgba(80,140,235,0.16); background: color-mix(in srgb, var(--comfy-input-bg, #111111) 88%, var(--p-primary-color, #6db4ff)); }
.koolook-recovery-row-info { flex: 1; min-width: 0; cursor: pointer; font-size: 12px; }
.koolook-recovery-row-info:hover .koolook-recovery-row-meta { color: var(--p-primary-color, rgba(120,170,255,1)); }
.koolook-recovery-row-meta { color: var(--input-text, #8f959c); opacity: 0.68; font-size: 11px; line-height: 1.35; }
.koolook-snapshot-row { display: flex; align-items: center; gap: 8px; padding: 10px 12px; border-bottom: 1px solid var(--border-color, #302f2f); background: var(--comfy-input-bg, #111111); }
.koolook-snapshot-row-selected { background: rgba(80,140,235,0.16); background: color-mix(in srgb, var(--comfy-input-bg, #111111) 88%, var(--p-primary-color, #6db4ff)); }
.koolook-snapshot-row:last-child { border-bottom: none; }
.koolook-snapshot-row-info { flex: 1; min-width: 0; cursor: pointer; }
.koolook-snapshot-row-info:hover .koolook-snapshot-row-name { color: var(--p-primary-color, rgba(120,170,255,1)); }
.koolook-snapshot-row-name { color: var(--input-text, #e6e8ec); font-size: 13px; font-weight: 700; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.koolook-snapshot-row-meta { color: var(--input-text, #8f959c); opacity: 0.68; font-size: 11px; margin-top: 2px; }
.koolook-snapshot-row-actions { display: flex; gap: 4px; flex-shrink: 0; }
.koolook-snapshot-row-btn { min-width: 26px; height: 24px; padding: 0 8px; background: var(--comfy-input-bg, #1a1a1f); border: 1px solid var(--border-color, #4f4f54); border-radius: 4px; cursor: pointer; color: var(--input-text, #e6e8ec); font-size: 16px; font-weight: 800; line-height: 1; }
.koolook-snapshot-row-btn:hover { background: rgba(255,255,255,0.08); }
.koolook-snapshot-row-btn-danger:hover { background: rgba(74,5,5,0.45); border-color: rgba(255,138,138,0.45); color: #ffd9d9; }
.koolook-snapshot-empty { padding: 20px; opacity: 0.5; font-size: 12px; text-align: center; }
.koolook-mode-toggle { display: inline-flex; gap: 0; margin-right: 3px; border: 1px solid var(--border-color, rgba(255,255,255,0.1)); border-radius: 3px; overflow: hidden; }
.koolook-mode-toggle-btn { padding: 0; width: 28px; height: 17px; display: inline-flex; align-items: center; justify-content: center; box-sizing: border-box; background: transparent; border: none; cursor: pointer; color: rgba(218,221,226,0.78); font-size: 11px; opacity: 1; line-height: 1; }
.koolook-mode-toggle-btn:hover { background: rgba(255,255,255,0.06); color: rgba(248,249,250,0.96); }
.koolook-mode-toggle-btn.koolook-mode-active { background: rgba(80,140,235,0.25); color: rgba(248,249,250,0.96); }
.koolook-leaf-unresolved { opacity: 0.55; font-style: italic; }
.koolook-leaf-crumb { opacity: 0.5; font-size: 11px; margin-right: 1px; }
.koolook-pack-badge { opacity: 0.5; font-size: 11px; margin-left: 6px; flex-shrink: 0; white-space: nowrap; }
.koolook-preview-card { position: fixed; z-index: 10000; background: #353535; border: 1px solid rgba(190,190,190,0.38); border-radius: 12px; box-shadow: 0 6px 24px rgba(0,0,0,0.55); color: var(--input-text, #ddd); font-size: 12px; width: 300px; max-width: calc(100vw - 16px); box-sizing: border-box; max-height: calc(100vh - 16px); pointer-events: none; overflow: hidden auto; padding-bottom: 10px; }
.koolook-preview-header { display: flex; align-items: center; gap: 9px; padding: 8px 13px 7px; font-size: 14px; line-height: 1; white-space: nowrap; overflow: hidden; background: rgba(255,255,255,0.045); border-bottom: 1px solid rgba(255,255,255,0.07); }
.koolook-preview-headtitle { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; min-width: 0; }
.koolook-preview-header-unresolved .koolook-preview-headtitle { font-style: italic; opacity: 0.75; }
.koolook-preview-headdot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; box-shadow: 0 0 0 1px rgba(0,0,0,0.4); }
.koolook-preview-badge { padding: 0 13px 7px; color: #ff5f6d; font-size: 11px; letter-spacing: 0.08em; font-weight: 600; text-transform: uppercase; }
.koolook-preview-stub { margin: 4px 10px 6px; padding: 8px 10px; background: rgba(0,0,0,0.18); border-radius: 6px; font-style: italic; opacity: 0.78; line-height: 1.4; }
.koolook-preview-row { display: grid; grid-template-columns: 14px 1fr 14px 1fr 14px; column-gap: 8px; align-items: center; padding: 0 9px; line-height: 1.55; min-height: 18px; }
.koolook-preview-row-slot { padding: 2px 9px; }
.koolook-preview-col { overflow: hidden; min-width: 0; }
.koolook-preview-col-input { text-align: left; text-overflow: ellipsis; white-space: nowrap; }
.koolook-preview-col-output { text-align: right; text-overflow: ellipsis; white-space: nowrap; }
.koolook-preview-col-middle { /* spacer, intentionally empty */ }
.koolook-preview-slot-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; box-shadow: 0 0 0 1px rgba(0,0,0,0.4); vertical-align: middle; }
.koolook-preview-slot-optional { opacity: 0.7; font-style: italic; }
.koolook-preview-row-widget { background: rgba(0,0,0,0.24); border: 1px solid rgba(255,255,255,0.14); margin: 5px 6px 0; border-radius: 9px; padding: 0 6px; line-height: 1.65; }
.koolook-preview-arrow { color: rgba(255,255,255,0.6); font-size: 9px; text-align: center; user-select: none; }
.koolook-preview-widget-name { text-align: left; text-overflow: ellipsis; white-space: nowrap; }
.koolook-preview-widget-value { text-align: right; font-size: 11px; opacity: 0.85; font-family: monospace; text-overflow: ellipsis; white-space: nowrap; }
.koolook-preview-desc { margin: 10px 9px 0; padding: 7px 9px; background: rgba(0,0,0,0.18); border: 1px solid rgba(255,255,255,0.05); border-radius: 6px; font-style: italic; font-weight: 500; font-size: 11px; line-height: 1.45; word-break: break-word; opacity: 0.86; max-height: 120px; overflow: hidden; }
.koolook-build-tag { flex-shrink: 0; padding: 6px 10px 14px; font-size: 10px; opacity: 0.5; text-align: center; letter-spacing: 0.04em; font-family: monospace; color: var(--input-text, inherit); line-height: 1.45; }
.koolook-build-sha { font-size: 13px; letter-spacing: 0.06em; }
.koolook-build-scope { display: block; font-size: 12px; margin-top: 3px; letter-spacing: 0.02em; font-family: var(--font-family, sans-serif); font-style: italic; }
/* Folder-browse picker (issue #137, mockup section 6). Navigate-into
   model: path input shows current location, clicking a folder row
   drills in, the Up button climbs one level. Files appear greyed so
   the user can confirm 'yes, this is the folder I expected' before
   committing.
   The picker reuses the modal shell (overlay, title, action row);
   only the body chrome (toolbar / list / row styling) is folder-
   picker-specific. Reminder per the recovery-list comment above:
   this whole CSS string is a JS template literal, so do NOT use
   backticks in comments here — they prematurely close the template
   literal and break the entire sidebar at load. */
.koolook-folder-picker { min-width: 480px; max-width: 600px; }
.koolook-folder-picker-toolbar { display: flex; gap: 8px; align-items: center; margin: 0 0 10px; }
.koolook-folder-picker-up { flex-shrink: 0; }
/* End-visible path overflow: 'direction: rtl' makes the input's
   overflow-axis right-to-left, 'unicode-bidi: plaintext' keeps the
   Latin path characters in natural left-to-right order. Net effect:
   long paths clip the LEFT (leading /Users/...) and always show the
   trailing folder name on the right. Per mockup section 6. */
.koolook-folder-picker-path { flex: 1; min-width: 0; direction: rtl; text-align: left; unicode-bidi: plaintext; font-family: ui-monospace, "Cascadia Mono", Menlo, monospace; font-size: 12px; }
.koolook-folder-picker-list { max-height: 320px; min-height: 120px; overflow-y: auto; border: 1px solid var(--border-color, #302f2f); border-radius: 4px; background: var(--comfy-input-bg, #111111); overflow: hidden auto; }
.koolook-folder-picker-row { display: flex; align-items: center; gap: 8px; width: 100%; text-align: left; padding: 10px 12px; border: 0; border-bottom: 1px solid var(--border-color, #302f2f); background: var(--comfy-input-bg, #111111); color: var(--input-text, #e6e8ec); cursor: pointer; font: inherit; font-size: 12px; }
.koolook-folder-picker-row:last-child { border-bottom: none; }
.koolook-folder-picker-row:hover { background: rgba(80,140,235,0.16); background: color-mix(in srgb, var(--comfy-input-bg, #111111) 88%, var(--p-primary-color, #6db4ff)); }
.koolook-folder-picker-row-file { cursor: default; color: var(--input-text, #8f959c); opacity: 0.68; }
.koolook-folder-picker-row-file:hover { background: var(--comfy-input-bg, #111111); }
.koolook-folder-picker-icon { font-size: 13px; line-height: 1; flex-shrink: 0; }
.koolook-folder-picker-row:not(.koolook-folder-picker-row-file) .koolook-folder-picker-icon { filter: hue-rotate(-30deg) saturate(1.4); }
.koolook-folder-picker-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.koolook-folder-picker-empty { padding: 20px; color: var(--input-text, #8f959c); opacity: 0.68; font-size: 12px; font-style: italic; text-align: center; }
.koolook-folder-picker-error { color: #ffae9a; font-style: normal; opacity: 0.9; }
.koolook-folder-picker-spacer { flex: 1; }
.koolook-folder-picker-newfolder-label { color: var(--input-text, #8f959c); opacity: 0.68; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; flex-shrink: 0; }
.koolook-folder-picker-newfolder-input { flex: 1; min-width: 0; }
/* Save dialog redesign (issue #137, mockup section 2). The library row
   at the top is purely informational — label + Open folder link sit on
   the top edge so a long absolute path can't overlap the link. Every
   action lives in the bottom command bar (Save to... | Cancel | Save
   as new... | Save). */
.koolook-snap-lib-row { padding: 10px 12px; border: 1px solid var(--border-color, #302f2f); border-radius: 4px; background: var(--comfy-input-bg, #181818); margin-bottom: 14px; }
.koolook-snap-lib-row-info { min-width: 0; }
.koolook-snap-lib-row-top { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; margin-bottom: 2px; }
.koolook-snap-lib-label { color: var(--input-text, #8f959c); opacity: 0.68; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }
.koolook-snap-open-folder-link { font-size: 11px; opacity: 1; color: #6db4ff; text-decoration: none; white-space: nowrap; }
.koolook-snap-open-folder-link:hover { opacity: 1; text-decoration: underline; }
.koolook-snap-save-message { margin: 6px 0 0; }
/* Four-state primary button (mockup section 5). The in-progress and
   done states sit on different button classes so the colour shift
   reads at a glance: default+dirty are primary blue (action), in-
   progress is dimmed primary (busy), done is subtle grey (confirmed,
   no further action expected). Disabled state is enforced via the
   'disabled' attribute, not just the class — keyboard activation
   short-circuits naturally. */
.koolook-snap-save-in-progress { opacity: 0.7; cursor: progress; }
.koolook-snap-save-done { background: rgba(120, 200, 120, 0.18); border-color: rgba(120, 200, 120, 0.5); color: #b3e3b3; opacity: 0.95; cursor: default; }
.koolook-delete-confirm-text { align-self: center; color: #ffb1b1; font-size: 12px; font-weight: 600; }
/* Inline delete state outlines the target row in red. Reminder: no
   backticks in comments inside this CSS template literal. */
.koolook-snapshot-row-pending-delete { outline: 1.5px solid rgba(220, 80, 80, 0.75); outline-offset: -1px; background: rgba(220, 80, 80, 0.06); }
/* Compare mode (issue 181): two live sidebars side by side, plus a bottom
   status bar. The columns reuse the normal koolook-sidebar render verbatim.
   Reminder: no backticks in comments inside this CSS template literal. */
.koolook-compare-host { display: flex; flex-direction: column; height: 100%; min-height: 0; }
.koolook-compare-split { display: flex; flex: 1 1 auto; gap: 8px; min-height: 0; overflow: auto; padding: 0 4px; }
.koolook-compare-col { flex: 1 1 0; min-width: 0; }
.koolook-compare-status { flex: 0 0 auto; padding: 5px 10px; font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; text-align: center; color: rgba(170, 200, 255, 0.95); background: rgba(80, 140, 235, 0.18); border-top: 1px solid var(--border-color, rgba(255, 255, 255, 0.1)); }
/* Text-only diff tint, applied only to the comparison column's rows. */
.koolook-cmp-new .koolook-name { color: #7be08a; }
.koolook-cmp-diff .koolook-name { color: #ff7d7d; }
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
// Critical toast — sticky variant for persist failures (server outage,
// localStorage rejected, both backends down). Distinct from `toast()` because:
//   • Auto-dismiss is wrong here — these are user-action-required signals,
//     not informational. The user MUST acknowledge.
//   • A "Copy details" button lets the user grab the unsaved JSON so a real
//     loss is recoverable manually (paste into a snapshot, or into devtools).
//   • `pointer-events: auto` on this variant — the regular toast is non-
//     interactive on purpose so it doesn't block clicks behind it; critical
//     toasts need to BE interactive (the buttons).
//
// Returns `{dismiss}` so callers can programmatically remove the toast when
// the underlying condition resolves (e.g. server reachable again).
// =============================================================================
export function criticalToast(msg, { copyText, onDismiss, actions: extraActions } = {}) {
    const t = document.createElement("div");
    t.className = "koolook-toast koolook-toast-critical";
    // Stack subsequent critical toasts at the top so they don't overlap the
    // bottom-anchored regular toasts. Each gets a vertical offset based on
    // its sibling count at append time.
    const existing = document.querySelectorAll(".koolook-toast-critical").length;
    if (existing > 0) {
        t.classList.add("koolook-toast-critical-stack");
        t.style.top = `${30 + existing * 90}px`;
    }

    const msgEl = document.createElement("div");
    msgEl.className = "koolook-toast-msg";
    msgEl.textContent = msg;
    t.appendChild(msgEl);

    const actionRow = document.createElement("div");
    actionRow.className = "koolook-toast-actions";

    const close = () => {
        if (t.parentNode) t.remove();
        if (typeof onDismiss === "function") onDismiss();
    };

    // Caller-supplied recovery actions render BEFORE Copy / Dismiss so the
    // primary path is leftmost. Each action's `onClick` may return a Promise
    // and may return `{ dismiss: false }` to keep the toast open after a
    // user-cancelled sub-flow (e.g. cancelled confirm modal). Default behavior
    // on resolve is to close the toast — these are recovery actions, so a
    // successful click means the underlying condition is resolved and the
    // toast should self-extinguish.
    if (Array.isArray(extraActions)) {
        for (const action of extraActions) {
            const btn = document.createElement("button");
            btn.className = "koolook-toast-btn";
            if (action.primary) btn.classList.add("koolook-toast-btn-primary");
            const originalLabel = action.label;
            btn.textContent = originalLabel;
            btn.addEventListener("click", async () => {
                if (btn.disabled) return;
                btn.disabled = true;
                if (action.busyLabel) btn.textContent = action.busyLabel;
                try {
                    const result = await Promise.resolve(action.onClick());
                    if (result && result.dismiss === false) {
                        btn.disabled = false;
                        btn.textContent = originalLabel;
                    } else {
                        close();
                    }
                } catch (e) {
                    console.error("[Koolook] critical-toast action failed:", e);
                    btn.disabled = false;
                    btn.textContent = originalLabel;
                }
            });
            actionRow.appendChild(btn);
        }
    }

    if (copyText) {
        const copyBtn = document.createElement("button");
        copyBtn.className = "koolook-toast-btn";
        copyBtn.textContent = "Copy details";
        copyBtn.addEventListener("click", () => {
            const finishCopy = () => {
                copyBtn.textContent = "Copied ✓";
                setTimeout(() => { copyBtn.textContent = "Copy details"; }, 1500);
            };
            // Async clipboard API first — works in secure contexts (https +
            // localhost). Fall back to an offscreen textarea + execCommand
            // for environments where clipboard.writeText rejects (some
            // browser sandboxes, http://192.168.x.y origins).
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(copyText).then(finishCopy).catch(() => {
                    legacyCopy(copyText);
                    finishCopy();
                });
            } else {
                legacyCopy(copyText);
                finishCopy();
            }
        });
        actionRow.appendChild(copyBtn);
    }

    const dismissBtn = document.createElement("button");
    dismissBtn.className = "koolook-toast-btn";
    dismissBtn.textContent = "Dismiss";
    dismissBtn.addEventListener("click", close);
    actionRow.appendChild(dismissBtn);

    t.appendChild(actionRow);
    document.body.appendChild(t);
    return { dismiss: close };
}

function legacyCopy(text) {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.cssText = "position:fixed;left:-9999px;top:-9999px;opacity:0;";
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    try { document.execCommand("copy"); } catch (e) { /* best effort */ }
    ta.remove();
}

// =============================================================================
// Sort helper — human-alphabetical (case-insensitive). Use everywhere we sort
// names, labels, or directory entries so the order in the sidebar matches what
// a user would expect from an A→Z list.
// =============================================================================
export function compareNames(a, b) {
    return String(a).localeCompare(String(b), undefined, { sensitivity: "base" });
}

// =============================================================================
// Library-path breadcrumb formatter — used by the Save / Load snapshot dialogs
// to surface "where am I saving to" without overflowing a single line. Strategy:
// keep the rightmost segments (the leaf is the most informative — "presets2"
// tells the user more than "e:\_AI\portable" does), snap to a path separator
// so we never truncate mid-segment, prefix with "…" when truncation happened.
// Pair the result with the full path in a `title=` attribute so hover reveals
// the unabbreviated form.
// =============================================================================
export function formatLibraryPathBreadcrumb(fullPath, maxChars = 50) {
    if (typeof fullPath !== "string" || !fullPath) return "";
    if (fullPath.length <= maxChars) return fullPath;
    // Reserve 1 char for the "…" prefix.
    const tail = fullPath.slice(-(maxChars - 1));
    // Snap forward to the first separator so we don't slice through a segment.
    // If no separator survives, fall back to the raw tail (very long single
    // segment is degenerate but shouldn't crash).
    const snapAt = tail.search(/[\\/]/);
    const cleanTail = snapAt >= 0 ? tail.slice(snapAt) : tail;
    return "…" + cleanTail;
}
