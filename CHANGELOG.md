# Changelog

All notable changes to this project should be documented in this file.

The format is inspired by Keep a Changelog and SemVer.

## [Unreleased]

### Changed
- **Sidebar node preview descriptions are now bounded.** Long node
  descriptions in the hover preview clamp to a compact bottom block instead
  of stretching the card and crowding out the slot grid/widget rows from
  #82's node-mock layout.
- **Sidebar second mode — "Theme" instead of "Category".** The sitemap-icon
  toggle in the Nodes action row now groups picks by **semantic theme**
  rather than by raw CATEGORY first-segment. The new algorithm strips the
  source pack's name from the front of each pick's CATEGORY (via the
  existing `findPackPathForType` precedence) before grouping, so
  `KJNodes/image/Get`, `Koolook/Image/EasyResize`, and `image/foo` all
  collapse to one top-level `image` folder. Each theme bucket renders as
  a flat sorted list — no further sub-folders. Picks-only (no
  REPOS-driven auto-pulled candidates) so the second mode genuinely
  reflects "my image-related favorites" rather than "every image-shaped
  node ComfyUI knows about." Synthetic buckets `(unresolved)` /
  `(uncategorized)` survive as before.
- **Pack-name badges removed from leaf rows.** The small dim labels
  (`Koolook`, `KJNodes`, `EasyUse`) that used to render at the right edge
  of every leaf in theme mode and in search-flatten are gone. The
  breadcrumb prefix on search-flatten rows already conveys origin in a
  cleaner form, and theme mode by definition doesn't care about source
  pack — the badge was redundant. `makeNodeLeafRow` no longer accepts a
  `packBadge` parameter; CSS rule `.koolook-pack-badge` removed from
  `constants.js`.

### Added
- **Modules — splice a saved cluster into your live canvas instead of
  replacing it.** Tag any saved workflow with the literal tag `module`
  and the Kforge Labs sidebar starts treating it as a reusable building
  block: a green `pi pi-plus-circle` icon replaces the file glyph,
  left-click **inserts** the cluster at the viewport center (with
  internal links re-created and the new nodes left selected), and a
  distinct hover tooltip surfaces the changed semantics. The
  selection-save modal now ships a **`Save as module`** checkbox
  (pre-checked for selection saves; unchecked for whole-canvas saves),
  which adds the `module` tag inside the same `persistMutation` as the
  entry write so a commit failure rolls back both. The right-click
  menu on every workflow row gains an explicit **Insert into canvas**
  entry alongside the existing Load — Load still works on every row,
  module-tagged or not. New primitive `insertWorkflowOntoCanvas` in
  `web/sidebar/canvas_io.js` pre-flights every referenced node type
  against `LiteGraph.registered_node_types` and aborts cleanly when
  any are missing (a partial insert with stub nodes is worse than no
  insert), then deep-clones the saved graph, lets `app.graph.add()`
  allocate fresh node ids so nothing collides with the live canvas,
  and recreates internal connections via `originNode.connect(...)` so
  link ids are auto-allocated too — no manual link remap needed.
  Designed for setups like an EXR-output stack, a depth-only render
  cluster, or a Wan 2.2 prompt module the user wants to drop into many
  workflows. Full reference in
  [`docs/maintainers/workflows-sidebar.md`](docs/maintainers/workflows-sidebar.md#modules--splice-a-saved-cluster-into-your-live-canvas).
- **Spotlight effect on add.** Clicking the toolbar `+` (or the canvas
  right-click "Add to Kforge Labs") now collapses every Nodes-section
  pack folder, then auto-expands just the pack + subcategory the just-saved
  node lives in — a pedagogical aid that helps new users internalize
  which pack each node belongs to. Multi-select adds light up every hit
  pack simultaneously. Duplicate adds (already in picks) trigger the
  same spotlight, since the educational reminder still applies. Helper
  `findPackPathForType` mirrors the gather code's REPO-precedence-then-
  user-pick-fallback to locate any node ID's sidebar path; new exported
  `spotlightAddedPicks(typeNames)` from `web/sidebar/tree.js` does the
  collapse + pin sequence and is called from both add paths.
- **Group-by-category mode for the Nodes section.** Segmented toggle in
  the Nodes action row (📦 Repository / 🌐 Category — `pi-database` /
  `pi-sitemap`); choice persists in localStorage (`koolook.groupMode.v1`)
  and survives reloads, including cross-tab via the existing `storage`
  event listener. In repo mode (default) every pick lives under its
  pack; in category mode picks are regrouped by their node-class
  `CATEGORY` path, ignoring REPOS affiliation. Categories that look the
  same after canonical-key normalization
  (`lower().replace(/[\s_\-]+/g, "")`) collapse into one folder —
  `Loaders`/`loaders`, `Image/Upscaling`/`image/upscaling`, and
  `style_model`/`StyleModel`/`style model` all merge. Display label is
  the most-common original casing seen for that key (ties → first-seen,
  stable across renders because Map iteration follows insertion order).
  Ports of the same path under different repos all collapse together,
  so a Loader from Pack A and a Loader from Pack B share the
  `Loaders/` folder. Two synthetic top-level buckets surface the edge
  cases: `(unresolved)` for picks whose pack isn't currently loaded
  (repo mode silently dropped these — category mode keeps them visible
  as italic-dim rows), `(uncategorized)` for picks whose node class has
  no `CATEGORY`. Each leaf in category mode carries a small pack-name
  badge so users still see "where did this come from" — load-bearing
  for the existing `↓ Install missing` flow. Folder paths in
  `pathStates` use the canonical key (not the resolved display label),
  so an upstream casing change to the most-common label doesn't reset
  user expansions. Closes the "sort by topic, by what they do" piece
  of #46. (#73)
- **Flatten-on-search for the Nodes section.** When the search field is
  non-empty, the Nodes section drops its tree structure and renders a
  flat list of matching leaves, sorted by display name, each with a
  small dim-grey breadcrumb prefix (`Loaders › LoRA › LoraLoader` in
  category mode, `Pack › Subcategory › Display` in repo mode) so the
  spatial-origin signal survives. Workflows and Tags sections retain
  their tree-under-filter behavior — the change is scoped to Nodes
  only. The breadcrumb collapses redundant synthetic labels: the
  `(root)` subcategory (which adds no info beyond the pack name) and
  any subcategory whose label duplicates the pack label (the
  "(uncategorized)" double-up). Always-flatten on any non-empty query
  is the deliberate choice over a count-threshold gate; the spatial
  cue from a tree is mostly already lost once a search has narrowed
  the set, and a flat sorted list scans faster. (#73)
- **Hover preview card for sidebar leaf rows.** Mirrors ComfyUI's
  official Node Library preview ([NodePreview.vue][np-vue]): plain
  HTML/CSS card with a colored title bar (HSL hue hashed from the
  `CATEGORY` string), category breadcrumb, two columns of type-colored
  slot dots (inputs left, outputs right), widgets section with
  truncated defaults, optional description. Hover a leaf for ~250ms to
  show; mouseleave dismisses; only one card visible at a time
  (module-level singleton). Card positions to the right of the row by
  default (12px offset), flips to the left when it'd overflow the
  viewport, clamps inside an 8px viewport-padding gutter. Tall cards
  (a node with many inputs) cap at `calc(100vh - 16px)` and scroll
  internally rather than clip. `pointer-events: none` on the card so
  it never intercepts events from rows it floats over. Slot dot
  colors read ComfyUI's runtime palette
  (`app.canvas.default_connection_color_byType`, populated at canvas
  init) first, then fall back to LiteGraph's static
  `LGraphCanvas.link_type_colors` (mostly empty in stock LiteGraph),
  then a neutral grey. Widget classification mirrors ComfyUI's
  frontend rule: scalars (`INT`/`FLOAT`/`STRING`/`BOOLEAN`) and
  arrays-of-choices (`COMBO`) become widgets, everything else is a
  connection slot. `INPUT_TYPES()` is read defensively (some
  custom-node packs throw under unusual conditions); when one does
  throw, the failure is logged via `console.warn` so a broken pack is
  visible during debugging. Picks whose pack isn't loaded show a
  "Pack not loaded" stub card pointing at the Tools-row install
  button. The hover card additionally tears down on
  `visibilitychange` (page hidden) and `window.blur` so it doesn't
  leak across tab switches or focus loss. New module
  `web/sidebar/node_preview.js` holds the preview engine;
  `attachHoverPreview(row, type)` is wired inside `makeNodeLeafRow`
  so every leaf-emit site (repo tree, category tree, search-flat) gets
  the preview for free. The renderer calls `teardownPreview()` before
  each `treeEl.innerHTML = ""` — without it, an active card whose
  anchor row gets detached during a re-render would leak (the
  `pointer-events: none` card can't be dismissed by clicking it).
  (#73)

[np-vue]: https://github.com/Comfy-Org/ComfyUI_frontend/blob/main/src/components/node/NodePreview.vue

### Changed
- **Hover preview rebuilt to read like a real node mock.** The original
  layout (HSL-hashed title strip + two-column slot table + flat widget
  text list) read as a generic info card. Reworked to mirror upstream
  `NodePreview.vue` more faithfully: a flat header row (small colored
  dot + node display name) over a red **PREVIEW** badge, slot rows
  laid out as a 5-column grid (`[dot] [input-name] [spacer] [output-name] [dot]`)
  so inputs and outputs line up horizontally as parallel sockets, and
  widgets rendered as individual rounded pills with `◀ name [spacer]
  value ▶` chrome to mimic LiteGraph widget UI. **Bigger fix
  underneath:** `readSlots` and the title / category / description
  reads were looking at static `nodeClass.INPUT_TYPES` /
  `RETURN_TYPES` properties, which **don't exist** on
  ComfyUI-registered nodes — ComfyUI parks the original V1 def at
  `nodeClass.nodeData` (set by `litegraphService.ts`'s
  `registerNodeDef` right before `LiteGraph.registerNodeType`). Every
  card was rendering as just-the-header-and-badge because the slot
  arrays came back empty. Reading from `nodeData` first (with the old
  property names as a legacy fallback for non-ComfyUI registered
  nodes) fixes the empty-card class. (#78)

### Fixed
- **`pinExpanded` paths now expand on the immediate render** instead of
  being delayed by one. `buildFolder`'s expansion-resolution chain gained
  a `pinnedPaths` check between `forceExpanded` and `pathStates`. Phase 3
  of `renderTree` writes pins into `pathStates` AFTER Phase 2 has built
  the DOM, so without the new check the pin had no effect on the current
  render — only on the next one. Side-effect-free correctness improvement
  for the workflow-save pinning that already existed; load-bearing for
  the new spotlight feature above which depends on pinned paths
  expanding immediately.
- **Modal drag-out-of-input no longer dismisses the dialog.** The shared
  modal shell (`makeModalShell`) used by every Koolook dialog gained a
  mousedown / click-intent check: a click on the overlay now only fires
  the dismiss when the gesture *started* on the overlay too. Drag-
  selecting text inside an input field that releases in the overlay's
  dark area used to trigger `click` with `target=overlay` and dismiss
  the modal mid-edit; now it stays open. Affects all dialogs including
  Snapshot Save / Load / Settings, Install Missing, Save Workflow, Tags,
  Confirm, Input — load-bearing especially for the Snapshot Settings
  path field where the long absolute path encourages drag-selection.
- **Snapshot Settings path field shows full path on hover.** Added a
  `title` attribute synced to the saved library path on every refresh
  and save, so the user can read the full path without scrolling /
  selecting inside the narrow input.

### Added (continued)
- **"Install missing for picks" toolbar button** in the Nodes section
  (`pi-cloud-download` icon, next to Add and Export). Walks the user's
  picks against ComfyUI-Manager's `/customnode/getmappings` mapping,
  buckets into already-installed / will-install / unresolved, queues
  unique git URLs through `/customnode/install/git_url`, polls
  `/manager/queue/status` to drive a progress bar, and prompts to
  reboot. Works on any install with ComfyUI-Manager loaded; falls back
  to a clipboard URL list (with a `comfy node install` hint) if Manager
  isn't reachable. New module `web/sidebar/installer.js` holds the
  Manager-API client + resolver; the modal in `web/sidebar/modals.js`
  drives the four-phase UI (discovery → confirm → progress → result).
  403s from Manager's security gate are surfaced as actionable language
  ("your security level forbids git-URL installs") rather than raw HTTP.
- **Snapshot library** — save your full Kforge Labs state (curated picks
  + the entire workflows store including tags + archive) as a named
  preset to a configurable filesystem path. New top-level **Snapshot**
  action row above the search field with three icon buttons, each
  opening its own focused dialog:
  - **Save (cloud-up icon)** — one click. If a preset is currently
    loaded, prompts "Save over '<name>'?" with three options
    (**Save** overwrite / **Save as new…** rename / **Cancel**). If
    no preset is loaded yet, prompts for a name (default
    `preset YYYY-MM-DD`, fully editable). The "current preset" is
    tracked in localStorage and persists across reloads, so Save
    keeps doing the obvious thing across sessions.
  - **Load (cloud-down icon)** — single dialog. Lists every preset
    in the library with metadata (workflow count · pick count · export
    date). Click a row → confirm "Replace current state?" → restores
    picks + workflows. Each row has an **×** button for delete (with
    confirm). Header line shows the current library path so you can
    see where you're loading from.
  - **Settings (cog icon)** — single field for the library's
    filesystem path. Save writes to a per-install settings file the
    server reads. **Reset to default** clears the saved path so the
    server falls back to env-var or built-in default. A read-only
    line shows the currently-resolved path + source (settings panel
    / env var / built-in default), so what's in effect right now is
    always visible.
- **Configurable storage location.** Resolution chain (highest first):
  1. Path saved via the Settings dialog
     (`<comfyui-userdata>/koolook-settings.json`'s `libraryPath`).
  2. `KFORGELABS_PRESETS` env var (deployment / facility config).
  3. Built-in default `<comfyui-userdata>/koolook-presets/`.
  Use cases:
  - **Personal cross-machine sync:** point at a Dropbox/iCloud/Drive
    folder; the library follows your machines.
  - **Facility shared library:** point at an NFS/SMB mount writable
    by every workstation; all workstations save to + load from the
    same library natively, no symlinks.
  - **Read-only distribution:** point at a path the workstation can
    read but not write; save fails cleanly with the server reason in
    a toast, load works.
  Snapshot files carry a `kind: "koolook-snapshot"` discriminator +
  `version` field for future schema migrations. Closes the "save to
  a custom location, take it elsewhere" piece of #46.

### Internal
- New server-side module `koolook_routes.py` registers the
  `/koolook/presets/*` aiohttp routes on ComfyUI's PromptServer.
  Endpoints: `info`, `list`, `file` (GET/POST/DELETE on a single
  query-string-keyed endpoint, with the GET handler short-circuiting
  HEAD requests via aiohttp's auto-HEAD-from-GET so existence checks
  don't read the full file body), and `settings` (GET/POST). The
  HEAD short-circuit lives inside the GET handler rather than as a
  dedicated `@routes.head` registration because ComfyUI's
  mirror-to-`/api` code in `server.py` blindly forwards
  `RouteDef.kwargs` into a `RouteTableDef.route(...)` closure that
  rejects `allow_head=False`, so any kwarg-based opt-out crashes
  startup. Path-traversal protection at the route boundary via a
  strict filename whitelist regex; symlink protection via
  post-resolve `is_relative_to` check on every file op. Settings
  file is written atomically (`tmp + os.replace`) so an interrupted
  process can't truncate the user's saved library path. The library
  directory is auto-created on first save.
- `replaceAllWorkflows` now uses the same `snapshotCache` rollback
  primitive as `persistMutation` so a snapshot apply that fails to
  persist rolls the in-memory cache back to its pre-call state — the
  load is fully atomic. The `workflows_store.js` mutator-invariants
  doc block lists `replaceAllWorkflows` as the fourth legitimate
  rebind site.
- The Load dialog now gates the `currentPresetName` tracker on
  `picksOk && workflowsOk`. Partial-failure paths clear the tracker
  so the next Save forces a fresh name prompt rather than offering
  to overwrite the on-disk preset with corrupted half-state.
- The Load dialog clears the tracker if the user deletes the
  currently-loaded preset.
- Client-side `sanitizeName` now mirrors the server's filename
  whitelist regex — invalid characters collapse to `_` rather than
  hitting an opaque HTTP 400 from the server. The `presetExists`
  probe is now tri-state (true/false/null); the Save flow refuses
  to write when it can't reach the library to verify the name.
- Server error reasons are surfaced via `await resp.text()` rather
  than `resp.statusText`, since HTTP/2 (RFC 7540) strips reason
  phrases — behind any HTTP/2-terminating proxy `statusText` is
  empty and the server's helpful "read-only mount" / "invalid
  filename" / "parent missing" messages would otherwise be lost.

### Changed
- **Save selection toast distinguishes "no selection" from "selection
  points at deleted nodes."** Previously both produced the generic
  "Select one or more nodes on the canvas first." Now the deleted-node
  case (selection set non-empty but every id refers to a removed node,
  common after undo/redo) shows "Selected node(s) no longer exist.
  Click a node on the canvas to re-select." `serializeSelection`
  returns a discriminated `{ kind: "empty" | "stale" | "ok", graph? }`
  result so the caller can route messaging precisely.
- **`curated_defaults.json` retired in favor of `starter_preset.json`.**
  Fresh installs no longer get picks seeded directly into localStorage
  on first load. Instead the bundled `web/starter_preset.json` (full
  snapshot format — picks + workflows + tags + archive) is copied into
  the user's snapshot library directory as `starter.json`, and the user
  opens Snapshot → Load to apply it in one click. The change unifies
  fresh-install distribution with the per-user snapshot library
  (#68): the maintainer flow becomes "build state in ComfyUI → click
  the ↓ Tools-row button → paste over `web/starter_preset.json` →
  commit." Existing users with non-empty picks are explicitly skipped
  by the new seeder, so their state is untouched. Removed:
  `seedDefaultsIfNeeded`, `exportPicks`, `web/curated_defaults.json`,
  `SEEDED_KEY`, `DEFAULTS_URL`. Added: `seedStarterPresetIfNeeded`,
  `exportStarterPreset` (both in `web/sidebar/snapshot.js`),
  `STARTER_SEEDED_KEY`, `STARTER_URL`, `STARTER_PRESET_FILENAME`,
  `web/starter_preset.json`.
- **Sidebar toolbar: Tools row split out from Nodes row.** The Export
  button (now "Export starter preset") and the "Install missing for
  picks" button moved up out of the Nodes row into a new dedicated
  **Tools** row above the search field, alongside a new "Drop
  placeholders onto canvas" button. The Nodes row keeps only the
  everyday `+` Add button. Reasoning: Export and Install-missing are
  admin/advanced operations; segregating them keeps the daily flow
  uncluttered and makes it harder to fire them by accident. The new
  Drop-placeholders button is the `security_level=normal` escape hatch
  for "install missing" — it instantiates one placeholder per missing
  pack on a fresh canvas tab so ComfyUI/Manager's standard "Install
  Missing Custom Nodes" detection picks them up and routes the install
  through Manager's UI flow (which doesn't go through
  `/customnode/install/git_url`'s security gate the way our
  programmatic call does).

### Internal (sidebar tidiness pass — no behavior change)
- **Lifted save-modal action sentinels and the cascade picker
  sentinels into named constants** at the top of `showSaveWorkflowModal`
  in `web/sidebar/modals.js` (`ACTION_NEW`, `ACTION_USE_EXISTING`,
  `ACTION_MODIFY_EXISTING`, alongside the existing `NEW_TOP` /
  `SAVE_HERE`). Magic strings (`"new"`, `"use_existing"`,
  `"modify_existing"`) are gone from the function body.
- **Section-id constants** (`SECTION_ID_NODES`, `SECTION_ID_WORKFLOWS`,
  `SECTION_ID_TAGS`) lifted to module scope in `web/sidebar/tree.js`.
  The `pinExpanded` save-flow callsite now references the constant
  instead of the literal `"workflows"` string, so a future section-id
  rename only needs the constant updated.
- **`modalLabel(text)`** factory in `web/sidebar/modals.js` replaces
  the seven repetitions of the four-line `createElement("label")` +
  `className` + `textContent` + `appendChild` pattern.
- **`USERDATA_OVERWRITE_QUERY = "?overwrite=true"`** named constant in
  `web/sidebar/workflows_store.js`. ComfyUI's userdata API requires
  this flag to allow POST over an existing file; pinning it as a
  constant makes the contract visible at a glance.
- **`buildFolder` no longer round-trips folder-expansion state through
  `wrapper.dataset.expanded`.** The DOM dataset forces every value
  to a string (and led to the slightly awkward `!== "false"` read);
  expansion state now lives in a closure variable. `pathStates` is
  the canonical store for cross-render persistence — the dataset was
  just mirroring it for reads.
- **`subcategoryFor` fallback path now logs a `console.warn`** with
  the offending category and `categoryRoot`. Reachable only via a
  logic error (typically a stale `REPOS` entry whose `categoryRoot`
  no longer matches the upstream node category prefix); previously
  it silently rewrote the category, masking the misconfig.
- **Save-modal `onSave` callback contract renamed `dir` → `dirPath`**
  to match the codebase-wide convention (`dirPath` for arrays of
  segments, `dirName` for single segments, `dir` for resolved
  DirNode objects). Caller updates in `tree.js` paired.
- **`makeToolbarButton({iconClass, title, onClick})`** factory in
  `web/sidebar/tree.js` replaces the four near-identical button-construction
  blocks in `renderPanel` (export, new-dir, save-canvas, save-selection).
  Each block was 5–7 lines of `createElement` / `className` / `innerHTML`
  / `title` / `addEventListener`; calls collapse to 4-line factory invocations.
  No behavior change. Closes the "extract `makeToolbarButton`" item on #47.
- **`makeToolbarButton({iconClass, title, onClick})`** factory in
  `web/sidebar/tree.js` replaces the four near-identical button-construction
  blocks in `renderPanel` (export, new-dir, save-canvas, save-selection).
  Each block was 5–7 lines of `createElement` / `className` / `innerHTML`
  / `title` / `addEventListener`; calls collapse to 4-line factory invocations.
  No behavior change. Closes the "extract `makeToolbarButton`" item on #47.
- **Dropped duplicate `.koolook-export-btn` CSS class.** It was identical to
  `.koolook-icon-btn` minus the `:disabled` state. The export button now
  uses only `.koolook-icon-btn` via `makeToolbarButton`. Closes the
  "drop duplicate CSS class" item on #47.
- **`buildFolder({path})` default removed.** `path` is now a required
  parameter; the previous `path = null` default was unreachable since every
  caller routes through `makeSectionCtx`, which always supplies a non-empty
  section-prefixed string. The runtime `if (path && …)` guard inside
  `buildFolder` is also dropped. Closes the "remove unreachable default"
  item on #47.
- **`workflowsCache` mutator invariants documented inline** above the public
  mutator block in `web/sidebar/workflows_store.js`. Four rules a future
  contributor needs before adding a new mutator: pair every mutate with a
  commit (or use `persistMutation`), return `false` for no-op vs. truthy for
  success, mutate in place, and never replace `workflowsCache` outside the
  seed/load paths. Closes the "document `workflowsCache` invariants" item
  on #47.

### Added
- **Right-click "Duplicate…"** on any workflow row in the sidebar tree.
  Opens a name modal pre-filled with `<name> (copy)`; saving deep-clones
  the source graph into a new entry in the same directory. The duplicate
  inherits the source's tags so the user's categorization carries over.
  Same-name duplicates fall through to the existing archive-on-collision
  behavior in `saveWorkflowEntry`. (Closes #58.)
- **Right-click "Tags…"** on any workflow row. Opens a chip-style modal
  to view, add, and remove the workflow's tags one at a time. Each
  edit fires its own `persistMutation` so changes survive a mid-edit
  close. (Part of #56.)
- **Tags sidebar section.** A new section between Workflows lists every
  tag in use across the active workflow tree. Each tag becomes a folder
  whose entries are the tagged workflows (sorted A→Z); click loads the
  workflow from its real directory. Archived workflows are filtered
  out of the section so the active view stays clean — their tags are
  still preserved on the entry, so a restore from the Archive folder
  brings them back. Search matches tag name OR workflow name. (Part
  of #56.)
- **Right-click "+ New directory…" / "+ New subdirectory under <path>…"**
  in the workflow row's Move-to flow. Both create a fresh directory and
  move the workflow into it as one atomic mutation; if the move can't
  land, the new directory is rolled back so the cache never leaks an
  empty orphan. (Closes #57.)
- **Recursive subdirectories under workflow directories.** Right-click any
  directory in the Workflows tree → "Create subdirectory…" to nest folders
  to arbitrary depth (e.g. `UP-scale / Type-A / Sharp`). Each nested
  directory behaves like a top-level one: it can hold workflows + an
  Archive subfolder + further subdirectories. The save modal directory
  picker is **cascading** (multi-step): pick a parent, then a child
  appears, then a grandchild, etc. — each child level has a `(save in
  "<path>")` option to stop drilling at that depth. The right-click
  workflow "Move to…" submenu lists every other path in the tree.
- **Delete archive (N) on the synthetic Archive folder.** Right-click the
  Archive folder under any directory → bulk-removes every archived
  workflow at that level in one confirm. Active workflows in the same
  directory are untouched.
- **Drag-and-drop in the workflows tree** (Tier 1 — moves only, no
  reordering). Drag a workflow onto a directory to move it. Drag a
  workflow onto an Archive folder to archive it (cross-directory drops
  move + archive in one go). Drag a directory onto another directory to
  nest it as a child. Cycle prevention rejects dropping a dir into
  itself or any of its descendants. Sort within a level stays
  alphabetical. (Custom ordering would be Tier 2.)
- **Schema is now recursive** — every directory node has a `workflows`
  object AND a `directories` object. Existing v0.2 stores load fine:
  `normalizeWorkflowsStore` treats a missing `directories` as `{}` and
  the rest of the code assumes it always exists post-normalization.

### Changed
- **Sidebar tab renamed:** "Curated Nodes" → **"Kforge Labs"**. Tooltip
  also updated. The tab id (`koolook.curatedNodes`) and the
  `app.registerExtension` name are unchanged so existing per-user tab
  state (pinning, ordering) is preserved.
- **Save modal — `Base on existing` candidates now walk leaf-up to root
  and dedupe (deepest wins).** Saving into an empty subdirectory like
  `UP-scale / seedvr2` no longer disables the existing-name actions —
  the modal pulls active workflows from every ancestor. Ancestor
  entries are labeled `<name>  ·  in <path>` so the candidate's
  source directory is unambiguous. The selected destination path is
  whatever the cascade resolves to — the ancestor source only seeds
  the workflow name; archive semantics still apply at the destination.
- **Save modal — `Action` dropdown is now hidden (not just disabled)
  when no base candidate exists** anywhere in the destination's
  ancestry. Disabled `<option>` elements render too subtly across
  browsers; the only useful path in that case is "type a fresh name
  and save," which the Workflow Name field below already provides.
  The underlying value is pinned to `new` so submit takes the
  by-name path.
- The right-click canvas-node menu item is now **"Add to Kforge Labs"**
  (was "Add to Curated Sidebar").
- Directory header counts in the workflows tree now show the total
  workflows in the **whole subtree** (active + archived + descendants),
  not just direct children. A parent with empty direct workflows but
  populated subdirectories no longer shows "0".

### Internal
- Workflow operations are now path-addressed: every mutator and lookup
  takes a `string[]` path (e.g. `["UP-scale", "Type-A"]`). The store's
  internal API: `addDirectory(parentPath, name)`,
  `renameDirectory(parentPath, old, new)`, `deleteDirectory(parentPath, name)`,
  `saveWorkflowEntry(path, wfName, graph)`, `archive/unarchive/rename/deleteWorkflow(path, wfName)`,
  `moveWorkflow(srcPath, wfName, dstPath)`, plus new helpers
  `listAllDirectoryPaths()` and `pathsEqual(a, b)`.
- Reserved-name check: subdirectory names cannot be `Archive`
  (case-insensitive) at any non-root level — collides with the synthetic
  Archive folder rendered for archived workflows.
- **Workflow entries gain optional `tags: string[]`.** `normalizeDirNode`
  trims, drops empties, and dedupes case-sensitively, so old entries
  without a tags field load as `tags: []` and the rest of the code can
  assume the field always exists. New mutators in `workflows_store.js`:
  `getWorkflowTags(path, wfName)`, `addTag(path, wfName, tag)`,
  `removeTag(path, wfName, tag)`. `getWorkflowGraph(path, wfName)` is
  now also exported so the duplicate flow can deep-clone without
  reaching into the cache directly.
- **`showTagsModal` in `modals.js`** — chip-row UI with add/remove
  callbacks, mirrors the existing `showInputModal` / `showConfirmModal`
  surface so the modal shell, escape-key teardown, and overlay-click
  dismissal stay centralized.

### Documentation
- Closed out issue #28 (de-vendor upstream code under `upscaler_FIX/`
  and `nuke_CAM_exporter/`) by adding the audit-trail layer that the
  v0.1.4 / v0.1.5 registry-hygiene cleanup deferred:
  - `forks/THIRD_PARTY.md` gained a "De-vendored upstream code" section
    listing all six untracked trees with former path, upstream (where
    pinned) and per-tree provenance notes. Preamble now also documents
    the publish-history finding that none of these files were ever in
    a successful Comfy Registry publish.
  - `forks/forks_manifest.yaml` gained six entries with the `_devendored`
    suffix and `status: "removed"`, populating `source_repo` /
    `source_ref` / `local_paths` / `removed_in_release` / `license` per
    issue #28's acceptance criterion (2). Best-effort where upstream
    URLs were not pinned at vendor time. Closes #28.

## [0.2.0] - 2026-05-04

### Removed (BREAKING for any saved workflow using `Easy_Version`)
- Dropped `Easy_Version` (and the `k_easy_version.py` source) — the
  whole node was just a one-liner that turned an integer N into the
  string `vNNN` (zero-padded). Maintainer concluded it was the first
  trivial node they ever wrote and saw no real value. Anyone who needs
  this exact behavior can either:
  - inline the format string in their workflow downstream of an INT
    primitive (`f"v{N:03d}"`), or
  - use any of the more general string-format nodes in the ecosystem
    (KJ Nodes, ComfyUI-Custom-Scripts, etc.).
- Saved workflows that reference the `Easy_Version` ID will fail to
  load — same migration as v0.1.5's `__koolook_v1_0_1` cleanup.

### Fixed (carried over from PR #39, [Unreleased] in 0.1.8)
- `Easy_hdr_VAE_encode` (Koolook v2.3.3) now wraps the encoded tensor in
  the standard ComfyUI `LATENT` dict (`{"samples": t}`) instead of
  returning the raw tensor. Wiring this node into KSampler previously
  crashed with `IndexError: too many indices for tensor of dimension 5`
  on Wan 2.2 video workflows, because KSampler does
  `latent["samples"]` and the raw 5-D tensor doesn't support string
  indexing. The decoder side was already correct.
- `Easy_hdr_VAE_encode` / `Easy_hdr_VAE_decode` (Koolook v2.3.3) now
  produce proper N-frame sequences end-to-end on Wan 2.2 / Hunyuan /
  CogVideoX / LTX video workflows. Two combined fixes:
  1. **Rank-aware dispatch.** 5-D `(B, F, H, W, C)` video tensors paired
     with a 2-D image VAE are iterated frame-by-frame and stacked along
     a temporal axis on encode (and per-frame decoded then concatenated
     to a `(B*F, H, W, C)` ComfyUI IMAGE batch on decode). 5-D tensors
     with a 3-D-native VAE — identified via `vae.latent_dim == 3`,
     which is the actual attribute ComfyUI sets on its VAE class for
     video VAEs — pass through unchanged.
  2. **5-D output normalization on decode.** 3-D-aware VAEs return a
     5-D `(B, F, H, W, C)` tensor from `vae.decode()`. Without an
     explicit reshape, downstream IMAGE-typed nodes (Get Image Size,
     SaveImage) misread that as a single 4-D image with `count=1` and
     `height=F` — exactly the symptom we were debugging
     (`1056×41 count=1` after a 41-frame Wan 2.2 sequence). The fix
     mirrors the stock ComfyUI VAEDecode node
     ([nodes.py:303-304](https://github.com/comfyanonymous/ComfyUI/blob/master/nodes.py#L303-L304)):
     `if img.ndim == 5: img = img.reshape(-1, H, W, C)` — the leading
     `(B, F, …)` dims collapse into the standard ComfyUI batch axis,
     so SaveImage now writes `_00001.png … _0004N.png` instead of one
     squashed frame.

  The new `path=image | video-iter | video-3d` field in `debug_info`
  surfaces which dispatch branch fired per run, for live verification.

### Added (Easy_hdr_VAE_encode / decode feature parity with upstream)
- Restored the cinema-grade color-management surface from upstream
  Radiance v2.3.3's `RadianceVAE4KEncode/Decode`, while keeping the
  slim no-tile-engine code path that lets these nodes work with
  Wan 2.2 video VAEs (the original motivation for the Koolook fork —
  upstream's 4K cosine-blend tile engine errors with `"size of tensor a
  (192) must match the size of tensor b (132) at non-singleton dimension 4"`
  on video VAEs because the tiler's spatial alignment fights the video
  VAE's internal temporal-aware encoding).
- **12 source / target color spaces** (was 4): Linear, ACEScg,
  **ACES 2065-1**, **Rec.2020 Linear**, sRGB, Raw, plus six cinema log
  curves — **ARRI LogC3** (EI-aware, default 800), **ARRI LogC4**,
  **Sony S-Log3**, **Panasonic V-Log**, **DaVinci Intermediate**,
  **RED Log3G10**. Round-trip math (linearize → encode) and matrix
  constants ported verbatim from upstream's `color_utils.py` with the
  bug-fix history preserved in the docstrings. Conversions are
  rank-agnostic and 4-channel-aware (alpha passes through untouched).
- **`Compress (Log)` HDR mode** — the upstream HDR-clamp-free pipeline.
  On encode, the linear-space tensor is re-encoded through a cinema log
  curve (matches `source_space` if it's a log space, else ARRI LogC4)
  and goes into the VAE without a hard clamp. On decode, the VAE output
  is run through `_soft_log_shoulder` (tanh rolloff at the per-profile
  knee instead of a hard clamp at 1.0) and `_denoise_log_highlights`
  (3×3 box blur in log space, ramped quadratically toward the highlight
  region) before log→linear conversion. Per-profile parameters
  (`LOG_PROFILE_HDR_PARAMS` table) tune knee / ceiling / denoise
  threshold / strength to each curve's slope at code 1.0 — RED Log3G10
  gets the most conservative settings, Sony S-Log3 / ARRI LogC4 the
  loosest. Pair `hdr_mode="Compress (Log)"` on both encode and decode
  for HDR-clean roundtrips; mismatched modes produce garbage by design.
- **`latent_sampling`** parameter (`sample` / `mean` / `mode`).
  `sample` is ComfyUI's default random posterior sample. `mean` and
  `mode` use the posterior mean for deterministic, lowest-noise
  encoding — best for img2img where minimum reconstruction noise
  matters. The mean/mode path replicates ComfyUI's preprocessing
  (BHWC → BCHW → [-1, 1] scaling) before reaching `first_stage_model`,
  with a graceful fallback chain that never crashes.
- **Real alpha output.** `Easy_hdr_VAE_encode` now returns
  `(LATENT samples, STRING debug_info, IMAGE alpha)`. With
  `alpha_handling="Preserve"` and a 4-channel input, the alpha channel
  is surfaced as a separate IMAGE for downstream re-compositing
  post-decode (VAEs don't encode alpha, so it's routed around them).
  With `alpha_handling="Ignore"` or a 3-channel input, alpha is a
  zeros tensor of compatible shape — downstream wiring never fails.
  The previous `alpha_handling` flag was a documented no-op.
- **Exposure now applied in linear space** for all source spaces. The
  previous behavior multiplied raw input bytes by `2^exposure` even
  when the input was sRGB-gamma or log-coded, which produced visually
  wrong results for non-linear sources. Linear sources are unaffected
  (linear × 2^stops is the correct semantic). Raw source still does a
  raw-domain multiplication so users who know what they're feeding the
  VAE keep their bytes intact.
- New helper module
  [`forks/radiance_koolook/versions/v2_3_3/color_helpers.py`](forks/radiance_koolook/versions/v2_3_3/color_helpers.py)
  contains all log curves, soft-shoulder / log-denoise helpers,
  `encode_with_sampling_mode`, color-space matrices, and dispatch
  tables. `nodes_vae.py` keeps the node wiring + sequence dispatch.

### Intentionally NOT ported (upstream-only features)
- 4K cosine-blend tile engine and its `tile_size` / `overlap` /
  `processing_mode` knobs — this is the broken plumbing that motivated
  the slim fork in the first place.
- `inverse_tonemap` / `target_stops` (SDR→HDR expansion).
- `.rhdr` sidecar export and `rhdr_precision` (Radiance Viewer-specific).
- `crop_padding` (only relevant when tiling).

These can be added back in future patches if needed; the slim wrapper
stays at ~600 lines of node wiring + ~450 lines of color helpers,
~40% the size of upstream's `vae.py` (2,638 lines).

### Added (carried over from PR #40, also unreleased on main)
- `scripts/sync_to_dev.py` — pure-stdlib helper that copies the curated
  runtime files (`__init__.py`, `config.json`, top-level `k_*.py`,
  `forks/`, `web/`) into a live ComfyUI `custom_nodes/<pack>/` folder
  for fast local iteration without bumping a version. Reads the target
  path from the new `KOLOOK_COMFYUI_DEV_PATH` env var (auto-loads `.env`
  from repo root); errors out cleanly when unset.
- `.env.example` — `KOLOOK_COMFYUI_DEV_PATH=` entry with comment.
- `CLAUDE.md` — new "`dev-sync`" trigger-phrase section so the agent
  knows to run `scripts/sync_to_dev.py` when the maintainer says
  "dev-sync" / "sync dev" / "copy those files" mid-session.

### Net effect
- Node count drops from 8 to **7**.
- `Koolook/Pipeline` subfolder now has only 1 entry (`Easy AI Pipeline`)
  instead of 2.
- Faster local dev loop: a one-line tweak no longer requires cutting a
  full release just to test it inside ComfyUI.

## [0.1.8] - 2026-05-03

### Changed (categories — affects ComfyUI node-add menu)
- `Koolook/VFX` is gone. Three more granular subfolders replace it:
  - **`Koolook/Pipeline`** — `Easy_Version`, `EasyAIPipeline` (workflow
    setup nodes)
  - **`Koolook/Image`** — `EasyResize_Koolook`, `easy_ImageBatch` (joins
    `EasyResize_Koolook`; `easy_ImageBatch` was previously in VFX)
  - **`Koolook/VAE`** — `Easy_hdr_VAE_encode`, `Easy_hdr_VAE_decode`
    (now have their own subfolder; previously also in VFX)
- Workflows continue to load and run unchanged — `CATEGORY` is purely a
  UI-organization hint and only affects the node-add menu hierarchy.

### Fixed (search discoverability)
- `Easy_hdr_VAE_encode` / `Easy_hdr_VAE_decode` display names are now
  **"Easy HDR VAE Encode (Koolook)"** / **"Easy HDR VAE Decode (Koolook)"**.
  Previously the display name equalled the node ID (`Easy_hdr_VAE_encode`
  with no "Koolook" string), so typing `koolook` in ComfyUI's node-add
  search filter excluded them — they appeared registered but invisible.
  Now they group with the rest of the pack under that filter.
- Class IDs (`Easy_hdr_VAE_encode`, `Easy_hdr_VAE_decode`) are unchanged —
  saved workflows that reference these IDs continue to load.

## [0.1.7] - 2026-05-03

### Changed
- All node display names now suffix `(Koolook)` so they surface together
  when users search "koolook" in the ComfyUI node-add menu. Previously
  only `KoolookLoadCameraPosesAbsolute` (and the v2.3.3 fork nodes via
  their version suffix) matched.
  - `Easy_Version` → `Easy Version (Koolook)`
  - `EasyAIPipeline` → `Easy AI Pipeline (Koolook)`
  - `easy_ImageBatch` → `Easy Image Batch (Koolook)`
  - `EasyWan22Prompt` → `Wan 2.2 Easy Prompt (Koolook)`
  - `Easy_Version` category moved from `VFX/Utils` to `Koolook/VFX`
    to match the rest of the pack.
- Class keys (`NODE_CLASS_MAPPINGS`) are unchanged — saved workflows keep
  loading and running unchanged.
- Carried in via PR #36, then bundled into this release alongside the
  documentation reorg below.

### Docs reorg (root cleanup)
- Moved out of repo root (preserving git history via `git mv`):
  - `Glossary.md` → [`docs/reference/glossary.md`](docs/reference/glossary.md)
  - `RELEASING.md` → [`docs/maintainers/releasing.md`](docs/maintainers/releasing.md)
- Root now keeps only the four files that *must* live there:
  `README.md` (GitHub repo home + Comfy Registry description),
  `LICENSE` (`pyproject.toml` reference + GitHub license badge),
  `CHANGELOG.md` (tooling convention),
  `CLAUDE.md` (Claude Code agent instructions).
- New `docs/` structure with three audience buckets:
  - [`docs/user_guide/`](docs/user_guide/) — end-user, per-node guides + screenshots (`img/`)
  - [`docs/reference/`](docs/reference/) — lookup material (glossary, node inventory)
  - [`docs/maintainers/`](docs/maintainers/) — project-internal procedures
- Each bucket gets a short index `README.md` so the structure is
  navigable from `docs/README.md` downwards without grep.

### Added (new maintainer docs)
- [`docs/maintainers/registry-api.md`](docs/maintainers/registry-api.md) —
  documents the **undocumented** Comfy Registry version-management API
  that we reverse-engineered while cleaning up v0.1.0–0.1.5
  (deprecate / undeprecate / yank endpoints, status enum values,
  auto-deprecation behavior, ready-to-paste curl recipes, and an
  optional `registry-mgmt.yml` GitHub Actions workflow for codified
  version management without ever exposing the token to a shell).
- [`docs/maintainers/node-versioning.md`](docs/maintainers/node-versioning.md) —
  codifies the rules from the v0.1.5 PR-review discussion for safely
  changing a node's `INPUT_TYPES` / `RETURN_TYPES` / class names without
  breaking saved user workflows. Five rules + the suffix-version pattern
  + the alias-then-deprecate migration path + concrete worked examples.
- Cross-references in `README.md`, `CLAUDE.md`,
  `.github/ISSUE_TEMPLATE/release_checklist.md`, and the
  `add-external-fork` skill all updated to point at the new paths.
- Imported the existing untracked `docs/` images that the maintainer
  had already started adding locally (node inventory screenshot under
  `reference/`, Easy Image Batch helper image under `user_guide/img/`).

## [0.1.6] - 2026-05-03

### Renamed (with back-compat alias)
- `EasyResize` is now exposed canonically as **`EasyResize_Koolook`**
  (display: `Easy Resize (Koolook)`) to resolve a node-ID collision with
  `ComfyUI-EasyFilePaths`, which also registers the bare name `EasyResize`.
  The old `EasyResize` ID is kept as a **deprecated alias** pointing at
  the same class, so saved workflows still load and run unchanged. The
  alias will be removed in a future major release once the deprecation
  has had time to propagate.

### Attribution
- Added a proper SPDX header + GPL-3.0 attribution block to
  `k_easy_resize.py`, crediting `kijai/ComfyUI-KJNodes` (`Resize Image V2`)
  as the original interface inspiration. The Koolook implementation is a
  fresh write that materially extended the surface (aspect-ratio parser,
  keep_proportion modes, mask + composed outputs, device selection,
  target/original W/H reporting, color-panel passthrough). KJ Nodes is
  GPL-3.0, same as our pack — no relicense required.
- `forks/THIRD_PARTY.md` and `forks/forks_manifest.yaml` upgraded the
  KJ Nodes entry from `license: "unknown"` /
  `sync_state: "needs-upstream-reference"` to verified GPL-3.0 with full
  per-feature change notes.

### Notes for users
- If a saved workflow uses the bare `EasyResize` ID, it still works but
  the node's display name now reads
  `Easy Resize (deprecated, use 'Easy Resize (Koolook)')` as a hint to
  swap. New workflows should pick `EasyResize_Koolook` from the node-add
  menu.

## [0.1.5] - 2026-05-03

### Removed (BREAKING for anyone using the v1_0_1 namespaced IDs)
- Dropped the entire `forks/radiance_koolook/versions/v1_0_1/` folder
  (~5,200 lines, 26 namespaced nodes). The wrappers were vestigial —
  Koolook authors never used them, no internal workflow referenced any
  `__koolook_v1_0_1` suffixed ID, and the VAE pair was already
  superseded by `Easy_hdr_VAE_encode` / `Easy_hdr_VAE_decode` in v2_3_3.
- IDs that no longer exist after this release:
  `ImageToFloat32__koolook_v1_0_1`, `Float32ColorCorrect__koolook_v1_0_1`,
  `HDRExpandDynamicRange__koolook_v1_0_1`, `HDRToneMap__koolook_v1_0_1`,
  `ColorSpaceConvert__koolook_v1_0_1`, `SaveImageEXR__koolook_v1_0_1`,
  `LoadImageEXR__koolook_v1_0_1`, `LoadImageEXRSequence__koolook_v1_0_1`,
  `SaveImage16bit__koolook_v1_0_1`, `HDRHistogram__koolook_v1_0_1`,
  `LogCurveEncode__koolook_v1_0_1`, `LogCurveDecode__koolook_v1_0_1`,
  `HDRExposureBlend__koolook_v1_0_1`,
  `HDRShadowHighlightRecovery__koolook_v1_0_1`,
  `OCIOColorTransform__koolook_v1_0_1`,
  `OCIOListColorspaces__koolook_v1_0_1`, `GPUTensorOps__koolook_v1_0_1`,
  `HDR360Generate__koolook_v1_0_1`, `SaveHDRI__koolook_v1_0_1`,
  `ACES2OutputTransform__koolook_v1_0_1`,
  `DaVinciWideGamut__koolook_v1_0_1`, `ARRIWideGamut4__koolook_v1_0_1`,
  `RadianceVAEEncode__koolook_v1_0_1`,
  `RadianceVAEDecode__koolook_v1_0_1`, `k_easy_OCIO_v101`
  (the short ID for `RadianceOCIOColorTransformV2`),
  `RadianceLogCurveDecode__koolook_v1_0_1`.
- Migration path: install upstream Radiance directly
  (https://github.com/fxtdstudios/radiance) for the HDR/EXR/OCIO
  functionality. Use `Easy_hdr_VAE_encode/decode` (already in v0.1.3+)
  for video VAE workflows.
- Source recoverable via `git checkout HEAD~1 -- forks/radiance_koolook/versions/v1_0_1/`
  if you ever realize you need any of those wrappers back.

### Added (registry hygiene from the original v0.1.4 plan)
- `.gitignore` now excludes `upscaler_FIX/` and `nuke_CAM_exporter/` —
  the maintainer's local dev workspaces accidentally committed in
  Dec 2025. These were never imported by the package's root
  `__init__.py`, so they had no runtime effect, but the Comfy Registry's
  static scanner picked up `NODE_CLASS_MAPPINGS` from vendored 3rd-party
  clones inside them and counted ~12 spurious nodes against this pack
  in ComfyUI-Manager (yielding the misleading "44 nodes / 13 conflicts"
  badge).

### Removed (registry hygiene)
- `git rm -r --cached upscaler_FIX nuke_CAM_exporter` — 70 files
  untracked from git, files stay on the maintainer's local disk for
  reference. ~3.7 MB of unrelated content out of the published archive.

### Net effect on Manager / registry
- Node count drops from 44 to **8** (6 root Koolook + 2 v2_3_3 VAE).
- Spurious "Conflict with `ComfyUI-SuperUltimateVaceTools`" warnings
  disappear.
- Published archive shrinks by ~9 MB total (3.7 MB dev workspaces +
  ~5.2 MB v1_0_1 fork code).

### Notes for the maintainer
- After merging this PR and `git pull`-ing main, your local working tree
  will lose the `upscaler_FIX/` and `nuke_CAM_exporter/` folders (git
  applies the deletion). Back up first or restore via
  `git checkout HEAD~1 -- ...`. The `upscaler_FIX/` folder has already
  been physically moved to `../ComfyUI-Forks-BK/`; the
  `_Utils-CAM-track/` subfolder of `nuke_CAM_exporter/` has been moved
  to `../ComfyUI-Tools-BK/nuke_CAM_exporter/`. The remainder of
  `nuke_CAM_exporter/` (your actual Nuke pipeline work) is still on
  disk in MAIN but no longer tracked by git.

## [0.1.4] - 2026-05-03 (test-published only, superseded by 0.1.5)

### Removed (registry hygiene)
- Untracked the maintainer's local dev workspaces from git: `upscaler_FIX/`
  and `nuke_CAM_exporter/`. These were never imported by the package's root
  `__init__.py`, so they had no effect on what ComfyUI loaded at runtime,
  but the Comfy Registry's static scanner picked up the `NODE_CLASS_MAPPINGS`
  dicts inside vendored 3rd-party clones and counted them as part of this
  pack — yielding a misleading "44 nodes / 13 conflicts" badge in
  ComfyUI-Manager (against `ComfyUI-SuperUltimateVaceTools`,
  `ComfyUI-multigpu`, etc.). Files removed from index via
  `git rm -r --cached` (still on the maintainer's local disk) and the two
  paths are now `.gitignore`d so they cannot leak again.
- Net effect: published archive shrinks by ~3.7 MB (70 files), Manager's
  node count drops from 44 to ~32 (the actual runtime registrations:
  6 root Koolook + ~26 namespaced fork variants), and the spurious
  conflict warnings disappear. Same class of issue as the v0.1.2 GPL-3.0
  relicense — vendored 3rd-party code in MAIN, this time without the
  license-compatibility risk because none of it was ever imported.

### Notes for the maintainer
- After merging this PR and `git pull`-ing main, your local working tree
  will lose the `upscaler_FIX/` and `nuke_CAM_exporter/` folders (git
  applies the deletion). If you want to keep working on those locally,
  back them up first (`cp -r upscaler_FIX nuke_CAM_exporter ~/backup/`)
  or restore from history afterwards
  (`git checkout HEAD~1 -- upscaler_FIX nuke_CAM_exporter`); they will
  then live as untracked files in your working tree, ignored by the new
  `.gitignore` rules.

## [0.1.3] - 2026-05-03

### Renamed
- The new v2_3_3 VAE nodes are now exposed as `Easy_hdr_VAE_encode` /
  `Easy_hdr_VAE_decode` (clean IDs and display names — no
  `__koolook_v2_3_3` suffix), to avoid visual collision with upstream
  Radiance v2.3.3's `RadianceVAEEncode` / `RadianceVAEDecode` aliases
  in the ComfyUI node-add search. The version is still tracked
  structurally (file lives in `versions/v2_3_3/`) and textually
  (file header + UPSTREAM_PIN.yaml + forks/THIRD_PARTY.md). Other
  Koolook nodes in the v2_3_3 set (none today, but possible in future)
  would still use the `__koolook_v2_3_3` suffix by default — opt out
  via the new `SKIP_VERSION_SUFFIX` set in `versions/v2_3_3/__init__.py`.
- 0.1.2 was test-published to the registry with the previous
  `RadianceVAEEncode__koolook_v2_3_3` IDs but never tagged or formally
  released; bumping to 0.1.3 publishes the renamed version cleanly.

## [0.1.2] - 2026-05-03 (test-published only, superseded by 0.1.3)

### License (BREAKING)
- **Relicensed entire package to GPL-3.0.** v0.1.0 and v0.1.1 shipped under
  a claimed MIT license while already incorporating GPL-3.0-derived code from
  [fxtdstudios/radiance](https://github.com/fxtdstudios/radiance) under
  `forks/radiance_koolook/`. GPL-3.0 §5(c) requires the entire combined work
  to be GPL-3.0; relicensing aligns the package with what we actually ship
  and matches the dominant license posture of the ComfyUI custom-node
  ecosystem. Downstream users incorporating, linking to, or deriving from
  ComfyUI-Koolook must now distribute under GPL-3.0 (or compatible).
- Added `LICENSE` file at repo root with the full GPL-3.0 text
  (`pyproject.toml` previously referenced a `LICENSE` file that did not
  exist on disk).
- README "License" section rewritten to declare GPL-3.0 and explain the
  §5(c) implication.

### Added
- **`forks/radiance_koolook/versions/v2_3_3/`** — slim, video-friendly
  re-implementation of `RadianceVAEEncode` / `RadianceVAEDecode` exposed
  under the namespace suffix `__koolook_v2_3_3`. Mirrors the *interface
  surface* of upstream `RadianceVAE4KEncode` / `RadianceVAE4KDecode`
  but skips the 4K cosine-blend tile engine, which conflicts with
  modern video VAEs (Wan 2.2, Hunyuan, CogVideoX, LTX) that already
  handle their own temporal/spatial stitching internally. Fixes the
  `"size of tensor a (192) must match the size of tensor b (132) at
  non-singleton dimension 4"` runtime error users hit when chaining
  upstream's VAE encoder into Wan 2.2 video workflows.
- `forks/THIRD_PARTY.md` — full attribution entries for the v1.0.1
  baseline (already in the repo) and the new v2.3.3 VAE subset, with
  per-class change notes.
- `.claude/skills/license-pre-check/` — blocking license-compatibility
  audit skill for Claude Code. Run **before** copying or porting any
  third-party code; refuses to proceed on incompatible combinations
  (e.g. GPL upstream into MIT downstream).
- `.claude/skills/add-external-fork/` — Claude Code port of the
  existing Cursor skill, with a mandatory Phase 0 ("run
  license-pre-check first") and a Phase 3c requirement to add GPL §5(a)
  modification headers on derived files.
- `RELEASING.md` — canonical, step-by-step release procedure (was previously
  ad-hoc; the gaps it closes are exactly what caused the `v0.1.0` orphan-tag
  and `CristianP` `PublisherId` incidents).
- README "Release & Stability" section now links `RELEASING.md` and the
  per-release checklist template.
- `CLAUDE.md` now references `RELEASING.md` for agents.

### Changed
- `forks/forks_manifest.yaml` — explicit `license: "GPL-3.0"` and
  `license_verified_at` fields on the existing radiance entry (was
  `to_verify_at_source_ref`); pre-registered the new v2.3.3 VAE entry
  with verified license metadata.
- `.github/ISSUE_TEMPLATE/release_checklist.md` rewritten to mirror
  `RELEASING.md`, including the registry-publisher validation step.

### Notes for users
- If your workflow currently references the bare-name `RadianceVAEEncode`
  or `RadianceVAEDecode` (which routes to upstream Radiance v2.3.3's
  `RadianceVAE4KEncode/Decode` via that package's alias), and you hit
  the 4D vs 5D tensor mismatch on video workflows, switch to the
  namespaced `RadianceVAEEncode__koolook_v2_3_3` /
  `RadianceVAEDecode__koolook_v2_3_3` (display name suffix
  *(Koolook v2.3.3)*).
- The existing `__koolook_v1_0_1` namespaced nodes are unchanged and
  remain available for backward compatibility with saved workflows.

## [0.1.1] - 2026-05-03

### Fixed
- Re-tag release at current `main` HEAD so `git describe --tags` resolves locally
  (the original `v0.1.0` tag points at an orphaned merge commit that is no longer
  in `main`'s ancestry, which caused ComfyUI-Manager to display the installed
  version as "unknown").
- Correct `[tool.comfy] PublisherId` from the placeholder `CristianP` to the
  real publisher `kforgelabs`, which was the actual cause of every
  `Publish to Comfy registry` workflow failure (registry returned a misleading
  `400 "Failed to validate token"` because the declared publisher did not exist).

### Changed
- Bumped `pyproject.toml` `version` to `0.1.1` to match the new tag.

### Chore
- `.gitignore`: fix syntax and exclude `__pycache__/`, `*.pyo`, `.DS_Store`
  (carried forward from commit `8fc28d1`).

## [0.1.0] - 2026-04-24

### Added
- Fork tracking moved to `forks/` with centralized workflow documentation.
- Versioned Radiance fork package layout under `forks/radiance_koolook/versions/v1_0_1`.
- `CLAUDE.md` and `Glossary.md` to keep workflow and naming conventions explicit.

### Changed
- Root node loader now imports the versioned Radiance fork entrypoint.
- Release/checklist/template references updated to current fork-based paths.
- Introduced compact node ID override for OCIO transform: `k_easy_OCIO_v101`.

### Removed
- Legacy `ACES_workflow/radiance` tree and duplicated tracking paths under `third_party/`.
- Deprecated docs/assets/workflow artifacts removed during repository cleanup.
