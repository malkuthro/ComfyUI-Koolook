# Workflows sidebar — quick reference

The "Workflows" section of the Kforge Labs sidebar tab; implementation
lives in [`web/sidebar/`](../../web/sidebar/), with the workflow store + ops
in [`workflows_store.js`](../../web/sidebar/workflows_store.js) and the tree
rendering in [`tree.js`](../../web/sidebar/tree.js).

## Save / load loop

1. Click **💾** (whole canvas) or **✂** (selection only) in the Workflows action bar.
2. Modal: **Directory** (flat path picker) → **Base on existing** (optional) → **Action** → **Workflow name** (shown only when needed).
3. Saved entry appears at the bottom of its directory in the tree.
4. Click any workflow row to load it onto a fresh tab; the tab title takes the workflow's name.

The directory dropdown lists every directory path in the tree as a flat
list (`UP-scale`, `UP-scale / Type-A`, `Depth`, …). The "+ New directory…"
option creates a new top-level directory. Subdirectories are created via
right-click on an existing directory (see below).

## Action semantics

| Action | Resulting name | Archive behavior |
|---|---|---|
| New name | typed | auto-archives only if the typed name happens to collide |
| Use existing name (archive previous) | base | always archives the existing entry |
| Modify existing name | base, edited | no archive (different name) |

The archive trigger is a name collision with any existing active entry,
regardless of which action was selected. The Action dropdown is UX
guidance; the underlying logic flows from the resulting `(name, dir)` pair.

## Modules — splice a saved cluster into your live canvas

A saved workflow tagged with the literal tag **`module`** behaves like a
reusable building block: clicking it **inserts** its nodes into the existing
canvas (placed at the viewport center, freshly id-remapped, internal links
re-created, the inserted nodes left selected) instead of replacing the
graph the way a normal Load does.

Typical use cases this is designed for:

- **EXR output stack** — a Save EXR + path-builder + format selector cluster
  the user wants to drop into any render-time workflow.
- **Depth-only render** — a depth-map node + preview wired up, ready to be
  spliced into a wider pipeline.
- **Wan 2.2 prompt module** — the `EasyWan22Prompt` node + a CLIP encoder
  + any conditioning glue, saved once, dropped into ten different
  workflows.

How it works:

| Surface | Behavior |
|---|---|
| **Save modal** (selection save) | Includes a **`Save as module`** checkbox, **pre-checked** for selection saves. Whole-canvas saves get the same checkbox but **unchecked** by default. The saved entry gets `module: true` and the compatibility `module` tag in the same atomic `persistMutation` as the save — a commit failure rolls both back. |
| **Workflows section** | Module-tagged entries get a green **`pi pi-plus-circle`** icon and a distinct hover tooltip. **Left-click inserts** instead of loading. Right-click still offers **both** `Load` and `Insert into canvas` — `Load` stays available for every row regardless of tag. |
| **Tags section** | Same flip — a workflow tagged `module` is treated as a module no matter which tag folder it shows up under. |
| **Archive folder** | Archived module entries still left-click to **Load** (they're old versions, treated as "review" rather than "splice"). Right-click still has both Load and Insert. |

Implementation pieces:

- The literal tag string lives in [`web/sidebar/constants.js`](../../web/sidebar/constants.js) as `MODULE_TAG`. New saves store module state as first-class `module: true`; the tag is still honored for old entries and for the manual right-click Tags workflow.
- The non-destructive insert primitive is `insertWorkflowOntoCanvas(dirPath, wfName)` in [`web/sidebar/canvas_io.js`](../../web/sidebar/canvas_io.js). It pre-flights every referenced node type against `LiteGraph.registered_node_types`, aborts cleanly with a toast when any are missing (a partial insert with stub nodes is worse than no insert), then deep-clones the saved graph, normalizes Comfy/LiteGraph link records, configures each node with stale saved link IDs stripped, sets `node.id = -1` so `app.graph.add()` allocates fresh ids that can't collide with the live canvas, and finally recreates internal connections via `originNode.connect(...)` (which auto-allocates fresh link ids). Selection saves that include ComfyUI subgraph wrapper nodes also carry the required `definitions.subgraphs` entries so a later native Load has the definitions available.
- Bbox-of-cluster placement uses CSS pixels (`clientWidth`/`clientHeight`), not the HiDPI backing buffer — same correction as the existing `placeAtCanvasCenter` helper.
- Selection saves preserve ComfyUI group boxes that overlap selected nodes.
  Insert translates those saved groups by the same placement delta as the
  inserted node cluster, so group titles and boxes survive module round trips.
  This is intentionally limited to groups touching the saved selection; unrelated
  canvas groups are not copied into partial workflow saves.

To turn an **existing** saved workflow into a module: right-click → **Tags…**
→ add `module`. (No re-save needed; the row re-renders on the next
`workflows-changed` event.) To un-module: remove the `module` tag the same way;
the tag editor keeps the first-class `module` flag in sync.

## Right-click menus

- **Workflow row**: Load / **Insert into canvas** / Update from selection or canvas / Rename / Duplicate / Tags / Move to archive (or Restore from archive if already archived) / + New directory / + New subdirectory / Delete. Existing-folder moves are handled by drag-and-drop so the menu stays short in large libraries.
- **Directory row** (any depth): **Create subdirectory…** / Rename / Delete (with confirm if non-empty — the message names workflow + subdirectory counts).
- **Archive folder** (synthetic — appears only when a directory has archived workflows): **Clean up archive** runs across the whole selected Archive folder, groups entries by original workflow/setup name, keeps the newest archived entry from the last 5 minutes, last hour, and last day for each group (plus one fallback if none land in those windows), and confirms the keep/delete counts before mutating; **Delete archive (N)** removes every archived workflow in this directory in one go. Active workflows in the same directory are untouched by both actions.

## Drag-and-drop (Tier 1 — moves only; reordering is alphabetical)

| Drag | Drop on | Effect |
|---|---|---|
| Workflow row | Directory row | Move workflow to that directory (no-op if same dir) |
| Workflow row | Archive folder | Archive in that directory (move first if cross-dir) |
| Directory row | Directory row | Nest the dragged dir as a child of the target |

**Cycle prevention:** dragging a directory onto itself, its current parent, or any of its descendants is rejected. **Name collisions** (the destination already has a sibling with the same name) are rejected. **Reserved name:** dropping a directory literally named `Archive` into a non-root parent is rejected (would shadow the synthetic Archive folder).

Visual feedback: drop targets get a blue outline on hover. Failed drops surface a toast.

Sort within a level remains alphabetical — Tier 1 doesn't introduce a custom-order schema. (A future Tier 2 could.)

## Subdirectories

Every directory can host nested subdirectories at arbitrary depth. Each
nested directory behaves like a top-level directory: it can hold workflows,
its own Archive subfolder (when any of its workflows are archived), and
further subdirectories.

- **Create**: right-click a directory → "Create subdirectory…"
- **Save into**: pick the nested path from the save modal's directory dropdown (`UP-scale / Type-A`)
- **Move across**: drag a workflow row onto the target directory row
- **Reserved name**: subdirectory names cannot be `Archive` (case-insensitive) — that string is reserved for the synthetic Archive folder rendered for archived workflows. Top-level directories named `Archive` are allowed (no collision at root).

## Storage

| Where | What |
|---|---|
| `/userdata/koolook_workflows.json` (ComfyUI install) | Primary. Per-install. |
| `localStorage["koolook.workflows.fallback.v1"]` | Used only if `/userdata` is unreachable. |
| `localStorage["koolook.workflows.seeded.v1"]` | `"1"` once defaults have been seeded for this install. |
| `web/workflow_defaults.json` (in repo) | Distributed starter pack — seeded once into `/userdata` on first load with empty workflow data. |

## Comfy workflow draft quota gotcha

ComfyUI keeps its own browser-side workflow draft cache in localStorage,
and the key scheme has changed across frontend versions. All generations
share the `Comfy.Workflow.Draft` prefix:

| Generation | Keys | Written when |
|---|---|---|
| V1 (legacy; still written on tab switches as of frontend 1.44) | `Comfy.Workflow.Drafts` (one blob of all drafts) + `Comfy.Workflow.DraftOrder` (LRU) | switching/loading workflows |
| V1 per-workspace (interim) | `Comfy.Workflow.Drafts:<ws>` + `Comfy.Workflow.DraftOrder:<ws>` | older 1.4x frontends; the 1.44 migration reads these once **without deleting them** (later frontends added cleanup on successful migration, and an interrupted migration can still leave them behind) |
| V2 (frontend 1.44+) | `Comfy.Workflow.DraftIndex.v2:<ws>` (index) + `Comfy.Workflow.Draft.v2:<ws>:<hash>` (one payload per draft) | ~512 ms after every graph edit |

Once the origin's storage quota is hit, draft saves fail and Comfy shows
**"Failed to save workflow draft"** toasts. Worse, the 1.44+ V2 layer only
evicts its own current-workspace drafts on a quota error, and when that
isn't enough it latches a session-wide *storage unavailable* flag — after
which **every** edit toasts until the page is reloaded, regardless of which
workflow is open.

Koolook hit the symptom first in v0.3.6 with sidebar workflow loads; the
fix was a stable temporary workflow ID derived from the sidebar path/name
so the same workflow replaces the same draft entry instead of creating a
new one per load. For Koolook frontend extensions that ship large
timeline/editor data, the standing rules remain:

- Do not churn workflow IDs for equivalent exported/imported workflow files.
- Keep hidden widget payloads plain JSON and strip preview-only blobs before
  serialization.

The systemic countermeasure is the global guard in
`web/koolook_draft_guard.js` (loaded on every page; behavior covered by
`tests/js/test_draft_guard.mjs`). It matches draft keys **by prefix**, so a
future upstream rename inside the prefix degrades to last-resort eviction
instead of going silently stale (which is how the previous exact-name
guard, then embedded in the LTX Director extension, died on frontend
1.44). Two mechanisms:

1. **Boot prune** — deletes suffixed-V1 families already migrated to V2
   (dead weight 1.44-era migrations leave behind), corrupt keys (scoped to
   the offending key only), oversized entries (>750k chars), and enforces a
   ~2M-char total draft budget, oldest drafts first.
2. **`localStorage.setItem` quota guard** — on `QuotaExceededError` for a
   draft key, evicts the oldest draft across *all* generations and retries
   inside `setItem`, so the V2 layer's own catch never sees the error: the
   save succeeds, no toast, and the storage-unavailable latch never trips.

The guard never touches Koolook `/userdata`, snapshots, workflow library
entries, render outputs, or any non-draft localStorage key.

JSON shape (recursive — `directories` lives both at the root AND inside
every directory node):

```json
{
  "directories": {
    "<dir>": {
      "workflows": {
        "<name>": {
          "savedAt": "<ISO timestamp>",
          "graph": { /* serialized ComfyUI workflow JSON */ },
          "module": true,
          "tags": ["module"],
          "archived": true
        }
      },
      "directories": {
        "<sub>": {
          "workflows": { /* … */ },
          "directories": { /* recurses */ }
        }
      }
    }
  }
}
```

`archived` is optional (false/missing means active). Pre-v0.3 stores
without the nested `directories` field still load fine — `normalize`
treats a missing `directories` as `{}` and the rest of the code assumes
it always exists after normalization.

## Reset (force re-seed from defaults)

Delete the userdata file in the live ComfyUI install:

```bash
rm <ComfyUI>/user/default/userdata/koolook_workflows.json
```

Then in the ComfyUI page DevTools console:

```js
localStorage.removeItem("koolook.workflows.seeded.v1");
localStorage.removeItem("koolook.workflows.fallback.v1");
location.reload();
```

## Things that surprised us — keep in mind

- **Selection saves are partial graphs.** Only the selected nodes + links between them survive; links into/out of non-selected nodes are nulled out, so the loaded workflow never has dangling references. Link handling accepts both serialized array links (`[id, origin, slot, target, slot, type]`) and object-shaped `LLink` records from `graph.links`.
- **Selection saves preserve relevant groups.** Any ComfyUI group box that
  overlaps selected nodes is stored with the partial graph and restored on Load
  or Insert. This supports the published-setup convention where
  `Koolook Input` and `Koolook Output` groups mark the setup's app surface.
- **Subgraph wrappers are special.** The saved selection includes transitively referenced `definitions.subgraphs`, but insert still cannot register a never-loaded subgraph definition because ComfyUI's subgraph registration path lives inside native `app.loadGraphData`. If Insert reports an unregistered subgraph definition, native-Load the workflow once in that browser session, then retry Insert.
- **Workflow tab name comes from `app.loadGraphData(graph, true, true, name, {})`'s 4th arg.** The tab flips from "Unsaved Workflow (N)" to the saved name, and Ctrl+S pre-fills with that name. (Verified against `Comfy-Org/ComfyUI_frontend` `src/scripts/app.ts`.)
- **Folder expansion state persists across re-renders** (the `pathStates` Map in `web/sidebar/tree.js`). Saving never collapses the directory you were viewing — and a save into a nested path opens every ancestor folder.
- **Archive sub-folder is rendered *above* active workflows** in each directory (and *below* nested subdirectories), so the latest active workflow sits closest to the bottom of its directory — easy to spot.
- **Directory header counts include all descendants.** A parent dir with no direct workflows but multiple subdirectories shows the recursive total.
- **No background auto-pruning of archives.** Re-saving the same name can still leave many timestamped archive entries. Right-click the synthetic Archive folder → **Clean up archive** to triage all archived entries in that folder by original workflow/setup name, keeping the newest archived entry from the last 5 minutes, last hour, and last day for each group. Use **Delete archive (N)** / per-row Delete when you want explicit removal.
- **Same-pattern distribution as `starter_preset.json`** — see [`curated-sidebar.md`](curated-sidebar.md) for the seeding semantics that also apply here.
