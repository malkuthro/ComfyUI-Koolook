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

## Right-click menus

- **Workflow row**: Load / Rename / `→ <other path>` (lists every other directory in the tree) / Move to archive (or Restore from archive if already archived) / Delete.
- **Directory row** (any depth): **Create subdirectory…** / Rename / Delete (with confirm if non-empty — the message names workflow + subdirectory counts).

## Subdirectories

Every directory can host nested subdirectories at arbitrary depth. Each
nested directory behaves like a top-level directory: it can hold workflows,
its own Archive subfolder (when any of its workflows are archived), and
further subdirectories.

- **Create**: right-click a directory → "Create subdirectory…"
- **Save into**: pick the nested path from the save modal's directory dropdown (`UP-scale / Type-A`)
- **Move across**: right-click a workflow → choose any other path in the move submenu
- **Reserved name**: subdirectory names cannot be `Archive` (case-insensitive) — that string is reserved for the synthetic Archive folder rendered for archived workflows. Top-level directories named `Archive` are allowed (no collision at root).

## Storage

| Where | What |
|---|---|
| `/userdata/koolook_workflows.json` (ComfyUI install) | Primary. Per-install. |
| `localStorage["koolook.workflows.fallback.v1"]` | Used only if `/userdata` is unreachable. |
| `localStorage["koolook.workflows.seeded.v1"]` | `"1"` once defaults have been seeded for this install. |
| `web/workflow_defaults.json` (in repo) | Distributed starter pack — seeded once into `/userdata` on first load with empty workflow data. |

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

- **Selection saves are partial graphs.** Only the selected nodes + links between them survive; links into/out of non-selected nodes are nulled out, so the loaded workflow never has dangling references.
- **Workflow tab name comes from `app.loadGraphData(graph, true, true, name, {})`'s 4th arg.** The tab flips from "Unsaved Workflow (N)" to the saved name, and Ctrl+S pre-fills with that name. (Verified against `Comfy-Org/ComfyUI_frontend` `src/scripts/app.ts`.)
- **Folder expansion state persists across re-renders** (the `pathStates` Map in `web/sidebar/tree.js`). Saving never collapses the directory you were viewing — and a save into a nested path opens every ancestor folder.
- **Archive sub-folder is rendered *above* active workflows** in each directory (and *below* nested subdirectories), so the latest active workflow sits closest to the bottom of its directory — easy to spot.
- **Directory header counts include all descendants.** A parent dir with no direct workflows but multiple subdirectories shows the recursive total.
- **No auto-pruning of archives.** Re-saving the same name many times leaves many timestamped archive entries. Right-click → Delete to remove individual ones when no longer useful.
- **Same-pattern distribution as `curated_defaults.json`** — see [`curated-sidebar.md`](curated-sidebar.md) for the seeding semantics that also apply here.
