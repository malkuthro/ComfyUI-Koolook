# Workflows sidebar — quick reference

The "Workflows" section of the Curated Nodes sidebar tab; implementation
shares [`web/koolook_sidebar.js`](../../web/koolook_sidebar.js) with the
Curated Nodes feature.

## Save / load loop

1. Click **💾** (whole canvas) or **✂** (selection only) in the Workflows action bar.
2. Modal: **Directory** → **Base on existing** (optional) → **Action** → **Workflow name** (shown only when needed).
3. Saved entry appears at the bottom of its directory in the tree.
4. Click any workflow row to load it onto a fresh tab; the tab title takes the workflow's name.

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

- **Workflow row**: Load / Rename / `→ <other dir>` / Move to archive (or Restore from archive if already archived) / Delete.
- **Directory row**: Rename / Delete (with confirm if non-empty).

## Storage

| Where | What |
|---|---|
| `/userdata/koolook_workflows.json` (ComfyUI install) | Primary. Per-install. |
| `localStorage["koolook.workflows.fallback.v1"]` | Used only if `/userdata` is unreachable. |
| `localStorage["koolook.workflows.seeded.v1"]` | `"1"` once defaults have been seeded for this install. |
| `web/workflow_defaults.json` (in repo) | Distributed starter pack — seeded once into `/userdata` on first load with empty workflow data. |

JSON shape:

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
      }
    }
  }
}
```

`archived` is optional (false/missing means active).

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
- **Folder expansion state persists across re-renders** (the `pathStates` Map in `web/sidebar/tree.js`). Saving never collapses the directory you were viewing.
- **Archive sub-folder is rendered *above* active workflows** in each directory, so the latest active workflow sits closest to the bottom — easy to spot.
- **No auto-pruning of archives.** Re-saving the same name many times leaves many timestamped archive entries. Right-click → Delete to remove individual ones when no longer useful.
- **Same-pattern distribution as `curated_defaults.json`** — see [`curated-sidebar.md`](curated-sidebar.md) for the seeding semantics that also apply here.
