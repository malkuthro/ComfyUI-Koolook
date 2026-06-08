# Published Setup Full-Circle Plan

This document captures the remaining work needed to make the whole loop work
from ComfyUI authoring to an external frontend job submitter.

Related design docs:

- [`published-setup-external-ui-contract.md`](published-setup-external-ui-contract.md)
- [`published-setups.md`](published-setups.md)
- [`external-frontend-job-submitter-brief.md`](external-frontend-job-submitter-brief.md)

## Target Loop

The intended end-to-end flow is:

1. A setup author builds a workflow visually in ComfyUI.
2. The author places Koolook publish contract nodes in reserved setup areas:
   `Koolook_PublishInput`, `Koolook_PublishOutput`, optional
   `Koolook_PublishRouter`, and optional `Koolook_PublishResult`.
3. The author right-clicks the saved setup and chooses **Publish setup...**.
4. Koolook automatically captures:
   - the visual graph, for app-surface inference and review;
   - ComfyUI's API prompt export, for execution;
   - the inferred `setupSurface.app`, for external frontend rendering.
5. Koolook records the setup's sidebar inventory location, including structured
   breadcrumbs for the folder hierarchy where the setup was authored.
6. Koolook writes one published setup record using the registry schema.
7. An external frontend lists or loads the setup record.
8. The frontend may use the inventory breadcrumbs to group setups by the same
   topics, models, or administrator-defined folders used in the sidebar.
9. The frontend renders controls from `setupSurface.app`.
10. The user chooses an input mode, provides the matching source path/file,
   edits output location/name/version, and runs the setup.
11. Koolook injects the submitted values into the stored `apiPrompt`, queues it
   on the same local ComfyUI server, and polls the run status.
12. ComfyUI runs with the server-installed custom nodes.
13. `Koolook_PublishRouter` selects the writer branch; optional
   `Koolook_PublishResult` returns a custom selected/resolved result path.
14. The frontend shows the run confirmation and result path.

## Canonical Data Shape

The canonical shape is the published setup record returned by:

```text
GET /koolook/api/setups/{id}
```

An exported/offline setup file should use the same shape, either as one setup
object or as a registry wrapper:

```json
{
  "setups": [
    {
      "schemaVersion": 1,
      "id": "rmgb-publish-v04",
      "metadata": {},
      "visualGraph": {},
      "apiPrompt": {},
      "inputContract": { "inputs": [] },
      "outputContract": { "outputs": [] },
      "setupSurface": {
        "sourceInputs": [],
        "outputs": [],
        "controls": [],
        "app": {
          "switch": {},
          "inputs": [],
          "outputs": [],
          "results": []
        }
      },
      "source": {
        "kind": "sidebar-workflow",
        "path": "Models/RMGB/Publish/rmgb-publish-v04",
        "inventoryPath": ["Models", "RMGB", "Publish"],
        "name": "rmgb-publish-v04"
      },
      "validation": { "status": "valid", "diagnostics": [] }
    }
  ]
}
```

Do not create a separate external-app-only format. If an external app reads a
file directly, it should read this published setup record and use the same
fields as the API path.

### Sidebar Inventory Breadcrumbs

Published setups should preserve where they came from in the sidebar inventory.
This is not only provenance for maintainers; it gives external frontends a
ready-made hierarchy for browsing available setup features.

The sidebar may eventually be organized perfectly by system administrators:
topics, models, departments, show folders, publishing targets, or any other
local convention. A production external frontend should be able to recreate
that structure without guessing from setup titles.

Record the hierarchy as structured data, not only as a display string:

```json
{
  "source": {
    "kind": "sidebar-workflow",
    "path": "Models/RMGB/Publish/rmgb-publish-v04",
    "inventoryPath": ["Models", "RMGB", "Publish"],
    "name": "rmgb-publish-v04"
  }
}
```

`source.path` remains useful for humans and compatibility. `source.inventoryPath`
is the folder breadcrumb array the external frontend can use to group published
setups. `source.name` is the workflow/setup name as it appeared inside that
folder.

## App Surface Contract

The external frontend uses `setupSurface.app`, not arbitrary ComfyUI nodes.

Required first-version shape:

```json
{
  "setupSurface": {
    "app": {
      "switch": {
        "key": "switch",
        "label": "Input type",
        "default": 2,
        "target": { "node": "100", "input": "mode" },
        "options": [
          { "value": 0, "label": "EXR", "input": "sequence_folder", "visible": true },
          { "value": 1, "label": "QT", "input": "qt_file", "visible": true },
          { "value": 2, "label": "Img", "input": "single_file", "visible": true },
          { "value": 3, "label": "Prompt", "input": "prompt", "visible": false }
        ]
      },
      "inputs": [
        {
          "key": "sequence_folder",
          "label": "Sequence folder",
          "target": { "node": "100", "input": "sequence_folder" },
          "default": ""
        },
        {
          "key": "qt_file",
          "label": "QT file",
          "target": { "node": "100", "input": "qt_file" },
          "default": ""
        },
        {
          "key": "single_file",
          "label": "Single file",
          "target": { "node": "100", "input": "single_file" },
          "default": ""
        }
      ],
      "outputs": [
        {
          "key": "folder",
          "label": "Output folder",
          "target": { "node": "200", "input": "folder" },
          "default": ""
        },
        {
          "key": "name",
          "label": "Output name",
          "target": { "node": "200", "input": "name" },
          "default": ""
        },
        {
          "key": "version",
          "label": "Version",
          "target": { "node": "200", "input": "version" },
          "default": "1"
        }
      ],
      "results": [
        {
          "key": "result",
          "label": "Result",
          "target": { "node": "300", "input": "result" },
          "default": ""
        }
      ]
    }
  }
}
```

If a future `config.ui` authoring script is introduced or restored, it should
normalize to this same `setupSurface.app` shape. The external frontend should
not need to know whether the surface came from publish nodes, a config UI
script, or another authoring helper.

## Completed In PR 231

- Published setup catalog and detail routes.
- Published setup run and run-status routes.
- Optional `apiPrompt` storage on publish requests.
- Server-side validation for declared app inputs, app output controls, and app
  switch values.
- `Koolook_PublishInput`, `Koolook_PublishOutput`, and
  `Koolook_PublishResult` contract nodes.
- `Koolook_PublishResult` emits UI text history so the runner can return a
  selected result path when a setup uses an explicit reporting node.
- External runner simulator under `web/setup_runner_simulator.html`.
- Simulator form rendering from `setupSurface.app`.

## Gaps To Close

### 1. Automatic API Prompt Capture

The sidebar **Publish setup...** action must automatically capture the ComfyUI
API prompt and pass it as `apiPrompt`.

Acceptance criteria:

- The author does not manually export or paste API JSON for normal setup
  publishing.
- The stored `apiPrompt` is the exact executable prompt shape ComfyUI would
  submit/export for the graph.
- Custom nodes, subgraphs, and ComfyUI widget serialization survive unchanged.
- If API prompt capture fails, publish fails with a clear diagnostic instead
  of silently falling back to a broken visual conversion for custom-node
  workflows.

### 2. Exported File Shape

The exporter must produce a published setup record or registry wrapper that
contains both `apiPrompt` and `setupSurface.app`.

Acceptance criteria:

- The exported file can be loaded by the simulator or an external frontend
  without guessing from visual graph nodes.
- The file uses the same schema as `GET /koolook/api/setups/{id}` or the
  registry `{ "setups": [...] }` wrapper.
- The file preserves `source.inventoryPath` and `source.name` so external
  frontends can rebuild the sidebar hierarchy if desired.
- The file includes validation diagnostics when a setup is draft/invalid.
- Exported setup records keep stable ids and metadata.

### 3. Publish Modal Simplification

The publish modal still exposes advanced contract JSON as primary UI. For
publish-node setups, this should become secondary.

Acceptance criteria:

- Normal publish-node setups require metadata and confirmation, not hand-written
  input/output contract JSON.
- Advanced JSON remains available for diagnostics or unusual workflows.
- The modal previews the inferred app surface so authors can verify what the
  external frontend will show.

### 4. Simulator File Loading

The simulator currently loads setups through the Koolook API and demo data. If
the product requires direct exported-file review, add a file loader that accepts
the canonical published setup shape.

Acceptance criteria:

- The simulator can load one setup object or `{ "setups": [...] }`.
- The same renderer is used for API-loaded and file-loaded setups.
- The raw JSON payload remains secondary/debug-only.

### 5. Run Result Robustness

The first result contract is a selected writer/history item, with an optional
path string from `Koolook_PublishResult` when authors need custom reporting.

Acceptance criteria:

- Successful runs return `outputs[]` with a `result` summary and at least one
  item whose `value` is the resolved path.
- The result path reflects the graph-selected branch and submitted output
  controls.
- Failed ComfyUI runs return actionable error payloads.

### 6. External Production Frontend

The simulator is a maintainer harness. A production external app still needs
its own design, auth/session model, job list, run history, and multiuser
behavior.

Acceptance criteria:

- It consumes the published setup API or exported setup file shape documented
  here.
- It can optionally group setup cards by `source.inventoryPath`, matching the
  administrator-defined sidebar folder structure.
- It renders controls only from `setupSurface.app`.
- It submits only declared keys.
- It shows queued/running/succeeded/failed/lost states.
- It displays the returned result path and can later add previews/open actions.

## Suggested Implementation Order

1. Implement and test automatic API prompt capture in sidebar publish.
2. Make published/exported setup files use the canonical setup record shape.
3. Simplify publish modal UX around inferred publish-node surfaces.
4. Add optional direct-file loading to the simulator if needed.
5. Hand the external frontend brief to the separate app session.
