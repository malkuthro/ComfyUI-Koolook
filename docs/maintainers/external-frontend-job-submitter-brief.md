# External Frontend Job Submitter Brief

This brief is for an AI or developer session building the real external
frontend that submits jobs to Koolook-published ComfyUI setups.

Primary reference:

- [`published-setup-external-ui-contract.md`](published-setup-external-ui-contract.md)
- [`published-setup-full-circle-plan.md`](published-setup-full-circle-plan.md)
- [`published-setups.md`](published-setups.md)

## Product Goal

Build an external app that lets a user run curated ComfyUI setups without
editing the ComfyUI graph.

The app should:

- list available published setups;
- render the user-facing controls declared by the setup author;
- accept a source path/file and output destination controls;
- submit a run to the local Koolook/ComfyUI server;
- show queued/running/succeeded/failed/lost state;
- show the returned result path.

The app must not inspect arbitrary ComfyUI nodes to decide what to render.
The setup author controls the frontend surface through the published setup
contract.

## What The Frontend Receives

The preferred source is the Koolook API.

List setups:

```text
GET /koolook/api/setups
```

Load one setup:

```text
GET /koolook/api/setups/{id}
```

The detail response is a published setup record. The important fields for the
frontend are:

```json
{
  "id": "rmgb-publish-v04",
  "metadata": {
    "title": "RMGB Publish",
    "description": "",
    "category": "",
    "tags": []
  },
  "validation": {
    "status": "valid",
    "diagnostics": []
  },
  "source": {
    "kind": "sidebar-workflow",
    "path": "Models/RMGB/Publish/rmgb-publish-v04",
    "inventoryPath": ["Models", "RMGB", "Publish"],
    "name": "rmgb-publish-v04"
  },
  "setupSurface": {
    "app": {
      "switch": {},
      "inputs": [],
      "outputs": [],
      "results": []
    }
  }
}
```

If the app is loading exported setup files instead of calling the API, use the
same shape. A file may contain one setup object or a registry wrapper:

```json
{
  "setups": []
}
```

## What To Look For

Use only:

- `setupSurface.app.switch`
- `setupSurface.app.inputs`
- `setupSurface.app.outputs`
- `setupSurface.app.results`
- `metadata`
- `validation`
- `source.inventoryPath`
- `source.name`

Ignore `visualGraph` for rendering. It exists for review/provenance.

Ignore `apiPrompt` for rendering. It exists for the server to execute the
setup.

## Catalog Grouping

Published setups preserve their sidebar inventory location in `source`.

Use:

```json
{
  "source": {
    "inventoryPath": ["Models", "RMGB", "Publish"],
    "name": "rmgb-publish-v04"
  }
}
```

`source.inventoryPath` is a structured breadcrumb array. A production frontend
can use it to recreate the same hierarchy the system administrator built in
the ComfyUI sidebar: topics, models, departments, show folders, or any other
local grouping convention.

Do not parse folder hierarchy from titles. Use `source.inventoryPath` when it
is present. `source.path` is a human-readable compatibility/display path.

## How The UI Is Constructed

### Mode Switch

Read `setupSurface.app.switch`.

Example:

```json
{
  "key": "switch",
  "label": "Input type",
  "default": 2,
  "options": [
    { "value": 0, "label": "EXR", "input": "sequence_folder", "visible": true },
    { "value": 1, "label": "QT", "input": "qt_file", "visible": true },
    { "value": 2, "label": "Img", "input": "single_file", "visible": true },
    { "value": 3, "label": "Prompt", "input": "prompt", "visible": false }
  ]
}
```

Render visible options only. Preserve and submit the option's `value`; do not
renumber options. The selected option's `input` field names the source input
field to show.

### Source Input Fields

Read `setupSurface.app.inputs`.

Typical first-version fields:

```text
sequence_folder  directory path for an EXR/image sequence
qt_file          full path to one specific QuickTime/video file
single_file      full path to one specific image/file
prompt           internal/shared prompt field, usually hidden
```

Path naming convention:

- A key ending in `_folder` means a directory path.
- A key ending in `_file` means one complete file path, including directory and
  filename.

When the user chooses `QT`, show `qt_file`.
When the user chooses `Img`, show `single_file`.
When the user chooses `EXR`, show `sequence_folder`.

### Output Controls

Read `setupSurface.app.outputs`.

Typical first-version fields:

```text
folder   output destination folder
name     output base name
version  version token/number
```

These values feed `Koolook_PublishOutput` in the ComfyUI graph. They are
shared naming/location controls, not final result values.

### Result

Read `setupSurface.app.results` to understand which result fields the setup
declares.

At runtime, display returned items from:

```json
{
  "run": {
    "outputs": [
      {
        "key": "result",
        "label": "Result",
        "type": "result",
        "items": [
          {
            "kind": "text",
            "value": "/resolved/output/path.mov"
          }
        ]
      }
    ]
  }
}
```

For the first version, treat `value` as the resolved path string. Previewing
images, movies, or folders can be added later based on extension/type.

## Run Submission

Submit only declared fields. For the first app surface, the payload normally
looks like:

```json
{
  "inputs": {
    "switch": 1,
    "qt_file": "/shots/shot010/source.mov",
    "folder": "/shots/shot010/output",
    "name": "shot010_rmgb",
    "version": "1"
  }
}
```

Run setup:

```text
POST /koolook/api/setups/{id}/run
```

Queued response:

```json
{
  "ok": true,
  "run": {
    "runId": "run-000001",
    "promptId": "comfy-prompt-id",
    "status": "queued"
  }
}
```

Poll run status:

```text
GET /koolook/api/runs/{runId}
```

Terminal statuses are:

```text
succeeded
failed
lost
```

Non-terminal statuses are:

```text
queued
running
```

`lost` means the prompt is no longer present in ComfyUI history or queue. Treat
it as terminal and show an actionable state failure instead of polling forever.

## Validation And Error Handling

Before rendering a run button:

- Require `validation.status === "valid"`.
- Require `setupSurface.app` to exist.
- Require a visible switch option or visible input fields.

On API errors, show the returned `errors[]` messages. Common causes:

- The setup is invalid or draft.
- The frontend submitted a key that is not declared by the setup.
- The local ComfyUI server could not queue the prompt.
- The server does not have a required custom node installed.

## Frontend Responsibilities

The external frontend should own:

- setup catalog browsing/filtering;
- optional catalog grouping by `source.inventoryPath`;
- form rendering from `setupSurface.app`;
- user input validation for empty paths and obvious missing fields;
- run submission and polling;
- run history/job list if the app is multiuser or multi-job;
- display of result paths;
- optional previews/open-in-folder actions later.

The frontend should not own:

- ComfyUI API prompt construction;
- custom-node compatibility;
- graph branch selection;
- output path calculation;
- inference from arbitrary node class names.

Those belong to the published setup contract and the ComfyUI workflow graph.

## Implementation Notes For An AI Session

Build the frontend around a small adapter:

```text
loadSetups() -> setup summaries
loadSetup(id) -> setup detail
buildForm(setup.setupSurface.app) -> fields and defaults
buildRunInputs(formState, appSurface) -> declared inputs only
runSetup(id, inputs) -> run id
pollRun(runId) -> status and outputs
extractResultPath(run.outputs) -> string or null
```

Use the existing simulator as a technical reference:

```text
web/setup_runner_simulator.html
web/setup_runner_simulator.js
```

The simulator is not the target UI quality bar for the production app. It is
a correctness harness that shows how to interpret the contract.
