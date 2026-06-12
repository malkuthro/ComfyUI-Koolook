# External Frontend Kit For Published Setups

This kit is the handoff package for an AI or developer session building a
production external frontend that runs Koolook-published ComfyUI setups.

It assumes the frontend is not embedded in ComfyUI. The frontend talks to the
running Koolook/ComfyUI server, renders controls from the published setup
contract, submits declared values, polls run status, and shows the resolved
result path.

## Current State

The current merged implementation has the core full-circle pieces needed by an
external frontend:

- Sidebar publishing captures ComfyUI's API-format prompt and stores it as
  `apiPrompt`.
- Published setup records preserve the visual graph, executable `apiPrompt`,
  inferred `setupSurface.app`, metadata, validation state, and sidebar source
  breadcrumbs.
- `source.inventoryPath` records the sidebar folder hierarchy so external apps
  can group setups the same way the ComfyUI sidebar was organized.
- The runner clones the stored `apiPrompt`, injects only declared fields,
  prunes switch-selected writer branches when an execution map is available,
  submits to the same local ComfyUI server, and reports queued/running/
  succeeded/failed/lost states.
- The simulator under `web/setup_runner_simulator.html` is a working technical
  reference for reading the API contract and submitting runs.

The remaining product work belongs mostly to the external app: interface
design, auth/session model if needed, job history, multiuser behavior, and
optional file previews/open actions.

## Files To Give The Other Session

Give the external frontend session these docs first:

- [`published-setup-external-ui-contract.md`](published-setup-external-ui-contract.md)
  explains the product contract, authoring nodes, app surface, result path, and
  source breadcrumb behavior.
- [`external-frontend-job-submitter-brief.md`](external-frontend-job-submitter-brief.md)
  is the shorter implementation brief.
- [`published-setups.md`](published-setups.md) documents storage, validation,
  routes, request/response shapes, and runner status behavior.
- [`published-setup-full-circle-plan.md`](published-setup-full-circle-plan.md)
  records the intended full loop and remaining non-core gaps.
- This file, [`external-frontend-kit.md`](external-frontend-kit.md), is the
  packaging checklist.

Give these code files as reference implementation, not as production UI:

- [`../../web/setup_runner_simulator.html`](../../web/setup_runner_simulator.html)
- [`../../web/setup_runner_simulator.js`](../../web/setup_runner_simulator.js)
- [`../../tests/sidebar/test_setup_runner_simulator.py`](../../tests/sidebar/test_setup_runner_simulator.py)

Give these code files if the session needs to understand the server boundary:

- [`../../koolook_routes.py`](../../koolook_routes.py)
- [`../../koolook_setups.py`](../../koolook_setups.py)
- [`../../koolook_setup_runner.py`](../../koolook_setup_runner.py)
- [`../../web/sidebar/published_setups.js`](../../web/sidebar/published_setups.js)

For the best start, also provide one real published setup detail JSON captured
from your live ComfyUI session after publishing from the sidebar:

```text
GET /koolook/api/setups/{id}
```

That real payload should include `apiPrompt`, `setupSurface.app`, and
`source.inventoryPath`. It is the most useful fixture for a separate frontend
AI session because it proves the exact setup shape coming out of your current
workflow.

## Runtime Dependencies

The external frontend needs:

- A running ComfyUI instance with the Koolook extension installed.
- The custom nodes required by the published setup installed in that same
  ComfyUI instance.
- Access to the Koolook API base URL, normally the local ComfyUI server such
  as `http://127.0.0.1:8188`.
- Published setup records in the Koolook registry, usually created by
  right-clicking a saved sidebar workflow and choosing **Publish setup...**.
- A browser/runtime strategy for same-origin requests, CORS, or a small proxy
  if the production frontend is served from a different origin.

The external frontend does not need to construct ComfyUI API prompts. The
Koolook server owns prompt cloning, value injection, custom-node execution,
branch pruning, and history/queue polling.

## API Surface

List setups:

```text
GET /koolook/api/setups
```

Load one setup:

```text
GET /koolook/api/setups/{id}
```

Run one setup:

```text
POST /koolook/api/setups/{id}/run
Content-Type: application/json

{ "inputs": { ...declared fields only... } }
```

Poll a run:

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

Treat `lost` as terminal. It means the prompt is missing from both ComfyUI
history and queue, so the frontend should stop polling and show an actionable
failure state.

## What The Frontend Should Render

Use only these setup detail fields to build the user-facing form:

- `metadata`
- `validation`
- `source.inventoryPath`
- `source.name`
- `setupSurface.app.switch`
- `setupSurface.app.inputs`
- `setupSurface.app.outputs`
- `setupSurface.app.results`

Do not render from arbitrary ComfyUI node classes. Do not render from
`apiPrompt`. Do not parse `visualGraph` to decide what the external user should
see.

The usual first-version app form is:

```text
Setup selector/catalog grouped by source.inventoryPath

Input type: EXR / QT / Img
Source path field:
  EXR -> sequence_folder
  QT  -> qt_file
  Img -> single_file

Output folder
Output name
Version

Run

Status and resolved result path
```

Path convention:

- Keys ending in `_folder` are directory paths.
- Keys ending in `_file` are full file paths, including directory and filename.

Preserve switch option values exactly. For example, if `QT` has `"value": 1`,
submit `1`; do not renumber visible options.

## Minimal Adapter The Frontend Should Build

The production frontend can be organized around a small adapter layer:

```text
loadSetups() -> setup summaries
loadSetup(id) -> setup detail
buildCatalogTree(setups, source.inventoryPath) -> grouped setup list
buildForm(setup.setupSurface.app) -> fields, defaults, visibility
buildRunInputs(formState, setup.setupSurface.app) -> declared inputs only
runSetup(id, inputs) -> { runId, promptId, status }
pollRun(runId) -> run status and outputs
extractResultPath(run.outputs) -> string or null
```

The simulator already contains working examples of most of this logic in
`web/setup_runner_simulator.js`.

## Example Run Payload

For a QuickTime input mode, the frontend normally submits:

```json
{
  "inputs": {
    "switch": 1,
    "qt_file": "/shots/shot010/source/source.mov",
    "folder": "/shots/shot010/output",
    "name": "shot010_rmgb",
    "version": "1"
  }
}
```

Only submit fields declared by the setup. If the frontend submits undeclared
keys, the API returns `400` with an `errors[]` payload.

## Result Handling

The first production behavior can be simple:

1. Show queued/running/succeeded/failed/lost status.
2. On terminal success, scan `run.outputs` for result items.
3. Display the returned `value` string as the resolved result path.

When present, `Koolook_PublishResult` is the preferred result source because
it represents the graph-selected resolved path. If it is not present, the
runner may still return selected writer/history output items for the selected
branch.

Previewing images, movies, or folders can be added later. The first frontend
does not need publish-time file-kind metadata as long as it displays the path
string returned by the run.

## Fixture Checklist

Before handing work to another AI session, capture these fixtures from the
current live server when possible:

- `GET /koolook/api/setups` response.
- `GET /koolook/api/setups/{id}` for one real published setup.
- One successful `POST /koolook/api/setups/{id}/run` response.
- One terminal `GET /koolook/api/runs/{runId}` success response that includes
  the resolved output path.
- One error response, such as an undeclared input key or missing setup id.

These fixtures let the external session build and test against the real
Koolook contract without needing to guess from prose.

## Gaps To Confirm Before Production

The core API contract is ready to build against, but a production app still
needs decisions that live outside this repository:

- How the app is served relative to ComfyUI: same origin, CORS, or proxy.
- Whether users browse a flat setup list or a tree grouped by
  `source.inventoryPath`.
- Whether the app is single-user/local only or needs auth and multiuser job
  ownership.
- Where run history lives if users need previous jobs after a page reload.
- Whether result paths should gain preview/open actions beyond showing the
  returned string.
