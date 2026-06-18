# Published Setup External UI Contract

This document captures the product intent for published setups as of the
issue #224 runner-simulator PR. It is intentionally about the external
application contract, not only the ComfyUI execution mechanics.

Companion docs:

- [`published-setup-full-circle-plan.md`](published-setup-full-circle-plan.md)
  tracks the remaining implementation work needed to complete the authoring,
  publish, export, simulator, and run loop.
- [`external-frontend-job-submitter-brief.md`](external-frontend-job-submitter-brief.md)
  is the handoff brief for an AI or developer session building the production
  external frontend.

## Problem Statement

A setup author builds a visual workflow inside ComfyUI. The author places
specific Koolook publish nodes in that workflow to declare which controls an
external application should expose to its own end user.

Publishing the setup must preserve enough machine-readable information for an
external frontend to auto-populate its interface without inspecting arbitrary
ComfyUI nodes or guessing from the executable prompt.

The external frontend's job is to:

- list published setups,
- render the exposed setup inputs and output controls,
- submit user-provided values,
- run the setup on the local ComfyUI server that has the required custom nodes,
- show the returned result/path information.

The external frontend should not reverse-engineer ComfyUI graph structure. It
should read the published setup contract.

The external frontend may also use the published setup's `source` metadata to
recreate the sidebar inventory hierarchy. This lets administrators organize
setups in the ComfyUI side panel by topic, model, department, show, or any
other local convention, and gives external apps the same grouping without
parsing setup titles.

## Actors

- **Setup author**: the developer/curator working visually inside ComfyUI.
- **External frontend**: a separate app that consumes the Koolook setup API.
- **External app user**: the person using that separate app to run a curated
  setup without editing the ComfyUI graph.
- **Koolook publish API**: the boundary that stores the authored app contract
  plus the executable ComfyUI API prompt.

## Source Of Truth

Published setups carry two related but separate artifacts:

1. **Visual graph**
   - Used to infer the external UI contract from Koolook publish nodes and
     reserved groups.
   - Used for review, provenance, and future editing.
   - Not the preferred executable source.

2. **ComfyUI API prompt**
   - Produced by ComfyUI's own API workflow export.
   - Used as the execution source of truth for `/prompt`.
   - Preserves custom nodes, subgraphs, and widget serialization exactly as
     ComfyUI expects.

The implementation now allows publish requests to supply `apiPrompt`; when it
is supplied, the registry stores it instead of re-converting the visual graph.
The fallback converter only exists for older/simple flows and should not be the
main product path.

## Authoring Nodes

Setup authors use controlled Koolook publish nodes to declare the external app
surface.

Reserved groups and publish nodes have separate jobs:

- Reserved groups (`Koolook Input`, `Koolook Output`) are spatial authoring
  areas. They let the publisher find the relevant part of the visual graph and
  keep a review/provenance summary of nodes in those areas.
- Publish contract nodes are the machine-readable external UI contract. The
  publisher looks for node class/type, not display title, inside the matching
  reserved group.

The external app surface should be inferred from these nodes:

| Group | Contract node class |
|---|---|
| `Koolook Input` | `Koolook_PublishInput` |
| `Koolook Output` | `Koolook_PublishOutput` |
| `Koolook Output` | `Koolook_PublishRouter` |
| `Koolook Output` | `Koolook_PublishResult` optional custom reporting |

Authors do not need to put every surrounding workflow node inside the reserved
groups. Prefer keeping the groups focused on the publish contract nodes and any
nearby nodes that are useful for human review. The publisher may preserve all
nodes that overlap the groups in `setupSurface.sourceInputs` and
`setupSurface.outputs`, but only the recognized `Koolook_Publish*` node classes
drive `setupSurface.app`.

Current implementation detail: if multiple nodes of the same publish contract
class are inside the same reserved group, the publisher uses the first matching
node in the group's top-to-bottom, left-to-right ordering. Authors should keep
one `Koolook_PublishInput` per setup input area, and one
`Koolook_PublishOutput` plus optional router/result nodes per setup output
area until multi-surface publishing is explicitly designed.

### `Koolook_PublishInput`

Placed in the `Koolook Input` area. It declares source-side controls for the
external app:

| Field | Purpose |
|---|---|
| `mode` | User-facing source mode. Current options: `EXR`, `QT`, `Img`, `Prompt`. |
| `sequence_folder` | Path to an EXR/image sequence folder. |
| `qt_file` | Full path to one specific QuickTime/video file, including directory and filename. |
| `single_file` | Full path to one specific single image/file, including directory and filename. |
| `prompt` | Shared/internal prompt field for setups that need it. |
| `switch` | Integer output derived from `mode`, used by the graph to route the chosen branch. |

The external frontend should render `mode` first, then show the matching path
field for the selected mode. It should preserve the numeric switch values
because the graph depends on them.

`prompt` is an **always-on** field: it is emitted with `standalone: true` and
renders independently of the `mode` switch (the user picks a source **and** can
describe the shot), so it is never offered as a source-mode option. Its widget
text is surfaced as a `placeholder` hint, not a submitted default — the default
is empty so an untouched hint is never sent as the real prompt. A `help` string
carries the phrasing guidance ("subject + action + setting"). Wire the
`Koolook_PublishInput` `prompt` output into the graph's positive prompt to use
it.

Naming convention: a field ending in `_folder` means a directory path. A field
ending in `_file` means a full file path string, including both the directory
path and the selected filename. This matters for folders that may contain many
candidate movies, stills, or other source files.

### `Koolook_PublishOutput`

Placed in the `Koolook Output` area. It declares shared output
naming/location controls for the external app and downstream writer/path nodes:

| Field | Purpose |
|---|---|
| `folder` | Destination folder for generated outputs. |
| `name` | Base output name. |
| `version` | Version token/number used by the graph's naming logic. |

The external frontend can expose these as editable output controls when the
setup author wants the user to choose where generated files should land.
`Koolook_PublishOutput` is an intermediate contract node: it does not choose
the final result and should not own a mode-specific `result` field. Its outputs
feed the writer/path-building nodes for each supported branch.

### `Koolook_PublishRouter`

Placed in the `Koolook Output` area when one setup switch selects between
multiple writer branches. It declares which writer branch should execute for
each `Koolook_PublishInput.switch` value:

| Field/output | Purpose |
|---|---|
| `selector` | INT input wired from `Koolook_PublishInput.switch`. |
| `payload` | Main setup payload, such as the mask/image result from the body graph. |
| `EXR` | Output slot 0, connected to the EXR writer branch. |
| `QT` | Output slot 1, connected to the QuickTime/movie writer branch. |
| `Img` | Output slot 2, connected to the image writer branch. |
| `Prompt` | Output slot 3, connected to a prompt/no-op branch when needed. |

When a setup is published, Koolook stores an `executionMap` from this router:
selected switch value -> router output slot -> writer nodes. During external
runs, the runner uses that map to prune unselected writer branches before
queueing the ComfyUI API prompt.

### `Koolook_PublishResult`

Optional node in the `Koolook Output` area. It represents a custom final result
value the external app should show after the graph has selected/resolved the
output.

The intended result is usually a path string. Depending on the selected mode,
the graph may calculate:

- an output image/file path,
- an output QuickTime/movie path,
- an output sequence folder/path,
- or another resolved output/status path string.

Those branch-specific calculated values may be routed through graph switch
logic into `Koolook_PublishResult`. The external frontend should display this
single selected result when present and may later decide how to preview or open
the path based on extension/type. When the result node is omitted, the runner
can still execute the selected writer branch through `Koolook_PublishRouter`;
the frontend should display returned writer/history items.

Decision: the first public result contract is the resolved path string. The
important requirement is that the string matches what the real setup resolved
inside ComfyUI. A frontend may optionally preview images or other known file
types later, but publish-time result metadata does not need to declare file
kind for the first version.

## Expected External Frontend UI

The external frontend should be able to build this kind of form from the
published setup detail:

```text
Input type:  [EXR] [QT] [Img]

If EXR:
  Sequence folder: <path>

If QT:
  QuickTime file: <path>

If Img:
  Single file: <path>

Output folder: <path>
Output name:   <name>
Version:       <version>

Run setup

Result:
  <resolved output path>
```

The frontend should read this from `setupSurface.app`, not from arbitrary
ComfyUI node classes:

- `setupSurface.app.switch`
- `setupSurface.app.inputs`
- `setupSurface.app.outputs`
- `setupSurface.app.results`

The executable `apiPrompt` exists so the backend can run the setup. It is not
the external UI schema.

## Inventory Source Metadata

Published setups should preserve their sidebar inventory location:

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

`source.inventoryPath` is the structured folder breadcrumb array. External
frontends can use it to group setup cards or rebuild the same hierarchy from
the ComfyUI sidebar. `source.path` is the human-readable full path kept for
compatibility and display. `source.name` is the workflow/setup name inside
that folder.

## Publish-Time Flow

The intended publish flow is:

1. The setup author builds the workflow visually in ComfyUI.
2. The author places publish contract nodes in the reserved setup areas.
3. Koolook captures the visual graph for UI-contract inference.
4. Koolook captures ComfyUI's exported API prompt for execution.
5. The registry stores both in one published setup record.
6. The registry infers `setupSurface.app` from the publish nodes.
7. External apps consume the published setup detail to render their UI.

The publish flow should not require the setup author to manually write JSON
contracts for ordinary publish-node setups. It also should not require the
setup author to manually export an API JSON file. When the author right-clicks
and chooses **Publish setup...**, Koolook should automatically capture the
same API prompt shape that ComfyUI would produce via its API export action and
store that prompt as `apiPrompt`. Advanced JSON contracts are a fallback for
unusual cases.

## Run-Time Flow

At run time:

1. The external frontend sends values only for declared fields.
2. Koolook validates the submitted keys against the setup contract.
3. Koolook clones the stored `apiPrompt`.
4. Koolook injects submitted values into the declared prompt targets.
5. Koolook queues the prompt on the same local ComfyUI server.
6. ComfyUI executes the workflow with installed custom nodes.
7. Koolook polls history/queue state and returns run status plus declared
   outputs/results.

The external frontend should not choose between right-side graph nodes. The
authoring graph should route the selected mode internally and expose one
published result field.

In other words, the intended output path flow is:

```text
Koolook_PublishInput.switch -> Koolook_PublishRouter.selector
main payload -> Koolook_PublishRouter.payload
Koolook_PublishOutput(folder/name/version) -> branch-specific path nodes
Koolook_PublishRouter outputN -> selected writer branch
optional calculated result path -> Koolook_PublishResult(result)
```

## Current Implementation

Implemented:

- Catalog and detail routes for published setup records.
- Run routes that clone the stored `apiPrompt`, inject declared inputs, queue
  ComfyUI `/prompt`, and poll history/queue status.
- Sidebar publish captures ComfyUI's API-format prompt automatically for normal
  publish-node setup publishing.
- External runner simulator as a separate app under `web/`; it renders the
  `setupSurface.app` mode switch, active source path field, output controls,
  generated JSON payload, run status, ComfyUI prompt id, and returned result
  path.
- The simulator can also load local published setup records for contract
  review; running still uses the live Koolook registry API.
- `apiPrompt` in publish requests is preserved as the execution source of
  truth.
- Fallback visual-to-API conversion repairs some legacy/editor artifacts such
  as labels, reroutes, and subgraph wrapper ids.
- `setupSurface.app` support in the server schema and runner summaries.
- Switch-selected writer execution maps let the runner prune unselected writer
  branches and report the selected result/writer output.
- Publish contract nodes:
  - `Koolook_PublishInput`
  - `Koolook_PublishOutput`
  - `Koolook_PublishRouter`
  - `Koolook_PublishResult`

Remaining gaps:

- The publish modal still exposes advanced contract JSON. For publish-node
  setups, this should become secondary because the graph nodes should define
  the app surface.
- The simulator is still a maintainer harness rather than the polished
  production external app, but the normal path is now the app-surface form
  instead of raw JSON.

## Non-Goals

- The external frontend should not parse arbitrary ComfyUI nodes to infer UI.
- The external frontend should not decide which right-side output branch to use.
- The fallback visual converter should not be treated as a replacement for
  ComfyUI's own API export.
- The runner simulator should not add debug controls to the ComfyUI sidebar.

## Resolved Design Decisions

The sidebar **Publish setup...** action should automatically capture
the ComfyUI API prompt. The setup author should not manually press "Export as
API" and attach a JSON file. The reference behavior is "whatever ComfyUI would
submit/export as API for this graph"; implementation should choose the safest
and simplest path that preserves that exact prompt shape.

External input visibility is encoded by the mode switch for the first version.
The external frontend should read `setupSurface.app.switch.options[*].input`:
each visible switch option points to the one input field to show for that mode.
Per-field `visibleWhen` metadata is reserved for future multi-condition
controls.
