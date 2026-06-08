# Published workflow setup registry

Issue #210 introduced the first server-side registry for published workflow
setups. Issue #211 adds the curator publish path from saved sidebar workflows,
and issue #212 adds publish-time conversion from visual workflow graphs to
Comfy API prompt JSON for the first supported callable setup shape.
The registry remains separate from sidebar workflow and snapshot storage so
external apps can consume a stable catalog without knowing Kforge Labs sidebar
internals.

For the product/design contract behind the external frontend surface, read
[`published-setup-external-ui-contract.md`](published-setup-external-ui-contract.md).
This document focuses on storage, validation, routes, and implementation
details.

## Storage

The registry loads JSON from:

```text
<ComfyUI user directory>/koolook-published-setups/setups.json
```

The file may be either a bare array of setup objects or an object with a
`setups` array:

```json
{
  "setups": []
}
```

When the user registry file does not exist, the catalog falls back to the
bundled sample at [`../../web/published_setups_sample.json`](../../web/published_setups_sample.json).
An existing empty `setups.json` remains empty. A corrupt or unreadable primary
`setups.json` does not fall back to the sample, because that would hide the
operator's real registry problem; the catalog returns no rows and logs a
diagnostic instead. Publishing a setup writes to the primary user registry file.

## Schema

Each published setup object uses this shape:

```json
{
  "schemaVersion": 1,
  "id": "stable-url-safe-id",
  "version": 1,
  "updatedAt": "2026-06-06T08:00:00Z",
  "metadata": {
    "title": "Setup title",
    "description": "Optional user-facing description",
    "category": "Video",
    "tags": ["tag"],
    "previewImage": ""
  },
  "visualGraph": {},
  "apiPrompt": {
    "12": {
      "class_type": "Text Multiline",
      "inputs": {
        "text": "Default prompt"
      }
    }
  },
  "inputContract": {
    "inputs": [
      {
        "key": "prompt",
        "label": "Prompt",
        "type": "text",
        "required": true,
        "target": {
          "node": "12",
          "input": "text"
        }
      }
    ]
  },
  "outputContract": {
    "outputs": [
      {
        "key": "video",
        "label": "Video",
        "type": "video"
      }
    ]
  },
  "setupSurface": {
    "sourceInputs": [
      {
        "group": "Koolook Input",
        "nodes": [{ "id": "12", "type": "Load Image", "title": "Source image" }]
      }
    ],
    "outputs": [
      {
        "group": "Koolook Output",
        "nodes": [{ "id": "20", "type": "Preview Image", "title": "Preview" }]
      }
    ],
    "controls": []
  },
  "source": {
    "kind": "sidebar-workflow",
    "path": "Models/RMGB/Publish/rmgb-publish-v04",
    "inventoryPath": ["Models", "RMGB", "Publish"],
    "name": "rmgb-publish-v04"
  },
  "validation": {
    "status": "valid",
    "diagnostics": []
  }
}
```

`apiPrompt` is nullable only for non-callable draft records, such as bundled
catalog smoke-test samples. A setup with `validation.status: "valid"` must have
a concrete API prompt. Prefer a ComfyUI-exported API prompt as the execution
source of truth; visual-graph conversion is only a fallback for older/simple
flows. Execution slices should require `validation.status: "valid"` and a
concrete `apiPrompt` before a setup can run. Only `schemaVersion: 1` is
accepted by this first registry; future schema versions must add explicit
migration or validation support before they pass.

## Public Boundary

The registry module is [`../../koolook_setups.py`](../../koolook_setups.py).
External callers should use:

- `PublishedSetupRegistry.listSetups()`
- `PublishedSetupRegistry.getSetup(id)`
- `PublishedSetupRegistry.publishSetup(visualGraph, metadata, inputContract, outputContract, source, apiPrompt=None)`

Execution lives behind [`../../koolook_setup_runner.py`](../../koolook_setup_runner.py).
External routes and future app adapters should use:

- `PublishedSetupRunner.runSetup(id, inputs)`
- `PublishedSetupRunner.getRun(runId)`

The runner hides prompt cloning, input injection, ComfyUI `/prompt`
submission, run-id mapping, history/queue polling, and output flattening from
route callers.

Invalid setup objects are omitted from list/detail results. Diagnostics are
kept on `registry.diagnostics` and logged by the HTTP adapter. File-level
storage diagnostics, such as unreadable JSON, are reported through the same
channel.

`publishSetup` builds a setup from a saved sidebar workflow graph and, when
provided, stores ComfyUI's exported API prompt as the executable source of
truth. If no API prompt is supplied, the registry falls back to the narrow
visual-graph converter. Publishing validates required metadata and contract
shape, checks that each declared input target points at both the visual graph
and the stored API prompt, then replaces any existing setup with the same id in
storage. Unsupported graphs fail publish with curator diagnostics rather than
being silently stored as callable records.

## Catalog API

`GET /koolook/api/setups`

Returns catalog summaries only: id, version, updated timestamp, metadata,
validation status, and input/output summaries. It intentionally does not return
`visualGraph`, `apiPrompt`, or raw storage internals.

`GET /koolook/api/setups/{id}`

Returns the full published setup contract for one setup, including visual graph,
optional API prompt, full input/output contracts, source reference, and
validation state. Treat `validation.status: "valid"` plus a non-null
`apiPrompt` as the public signal that the prompt is current and executable.
Unknown, invalid, or stale setup ids return `404`.

`POST /koolook/api/setups`

Publishes one setup. Body:

```json
{
  "visualGraph": {},
  "apiPrompt": {
    "12": { "class_type": "Text Multiline", "inputs": { "text": "prompt" } }
  },
  "metadata": {
    "id": "director-demo",
    "title": "Director Demo",
    "description": "A curated video workflow",
    "category": "Video",
    "tags": ["director", "video"],
    "previewImage": ""
  },
  "inputContract": {
    "inputs": [
      {
        "key": "prompt",
        "label": "Prompt",
        "type": "text",
        "required": true,
        "target": { "node": "12", "input": "text" }
      }
    ]
  },
  "outputContract": {
    "outputs": [{ "key": "preview", "label": "Preview", "type": "image" }]
  },
  "source": {
    "kind": "sidebar-workflow",
    "path": "Demos/Director Demo",
    "inventoryPath": ["Demos"],
    "name": "Director Demo"
  }
}
```

Success returns `{ "ok": true, "setup": { ... } }` with the stored
`apiPrompt` alongside the original `visualGraph`. Validation failures
return HTTP `400` with `{ "ok": false, "errors": [...] }`.

`source.inventoryPath` is the sidebar folder breadcrumb array for the workflow
that was published. External frontends may use it to recreate the sidebar's
administrator-defined catalog hierarchy. `source.path` remains the
human-readable compatibility path, and `source.name` is the workflow/setup name
inside that folder.

## Run API

`POST /koolook/api/setups/{id}/run`

Queues a callable published setup on the managed ComfyUI server. The body must
be a JSON object with an `inputs` object:

```json
{
  "inputs": {
    "prompt": "A cinematic close-up"
  }
}
```

Only fields declared by `inputContract.inputs`, `setupSurface.app.inputs`, or
`setupSurface.app.outputs` are accepted. `setupSurface.app.switch` is also
accepted when present. The runner deep-clones the stored `apiPrompt`, injects
approved values into their declared targets, and submits
`{ "prompt": <cloned prompt> }` to ComfyUI `/prompt`; the stored setup record
is not mutated by a run request.

Success returns:

```json
{
  "ok": true,
  "run": {
    "runId": "run-000001",
    "promptId": "f0c2...",
    "status": "queued"
  }
}
```

Failure responses use the standard route error shape:

```json
{
  "ok": false,
  "errors": ["input 'seed' is not declared by this setup"]
}
```

Missing setup ids return `404`; invalid inputs and non-callable draft setups
return `400`; ComfyUI queue failures return `502`.

`GET /koolook/api/runs/{runId}`

Looks up a stable Koolook run id and translates ComfyUI history/queue data into
external state:

```json
{
  "ok": true,
  "run": {
    "runId": "run-000001",
    "setupId": "ltx-director-demo",
    "promptId": "f0c2...",
    "status": "succeeded",
    "comfyStatus": {
      "completed": true,
      "status_str": "success"
    },
    "outputs": [
      {
        "key": "video",
        "label": "Video",
        "type": "video",
        "items": [
          {
            "nodeId": "20",
            "kind": "videos",
            "filename": "demo.mp4",
            "subfolder": "koolook",
            "type": "output"
          }
        ]
      }
    ]
  }
}
```

Status is one of `queued`, `running`, `succeeded`, `failed`, or `lost`.
History entries produce terminal state, the raw ComfyUI `status` object as
`comfyStatus`, and output items; otherwise the runner checks ComfyUI queue
data to distinguish `running` from still `queued`. If a prompt is missing from
both history and queue, the runner reports terminal status `lost` so clients do
not poll until timeout. Unknown run ids return `404`; ComfyUI status lookup
failures return `502`.

For group-authored setups, status output summaries also include
`setupSurface.app.outputs` and `setupSurface.app.results`. This keeps the
external app aligned with the publish-contract-node surface even when
`outputContract.outputs` is empty. Result fields include their declared target,
default value, visibility, and any matching ComfyUI history items for the
target node. `Koolook_PublishResult` emits its resolved string through
ComfyUI UI text history, and the runner flattens that into a result item whose
`value` is the selected output path.

## Comfy-Native Setup Surface

Issue #219 amends the #209 direction: curators should define a setup's app
surface visually in ComfyUI where possible, while the backend stores the
machine-readable contract. The reserved group names are:

- `Koolook Input`: required source/input area for app-style setups. Put
  `Koolook_PublishInput` here. Nearby source/helper nodes may overlap this
  group for human review, but they do not define the external app contract.
- `Koolook Output`: required output/result area. Put `Koolook_PublishOutput`
  and `Koolook_PublishResult` here. Nearby save/preview/helper nodes may
  overlap this group for human review, but they do not define the external app
  contract.
- `Koolook Controls`: future optional controls area for prompt, seed, strength,
  size, mode, or other user-tweakable fields.

Sidebar selection saves preserve ComfyUI groups that overlap selected nodes.
Publish infers a machine-readable `setupSurface` from the reserved groups.
Group membership is spatial: nodes whose rectangles overlap the reserved group
are listed in `sourceInputs` / `outputs`. The external app surface under
`setupSurface.app` is narrower: it is inferred only from recognized
`Koolook_Publish*` node classes inside the matching reserved group.

```json
{
  "sourceInputs": [
    {
      "group": "Koolook Input",
      "nodes": [{ "id": "12", "type": "Load Image", "title": "Source image" }]
    }
  ],
  "outputs": [
    {
      "group": "Koolook Output",
      "nodes": [{ "id": "20", "type": "Preview Image", "title": "Preview" }]
    }
  ],
  "controls": [],
  "app": {
    "inputs": [],
    "outputs": []
  }
}
```

When the publish dialog uses the group-first path, `inputContract.inputs` and
`outputContract.outputs` are submitted as empty arrays and the server requires
non-empty `Koolook Input` and `Koolook Output` groups. Explicit JSON contracts
still work as the advanced fallback for older or unusual workflows.

### Publish Contract Nodes

Use the controlled Koolook publish nodes instead of scattered third-party text
nodes when a setup should be callable from an external app:

```text
Koolook Publish Input   -> place in Koolook Input
Koolook Publish Output  -> place in Koolook Output
Koolook Publish Result  -> place in Koolook Output
```

`Koolook Publish Input` exposes stable multiline fields and outputs:

```text
mode             dropdown: EXR, QT, Img, Prompt
sequence_folder  STRING
qt_file          STRING
single_file      STRING
prompt           STRING
switch           INT output derived from mode
```

`Koolook Publish Output` exposes stable fields and outputs:

```text
folder   STRING
name     STRING
version  STRING
```

It is the shared destination/naming parameter node for downstream writer and
path-building branches. It does not own the final mode-selected result.

`Koolook Publish Result` exposes the resolved result value after workflow
writer/path logic has run:

```text
result   STRING
```

Route the calculated image/movie/sequence result path selected by the workflow
switch into `Koolook Publish Result`.

Publish detects these node classes and stores `setupSurface.app` with stable
keys, user-facing labels, defaults, injection targets, result targets, and
switch options. The external app should render the switch first, preserve the
numeric switch values, and hide internal-only options such as Prompt while
keeping their index stable for the workflow. For the first version, field
visibility comes from `setupSurface.app.switch.options[*].input`: the selected
visible option names the one source input field to show.

## Callable API Prompt Standard

Prefer ComfyUI's own API workflow export when publishing a callable setup. That
export is the same prompt shape submitted to `/prompt`, so it preserves custom
nodes exactly as the live server executes them and avoids editor-only artifacts
such as labels, reroutes, EasyUse state helpers, and subgraph wrapper ids.
The sidebar publish action should capture this automatically; manual API JSON
export is a diagnostic/testing workaround, not the product workflow.

Koolook still stores the visual workflow separately for setup surface inference,
review, and future editing. The visual graph is not treated as the executable
source when a Comfy-exported `apiPrompt` is present.

## Callable Visual Workflow Fallback

The first supported conversion shape is intentionally narrow:

- `visualGraph.nodes` must be a list of node objects with stable `id` and
  non-empty `type`.
- Each converted API node is keyed by the visual node id as text and stores
  `class_type` from the visual node `type`.
- Widget-backed inputs must appear in `node.inputs` with a `name` and
  `widget` object. Their values are read from `node.widgets_values` in widget
  input order.
- Nodes may also save `widgets_values` as an object keyed by widget/input name;
  those values are read by name.
- Subgraph wrapper nodes are expanded from the matching
  `definitions.subgraphs` entry and internal node ids are namespaced as
  `<wrapper-id>:<internal-id>`.
- Some simple Comfy nodes serialize widget values without corresponding
  `node.inputs` entries. The converter supports known widget-only mappings for
  `Text Multiline` (`text`) and Koolook `EasyAIPipeline` so simple grouped
  setup workflows can still publish into runnable API prompt inputs.
- Linked inputs must have a `link` id that resolves in `visualGraph.links`.
  Array links such as `[101, 12, 0, 20, 0, "STRING"]` and object links with
  `origin_id` / `origin_slot` are supported. The API prompt value becomes
  `["12", 0]`.
- Visual-only nodes such as `Label (rgthree)` and `Note` are omitted. `Reroute`
  nodes are resolved as passthrough links instead of being submitted as API
  nodes.
- Partial/module workflow sentinel links, such as links from node `-10`, are
  not callable yet and fail publish.

Example visual input:

```json
{
  "nodes": [
    {
      "id": 12,
      "type": "Text Multiline",
      "inputs": [{ "name": "text", "widget": { "name": "text" } }],
      "widgets_values": ["Default prompt"]
    },
    {
      "id": 20,
      "type": "Text Concatenate",
      "inputs": [
        { "name": "text_a", "link": 101 },
        { "name": "delimiter", "widget": { "name": "delimiter" } }
      ],
      "widgets_values": ["_"]
    }
  ],
  "links": [[101, 12, 0, 20, 0, "STRING"]]
}
```

Generated API prompt:

```json
{
  "12": {
    "class_type": "Text Multiline",
    "inputs": { "text": "Default prompt" }
  },
  "20": {
    "class_type": "Text Concatenate",
    "inputs": {
      "text_a": ["12", 0],
      "delimiter": "_"
    }
  }
}
```

## Publish Diagnostics

Curators should treat publish diagnostics as setup authoring feedback:

- `visualGraph.nodes must be a list`: the submitted workflow is not a visual
  graph object.
- `visualGraph.nodes[N].type must be non-empty text`: a node is missing the
  Comfy class/type needed for API prompt conversion.
- `visualGraph.nodes[N].id duplicates visualGraph.nodes[M].id`: duplicate node
  ids would overwrite generated API prompt entries, so publish is rejected.
- `visualGraph.nodes[N].inputs must be a list when present`: the node has a
  malformed input list. Missing `inputs` is allowed for nodes with no inputs.
- `visualGraph.links[N].id duplicates visualGraph.links[M].id`: duplicate link
  ids would make linked inputs ambiguous, so publish is rejected.
- `visualGraph.links[ID].origin_id not found in visualGraph`: a linked input
  points to a source node that is not present in the submitted graph.
- `visualGraph.links[ID].origin_slot must be a non-negative integer`: a linked
  input has a malformed source output slot.
- `visualGraph.links[ID].target does not match visualGraph.nodes[N].inputs[M]`:
  a visual input references a link whose target node/slot metadata points
  somewhere else.
- `visualGraph.nodes[N].inputs.<name> references missing link`: a linked input
  points at a link id that is absent from `visualGraph.links`.
- `visualGraph.links[ID] uses unsupported module graph sentinel node`: the
  graph includes partial/module placeholder links and cannot be made callable by
  this slice.
- `inputContract.inputs[N].target.node not found in visualGraph`: the input
  contract points at a missing visual node.
- `inputContract.inputs[N].target.input not found in generated apiPrompt`: the
  contract points at a visual input that is not injectable in the stored API
  prompt.
- `setupSurface must be a JSON object for group-authored setups`: a stored
  group-first setup has empty input/output contracts but lacks its persisted
  inferred app surface, so it is hidden from list/detail responses.
- `setupSurface.sourceInputs[N].nodes[M].type must be non-empty text`: a stored
  setup surface node summary is malformed.
- `visualGraph.groups[N].bounding must contain numeric x, y, width, height`: a
  reserved setup group has malformed rectangle data, so publish cannot safely
  infer membership.
- `visualGraph.nodes[N].pos must contain numeric x and y for setup surface inference`:
  a visual node has malformed placement data that could infer the wrong group.
- `setupSurface.sourceInputs requires a non-empty Koolook Input group`: the
  group-first publish path was used, but no node overlapped a `Koolook Input`
  group.
- `setupSurface.outputs requires a non-empty Koolook Output group`: the
  group-first publish path was used, but no node overlapped a `Koolook Output`
  group.
- Legacy prompts containing visual-only artifacts such as `Reroute`,
  `Label (rgthree)`, `Note`, `SetNode`, or subgraph wrapper ids are normalized
  from the visual graph when possible. ComfyUI-exported API prompts are
  preserved even when they differ from the fallback converter.

## Curator Publish Flow

In the sidebar Workflows section, right-click an active saved workflow and
choose **Publish setup...**. Archived workflows do not expose this action.

The dialog captures:

- setup id, title, optional description, category, tags, and optional preview/card reference
- the source workflow reference, shown read-only as `Folder/Workflow name`
  and stored with structured inventory breadcrumbs
- inferred `Koolook Input` / `Koolook Output` node summaries
- ComfyUI's API prompt for the workflow, captured automatically by the publish
  action when that UI integration is complete
- advanced input/output contract JSON when group inference is not enough

The client validates that the selected saved workflow still exists before
calling the publish API. The intended path is that the client captures the same
API prompt shape ComfyUI would export and sends it as `apiPrompt`; the setup
author should not manually attach an exported JSON file. The server validates
metadata/schema shape, stores the provided ComfyUI API prompt when present,
falls back to visual conversion only when necessary, stores the inferred setup
surface, and checks explicit input targets against both the submitted graph and
stored prompt when advanced contracts are used. Ordinary saved workflows are not
published automatically; only the explicit context-menu publish action writes
to the registry.

## External App Simulator

Use [`../../web/setup_runner_simulator.html`](../../web/setup_runner_simulator.html)
to validate the external-app path without adding maintainer-only controls to
the Kforge Labs sidebar. First publish at least one setup from the sidebar
Workflows context menu; the simulator consumes published records, it does not
publish workflows itself.

For live review, open the simulator through the stable Koolook route on the
same host/port as the running ComfyUI instance:

```text
http://127.0.0.1:<comfy-port>/koolook/setup_runner_simulator.html
```

Opening the file directly from disk is allowed for inspection, but same-origin
API calls are not available from `file://`. In that case the simulator prefills
`http://127.0.0.1:8188` as a common ComfyUI API base; change the port if the
server is running elsewhere. If the browser blocks that cross-origin request,
use the Koolook route above. Use
`web/setup_runner_simulator.html?demo=1` only to verify the simulator UI without
a live published setup.

The simulator uses the same public execution boundary an external frontend
uses:

1. `GET /koolook/api/setups` to list published setups.
2. `GET /koolook/api/setups/{id}` to inspect the selected setup contract.
3. `POST /koolook/api/setups/{id}/run` with the JSON inputs supplied in the
   simulator.
4. `GET /koolook/api/runs/{runId}` until the run reaches `succeeded`,
   `failed`, or `lost`, or the client-side timeout expires.

The simulator displays the stable Koolook run id, ComfyUI prompt id,
queued/running/final status, returned output summaries, raw ComfyUI terminal
status details, and raw Koolook error payloads. It does not re-convert sidebar
workflow graphs at run time and does not call ComfyUI `/prompt` directly; the
runner owns prompt cloning, input injection, queue submission, and history/queue
translation.

## Contract Authoring Rules

Input contract fields should describe the external app's friendly controls and
where each value lands in the workflow:

```json
{
  "key": "prompt",
  "label": "Prompt",
  "type": "text",
  "required": true,
  "target": { "node": "12", "input": "text" }
}
```

- `key`: stable external field key, such as `prompt`, `seed`, or `aspect_ratio`.
- `label`: curator-facing/user-facing label.
- `type`: simple value type such as `text`, `number`, `boolean`, `image`, or `video`.
- `required`: boolean; omit or set `false` for optional controls.
- `target.node`: visual graph node id as text.
- `target.input`: input/port/widget name expected on that node.

Output contract fields describe what an external app can read back:

```json
{
  "key": "preview",
  "label": "Preview",
  "type": "image"
}
```

Keep outputs minimal until the runner slice defines richer execution payloads.
