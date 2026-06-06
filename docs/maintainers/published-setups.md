# Published workflow setup registry

Issue #210 introduced the first server-side registry for published workflow
setups. Issue #211 adds the curator publish path from saved sidebar workflows.
The registry remains separate from sidebar workflow and snapshot storage so
external apps can consume a stable catalog without knowing Kforge Labs sidebar
internals.

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
    "description": "User-facing description",
    "category": "Video",
    "tags": ["tag"],
    "previewImage": ""
  },
  "visualGraph": {},
  "apiPrompt": null,
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
  "source": {
    "kind": "sidebar-workflow",
    "path": "Folder/Workflow name"
  },
  "validation": {
    "status": "valid",
    "diagnostics": []
  }
}
```

`apiPrompt` is nullable in this first slice so a backend-created fixture can
prove the registry and catalog shape before visual-to-API conversion lands.
Execution slices should require a concrete API prompt before a setup can run.
Only `schemaVersion: 1` is accepted by this first registry; future schema
versions must add explicit migration or validation support before they pass.

## Public Boundary

The registry module is [`../../koolook_setups.py`](../../koolook_setups.py).
External callers should use:

- `PublishedSetupRegistry.listSetups()`
- `PublishedSetupRegistry.getSetup(id)`
- `PublishedSetupRegistry.publishSetup(visualGraph, metadata, inputContract, outputContract, source)`

Invalid setup objects are omitted from list/detail results. Diagnostics are
kept on `registry.diagnostics` and logged by the HTTP adapter. File-level
storage diagnostics, such as unreadable JSON, are reported through the same
channel.

`publishSetup` builds a draft setup from a saved sidebar workflow graph,
validates required metadata and contract shape, checks that each declared input
target points at a plausible node/input in the visual graph, then replaces any
existing setup with the same id in storage. Draft publishes use
`apiPrompt: null` and `validation.status: "draft"` until the visual-to-API
conversion slice lands.

## Catalog API

`GET /koolook/api/setups`

Returns catalog summaries only: id, version, updated timestamp, metadata,
validation status, and input/output summaries. It intentionally does not return
`visualGraph`, `apiPrompt`, or raw storage internals.

`GET /koolook/api/setups/{id}`

Returns the full published setup contract for one setup, including visual graph,
optional API prompt, full input/output contracts, source reference, and
validation state. Unknown or invalid setup ids return `404`.

`POST /koolook/api/setups`

Publishes one setup draft. Body:

```json
{
  "visualGraph": {},
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
    "path": "Demos/Director Demo"
  }
}
```

Success returns `{ "ok": true, "setup": { ... } }`. Validation failures return
HTTP `400` with `{ "ok": false, "errors": [...] }`.

## Curator Publish Flow

In the sidebar Workflows section, right-click an active saved workflow and
choose **Publish setup...**. Archived workflows do not expose this action.

The dialog captures:

- setup id, title, description, category, tags, and optional preview/card reference
- the source workflow reference, shown read-only as `Folder/Workflow name`
- input contract JSON
- output contract JSON

The client validates that the selected saved workflow still exists before
calling the publish API. The server validates metadata/schema shape and checks
input targets against the submitted graph. Ordinary saved workflows are not
published automatically; only the explicit context-menu publish action writes
to the registry.

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
