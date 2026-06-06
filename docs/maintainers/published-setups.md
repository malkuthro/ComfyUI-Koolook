# Published workflow setup registry

Issue #210 introduces the first read-only server-side registry for published
workflow setups. It is separate from sidebar workflow and snapshot storage so
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
An existing empty `setups.json` remains empty; the sample is only a demo
fallback until the sidebar publish UX exists.

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

## Public Boundary

The registry module is [`../../koolook_setups.py`](../../koolook_setups.py).
External callers should use:

- `PublishedSetupRegistry.listSetups()`
- `PublishedSetupRegistry.getSetup(id)`

Invalid setup objects are omitted from list/detail results. Diagnostics are
kept on `registry.diagnostics` and logged by the HTTP adapter.

## Catalog API

`GET /koolook/api/setups`

Returns catalog summaries only: id, version, updated timestamp, metadata,
validation status, and input/output summaries. It intentionally does not return
`visualGraph`, `apiPrompt`, or raw storage internals.

`GET /koolook/api/setups/{id}`

Returns the full published setup contract for one setup, including visual graph,
optional API prompt, full input/output contracts, source reference, and
validation state. Unknown or invalid setup ids return `404`.
