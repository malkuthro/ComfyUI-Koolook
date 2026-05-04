# Workflow regression fixtures

Pinned ComfyUI workflow JSONs that exercise Koolook nodes. Used by the
pre-flight script (TBD — see `tools/preflight_release.py`) to verify
that every Koolook node ID referenced in any of these workflows still
exists in the current `NODE_CLASS_MAPPINGS` (or has a back-compat alias).

The pre-flight test only checks **Koolook-pack node IDs**. Other-pack
node IDs (DepthAnything, rgthree, etc.) are silently ignored — those
packs may or may not be installed on a given machine, and tracking
their stability isn't our job.

## How to add a new fixture

1. In ComfyUI, build a workflow that exercises one or more Koolook nodes.
2. Click **Save** to export the workflow JSON.
3. Name the file descriptively: `<feature>-<short-tag>-v<NN>.json`
   (e.g. `vae-roundtrip-srgb-v01.json`).
4. Drop it in this folder.
5. Add a one-line entry to the inventory table below.
6. **Sanity-check before committing:**
   - No Windows-style absolute paths (`C:\…`, `D:\…`).
   - No huge base64 blobs (embedded preview thumbnails) — strip them in
     ComfyUI's save dialog if present.
   - No tokens / credentials / personal info in any node parameter.
   - File size under ~50 KB ideally; certainly under 500 KB.

The pre-flight script picks the file up automatically — no need to
register it anywhere else.

## Fixture inventory

| File | Koolook nodes referenced | Notes |
|---|---|---|
| [`Depth-Anything3-v01.json`](Depth-Anything3-v01.json) | `EasyAIPipeline` | Depth-Anything v3 workflow with Koolook pipeline node + EXR I/O |
| [`Hires-FIX-wanT2V-v01.json`](Hires-FIX-wanT2V-v01.json) | `EasyAIPipeline` | Hires-fix variant for Wan T2V workflows |

## What the pre-flight script does (for reference)

For each `*.json` in this folder:

1. Parse the JSON.
2. Walk the `nodes` array (ComfyUI canvas format) — for each node, read its `type` field.
3. Filter: keep only types that look like Koolook node IDs (heuristics: starts with `Easy`, `Koolook`, contains `_koolook_`, etc. — full pattern lives in the script).
4. For each Koolook ID, verify it appears in the current `NODE_CLASS_MAPPINGS` (computed by importing the package).
5. **Fail** the pre-flight if any referenced ID is missing — that's a breaking change to a saved workflow that needs either a back-compat alias or an explicit major version bump.

If the JSON is in API format (different shape, often produced by `ComfyUI` API export), the script also handles that: walk the top-level dict where each key is a node instance and each value has `class_type`.
