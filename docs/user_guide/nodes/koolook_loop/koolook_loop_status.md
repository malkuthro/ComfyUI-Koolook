# Koolook Loop Status

| Field | Value |
|---|---|
| Source | Koolook-native |
| Node ID | `Koolook_LoopStatus` |
| Display name | `Koolook Loop Status` |
| Category | `Koolook/Loop` |

`Koolook_LoopStatus` is a pass-through loop status node. It prints progress
such as `EXR_SAFE: 1/4 frame 0 -> path/to/frame.0000.exr`, returns the input
value unchanged, and can queue the next prompt by advancing a connected
`easy int` frame-index node.

Use it when a workflow must process one frame per ComfyUI execution instead of
passing a whole image sequence through a deep subgraph. The frame-index node
drives `ImageFromBatch.batch_index` and the saver frame number. Queue the graph
once at frame `0`; with `auto_queue_next` enabled, the status node submits the
next prompt until `index + 1 == total`.

Key inputs:

- `value`: the image, latent, or other payload to pass through.
- `index`: current zero-based frame index.
- `total`: total number of frames to run.
- `filepath`: optional sequence path used only for the printed status line.
- `label`: status label, for example `EXR_SAFE`.
- `auto_queue_next`: when enabled, submit the next prompt automatically.
- `index_node_id`: node id of the connected `easy int` frame-index node. If this
  is blank, the node attempts to infer it from the connected `index` input.
- `server_url`: local ComfyUI server URL, usually `http://127.0.0.1:8188`.
