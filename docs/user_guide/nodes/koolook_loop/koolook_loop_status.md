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
  is blank, the node infers it from the connected `index` input. If the supplied
  id is not actually a node in the workflow (e.g. a widget value shifted to `0`
  in an older save), the node falls back to that same inference so the loop
  self-heals; if it still cannot resolve a real node, auto-queue fails up front
  with a clear message instead of silently aborting into a marker file.
- `server_url`: local ComfyUI server URL. Defaults to `http://127.0.0.1:8188`,
  but when left at that default (or blank) the node auto-detects the address the
  running server actually bound to — so an install launched with `--port 8000`
  (or `--listen`) queues correctly without editing the widget. Set an explicit
  value only to target a different host/port; a custom value is used verbatim.
- `max_auto_queue_depth`: hard safety cap for how many child prompts this node
  may chain from the current frame.
- `remaining_auto_queue_depth`: internal countdown carried into child prompts.
  Leave this at `-1` in normal canvas use.

If an older saved workflow accidentally shifts widget values and puts a numeric
node id into `label`, the node treats that numeric label as `index_node_id`,
prints a recovery note, and uses `EXR_SAFE` as the label. This keeps older loop
demo saves from crashing, but new workflows should set `label` and
`index_node_id` explicitly.
