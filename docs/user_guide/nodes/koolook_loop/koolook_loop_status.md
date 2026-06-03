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
- `index_node_id`: advanced override for the connected `easy int` frame-index
  node. Leave this blank for normal use. The node infers the frame-index node
  from the connected `index` input and logs the detected node class/id when it
  queues the next frame. If an old saved workflow contains a stale manual id,
  the connected `index` input is used instead.
- `server_url`: local ComfyUI server URL, usually `http://127.0.0.1:8188`.
- `max_auto_queue_depth`: hard safety cap for how many child prompts this node
  may chain from the current frame.
- `remaining_auto_queue_depth`: internal countdown carried into child prompts.
  Leave this at `-1` in normal canvas use.

If an older saved workflow accidentally shifts widget values and puts a numeric
node id into `label`, the node treats that numeric label as `index_node_id`,
prints a recovery note, and uses `EXR_SAFE` as the label. This keeps older loop
demo saves from crashing, but new workflows should set `label` explicitly and
leave `index_node_id` blank unless there is a specific override reason.
