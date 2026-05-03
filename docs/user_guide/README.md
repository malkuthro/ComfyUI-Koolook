# User guide

End-user documentation for the nodes shipped by `ComfyUI-Koolook`. Search for a node by display name in ComfyUI (type "koolook" in the node-add menu) and see the corresponding guide here.

## Available guides

*(Stub — guides are added as they're written. The starter image in [`img/Easy_Image_Batch_v0.1.5.png`](img/Easy_Image_Batch_v0.1.5.png) is the first one queued.)*

## When to add a guide

If a node has any of:
- non-obvious input semantics (e.g. aspect-ratio strings, frame indexing conventions),
- a workflow recipe more useful than the per-input tooltips,
- a known limitation or compatibility caveat,

→ create a per-node markdown file here, name it after the canonical node ID
(e.g. `easy_image_batch.md`, `easy_resize_koolook.md`), and add it to the
"Available guides" table above.

Screenshots and helper images go in [`img/`](img/), named to match the file
they're embedded in (e.g. `Easy_Image_Batch_v0.1.5.png`).

## Where to find the node IDs

The canonical list is in the project [`README.md`](../../README.md) and
[`reference/`](../reference/).
