# Kforge Labs Workflows

These workflows are bundled into the Kforge Labs sidebar as starter demos.

| Workflow | What it demonstrates |
|---|---|
| [`Easy Image Batch - Select and Rebuild.json`](Easy%20Image%20Batch%20-%20Select%20and%20Rebuild.json) | Selecting sparse frames from a batch, then rebuilding sequence spacing with Easy Image Batch. |
| [`LOOP Demo.json`](LOOP%20Demo.json) | Queue-driven per-frame processing with `Koolook_LoopStatus`. |

## Requirements

Both workflows need Kforge Labs / Koolook and ComfyUI core nodes. The demo
can still open when optional node packs are missing, but those nodes must be
installed before the workflow can run end-to-end.

| Workflow | Extra node packs |
|---|---|
| Easy Image Batch - Select and Rebuild | ComfyUI Custom Scripts (`ShowText|pysssss`) |
| LOOP Demo | ComfyUI Easy Use (`easy int`), ComfyUI-KJNodes (`GetImageSizeAndCount`), ComfyUI-HQ-Image-Save (`LoadEXR`, `SaveEXRFrames`), ComfyUI-RMBG (`RMBG`) |

The large text labels in both workflows use rgthree label nodes. They are
visual annotations only; missing labels do not change the core demo logic.

The sidebar seed lives in [`../../../web/workflow_defaults.json`](../../../web/workflow_defaults.json)
under **Kforge Labs Workflows**.

Before running `LOOP Demo`, replace `CHANGE_ME_OUTPUT_FOLDER` in the Easy AI
Pipeline node with a writable local output folder, then enable
`auto_queue_next` on `Koolook_LoopStatus` if you want it to queue the remaining
frames automatically. The bundled copy leaves auto-queue disabled so the first
load cannot write to a placeholder path by accident.
