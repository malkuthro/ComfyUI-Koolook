# Loop Demo Handoff

## Context

This demo captures the loop debugging work from PR #205. The original maintainer
workflow created a batch of test frames, picked one frame by index, processed it
through a placeholder subgraph, built an EXR path with `EasyAIPipeline`, and
saved with `SaveEXRFrames`.

The problem was that the workflow wrote only the first frame. The intended
behavior was a true sequence of ComfyUI executions:

1. frame `0` writes `1/4`;
2. frame `1` writes `2/4`;
3. frame `2` writes `3/4`;
4. frame `3` writes `4/4`.

That matters because the future production subgraph may not accept batched image
sequences. It must receive one frame per prompt execution.

## Findings

Two false starts were useful:

- Replacing the EXR saver was the wrong abstraction. Saver nodes have their own
  backend-specific inputs and should keep owning file-writing behavior.
- Accumulating frames into one image batch and saving after the loop wrote four
  EXRs, but it produced saver progress like `4/4`, not four independent subgraph
  executions.

The final direction is queue-driven:

- `Frame Index` is an `easy int` node.
- `Frame Index` drives `ImageFromBatch.batch_index` and
  `SaveEXRFrames.start_frame`.
- `Koolook_LoopStatus` receives the processed single-frame value and passes it
  through to the saver.
- `Koolook_LoopStatus` prints `EXR_SAFE: 1/4`, `2/4`, and so on.
- When `auto_queue_next` is enabled, it submits the next API prompt with the
  `Frame Index` node advanced by one.

## Demo Files

- Canvas workflow: `docs/automations/loop-demo/LOOP_demo_pipeline.json`
- API harness: `scripts/run_loop_demo_api_test.py`
- Maintainer API-testing notes:
  `docs/maintainers/comfyui-server-api-testing.md`

The committed workflow uses a neutral placeholder output base:
`C:/koolook-loop-demo-output/`. Change that in ComfyUI before running it in a
real project.

## Validation

Validated against a running local ComfyUI server at `http://127.0.0.1:8188`.

```powershell
.\.venv-codex\Scripts\python scripts\run_loop_demo_api_test.py
```

Expected result:

- ComfyUI returns `status_str: success` for the initial prompt.
- Child prompts are submitted by `Koolook_LoopStatus`.
- Four EXRs appear under `.tmp/comfy-loop-test-output/run-*/v001/`.

The harness flattens the demo subgraph to its inner `EasyResize_Koolook` node
because ComfyUI's `/prompt` endpoint accepts API-format prompts, not visual
canvas JSON with subgraph wrapper nodes.

## Operational Notes

- Python node changes require `dev-sync` and a ComfyUI restart.
- `Koolook_LoopStatus` uses hidden `PROMPT` and `UNIQUE_ID` inputs, so the
  auto-queue behavior only exists inside real ComfyUI execution.
- Connected optional widgets can shift `widgets_values`; the node infers the
  frame-index node from its connected `index` input when `index_node_id` is
  blank.
- The node has a hard queue-depth cap and writes an abort marker near the target
  sequence path when a background child-prompt submission fails.
