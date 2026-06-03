# LOOP demo pipeline handoff

## Context

The user is testing a ComfyUI loop setup from:

`E:/G-Drive-BaconX/Jobs/3252_arena/ComfyUI/setups/LOOP_demo_pipeline.json`

The graph is a minimal repro:

- `Easy_Pattern` creates a batch of test frames.
- `ImageFromBatch` picks one frame by loop index.
- A placeholder subgraph processes the picked frame.
- `EasyAIPipeline` builds the EXR path.
- `SaveEXRFrames` writes the final EXR.
- `easy forLoopStart` / `easy forLoopEnd` already exist.

The observed issue: the graph writes only once. It does not execute the save branch for every loop iteration.

## Current diagnosis

The problem is not `SaveEXRFrames` and not the path builder.

Easy-Use loop nodes recurse over the graph section that is connected through the loop value sockets:

- `easy forLoopStart.value1` / loop-dependent work
- work output
- `easy forLoopEnd.initial_value1`

The original workflow had `forLoopEnd.initial_value1` empty. That means the loop end had no body result to recurse over. A saver node has no normal output socket, so it cannot by itself be the returned body value.

The useful internet example says the same thing: the body output must return to the matching `For Loop End.initial_valueN` input. The local Easy-Use source confirms this: `whileLoopEnd` collects nodes upstream of the loop end, then expands that contained graph for the next iteration.

The second important point: `SaveEXRFrames` must be downstream of `forLoopEnd`.
If the saver is fed directly from the body image, Comfy can satisfy the output
node by running the body once and never executing the loop end. That was the
reason the first attempted return-link-only fix still behaved like a one-pass
workflow.

## Correct direction

Do not create a replacement EXR saver.

Do not use a separate stateful frame-counter node for this demo. That only advances once per queued prompt and does not solve one-click loop execution.

Use the existing Easy-Use loop properly:

- Keep `easy forLoopStart`.
- Keep `easy forLoopEnd`.
- Keep `ImageFromBatch.batch_index` driven by `forLoopStart.index`.
- Use `easy batchAnything` as the loop-state accumulator:
  - `forLoopStart.value1` -> `Batch Any.any_1`
  - processed image -> `Batch Any.any_2`
  - `Batch Any.batch` -> `forLoopEnd.initial_value1`
- Feed `forLoopEnd.value1` into `SaveEXRFrames.images`.
- Leave `SaveEXRFrames.start_frame` as the local widget value `0`.

This makes the loop end part of the dependency chain for the output saver. The
loop accumulates the processed frames into one image batch, then the EXR saver
writes that batch as `0000`, `0001`, `0002`, `0003`.

## Generated setup copy

Repo-local generated setup:

`agent-tools/LOOP_demo_pipeline_Codex.json`

It was regenerated from the original external JSON and now has:

- No `Koolook_EXRLoop`.
- No `Koolook_LoopIndex`.
- Original `SaveEXRFrames` restored and untouched.
- Original `EasyAIPipeline` path builder retained.
- Existing subgraph retained.
- `easy batchAnything` node added as the loop-body accumulator.
- `SaveEXRFrames.images` now receives `easy forLoopEnd.value1`, not the direct
  processed image output.
- `SaveEXRFrames.start_frame` is unlinked and uses widget value `0`, because the
  saver now writes the final accumulated batch.

The external setup folder was not modified because project rules forbid Codex from writing outside the repository.

## Repo cleanup state

The rejected loop-index implementation has been removed from source:

- `k_loop.py` deleted.
- `tests/nodes/test_loop.py` deleted.
- `__init__.py` no longer registers `.k_loop`.
- `scripts/sync_to_dev.py` no longer syncs `k_loop.py`.
- `CHANGELOG.md` no longer advertises the loop-index node.

If the live ComfyUI dev install was previously synced, it may still contain stale `k_loop.py` on disk because `dev-sync` copies files but does not delete removed runtime files. Since `__init__.py` no longer registers `.k_loop`, the stale file should not load after restart. Manual cleanup of stale files in the live external install must be done by the user because this repo forbids external deletes.

## Internet references checked

- `ali1234/comfyui-job-iterator`: shows a workflow-native iterator approach where values are exposed to normal nodes, not by wrapping saver nodes.
- Reddit example "Using Loops on ComfyUI": states the key Easy-Use-style loop rule: pass the start value into the body, then return the same-format output to the matching `For Loop End.initial_valueN`.

## Validation

Validated against the live local ComfyUI server at `http://127.0.0.1:8188`.

Repeatable harness:

```powershell
.\.venv-codex\Scripts\python agent-tools\run_loop_demo_api_test.py
```

Result:

- ComfyUI returned `status_str: success`.
- Four EXRs were written under
  `agent-tools/comfy-loop-test-output/run-*/v001/`.
- `agent-tools/comfy-loop-test-output/` is ignored in git.

The harness flattens the current demo subgraph to its inner
`EasyResize_Koolook` node because Comfy's `/prompt` API accepts API-format
prompts, not the visual canvas JSON subgraph wrapper. The canvas workflow file
still keeps the subgraph.

## Important repo rules

- Do not write to the external `E:/G-Drive-BaconX/...` setup folder from Codex.
- `dev-sync` is user-initiated only.
- For UI/browser-visible changes, visual QA is mandatory. This work is workflow JSON / Python registry cleanup, not frontend UI.
