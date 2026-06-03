# ComfyUI Server API Testing

Use this when an agent or maintainer needs to validate real ComfyUI execution,
not just static workflow JSON. It is useful for nodes that depend on ComfyUI's
runtime registry, hidden inputs, output nodes, prompt history, or file writes.

## What The AI Session Needs

The AI session needs three kinds of access:

1. Network access to the ComfyUI server. Usually this is
   `http://127.0.0.1:8188`. If the agent sandbox blocks local networking, grant
   network permission for the session.
2. Read access to the repository and test workflow files. The agent should
   build or load API prompts from repo-local files where possible.
3. Write access to a safe output folder. Prefer a repo-local ignored folder,
   such as `.tmp/comfy-loop-test-output/`, so generated images or EXRs do
   not land in production show folders while the test is still experimental.

Do not give the agent broad write access to project drives just to test a node.
If the workflow under test normally writes to an external shot path, override the
output base path in the submitted API prompt.

## ComfyUI Settings And Server State

ComfyUI must already be running and listening on a reachable host/port.

Useful checks:

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8188/system_stats -UseBasicParsing
```

or from Python:

```python
import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8188/system_stats") as r:
    print(json.load(r)["system"]["comfyui_version"])
```

For custom node changes:

- Run `dev-sync` only when explicitly requested.
- Restart ComfyUI after syncing Python files. The running process will not see a
  new Python node until restart.
- Confirm node registration with `/object_info`:

```python
import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8188/object_info") as r:
    info = json.load(r)
print("Koolook_LoopStatus" in info)
```

In the ComfyUI UI, enable the developer/API workflow option when you need to
manually export API-format workflows. The exact label can vary by frontend
version, but it is commonly under settings as developer mode / dev mode options,
which exposes "Save (API Format)".

## Canvas JSON Is Not API JSON

The normal saved workflow/canvas JSON is not what `/prompt` accepts. The API
prompt is a stripped graph shaped like this:

```json
{
  "4": {
    "class_type": "SaveEXRFrames",
    "inputs": {
      "images": ["21", 0],
      "filepath": ["8", 0],
      "start_frame": ["22", 0]
    }
  }
}
```

Each node id maps to `class_type` and `inputs`. Inputs are either literal widget
values or links in `[source_node_id, output_index]` form.

Subgraphs are another important difference. A canvas workflow can contain a
visual subgraph wrapper node, but `/prompt` needs the executable API graph. For
small tests, flatten the demo subgraph to the actual internal nodes. For larger
subgraphs, export API format from ComfyUI or build a converter that understands
the subgraph definition.

## Endpoints Used Most Often

| Endpoint | Use |
|---|---|
| `GET /system_stats` | Confirms the server is reachable and reports ComfyUI/Python/GPU state. |
| `GET /object_info` | Confirms node classes and input schemas are registered in the live server. |
| `POST /prompt` | Queues an API-format prompt. |
| `GET /history/{prompt_id}` | Checks completion status, errors, and output records for a queued prompt. |
| `GET /queue` | Shows running and pending prompts when debugging auto-queue behavior. |

## Safe Validation Pattern

1. Query `/system_stats`.
2. Query `/object_info` for every custom node used by the test.
3. Build an API prompt with a unique repo-local output folder.
4. Submit the prompt to `/prompt`.
5. Poll `/history/{prompt_id}` until the prompt completes or errors.
6. Check the expected files on disk.
7. For queue-controller nodes, keep polling the output folder or `/queue`,
   because child prompts may be submitted after the first prompt finishes.

The loop demo has a reusable harness:

```powershell
.\.venv-codex\Scripts\python scripts\run_loop_demo_api_test.py
```

It writes to `.tmp/comfy-loop-test-output/run-*/`, which is ignored by
git.

## Lessons From The Loop Controller Work

- A saver that receives a batch and prints `4/4` is not the same as four
  independent frame executions. Deep subgraphs that cannot accept sequences need
  one prompt per frame.
- Easy-Use `forLoopStart` / `forLoopEnd` did not reliably re-enter this demo
  graph as needed, so the working model became queue-driven: process frame 0,
  save frame 0, then submit a child prompt with frame index 1.
- Hidden inputs such as `PROMPT` and `UNIQUE_ID` are available only inside
  ComfyUI execution. They are what let a node inspect and resubmit its own API
  prompt.
- Connected widgets can disappear from the visible widget value list. Do not
  rely only on `widgets_values` order when a node has connectable optional
  inputs. Prefer connected inputs, explicit node ids, or runtime inference from
  the API prompt.
- ComfyUI caching can make a test look successful without re-running the saver.
  Use unique output paths per test run when file writes are part of the proof.

## What To Document In PRs

When a PR includes ComfyUI-server validation, include:

- the server URL used, usually `http://127.0.0.1:8188`;
- the harness or command run;
- whether custom node code was dev-synced and whether ComfyUI was restarted;
- the expected output folder;
- exact success evidence, such as `status_str: success` and output file count;
- any warnings about API prompt flattening or canvas subgraphs.
