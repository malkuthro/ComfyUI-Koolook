# LTX 2.3 audio-lipsync — per-render iteration protocol

**Automation:** [LTX 2.3 audio-lipsync](../README.md)
**Tracking issue:** [malkuthro/ComfyUI-Koolook#177](https://github.com/malkuthro/ComfyUI-Koolook/issues/177)
**Started:** 2026-05-28

---

## The loop

| Step | Who | What |
|---|---|---|
| 1 | Maintainer | Either edit `relay_overrides` JSON on the **`LTX Director (Koolook v1.3.2)`** node in ComfyUI, OR edit a `forks/whatdreamscost_koolook/versions/v1_3_2/*.py` source in this repo + run `dev-sync` + restart Comfy. |
| 2 | Maintainer | **Save** workflow → overwrites the current workflow file at the working folder. |
| 3 | Maintainer | Queue render in ComfyUI. |
| 4 | Maintainer | Report result in chat (verbal feedback — sync state, motion state, prompt adherence). |
| 5 | Agent | Snapshot current state into a new `run-NNN_<label>/` folder here + append a row to [`log.md`](log.md). |
| 6 | Agent | Confirm what was captured. |

**Save-before-render** is the anchor: the workflow file on disk at submission time = exactly what produced the render being described. No copy-paste, no version drift.

## Per-run snapshot contents

```
run-NNN_<label>/
├── workflow.json           ← copy of the working-folder workflow at submission
├── relay_overrides.txt     ← the LTX Director (Koolook) node's relay_overrides widget value
├── patch_state.txt         ← MAIN SHA + dev-sync SHA + whether forks/.../v1_3_2/*.py differs from MAIN
└── notes.md                ← maintainer's verbal feedback + agent's one-line interp
```

The `relay_overrides` value is also stored inside `workflow.json` (Koolook
LTXDirector node widget), but the `.txt` copy makes it diff-friendly in the
run folder.

## Trigger detection

No magic phrase. Whenever the maintainer describes a render result in chat, the agent treats it as a feedback signal and creates a new run folder. The agent confirms in chat what was captured; the maintainer can rename, delete, or correct.

To share a workflow file *without* logging it as a run, say "no log" or "don't log this" in the chat message.

## Run labels

Auto-generated from key knob state + feedback summary, e.g.:

- `run-001_no-override_baseline-fork-only`
- `run-002_vstr10_no-change`
- `run-003_vstr25_some-motion-recovered`
- `run-004_vstr10_astr0.5_sync-partial`
- `run-005_codechange_sigma-bumped_sync-clean`

The maintainer can override any label by stating it explicitly.

## Syncing fork-code edits to the live install

When the iteration step requires editing a `forks/whatdreamscost_koolook/`
source (not just the widget), the standard
[`dev-sync`](../../../../scripts/sync_to_dev.py) flow applies — it's a
user-initiated, never-automatic action per the project `CLAUDE.md`. Quick
recipe:

```
python scripts/sync_to_dev.py --scope "audio-lipsync fork edit"
```

Then restart Comfy so the Python module re-imports. Widget-only changes
(editing `relay_overrides` JSON on the canvas) do **not** require sync or
restart — just queue the next render.
