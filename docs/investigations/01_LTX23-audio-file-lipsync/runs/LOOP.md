# Audio-file lip-sync investigation — iteration loop

**Investigation:** [01_LTX23-audio-file-lipsync](../README.md)
**Tracking issue:** [malkuthro/ComfyUI-Koolook#177](https://github.com/malkuthro/ComfyUI-Koolook/issues/177)
**Started:** 2026-05-28

---

## The loop

| Step | Who | What |
|---|---|---|
| 1 | Maintainer | Edit workflow in ComfyUI (canvas — node widgets, including the `relay_overrides` JSON field on the LTX Director) |
| 2 | Maintainer | **Save** workflow → overwrites `E:\_AI\portable\ComfyUI_windows_312\ComfyUI\user\default\workflows\LTX-23-audio_tests_v01.json` |
| 3 | Maintainer | Queue render in ComfyUI |
| 4 | Maintainer | Report result in chat (verbal feedback — sync state, motion state, prompt adherence) |
| 5 | Agent | Snapshot current state into a new `run-NNN_<label>/` folder here + append a row to [`log.md`](log.md) |
| 6 | Agent | Confirm what was captured |

**Save-before-render** is the anchor: the file on disk at the JSON file folder = exactly what produced the render being described. No copy-paste, no version drift.

## Per-run snapshot contents

```
run-NNN_<label>/
├── workflow.json           ← copy of LTX-23-audio_tests_v01.json at submission
├── relay_overrides.txt     ← the LTX Director node's relay_overrides widget value
├── patch_state.txt         ← MAIN SHA + upstream SHA + which patches were synced
└── notes.md                ← maintainer's verbal feedback + agent's one-line interp
```

The `relay_overrides` value is also stored inside `workflow.json` (LTXDirector node widget), but the `.txt` copy makes it diff-friendly in the run folder.

## Trigger detection

No magic phrase. Whenever the maintainer describes a render result in chat, the agent treats it as a feedback signal and creates a new run folder. The agent confirms in chat what was captured; the maintainer can rename, delete, or correct.

To share a workflow file *without* logging it as a run, say "no log" or "don't log this" in the chat message.

## Run labels

Auto-generated from key knob state + feedback summary, e.g.:

- `run-001_no-patch_broken-baseline`
- `run-002_vstr10_no-change`
- `run-003_vstr25_some-motion-recovered`
- `run-004_vstr10_astr0.5_sync-partial`

The maintainer can override any label by stating it explicitly.

## Sync to live install — `sync-audio`

Before the first render with new patches, sync MAIN → live install:

```
python scripts/sync_investigation_patches.py 01_LTX23-audio-file-lipsync
```

Or just say "**sync-audio**" in chat and the agent runs it. Backs up the target's current state to `.bak.<YYYYMMDD>` before overwriting. See the investigation [README](../README.md) and [`upstream.json`](../upstream.json) for details.

## Asking "what's the loop?"

Point future-session-me at this file: `docs/investigations/01_LTX23-audio-file-lipsync/runs/LOOP.md`. Everything we agreed lives here.

## Glossary

See [`../../../reference/glossary.md`](../../../reference/glossary.md) for definitions of *JSON file folder*, *Loop*, *Run*, *Investigation patches*, *Investigation folder*.
