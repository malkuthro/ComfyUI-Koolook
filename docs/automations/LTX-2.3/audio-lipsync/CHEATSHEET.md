# LTX 2.3 audio-lipsync — cheat sheet

The minimum the maintainer needs to drive this module on any machine.
Full prose: [`README.md`](README.md). Per-render protocol:
[`runs/LOOP.md`](runs/LOOP.md). Backstory:
[`backstory/audio-lipsync-rationale.md`](backstory/audio-lipsync-rationale.md).

## What this module is for

Iterating on the LTX 2.3 audio-file lip-sync failure mode by sweeping
the new `relay_overrides` knobs on the Koolook fork of `LTXDirector`
(`LTXDirector__koolook_v1_3_2`). Baseline = upstream `LTXDirector` with
`use_custom_audio=False`. Goal: get `use_custom_audio=True` to look as
clean as the baseline.

## Chat triggers (user-initiated only — never automatic)

| Phrase | Script | What it does |
|---|---|---|
| `dev-sync-audio` | [`../../../../scripts/sync_to_dev_audio.py`](../../../../scripts/sync_to_dev_audio.py) | Copies `forks/whatdreamscost_koolook/` + root `__init__.py` to `$KOLOOK_COMFYUI_DEV_PATH`, auto-restarts Comfy. Use after editing fork code. |
| `loop-audio` | [`../../../../scripts/loop_audio.py`](../../../../scripts/loop_audio.py) | Snapshots the most recent render into `runs/run-NNN_<label>/` (workflow.json + relay_overrides + patch state + notes + card.png) and appends a row to [`runs/log.md`](runs/log.md). |

Both scripts read their config from `<script>.config.json` next to them. Edit the config (not the Python) when paths or node titles change.

## Required workflow shape

Five `Text Multiline` nodes on the canvas, titled (case-insensitive `contains` match):

| Title contains | Purpose |
|---|---|
| **`Working_Folder_PATH`** | Per-project working folder for renders (one absolute path). |
| **`NAME`** | Short run identifier — feeds the snapshot folder name + card title. |
| **`RELAY_OVERRIDES`** | JSON object of Prompt-Relay knob overrides. **Wired** into the LTX Director's `relay_overrides` input via link (don't paste into the Director widget directly). |
| **`OVERLAY - INFO`** | Free-form run summary + `BASE (notes):` section for Δ-from-baseline. |
| **`OVERLAY - FEEDBACK`** | Post-render observations + `motion: N/5 · sync: N/5 · sharp: N/5` score lines. |

Plus the **`LTX Director (Koolook v1.3.2)`** node (registered ID `LTXDirector__koolook_v1_3_2`) — NOT upstream `LTXDirector`. Otherwise the `relay_overrides` value drains into nothing and the per-segment σ formula is also unused.

Save as `LTX-23-audio_tests_v01.json` inside ComfyUI's `user/default/workflows/` (or whatever filename matches `workflow_pattern` in [`loop_audio.config.json`](../../../../scripts/loop_audio.config.json)).

## The render cycle

1. **Edit the knob.** Either change the `relay_overrides` multiline JSON, or edit a `forks/whatdreamscost_koolook/versions/v1_3_2/*.py` source then say `dev-sync-audio`.
2. **`Workflow → Save (API Format)`** into the workflows dir. Mtime decides which file `loop-audio` picks.
3. **Queue the render** in ComfyUI.
4. **Type `loop-audio`** in chat once it finishes. The agent runs the script.
5. **Tell the agent what you saw.** It logs the row + creates the run folder. No extra command for "keep" — every render is captured (the config decides what's recorded).

## What lands in `runs/run-NNN_<label>/`

```
workflow.json          ← copy of the Comfy file at submission
relay_overrides.txt    ← RELAY_OVERRIDES body + INERT warning if Director is upstream
patch_state.txt        ← MAIN sha + last dev-sync-audio sha + fork-dir clean/dirty
notes.md               ← maintainer feedback + scores + mechanical interp
card.png               ← side-by-side card for the NLE (KNOB / FORK / SAMPLER / BASE / OUTCOME)
```

`runs/log.md` gets a rolling-table row pointing at this folder.

## Promoting findings

When a knob's behaviour is stable across **≥ 3 confirming runs**, move the conclusion into [`findings.md`](findings.md):

1. Describe it as a single declarative sentence (*"X causes Y"* / *"do X, don't do Y"*).
2. Add a one-paragraph "why" and link the confirming run folders.
3. Remove the corresponding hypothesis from the open list in [`README.md`](README.md) or [`backstory/audio-lipsync-rationale.md`](backstory/audio-lipsync-rationale.md).

## Where settings live (no code edits for the common cases)

| What | Where |
|---|---|
| Which workflow filename pattern | [`scripts/loop_audio.config.json`](../../../../scripts/loop_audio.config.json) → `workflow_pattern` |
| Which ComfyUI workflows dir | Derived from `KOLOOK_COMFYUI_DEV_PATH` in `.env`, or `KOLOOK_COMFYUI_WORKFLOWS_DIR` override |
| Which Text Multiline titles to extract | `loop_audio.config.json` → `tracked_multilines` |
| Which fork dir to pin | `loop_audio.config.json` → `fork_to_track` |
| Whether to render the card | `loop_audio.config.json` → `render_card` |
| ComfyUI dev install target (for `dev-sync-audio`) | `.env` → `KOLOOK_COMFYUI_DEV_PATH` |

## Card section legend

| Section | Accent | Source |
|---|---|---|
| **KNOB STATE** | amber | `RELAY_OVERRIDES` multiline + LTX Director's `use_custom_audio` |
| **FORK STATE** | violet | `git rev-parse HEAD` + `web/_dev_build.json` (written by `dev-sync-audio`) + `git status` of the fork dir |
| **SAMPLER** | sky | Workflow's `BasicScheduler` / `KSamplerSelect` / `RandomNoise` / `CFGGuider` widgets |
| **BASE notes** | sky | `OVERLAY - INFO` multiline (verbatim) |
| **OUTCOME** | green | `OVERLAY - FEEDBACK` multiline → score lines + free text |
| **⚠ INERT warning** | amber outline | shown only when the workflow's Director is upstream `LTXDirector` and `RELAY_OVERRIDES` is non-empty |
