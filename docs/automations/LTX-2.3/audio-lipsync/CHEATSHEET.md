# LTX 2.3 audio-lipsync — cheat sheet

The minimum needed to drive this module on any machine. Visual map:
[`reading-graph.html`](reading-graph.html). Machine-readable schema:
[`reading-graph.schema.yaml`](reading-graph.schema.yaml). Per-render
protocol: [`runs/LOOP.md`](runs/LOOP.md). Full prose: [`README.md`](README.md).
Backstory: [`backstory/audio-lipsync-rationale.md`](backstory/audio-lipsync-rationale.md).

## What this module is for

Iterating on the LTX 2.3 audio-file lip-sync failure mode by sweeping
the `relay_overrides` knobs on the Koolook fork of `LTXDirector`
(`LTXDirector__koolook_v1_3_2`). Baseline = `use_custom_audio=False`
(model-generated audio path). Goal: get `use_custom_audio=True` to
look as clean as baseline.

## Chat triggers (user-initiated only — never automatic)

| Phrase | Script | What it does |
|---|---|---|
| `loop-audio` | [`scripts/loop_audio.py`](../../../../scripts/loop_audio.py) | Snapshots the most recent render into `runs/run-NNN_<label>/` (workflow.json + relay_overrides.txt + patch_state.txt + notes.md + card.png) and appends a row to [`runs/log.md`](runs/log.md). |
| `dev-sync-audio` | [`scripts/sync_to_dev_audio.py`](../../../../scripts/sync_to_dev_audio.py) | Copies `forks/whatdreamscost_koolook/` + root `__init__.py` into `$KOLOOK_COMFYUI_DEV_PATH`, auto-restarts Comfy. Use after editing fork source. Widget-only changes on the canvas don't need it. |

Config lives in `<script>.config.json` next to each script — edit the
config, not the Python, when paths or node titles change.

## Required canvas shape

Five `Text Multiline` nodes (case-insensitive `contains` match against
the title) + the Koolook Director node:

| Title contains | Used for |
|---|---|
| `NAME` | Run identifier — feeds snapshot folder + card title |
| `RELAY_OVERRIDES` | JSON dict of Prompt-Relay knob overrides. **Wired** into the Director's `relay_overrides` input |
| `OVERLAY - INFO` | Free-form Δ-this-run note. Card renders verbatim |
| `OVERLAY - FEEDBACK` | Observations + `motion: N/5 · sync: N/5 · sharp: N/5` score lines |
| `Working_Folder_PATH` | Per-project working folder. Duplicates allowed (mount + mirror); first existing wins |
| **node** `LTX Director (Koolook v1.3.2)` | Registered as `LTXDirector__koolook_v1_3_2`. Required — upstream `LTXDirector` won't have the `relay_overrides` input and the per-segment σ formula is also absent |

Save the workflow as `LTX-23-audio_tests_v01.json` in ComfyUI's
`user/default/workflows/` (or whatever filename matches `workflow_pattern`
in [`loop_audio.config.json`](../../../../scripts/loop_audio.config.json)).

## The render cycle

1. Edit the knob — either change the `RELAY_OVERRIDES` multiline JSON,
   or edit a `forks/whatdreamscost_koolook/versions/v1_3_2/*.py` source
   then say `dev-sync-audio`.
2. **Save (API format)** into the workflows dir. Mtime decides which
   file `loop-audio` picks.
3. Queue the render in ComfyUI.
4. Type `loop-audio` in chat once it finishes.
5. Describe the result in chat — the agent has already logged the row
   and snapshotted the folder.

## Card source rule

The card draws **only** from two source families:

1. The five `Text Multiline` nodes above (the per-render notes).
2. The `LTXDirector__koolook_v1_3_2` node's own widget values and
   input wiring.

Forbidden on the card: `BasicScheduler` / `KSamplerSelect` / `RandomNoise` /
`CFGGuider` widgets, `_dev_build.json`, `git status` output, hardcoded
sampler/CFG/seed constants, substring matches on prompt text. Full
breakdown: [`reading-graph.html`](reading-graph.html).

## Audio src — the four structural states

Derived from `audio_vae` input link + `use_custom_audio` widget +
`timeline_data.audioSegments` count. Mirrors what the director does at
runtime (`forks/whatdreamscost_koolook/versions/v1_3_2/ltx_director.py:610-622`).

| Label | Conditions |
|---|---|
| `model-gen` | audio_vae wired · use_custom_audio = False |
| `custom` | audio_vae wired · use_custom_audio = True · audioSegments non-empty |
| `custom (empty)` | audio_vae wired · use_custom_audio = True · audioSegments empty |
| `off (no VAE)` | audio_vae not wired (no audio latent produced) |

## What lands in `runs/run-NNN_<label>/`

```
workflow.json          ← copy of the Comfy file at submission (ground truth)
relay_overrides.txt    ← RELAY_OVERRIDES body (diff-friendly plain text)
patch_state.txt        ← MAIN sha + last dev-sync-audio sha + fork-dir clean/dirty (audit trail; NOT on the card)
notes.md               ← OVERLAY-FEEDBACK + OVERLAY-INFO verbatim + director's structural state
card.png               ← side-by-side card for the NLE
```

Each render also appends a row to [`runs/log.md`](runs/log.md) with the
same source rule (no scheduler/sampler columns).

## Promoting findings

A pattern graduates from [`runs/log.md`](runs/log.md) into
[`findings.md`](findings.md) when it's stable across **≥ 3 confirming
runs**:

1. Describe it as a single declarative sentence (*"X causes Y"* /
   *"do X, don't do Y"*).
2. Add a one-paragraph "why" and link the confirming run folders.
3. Remove the corresponding hypothesis from [`README.md`](README.md) or
   [`backstory/audio-lipsync-rationale.md`](backstory/audio-lipsync-rationale.md).

## Where settings live (no code edits for the common cases)

| What | Where |
|---|---|
| Workflow filename pattern | [`scripts/loop_audio.config.json`](../../../../scripts/loop_audio.config.json) → `workflow_pattern` |
| ComfyUI workflows dir | Derived from `KOLOOK_COMFYUI_DEV_PATH` in `.env`, or `KOLOOK_COMFYUI_WORKFLOWS_DIR` override |
| Tracked multiline titles | `loop_audio.config.json` → `tracked_multilines` |
| Fork dir to pin in `patch_state.txt` | `loop_audio.config.json` → `fork_to_track` |
| Whether to render the card | `loop_audio.config.json` → `render_card` |
| ComfyUI dev install target (`dev-sync-audio`) | `.env` → `KOLOOK_COMFYUI_DEV_PATH` |

`.env` is read from the worktree first, then the main repo when running
from a worktree — same fallback as `scripts/make_card.py`.
