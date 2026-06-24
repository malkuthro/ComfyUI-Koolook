# LTX 2.3 audio-lipsync — cheat sheet

The minimum needed to drive this module on any machine. Visual map:
[`reading-graph.html`](reading-graph.html). Machine-readable schema:
[`reading-graph.schema.yaml`](reading-graph.schema.yaml). Per-render
protocol: [`runs/LOOP.md`](runs/LOOP.md). Full prose: [`README.md`](README.md).
Backstory: [`backstory/audio-lipsync-rationale.md`](backstory/audio-lipsync-rationale.md).

## What this module is for

Iterating on the LTX 2.3 audio-file lip-sync failure mode by sweeping
the `relay_overrides` knobs on the Koolook fork of `LTXDirector`
(`LTXDirector__koolook`). Baseline = `use_custom_audio=False`
(model-generated audio path). Goal: get `use_custom_audio=True` to
look as clean as baseline.

As of the v2.0.2 fork, `LTXDirector__koolook` also exposes
`snap_keyframes_to_grid` (default **on**): it snaps each image keyframe to
its LTX latent-bucket center so hard pins stop "jumping" at bucket edges and
two pins never collide in one bucket. Resolution ceiling is one pin per
temporal-stride frames (~333 ms @ 24 fps). See
[`findings.md`](findings.md) and ComfyUI-Koolook#258.

## Chat triggers (user-initiated only — never automatic)

| Phrase | Script | What it does |
|---|---|---|
| `loop-audio` | [`scripts/loop_audio.py`](../../../../scripts/loop_audio.py) | Snapshots the most recent render into `runs/run-NNN_<label>/` (redacted runNNN_workflow.json + relay_overrides.txt + patch_state.txt + metadata.json + notes.md + card.png), delivers the external card copy, and appends a row to [`runs/log.md`](runs/log.md). |
| `dev-sync-audio` | [`scripts/sync_to_dev_audio.py`](../../../../scripts/sync_to_dev_audio.py) | Copies `forks/whatdreamscost_koolook/`, `web/whatdreamscost_koolook/`, and root `__init__.py` into `$KOLOOK_COMFYUI_DEV_PATH`; removes stale pre-v1.3.9 web extension folder. Restart ComfyUI manually after Python changes need to be re-imported. Widget-only changes on the canvas don't need it. |

Config lives in `<script>.config.json` next to each script — edit the
config, not the Python, when paths or node titles change.

## Required canvas shape

Five semantic `Text Multiline` captures (case-insensitive alias match
against the title) + one Director node:

| Capture key | Preferred title alias | Used for |
|---|---|---|
| `name` | `GLOBAL [ base name ]`, fallback `NAME` | Run identifier — feeds snapshot folder + card title |
| `relay_overrides` | `RELAY_OVERRIDES` | JSON dict of Prompt-Relay knob overrides. Counts as active only when **wired** into the Director's `relay_overrides` input. |
| `overlay - info` | `OVERLAY - INFO` | Free-form Δ-this-run note. Card renders verbatim |
| `overlay - feedback` | `OVERLAY - FEEDBACK` | Observations + `motion: N/5 · sync: N/5 · sharp: N/5` score lines |
| `working_folder` | `GLOBAL [ path ] - working folder`, fallbacks `working_folder` / `working folder` | Per-project working folder. Duplicates allowed (mount + mirror); first existing wins |
| director node | `LTX Director (Koolook)` or upstream `LTX Director` | Koolook is registered as `LTXDirector__koolook`; legacy `LTXDirector__koolook_v1_3_2` still loads through a compatibility alias. Upstream `LTXDirector` is accepted for A/B comparison and is labeled on the card as original upstream, but `relay_overrides` + the per-segment sigma patch are inert on that path. |

The alias list lives in `scripts/loop_audio.config.json` →
`tracked_multilines`. Earlier aliases win, so the v02 card structure can
prefer `GLOBAL [ base name ]` without losing compatibility with older
`NAME` setups.

The v02 setup note also lists audit variables that are captured into
`notes.md` under `SETUP variables (captured)`. These live in
`tracked_setup_variables` because some are not multiline nodes:

| Capture key | Preferred title alias | Used for |
|---|---|---|
| `input_path_exr` | `INPUT Path [ EXR ]` | Source EXR directory |
| `version` | `GLOBAL [ version ]`, fallback `Version [ Global ]` | Version integer from the setup primitive |
| `run_offset` | `GLOBAL [ run offset ]` | Offset used by the workflow's calculated run number |

The `BASE` setup note is treated as a checklist, not as data. Rows such
as `Image Segments = [ai]`, prompt similarity, commit number, and
Director flavour are filled from the Director timeline and repo state.

The capture run number is not read from the canvas. `loop-audio`
calculates it from the next available `runs/run-NNN_*` folder/log row,
so it keeps incrementing correctly even when the Comfy setup name or
internal naming offset changes.

Save the workflow as `LTX-23-audio_tests_v01.json` in ComfyUI's
`user/default/workflows/` (or whatever filename matches `workflow_pattern`
in [`loop_audio.config.json`](../../../../scripts/loop_audio.config.json)).

## The render cycle

1. Edit the knob — either change the `RELAY_OVERRIDES` multiline JSON,
   or edit a `forks/whatdreamscost_koolook/versions/v1_3_9/*.py` source
   then say `dev-sync-audio`.
2. **Save (API format)** into the workflows dir. Mtime decides which
   file `loop-audio` picks.
3. Queue the render in ComfyUI.
4. Type `loop-audio` in chat once it finishes.
5. Describe the result in chat — the agent has already logged the row
   and snapshotted the folder.

## Card source rule

The card draws **only** from two source families:

1. The `Text Multiline` nodes above for run naming, notes, feedback, and working folder. `RELAY_OVERRIDES` is read through the active Director input link, not merely by title.
2. The active Director node's own widget values and input wiring
   (`LTXDirector__koolook` / legacy Koolook / upstream `LTXDirector`).

Forbidden on the card: `BasicScheduler` / `KSamplerSelect` / `RandomNoise` /
`CFGGuider` widgets, `_dev_build.json`, `git status` output, hardcoded
sampler/CFG/seed constants, substring matches on prompt text. Full
breakdown: [`reading-graph.html`](reading-graph.html).

## Audio src — the five structural states

Derived from director-presence + `audio_vae` input link + `use_custom_audio`
widget + `timeline_data.audioSegments` count. Mirrors what the director
does at runtime
(`forks/whatdreamscost_koolook/versions/v1_3_9/ltx_director.py`).

| Label | Conditions |
|---|---|
| `(no director)` | No Koolook or upstream Director node on the canvas |
| `off (no VAE)` | Director present · audio_vae not wired |
| `model-gen` | audio_vae wired · use_custom_audio = False |
| `custom` | audio_vae wired · use_custom_audio = True · audioSegments non-empty |
| `custom (empty)` | audio_vae wired · use_custom_audio = True · audioSegments empty |

## Segment prompt check

`BASE · SCENE` includes `Prompt mode`, derived from
`timeline_data.segments[].prompt`:

| Label | Meaning |
|---|---|
| `none` | No video segments found |
| `single` | One segment with a prompt |
| `same` | Multiple segments, same prompt after whitespace normalization |
| `per-segment` | Multiple segments with different prompts |
| `missing` | At least one segment has no prompt |

The card also shows `Audio segments`, the count of
`timeline_data.audioSegments`. In the current v02 comparison setup this
is expected to be `2` when both edited audio clips are present on the
Director timeline.

## What lands in `runs/run-NNN_<label>/`

```
runNNN_workflow.json   ← redacted copy of the Comfy file at submission
relay_overrides.txt    ← active Director relay_overrides body; empty/defaults when the socket is unwired
patch_state.txt        ← MAIN sha + last dev-sync-audio sha + fork-dir clean/dirty (audit trail; NOT on the card)
metadata.json          ← structured run/setup/director/repo metadata
notes.md               ← OVERLAY-FEEDBACK + OVERLAY-INFO verbatim + SETUP variables + director's structural state
card.png               ← stable archive card with embedded `koolook_audio_loop` JSON metadata
```

`metadata.json` and the PNG metadata include the technical report:
run number, copied workflow source, setup values, expected output
folder/name/version, Director structure, Director pin tag, MAIN SHA,
last `dev-sync-audio`, sync scope, and fork-dir status. The visual card
keeps only scan-friendly rows; BASE/RUN shows the copied workflow stem
without the `.json` extension.

For compositing beside the rendered QuickTime, the card script copies the
card into the setup's stable output folder under `cards/` as
`cards/<Output name>_runNNN_card.png` (for example
`cards/Bear_2x-FR_AudioFile_K-Dir_h264_v002_run006_card.png`). The
run-folder `card.png` remains the single archive card; the long filename is
only for the external delivery copy. Re-rendering an existing run through
`scripts/make_card_audio.py <run_dir>` refreshes `card.png`, `metadata.json`,
`notes.md`, and that delivery copy. Add `--no-delivery` when refreshing
committed evidence only.

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

## Timed transcript experiment (removed)

The standalone `Koolook Audio Transcript Timeline` node and the
`scripts/transcribe_audio_timeline.py` Whisper helper were removed in #198.
The capability was experimental and superseded by the Koolook Director's
built-in `audio_transcript_json` input — supply it a transcript JSON from
your own tooling if you still want semantic speech timing on top of the raw
audio latent. Nothing in the bundled loop generates that JSON anymore.

## Where settings live (no code edits for the common cases)

| What | Where |
|---|---|
| Workflow filename pattern | [`scripts/loop_audio.config.json`](../../../../scripts/loop_audio.config.json) → `workflow_pattern` |
| ComfyUI workflows dir | Derived from `KOLOOK_COMFYUI_DEV_PATH` in `.env`, or `KOLOOK_COMFYUI_WORKFLOWS_DIR` override |
| Tracked multiline capture keys + title aliases | `loop_audio.config.json` → `tracked_multilines` |
| Tracked setup variables for `notes.md` | `loop_audio.config.json` → `tracked_setup_variables` |
| Fork dir to pin in `patch_state.txt` | `loop_audio.config.json` → `fork_to_track` |
| Whether to render the card | `loop_audio.config.json` → `render_card` |
| ComfyUI dev install target (`dev-sync-audio`) | `.env` → `KOLOOK_COMFYUI_DEV_PATH` |

`.env` is read from the worktree first, then the main repo when running
from a worktree — same fallback as `scripts/make_card.py`.
