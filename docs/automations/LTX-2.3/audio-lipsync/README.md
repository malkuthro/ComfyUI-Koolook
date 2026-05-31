# LTX 2.3 audio-file lip-sync — iteration loop

AI-assisted iteration loop for getting audio-file-driven lip-sync working
on the WhatDreamsCost `LTXDirector` (LTX 2.3). The vanilla `use_custom_audio = True`
path produces broken video — no lip-sync, no meaningful animation, prompt
adherence collapses. This automation iterates on a Koolook-modified
`LTXDirector` that adds a `relay_overrides` widget and a per-segment-σ
formula in `build_segments`; the loop sweeps those knobs to find a setting
that preserves video coherence with the external audio.

**Tracking issue:** [#177](https://github.com/malkuthro/ComfyUI-Koolook/issues/177)
**Sibling automation:** [`../base-1step/`](../base-1step/) — same model, the single-stage path with model-generated audio. Locked-in LTX-2.3 architecture facts live in its [`findings.md`](../base-1step/findings.md).
**Backstory:** [`backstory/audio-lipsync-rationale.md`](backstory/audio-lipsync-rationale.md) — why the audio path fails by default, the Prompt-Relay paper, what the two patches are doing mechanically.
**Started:** 2026-05-28

## What the Koolook fork ships

Lives at [`../../../../forks/whatdreamscost_koolook/versions/v1_3_9/`](../../../../forks/whatdreamscost_koolook/versions/v1_3_9/),
loaded by ComfyUI as `LTX Director (Koolook)` in the picker. Two upstream
modifications:

### Modification 1 — per-segment σ in `prompt_relay.py`

`build_segments` now computes σ per-segment using the Prompt-Relay paper
formula:

```
σ = (L − w_eff) / (2 · √ln(1/ε))
```

Upstream used a length-independent `σ = 1/ln(1/ε)` (≈ 0.1448 at ε=0.001) —
a razor-sharp boundary that did not scale with segment length L. The new
formula calibrates the penalty so it hits threshold ε exactly at the
segment boundary regardless of L. A `SIGMA_FALLBACK = 0.1448` preserves
the prior constant for the degenerate `L ≤ w_eff` corner. Per-segment σ
is logged as `[PromptRelay] seg L=… w_v=… sigma_v=…`.

### Modification 2 — `relay_overrides` widget on `LTXDirector`

A new optional multiline-string input on the LTXDirector canvas widget.
The maintainer pastes a JSON dict of Prompt-Relay knobs directly into the
node — values live entirely inside the workflow JSON, no disk file, no
env var.

Supported keys (all optional):

| Key | Type | Default | Effect |
|---|---|---|---|
| `video_strength` | float | 1.0 | Multiplies `attn2` penalty (video × text). Raise to tighten per-segment text routing. |
| `video_window_scale` | float | 1.0 | Scales the flat-top window width on the video side. < 1.0 = penalty kicks in closer to segment center. |
| `audio_strength` | float | 1.0 | Same for `audio_attn2`. **Note:** no-op for video output when `use_custom_audio=true` (audio latent is frozen at `noise_mask=0`). |
| `audio_window_scale` | float | 1.0 | Audio-side window. |
| `audio_epsilon` | float | inherit | Independent σ for the audio path. |

Underscore-prefixed keys are ignored (use for inline JSON comments). Preferred example value to paste:

```json
{"video_strength": 10.0}
```

Empty field → upstream Prompt-Relay defaults (Modification 1 still active —
the new σ formula is unconditional).

If a connected Text Multiline node mangles strict JSON, the parser also accepts
one setting per line:

```text
video_strength: 10.0
video_window_scale: 0.75
```

## Folder map

```
audio-lipsync/
├── README.md           ← this file
├── handoff-checklist.md ← bootstrap for picking up this automation cold
├── findings.md          ← locked-in (currently empty; populated as runs validate)
├── backstory/
│   └── audio-lipsync-rationale.md
└── runs/                ← iteration log + per-render snapshots
    ├── LOOP.md          ← the iteration protocol
    ├── log.md           ← rolling table — one row per render, always
    └── run-NNN_<label>/  ← created by each `loop-audio` capture
        ├── runNNN_workflow.json
        ├── relay_overrides.txt
        ├── patch_state.txt
        ├── metadata.json
        ├── notes.md
        └── card.png
```

Each `loop-audio` capture adds a row to `log.md` and creates a matching
`run-NNN/` snapshot folder. The folder number is derived from both
existing folders and `log.md`, so a missing folder cannot make the next
capture reuse an already logged number.

This module keeps every `loop-audio` capture as a reproducibility snapshot.
There is no extra "keep this run" prompt in this fork-touching loop; scratch
renders should be skipped with "no log" / "don't log this" before capture.

## Iteration loop

1. **Edit a knob.** Either:
   - **Widget-only swap** — change `relay_overrides` JSON on the LTXDirector node in ComfyUI. No sync, no restart; just queue.
   - **Code-level change** — edit
     [`../../../../forks/whatdreamscost_koolook/versions/v1_3_9/prompt_relay.py`](../../../../forks/whatdreamscost_koolook/versions/v1_3_9/prompt_relay.py)
     or [`ltx_director.py`](../../../../forks/whatdreamscost_koolook/versions/v1_3_9/ltx_director.py)
     in this repo. Then run **`dev-sync-audio`** (chat phrase or directly
     [`scripts/sync_to_dev_audio.py`](../../../../scripts/sync_to_dev_audio.py)) - a scoped variant of `dev-sync` that copies just the fork dir, Director web extension, and root `__init__.py` into the live ComfyUI install at `$KOLOOK_COMFYUI_DEV_PATH`, removes the stale pre-v1.3.9 web extension folder if present, and leaves `forks/radiance_koolook/` plus the rest of the tree alone. Restart ComfyUI manually after Python changes so modules re-import. User-initiated only - same rule as `dev-sync`; see project `CLAUDE.md`.
2. **Save workflow** — `Workflow → Save (API Format)` into the working folder.
3. **Render** — queue.
4. **Report** in chat — verbal feedback on sync state, motion, prompt adherence.
5. **Agent captures the run** by appending a row to [`runs/log.md`](runs/log.md) and creating `runs/run-NNN_<label>/` with the full snapshot (workflow JSON copy + relay_overrides + patch state + metadata + notes + card).

The card's BASE/RUN area shows the original copied workflow stem (for
example `LTX-23-audio_tests_03`) instead of the predictable archive name
`runNNN_workflow.json`. The full archive filename and technical details stay
in `metadata.json` and the PNG `koolook_audio_loop` text metadata.

See [`runs/LOOP.md`](runs/LOOP.md) for the full per-render protocol and the retention rationale.

## Timed transcript experiment

Raw custom audio alone can fail to produce usable mouth timing even when
the same words improve the render when typed into the prompt. The loop now
has an optional helper for that gap.

Inside ComfyUI, add `Koolook Audio Transcript Timeline`, set the same
`audio_file`, `image_file`, duration, and FPS as the Director, then link
its `transcript_json` output into the Koolook Director's
`audio_transcript_json` input. Keep `use_custom_audio=True`; use the same
`transcript_json` output with a text preview node to inspect what Whisper
recognized. The Director converts the phrase timings into its internal
`timeline_data`, `local_prompts`, and `segment_lengths` immediately before
Prompt Relay conditioning runs.

For Timeline Editor workflows, link `Koolook Timeline Editor.timeline_data`
into `Koolook Audio Transcript Timeline.timeline_data`. The transcript node
then reads every `audioSegments` clip, including separated clips, and applies
that clip's start frame, trim start, and visible length before merging the
phrases into the final Prompt Relay timing. The node also composes each
speech/pause timing instruction with the active image segment prompt, so
visual directions stay on the timeline image clips while speech timing is
generated automatically.

The same helper can also run from a script when an export file is useful:

```powershell
.\.venv\Scripts\python -m pip install -e ".[audio]"
.\.venv\Scripts\python scripts\transcribe_audio_timeline.py <audio.mp3> --workflow <workflow.json> --out timed-prompts.json --patched-workflow timed-workflow.json
```

The helper uses `faster-whisper` to transcribe speech into timestamped
phrases, inserts closed-mouth pause segments for silence, and emits
Director-shaped `timeline_data`, `local_prompts`, and `segment_lengths`.
Those fields remain useful for debugging or patched-workflow exports. Use
`--patched-workflow` to write a loadable workflow JSON with those fields
already applied to the Koolook Director. Render it with
`use_custom_audio=True` to test whether LTX needs semantic speech timing
in addition to the raw audio latent.

## Workflow JSON — Koolook node ID

New modified runs should use the stable node ID `LTXDirector__koolook`
(display name *"LTX Director (Koolook)"*). Upstream `LTXDirector` is still
accepted for original-vs-Koolook comparison captures; the card labels it as
original upstream, marks `relay_overrides` inert, and shows the upstream pin
tag read automatically from the live `WhatDreamsCost-ComfyUI/pyproject.toml`
beside `$KOLOOK_COMFYUI_DEV_PATH`. Old workflows saved with
`LTXDirector__koolook_v1_3_2` still load through a compatibility alias backed
by the same v1.3.9 implementation, not the byte-identical v1.3.2 class, so
future fork upgrades do not require repeated node replacement.

The upstream node and the Koolook variant still appear side-by-side —
upstream stays vanilla, the Koolook variant carries our two modifications.

If you're starting from an existing workflow that wires the upstream
`LTXDirector`:

1. Open the workflow in ComfyUI.
2. Right-click the `LTX Director` node → **Convert / Replace** with `LTX Director (Koolook)`. Same input/output socket layout — wires are preserved.
3. The new `relay_overrides` widget appears at the bottom of the node — paste your JSON there (empty field is fine).
4. `Workflow → Save (API Format)` into the working folder.

Reverting to upstream behaviour for an A/B comparison: drop in the upstream
`LTXDirector` node next to the Koolook one and toggle which feeds the
downstream `LTXDirectorGuide`. No file backups needed — the upstream
install is untouched.

## Reverting changes to the fork

Edits to `forks/whatdreamscost_koolook/versions/v1_3_9/*.py` are normal
git-tracked changes — `git restore` / `git stash` work as usual. Once
`dev-sync-audio` is run, the running ComfyUI install is updated; the
manual ComfyUI restart after the sync re-imports the Python module.
