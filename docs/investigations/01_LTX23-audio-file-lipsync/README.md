# 01 — LTX 2.3 Director: audio-file lip-sync investigation

**Tracking issue:** [#177](https://github.com/malkuthro/ComfyUI-Koolook/issues/177)
**Related:** [`../00_LTX23-base-1-step/README.md`](../00_LTX23-base-1-step/README.md) (the foundational LTX 2.3 investigation — 4K dissolves, 2-stage workflow, scheduler).
**Started:** 2026-05-28
**Status:** investigation-phase patches applied to live install; sync via `sync-audio`.

---

## Problem

When `use_custom_audio: true` and an external audio file is provided to `LTXDirector`, the resulting video is broken: no lip-sync, no meaningful animation, prompt adherence collapses. The only mode that produces good output is `use_custom_audio: false` with `[Audio]: …` in the prompt (model-generated audio).

Goal: investigate whether the Prompt Relay cross-attention math can be tuned to give the text prompt enough pull on the video to overcome the audio-driven prior baked into LTX 2.3's training distribution.

## Folder map

```
01_LTX23-audio-file-lipsync/
├── README.md           ← this file
├── upstream.json       ← upstream version pin + sync target + file list
├── patches/            ← source of truth for modified upstream code
│   ├── ltx_director.py
│   └── prompt_relay.py
└── runs/               ← per-render snapshots
    ├── LOOP.md         ← the iteration protocol
    ├── log.md          ← rolling table of runs
    └── run-NNN_<label>/
        ├── workflow.json
        ├── relay_overrides.txt
        ├── patch_state.txt
        └── notes.md
```

## Patch summary

### Patch 1 — `prompt_relay.py`

`build_segments` now computes σ per-segment using the Prompt-Relay paper formula:

```
σ = (L − w_eff) / (2 · √ln(1/ε))
```

Original code used a length-independent `σ = 1/ln(1/ε)` (≈ 0.1448 at ε=0.001), producing a razor-sharp boundary that did not scale with L. The new formula calibrates the penalty so it hits threshold ε exactly at the segment boundary regardless of segment length.

A `SIGMA_FALLBACK = 0.1448` preserves the prior constant for the degenerate `L ≤ w_eff` corner. Per-segment σ is now logged: `[PromptRelay] seg L=… w_v=… sigma_v=…`.

### Patch 2 — `ltx_director.py`

The `LTXDirector` node gains a new optional input `relay_overrides` (multiline string, default `""`). The maintainer types a JSON dict of Prompt Relay knobs directly into the node's canvas widget. The values live entirely inside the workflow JSON — no disk file, no env var.

Supported keys (all optional):

| Key | Type | Default | Effect |
|---|---|---|---|
| `video_strength` | float | 1.0 | Multiplies `attn2` penalty (video × text). Raise to tighten per-segment text routing. |
| `video_window_scale` | float | 1.0 | Scales the flat-top window width on the video side. < 1.0 = penalty kicks in closer to segment center. |
| `audio_strength` | float | 1.0 | Same for `audio_attn2`. **Note:** no-op for video output when `use_custom_audio=true` (audio latent is frozen at `noise_mask=0`). |
| `audio_window_scale` | float | 1.0 | Audio-side window. |
| `audio_epsilon` | float | inherit | Independent σ for the audio path. |

Underscore-prefixed keys are ignored (use for inline JSON comments). Example value to paste:

```json
{"video_strength": 10.0}
```

Empty field → upstream Prompt Relay behaviour.

## Sync — pushing patches to the live install

MAIN is the source of truth. The live install at `$KOLOOK_WHATDREAMSCOST_PATH` is a destination only.

**Trigger phrase: `sync-audio`** (declared in [`upstream.json`](upstream.json)). Equivalent CLI:

```
python scripts/sync_investigation_patches.py 01_LTX23-audio-file-lipsync
```

The script reads [`upstream.json`](upstream.json), backs up the target's current state (`.bak.<YYYYMMDD>`, once per day), copies `patches/*.py` over, prints a 2-line report.

## Test workflow

`E:/_AI/portable/ComfyUI_windows_312/ComfyUI/user/default/workflows/LTX-23-audio_tests_v01.json` — the maintainer overwrites this single file from inside ComfyUI between iterations.

See [`runs/LOOP.md`](runs/LOOP.md) for the per-render iteration protocol.

## Reversion (any one works)

1. Restore `.bak.<date>` files in the live install dir (created automatically by the sync script).
2. `git -C "$KOLOOK_WHATDREAMSCOST_PATH" checkout -- prompt_relay.py ltx_director.py` — upstream is a git clone (verified: `.git` present).
3. ComfyUI-Manager → Try Update — re-clones from upstream.

## Promotion path

If the patches validate:

1. Run [`license-pre-check`](../../../.claude/skills/license-pre-check/) — upstream is GPL-3.0, Koolook is GPL-3.0 (compatible).
2. Run [`add-external-fork`](../../../.claude/skills/add-external-fork/) to register `WhatDreamsCost-ComfyUI` in [`../../../forks/forks_manifest.yaml`](../../../forks/forks_manifest.yaml).
3. Create `forks/ltx_director_koolook/` with namespaced node ID (`LTXDirector_Koolook`) so it coexists with installed upstream.
4. Update [`../../../forks/THIRD_PARTY.md`](../../../forks/THIRD_PARTY.md).
5. Promote validated findings to [`../../automations/LTX-2.3/findings.md`](../../automations/LTX-2.3/findings.md) per its "How to add a new finding" rule (≥3 confirming runs).
