# Why we forked LTXDirector for audio-file lip-sync

Backstory for the [LTX 2.3 audio-lipsync automation](../README.md): the
symptom, the mechanism, and what the two fork modifications are doing.
The running iteration narrative lives in
[`../runs/log.md`](../runs/log.md); locked findings in
[`../findings.md`](../findings.md).

## The symptom

When `use_custom_audio: true` and an external audio file is provided to
upstream `LTXDirector`, the resulting video is broken: no lip-sync,
no meaningful animation, prompt adherence collapses. The only mode that
produces good output is `use_custom_audio: false` with `[Audio]: …` in
the prompt (model-generated audio).

We want audio-file mode to work — the model-generated path can't be
art-directed against an existing soundtrack.

## Why upstream fails

Two mechanisms compounded:

### 1. Length-independent σ in `build_segments`

Upstream `prompt_relay.py` computes the segment cost penalty with:

```python
sigma = 1.0 / math.log(1.0 / epsilon) if 0 < epsilon < 1 else 0.1448
```

A constant ≈ 0.1448 at the paper-default ε=0.001. The Prompt-Relay paper
(*"Prompt Relay: Inference-Time Temporal Control for Multi-Event Video
Generation"*, [arXiv:2604.10030](https://arxiv.org/abs/2604.10030))
defines:

```
σ = (L − w) / (2 · √ln(1/ε))
```

— scaling σ with segment length L so the penalty hits threshold ε
exactly at the segment boundary. Upstream's constant-σ approximation
holds when all segments are roughly the same length, but breaks down when
segment lengths vary widely (which is what happens when audio drives the
segment boundaries — speech segments are not uniform).

### 2. No knob for per-render Prompt-Relay tuning

The Prompt-Relay penalty has knobs (`video_strength`, `video_window_scale`,
`audio_strength`, `audio_window_scale`, `audio_epsilon`) that the paper
varies depending on the data. Upstream hardcodes them, so the only way to
tune was edit-and-restart of `prompt_relay.py`. For an iteration loop
sweeping these knobs across runs, that's heavyweight.

## What the fork does

Two upstream files modified, one vendored verbatim:

| File | Change | Why |
|---|---|---|
| `prompt_relay.py` | Per-segment σ per the paper formula, with `SIGMA_FALLBACK = 0.1448` for the `L ≤ w_eff` degenerate corner. | Tracks the paper precisely — same behaviour as upstream when L is uniform; differs when L varies. |
| `ltx_director.py` | Adds a `relay_overrides` multiline JSON input on the LTXDirector node. Underscore-prefixed keys ignored (inline JSON comments). | Per-render knob sweeps without editing the fork file. Workflow JSON carries the override; reproducible across machines. |
| `patches.py` | **Unmodified**. Vendored verbatim because `ltx_director.py` imports `detect_model_type, apply_patches` from it. | Self-contained fork; no runtime dependency on the user's upstream WhatDreamsCost install for these two functions. |

The fork is loaded as `LTXDirector__koolook_v1_3_2` (display name
*"LTX Director (Koolook v1.3.2)"*). Upstream `LTXDirector` continues to
exist in the node picker for A/B comparison.

## Hypothesis space being swept

What's not yet validated and is what the iteration loop is for:

1. **Whether the paper σ formula alone fixes the symptom.** Empty
   `relay_overrides` → fork's σ change is active, all other knobs at
   upstream defaults. If sync recovers, σ was the dominant cause.
2. **Whether `video_strength` > 1 helps.** Tightening per-segment text
   routing should reduce the model's freedom to ignore the prompt under
   audio pressure. Sweep `1, 2.5, 10`.
3. **Whether window scaling helps.** Narrowing the flat-top makes the
   penalty kick in closer to segment center, biasing the model to commit
   to one prompt segment at a time. Sweep `video_window_scale` `0.7 / 1.0`.
4. **Whether audio-side knobs matter at all when `use_custom_audio=true`.**
   The audio latent is frozen at `noise_mask=0` in that mode, so
   `audio_attn2`'s penalty has no effect on the video output. Predict
   `audio_strength` is a no-op; confirm empirically.

Each of these turns into one or more rows in
[`../runs/log.md`](../runs/log.md). Once a pattern shows up across
≥ 3 confirming runs, it graduates to [`../findings.md`](../findings.md).

## Related

- [Prompt-Relay paper](https://arxiv.org/abs/2604.10030) — Chen, Huang, Liu (S-Lab, NTU, April 2026).
- [Reference implementation by paper authors](https://github.com/GordonChen19/Prompt-Relay).
- [Kijai's ComfyUI port](https://github.com/kijai/ComfyUI-PromptRelay) — the basis of WhatDreamsCost's LTXDirector.
- Sibling automation: [`../../base-1step/`](../../base-1step/) — same model, the model-generated-audio path. Locked-in LTX-2.3 architecture facts (spatial upscaler ×2 hardcoding, attention scaling) live in its [`findings.md`](../../base-1step/findings.md) and apply here too.
