# LTX 2.3 — promoted findings

Settled conclusions from the iteration loop. Entries graduate here from the running [iterations log](../../../).<br>
Anything in this file is **locked in** — don't sweep it again without a reason.

## Sampling — scheduler shape

**`linear_quadratic` with 8 steps is the right scheduler for LTX Director's multi-segment workflows.**

The sigma curve it produces — flat plateau near σ ≈ 1.0 for the first ~5 steps, then ~80% of the denoising compressed into steps 5→8 — gives the cleanest inter-keyframe motion and fast-transition quality. Empirically confirmed across multiple renders during the kickoff investigation.

- Other scheduler shapes degrade fast motion and keyframe transitions.
- Both Phase 1 and Phase 2 keep this scheduler. Phase 2 starts slightly lower on σ because the lower denoise value shifts the entry point down the curve.
- Source: [investigation Round 4](../../investigations/ltx-director-4k-transitions.md#round-4--scheduler-finding--tracking-system-plan).

## Sampling — locked knobs

| Knob | Value | Why |
|---|---|---|
| Sampler | `euler` | No reason yet to vary. |
| CFG | `1.0` | LTX 2.3 default. |
| Seed | `12` (fixed) | Held constant to isolate single-variable changes. Sweep seed only after a setting stabilises and we want to test variance. |
| ε (epsilon) | `0.001` | Paper default — sharpest segment boundaries. |
| Custom audio | `off` | Audio is model-generated from text prompts. See §"Audio mode" below. |

## Resolution behaviour

**LTX 2.3 transitions degrade above ~2K.**

- HD multi-segment + multi-keyframe → motion synthesises cleanly.
- 2K multi-segment + multi-keyframe → still morphs properly between keyframes.
- 4K multi-segment + multi-keyframe → degrades into a **dissolve** between pinned keyframes (model out-of-distribution; cheapest extrapolant is a linear blend).
- 4K *single-keyframe* (no interpolation required) → works perfectly. Confirms it's an interpolation-prior limit, not a sampling failure.

Implication: **single-stage 2K is currently the recommended target** for multi-segment work. Two-stage 2K-then-upscale is being investigated separately.

## Audio mode

`use_custom_audio = False` (model-generated audio) is what the current scene uses. This changes the noise-mask story relative to what we initially assumed:

- With custom audio (True), `audio_attn2`'s audio latent has noise_mask = 0 throughout, preserved across stages.
- With model-generated audio (False), the audio latent is *generated* alongside the video. Stage 2 denoise re-samples both. Lip sync drift in stage 2 is the visible symptom.

## Two-stage workflow — open

Two-stage = render at half-res Phase 1, upscale + resample at Phase 2 via `LTXVLatentUpsampler [BETA]`.

**Suspects, not yet resolved:**
1. Phase 2 `denoise = 0.80` is too high for "preserve stage 1" semantics. Most LTX reference workflows use 0.30–0.40 for this step. Sweep this first.
2. The learned `LTXVLatentUpsampler` may hallucinate mouth-region detail before Phase 2 begins denoising. Compare against bicubic latent upscale.
3. LTX 2.3 prior weakens at the Phase 2 resolution itself.

Current working hypothesis (Round 3 of investigation): #1 is dominant; #2 is a secondary contributor; #3 is the floor.

See [`../../investigations/ltx-director-4k-transitions.md`](../../investigations/ltx-director-4k-transitions.md) for the full hypothesis tree and test plan.

## How to add a new finding

1. Pattern is stable across ≥ 3 runs in `_AI/iterations.md` (per-project working folder).
2. You can describe it as a single declarative sentence ("X causes Y" or "do X, don't do Y").
3. Move it here under an appropriate heading, with a one-paragraph "why" and a link to the run(s) that confirmed it.
4. Remove the corresponding hypothesis from the open list above so the file shrinks toward truth.
