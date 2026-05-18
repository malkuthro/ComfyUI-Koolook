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

## Spatial upscaler architecture (locked-in facts)

These are facts derivable from inspecting the workflow graph and the upscaler model, not patterns that needed run-count to settle. Useful to know before designing any stage-2 variant.

### The spatial upscaler is hardcoded ×2

`LTXVLatentUpsampler` with `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` has **no factor widget** — final pixel dims = (encoder input dims) × 2, always. There is no "upscale by 1.5×" path through the learned upscaler. To target anything other than 2× source, either:

- Pre-scale the source before VAE encode (then take the ×2 hit on a smaller input).
- Replace `LTXVLatentUpsampler` with a generic pixel-space `ImageScaleBy` at any factor (loses the learned-upscaler quality, but factor-flexible).

### The brief lives only on the Director → DirectorGuide path

There is **one** `LTXDirector` per workflow even when both stages have their own `LTXDirectorGuide`. Both guides consume the *same* `guide_data` output from the single Director — Stage 2 doesn't re-brief, it re-bakes the same brief at the upscaled latent resolution. The brief/keyframes/audio embeddings only enter conditioning via `LTXDirector → guide_data → LTXDirectorGuide`.

The brief-free counterpart to `LTXDirectorGuide` is `LTXVConditioning` (positive/negative CONDITIONING + frame_rate → LTX-shaped conditioning). Anything downstream that accepts the LTXDirectorGuide output also accepts LTXVConditioning output. This is what makes a Director-less stage-2-only graph possible.

### Attention memory scales `(W × H × T)²`

The LTX 2.3 transformer's attention is over flattened spatio-temporal latent tokens. Memory is **quadratic in the latent token count**. Implication: an HD source pushed through the ×2 upscaler hits ~16× the attention memory of the same model running at HD output. This is why the Director workflow can target 4K (its Stage 1 generates at a smaller base, then ×2 hits a smaller-than-4K target) but a stage-2-only path from HD source OOMs trying to do the same — the latent fed to attention is much bigger.

The same arithmetic constrains the Director: practical resolution ceiling per GPU is set by `(W × H × T)²` of the largest latent any stage holds, not by the model size alone.

## Stage-2-only refine path — open

Stage-2-only = "take a previously-rendered LTX 2.3 clip and run only the spatial ×2 upscaler + low-denoise refine, with no brief, no keyframes, no audio reencode". Recipe:

```
VHS_LoadVideo → ImageScaleBy → VAEEncode → LTXVLatentUpsampler → SamplerCustomAdvanced → VAEDecode → save
                                              ↑                        ↑
                                          video VAE              CFGGuider ← LTXVConditioning ← CLIPTextEncode(+/-)
```

`ImageScaleBy.scale_by` is the memory dial; `BasicScheduler.denoise` is the refine-strength dial. Initial settings to sweep:

| Knob | Starting value | Range to explore |
|---|---|---|
| `ImageScaleBy.scale_by` | `1.0` (passthrough) | `0.5–1.0` — lower to fight OOM |
| `BasicScheduler.denoise` | `0.20` | `0.15–0.30` for refine-only |
| `BasicScheduler.steps` | `8` (matches Director Phase 2) | `6–8` |

**Open hypotheses (not yet validated across runs):**

1. Whether `LTXVConcatAVLatent` is required when the sampler model is the same `ltx-2.3-22b-dev-fp8` checkpoint used in the Director chain. First test build assumed video-only latent is fine; first user attempt OOMed before reaching the sampler so this hasn't been observed end-to-end. If the sampler complains about an AV-paired latent shape, add LTXVConcatAVLatent with a zero/silent audio latent.
2. Whether a stage-2-only refine at HD-output (source × 0.5 × 2 = HD) produces output quality comparable to the Director's own Stage 2 at HD, or whether the missing keyframe-conditioning shows up as a quality gap.
3. Practical `scale_by` ceiling per GPU memory — needs an OOM/no-OOM sweep on the maintainer's hardware to put a number on it.

Promote any of these to a locked-in section once a pattern shows up across ≥ 3 runs (per the file's "How to add a new finding" rule).

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
