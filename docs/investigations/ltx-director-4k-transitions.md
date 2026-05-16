# LTX Director — 4K multi-keyframe transition breakdown

**Branch:** `investigate/ltx-director-4k-transitions`
**Target node:** WhatDreamsCost-ComfyUI → `LTXDirector` (v1.3.2)
**Status:** investigation kickoff — no code changes yet.
**Date:** 2026-05-16

---

## Problem statement

(Maintainer's own framing, preserved.)

> I have a problem with 4K resolution.
>
> HD works fine — you don't see the transitions between segments and the reference frames; the video flows. But at 4K it breaks down when the timeline has multiple segments and multiple keyframes.
>
> What it does at 4K is it doesn't properly interpolate between the keyframes — it just makes a **dissolve** between one position and the other one somewhere in between. So instead of actual motion from keyframe A to keyframe B, you get a cross-fade with the subject planted at A on one side and at B on the other side, somewhere in the middle of the segment.
>
> I don't understand why this changes with higher resolution. I want to investigate whether we can fix this in the node, or whether it's a limit baked into how the LTX model itself was trained.
>
> **Observations so far:**
> - HD, multi-segment + multi-keyframe → works fine.
> - 4K, single-keyframe only (only the first frame is given; no transitions required) → 4K video is perfect.
> - 4K, multi-segment + multi-keyframe → breaks down into dissolves.

---

## Conclusion about the current implementation

The LTX Director is a UI wrapper around three orthogonal mechanisms. Two of them are plausible loci for the 4K dissolve symptom; one is unrelated.

### Lineage

The algorithmic core is the **Prompt Relay** technique from the paper *"Prompt Relay: Inference-Time Temporal Control for Multi-Event Video Generation"* (Chen, Huang, Liu — S-Lab, Nanyang Technological University — [arXiv:2604.10030](https://arxiv.org/abs/2604.10030), April 2026).

- Reference implementation by the paper's authors: [GordonChen19/Prompt-Relay](https://github.com/GordonChen19/Prompt-Relay).
- ComfyUI port the WhatDreamsCost node is built on: [kijai/ComfyUI-PromptRelay](https://github.com/kijai/ComfyUI-PromptRelay).

The WhatDreamsCost author took Kijai's port, wrapped it in a canvas-based timeline editor, and bolted on audio compositing + image-guide handling. The cross-attention math itself is essentially unchanged from the paper (with one minor implementation deviation — see below).

### 1. Prompt Relay temporal attention penalty — the algorithmic core

The technique injects a penalty matrix `C(Q, K)` inside the cross-attention softmax:

```
Attn(Q, K, V) = softmax( Q·Kᵀ / √d  −  C(Q, K) ) · V
```

For query frame *f* and key token *k* belonging to segment *s*:

```
cost(f, k) = strength · ReLU( |f − midpoint_s| − w_s )² / (2 · σ_s²)
```

— a flat-top Gaussian: zero penalty inside the segment window `w`, quadratic ramp outside, with σ derived from `epsilon` (default 0.001). The point of the penalty is that during sampling, each video frame is steered to attend only to its own segment's prompt tokens and ignore the others.

Files:
- `prompt_relay.py` — builds the penalty matrices; derives `tokens_per_frame` from the latent's shape.
- `patches.py` — monkey-patches `attn2.forward` (and `audio_attn2.forward`) on every LTX transformer block to inject the penalty term inside the softmax.

**Relevance to the 4K symptom:** `tokens_per_frame` scales with `latent_w × latent_h`. A 4K latent has roughly 16× the tokens-per-frame of an HD latent — i.e. the cross-attention's query side is enormously larger. This implementation uses a **length-independent σ**, so the penalty geometry may not generalise across the token-grid change as resolution scales.

### 2. Guide-image keyframes — the spatial mechanism

Image segments on the timeline emit a `guide_data` payload (image tensor + frame index + strength). The companion `LTXDirectorGuide` node (a subclass of ComfyUI's stock `LTXVAddGuide`) VAE-encodes each image and **splices it into the latent at its assigned frame index** with a noise mask that pins that latent slot. This is the standard LTX keyframe mechanism, unchanged from upstream LTX.

**Relevance to the 4K symptom:** the node only inserts boundary conditions — it does no interpolation itself. **The actual motion between two pinned keyframes is synthesised by the LTX diffusion model at sampling time.** If the model wasn't trained to generate convincing motion at 4K (LTX 2.3's training distribution is well below 4K), the path of least resistance is to linearly blend the two pinned latent states — which is visually indistinguishable from a dissolve. The observation that 4K *single-keyframe* works perfectly is consistent with this: with only one anchor, no inter-frame interpolation is required, so the model never has to reach beyond what it knows how to do.

### 3. Audio + image preprocessing — unrelated

PyAV audio decode/resample/composite, PIL image resize, optional in-memory H.264 round-trip compression of guide images (`img_compression`). None of these touch the temporal generation path; they affect only the audio output and the spatial encoding of the guide frames. Not a suspect.

---

## Minor implementation deviation worth flagging

`prompt_relay.py:93`:

```python
# Paper uses constant sigma = 1/ln(1/epsilon) regardless of segment length
sigma = 1.0 / math.log(1.0 / epsilon) if 0 < epsilon < 1 else 0.1448
```

The author's comment claims this matches the paper, but the paper's stated formula for σ is:

```
σ = (L − w) / ( 2 · √(ln(1/ε)) )
```

— i.e. σ scales with segment length `L` so that the penalty reaches threshold `ε` exactly at the segment boundary. The WhatDreamsCost code uses a length-independent σ instead. Whether this matters at 4K, and how Kijai's port handles it, is worth checking when we start instrumenting.

---

## Round 2 — 2K vs 4K, and the two-stage upscale problem

**Date:** 2026-05-16 (same session)

### New observations (maintainer's framing, preserved)

> 2K works **much better** than 4K. The same three segments don't seem to have nearly as much dissolve — instead they're morphing properly. So the model is better at transitioning from one keyframe to the next without going to a dissolve, which is what we want.
>
> The new problem is the **two-stage workflow**: the second stage (the upscale stage) is not holding the audio instruction that well. It's kind of overwriting stage 1, which is done at half resolution. Stage 1's output is good — I could see this by rendering out that section on its own. But when stage 2 runs, the animation is a bit lost and the lip sync isn't as good.

### Why this could happen

**Why 2K behaves much better than 4K**

Two effects compound:

1. **Model prior, not node code.** LTX 2.3 was trained on a distribution centred well below 4K. Synthesising motion *between* two pinned keyframes is a learned task — the model has to invent latent states that plausibly bridge them. At 4K the model is extrapolating outside its training distribution, and the cheapest extrapolant is a linear blend between the two pinned latents (which looks like a dissolve). At 2K the model is at or near its training distribution, so it can synthesise real motion. The original observation that 4K *single-keyframe* works perfectly is consistent with this — single-keyframe doesn't ask the model to interpolate anything.

2. **Token-grid scaling.** Tokens-per-frame = `(latent_h / patch) × (latent_w / patch)`. From HD → 2K → 4K, this grows roughly 4× and 16×. The Prompt Relay penalty matrix has to span that whole token grid per frame, but its σ is length-independent in this implementation, so the per-frame attention concentration behaves slightly differently as the grid grows. This is a secondary effect — not the dominant cause of the dissolve — but it's a second reason the same workflow gets worse with resolution rather than just being neutrally rescaled.

**Why stage 2 overwrites stage 1 (audio + animation drift)**

I don't have the workflow JSON, so this is hypothesis-shaped, but the standard failure mode of a "low-res generate → upscale + resample" LTX workflow is the combination of three things:

1. **Stage 2 is a fresh KSampler pass with non-trivial denoise.** Anything denoised above ~zero is partially regenerated from noise, and during that regeneration the model's conditioning is re-applied. If the LTX Director node (i.e. the patched model with Prompt Relay + audio_attn2 patches) isn't wired into stage 2, the model defaults to vanilla cross-attention — every video frame attends to every prompt token equally, and the segment-wise routing that stage 1 carefully constructed gets washed out. Symptom: animation "loses register."

2. **Audio conditioning is denoise-fragile.** Lip sync requires sub-frame audio-to-mouth correspondence, encoded in fine-grained latent detail that the `audio_attn2` cross-attention shaped during stage 1. A stage-2 denoise even at 0.2–0.3 partially overwrites those fine details. Unless the audio VAE latent *and* the audio_attn2 patch are re-applied identically on stage 2, the audio prior is diluted relative to the now-stronger spatial prior at high resolution. Symptom: lip sync degrades.

3. **Latent shape changes between stages.** The Prompt Relay penalty matrix is dimensioned to `(latent_frames × tokens_per_frame)`. Stage 2's spatially upscaled latent has 4× the tokens per frame (for a 2× upscale). The penalty matrix has to be rebuilt with the new dimensions. If the rebuild isn't happening — or happens with un-rescaled σ — the temporal binding between segments weakens, which presents as exactly the symptom described.

"Kind of overwriting stage 1" is precisely what high-denoise + missing-PromptRelay-patch + diluted audio conditioning would look like.

### Pros and cons

|  | 4K direct | 2K direct | 2-stage (½-res → upscale) |
|---|---|---|---|
| Inter-keyframe motion | **dissolve** (model out of distribution) | morphs properly | stage 1 good, stage 2 weakens it |
| Audio / lip sync | n/a — motion already broken | holds well in a single pass | **degrades in stage 2** |
| Single-keyframe quality | excellent (no interpolation needed) | excellent | excellent through stage 1 |
| Compute cost | very high | moderate | moderate stage 1 + high stage 2 |
| Within LTX's training distribution? | no | borderline / yes | stage 1 yes, stage 2 no |

The cheapest win, if it's an option, is **2K direct, no two-stage**. The two-stage pattern earns its keep when you need 4K delivery *and* 2K→upscale actually beats 4K-direct — but the stage-2 path has to *preserve* stage 1's conditioning rather than redo it from scratch.

### Directions to investigate (cheapest first)

1. **Inspect the stage-2 wiring.** Find out: what is stage 2 actually doing? Latent upscale + KSampler with what denoise? Is the LTX Director node feeding the patched model into stage 2, or does stage 2 use a stock (unpatched) LTX model? This single question probably explains 80% of the symptom.
2. **Drop stage 2's denoise.** If stage 2 is at 0.3+, try 0.15 or 0.10. Lower denoise preserves more of stage 1's audio-aligned detail at the cost of refinement.
3. **Re-apply the LTX Director on stage 2** with the same `timeline_data` JSON, so the Prompt Relay penalty and audio cross-attention patch are present on stage 2's model clone too. If the dissolve / audio drift still reappears at the upscaled resolution, that confirms it's an LTX-model-side prior limit at high res, not a missing-patch issue.
4. **Run 2K direct at full quality and compare to the 2-stage 4K.** If 2K direct delivers an acceptable result for the delivery target, retire the two-stage path. Cheapest fix is "don't do it."
5. **If 4K delivery is non-negotiable,** consider replacing stage 2's KSampler with a *pixel-space* upscaler (SUPIR, ESRGAN, or a tiled sampler at very low denoise with the Director re-wired). Pixel-space upscale preserves the temporal structure stage 1 built — at the cost of not inventing new motion detail. If stage 2 isn't generating new motion anyway, this is likely a net win.
6. **Diff against Kijai's port.** Plug `kijai/ComfyUI-PromptRelay` into a stage-2-style workflow and see whether audio attention behaves differently. If Kijai's port preserves audio better, the difference points at where the WhatDreamsCost wrapper drops the ball — cheaper than reading Gordon Chen's paper-reference impl line-by-line.
