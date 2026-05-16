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
