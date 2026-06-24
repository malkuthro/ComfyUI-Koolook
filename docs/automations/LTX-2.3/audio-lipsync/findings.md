# LTX 2.3 audio-lipsync — promoted findings

Settled conclusions from the iteration loop. Entries graduate here from the
running [`runs/log.md`](runs/log.md) when a pattern shows up across multiple
renders.

This file is scoped to the **audio-file lip-sync** automation. Model-level
LTX-2.3 architecture facts (spatial upscaler ×2 hardcoding, brief path,
attention quadratic in token count) live in the sibling
[`../base-1step/findings.md`](../base-1step/findings.md) and apply here too.

Anything in this file is **locked in** — don't sweep it again without a reason.

---

## Audio→video binding must be sigma-scheduled, not full-on, to keep motion smooth AND lips synced

**Finding.** In the LTX 2.3 AV transformer the audio→video coupling is one gated
cross-attention per block (`audio_to_video_attn`, gated by
`transformer_options["a2v_cross_attn"]`). Left full-on every denoise step, that
coupling fights hard keyframe pose changes that land on an audio peak → the
keyframe-boundary **jump**. The fix is the Koolook `LTXAVBindSchedule` node: it
scales the a2v output by a gain that **ramps from `early_gain` (early/high-sigma
steps) to 1.0 (late/low-sigma steps)**.

**Promising knee: `early_gain=0.35, bind_start=0.25, bind_end=0.55`** (8-step base
pass) → almost-perfect lip-sync **and** smooth motion, no jumps on the 4-keyframe
goat clip (`Grayscale-BLK` v003 / 00020-1x, single stage).

> ⚠️ **Not yet a clean validation — confound.** That clip's transitions
> (34/63/93) were deliberately placed on audio **lows**, so the absence of jumps
> may be the placement, not the node. The controlled test is still **pending**:
> same on-**peak** timeline, render `early_gain=1.0` (stock — should jump at the
> on-peak transition) vs `early_gain=0.35`. The node is only proven if `0.35`
> removes a jump that `1.0` shows on the *same* on-peak layout. Until then this
> is a strong lead, not a settled finding.

**Why.** Lip-sync *timing* is itself coarse motion — decided in the early steps,
the same place pose transitions resolve. So the knob is a single trade axis:

| `early_gain` | result |
|---|---|
| `0.0` (audio off early, e.g. `0.0 / 0.50 / 0.80`) | smooth motion, **lips never sync** (mouth posed audio-blind) |
| `1.0` (stock, full audio) | lips sync, **transitions jump** |
| **`~0.35`** (`0.35 / 0.25 / 0.55`) | **the knee — both** |

A ~35% audio floor from step 0 (plus full binding by mid-denoise) is enough to
time the lips while staying low enough early that pose changes settle smoothly.
Tuning: lips soft → raise `early_gain` / lower `bind_start`; motion jumps → the
reverse.

**Notes.** The global (scalar) ramp is the reliable lever. A per-frame
"transition-window" mode exists in the node but currently falls back to scalar
on the AV model (`grid_sizes` is `None` at the a2v call and the video token
stream is padded with appended keyframe-guide frames — e.g. `19 timeline + 4
guide = 23` frames × 819 tokens), so mapping tokens→frames isn't reliable yet.
The architecturally-cleaner path for motion-rich lip-sync remains the IC-LoRA
Video track. Node + ramp math: `k_ltx_av_bind_schedule.py` (ComfyUI-Koolook#259).

---

## Hard-pinned image keyframes must land on a latent-bucket center, one per bucket

**Finding.** A hard-pinned LTX Director image keyframe is injected as a *single
latent frame* at the segment's left edge (`insert_frame = seg_start`), and its
effective time is quantized to the model's temporal stride `s`
(`s = scale_factors[0]`, 8 for LTXV) via `L(f) = 0 if f==0 else 1 + (f-1)//s`.
So a pin dropped near a bucket boundary rounds ambiguously (reads as a whole
bucket early/late), and two pins inside the same `s`-frame bucket collapse onto
one latent frame and fight. Either shows up as the keyframe-boundary **"jump"**.
Resolution is at most **one pin per `s` frames** (~333 ms / ~3 pins-per-sec @
24 fps, s=8).

**Confirmed across runs** (`Grayscale-BLK_h264_*`, same audio/seed/sampler):
3-pin layout that happened to land mid-bucket rendered smooth (00016, 00019); a
4-pin layout that pushed a switch near a bucket edge jumped (00018). Audio gaps
were ruled out — the 00018 audio had no leading/interior gap.

**Fix (shipped).** `forks/whatdreamscost_koolook/versions/v2_0_2/` adds
`snap_keyframes_to_grid` (default on): image keyframes are snapped to their
latent-bucket center before the guide is built, and colliding pins are bumped to
the next free bucket with a logged warning. Snapping within a bucket doesn't
change which latent frame a pin lands on, so artistic timing is preserved — it
only removes the boundary ambiguity. Pure math + tests:
[`keyframe_grid.py`](../../../../forks/whatdreamscost_koolook/versions/v2_0_2/keyframe_grid.py),
[`test_whatdreamscost_v2_0_2_keyframe_grid.py`](../../../../tests/forks/test_whatdreamscost_v2_0_2_keyframe_grid.py).
Issue: ComfyUI-Koolook#258.

**Animator note.** Viseme/pose *content* matching (mouth open on a vocal, etc.)
is the separate manual lever and stays the animator's job; the grid-snap only
fixes *placement*, and the per-pin resolution ceiling above bounds how finely a
still pin can track speech (use the IC-LoRA video track for sub-bucket motion).

---

## How to add a new finding

1. Pattern is stable across ≥ 3 runs in [`runs/log.md`](runs/log.md).
2. You can describe it as a single declarative sentence ("X causes Y" or "do X, don't do Y").
3. Move it here under an appropriate heading, with a one-paragraph "why" and links to the run(s) that confirmed it.
4. Remove the corresponding hypothesis from the open list in [`README.md`](README.md) (and from [`backstory/audio-lipsync-rationale.md`](backstory/audio-lipsync-rationale.md) if it was raised there) so those files shrink toward truth.
