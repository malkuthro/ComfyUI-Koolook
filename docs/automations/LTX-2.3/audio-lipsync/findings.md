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
