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

*No findings yet. First entry appears once a knob's effect is stable across
≥ 3 confirming runs.*

---

## How to add a new finding

1. Pattern is stable across ≥ 3 runs in [`runs/log.md`](runs/log.md).
2. You can describe it as a single declarative sentence ("X causes Y" or "do X, don't do Y").
3. Move it here under an appropriate heading, with a one-paragraph "why" and links to the run(s) that confirmed it.
4. Remove the corresponding hypothesis from the open list in [`README.md`](README.md) (and from [`backstory/audio-lipsync-rationale.md`](backstory/audio-lipsync-rationale.md) if it was raised there) so those files shrink toward truth.
