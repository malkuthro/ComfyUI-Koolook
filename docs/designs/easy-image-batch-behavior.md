<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# Easy Image Batch — behavior matrix (proposed)

Status: **Canonical** (approved 2026-06-18). Implemented in
`k_easy_image_batch.py` and mirrored in
[`docs/user_guide/nodes/koolook_image/easy_image_batch.md`](../user_guide/nodes/koolook_image/easy_image_batch.md).
Companion:
- [`easy-image-batch-demo-setup.json`](easy-image-batch-demo-setup.json) — a
  loadable ComfyUI workflow with one `Easy Pattern → Easy Image Batch →
  Preview Image` cluster per matrix row (regenerate via
  [`build_easy_image_batch_demo.py`](build_easy_image_batch_demo.py)). Load it
  in ComfyUI to run the modes side by side against the live node. (Generated
  from an earlier draft — if a row disagrees with the matrix below, the matrix
  and the node are canonical.)

This captures the agreed changes:

1. `keyframe_batch` → **`keyframes_insert`** (rename).
2. **No image source connected → clean placeholder batch** sized by new
   `width`/`height` widgets (no error).
3. **Insert-over-source mode** — `keyframes_insert` + `source_batch` composite
   the inserts over the source cut window.
4. **`image1`–`image4` are "super-overwrites"** — when connected they replace
   whatever is at their index in **every** mode (top priority). When **not**
   connected they do **nothing** (the old "pull from `source_batch` via
   `imageN_frame`" fallback — Mode 2 — is removed).

---

## The layered model (single source of truth)

The output is built in three layers, lowest first. A later layer overwrites an
earlier one at the same output index.

```
output index i  ↔  VFX frame v = cut_start_frame + i        (window: i in [0, total_frames))

┌─ Layer 3 — SUPER-OVERWRITE (image1–image4) ────────────────┐  highest priority
│  Each CONNECTED imageN → placed at index (imageN_frame      │
│  − cut_start_frame). Wins over everything. Unconnected      │
│  slots contribute NOTHING. (imageN_frame matters only       │
│  when imageN is connected.)                                 │
├─ Layer 2 — SEQUENCE ───────────────────────────────────────┤
│  • keyframes_insert connected (INSERT): i-th incoming frame │
│    → i-th source_frames position (ascending, deduped).      │
│  • else (SELECT): each source_frames number N picks         │
│    source_batch[N−1] → index (N − cut_start_frame).         │
├─ Layer 1 — BACKGROUND ─────────────────────────────────────┤  lowest priority
│  • insert-over-source (keyframes_insert AND source_batch):  │
│    source_batch cut window (index i ← source_batch[v−1];    │
│    frames beyond the source fall back to placeholder).      │
│  • otherwise: placeholder_color everywhere.                 │
└─────────────────────────────────────────────────────────────┘
```

**Background rule.** Layer 1 is the source cut window whenever `source_batch`
is connected **and** either an insert is active (`keyframes_insert`) **or** the
`source_frames` list is empty; otherwise it is `placeholder_color`.

**What counts as "placed"** (drives `alpha_batch`, `selected_image_batch`,
`selected_frames`): normally the Layer 2 placements **and** Layer 3 overwrites;
the Layer 1 `source_batch` backdrop is *not* placed (just the background). So
`alpha_batch` is `0.0` at sequence inserts/picks and connected-slot overwrites,
`1.0` everywhere else (flipped by `invert_alpha`).

**Select passthrough is the exception (row 5).** When the source passes through
with an empty list, the covered source frames *are* kept content (`alpha`
`0.0`), and the **uncovered tail becomes the selection** (`alpha` `1.0`,
emitted in `selected_*`) — an extend-a-clip inpaint mask. The selection here is
the *gap*, the inverse of every other row. Insert-mode passthrough (row 9)
keeps the plain backdrop semantics (nothing placed).

**Resolution / dtype** come from the first connected image (insert: from
`keyframes_insert`; select: `image1` → `source_batch` → `image2…4`). With **no**
image connected at all, the output is sized by the `width`/`height` widgets
(channels = 3), on CPU/float — a clean placeholder batch.

**Tie-break:** if two connected slots target the same index, the
higher-numbered slot wins (`image4` > … > `image1`). Sequence + slot at the
same index → the slot wins (Layer 3 > Layer 2).

---

## Mode matrix

Legend — `keyframes_insert` = **KI**, `source_batch` = **SB**, `source_frames`
list = **list**, connected `image1–4` = **slots**.

| # | KI | SB | list | slots | Background | Placed (alpha 0.0) | `selected_*` | Mode label |
|---|----|----|------|-------|-----------|--------------------|--------------|-----------|
| 1 | ✗ | ✗ | — | none | placeholder (`width`×`height`) | nothing | empty | **clean batch** |
| 2 | ✗ | ✓ | has | none | placeholder | source picks at listed positions | the picks | select |
| 3 | ✗ | ✗ | — | some | placeholder | connected slots | the slots | select (slots) |
| 4 | ✗ | ✓ | has | some | placeholder | source picks **+** slots (slots win on overlap) | picks + slots | select |
| 5 | ✗ | ✓ | empty | none | **source cut window** | covered source (kept) | the uncovered **gap** | source passthrough / extend |
| 6 | ✓ | ✗ | has | none | placeholder | scattered inserts | the inserts | offset / reconstruct |
| 7 | ✓ | ✗ | empty | none | placeholder | nothing | empty | **clean** (nothing inserted) |
| 8 | ✓ | ✓ | has | none | **source cut window** | scattered inserts (overwrite source) | the inserts | insert-over-source |
| 9 | ✓ | ✓ | empty | none | **source cut window** | nothing | empty | source passthrough |
| 10 | ✓ | ✓ | has | some | **source cut window** | inserts **+** slots (slots win) | inserts + slots | insert-over-source |
| 11 | any | any | any | **set but NOT connected** | (per above) | **slots contribute nothing** | (per above) | (per above) |

Row **11** was the bug fixed here: a slot with a non-default `imageN_frame`
that is **not wired** has zero effect on the output. The old behaviour (an
unconnected slot pulling from `source_batch`) is gone.

---

## Edge cases & logging

- **List ranges.** `N-M` (e.g. `14-17`) expands inclusively; a descending
  `M-N` or any non-integer token warns and is skipped. Applies in both modes.
- **List token not in `source_batch`** (select): warn + skip that number.
- **List/insert position outside the cut window**: dropped, summarized once.
- **More inserts than list positions** / **more positions than inserts**:
  reported as "extra insert frames" / "positions with no insert frame".
- **`source_batch` shorter than the cut window** (insert-over-source): the
  uncovered tail frames stay placeholder; count reported.
- **Empty list** in an insert mode: clean placeholder (no SB) or clean source
  passthrough (with SB) — console note "frame list is empty".
- **Mismatched H/W/C** among connected inputs: hard `ValueError` (unchanged).
- Every run still prints one summary line: mode, cut window, placed count, and
  any of the notes above.

---

## Decisions (resolved)

1. **Tie-break direction** — higher-numbered slot wins (`image4` > `image1`);
   a slot beats a sequence pick / insert at the same index (Layer 3 > Layer 2).
2. **Empty-list passthrough (select)** — `source_batch` + an empty list passes
   the source cut window through (row 5). Covered source frames are kept content
   (`alpha` 0.0); a source shorter than the output leaves a placeholder tail
   that becomes the selection (`selected_*`) — an extend/inpaint mask. The
   selection here is the *gap*, the inverse of other rows.
3. **Mode-2 fallback removed** — an unconnected slot never pulls from
   `source_batch`; `imageN_frame` is only a placement target for a wired image.
4. **Range syntax** — the frame list accepts inclusive ranges (`14-17` →
   14,15,16,17) in both select and insert; descending/non-integer tokens warn.
