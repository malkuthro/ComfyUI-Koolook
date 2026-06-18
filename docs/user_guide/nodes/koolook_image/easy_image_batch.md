# Easy Image Batch (Koolook)

| Aspect | Value |
|---|---|
| Display name | `Easy Image Batch (Koolook)` |
| Node ID | `easy_ImageBatch` |
| ComfyUI category | `Koolook/Image` |
| Source | Koolook-native |
| Source file | [`k_easy_image_batch.py`](../../../../k_easy_image_batch.py) |

## What it does

Places up to *N* keyframes on a 1-based VFX timeline, then outputs a
**cut window** of that timeline as a fixed-length `IMAGE` batch. Empty
positions are filled with a chosen placeholder colour (Black / 50 % Gray
/ White). Pairs the image batch with an `alpha_batch` mask marking
selected vs empty frames, and also returns just the in-cut keyframes
packed back-to-back as a separate short-batch output.

Designed for sparse control sequences — e.g. feeding a video model like
Wan 2.2 a 81-frame guide where only frames 41, 63, 78, 88, 98 and 121
carry real image data and the rest tell the model "no latent update
here". The cut window lets you grab any sub-range of a longer sequence
without needing an external `ImageFromBatch` slicer.

**Insert modes (the inverse).** Connect a packed sequence to
`keyframes_insert` and the node flips around: instead of *selecting* sparse
keyframes out of a dense source, it *scatters* an already-dense short
sequence onto the keyframe positions from the frame list — placing frame
*i* at the *i*-th listed number. The background depends on whether
`source_batch` is connected:

- **`source_batch` not connected (offset / reconstruct mode):** gaps are
  filled with the placeholder. Use it to rebuild a full-length sequence
  from frames you selected, exported, processed, and want back in their
  original spacing (step 2 of a select → process → reconstruct workflow).
- **`source_batch` connected (insert-over-source mode):** the source cut
  window is the background and the listed positions are overwritten with
  the insert frames — composite more-than-one processed frames back into
  the source video at chosen indexes.

**No inputs ⇒ clean batch.** With no image source connected at all, the
node emits a clean placeholder batch sized by the `width`/`height` widgets
instead of raising an error.

## Inputs

### Required widgets

| Input | Type | Default | Description |
|---|---|---|---|
| `total_frames` | `INT` | `81` | Length of the cut window (= length of the output). Range: `1..1024`. |
| `cut_start_frame` | `INT` | `1` | First VFX frame in the output. The output represents `[cut_start_frame .. cut_start_frame + total_frames - 1]`. Frames placed outside this window are dropped from the cut and listed in a single end-of-run summary log. |
| `placeholder_color` | enum | `Black` | Fill colour for unoccupied frames in `image_batch`. `Black = 0.0`, `Gray = 0.5`, `White = 1.0`. Independent of `alpha_batch`. |
| `invert_alpha` | `BOOLEAN` | `false` (`inpaint`) | Off = inpaint convention (selected = `0.0` black, empty = `1.0` white). On = compositing alpha (selected = `1.0`, empty = `0.0`). |
| `source_frames` | multiline `STRING` | `""` | Optional list of VFX frame numbers to pick from `source_batch`. Tokens may be separated by commas, newlines, spaces, tabs, or any mix (`"1, 27\n41 63"` → `[1, 27, 41, 63]`). Each token picks `source_batch[N - 1]` and places it at output index `N - cut_start_frame`. Bad tokens warn and skip. Frames not present in `source_batch` warn; frames outside the cut window are dropped silently and reported in the end-of-run summary. |
| `image1_frame` … `image4_frame` | `INT` | `5`, `9`, `13`, `17` | VFX timeline position (1-based) for the slot's keyframe. Used **only** when the matching `imageN` is connected — it's the output frame that image overwrites (the top layer). No longer a `source_batch` pick index. |

### Optional input slots

| Input | Type | Description |
|---|---|---|
| `keyframes_insert` | `IMAGE` | Packed sequence for **insert modes**. When connected, the node switches from *select* to *insert*: frame *i* is placed at the *i*-th position in `source_frames` (ascending). Connected `image1…image4` still composite on top of the inserts. The background is placeholder (no `source_batch`) or the `source_batch` cut window (insert-over-source). |
| `source_batch` | `IMAGE` | Pre-batched image stack, e.g. straight from `Load Video`. `source_batch[0]` is treated as VFX frame 1; `source_batch[k]` is VFX frame `k + 1`. In insert mode it becomes the composite background. |
| `image1` … `image4` | `IMAGE` | Per-slot keyframes. Each must be a single-frame `IMAGE` (batch size 1). Pre-batched stacks must come through `source_batch` instead. |

### Fallback widgets (last two)

| Input | Type | Default | Description |
|---|---|---|---|
| `width` | `INT` | `512` | Fallback output width. Used **only** when no image input provides dimensions (a fully empty node producing a clean placeholder batch). Ignored whenever an image source is connected. Range: `1..8192`. |
| `height` | `INT` | `512` | Fallback output height. Same fallback-only semantics as `width`. Range: `1..8192`. |

These two are deliberately the **last** widgets on the node so they append
to the end of the saved widget list — keeping older saved workflows (which
predate them) correctly aligned.

With **no image source connected at all**, the node emits a clean
placeholder batch sized by the `width`/`height` widgets (no error). When a
source *is* connected, all connected inputs must share the same
`H × W × C`. (In **insert modes** — `keyframes_insert` connected — that
input drives the sequence placement; `source_batch`, if present, supplies
the composite background; and connected `image1…image4` overwrite on top.)

## Outputs

| Output | Type | Description |
|---|---|---|
| `image_batch` | `IMAGE` | The cut window as a `total_frames`-long batch starting at VFX frame `cut_start_frame`. Selected positions carry the picked images; everything else is the chosen placeholder colour. |
| `alpha_batch` | `MASK` | Same length as `image_batch`. Default convention: selected = `0.0`, empty = `1.0`. Toggle `invert_alpha` to flip. |
| `selected_image_batch` | `IMAGE` | **Only** the keyframes that actually landed inside the cut window, packed back-to-back in ascending order. Length = number of placed keyframes; empty (length 0) if nothing was placed. |
| `selected_frames` | `STRING` | Comma-separated VFX frame numbers of the placed-inside-cut keyframes, sorted ascending and deduped (e.g. `"41, 63, 78, 88, 98, 121"`). Same format as the `source_frames` input — paste it (or wire it) into another `easy_ImageBatch` to reuse the same selection. |

## Frame-numbering model

Two indices are computed per VFX frame number `N`:

```text
source pick index   = N - 1                  # source_batch[0] is VFX frame 1
output (cut) index  = N - cut_start_frame    # cut window
```

These are independent. A frame is *in the cut* iff its output index is
in `[0, total_frames)`. The two are equal only when `cut_start_frame =
1` (the default), so the previously-shipped behaviour is preserved by
default.

There is no exposed knob to renumber `source_batch` (i.e. say "my video
starts at VFX frame 100"). If your stack represents a different VFX
range, renumber externally before connecting.

## Select mode — the three layers

When `keyframes_insert` is **not** connected, the output is built in three
layers, lowest first. A higher layer overwrites a lower one at the same
output index.

### Layer 1 — Background

- **`source_batch` connected and `source_frames` empty → passthrough.** The
  source cut window is laid down as the output (output index *i* ←
  `source_batch[cut_start_frame + i - 1]`; frames beyond the source stay
  placeholder). "Source in → source out" with nothing else to do — the same
  backdrop insert-over-source uses, so the empty-list case behaves the same
  whether or not an insert is wired.
- **Otherwise → placeholder.** A non-empty `source_frames` list means *select*
  mode: the named frames are pulled onto a neutral `placeholder_color`
  background. With no `source_batch` at all the background is placeholder too.

The background is never counted as *placed* — `alpha_batch` and the
`selected_*` outputs describe only the picks and slots placed on top of it.

### Layer 2 — `source_frames` picks

Each VFX number in `source_frames` picks `source_batch[N - 1]` and places it
at output index `N - cut_start_frame` (if inside the cut), on top of the
background. Unlimited keyframes — removes the old 4-slot ceiling. Bad tokens
warn and skip; frames absent from `source_batch` warn; outside-cut frames are
dropped and summarised.

### Layer 3 — Manual slots (`image1`-`image4`), top priority

Connect `imageN` and set `imageN_frame` to its VFX position; the image
overwrites that output frame **on top of everything** — background and picks
alike. On a collision the **higher-numbered slot wins** (`image4` > `image1`),
and a slot beats a Layer-2 pick at the same position.

An **unconnected** slot does nothing. `imageN_frame` is only a placement
target for a *wired* `imageN` — it is no longer a `source_batch` pick index.
(The old behaviour, where an unwired slot pulled `source_batch[imageN_frame -
1]` whenever the list was empty, is removed: it surfaced seemingly-random
source frames. To pick frames from a video, use the `source_frames` list.)

The set of placed positions is deduped automatically (`selected_*` reflect the
final, deduped state).

### Mode 4 — Insert (`keyframes_insert`)

The inverse of Modes 1–3. Connect a packed `IMAGE` sequence (e.g. frames
you previously pulled via `selected_image_batch`, processed, and
re-loaded) to `keyframes_insert`. The node switches to **insert mode** and
reads `source_frames` as *destination positions*: the *i*-th frame of
`keyframes_insert` is placed at output index `position[i] - cut_start_frame`.
Positions are taken ascending and de-duplicated, so it round-trips cleanly
with the ascending `selected_image_batch` + `selected_frames` a select-mode
node produces.

The background depends on `source_batch`:

- **Offset / reconstruct (no `source_batch`).** Gaps are filled with
  `placeholder_color`, at the original sequence length.
- **Insert-over-source (`source_batch` connected).** The `source_batch` cut
  window is laid down first (output index *i* ← `source_batch[cut_start_frame
  + i - 1]`; cut frames beyond the source fall back to the placeholder and
  are reported), then the listed positions are overwritten with the insert
  frames. This composites several processed frames back into the source
  video at chosen indexes.

- **Wire-driven switch.** Insert mode is active whenever `keyframes_insert`
  is connected. Connected `image1…image4` still apply — they composite on top
  of the inserts (Layer 3): a slot beats an insert at the same position, and
  the higher-numbered slot wins a slot-vs-slot collision.
- **Alpha marks the placed frames.** Regardless of background, `alpha_batch`
  is `0.0` at the inserted positions and any slot overwrites, and
  `selected_image_batch` / `selected_frames` describe just those.
- **Counts needn't match.** Extra incoming frames beyond the list are
  reported as unused; positions beyond the available frames are reported as
  missing. Outside-cut positions (inserts or slots) are dropped, same as
  select mode.
- **Empty list ⇒ nothing inserted** — a clean placeholder batch (no
  `source_batch`) or a clean `source_batch` cut-window passthrough — with a
  console note. (Connected slots still composite on top.)

## Conventions

- **1-based VFX numbering.** `imageN_frame`, `source_frames`,
  `cut_start_frame` are all 1-based. `source_batch[0]` is VFX frame 1.
  Frame `0` is invalid and out-of-source-range; frame `1` is the first
  real frame.
- **Cut-window drops.** Frames placed outside the cut window aren't
  errors — they're intentionally absent. The console prints one summary
  line per run noting the cut range, the placed count, and any
  outside-cut drops.
- **Alpha vs mask vocabulary.** `alpha_batch` is a `MASK`-typed output
  but follows whichever convention `invert_alpha` selects. The default
  (`inpaint`) matches the previous `mask_batch` semantics for backward
  compatibility (selected = black, empty = white).

## Example A — `cut_start_frame = 1` (no cut, full timeline)

| Field | Value |
|---|---|
| `source_batch` | `Load Video` (121 frames) |
| `source_frames` | `"1, 27, 41, 63, 78, 88, 98, 121"` |
| `cut_start_frame` | `1` |
| `total_frames` | `121` |
| `placeholder_color` | `Gray` |
| `invert_alpha` | `false` (inpaint) |

| Output | What you get |
|---|---|
| `image_batch` | 121 frames; the 8 picks land at indices 0 / 26 / 40 / 62 / 77 / 87 / 97 / 120. Rest is gray. |
| `alpha_batch` | 121-frame mask; those 8 indices are `0.0`, the other 113 are `1.0`. |
| `selected_image_batch` | 8 frames, packed back-to-back in timeline order. |
| `selected_frames` | `"1, 27, 41, 63, 78, 88, 98, 121"` |

Console summary: `cut window: frames 1..121 (121 frames). 8 placed.`

## Example B — `cut_start_frame = 41`, `total_frames = 81` (cut)

Same inputs as Example A, but with the cut window changed to start at
VFX frame 41 for 81 frames.

| Output | What you get |
|---|---|
| `image_batch` | 81 frames representing VFX frames 41..121. Frames 41 / 63 / 78 / 88 / 98 / 121 land at output indices 0 / 22 / 37 / 47 / 57 / 80. Rest is gray. |
| `alpha_batch` | 81-frame mask, 6 indices selected. |
| `selected_image_batch` | 6 frames, packed back-to-back. |
| `selected_frames` | `"41, 63, 78, 88, 98, 121"` |

Console summary: `cut window: frames 41..121 (81 frames). 6 placed. 2 outside cut: 1, 27.`

This replaces the workflow pattern of feeding the Easy Image Batch
output through an external `ImageFromBatch (batch_index=40, length=81)`
node — the cut is built in.

## Example C — Offset mode (reconstruct a selection)

Step 1 produced 5 enhanced frames (a 5-frame sequence) that originally
came from VFX frames 1, 10, 19, 30, 40 of a 41-frame shot. Step 2 rebuilds
the original length:

| Field | Value |
|---|---|
| `keyframes_insert` | the 5 processed frames (e.g. `Load EXR` / image batch) |
| `source_frames` | `"1, 10, 19, 30, 40"` |
| `cut_start_frame` | `1` |
| `total_frames` | `41` |
| `placeholder_color` | `Black` |

| Output | What you get |
|---|---|
| `image_batch` | 41 frames; the 5 processed frames land at indices 0 / 9 / 18 / 29 / 39. Rest is black. |
| `alpha_batch` | 41-frame mask; those 5 indices are `0.0`, the other 36 are `1.0` — ready as an inpaint mask for the gaps. |
| `selected_image_batch` | the 5 frames, packed back-to-back. |
| `selected_frames` | `"1, 10, 19, 30, 40"` |

Console summary: `offset mode: cut window frames 1..41 (41 frames). 5 placed.`

## Example D — Insert-over-source (composite into a video)

Same 5 processed frames as Example C, but now you want them dropped back
**into the original 41-frame video** at their positions rather than onto a
black background — the untouched frames keep the source footage.

| Field | Value |
|---|---|
| `keyframes_insert` | the 5 processed frames |
| `source_batch` | the original 41-frame `Load Video` stack |
| `source_frames` | `"1, 10, 19, 30, 40"` |
| `cut_start_frame` | `1` |
| `total_frames` | `41` |

| Output | What you get |
|---|---|
| `image_batch` | 41 frames; the 5 processed frames overwrite indices 0 / 9 / 18 / 29 / 39, every other frame is the original source frame. |
| `alpha_batch` | 41-frame mask; only those 5 indices are `0.0` (the inserts), the rest `1.0`. |
| `selected_image_batch` | the 5 inserted frames, packed back-to-back. |
| `selected_frames` | `"1, 10, 19, 30, 40"` |

Console summary: `insert-over-source mode: cut window frames 1..41 (41 frames). 5 placed.`

## Tips

- **Round-tripping selections.** Wire `selected_frames` of one node
  into `source_frames` of another to chain or fan out the same picks
  across workflows.
- **A few explicit keyframes.** Plug single-frame images into `image1`…
  `image4` and set each `imageN_frame`. They overwrite those positions on top
  of whatever else the node produces — no `source_batch` needed.
- **Picking many frames from a video.** Connect `source_batch` and list the
  VFX numbers in `source_frames` (no 4-keyframe ceiling). Reach for `image1`…
  `image4` only to *override* individual positions on top of the picks.
- **Cutting an output.** Set `cut_start_frame` to the first VFX frame
  you want and `total_frames` to the length of the cut. No external
  slicer node needed.
- **Inverted alpha for compositing.** Toggle `invert_alpha` on when you
  want to use the alpha output as a true compositing alpha (selected =
  white = visible).
