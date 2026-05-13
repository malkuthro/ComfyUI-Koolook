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

## Inputs

### Required widgets

| Input | Type | Default | Description |
|---|---|---|---|
| `total_frames` | `INT` | `81` | Length of the cut window (= length of the output). Range: `1..1024`. |
| `cut_start_frame` | `INT` | `1` | First VFX frame in the output. The output represents `[cut_start_frame .. cut_start_frame + total_frames - 1]`. Frames placed outside this window are dropped from the cut and listed in a single end-of-run summary log. |
| `placeholder_color` | enum | `Black` | Fill colour for unoccupied frames in `image_batch`. `Black = 0.0`, `Gray = 0.5`, `White = 1.0`. Independent of `alpha_batch`. |
| `invert_alpha` | `BOOLEAN` | `false` (`inpaint`) | Off = inpaint convention (selected = `0.0` black, empty = `1.0` white). On = compositing alpha (selected = `1.0`, empty = `0.0`). |
| `source_frames` | multiline `STRING` | `""` | Optional list of VFX frame numbers to pick from `source_batch`. Tokens may be separated by commas, newlines, spaces, tabs, or any mix (`"1, 27\n41 63"` → `[1, 27, 41, 63]`). Each token picks `source_batch[N - 1]` and places it at output index `N - cut_start_frame`. Bad tokens warn and skip. Frames not present in `source_batch` warn; frames outside the cut window are dropped silently and reported in the end-of-run summary. |
| `image1_frame` … `image4_frame` | `INT` | `1`, `5`, `9`, `13` | VFX frame number (1-based) for the slot's keyframe. In Mode 2 also used as the `source_batch` pick index. |

### Optional input slots

| Input | Type | Description |
|---|---|---|
| `source_batch` | `IMAGE` | Pre-batched image stack, e.g. straight from `Load Video`. `source_batch[0]` is treated as VFX frame 1; `source_batch[k]` is VFX frame `k + 1`. |
| `image1` … `image4` | `IMAGE` | Per-slot keyframes. Each must be a single-frame `IMAGE` (batch size 1). Pre-batched stacks must come through `source_batch` instead. |

At least one of `image1` or `source_batch` must be connected; otherwise
the node raises a clear `ValueError`. All connected inputs must share
the same `H × W × C`.

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

## Modes

The node supports three input modes that can be freely mixed.

### Mode 1 — Per-slot images (up to 4 keyframes)

Connect `imageN` and set `imageN_frame` to its VFX timeline position.
The image lands at output index `imageN_frame - cut_start_frame` if
that's inside the cut. No `source_batch` needed. Up to 4 keyframes
total.

### Mode 2 — `source_batch` + the 4 slots (up to 4 keyframes from a video)

Connect a pre-batched stack to `source_batch` (e.g. from `Load Video`)
and leave `imageN` unconnected. Each unconnected slot pulls
`source_batch[imageN_frame - 1]` and places it at output index
`imageN_frame - cut_start_frame` — collapsing the typical "Load Video →
Get Images From Batch In Range × N → Easy Image Batch" helper chain
into a single node.

This mode is active when `source_frames` is empty.

### Mode 3 — `source_batch` + `source_frames` list (unlimited keyframes)

Connect `source_batch` and fill `source_frames` with as many VFX frame
numbers as you need. Removes the 4-keyframe ceiling.

**Important interaction rule:** when `source_frames` is non-empty, the
4 manual slots contribute ONLY where `imageN` is explicitly connected —
the `imageN_frame` defaults do **not** pick from `source_batch` in this
mode. The list fully controls the selection, with explicit `imageN`
connections used solely to override individual positions.

### Priority order

When more than one mode contributes:

1. `source_frames` list runs first.
2. Manual slots (`image1` … `image4` with their `imageN_frame`) run
   after, so explicit `imageN` inputs always **override** list entries
   at the same cut-output position.
3. The set of placed keyframes is deduped automatically (`selected_*`
   outputs reflect the final, deduped state).

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

Console summary: `cut window: frames 41..121 (81 frames). 6 placed; 2 outside cut: 1, 27.`

This replaces the workflow pattern of feeding the Easy Image Batch
output through an external `ImageFromBatch (batch_index=40, length=81)`
node — the cut is built in.

## Tips

- **Round-tripping selections.** Wire `selected_frames` of one node
  into `source_frames` of another to chain or fan out the same picks
  across workflows.
- **Picking from a video without writing the list manually.** If you
  already have helper nodes producing single frames, plug them into
  `image1`…`image4` (Mode 1) and leave `source_frames` empty. That's
  the legacy 4-keyframe pattern, kept fully working.
- **Going past 4 keyframes.** Switch to Mode 3: connect `source_batch`,
  fill `source_frames`, leave `imageN` unconnected. Use `image1`…
  `image4` only for *overrides* (e.g. patching one frame manually).
- **Cutting an output.** Set `cut_start_frame` to the first VFX frame
  you want and `total_frames` to the length of the cut. No external
  slicer node needed.
- **Inverted alpha for compositing.** Toggle `invert_alpha` on when you
  want to use the alpha output as a true compositing alpha (selected =
  white = visible).
