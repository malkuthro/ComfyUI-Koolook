# Easy Image Batch (Koolook)

| Aspect | Value |
|---|---|
| Display name | `Easy Image Batch (Koolook)` |
| Node ID | `easy_ImageBatch` |
| ComfyUI category | `Koolook/Image` |
| Source | Koolook-native |
| Source file | [`k_easy_image_batch.py`](../../../../k_easy_image_batch.py) |

## What it does

Builds a fixed-length `IMAGE` batch (`total_frames` long) where a small set
of *keyframes* sit at chosen timeline positions and every other frame is a
solid placeholder colour (Black / 50 % Gray / White). Pairs the image
batch with an `alpha_batch` mask marking selected vs empty frames, and
also returns just the keyframes packed back-to-back as a separate
short-batch output.

Designed for sparse control sequences — e.g. feeding a video model like
Wan 2.2 a 64-frame guide where only frames 1, 27, 41 and 63 carry real
image data and the rest tell the model "no latent update here".

## Inputs

### Required widgets

| Input | Type | Default | Description |
|---|---|---|---|
| `total_frames` | `INT` | `16` | Length of the output timeline. Range: `1..1024`. |
| `start_frame` | `INT` | `0` | VFX-style frame offset. `imageN_frame` and `source_frames` values are interpreted as VFX frame numbers; the internal index is `frame - start_frame` (0-based). |
| `placeholder_color` | enum | `Black` | Fill colour for unoccupied frames in `image_batch`. `Black = 0.0`, `Gray = 0.5`, `White = 1.0`. Independent of `alpha_batch`. |
| `invert_alpha` | `BOOLEAN` | `false` (`inpaint`) | Off = inpaint convention (selected = `0.0` black, empty = `1.0` white). On = compositing alpha (selected = `1.0`, empty = `0.0`). |
| `source_frames` | multiline `STRING` | `""` | Optional list of VFX frame numbers to pick from `source_batch`. Tokens may be separated by commas, newlines, spaces, tabs, or any mix (`"1, 27\n41 63"` → `[1, 27, 41, 63]`). Each token both picks `source_batch[N - start_frame]` and places it at the same timeline position. Bad tokens warn and skip. |
| `image1_frame` … `image4_frame` | `INT` | `0`, `4`, `8`, `12` | VFX frame number where the slot's keyframe lands. Doubles as the source-pick index when `imageN` is unconnected (see [Modes](#modes) below). |

### Optional input slots

| Input | Type | Description |
|---|---|---|
| `source_batch` | `IMAGE` | Pre-batched image stack, e.g. straight from `Load Video`. Used as a frame source for `source_frames` and (in mode 2) for unconnected `imageN` slots. |
| `image1` … `image4` | `IMAGE` | Per-slot keyframes. Each must be a single-frame `IMAGE` (batch size 1). Pre-batched stacks must come through `source_batch` instead. |

At least one of `image1` or `source_batch` must be connected; otherwise
the node raises a clear `ValueError`. All connected inputs must share
the same `H × W × C`.

## Outputs

| Output | Type | Description |
|---|---|---|
| `image_batch` | `IMAGE` | Full-length timeline (`total_frames` frames). Selected positions carry the picked images; everything else is the chosen placeholder colour. |
| `alpha_batch` | `MASK` | Same length as `image_batch`. Default convention: selected = `0.0`, empty = `1.0`. Toggle `invert_alpha` to flip. |
| `selected_image_batch` | `IMAGE` | **Only** the keyframes that actually landed, packed back-to-back in ascending timeline order. Length = number of placed keyframes; empty (length 0) if nothing was placed. |
| `selected_frames` | `STRING` | Comma-separated VFX frame numbers of the placed keyframes, sorted ascending and deduped (e.g. `"1, 27, 41, 63"`). Same format as the `source_frames` input — paste it (or wire it) into another `easy_ImageBatch` to reuse the same selection. |

## Modes

The node supports three input modes that can be freely mixed.

### Mode 1 — Per-slot images (up to 4 keyframes)

Connect `imageN` and set `imageN_frame` to its VFX timeline position. No
`source_batch` needed. Up to 4 keyframes total.

### Mode 2 — `source_batch` + the 4 slots (up to 4 keyframes from a video)

Connect a pre-batched stack to `source_batch` (e.g. from `Load Video`)
and leave `imageN` unconnected. Each `imageN_frame` then doubles as
*both* the source-pick index *and* the timeline target — collapsing the
typical "Load Video → Get Images From Batch In Range × N → Easy Image
Batch" helper chain into a single node.

This mode is active when `source_frames` is empty.

### Mode 3 — `source_batch` + `source_frames` list (unlimited keyframes)

Connect `source_batch` and fill `source_frames` with as many VFX frame
numbers as you need. Removes the 4-keyframe ceiling.

**Important interaction rule:** when `source_frames` is non-empty, the
4 manual slots contribute ONLY where `imageN` is explicitly connected —
the `imageN_frame` defaults (0, 4, 8, 12) do **not** pick from
`source_batch` in this mode. The list fully controls the selection,
with explicit `imageN` connections used solely to override individual
positions.

### Priority order

When more than one mode contributes:

1. `source_frames` list runs first.
2. Manual slots (`image1` … `image4` with their `imageN_frame`) run
   after, so explicit `imageN` inputs always **override** list entries
   at the same timeline position.
3. The set of placed keyframes is deduped automatically (`selected_*`
   outputs reflect the final, deduped state).

## Conventions

- **VFX-numbered frames.** `imageN_frame` and `source_frames` are
  *VFX-numbered*. Internally the node converts to 0-based indices via
  `target_index = frame - start_frame`. Out-of-range values warn and
  skip.
- **Alpha vs mask vocabulary.** `alpha_batch` is a `MASK`-typed output
  but follows whichever convention `invert_alpha` selects. The default
  (`inpaint`) matches the previous `mask_batch` semantics for backward
  compatibility (selected = black, empty = white).

## Example

Inputs:

| Field | Value |
|---|---|
| `source_batch` | `Load Video` (100 frames) |
| `source_frames` | `"1, 27, 41, 63"` |
| `start_frame` | `1` |
| `total_frames` | `64` |
| `placeholder_color` | `Black` |
| `invert_alpha` | `false` (inpaint) |

Outputs:

| Output | What you get |
|---|---|
| `image_batch` | 64 frames; timeline indices 0/26/40/62 carry the picked source frames, all other 60 frames are black placeholder. |
| `alpha_batch` | 64-frame mask; the 4 selected indices are `0.0` (black), the 60 placeholders are `1.0` (white). |
| `selected_image_batch` | 4 frames, packed back-to-back in timeline order (the picks at frames 1 / 27 / 41 / 63). |
| `selected_frames` | `"1, 27, 41, 63"` — paste-able into another `easy_ImageBatch.source_frames` to reuse the same selection. |

## Tips

- **Round-tripping selections.** Wire `selected_frames` of one node into
  `source_frames` of another to chain or fan out the same picks across
  workflows.
- **Picking from a video without writing the list manually.** If you
  already have helper nodes producing single frames, plug them into
  `image1`…`image4` (Mode 1) and leave `source_frames` empty. That's the
  legacy 4-keyframe pattern, kept fully working.
- **Going past 4 keyframes.** Switch to Mode 3: connect `source_batch`,
  fill `source_frames`, leave `imageN` unconnected. Use `image1`…`image4`
  only for *overrides* (e.g. patching one frame manually).
- **Inverted alpha for compositing.** Toggle `invert_alpha` on when you
  want to use the alpha output as a true compositing alpha (selected =
  white = visible).
