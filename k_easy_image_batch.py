from __future__ import annotations

from dataclasses import dataclass
import re

import torch


# Placeholder fill values (0..1) for unoccupied frames in the output IMAGE batch.
# Alpha output is independent of this choice; see `invert_alpha` for convention.
_PLACEHOLDER_FILL = {
    "Black": 0.0,
    "Gray": 0.5,
    "White": 1.0,
}

# Upper bound on how many frames a single ``N-M`` range token may expand to, so
# a typo like ``1-99999999`` can't balloon the frame list into millions of
# entries. Comfortably above the node's max cut window (total_frames <= 1024).
_MAX_RANGE_SPAN = 8192


@dataclass(frozen=True)
class OffsetPlan:
    """Result of planning offset-mode placements (pure; no torch).

    - ``placements``        — ``(source_index, output_index, vfx_frame)`` for
      each incoming frame that lands inside the cut window, ascending by
      position. ``source_index`` is the frame's ordinal in ``keyframes_insert``.
    - ``outside_cut``       — VFX frame numbers paired to a frame but whose
      output index fell outside ``[0, total_frames)`` (dropped, like select
      mode).
    - ``unused_source``     — count of incoming frames with no position to fill
      (more frames than listed positions).
    - ``missing_positions`` — listed positions with no incoming frame left
      (more positions than frames).
    """

    placements: list[tuple[int, int, int]]
    outside_cut: list[int]
    unused_source: int
    missing_positions: list[int]


def plan_offset_placements(
    positions: list[int],
    source_count: int,
    total_frames: int,
    cut_start_frame: int,
) -> OffsetPlan:
    """Map a packed sequence of ``source_count`` frames onto keyframe positions.

    Offset mode (the inverse of selecting): the i-th incoming frame is assigned
    to the i-th position, with positions taken ascending and de-duplicated so
    the mapping round-trips cleanly with the ascending ``selected_frames``
    output. A position is placed iff its output index ``vfx - cut_start_frame``
    falls inside ``[0, total_frames)``; otherwise it is reported in
    ``outside_cut`` and dropped (matching select-mode cut behaviour).

    Pure Python — no torch — so the placement logic is unit-testable in CI
    without the heavyweight tensor dependency.
    """
    ordered = sorted(dict.fromkeys(int(p) for p in positions))
    paired = min(source_count, len(ordered))
    placements: list[tuple[int, int, int]] = []
    outside_cut: list[int] = []
    for source_index in range(paired):
        vfx = ordered[source_index]
        output_index = vfx - cut_start_frame
        if 0 <= output_index < total_frames:
            placements.append((source_index, output_index, vfx))
        else:
            outside_cut.append(vfx)
    unused_source = max(0, source_count - len(ordered))
    missing_positions = ordered[source_count:] if source_count < len(ordered) else []
    return OffsetPlan(placements, outside_cut, unused_source, missing_positions)


def plan_source_base_fill(
    total_frames: int, cut_start_frame: int, base_len: int
) -> tuple[list[tuple[int, int]], int]:
    """Map each output index in the cut window onto a ``source_batch`` index.

    Used by insert-over-source mode to lay down the source video as the
    background before the inserts overwrite it. Output index ``i`` represents
    VFX frame ``cut_start_frame + i``, and ``source_batch[0]`` is VFX frame 1,
    so ``source_index = i + cut_start_frame - 1``.

    Returns ``(placements, fallback_count)`` where ``placements`` is the list
    of ``(output_index, source_index)`` the source actually covers and
    ``fallback_count`` is how many cut frames fell beyond the source (and stay
    placeholder). Pure Python — no torch — so it is unit-testable in CI.
    """
    placements: list[tuple[int, int]] = []
    fallback = 0
    for output_index in range(total_frames):
        source_index = output_index + cut_start_frame - 1
        if 0 <= source_index < base_len:
            placements.append((output_index, source_index))
        else:
            fallback += 1
    return placements, fallback


def plan_slot_overwrites(
    slot_frames: list[tuple[int, int]],
    total_frames: int,
    cut_start_frame: int,
) -> tuple[list[tuple[int, int, int]], list[int]]:
    """Map connected manual slots (image1-4) onto output indices — Layer 3.

    ``slot_frames`` is ``(slot_index, vfx_frame)`` for each CONNECTED slot, in
    ascending slot order (image1 = 0 … image4 = 3). Manual slots are the top
    compositing layer: they overwrite whatever the sequence or background put
    at the same output index. On a collision between two slots the
    higher-numbered slot wins (image4 > image1). A slot lands iff its output
    index ``vfx - cut_start_frame`` is inside ``[0, total_frames)``; otherwise
    the frame is reported in ``outside_cut``.

    Returns ``(placements, outside_cut)`` where ``placements`` is
    ``(slot_index, output_index, vfx)`` ascending by output index, one winner
    per index. Pure Python — no torch — so it is unit-testable in CI.
    """
    winner: dict[int, tuple[int, int]] = {}  # output_index -> (slot_index, vfx)
    outside_cut: list[int] = []
    for slot_index, vfx in slot_frames:
        output_index = vfx - cut_start_frame
        if 0 <= output_index < total_frames:
            # Higher slot wins on a collision (image4 > image1), independent of
            # the order slots are supplied in.
            prev = winner.get(output_index)
            if prev is None or slot_index > prev[0]:
                winner[output_index] = (slot_index, vfx)
        else:
            outside_cut.append(vfx)
    placements = [
        (slot_index, output_index, vfx)
        for output_index, (slot_index, vfx) in sorted(winner.items())
    ]
    return placements, outside_cut


def parse_frame_tokens(source_frames: str) -> tuple[list[int], list[str]]:
    """Parse a frame-list string into 1-based VFX frame numbers.

    Tokens are separated by commas, spaces, tabs, or newlines. A token is
    either a single integer (``7``) or an inclusive ascending range
    (``14-17`` → 14, 15, 16, 17). Returns ``(values, bad_tokens)`` where
    ``bad_tokens`` are the tokens that could not be parsed (non-integer, a
    descending range like ``5-1``, or a range wider than ``_MAX_RANGE_SPAN``)
    so the caller can warn. Pure Python — no torch — so it is unit-testable in
    CI. Shared by select picks and insert positions.
    """
    values: list[int] = []
    bad_tokens: list[str] = []
    for tok in re.split(r"[,\s]+", (source_frames or "").strip()):
        if not tok:
            continue
        span = re.fullmatch(r"(\d+)-(\d+)", tok)
        if span:
            lo, hi = int(span.group(1)), int(span.group(2))
            # Reject descending or absurdly large ranges (a typo like
            # "1-99999999" would otherwise expand into millions of entries).
            if lo <= hi <= lo + _MAX_RANGE_SPAN - 1:
                values.extend(range(lo, hi + 1))
            else:
                bad_tokens.append(tok)
            continue
        try:
            values.append(int(tok))
        except ValueError:
            bad_tokens.append(tok)
    return values, bad_tokens


class easy_ImageBatch:
    """
    Place up to N keyframes on a 1-based VFX timeline, then output a
    *cut window* of that timeline as a fixed-length IMAGE batch with
    placeholder fill (Black / 50% Gray / White) for unoccupied frames.
    Returns four outputs:

    - `image_batch`           — IMAGE batch of length `total_frames`,
        representing the cut window starting at VFX frame
        `cut_start_frame` and lasting `total_frames` frames. Selected
        positions carry the picked images; everything else is the
        placeholder.
    - `alpha_batch`           — MASK at the same length. Default
        convention: selected = 0.0 (black), empty = 1.0 (white). Toggle
        `invert_alpha` flips to compositing-style (selected = 1.0).
    - `selected_image_batch`  — IMAGE containing only the keyframes that
        actually landed *inside the cut*, packed back-to-back in
        ascending order (no placeholders). Empty if nothing was placed.
    - `selected_frames`       — STRING, comma-separated VFX frame
        numbers of the placed-inside-cut keyframes (e.g.
        `"41, 63, 78, 88, 98, 121"`). Same format as the `source_frames`
        input → round-trippable into another `easy_ImageBatch`.

    Frame-numbering conventions
    ---------------------------
    All frame numbers (`source_frames`, `imageN_frame`, `cut_start_frame`)
    are **VFX-numbered, 1-based**. Internally:

        source pick index   = vfx_frame - 1                # always
        output (cut) index  = vfx_frame - cut_start_frame  # cut window

    `source_batch[0]` is treated as VFX frame 1. There is no exposed
    knob to renumber `source_batch` — if your stack represents a
    different VFX range, renumber externally before connecting.

    Cut window
    ----------
    The output represents VFX frames `[cut_start_frame ..
    cut_start_frame + total_frames - 1]`. Frames outside this window
    are placed but silently *dropped from the cut* (a single summary
    line is logged at the end listing them). Frames inside the window
    are placed at output index `vfx - cut_start_frame`. The default
    `cut_start_frame = 1` means the cut starts at VFX frame 1, so the
    full timeline shows up — equivalent to "no cut".

    Select mode — the layers (lowest first; a later layer overwrites an
    earlier one at the same output index)
    -----------------------------------------------------------------------
    1. Background: with `source_batch` connected and an EMPTY `source_frames`
       list, the source cut window passes straight through (source in →
       source out) as kept content (`alpha` 0.0). Any tail beyond the source
       stays placeholder, and that uncovered gap becomes the `selected_*`
       output — the frames to inpaint/generate (the inverse of the usual
       "selected = picks/slots"). Otherwise the background is
       `placeholder_color` and is not counted as "placed".
    2. `source_frames` (optional comma/newline/whitespace string like
       `"1, 27, 41, 63"`, with inclusive ranges `"14-17"` → 14,15,16,17): each
       number picks `source_batch[N - 1]` and places it at output index
       `N - cut_start_frame` (if inside the cut), onto the placeholder. Bad
       tokens warn and are skipped; frames not present in `source_batch` warn;
       frames outside the cut window are dropped and summarised at the end.
    3. Manual slots (`image1`-`image4`): connect `imageN` and set
       `imageN_frame` to its VFX position; the image overwrites that output
       frame on TOP of everything. A higher-numbered slot wins a slot-vs-slot
       collision (`image4` > `image1`). An UNconnected slot does nothing —
       `imageN_frame` is only a placement target for a wired `imageN`, never a
       `source_batch` pick.

    Output dedup is automatic (set-based). All inputs must share the same
    H/W/C. When no image source is connected at all, the node emits a clean
    placeholder batch sized by the `width`/`height` widgets instead of
    erroring.

    Insert modes (the inverse) — connect a packed sequence to
    `keyframes_insert` and the node instead *scatters* those frames onto the
    `source_frames` positions (i-th frame → i-th position, ascending). The
    background depends on `source_batch`:

    - `source_batch` NOT connected → gaps filled with the placeholder
      (offset / reconstruct mode). Round-trips with `selected_image_batch` +
      `selected_frames`.
    - `source_batch` connected → the inserts are composited *over* the
      `source_batch` cut window (insert-over-source mode): the source video
      is the background and the listed positions are overwritten with the
      insert frames.

    Connected manual slots (`image1`-`image4`) still apply here — they
    composite on TOP of the inserts (a slot beats an insert at the same
    position; the higher slot wins a slot collision). With an empty
    `source_frames` list and no slots nothing is placed: a clean placeholder
    batch (no `source_batch`) or a clean `source_batch` cut-window
    passthrough. `alpha_batch` marks the inserted positions and slot
    overwrites as placed. See `create_batch_from_frames`.

    Useful for preparing sparse control sequences for video models like
    Wan 2.2, where placeholder frames indicate no latent update.

    Example A — cut_start_frame=1 (no cut, full timeline)
    -----------------------------------------------------
    Inputs:
        source_batch       = Load Video (121 frames)
        source_frames      = "1, 27, 41, 63, 78, 88, 98, 121"
        cut_start_frame    = 1
        total_frames       = 121
        placeholder_color  = "Gray"

    Outputs:
        image_batch          — 121 frames; the 8 picked frames sit at
                               their natural positions (1→idx 0,
                               27→idx 26, …, 121→idx 120). Rest is gray.
        selected_image_batch — 8 frames, packed back-to-back.
        selected_frames      — "1, 27, 41, 63, 78, 88, 98, 121"

    Example B — cut_start_frame=41, total_frames=81 (cut)
    -----------------------------------------------------
    Same inputs as Example A, but cut_start_frame=41 / total_frames=81.

    Outputs:
        image_batch          — 81 frames representing VFX 41..121.
                               Frames 41/63/78/88/98/121 land at output
                               indices 0/22/37/47/57/80. Frames 1, 27
                               are *outside the cut* and dropped.
        selected_image_batch — 6 frames (the in-cut picks).
        selected_frames      — "41, 63, 78, 88, 98, 121"
        Console summary:     — "cut window: frames 41..121 (81 frames).
                               6 placed; 2 outside cut: 1, 27."
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_frames": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": 1024,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Length of the output cut window in frames.",
                }),
                "cut_start_frame": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 999999,
                    "step": 1,
                    "display": "number",
                    "tooltip": "First VFX frame represented by output index 0. All frame numbers are 1-based.",
                }),
                "placeholder_color": (list(_PLACEHOLDER_FILL.keys()), {
                    "default": "Black",
                    "tooltip": "Fill color for empty frames in image_batch. Alpha output is controlled separately.",
                }),
                "invert_alpha": ("BOOLEAN", {
                    "default": False,
                    "label_on": "compositing",
                    "label_off": "inpaint",
                    "tooltip": "Off: selected frames are black/0 and empty frames are white/1. On: selected frames are white/1 for compositing.",
                }),
                "source_frames": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "frames from source_batch, e.g. 1, 27, 41-50, 63 (commas/newlines; ranges with -)",
                    "tooltip": "Optional 1-based VFX frame list to pick from source_batch (in insert mode, the destination positions). Separators: commas, spaces, tabs, or newlines. Inclusive ranges like 14-17 expand to 14,15,16,17.",
                }),
                "image1_frame": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 999999,
                    "step": 1,
                    "display": "number",
                    "tooltip": "1-based VFX frame number for image1; the connected image is placed here on top (no effect when the slot is unwired).",
                }),
            },
            "optional": {
                "keyframes_insert": ("IMAGE", {
                    "tooltip": (
                        "Connect a packed sequence of frames to INSERT them onto the "
                        "timeline at the positions from the frame list (source_frames): "
                        "1st frame -> 1st number, 2nd -> 2nd, ... Connected image1-4 "
                        "still composite on top of the inserts. The background depends on "
                        "source_batch: if source_batch is NOT connected, the gaps are "
                        "filled with placeholder_color (offset/reconstruct mode); if "
                        "source_batch IS connected, the inserts are composited over the "
                        "source frames in the cut window (insert-over-source mode). With "
                        "an empty frame list and no slots nothing is placed: a clean "
                        "placeholder batch (no source_batch) or a clean source-batch "
                        "passthrough (with source_batch)."
                    ),
                }),
                "source_batch": ("IMAGE", ),
                "image1": ("IMAGE", ),
                "image2": ("IMAGE", ),
                "image2_frame": ("INT", {
                    "default": 9,
                    "min": 1,
                    "max": 999999,
                    "step": 1,
                    "display": "number",
                    "tooltip": "1-based VFX frame number for image2; the connected image is placed here on top (no effect when the slot is unwired).",
                }),
                "image3": ("IMAGE", ),
                "image3_frame": ("INT", {
                    "default": 13,
                    "min": 1,
                    "max": 999999,
                    "step": 1,
                    "display": "number",
                    "tooltip": "1-based VFX frame number for image3; the connected image is placed here on top (no effect when the slot is unwired).",
                }),
                "image4": ("IMAGE", ),
                "image4_frame": ("INT", {
                    "default": 17,
                    "min": 1,
                    "max": 999999,
                    "step": 1,
                    "display": "number",
                    "tooltip": "1-based VFX frame number for image4; the connected image is placed here on top (no effect when the slot is unwired).",
                }),
                # width/height are intentionally LAST so they append to the end
                # of the widget list — inserting them earlier would shift the
                # positional widgets_values of every previously-saved workflow.
                "width": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "step": 8,
                    "display": "number",
                    "tooltip": "Fallback output width. Used ONLY when no image input provides dimensions (e.g. a fully empty node producing a clean placeholder batch). Ignored whenever an image source is connected.",
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "step": 8,
                    "display": "number",
                    "tooltip": "Fallback output height. Used ONLY when no image input provides dimensions (e.g. a fully empty node producing a clean placeholder batch). Ignored whenever an image source is connected.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING", )
    RETURN_NAMES = ("image_batch", "alpha_batch", "selected_image_batch", "selected_frames", )
    FUNCTION = "create_batch"
    CATEGORY = "Koolook/Image"
    OUTPUT_NODE = False  # Not necessarily an output node, but can be chained

    def create_batch(
        self,
        total_frames,
        cut_start_frame,
        placeholder_color,
        invert_alpha,
        source_frames,
        image1_frame,
        source_batch=None,
        image1=None,
        image2=None,
        image2_frame=None,
        image3=None,
        image3_frame=None,
        image4=None,
        image4_frame=None,
        keyframes_insert=None,
        width=512,
        height=512,
    ):
        # Validate per-slot image inputs (must be single-frame tensors, not
        # pre-batched) for BOTH modes — manual slots are now composited as the
        # top layer in insert mode too, not just select. source_batch is the
        # only accepted multi-frame input.
        for slot_name, slot_img in (
            ("image1", image1),
            ("image2", image2),
            ("image3", image3),
            ("image4", image4),
        ):
            if slot_img is not None and slot_img.shape[0] != 1:
                raise ValueError(
                    f"{slot_name} should be a single IMAGE (batch size 1), not a pre-batched "
                    "tensor. Use the 'source_batch' input for pre-batched stacks."
                )

        # Manual slots (image1-4) are the top compositing layer (Layer 3) in
        # every mode: a connected imageN overwrites output frame imageN_frame on
        # top of the sequence/background. Higher slot wins on a collision.
        slots = (
            (image1, image1_frame),
            (image2, image2_frame),
            (image3, image3_frame),
            (image4, image4_frame),
        )

        # Insert modes: when keyframes_insert is connected, switch from "select
        # frames out of source_batch" to "scatter this packed sequence onto the
        # frame-list positions", then composite the slots over the result. The
        # background depends on source_batch:
        #   - no source_batch  -> placeholder gaps (offset / reconstruct mode)
        #   - with source_batch -> the inserts sit over the source cut window
        #     (insert-over-source mode)
        # See create_batch_from_frames.
        if keyframes_insert is not None:
            return self.create_batch_from_frames(
                keyframes_insert,
                source_frames,
                total_frames,
                cut_start_frame,
                placeholder_color,
                invert_alpha,
                base_batch=source_batch,
                slots=slots,
            )

        # Determine reference H/W/C/device/dtype from the first available source.
        # Priority: image1 → source_batch → image2 → image3 → image4. With no
        # image source connected at all, fall back to the width/height widgets
        # and emit a clean placeholder batch instead of erroring.
        reference = None
        for candidate in (image1, source_batch, image2, image3, image4):
            if candidate is not None:
                reference = candidate
                break
        if reference is None:
            if (source_frames or "").strip():
                print(
                    "[easy_ImageBatch] source_frames is set but no image source "
                    "(source_batch / image1-4) is connected; the list is ignored "
                    "and a clean placeholder batch is returned."
                )
            return self._clean_placeholder_batch(
                total_frames, height, width, 3, placeholder_color, invert_alpha
            )

        h, w, c = reference.shape[1:]
        device = reference.device
        dtype = reference.dtype

        # Cross-check that all provided inputs share H/W/C.
        for label, tensor in (
            ("image1", image1),
            ("source_batch", source_batch),
            ("image2", image2),
            ("image3", image3),
            ("image4", image4),
        ):
            if tensor is not None and tensor.shape[1:] != (h, w, c):
                raise ValueError(
                    f"{label} shape {tuple(tensor.shape[1:])} does not match reference "
                    f"({h}, {w}, {c}). All inputs must share the same H/W/C."
                )

        # Compute the frame list early — it decides the background.
        source_frames_str = (source_frames or "").strip()
        source_len = source_batch.shape[0] if source_batch is not None else 0

        # Background (Layer 1): with source_batch connected AND an empty frame
        # list, pass the source cut window straight through (the same backdrop
        # insert-over-source lays down) — "source in, source out" with nothing
        # else to do. Otherwise fill with the placeholder; a non-empty list
        # means select mode (pull the named frames onto a neutral background).
        # The background is NOT counted as "placed": only list picks and slots
        # are, so alpha / selected_* describe just those.
        fill_value = _PLACEHOLDER_FILL[placeholder_color]
        image_batch = torch.full((total_frames, h, w, c), fill_value, device=device, dtype=dtype)
        # Alpha (default convention): 1.0 (white) = empty/to-fill; set to 0.0 on
        # frames that carry real content. `invert_alpha` flips at the end.
        alpha_batch = torch.ones((total_frames, h, w), device=device, dtype=torch.float32)

        # Timeline indices with a keyframe placed (picks + slots), used to build
        # the selected_* outputs. `outside_cut` collects referenced VFX frames
        # that fell outside the window (summarised once at the end).
        placed_indices: set[int] = set()
        outside_cut: list[int] = []

        # Passthrough (empty list + source): the source cut window is the kept
        # background — covered frames carry real content (alpha 0.0), the
        # uncovered tail stays placeholder. `covered_indices` lets the selection
        # below resolve to the GAP (the to-inpaint frames) rather than to the
        # picks/slots that drive select mode.
        passthrough = source_batch is not None and not source_frames_str
        covered_indices: set[int] = set()
        if passthrough:
            base_placements, _ = plan_source_base_fill(
                total_frames, cut_start_frame, source_len
            )
            for output_index, source_index in base_placements:
                image_batch[output_index] = source_batch[source_index]
                alpha_batch[output_index] = 0.0
                covered_indices.add(output_index)

        # 1) source_frames list (optional). Each token is a VFX-numbered frame:
        #    it picks source_batch[N - 1] (VFX 1-based) AND places it at output
        #    index N - cut_start_frame, onto the placeholder. Runs only when the
        #    list is non-empty (an empty list is the passthrough case above);
        #    the manual slots run after and override per-position.
        #
        #    The field is multiline: tokens may be separated by commas, newlines,
        #    spaces, tabs, or any mix of them ("1, 27\n41 63" → [1, 27, 41, 63]).
        if source_frames_str:
            if source_batch is None:
                print(
                    "[easy_ImageBatch] source_frames is set but source_batch is not connected; "
                    "ignoring the list (it only picks frames from source_batch)."
                )
            else:
                values, bad_tokens = parse_frame_tokens(source_frames_str)
                for tok in bad_tokens:
                    print(
                        f"[easy_ImageBatch] source_frames: ignoring invalid token '{tok}'."
                    )
                for f in values:
                    source_index = f - 1
                    output_index = f - cut_start_frame
                    if not (0 <= source_index < source_len):
                        print(
                            f"[easy_ImageBatch] source_frames: VFX frame {f} doesn't exist in "
                            f"source_batch (size {source_len}, covers frames 1..{source_len}); "
                            "skipping."
                        )
                        continue
                    if not (0 <= output_index < total_frames):
                        outside_cut.append(f)
                        continue
                    image_batch[output_index] = source_batch[source_index]
                    alpha_batch[output_index] = 0.0
                    placed_indices.add(output_index)

        # 2) Manual slots (Layer 3, top priority): a connected image1-4
        #    overwrites its output frame on top of the list picks / passthrough
        #    background. Higher slot wins on a collision (image4 > image1).
        #    Unconnected slots contribute nothing — imageN_frame is only a
        #    placement target for a wired imageN, never a source_batch pick.
        self._apply_slot_overwrites(
            image_batch, alpha_batch, placed_indices, slots,
            total_frames, cut_start_frame, outside_cut, (h, w, c),
        )

        # Passthrough selection: the "selected" frames are the GAP — placeholder
        # frames with no source coverage and no slot, i.e. the ones to inpaint /
        # generate. This is the inverse of select mode (where selected = the
        # placed picks/slots); covered source and slot overwrites are kept
        # content and excluded from the selection.
        if passthrough:
            content = covered_indices | placed_indices
            placed_indices = {
                i for i in range(total_frames) if i not in content
            }

        if invert_alpha:
            alpha_batch = 1.0 - alpha_batch

        # Build the "selected only" outputs: keyframes packed back-to-back in
        # ascending timeline order, plus their VFX frame numbers as a string
        # that round-trips cleanly into the `source_frames` input.
        sorted_indices = sorted(placed_indices)
        if sorted_indices:
            selected_image_batch = image_batch[sorted_indices]
        else:
            # No keyframes placed — emit a 0-length batch with the same H/W/C.
            selected_image_batch = image_batch[:0]
        selected_frames_out = ", ".join(
            str(idx + cut_start_frame) for idx in sorted_indices
        )

        # One-line summary of the cut window. Always logged so it's easy to
        # confirm what the node produced; surfaces outside-cut drops, the
        # passthrough background, and any source-shorter-than-cut fill.
        cut_end = cut_start_frame + total_frames - 1
        notes: list[str] = []
        if outside_cut:
            unique_outside = sorted(set(outside_cut))
            notes.append(
                f"{len(unique_outside)} outside cut: "
                + ", ".join(str(f) for f in unique_outside)
            )
        if passthrough:
            # kept = everything not in the gap (covered source + any slot fills).
            notes.append(
                f"source passthrough: {total_frames - len(sorted_indices)} kept, "
                f"{len(sorted_indices)} gap frame(s) selected to inpaint"
            )
        suffix = (" " + "; ".join(notes) + ".") if notes else ""
        print(
            f"[easy_ImageBatch] cut window: frames {cut_start_frame}..{cut_end} "
            f"({total_frames} frames). {len(sorted_indices)} placed.{suffix}"
        )

        return (image_batch, alpha_batch, selected_image_batch, selected_frames_out, )

    def _clean_placeholder_batch(
        self, total_frames, h, w, c, placeholder_color, invert_alpha, device=None, dtype=None
    ):
        """Build a fully-empty batch (no placed frames) of the placeholder color.

        Used when nothing drives placement: a totally empty node (sized by the
        width/height widgets, so ``device``/``dtype`` are absent → CPU/float).
        ``image_batch`` is uniform placeholder; ``alpha_batch`` is all-empty
        (1.0, flipped by ``invert_alpha``); ``selected_*`` outputs are
        zero-length. Returns the standard 4-tuple.
        """
        fill_value = _PLACEHOLDER_FILL[placeholder_color]
        if device is None:
            image_batch = torch.full((total_frames, h, w, c), fill_value)
            alpha_batch = torch.ones((total_frames, h, w), dtype=torch.float32)
        else:
            image_batch = torch.full(
                (total_frames, h, w, c), fill_value, device=device, dtype=dtype
            )
            alpha_batch = torch.ones(
                (total_frames, h, w), device=device, dtype=torch.float32
            )
        if invert_alpha:
            alpha_batch = 1.0 - alpha_batch
        print(
            f"[easy_ImageBatch] clean batch: {total_frames} frames of "
            f"{placeholder_color} placeholder; nothing placed."
        )
        return (image_batch, alpha_batch, image_batch[:0], "")

    def _apply_slot_overwrites(
        self,
        image_batch,
        alpha_batch,
        placed_indices,
        slots,
        total_frames,
        cut_start_frame,
        outside_cut,
        ref_hwc,
    ):
        """Composite connected manual slots on top of the batch (Layer 3).

        ``slots`` is ``((image, frame), …)`` for image1..image4 in ascending
        order. Only connected slots with a frame value participate; each must
        match ``ref_hwc`` (H/W/C). Higher slot wins on a collision (see
        ``plan_slot_overwrites``). Mutates ``image_batch`` / ``alpha_batch`` /
        ``placed_indices`` and extends ``outside_cut`` in place.
        """
        h, w, c = ref_hwc
        connected: list[tuple[int, int]] = []
        for slot_index, (img, frame) in enumerate(slots):
            if img is None or frame is None:
                continue
            if tuple(img.shape[1:]) != (h, w, c):
                raise ValueError(
                    f"image{slot_index + 1} shape {tuple(img.shape[1:])} does not match "
                    f"reference ({h}, {w}, {c}). All inputs must share the same H/W/C."
                )
            connected.append((slot_index, frame))
        placements, slot_outside = plan_slot_overwrites(
            connected, total_frames, cut_start_frame
        )
        for slot_index, output_index, _vfx in placements:
            image_batch[output_index] = slots[slot_index][0][0]
            alpha_batch[output_index] = 0.0
            placed_indices.add(output_index)
        outside_cut.extend(slot_outside)

    def create_batch_from_frames(
        self,
        keyframes_insert,
        source_frames,
        total_frames,
        cut_start_frame,
        placeholder_color,
        invert_alpha,
        base_batch=None,
        slots=((None, None), (None, None), (None, None), (None, None)),
    ):
        """Insert mode — active when ``keyframes_insert`` is connected.

        Scatter the packed ``keyframes_insert`` sequence onto the timeline at
        the positions listed in ``source_frames`` (i-th frame → i-th position,
        ascending). The background depends on ``base_batch`` (the connected
        ``source_batch``, or ``None``):

        - **No ``base_batch`` (offset / reconstruct mode).** Gaps are filled
          with ``placeholder_color``. Feed back a processed copy of a previous
          ``selected_image_batch`` plus the same ``selected_frames`` list to
          rebuild a full-length sequence in the original spacing.
        - **With ``base_batch`` (insert-over-source mode).** The cut window of
          ``base_batch`` is laid down first (VFX frame ``cut_start_frame + i``
          → ``base_batch[cut_start_frame + i - 1]``; frames beyond the source
          fall back to the placeholder), then the inserts overwrite at the
          listed positions. The inserts are composited *over* the source video.

        Connected manual slots (``image1``-``image4``) are then composited on
        top as the highest layer — a slot overwrites an insert at the same
        position, and the higher-numbered slot wins a slot-vs-slot collision.

        ``alpha_batch`` marks the inserted positions and any slot overwrites as
        placed (0.0), regardless of the background — so ``selected_image_batch``
        / ``selected_frames`` describe just those. With an empty frame list and
        no slots nothing is placed: a clean placeholder batch (no ``base_batch``)
        or a clean ``base_batch`` cut-window passthrough.
        """
        h, w, c = keyframes_insert.shape[1:]
        device = keyframes_insert.device
        dtype = keyframes_insert.dtype

        # Background: source cut window (insert-over-source) or placeholder.
        fill_value = _PLACEHOLDER_FILL[placeholder_color]
        image_batch = torch.full((total_frames, h, w, c), fill_value, device=device, dtype=dtype)
        base_short_fallback = 0
        if base_batch is not None:
            if tuple(base_batch.shape[1:]) != (h, w, c):
                raise ValueError(
                    f"source_batch shape {tuple(base_batch.shape[1:])} does not match "
                    f"keyframes_insert ({h}, {w}, {c}). Both must share the same H/W/C."
                )
            base_placements, base_short_fallback = plan_source_base_fill(
                total_frames, cut_start_frame, base_batch.shape[0]
            )
            for output_index, source_index in base_placements:
                image_batch[output_index] = base_batch[source_index]
        # Convention: 1.0 (white/empty), 0.0 on inserted positions only.
        alpha_batch = torch.ones((total_frames, h, w), device=device, dtype=torch.float32)

        # Parse the position list (commas, spaces, tabs, newlines, or any mix;
        # inclusive ranges like "5-7" expand via parse_frame_tokens).
        source_frames_str = (source_frames or "").strip()
        positions, bad_tokens = parse_frame_tokens(source_frames_str)
        for tok in bad_tokens:
            print(f"[easy_ImageBatch] source_frames: ignoring invalid token '{tok}'.")

        source_count = keyframes_insert.shape[0]
        plan = plan_offset_placements(positions, source_count, total_frames, cut_start_frame)

        placed_indices: set[int] = set()
        for source_index, output_index, _vfx in plan.placements:
            image_batch[output_index] = keyframes_insert[source_index]
            alpha_batch[output_index] = 0.0
            placed_indices.add(output_index)

        # Manual slots (Layer 3): composite connected image1-4 on top of the
        # inserts and the source background. A slot beats an insert at the same
        # position; higher slot wins a slot-vs-slot collision.
        slot_outside: list[int] = []
        self._apply_slot_overwrites(
            image_batch, alpha_batch, placed_indices, slots,
            total_frames, cut_start_frame, slot_outside, (h, w, c),
        )

        if invert_alpha:
            alpha_batch = 1.0 - alpha_batch

        sorted_indices = sorted(placed_indices)
        if sorted_indices:
            selected_image_batch = image_batch[sorted_indices]
        else:
            selected_image_batch = image_batch[:0]
        selected_frames_out = ", ".join(
            str(idx + cut_start_frame) for idx in sorted_indices
        )

        # One-line summary (mirrors select mode) with insert-specific notes so
        # silent surprises (extra frames, short lists, outside-cut, empty list,
        # source shorter than the cut) are visible without per-frame logs.
        mode_label = "insert-over-source mode" if base_batch is not None else "offset mode"
        cut_end = cut_start_frame + total_frames - 1
        notes: list[str] = []
        if plan.outside_cut:
            notes.append(
                f"{len(plan.outside_cut)} outside cut: "
                + ", ".join(str(f) for f in plan.outside_cut)
            )
        if plan.unused_source:
            notes.append(
                f"{plan.unused_source} extra insert frame(s) had no listed position"
            )
        if plan.missing_positions:
            notes.append(
                f"{len(plan.missing_positions)} position(s) had no insert frame: "
                + ", ".join(str(f) for f in plan.missing_positions)
            )
        if base_short_fallback:
            notes.append(
                f"{base_short_fallback} cut frame(s) beyond source_batch filled with placeholder"
            )
        if slot_outside:
            notes.append(
                f"{len(set(slot_outside))} slot frame(s) outside cut: "
                + ", ".join(str(f) for f in sorted(set(slot_outside)))
            )
        if not source_frames_str:
            notes.append("frame list is empty — nothing inserted")
        suffix = (" " + "; ".join(notes) + ".") if notes else ""
        print(
            f"[easy_ImageBatch] {mode_label}: cut window frames {cut_start_frame}.."
            f"{cut_end} ({total_frames} frames). {len(sorted_indices)} placed.{suffix}"
        )

        return (image_batch, alpha_batch, selected_image_batch, selected_frames_out, )

# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "easy_ImageBatch": easy_ImageBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "easy_ImageBatch": "Easy Image Batch (Koolook)"
}
