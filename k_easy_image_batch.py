import re

import torch


# Placeholder fill values (0..1) for unoccupied frames in the output IMAGE batch.
# Alpha output is independent of this choice; see `invert_alpha` for convention.
_PLACEHOLDER_FILL = {
    "Black": 0.0,
    "Gray": 0.5,
    "White": 1.0,
}


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

    Three input modes, freely mixable
    ---------------------------------
    1. Per-slot images (up to 4): connect `imageN` and set `imageN_frame`
       to its VFX timeline position. The image lands at output index
       `imageN_frame - cut_start_frame` if that's inside the cut.
    2. `source_batch` (optional pre-batched IMAGE, e.g. straight from
       Load Video) + the 4 slots: when a slot's `imageN` is *not*
       connected, the node pulls VFX frame `imageN_frame` from
       `source_batch` (i.e. `source_batch[imageN_frame - 1]`) and
       places it at output index `imageN_frame - cut_start_frame`.
       Collapses the "Load Video → Get Images From Batch In Range × N"
       helper chain into a single node.
    3. `source_batch` + `source_frames` (optional comma/newline/whitespace
       separated string like `"1, 27, 41, 63, 85, 120"`): for keyframe
       counts beyond 4. Each number picks
       `source_batch[N - 1]` and places it at output index
       `N - cut_start_frame` (if inside the cut). Bad tokens warn and
       are skipped; frames not present in `source_batch` warn; frames
       outside the cut window are dropped silently and summarised at
       the end.

       Important: when `source_frames` is non-empty, the 4 manual slots
       contribute ONLY where `imageN` is explicitly connected — the
       `imageN_frame` defaults do NOT pick from `source_batch` in this
       mode. The list fully controls the selection, with explicit
       `imageN` connections used solely to override individual positions.

    Priority order (later wins): `source_frames` list → 4 manual slots.
    Explicit `imageN` inputs always override list entries at the same
    cut-output position. Output dedup is automatic (set-based). At least
    one of `image1` or `source_batch` must be provided. All keyframes
    (and `source_batch` frames) must share the same H/W/C.

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
                    "placeholder": "extra frames from source_batch, e.g. 1, 27, 41, 63, 85 (commas and/or newlines)",
                    "tooltip": "Optional 1-based VFX frame list to pick from source_batch. Separators can be commas, spaces, tabs, or newlines.",
                }),
                "image1_frame": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 999999,
                    "step": 1,
                    "display": "number",
                    "tooltip": "1-based VFX frame number for image1, or source_batch pick when source_frames is empty.",
                }),
            },
            "optional": {
                "source_batch": ("IMAGE", ),
                "image1": ("IMAGE", ),
                "image2": ("IMAGE", ),
                "image2_frame": ("INT", {
                    "default": 9,
                    "min": 1,
                    "max": 999999,
                    "step": 1,
                    "display": "number",
                    "tooltip": "1-based VFX frame number for image2, or source_batch pick when source_frames is empty.",
                }),
                "image3": ("IMAGE", ),
                "image3_frame": ("INT", {
                    "default": 13,
                    "min": 1,
                    "max": 999999,
                    "step": 1,
                    "display": "number",
                    "tooltip": "1-based VFX frame number for image3, or source_batch pick when source_frames is empty.",
                }),
                "image4": ("IMAGE", ),
                "image4_frame": ("INT", {
                    "default": 17,
                    "min": 1,
                    "max": 999999,
                    "step": 1,
                    "display": "number",
                    "tooltip": "1-based VFX frame number for image4, or source_batch pick when source_frames is empty.",
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
    ):
        # Validate per-slot image inputs (must be single-frame tensors,
        # not pre-batched). source_batch is the only accepted multi-frame input.
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

        # Determine reference H/W/C/device/dtype from the first available source.
        # Priority: image1 → source_batch → image2 → image3 → image4.
        reference = None
        for candidate in (image1, source_batch, image2, image3, image4):
            if candidate is not None:
                reference = candidate
                break
        if reference is None:
            raise ValueError(
                "easy_ImageBatch needs at least one of: image1, source_batch, image2, image3, image4."
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

        fill_value = _PLACEHOLDER_FILL[placeholder_color]
        image_batch = torch.full((total_frames, h, w, c), fill_value, device=device, dtype=dtype)
        # Alpha (default convention): 1.0 (white) = empty, set to 0.0 on occupied frames below.
        # `invert_alpha` flips the result to compositing-style at the end.
        alpha_batch = torch.ones((total_frames, h, w), device=device, dtype=torch.float32)

        # Track timeline indices that actually got a keyframe placed. Used at the
        # end to build `selected_image_batch` (just the placed frames, packed)
        # and `selected_frames` (their VFX numbers as a comma-separated string).
        placed_indices: set[int] = set()
        # VFX frame numbers that were referenced but landed outside the cut
        # window. Collected once and printed as a single summary line at the
        # end so each frame doesn't generate its own console warning.
        outside_cut: list[int] = []

        source_len = source_batch.shape[0] if source_batch is not None else 0

        # 1) source_frames list (optional, additive). Each token is a VFX-numbered
        #    frame: it picks source_batch[N - 1] (VFX 1-based, hardcoded) AND
        #    places it at output index N - cut_start_frame. The 4 manual slots
        #    run after this and override per-position when needed.
        #
        #    The field is multiline: tokens may be separated by commas, newlines,
        #    spaces, tabs, or any mix of them ("1, 27\n41 63" → [1, 27, 41, 63]).
        source_frames_str = (source_frames or "").strip()
        if source_frames_str:
            if source_batch is None:
                print(
                    "[easy_ImageBatch] source_frames is set but source_batch is not connected; "
                    "ignoring the list (it only picks frames from source_batch)."
                )
            else:
                tokens = [t for t in re.split(r"[,\s]+", source_frames_str) if t]
                for tok in tokens:
                    try:
                        f = int(tok)
                    except ValueError:
                        print(
                            f"[easy_ImageBatch] source_frames: ignoring non-integer token '{tok}'."
                        )
                        continue
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

        # 2) Resolve each manual slot to (frame_image, timeline_frame). A slot
        #    contributes iff (a) its imageN is connected, or (b) source_batch
        #    is connected (in which case imageN_frame doubles as the source
        #    pick index). Manual slots run AFTER the source_frames list so an
        #    explicit imageN always wins at its target position.
        slots = (
            (image1, image1_frame, "image1"),
            (image2, image2_frame, "image2"),
            (image3, image3_frame, "image3"),
            (image4, image4_frame, "image4"),
        )

        # When the user is driving the node from `source_frames`, the 4 manual
        # `imageN_frame` defaults (0, 4, 8, 12) shouldn't sneak in extra picks
        # from source_batch — only the explicit list (and any explicitly
        # connected imageN inputs) should contribute. So the source_batch
        # fallback for an UN-connected slot is only active when the list is
        # empty (the original 4-keyframe workflow).
        slot_source_fallback = source_batch is not None and not source_frames_str

        for slot_img, slot_frame, slot_label in slots:
            if slot_frame is None:
                continue  # Optional slot with no frame value — skip.

            output_index = slot_frame - cut_start_frame

            if slot_img is not None:
                frame_tensor = slot_img[0]
            elif slot_source_fallback:
                # Pull VFX frame `slot_frame` from source_batch (1-based).
                source_index = slot_frame - 1
                if not (0 <= source_index < source_len):
                    print(
                        f"[easy_ImageBatch] {slot_label}: VFX frame {slot_frame} doesn't "
                        f"exist in source_batch (size {source_len}, covers frames "
                        f"1..{source_len}); skipping."
                    )
                    continue
                frame_tensor = source_batch[source_index]
            else:
                # Either source_batch isn't connected, or source_frames is
                # being used and no explicit imageN was wired — leave this
                # slot empty so it doesn't pollute the selection.
                continue

            if 0 <= output_index < total_frames:
                image_batch[output_index] = frame_tensor
                alpha_batch[output_index] = 0.0
                placed_indices.add(output_index)
            else:
                outside_cut.append(slot_frame)

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
        # confirm what the node produced; mentions outside-cut drops when any.
        cut_end = cut_start_frame + total_frames - 1
        if outside_cut:
            unique_outside = sorted(set(outside_cut))
            outside_str = ", ".join(str(f) for f in unique_outside)
            print(
                f"[easy_ImageBatch] cut window: frames {cut_start_frame}..{cut_end} "
                f"({total_frames} frames). {len(sorted_indices)} placed; "
                f"{len(unique_outside)} outside cut: {outside_str}."
            )
        else:
            print(
                f"[easy_ImageBatch] cut window: frames {cut_start_frame}..{cut_end} "
                f"({total_frames} frames). {len(sorted_indices)} placed."
            )

        return (image_batch, alpha_batch, selected_image_batch, selected_frames_out, )

# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "easy_ImageBatch": easy_ImageBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "easy_ImageBatch": "Easy Image Batch (Koolook)"
}
