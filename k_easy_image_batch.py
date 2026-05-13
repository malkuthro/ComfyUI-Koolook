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
    Build an IMAGE batch of `total_frames` length from up to 4 keyframes,
    filling undefined frames with a chosen placeholder color (Black / 50% Gray /
    White). Returns four outputs:

    - `image_batch`           — full timeline IMAGE batch with placeholders.
    - `alpha_batch`           — MASK at full timeline length. Default
        convention: selected = 0.0 (black), empty = 1.0 (white). Toggle
        `invert_alpha` flips to compositing-style (selected = 1.0).
    - `selected_image_batch`  — IMAGE containing only the keyframes that
        actually landed, packed back-to-back in ascending timeline order
        (no placeholders). Empty (length 0) if nothing was placed.
    - `selected_frames`       — STRING, comma-separated VFX frame numbers
        of the placed keyframes (e.g. "1, 27, 41, 63"). Same format as
        the `source_frames` input → round-trippable into another
        `easy_ImageBatch` instance.

    Three input modes, freely mixable:

    1. Per-slot images (up to 4): connect `imageN` and set `imageN_frame`
       to its VFX timeline position. `start_frame` shifts the timeline
       (internal indices are 0-based).
    2. `source_batch` (optional pre-batched IMAGE, e.g. straight from Load
       Video) + the 4 slots: when a slot's `imageN` is *not* connected,
       the node pulls frame `imageN_frame - start_frame` from
       `source_batch` instead. `imageN_frame` controls *both* which source
       frame to pick *and* where to place it on the output timeline —
       collapsing the typical "Load Video → Get Images From Batch In
       Range × N" helper chain into a single node.
    3. `source_batch` + `source_frames` (optional comma/newline/whitespace-
       separated string like `"1, 27, 41, 63, 85, 120"`): for keyframe
       counts beyond 4. Each number is a VFX-numbered frame that both
       picks `source_batch[N - start_frame]` and places it at the same
       timeline position. Bad tokens warn and are skipped.

       Important: when `source_frames` is non-empty, the 4 manual slots
       contribute ONLY where `imageN` is explicitly connected — the
       `imageN_frame` defaults (0, 4, 8, 12) do NOT pick from
       `source_batch` in this mode. This way the list fully controls the
       selection, with explicit `imageN` connections used solely to
       override individual positions.

    Priority order (later wins): `source_frames` list → 4 manual slots.
    So explicit `imageN` inputs always override list entries at the same
    timeline position. At least one of `image1` or `source_batch` must
    be provided. All keyframes (and source_batch frames) must share the
    same H/W/C.

    Useful for preparing sparse control sequences for video models like
    Wan 2.2, where placeholder frames indicate no latent update.

    Example
    -------
    Inputs:
        source_batch        = Load Video (100 frames)
        source_frames       = "1, 27, 41, 63"
        start_frame         = 1
        total_frames        = 64
        placeholder_color   = "Black"
        invert_alpha        = False  (inpaint convention)

    Outputs:
        image_batch          — 64 frames; timeline indices 0/26/40/62
                               carry the picked source frames, all other
                               64-4=60 frames are black placeholder.
        alpha_batch          — 64-frame MASK; the 4 selected indices are
                               0.0 (black), the 60 placeholders are 1.0
                               (white).
        selected_image_batch — 4 frames, packed back-to-back in timeline
                               order (the picks at frames 1/27/41/63).
        selected_frames      — "1, 27, 41, 63"  (paste-able into the
                               `source_frames` input of another
                               `easy_ImageBatch` instance to reuse the
                               same selection).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_frames": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 1024,
                    "step": 1,
                    "display": "number"
                }),
                "start_frame": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "display": "number"
                }),
                "placeholder_color": (list(_PLACEHOLDER_FILL.keys()), {
                    "default": "Black"
                }),
                "invert_alpha": ("BOOLEAN", {
                    "default": False,
                    "label_on": "compositing",
                    "label_off": "inpaint",
                }),
                "source_frames": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "extra frames from source_batch, e.g. 1, 27, 41, 63, 85 (commas and/or newlines)",
                }),
                "image1_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                "source_batch": ("IMAGE", ),
                "image1": ("IMAGE", ),
                "image2": ("IMAGE", ),
                "image2_frame": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "display": "number"
                }),
                "image3": ("IMAGE", ),
                "image3_frame": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "display": "number"
                }),
                "image4": ("IMAGE", ),
                "image4_frame": ("INT", {
                    "default": 12,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "display": "number"
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
        start_frame,
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

        source_len = source_batch.shape[0] if source_batch is not None else 0

        # 1) source_frames list (optional, additive). Each token is a VFX-numbered
        #    frame: it picks source_batch[N - start_frame] AND places it at the
        #    same timeline position. The 4 manual slots run after this and
        #    override per-position when needed.
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
                    target_index = f - start_frame
                    if not (0 <= target_index < source_len):
                        print(
                            f"[easy_ImageBatch] source_frames: source frame index {target_index} "
                            f"(frame {f} - start {start_frame}) is out of range for "
                            f"source_batch of length {source_len}; skipping."
                        )
                        continue
                    if not (0 <= target_index < total_frames):
                        print(
                            f"[easy_ImageBatch] source_frames: frame {f} maps to timeline "
                            f"index {target_index} which is out of range "
                            f"(0..{total_frames - 1}); skipping."
                        )
                        continue
                    image_batch[target_index] = source_batch[target_index]
                    alpha_batch[target_index] = 0.0
                    placed_indices.add(target_index)

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

            target_index = slot_frame - start_frame

            if slot_img is not None:
                frame_tensor = slot_img[0]
            elif slot_source_fallback:
                # Pull the same source index as the target timeline index.
                source_index = target_index
                if not (0 <= source_index < source_len):
                    print(
                        f"[easy_ImageBatch] {slot_label}: source frame index {source_index} "
                        f"(frame {slot_frame} - start {start_frame}) is out of range for "
                        f"source_batch of length {source_len}; skipping."
                    )
                    continue
                frame_tensor = source_batch[source_index]
            else:
                # Either source_batch isn't connected, or source_frames is
                # being used and no explicit imageN was wired — leave this
                # slot empty so it doesn't pollute the selection.
                continue

            if 0 <= target_index < total_frames:
                image_batch[target_index] = frame_tensor
                alpha_batch[target_index] = 0.0
                placed_indices.add(target_index)
            else:
                print(
                    f"[easy_ImageBatch] {slot_label}: frame {slot_frame} maps to index "
                    f"{target_index} which is out of range (0..{total_frames - 1}); "
                    "skipping this keyframe."
                )

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
        selected_frames_out = ", ".join(str(idx + start_frame) for idx in sorted_indices)

        return (image_batch, alpha_batch, selected_image_batch, selected_frames_out, )

# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "easy_ImageBatch": easy_ImageBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "easy_ImageBatch": "Easy Image Batch (Koolook)"
}