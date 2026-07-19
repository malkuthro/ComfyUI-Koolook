# SPDX-License-Identifier: GPL-3.0-or-later
#
# ComfyUI-Koolook — LTX Keyframe Soften Schedule node.
# Copyright (C) 2026 ComfyUI-Koolook contributors (kforgelabs).
"""LTX Keyframe Soften Schedule (Koolook) — smooth motion early, sharp poses late.

The LTX Director pins each keyframe via the latent's denoise mask. Held hard from
step 0, the model must hit exact poses immediately, so a transition between two
poses snaps as a CUT. This node eases those pins in the HIGH-SIGMA (early) part of
denoise — where coarse motion forms — so the model glides between poses, then
restores them so the poses sharpen. It adds no frames; only the keyframes you have.

Two simple controls:
  * ``max_soften`` — ease strength: the fraction the pins unpin at the peak
    (0 = off, ~0.1 = gentle, 1 = max). Direct, schedule-independent.
  * ``soften_range`` — ease the TOP ``soften_range`` of the sigma range
    (sigma-progress ``1 - sigma/sigma_start``). 0.03 = top 3% of the range. This
    is a LEVEL on the noise curve, not a step count; the number of steps it spans
    depends on the schedule, so wire ``sigmas`` to see the exact steps on ``info``.

Audio-safe: the A/V mask is a ``NestedTensor((video, audio))`` — only the video
pins are softened; the audio mask passes through untouched (lip-sync unaffected).

NOTE: ``soften_gain`` and ``steps_in_range`` are unit-tested; the model wiring is
validated by rendering against a live LTX 2.3 model.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def soften_gain(sigma, sigma_start, soften_range, max_soften):
    """Ease strength at the current ``sigma``.

    Progress = ``1 - sigma/sigma_start`` (0 at the schedule start, growing as sigma
    falls). Returns ``max_soften`` at progress 0 and smoothsteps to 0 by
    ``soften_range`` progress (the top ``soften_range`` of the range), then 0.
    A gain ``g`` raises a pinned mask value ``m`` toward 1 via ``m + (1-m)*g``.
    """
    sigma_start = float(sigma_start)
    if sigma_start <= 0.0:
        return 0.0
    progress = 1.0 - (float(sigma) / sigma_start)
    progress = 0.0 if progress < 0.0 else (1.0 if progress > 1.0 else progress)
    soften_range = float(soften_range)
    if soften_range <= 0.0 or progress >= soften_range:
        return 0.0
    t = progress / soften_range            # 0 at start -> 1 at the edge of the range
    s = t * t * (3.0 - 2.0 * t)            # smoothstep
    return float(max_soften) * (1.0 - s)   # max_soften at start -> 0 at the edge


def soften_mask_value(mask_value, gain):
    """Raise a denoise-mask value toward 1 (unpinned) by ``gain``."""
    return mask_value + (1.0 - mask_value) * float(gain)


def soften_denoise_mask(denoise_mask, gain, nested_ctor=None):
    """Soften a denoise mask, sparing the audio half of an A/V mask.

    The A/V mask is a ``NestedTensor((video, audio))`` (``LTXVConcatAVLatent`` —
    video FIRST). Soften ONLY part 0 (video pins); pass part 1 (audio) through
    untouched. A plain (video-only) mask is softened whole. ``nested_ctor`` is
    injectable for tests; ``gain<=0`` / ``None`` is a no-op.
    """
    if gain <= 0.0 or denoise_mask is None:
        return denoise_mask
    if getattr(denoise_mask, "is_nested", False):
        parts = list(denoise_mask.unbind())
        parts[0] = soften_mask_value(parts[0], gain)   # video only; audio untouched
        if nested_ctor is None:
            import comfy.nested_tensor
            nested_ctor = comfy.nested_tensor.NestedTensor
        return nested_ctor(tuple(parts))
    return soften_mask_value(denoise_mask, gain)


def steps_in_range(sigmas, soften_range):
    """How many leading steps fall in the top ``soften_range`` of the sigma range.

    ``sigmas`` is the sampler's sigma list (length n_steps+1; last ~0). A step is
    "in range" while its sigma-progress (``1 - sigma/sigma_start``) is below
    ``soften_range``. Returns ``(count, n_steps)`` — the bridge from the level to
    actual steps for display only.
    """
    vals = [float(s) for s in sigmas]
    n = len(vals)
    if n < 2:
        return 0, max(0, n - 1)
    ref = vals[0]
    n_steps = n - 1
    if ref <= 0.0:
        return 0, n_steps
    count = 0
    for i in range(n_steps):
        if (1.0 - vals[i] / ref) < float(soften_range):
            count += 1
        else:
            break
    return count, n_steps


def _format_info(max_soften, soften_range, window):
    """Build the short ``info`` string. ``window`` is (count, n_steps) or None."""
    if max_soften <= 0.0 or soften_range <= 0.0:
        return "Soften: OFF"
    head = f"Soften: ease {max_soften:.2f}, top {soften_range * 100:.0f}% of range"
    if window is not None:
        count, n_steps = window
        return f"{head}  = first {count} of {n_steps} steps"
    return head + "\n(connect SIGMAS to see steps)"


class LTXKeyframeSoftenSchedule:
    """Ease the LTX keyframe pins in the high-sigma range: smooth motion, sharp poses."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "LTX 2.3 model."}),
                "max_soften": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Ease strength — how much to unpin the keyframes. "
                               "0 = off, ~0.1 = gentle, 1 = max.",
                }),
                "soften_range": ("FLOAT", {
                    "default": 0.03, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Ease the TOP % of the sigma range (where motion forms). "
                               "0.03 = top 3%. It's a level, not a step count — wire "
                               "SIGMAS to see which steps that is.",
                }),
            },
            "optional": {
                "sigmas": ("SIGMAS", {
                    "tooltip": "Optional — connect your sampler's SIGMAS to show the "
                               "exact steps on the 'info' output.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "patch"
    CATEGORY = "Koolook/LTX"
    DESCRIPTION = (
        "Eases the keyframe pins in the high-sigma (early) part of denoise so motion "
        "between poses is smooth, then restores them so poses sharpen. Adds no frames; "
        "lip-sync safe. `info` shows the eased steps (wire SIGMAS)."
    )

    def patch(self, model, max_soften, soften_range, sigmas=None):
        m = model.clone()
        max_soften = float(max_soften)
        soften_range = float(soften_range)

        # info string (display only)
        window = None
        if sigmas is not None:
            try:
                window = steps_in_range(sigmas, soften_range)
            except Exception as e:
                log.warning("[LTXKeyframeSoftenSchedule] could not read preview sigmas: %s", e)
        info = _format_info(max_soften, soften_range, window)

        if max_soften <= 0.0 or soften_range <= 0.0:
            log.info("[LTXKeyframeSoftenSchedule] off (max_soften/soften_range = 0).")
            return (m, info)

        prev_fn = m.model_options.get("denoise_mask_function")
        # Fallback anchor from the connected preview SIGMAS (if wired). Used only
        # when a run omits extra_options["sigmas"] — gives the wired input a real
        # runtime role, not just the info readout.
        try:
            connected_ref = float(sigmas[0]) if sigmas is not None else None
        except Exception:
            connected_ref = None
        logged = [False]
        warned = [False]

        def denoise_mask_function(sigma, denoise_mask, extra_options=None):
            if prev_fn is not None:
                denoise_mask = prev_fn(sigma, denoise_mask, extra_options=extra_options)
            if denoise_mask is None:
                return denoise_mask

            # Anchor = the schedule's START sigma, read FRESH each step from the
            # runtime sigmas. It's constant within a run (no per-step drift) but
            # re-derives itself if this patched clone is later run under a
            # different sampler schedule — so no stale cached anchor, and a run
            # that momentarily lacks runtime sigmas can never latch the node
            # permanently off (the old cache-0.0 path). Falls back to the wired
            # SIGMAS when the runtime doesn't supply them.
            rt = (extra_options or {}).get("sigmas")
            ref = None
            if rt is not None:
                try:
                    ref = float(rt[0])
                except Exception:
                    ref = None
            if ref is None:
                ref = connected_ref
            if not ref or ref <= 0.0:
                # No usable reference sigma from the run and none wired: the node
                # is patched but inert this run. Warn ONCE so a silent no-op on a
                # sampler path that doesn't pass sigmas doesn't look like a bug.
                if not warned[0]:
                    warned[0] = True
                    log.warning(
                        "[LTXKeyframeSoftenSchedule] no usable sigmas from the run "
                        "and none wired to the 'sigmas' input — soften inactive this "
                        "run. Connect your sampler's SIGMAS to enable it.",
                    )
                return denoise_mask

            try:
                s = float(sigma.max()) if hasattr(sigma, "max") else float(sigma)
            except Exception:
                s = ref
            gain = soften_gain(s, ref, soften_range, max_soften)
            if gain <= 0.0:
                return denoise_mask
            if not logged[0]:
                logged[0] = True
                nested = getattr(denoise_mask, "is_nested", False)
                try:
                    count, n_steps = steps_in_range(
                        rt if rt is not None else [ref, 0.0], soften_range
                    )
                except Exception:
                    count, n_steps = 0, 0
                log.info(
                    "[LTXKeyframeSoftenSchedule] active (%s)%s — ease %.2f, top %.0f%% "
                    "of range = first %d of %d steps",
                    "A/V" if nested else "video-only",
                    " — audio untouched" if nested else "",
                    max_soften, soften_range * 100, count, n_steps,
                )
            return soften_denoise_mask(denoise_mask, gain)

        m.set_model_denoise_mask_function(denoise_mask_function)
        log.info(
            "[LTXKeyframeSoftenSchedule] patched: ease=%.2f top_range=%.2f",
            max_soften, soften_range,
        )
        return (m, info)


NODE_CLASS_MAPPINGS = {"LTXKeyframeSoftenSchedule": LTXKeyframeSoftenSchedule}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXKeyframeSoftenSchedule": "LTX Keyframe Soften Schedule (Koolook)"}

__all__ = [
    "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
    "soften_gain", "soften_mask_value", "soften_denoise_mask", "steps_in_range",
]
