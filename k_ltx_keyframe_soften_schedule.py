# SPDX-License-Identifier: GPL-3.0-or-later
#
# ComfyUI-Koolook — LTX Keyframe Soften Schedule node.
# Copyright (C) 2026 ComfyUI-Koolook contributors (kforgelabs).
"""LTX Keyframe Soften Schedule (Koolook) — smooth motion early, sharp poses late.

The LTX Director pins each keyframe by setting that latent frame's denoise mask
to ``1 - strength`` (a value < 1). The sampler applies it EVERY step
(``comfy/samplers.py``)::

    if "denoise_mask_function" in model_options:
        denoise_mask = model_options["denoise_mask_function"](sigma, denoise_mask, ...)
    x = x * denoise_mask + guide * (1 - denoise_mask)   # the pin

Hard pins held from step 0 force the model to hit exact poses immediately, so a
transition between two different poses resolves as a CUT. This node schedules the
pin by sigma instead: it **softens** the keyframe pins early (high sigma), when
the model is deciding coarse motion — letting it form a smooth trajectory between
the poses instead of snapping — then **restores** them by a crossover point (~step
9 of 16) so the poses sharpen for the back half of denoise.

It adds NOTHING (no extra guide frames / poses — that approach backfires; every
guide is a pose target). It only loosens/tightens the keyframes you already have.

Audio-safe: the A/V latent mask is a ``NestedTensor((video_mask, audio_mask))``
(``LTXVConcatAVLatent``); we soften ONLY the video mask and pass the audio mask
through untouched, so lip-sync timing is never disturbed.

NOTE: the pure ``soften_gain`` ramp is unit-tested; the model wiring (the
denoise-mask function) is validated by rendering against a live LTX 2.3 model.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# The node widgets are normalized 0-1 sliders that map onto the *useful* internal
# range (the effect is very sensitive, so the raw values bunch up near 0). Widget
# value * SCALE = the internal value passed to soften_gain().
_MAX_SOFTEN_SCALE = 0.2    # widget 0..1 -> internal max_soften 0..0.2
_CROSSOVER_SCALE = 0.05    # widget 0..1 -> internal crossover 0..0.05


def soften_gain(sigma, sigma_max, crossover=0.55, max_soften=0.6):
    """How much to soften the keyframe pins at the current ``sigma``.

    Denoise progress runs 0 at the start (``sigma == sigma_max``) to 1 at the end
    (``sigma -> 0``). Returns ``max_soften`` at progress 0, smoothsteps down to 0
    by ``crossover`` progress, and stays 0 afterwards (pins fully restored).

    A returned gain ``g`` raises a pinned mask value ``m`` toward 1 (unpinned)
    via ``m + (1 - m) * g``: g=0 keeps the exact pin, g=1 fully frees it.
    """
    sigma_max = float(sigma_max)
    if sigma_max <= 0.0:
        return 0.0
    progress = 1.0 - (float(sigma) / sigma_max)
    progress = 0.0 if progress < 0.0 else (1.0 if progress > 1.0 else progress)
    crossover = float(crossover)
    if crossover <= 0.0 or progress >= crossover:
        return 0.0
    t = progress / crossover               # 0 at start -> 1 at crossover
    s = t * t * (3.0 - 2.0 * t)            # smoothstep 0 -> 1
    return float(max_soften) * (1.0 - s)   # max_soften at start -> 0 at crossover


def soften_mask_value(mask_value, gain):
    """Raise a single denoise-mask value toward 1 (unpinned) by ``gain``."""
    return mask_value + (1.0 - mask_value) * float(gain)


class LTXKeyframeSoftenSchedule:
    """Sigma-schedule the LTX keyframe pins: soft early (smooth motion), sharp late."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "LTX 2.3 model (place after the other model patchers)."}),
                "max_soften": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "EASE amount (normalized 0-1; maps to internal 0-0.20). "
                               "How gently the keyframe pins loosen at the START of "
                               "denoise — the smooth approach into each key. 0 = stock "
                               "(no ease). Higher = smoother, but also more invented "
                               "in-between motion. Default 0.5 = internal 0.10.",
                }),
                "crossover": ("FLOAT", {
                    "default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "INVENTION window (normalized 0-1; maps to internal "
                               "0-0.05). How long the pins stay loose before re-locking. "
                               "LOWER = poses lock back sooner = FEWER invented in-between "
                               "frames (truer to your keyframes). Higher = more invented "
                               "motion (and risk of a noisy/under-resolved image). "
                               "Default 0.65 = internal ~0.033.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Koolook/LTX"
    DESCRIPTION = (
        "Smooth motion early, sharp poses late: sigma-schedule the LTX keyframe "
        "pins — loosen them at high sigma so the model forms smooth motion between "
        "poses, restore them by a crossover (~step 9) so the poses sharpen. Adds no "
        "frames; audio mask untouched (lip-sync safe)."
    )

    def patch(self, model, max_soften, crossover):
        m = model.clone()

        # Widgets are normalized 0-1; map onto the sensitive internal range.
        max_soften_w, crossover_w = float(max_soften), float(crossover)
        max_soften = max_soften_w * _MAX_SOFTEN_SCALE
        crossover = crossover_w * _CROSSOVER_SCALE

        try:
            sigma_max = float(m.get_model_object("model_sampling").sigma_max)
        except Exception as e:  # pragma: no cover - live model
            log.warning("[LTXKeyframeSoftenSchedule] no model_sampling.sigma_max: %s", e)
            sigma_max = 1.0

        if max_soften <= 0.0:
            log.info("[LTXKeyframeSoftenSchedule] max_soften=0 — model passed through.")
            return (m,)

        diag = [False]

        def denoise_mask_function(sigma, denoise_mask, extra_options=None):
            if denoise_mask is None:
                return denoise_mask
            try:
                s = float(sigma.max()) if hasattr(sigma, "max") else float(sigma)
            except Exception:
                s = sigma_max
            gain = soften_gain(s, sigma_max, crossover, max_soften)
            if gain <= 0.0:
                return denoise_mask

            # A/V latent: mask is NestedTensor((video_mask, audio_mask)). Soften
            # ONLY the video mask (parts[0]); leave audio (parts[1]) untouched.
            if getattr(denoise_mask, "is_nested", False):
                import comfy.nested_tensor
                parts = list(denoise_mask.unbind())
                parts[0] = soften_mask_value(parts[0], gain)
                if not diag[0]:
                    diag[0] = True
                    log.info(
                        "[LTXKeyframeSoftenSchedule] active (A/V): softening video "
                        "keyframe pins, audio untouched. max_soften=%.2f crossover=%.2f",
                        max_soften, crossover,
                    )
                return comfy.nested_tensor.NestedTensor(tuple(parts))

            # Video-only latent: the whole mask is video.
            if not diag[0]:
                diag[0] = True
                log.info(
                    "[LTXKeyframeSoftenSchedule] active (video-only): softening "
                    "keyframe pins. max_soften=%.2f crossover=%.2f", max_soften, crossover,
                )
            return soften_mask_value(denoise_mask, gain)

        m.set_model_denoise_mask_function(denoise_mask_function)
        log.info(
            "[LTXKeyframeSoftenSchedule] patched: max_soften=%.2f (->%.3f) "
            "crossover=%.2f (->%.3f) sigma_max=%.3f",
            max_soften_w, max_soften, crossover_w, crossover, sigma_max,
        )
        return (m,)


NODE_CLASS_MAPPINGS = {"LTXKeyframeSoftenSchedule": LTXKeyframeSoftenSchedule}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXKeyframeSoftenSchedule": "LTX Keyframe Soften Schedule (Koolook)"}

__all__ = [
    "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
    "soften_gain", "soften_mask_value",
]
