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
the poses instead of snapping — then **restores** them by a configurable
``crossover`` measured in denoise *progress* (``1 - sigma/sigma_max``) so the poses
sharpen as denoise finishes. Because LTX's sigma schedule is front-loaded (sigma
held high, dropping late), a SMALL progress fraction already covers the early
motion-forming steps — hence the ``crossover`` widget maps its 0-1 range onto a
small internal ``0-0.05`` progress band (see ``_CROSSOVER_SCALE``).

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


def soften_denoise_mask(denoise_mask, gain, nested_ctor=None):
    """Soften a denoise mask, sparing the audio half of an A/V mask.

    The A/V latent mask is a ``NestedTensor((video_mask, audio_mask))`` produced by
    ``LTXVConcatAVLatent`` — video FIRST, audio SECOND. We soften ONLY part 0
    (video keyframe pins) and pass part 1 (audio) through untouched, so lip-sync
    timing is never disturbed. A plain (video-only) mask is softened whole.

    ``nested_ctor`` rebuilds the nested mask; defaults to
    ``comfy.nested_tensor.NestedTensor`` and is injectable so the audio-safety
    behavior can be unit-tested with a stub. ``gain<=0`` / ``None`` is a no-op.
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


class LTXKeyframeSoftenSchedule:
    """Sigma-schedule the LTX keyframe pins: soft early (smooth motion), sharp late."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "LTX 2.3 model (place after the other model patchers)."}),
                "max_soften": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "EASE amount — HOW HARD the keyframe pins loosen at the "
                               "softening peak. Amplitude only: independent of the "
                               "sampler/sigma schedule, so it always means the same "
                               "thing. Normalized 0-1 -> internal 0-0.20. Default 0.5 "
                               "= ~10% unpinned at peak (a hard pin 0.20 -> 0.28). "
                               "0 = stock (no ease); higher = smoother but more "
                               "invented in-between motion.",
                }),
                "crossover": ("FLOAT", {
                    "default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "INVENTION window — HOW LONG the pins stay loose before "
                               "re-locking, measured in denoise sigma-progress "
                               "(1 - sigma/sigma_max), NOT step count. NOTE: the number "
                               "of steps it actually covers SHIFTS if you change the "
                               "sampler/sigma schedule. Normalized 0-1 -> internal "
                               "0-0.05; default 0.65 protects roughly the first few "
                               "(high-sigma) steps. LOWER = lock sooner = FEWER invented "
                               "in-between frames; higher = more invented motion (and "
                               "risk of a noisy/under-resolved image).",
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

        # Chain any pre-existing denoise_mask_function instead of clobbering it
        # (mirrors how the sibling LTX patchers chain model_function_wrapper).
        prev_fn = m.model_options.get("denoise_mask_function")
        diag = [False]
        warned = [False]

        def denoise_mask_function(sigma, denoise_mask, extra_options=None):
            if prev_fn is not None:
                denoise_mask = prev_fn(sigma, denoise_mask, extra_options=extra_options)
            if denoise_mask is None:
                return denoise_mask
            try:
                s = float(sigma.max()) if hasattr(sigma, "max") else float(sigma)
            except Exception as _e:
                s = sigma_max
                if not warned[0]:
                    warned[0] = True
                    log.warning(
                        "[LTXKeyframeSoftenSchedule] could not read sigma (%s); "
                        "softening at max this step.", _e,
                    )
            gain = soften_gain(s, sigma_max, crossover, max_soften)
            if gain <= 0.0:
                return denoise_mask
            if not diag[0]:
                diag[0] = True
                nested = getattr(denoise_mask, "is_nested", False)
                log.info(
                    "[LTXKeyframeSoftenSchedule] active (%s): softening video keyframe "
                    "pins%s. max_soften=%.3f crossover=%.3f",
                    "A/V" if nested else "video-only",
                    ", audio untouched" if nested else "", max_soften, crossover,
                )
            return soften_denoise_mask(denoise_mask, gain)

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
    "soften_gain", "soften_mask_value", "soften_denoise_mask",
]
