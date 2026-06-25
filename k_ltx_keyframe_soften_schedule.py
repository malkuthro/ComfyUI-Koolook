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
pin instead: it **softens** the keyframe pins early, when the model is deciding
coarse motion — letting it form a smooth trajectory between the poses — then
**restores** them so the poses sharpen as denoise finishes.

The softening window is set by ``protect_until`` — a FRACTION OF THE STEPS (0-1),
resolved against the sampler's ACTUAL sigma schedule at run time. ``0.4`` always
means "soften the first 40% of the steps", on any sigma curve / step count, so it
needs no re-tuning when the schedule changes. (This replaces the old ``crossover``
knob, which was measured in sigma-progress and therefore covered a wildly
different number of steps on different curves — flat front-loaded curves stretched
it across many steps, steep curves collapsed it to ~1.)

It adds NOTHING (no extra guide frames / poses — that approach backfires; every
guide is a pose target). It only loosens/tightens the keyframes you already have.

Audio-safe: the A/V latent mask is a ``NestedTensor((video_mask, audio_mask))``
(``LTXVConcatAVLatent``); we soften ONLY the video mask and pass the audio mask
through untouched, so lip-sync timing is never disturbed.

The node emits an ``info`` STRING describing the resolved window — wire a SIGMAS
input (the same one your sampler uses) to read the exact softened steps + sigma
range in-graph via a text node.

NOTE: the pure ``soften_gain`` ramp and ``resolve_soften_window`` are unit-tested;
the model wiring (the denoise-mask function) is validated by rendering against a
live LTX 2.3 model.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# max_soften widget is a normalized 0-1 slider mapping onto the *useful* internal
# amplitude range (the effect is sensitive, so raw values bunch up near 0).
_MAX_SOFTEN_SCALE = 0.2    # widget 0..1 -> internal max_soften 0..0.2


def soften_gain(sigma, sigma_max, crossover=0.55, max_soften=0.6):
    """How much to soften the keyframe pins at the current ``sigma``.

    Denoise progress runs 0 at the start (``sigma == sigma_max``) to 1 at the end
    (``sigma -> 0``). Returns ``max_soften`` at progress 0, smoothsteps down to 0
    by ``crossover`` progress, and stays 0 afterwards (pins fully restored).

    ``crossover`` here is an internal sigma-progress threshold; the node derives it
    from the step-fraction ``protect_until`` against the live schedule
    (see ``resolve_soften_window``). A returned gain ``g`` raises a pinned mask
    value ``m`` toward 1 (unpinned) via ``m + (1 - m) * g``.
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


def resolve_soften_window(sigmas, protect_until):
    """Resolve the step-fraction ``protect_until`` against an actual sigma schedule.

    ``sigmas`` is the sampler's sigma list (length ``n_steps + 1``; the last entry
    is the final ~0 sigma). ``protect_until`` in [0, 1] is the fraction of the
    STEPS to soften. Returns ``(cutoff_step, threshold_sigma, n_steps)``: the
    soften ramps from full at step 0 down to 0 at ``cutoff_step``, and
    ``threshold_sigma`` is the sigma at that step. Reading from the actual schedule
    makes "first X% of steps" mean the same thing on any sigma curve.
    """
    vals = [float(s) for s in sigmas]
    n = len(vals)
    if n <= 1:
        return 0, (vals[0] if vals else 0.0), max(0, n - 1)
    n_steps = n - 1
    cutoff = int(round(float(protect_until) * n_steps))
    cutoff = max(0, min(cutoff, n_steps))
    return cutoff, vals[cutoff], n_steps


def _format_info(max_soften_w, eff_max_soften, protect_until, window):
    """Build the human-readable ``info`` string. ``window`` is the
    (cutoff, threshold, n_steps) tuple from resolve_soften_window, or None."""
    head = (
        "LTX Keyframe Soften\n"
        f"  max_soften {max_soften_w:.2f} -> {eff_max_soften:.3f} "
        f"(~{eff_max_soften * 100:.0f}% unpinned at peak)"
    )
    if eff_max_soften <= 0.0 or protect_until <= 0.0:
        return head + "\n  [inactive — model passed through]"
    if window is not None:
        cutoff, threshold, n_steps, sigma0 = window
        return (
            head
            + f"\n  protect_until {protect_until:.2f} -> soften steps 0-{cutoff} of {n_steps}"
            + f"\n  sigma {sigma0:.3f} -> {threshold:.3f}  (gain ramps max->0)"
        )
    return (
        head
        + f"\n  protect_until {protect_until:.2f} = first {protect_until * 100:.0f}% of steps"
        + "\n  (wire SIGMAS to preview exact steps / sigma range)"
    )


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
                               "sampler/sigma schedule. Normalized 0-1 -> internal "
                               "0-0.20. Default 0.5 = ~10% unpinned at peak (a hard pin "
                               "0.20 -> 0.28). 0 = stock; higher = smoother but more "
                               "invented in-between motion.",
                }),
                "protect_until": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "HOW LONG to soften, as a FRACTION OF THE STEPS (resolved "
                               "against the live sigma schedule). 0.4 = soften the first "
                               "40% of the steps, on ANY curve / step count — no "
                               "re-tuning when the schedule changes. LOWER = lock sooner "
                               "= FEWER invented in-between frames; higher = more invented "
                               "motion (+ risk of a noisy/under-resolved image). 0 = off.",
                }),
            },
            "optional": {
                "sigmas": ("SIGMAS", {
                    "tooltip": "Optional. Wire the SAME sigmas you feed the sampler to "
                               "preview the exact softened steps + sigma range on the "
                               "'info' output (use a text / show-anything node). Display "
                               "only — behavior always uses the sampler's actual schedule "
                               "at run time.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "patch"
    CATEGORY = "Koolook/LTX"
    DESCRIPTION = (
        "Smooth motion early, sharp poses late: sigma-schedule the LTX keyframe "
        "pins — loosen them early so the model forms smooth motion between poses, "
        "restore them by `protect_until` (a fraction of the steps) so the poses "
        "sharpen. Adds no frames; audio mask untouched (lip-sync safe). The `info` "
        "output reports the resolved window (wire SIGMAS to preview it in-graph)."
    )

    def patch(self, model, max_soften, protect_until, sigmas=None):
        m = model.clone()

        max_soften_w = float(max_soften)
        eff_max_soften = max_soften_w * _MAX_SOFTEN_SCALE
        protect_until = float(protect_until)

        # --- info string (display only; never drives behavior) ---
        preview = None
        if sigmas is not None:
            try:
                cutoff, threshold, n_steps = resolve_soften_window(sigmas, protect_until)
                preview = (cutoff, threshold, n_steps, float(sigmas[0]))
            except Exception as e:
                log.warning("[LTXKeyframeSoftenSchedule] could not read preview sigmas: %s", e)
        info = _format_info(max_soften_w, eff_max_soften, protect_until, preview)

        if eff_max_soften <= 0.0 or protect_until <= 0.0:
            log.info("[LTXKeyframeSoftenSchedule] max_soften/protect_until = 0 — model passed through.")
            return (m, info)

        # Chain any pre-existing denoise_mask_function instead of clobbering it.
        prev_fn = m.model_options.get("denoise_mask_function")
        state = {"crossover": None, "ref": 0.0}   # resolved once at run time
        diag = [False]
        warned = [False]

        def denoise_mask_function(sigma, denoise_mask, extra_options=None):
            if prev_fn is not None:
                denoise_mask = prev_fn(sigma, denoise_mask, extra_options=extra_options)
            if denoise_mask is None:
                return denoise_mask

            # Resolve protect_until -> sigma-progress crossover from the ACTUAL
            # running schedule, once. This is what makes the step-fraction mean the
            # same thing on any curve.
            if state["crossover"] is None:
                rt = (extra_options or {}).get("sigmas")
                try:
                    cutoff, threshold, n_steps = resolve_soften_window(rt, protect_until)
                    ref = float(rt[0])   # anchor the ramp to the schedule's START sigma
                    state["ref"] = ref
                    state["crossover"] = max(0.0, 1.0 - threshold / ref) if ref > 0 else 0.0
                    log.info(
                        "[LTXKeyframeSoftenSchedule] soften steps 0-%d of %d "
                        "(sigma %.3f->%.3f), max_soften=%.3f",
                        cutoff, n_steps, ref, threshold, eff_max_soften,
                    )
                except Exception as _e:
                    state["ref"] = 0.0
                    state["crossover"] = 0.0
                    if not warned[0]:
                        warned[0] = True
                        log.warning(
                            "[LTXKeyframeSoftenSchedule] could not resolve the run's "
                            "sigma schedule (%s); softening disabled this run.", _e,
                        )

            cp = state["crossover"]
            if cp <= 0.0:
                return denoise_mask
            try:
                s = float(sigma.max()) if hasattr(sigma, "max") else float(sigma)
            except Exception:
                s = state["ref"]
            gain = soften_gain(s, state["ref"], cp, eff_max_soften)
            if gain <= 0.0:
                return denoise_mask
            if not diag[0]:
                diag[0] = True
                nested = getattr(denoise_mask, "is_nested", False)
                log.info(
                    "[LTXKeyframeSoftenSchedule] active (%s)%s.",
                    "A/V" if nested else "video-only",
                    " — audio mask untouched" if nested else "",
                )
            return soften_denoise_mask(denoise_mask, gain)

        m.set_model_denoise_mask_function(denoise_mask_function)
        log.info(
            "[LTXKeyframeSoftenSchedule] patched: max_soften=%.2f (->%.3f) "
            "protect_until=%.2f", max_soften_w, eff_max_soften, protect_until,
        )
        return (m, info)


NODE_CLASS_MAPPINGS = {"LTXKeyframeSoftenSchedule": LTXKeyframeSoftenSchedule}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXKeyframeSoftenSchedule": "LTX Keyframe Soften Schedule (Koolook)"}

__all__ = [
    "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
    "soften_gain", "soften_mask_value", "soften_denoise_mask",
    "resolve_soften_window",
]
