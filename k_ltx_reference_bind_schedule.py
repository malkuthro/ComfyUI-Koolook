# SPDX-License-Identifier: GPL-3.0-or-later
#
# ComfyUI-Koolook — LTX Reference Bind Schedule node.
# Copyright (C) 2026 ComfyUI-Koolook contributors (kforgelabs).
"""LTX Reference Bind Schedule (Koolook) — ramp reference identity up at low sigma.

Companion to the LTX Director Ghost Mask reference. The reference image is added
as a guide frame; the core LTX transformer attends to it with a per-guide
**attention strength** (`guide_attention_entries[].strength`) which it turns into
a log-space additive self-attention bias, rebuilt **every denoise step** in
`comfy/ldm/lightricks/model.py::_build_guide_self_attention_mask`. That builder
explicitly supports `strength > 1.0` to *amplify* a guide's attention.

This node schedules that amplification over the denoise so identity locks late
without disturbing motion that forms early:

* **Early steps (high sigma)** — gain ~1.0: the reference attention is neutral,
  big motion resolves freely.
* **Late steps (low sigma)** — gain ramps up to `peak_strength`: the refinement
  conforms harder to the reference's identity (face / mouth).

It's the inverse of the motion curve, and the mirror of `LTXAVBindSchedule`
(which ramps audio->video up at low sigma for lip-sync). It touches ONLY the
attention bias of the trailing `num_references` guide entries (references are
appended after the keyframes) — the keyframe pins and the noise-mask freezing
are untouched.

Mechanics (same as the A/V node): a `model_function_wrapper` captures the
current sigma each step and stashes the gain in `transformer_options`; an
object-patch on `_build_guide_self_attention_mask` reads it and scales the
reference entries' `strength` before the mask is built.

NOTE: the `ref_gain` ramp is unit-tested; the model wiring is validated by
rendering against a live LTX 2.3 model (not in CI).
"""
from __future__ import annotations

import types
import logging

log = logging.getLogger(__name__)

_REF_GAIN_KEY = "nghtdrp_ref_gain"  # per-step reference-attention multiplier


def ref_gain(sigma, sigma_max, ramp_start=0.5, ramp_end=0.9, peak=2.0):
    """Reference-attention multiplier for the current ``sigma``.

    Progress runs 0 at the start of denoise (``sigma == sigma_max``) to 1 at the
    end (``sigma -> 0``). Holds at 1.0 (neutral) until ``ramp_start``, then
    smoothsteps up to ``peak`` by ``ramp_end`` (progress fractions in [0, 1]).
    Returning 1.0 means "no change" — the mask builder then skips entirely.
    """
    if sigma_max <= 0.0:
        return float(peak)
    progress = 1.0 - (float(sigma) / float(sigma_max))
    progress = 0.0 if progress < 0.0 else (1.0 if progress > 1.0 else progress)
    if ramp_end <= ramp_start:
        return float(peak) if progress >= ramp_end else 1.0
    if progress <= ramp_start:
        return 1.0
    if progress >= ramp_end:
        return float(peak)
    t = (progress - ramp_start) / (ramp_end - ramp_start)
    s = t * t * (3.0 - 2.0 * t)  # smoothstep
    return 1.0 + (float(peak) - 1.0) * s


def _boosted_entries(resolved_entries, ref_count, gain):
    """Return a copy of ``resolved_entries`` with the LAST ``ref_count`` entries'
    ``strength`` multiplied by ``gain`` (references are appended after keyframes).
    Returns the original list unchanged when there's nothing to do."""
    if not resolved_entries or gain == 1.0 or ref_count <= 0:
        return resolved_entries
    n = len(resolved_entries)
    k = min(int(ref_count), n)
    if k <= 0:
        return resolved_entries
    out = list(resolved_entries)
    for i in range(n - k, n):
        e = dict(out[i])
        e["strength"] = float(e.get("strength", 1.0)) * float(gain)
        out[i] = e
    return out


def _make_guide_mask_wrapper(underlying, ref_count, diag):
    """Bound-method wrapper around ``_build_guide_self_attention_mask``: scale the
    reference entries' attention strength by the per-step gain before the mask is
    built. ``diag`` is a shared 1-elem list for one-shot logging."""

    def wrapped(_self_module, x, transformer_options, merged_args, *args, **kwargs):
        to = transformer_options or {}
        gain = float(to.get(_REF_GAIN_KEY, 1.0))
        if gain != 1.0 and ref_count > 0 and isinstance(merged_args, dict):
            entries = merged_args.get("resolved_guide_entries")
            boosted = _boosted_entries(entries, ref_count, gain)
            if boosted is not entries:
                merged_args = {**merged_args, "resolved_guide_entries": boosted}
                if not diag[0]:
                    diag[0] = True
                    log.info(
                        "[LTXReferenceBindSchedule] reference attention scheduling "
                        "active: %d trailing guide entr(y/ies), gain reaches x%.2f "
                        "at low sigma.", min(int(ref_count), len(entries or [])), gain,
                    )
        return underlying(x, transformer_options, merged_args, *args, **kwargs)

    return wrapped


class LTXReferenceBindSchedule:
    """Ramp the LTX Ghost Mask reference's attention up as sigma falls."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "LTX 2.3 model (the same one feeding the guide pipeline)."}),
                "num_references": ("INT", {
                    "default": 1, "min": 0, "max": 64, "step": 1,
                    "tooltip": "How many trailing guide frames are references (= how many images you fed LTX Director's reference_images). They're appended after the keyframes, so the LAST N guide entries are the references.",
                }),
                "peak_strength": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05,
                    "tooltip": "Reference attention multiplier at the end of denoise (low sigma). 1.0 = no boost; 2.0 = double the identity pull during refinement.",
                }),
            },
            "optional": {
                "ramp_start": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoise progress (0=start, 1=end) where the reference boost begins. Keep above where big motion finalizes (~0.5-0.6) so motion forms first.",
                }),
                "ramp_end": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoise progress where the boost reaches peak_strength.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Koolook/LTX"
    DESCRIPTION = (
        "Ramp the Ghost Mask reference's attention UP at low sigma so identity "
        "locks during refinement without disturbing the motion that forms early. "
        "The inverse of the motion curve; mirror of LTX A/V Bind Schedule. One "
        "pass, no re-noise; touches only the reference guide entries."
    )

    def patch(self, model, num_references, peak_strength, ramp_start=0.5, ramp_end=0.9):
        m = model.clone()

        if int(num_references) <= 0:
            log.info("[LTXReferenceBindSchedule] num_references=0 — model passed through.")
            return (m,)

        try:
            sigma_max = float(m.get_model_object("model_sampling").sigma_max)
        except Exception as e:  # pragma: no cover - live model
            log.warning("[LTXReferenceBindSchedule] no model_sampling.sigma_max: %s", e)
            sigma_max = 1.0

        key = "diffusion_model._build_guide_self_attention_mask"
        try:
            underlying = m.get_model_object(key)
            dm = m.get_model_object("diffusion_model")
        except Exception as e:  # pragma: no cover - live model
            log.warning(
                "[LTXReferenceBindSchedule] no %s on this model (not an LTX guide "
                "model?): %s — passed through unchanged.", key, e,
            )
            return (m,)

        diag = [False]
        wrapper = _make_guide_mask_wrapper(underlying, int(num_references), diag)
        try:
            m.add_object_patch(key, types.MethodType(wrapper, dm))
        except Exception as e:  # pragma: no cover - live model
            log.warning("[LTXReferenceBindSchedule] guide-mask patch failed: %s", e)
            return (m,)

        prev = m.model_options.get("model_function_wrapper")

        def unet_wrapper(model_function, params):
            try:
                sigma = float(params["timestep"].max())
            except Exception:
                sigma = sigma_max
            gain = ref_gain(sigma, sigma_max, ramp_start, ramp_end, peak_strength)
            c = params["c"]
            to = dict(c.get("transformer_options", {}))
            to[_REF_GAIN_KEY] = gain
            c = {**c, "transformer_options": to}
            params = {**params, "c": c}
            if prev is not None:
                return prev(model_function, params)
            return model_function(params["input"], params["timestep"], **params["c"])

        m.set_model_unet_function_wrapper(unet_wrapper)
        log.info(
            "[LTXReferenceBindSchedule] scheduling %d reference(s): peak x%.2f, "
            "ramp %.2f..%.2f of denoise.", int(num_references), float(peak_strength),
            float(ramp_start), float(ramp_end),
        )
        return (m,)


NODE_CLASS_MAPPINGS = {"LTXReferenceBindSchedule": LTXReferenceBindSchedule}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXReferenceBindSchedule": "LTX Reference Bind Schedule (Koolook)"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
