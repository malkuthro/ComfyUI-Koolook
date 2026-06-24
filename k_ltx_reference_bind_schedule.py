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

WORKS WITHOUT AN IC-LoRA. The attention-bias machinery is BASE-model
functionality — `_build_guide_self_attention_mask` builds the mask from
`resolved_guide_entries` regardless of any LoRA. The catch is that
LTXDirectorGuide only *populates* those entries `if is_lora_active`, so in the
no-LoRA Ghost Mask path the list is empty. This node closes that gap: when the
entries are missing it **fabricates them itself** for the trailing reference
tokens (splitting `num_guide_tokens` across the `num_keyframes + num_references`
guide frames and assigning the ramped strength to the reference share). So the
reference gets the same amplified-attention channel an IC-LoRA would use, inside
the Director, with no LoRA, no guide-node vendor, and no graph change. When an
IC-LoRA *is* active and real entries exist, it scales those instead.

Mechanics (same as the A/V node): a `model_function_wrapper` captures the
current sigma each step and stashes the gain in `transformer_options`; an
object-patch on `_build_guide_self_attention_mask` reads it and fabricates (or
scales) the reference entries' `strength` before the mask is built.

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


def fabricate_reference_entries(num_guide_tokens, num_references, num_keyframes, strength):
    """Build ``resolved_guide_entries`` for the trailing reference tokens when the
    guide node created none (the no-IC-LoRA path).

    Splits ``num_guide_tokens`` evenly across the ``num_keyframes + num_references``
    guide frames and gives the trailing reference share ``strength`` while the
    keyframe share stays 1.0 (neutral). The entries match the shape the core
    builder consumes (``strength`` + ``surviving_count`` + optional pixel_mask /
    latent_shape). When ``num_keyframes`` is 0 the whole guide region is treated
    as reference. Returns ``[]`` when there's nothing to amplify.
    """
    n_guide = int(num_guide_tokens)
    n_ref = int(num_references)
    n_kf = max(0, int(num_keyframes))
    if n_guide <= 0 or n_ref <= 0:
        return []
    total_frames = n_ref + n_kf
    ref_tokens = round(n_guide * n_ref / total_frames) if total_frames > 0 else n_guide
    ref_tokens = max(1, min(int(ref_tokens), n_guide))
    kf_tokens = n_guide - ref_tokens
    entries = []
    if kf_tokens > 0:
        entries.append({"strength": 1.0, "surviving_count": kf_tokens, "pixel_mask": None, "latent_shape": None})
    entries.append({"strength": float(strength), "surviving_count": ref_tokens, "pixel_mask": None, "latent_shape": None})
    return entries


def _make_guide_mask_wrapper(underlying, ref_count, kf_count, peak, diag):
    """Bound-method wrapper around ``_build_guide_self_attention_mask``.

    If the guide node already populated entries (IC-LoRA path) scale the trailing
    ``ref_count`` of them by the per-step gain. If not (no-LoRA Ghost Mask path)
    fabricate entries for the trailing reference tokens at the gain strength. Then
    hand off to the original builder, which turns it into the additive bias.
    """

    def wrapped(_self_module, x, transformer_options, merged_args, *args, **kwargs):
        to = transformer_options or {}
        gain = float(to.get(_REF_GAIN_KEY, 1.0))
        ma = merged_args if isinstance(merged_args, dict) else {}
        entries = ma.get("resolved_guide_entries")
        n_guide = int(ma.get("num_guide_tokens") or 0)

        mode = "passthrough"
        if entries:  # IC-LoRA path — scale real entries
            if gain != 1.0 and ref_count > 0:
                boosted = _boosted_entries(entries, ref_count, gain)
                if boosted is not entries:
                    merged_args = {**ma, "resolved_guide_entries": boosted}
                    mode = "scaled"
        elif n_guide > 0 and ref_count > 0 and gain != 1.0:  # no-LoRA path — fabricate
            fab = fabricate_reference_entries(n_guide, ref_count, kf_count, gain)
            if fab:
                merged_args = {**ma, "resolved_guide_entries": fab}
                mode = "fabricated"

        if not diag.get("first"):
            diag["first"] = True
            log.info(
                "[LTXReferenceBindSchedule] active: num_guide_tokens=%s entries=%s "
                "ref_count=%d num_keyframes=%d peak=%.2f (amplifies the trailing "
                "reference tokens at low sigma).",
                n_guide, (len(entries) if entries else 0), int(ref_count),
                int(kf_count), float(peak),
            )
        if mode in ("scaled", "fabricated") and not diag.get(mode):
            diag[mode] = True
            log.info("[LTXReferenceBindSchedule] %s reference attention (gain x%.2f).", mode, gain)
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
                    "tooltip": "How many trailing guide frames are references (= how many images you fed LTX Director's reference_images). They're appended after the keyframes, so the LAST N guide frames are the references.",
                }),
                "peak_strength": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05,
                    "tooltip": "Reference attention multiplier at the end of denoise (low sigma). 1.0 = no boost; 2.0 = double the identity pull during refinement. Try 2-4.",
                }),
            },
            "optional": {
                "num_keyframes": ("INT", {
                    "default": 0, "min": 0, "max": 256, "step": 1,
                    "tooltip": "No-LoRA path only: how many timeline image keyframes precede the references, so the node knows the keyframe/reference token split. Set this = your number of timeline image segments (e.g. 4). Leave 0 and the WHOLE guide region is treated as reference (also boosts keyframes). Ignored when an IC-LoRA populated real entries.",
                }),
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
        "Ramp the reference's guide-attention UP at low sigma so identity locks "
        "during refinement without disturbing early motion. Works WITHOUT an "
        "IC-LoRA: it fabricates the per-guide attention entries for the trailing "
        "reference tokens itself (set num_keyframes so it knows the split), so the "
        "reference gets the same amplified-attention channel an IC-LoRA would use "
        "— inside the Director, no guide-node changes. Mirror of LTX A/V Bind "
        "Schedule. One pass, no re-noise."
    )

    def patch(self, model, num_references, peak_strength, num_keyframes=0, ramp_start=0.5, ramp_end=0.9):
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

        diag = {}
        wrapper = _make_guide_mask_wrapper(
            underlying, int(num_references), int(num_keyframes), float(peak_strength), diag,
        )
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
