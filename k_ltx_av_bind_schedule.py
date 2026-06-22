# SPDX-License-Identifier: GPL-3.0-or-later
#
# ComfyUI-Koolook — LTX A/V Bind Schedule node.
# Copyright (C) 2026 ComfyUI-Koolook contributors (kforgelabs).
"""LTX A/V Bind Schedule (Koolook) — decouple big motion from lip-sync.

LTX 2.3's audio-video transformer (`comfy/ldm/lightricks/av_model.py`,
`BasicAVTransformerBlock`) injects audio into the video stream through a single
gated cross-attention per block::

    a2v_out = self.audio_to_video_attn(vx_scaled, context=ax_scaled, ...)
    vx.addcmul_(a2v_out, gate_out_a2v)

…and only runs it when ``transformer_options["a2v_cross_attn"]`` is truthy. That
audio->video term is what makes the lips drive the face — and what fights a hard
keyframe pose change when a transition lands on an audio peak (the jump).

This node schedules that binding across the denoise instead of leaving it fully
on every step: it scales ``a2v_out`` by a per-step **gain** that ramps from
``early_gain`` (high sigma / early steps) up to ``1.0`` (low sigma / late steps).

    high sigma  ->  gain ~ early_gain (e.g. 0.0)  ->  big motion resolves audio-blind
    low  sigma  ->  gain = 1.0                    ->  lips bind onto settled motion

So the coarse motion (decided early in diffusion) settles without audio
interference, and lip-sync (high-frequency, decided late) is applied on top —
one pass, no re-noise.

Mechanism (mirrors the PromptRelay attention patching in the fork's patches.py):
  * an object-patch on each block's ``audio_to_video_attn.forward`` multiplies
    its output by ``transformer_options["nghtdrp_a2v_gain"]`` (default 1.0 = no-op),
  * a unet-function-wrapper computes that gain from the current sigma each step
    and writes it into the cond's ``transformer_options`` before the model runs.

gain == 1.0 everywhere reproduces stock behavior, so a graph without this node
(or with ``early_gain=1.0``) is unchanged.

NOTE: the pure ramp (`a2v_gain`) is unit-tested; the model wiring is validated by
rendering on a machine with the LTX 2.3 AV model loaded (not in CI).
"""
from __future__ import annotations

import math
import types
import logging

log = logging.getLogger(__name__)

# key under which the per-step gain is threaded through transformer_options
_GAIN_KEY = "nghtdrp_a2v_gain"


def a2v_gain(
    sigma: float,
    sigma_max: float,
    bind_start: float = 0.5,
    bind_end: float = 0.7,
    early_gain: float = 0.0,
) -> float:
    """Audio->video gain for the current ``sigma``.

    Progress runs 0 at the start of denoise (``sigma == sigma_max``) to 1 at the
    end (``sigma -> 0``). Gain holds at ``early_gain`` until ``bind_start``, then
    smoothsteps up to ``1.0`` by ``bind_end`` (both expressed as progress
    fractions in [0, 1]).
    """
    if sigma_max <= 0.0:
        return 1.0
    progress = 1.0 - (float(sigma) / float(sigma_max))
    progress = 0.0 if progress < 0.0 else (1.0 if progress > 1.0 else progress)

    if bind_end <= bind_start:  # degenerate -> hard step at bind_end
        return 1.0 if progress >= bind_end else early_gain
    if progress <= bind_start:
        return early_gain
    if progress >= bind_end:
        return 1.0
    t = (progress - bind_start) / (bind_end - bind_start)
    s = t * t * (3.0 - 2.0 * t)  # smoothstep
    return early_gain + (1.0 - early_gain) * s


def _make_a2v_scale_method(underlying):
    """Bound-method wrapper: run the original a2v cross-attn, scale its output by
    the per-step gain found in transformer_options (1.0 -> untouched)."""

    def wrapped(_self_module, *args, **kwargs):
        out = underlying(*args, **kwargs)
        to = kwargs.get("transformer_options", {}) or {}
        gain = to.get(_GAIN_KEY, 1.0)
        if gain == 1.0:
            return out
        return out * gain

    return wrapped


class LTXAVBindSchedule:
    """Schedule LTX 2.3 audio->video binding strength across the denoise."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "LTX 2.3 audio-video model."}),
                "early_gain": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Audio->video strength during the early (high-sigma) "
                               "steps where big motion resolves. 0.0 = fully "
                               "audio-blind motion; 1.0 = stock behavior.",
                }),
                "bind_start": ("FLOAT", {
                    "default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoise progress (0=start, 1=end) where audio begins "
                               "ramping in. Below this, gain stays at early_gain.",
                }),
                "bind_end": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoise progress where audio binding reaches full "
                               "strength (1.0). Lip-sync lands between bind_start "
                               "and here.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Koolook/LTX"
    DESCRIPTION = (
        "Resolve big motion before lip-sync: ramps LTX 2.3's audio->video "
        "cross-attention from early_gain (early steps) to full (late steps), so "
        "keyframe transitions on audio peaks stop jumping. No re-noise; one pass."
    )

    def patch(self, model, early_gain, bind_start, bind_end):
        m = model.clone()

        # sigma_max for normalizing progress
        try:
            ms = m.get_model_object("model_sampling")
            sigma_max = float(ms.sigma_max)
        except Exception as e:  # pragma: no cover - depends on live model
            log.warning("[LTXAVBindSchedule] could not read model_sampling.sigma_max: %s", e)
            sigma_max = 1.0

        # 1) patch every block's audio_to_video_attn.forward to scale by the gain
        patched = 0
        try:
            dm = m.get_model_object("diffusion_model")
            blocks = getattr(dm, "transformer_blocks", None) or []
            for idx, block in enumerate(blocks):
                if getattr(block, "audio_to_video_attn", None) is None:
                    continue
                key = f"diffusion_model.transformer_blocks.{idx}.audio_to_video_attn.forward"
                underlying = m.get_model_object(key)
                wrapper = _make_a2v_scale_method(underlying)
                m.add_object_patch(
                    key, types.MethodType(wrapper, block.audio_to_video_attn)
                )
                patched += 1
        except Exception as e:  # pragma: no cover - depends on live model
            log.warning("[LTXAVBindSchedule] block patch failed: %s", e)

        if patched == 0:
            log.warning(
                "[LTXAVBindSchedule] no audio_to_video_attn blocks found — is this "
                "an LTX 2.3 audio-video model? Passing model through unchanged."
            )
            return (m,)

        # 2) per-step wrapper: compute gain from current sigma, thread it down
        prev = m.model_options.get("model_function_wrapper")

        def unet_wrapper(model_function, params):
            try:
                sigma = float(params["timestep"].max())
            except Exception:
                sigma = sigma_max
            gain = a2v_gain(sigma, sigma_max, bind_start, bind_end, early_gain)
            c = params["c"]
            to = dict(c.get("transformer_options", {}))
            to[_GAIN_KEY] = gain
            c = {**c, "transformer_options": to}
            params = {**params, "c": c}
            if prev is not None:
                return prev(model_function, params)
            return model_function(params["input"], params["timestep"], **params["c"])

        m.set_model_unet_function_wrapper(unet_wrapper)
        log.info(
            "[LTXAVBindSchedule] patched %d a2v blocks; early_gain=%.2f bind=%.2f..%.2f",
            patched, early_gain, bind_start, bind_end,
        )
        return (m,)


NODE_CLASS_MAPPINGS = {"LTXAVBindSchedule": LTXAVBindSchedule}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXAVBindSchedule": "LTX A/V Bind Schedule (Koolook)"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "a2v_gain"]
