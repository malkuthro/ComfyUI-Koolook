# SPDX-License-Identifier: GPL-3.0-or-later
#
# ComfyUI-Koolook — LTX Guide Reference Strength node.
# Copyright (C) 2026 ComfyUI-Koolook contributors (kforgelabs).
"""LTX Guide Reference Strength (Koolook) — boost ONLY the reference frames.

Companion to LTX Director's Ghost Mask reference feature. The Director appends
the reference image(s) as the LAST entries in ``guide_data`` (tagged with
``reference_count``) so they ride the same append-and-crop path as the timeline
keyframes. Every guide frame's strength becomes a noise-mask pin
(``noise_mask = 1 - strength``) in the downstream LTXDirectorGuide.

This node rewrites the strength of just those trailing reference entries,
leaving the keyframe pins untouched. The intended use is a two-stage pipeline:
feed stage 1 the Director's ``guide_data`` unchanged (light reference, so motion
forms freely), and route stage 2's guide_data through this node at a higher
reference strength — so the identity locks during the low-denoise refinement
without disturbing the motion already set in stage 1.

Pure-python; no torch. The strength math is unit-tested.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def scaled_reference_strengths(strengths, reference_count, value, mode="absolute"):
    """Return a new strengths list with the last ``reference_count`` entries
    rewritten. ``mode='absolute'`` sets them to ``value``; ``mode='multiply'``
    multiplies the existing strength by ``value``. Other entries (keyframes) are
    left untouched. Clamps to [0, 1] (guide strength -> noise-mask range)."""
    out = [float(s) for s in strengths]
    n = len(out)
    k = max(0, min(int(reference_count), n))
    for i in range(n - k, n):
        v = float(value) if mode == "absolute" else out[i] * float(value)
        out[i] = max(0.0, min(1.0, v))
    return out


class LTXGuideReferenceStrength:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guide_data": ("GUIDE_DATA",),
                "reference_strength": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "New strength for the reference frames only. Higher = harder identity pin. Use ~0.9 on a stage-2 refinement guide.",
                }),
                "mode": (["absolute", "multiply"], {
                    "default": "absolute",
                    "tooltip": "absolute: set reference strength to this value. multiply: scale the Director's reference strength by it.",
                }),
            }
        }

    RETURN_TYPES = ("GUIDE_DATA",)
    RETURN_NAMES = ("guide_data",)
    FUNCTION = "apply"
    CATEGORY = "Koolook/LTX"
    DESCRIPTION = (
        "Rewrite the strength of just the Ghost Mask reference frames in "
        "guide_data (the trailing entries tagged by LTX Director), leaving the "
        "keyframe pins untouched. Insert before a stage-2 refinement guide to "
        "lock identity harder without disturbing stage-1 motion."
    )

    def apply(self, guide_data, reference_strength, mode):
        ref_count = int((guide_data or {}).get("reference_count", 0) or 0)
        if not guide_data or ref_count <= 0:
            log.info("[LTXGuideReferenceStrength] no reference frames in guide_data; passthrough.")
            return (guide_data,)
        out = dict(guide_data)
        out["strengths"] = scaled_reference_strengths(
            guide_data.get("strengths", []), ref_count, reference_strength, mode,
        )
        log.info(
            "[LTXGuideReferenceStrength] set %d reference frame(s) -> %.2f (%s).",
            ref_count, float(reference_strength), mode,
        )
        return (out,)


NODE_CLASS_MAPPINGS = {"LTXGuideReferenceStrength": LTXGuideReferenceStrength}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXGuideReferenceStrength": "LTX Guide Reference Strength (Koolook)"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
