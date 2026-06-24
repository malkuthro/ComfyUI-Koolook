# SPDX-License-Identifier: GPL-3.0-or-later
#
# ComfyUI-Koolook — Clean Latent Slice node.
# Copyright (C) 2026 ComfyUI-Koolook contributors (kforgelabs).
"""Clean Latent Slice (Koolook) — trim frames off a video latent.

Companion to the LTX Director "Ghost Mask" reference feature. That mode appends
reference image(s) as guide frames *after* the clean video region so the model
anchors identity (face / mouth shape) without the references appearing in the
output. After sampling, the trailing reference frames must be removed: wire the
Director's ``clean_latent_frames`` output to ``length`` (with ``start=0``) and
this node slices the latent back down to just the video.

The trick — and the use of ``torch.narrow`` to avoid PyTorch NestedTensor
slicing issues — is adapted from CGlide's CleanLatentSliceCS
(WhatDreamsCost-CSGlide, branch main_cs @ 21dd076, GPL-3.0).

The clamp math mirrors ``reference_ghost.clean_slice_bounds`` (unit-tested in
tests/forks); torch is imported lazily so this module loads without it.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def _clamp_slice(size: int, start: int, length: int) -> tuple[int, int]:
    """Clamp (start, length) to a frame axis of ``size``. Mirrors
    ``forks.whatdreamscost_koolook.versions.v2_0_2.reference_ghost.clean_slice_bounds``.
    """
    sz = int(size)
    if sz <= 0:
        return 0, 0
    st = min(max(0, int(start)), sz - 1)
    ln = min(max(0, int(length)), sz - st)
    return st, ln


class CleanLatentSlice:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "start": ("INT", {
                    "default": 0, "min": 0, "max": 100000, "step": 1,
                    "tooltip": "First latent frame to keep. 0 for Ghost Mask (refs are at the end).",
                }),
                "length": ("INT", {
                    "default": 1, "min": 1, "max": 100000, "step": 1,
                    "tooltip": "Number of latent frames to keep. Wire the Director's 'clean_latent_frames'.",
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "slice_latent"
    CATEGORY = "Koolook/LTX"
    DESCRIPTION = (
        "Slice a video latent to [start, start+length) along the temporal axis. "
        "Use it to drop the trailing reference frames added by LTX Director's "
        "Ghost Mask reference feature (start=0, length=clean_latent_frames)."
    )

    def slice_latent(self, latent, start, length):
        import torch  # lazy: keep module importable without torch

        out = latent.copy()

        def _narrow(tensor):
            dims = tensor.ndim if hasattr(tensor, "ndim") else len(tensor.shape)
            # 5D [B, C, F, H, W] -> temporal dim 2; 4D/3D [F, ...] -> dim 0.
            axis = 2 if dims == 5 else 0
            size = tensor.size(axis)
            st, ln = _clamp_slice(size, start, length)
            if ln <= 0:
                return tensor
            try:
                return torch.narrow(tensor, axis, st, ln)
            except Exception as e:
                log.warning("[CleanLatentSlice] torch.narrow failed (%s); using python slice", e)
                idx = [slice(None)] * dims
                idx[axis] = slice(st, st + ln)
                return tensor[tuple(idx)]

        if "samples" in out:
            out["samples"] = _narrow(out["samples"])
        # Only slice a real per-frame mask, not a broadcasted (size-1) one.
        if "noise_mask" in out:
            nm = out["noise_mask"]
            nm_dims = nm.ndim if hasattr(nm, "ndim") else len(nm.shape)
            nm_axis = 2 if nm_dims == 5 else 0
            if nm.size(nm_axis) > 1:
                out["noise_mask"] = _narrow(nm)

        return (out,)


NODE_CLASS_MAPPINGS = {"CleanLatentSlice": CleanLatentSlice}
NODE_DISPLAY_NAME_MAPPINGS = {"CleanLatentSlice": "Clean Latent Slice (Koolook)"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
