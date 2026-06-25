# SPDX-License-Identifier: GPL-3.0-or-later
"""
Koolook-original pure helpers for the LTX Director "Ghost Mask" reference
feature (REF image / character sheet identity anchor).

The trailing-reference trick — append the reference image(s) as guide frames
*after* the clean video region, let the transformer attend to them for
identity, then slice them off the output latent — is adapted from CGlide's
WhatDreamsCost-CSGlide (branch main_cs @ 21dd076, GPL-3.0). Only the integer
geometry lives here; the tensor work stays in the Director / the Clean Latent
Slice node. Pure stdlib so it unit-tests without torch or ComfyUI.

Geometry recap (LTXV stride-8 temporal latent):
  latent index of pixel frame f>0 : L(f) = 1 + (f - 1) // 8
  so pixel f = (clean_latent_frames + i) * 8  ->  L(f) = clean_latent_frames + i
which is exactly the i-th trailing latent slot. We append one reference per
trailing slot, then the video latent must be grown to make room for them.
"""
from __future__ import annotations

# LTXV temporal downscale stride. The Director's clean-latent math hardcodes
# the same 8, so the Ghost Mask geometry must agree with it.
LTXV_STRIDE = 8


def ghost_insert_frames(clean_latent_frames: int, n_refs: int, stride: int = LTXV_STRIDE) -> list[int]:
    """Pixel-space insert positions for ``n_refs`` Ghost Mask reference frames.

    Each reference pins the trailing latent slot ``clean_latent_frames + i``.
    Returns ``[(clean_latent_frames + i) * stride for i in range(n_refs)]``.
    """
    clf = int(clean_latent_frames)
    n = int(n_refs)
    s = int(stride)
    if clf < 0:
        raise ValueError(f"clean_latent_frames must be >= 0, got {clf}")
    if n < 0:
        raise ValueError(f"n_refs must be >= 0, got {n}")
    if s < 1:
        raise ValueError(f"stride must be >= 1, got {s}")
    return [(clf + i) * s for i in range(n)]


def ghost_total_latent_frames(clean_latent_frames: int, n_refs: int) -> int:
    """Total latent frame count once the reference slots are appended.

    The auto-generated video latent must be this long so the sampler produces
    the trailing reference slots the guide node pins; they are removed
    afterwards with the Clean Latent Slice node.
    """
    clf = int(clean_latent_frames)
    n = int(n_refs)
    if clf < 0 or n < 0:
        raise ValueError(f"counts must be >= 0, got clf={clf}, n_refs={n}")
    return clf + n


def clean_slice_bounds(size: int, start: int, length: int) -> tuple[int, int]:
    """Clamp a (start, length) request to a tensor axis of ``size`` frames.

    Mirrors the safety logic of the Clean Latent Slice node so it can be
    unit-tested without torch. Returns ``(actual_start, actual_length)`` such
    that ``0 <= actual_start < size`` (or 0 when empty) and the window never
    runs past the end. For Ghost Mask the caller passes start=0,
    length=clean_latent_frames to drop the trailing reference frames.
    """
    sz = int(size)
    if sz <= 0:
        return 0, 0
    st = max(0, int(start))
    st = min(st, sz - 1)
    ln = max(0, int(length))
    ln = min(ln, sz - st)
    return st, ln
