# SPDX-License-Identifier: GPL-3.0-or-later
"""
Latent-grid keyframe snapping for the Koolook LTX Director fork.

Original ComfyUI-Koolook code (not derived from upstream WhatDreamsCost).
Authored 2026-06-22. License: GPL-3.0-or-later (matches the pack).

Background
----------
LTX encodes time with a temporal stride ``s`` (``scale_factors[0]`` at
runtime; 8 for LTXV). A guide image pinned at pixel frame ``f`` is injected
into a single latent frame at index::

    L(f) = 0                if f <= 0          # causal first frame
         = 1 + (f - 1)//s    otherwise

So a pin's effective time is quantized to ~``s``-frame buckets. Two
consequences this module addresses:

1. A pin dropped near a bucket boundary rounds ambiguously and can read as
   placed a whole bucket early/late. Snapping each pin to its bucket
   *center* removes the ambiguity without changing which latent frame it
   lands on (any pixel in the same bucket maps to the same ``L``).
2. Two pins inside one bucket collapse to the same latent frame and fight
   (the later overwrites the earlier). We bump the later pin to the next
   free bucket and surface a warning instead of silently clobbering.

This is a placement fix only: pose/phoneme *content* matching is the
animator's job; strength stays whatever the caller passes (no softening).
"""

from __future__ import annotations

from typing import List, Tuple


def latent_index(frame: int, stride: int) -> int:
    """Latent frame index that pixel ``frame`` lands in (causal first frame)."""
    if frame <= 0:
        return 0
    return 1 + (frame - 1) // stride


def bucket_center(index: int, stride: int) -> int:
    """Pixel frame at the center of latent bucket ``index``.

    ``latent_index(bucket_center(i, s), s) == i`` for every ``i >= 0``.
    Bucket 0 is the lone causal frame 0; bucket ``i >= 1`` spans pixels
    ``[1 + (i-1)*s, i*s]`` and its center is the lower-middle pixel.
    """
    if index <= 0:
        return 0
    return (index - 1) * stride + 1 + (stride - 1) // 2


def snap_keyframes_to_grid(
    insert_frames: List[int],
    stride: int,
    latent_length: int,
) -> Tuple[List[int], List[str]]:
    """Snap guide-image pixel positions to latent-bucket centers, 1 per bucket.

    Parameters
    ----------
    insert_frames:
        Pixel-frame positions of the image guides, in timeline order. Frame
        ``0`` (the identity/start pin on the causal first latent frame) is
        passed through untouched.
    stride:
        Temporal stride ``s`` (``scale_factors[0]``). Must be >= 1.
    latent_length:
        Number of latent frames available. Highest assignable latent index
        is ``latent_length - 1``; pins that cannot find a free bucket at or
        below it are left on their (clamped) target and a warning is added.

    Returns
    -------
    (snapped_frames, warnings)
        ``snapped_frames`` is the same length/order as ``insert_frames``.
        ``warnings`` is a (possibly empty) list of human-readable strings.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    warnings: List[str] = []
    max_index = max(0, latent_length - 1)

    # Resolve in chronological order so an earlier pin keeps its bucket and a
    # colliding later pin is the one that moves.
    order = sorted(range(len(insert_frames)), key=lambda i: insert_frames[i])

    used: set[int] = set()
    snapped: List[int] = [0] * len(insert_frames)

    for pos in order:
        f = insert_frames[pos]
        target = latent_index(f, stride)
        target = min(target, max_index)

        if target in used:
            bumped = target
            while bumped in used and bumped < max_index:
                bumped += 1
            if bumped in used:
                warnings.append(
                    f"keyframe at frame {f} collides on latent frame {target} "
                    f"and no free bucket remains below the clip end "
                    f"(latent_length={latent_length}, stride={stride}); "
                    f"too many pins for this duration — left overlapping."
                )
            else:
                warnings.append(
                    f"keyframe at frame {f} shared latent frame {target} with an "
                    f"earlier pin; bumped to latent frame {bumped} "
                    f"(~{bucket_center(bumped, stride)} px). Pins resolve at most "
                    f"one per {stride} frames."
                )
                target = bumped

        used.add(target)
        snapped[pos] = bucket_center(target, stride)

    return snapped, warnings


def expand_keyframe_ease(insert_frames, strengths, stride, ease, falloff, max_frame):
    """Expand each hard keyframe into a strength-ramped cluster (ease in/out).

    Keeps every center pin at its original (exact) strength, and adds neighbor
    pins one latent-bucket apart (``+/- k*stride`` pixel frames) at decreasing
    strength ``center * falloff**k`` for ``k in 1..ease``. The model is then
    softly pulled toward the pose approaching and leaving the locked frame
    instead of snapping to a dead stop — smoothing the "stop/dissolve/stop"
    without softening the exact pose at the center.

    Pure list math (frames/strengths only). Returns ``(idxs, frames, strengths)``
    parallel lists where ``idxs[i]`` indexes the ORIGINAL keyframe, so the caller
    reuses the same image tensor. Neighbors that fall off the clip or onto a
    frame already claimed by a center/neighbor are skipped (no clobbering).
    """
    n = len(insert_frames)
    if ease <= 0 or n == 0:
        return list(range(n)), list(insert_frames), [float(s) for s in strengths]

    stride = max(1, int(stride))
    claimed = set(int(f) for f in insert_frames)  # centers own their frames
    out_idx, out_f, out_s = [], [], []
    # centers first (exact)
    for i, (f, s) in enumerate(zip(insert_frames, strengths)):
        out_idx.append(i); out_f.append(int(f)); out_s.append(float(s))
    # then ramped neighbors
    for i, (f, s) in enumerate(zip(insert_frames, strengths)):
        f = int(f); s = float(s)
        for k in range(1, int(ease) + 1):
            ns = s * (float(falloff) ** k)
            for nf in (f - k * stride, f + k * stride):
                if nf < 0 or nf > max_frame:
                    continue
                if nf in claimed:
                    continue
                claimed.add(nf)
                out_idx.append(i); out_f.append(nf); out_s.append(ns)
    return out_idx, out_f, out_s
