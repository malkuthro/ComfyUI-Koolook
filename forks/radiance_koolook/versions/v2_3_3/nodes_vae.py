# SPDX-License-Identifier: GPL-3.0-or-later
#
# ComfyUI-Koolook — Radiance VAE Encode/Decode (Koolook v2.3.3 fork)
# Copyright (C) 2026 ComfyUI-Koolook contributors (kforgelabs).
#
# This file is part of ComfyUI-Koolook.
#
# ComfyUI-Koolook is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file at the repo
# root for the full text.
#
# Inspired by the API surface of `RadianceVAE4KEncode` / `RadianceVAE4KDecode`
# in fxtdstudios/radiance v2.3.3 (commit f262f47ddfda01ece154bf80c22769b1e4cef795,
# GPL-3.0). This implementation does not copy upstream's 4K cosine-blend tile
# engine — see forks/THIRD_PARTY.md for the full attribution + rationale.
#
# Modified by ComfyUI-Koolook on 2026-05-03 (initial port).

"""
Slim, video-friendly VAE Encode/Decode wrappers with HDR / color-space
awareness. Designed to work with image VAEs (SD1.5, SDXL, Flux) AND with
video VAEs (Wan 2.2, Hunyuan, CogVideoX, LTX) without a separate code path.

Why this exists:
  Upstream Radiance v2.3.3 ships RadianceVAE4KEncode/Decode, which run
  every input through a 4K cosine-blend tile engine. That engine errors
  with `size of tensor a (192) must match the size of tensor b (132)
  at non-singleton dimension 4` on Wan 2.2 video workflows, because
  the tiler's spatial layout doesn't agree with the video VAE's internal
  temporal-aware encoding. Most modern video VAEs already handle their
  own internal tiling/stitching, so the upstream tiler is redundant and
  actively breaks the chain.

  This Koolook variant keeps the upstream's color-management *intent*
  (linear / sRGB / ACEScg / raw, exposure adjust, HDR clamp) but routes
  the prepped tensor straight into vae.encode(), letting the underlying
  VAE deal with whatever rank / temporal layout it needs.

What changed from upstream's interface:
  - Dropped: tile_size, overlap, processing_mode (no tiling).
  - Dropped: latent_sampling (defer to ComfyUI's VAE.encode default).
  - Dropped: "Compress (Log)" hdr_mode (the log-curve compression family
    is upstream's most complex feature and is not required for video
    VAE encoding).
  - Kept: pixels, vae, source_space, exposure, alpha_handling, hdr_mode
    (with reduced choices: Clip / Soft Clip / Passthrough).
  - Added: a STRING `debug_info` second return that reports input shape
    and selected color path, mirroring the pattern from
    RadianceOCIOColorTransformV2 in versions/v1_0_1/.
"""

from typing import Any, Dict, Tuple

import torch


# ───────────────────────────────────────────────────────────────────────
# Local color-space helpers (kept inline so this module is self-contained).
# Copied conceptually from forks/radiance_koolook/versions/v1_0_1/nodes_hdr.py
# (which itself derives from upstream Radiance's color helpers, GPL-3.0).
# ───────────────────────────────────────────────────────────────────────


def _tensor_srgb_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    """sRGB EOTF → linear. Rank-agnostic; handles negative values."""
    sign = torch.sign(tensor)
    abs_t = torch.abs(tensor)
    linear = torch.where(
        abs_t <= 0.04045,
        abs_t / 12.92,
        torch.pow((abs_t + 0.055) / 1.055, 2.4),
    )
    return sign * linear


def _tensor_linear_to_srgb(tensor: torch.Tensor) -> torch.Tensor:
    """Linear → sRGB EOTF. Rank-agnostic; handles negative values."""
    sign = torch.sign(tensor)
    abs_t = torch.abs(tensor)
    srgb = torch.where(
        abs_t <= 0.0031308,
        abs_t * 12.92,
        1.055 * torch.pow(abs_t, 1.0 / 2.4) - 0.055,
    )
    return sign * srgb


def _apply_3x3_color_matrix(img: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """
    Apply a 3x3 color matrix to the first 3 channels of `img` (last dim),
    for any tensor rank (3D, 4D, 5D, ...). Channels beyond the first 3
    (e.g. alpha) are preserved unchanged.
    """
    if img.shape[-1] < 3:
        raise ValueError(
            f"Color matrix needs >= 3 channels in last dim; got shape "
            f"{tuple(img.shape)}."
        )
    rgb = img[..., :3]
    leading = rgb.shape[:-1]
    rgb_t = (rgb.reshape(-1, 3) @ matrix).reshape(*leading, 3)
    if img.shape[-1] == 3:
        return rgb_t
    out = img.clone()
    out[..., :3] = rgb_t
    return out


# ACEScg (AP1) → Linear sRGB (Rec.709). Standard ACES matrix, transposed for
# right-multiply on (..., 3) tensors.
_AP1_TO_REC709 = torch.tensor(
    [
        [1.6410, -0.3249, -0.2365],
        [-0.6636, 1.6153, 0.0168],
        [0.0117, -0.0084, 0.9884],
    ],
    dtype=torch.float32,
).T

# Linear sRGB (Rec.709) → ACEScg (AP1). Inverse of the above.
_REC709_TO_AP1 = torch.tensor(
    [
        [0.613097, 0.339523, 0.047379],
        [0.070194, 0.916354, 0.013452],
        [0.020616, 0.109570, 0.869815],
    ],
    dtype=torch.float32,
).T


# ───────────────────────────────────────────────────────────────────────
# RadianceVAEEncode (Koolook v2.3.3)
# ───────────────────────────────────────────────────────────────────────


class RadianceVAEEncode:
    """
    Slim, video-friendly Radiance-style VAE encoder.

    See module docstring for the rationale and the upstream attribution.
    """

    SOURCE_SPACES = ["Linear", "sRGB", "ACEScg", "Raw"]
    HDR_MODES = ["Clip (SDR)", "Soft Clip", "Passthrough"]

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
                "source_space": (
                    cls.SOURCE_SPACES,
                    {
                        "default": "Linear",
                        "tooltip": "Color space of the input pixels. "
                        "Auto-converted to sRGB-ish [0,1] before vae.encode().",
                    },
                ),
            },
            "optional": {
                "exposure": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Exposure adjustment in stops (linear space).",
                    },
                ),
                "alpha_handling": (
                    ["Preserve", "Ignore"],
                    {
                        "default": "Preserve",
                        "tooltip": "Whether to keep extra channels (e.g. alpha) "
                        "around for downstream nodes. The VAE itself only "
                        "encodes the first 3 channels regardless.",
                    },
                ),
                "hdr_mode": (
                    cls.HDR_MODES,
                    {
                        "default": "Soft Clip",
                        "tooltip": "How to handle values outside [0,1] before "
                        "encoding. Soft Clip applies a tanh rolloff above 0.85.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("samples", "debug_info")
    FUNCTION = "encode"
    CATEGORY = "Koolook/VFX"
    DESCRIPTION = (
        "Encode 32-bit Linear/ACEScg images or image sequences (video) to "
        "VAE latents. Skips upstream Radiance's 4K tile engine for "
        "compatibility with video VAEs (Wan 2.2, Hunyuan, CogVideoX, LTX)."
    )

    def encode(
        self,
        pixels: torch.Tensor,
        vae: Any,
        source_space: str = "Linear",
        exposure: float = 0.0,
        alpha_handling: str = "Preserve",
        hdr_mode: str = "Soft Clip",
    ) -> Tuple[Dict[str, Any], str]:
        img = pixels.clone().float()

        if img.shape[-1] < 3:
            raise ValueError(
                f"RadianceVAEEncode (Koolook v2.3.3): input tensor must have "
                f">= 3 channels in the last dimension; got shape "
                f"{tuple(img.shape)}."
            )

        # 1. Exposure adjust (rank-agnostic elementwise op).
        if exposure != 0.0:
            img = img * (2.0 ** exposure)

        # 2. Color-space conversion to sRGB Gamma — the standard VAE input space.
        if source_space == "Linear":
            img = _tensor_linear_to_srgb(img)
        elif source_space == "ACEScg":
            matrix = _AP1_TO_REC709.to(dtype=img.dtype, device=img.device)
            img = _apply_3x3_color_matrix(img, matrix)
            img = _tensor_linear_to_srgb(img)
        # "sRGB" -> already in gamma space, passthrough.
        # "Raw"  -> total passthrough.

        # 3. HDR handling.
        if hdr_mode == "Clip (SDR)":
            img = torch.clamp(img, 0.0, 1.0)
        elif hdr_mode == "Soft Clip":
            # Tanh rolloff above 0.85 — gives super-whites a natural curve
            # instead of a hard cutoff.
            knee = 0.85
            over = (img - knee).clamp(min=0.0)
            img = torch.where(img > knee, knee + torch.tanh(over) * (1.0 - knee), img)
            img = torch.clamp(img, 0.0, 1.0)
        # "Passthrough" -> no clamp (HDR experiments).

        # 4. Drop alpha for the VAE; alpha is preserved separately if requested.
        img_rgb = img[..., :3]
        if alpha_handling == "Preserve" and img.shape[-1] > 3:
            # We don't currently surface alpha back through LATENT — most
            # downstream samplers strip it anyway. The flag is kept so future
            # versions can add a second IMAGE return. For now, this is a noop
            # but logged in debug_info.
            pass

        # 5. Encode and surface clearer errors on shape mismatch.
        try:
            latent = vae.encode(img_rgb)
        except RuntimeError as exc:
            msg = str(exc)
            if "must match" in msg and "non-singleton dimension" in msg:
                raise RuntimeError(
                    f"RadianceVAEEncode (Koolook v2.3.3): vae.encode() failed "
                    f"with a tensor-shape mismatch ({msg}). Input shape was "
                    f"{tuple(img_rgb.shape)}. For video VAEs (Wan, Hunyuan, "
                    f"CogVideoX, LTX) this usually means another tensor in the "
                    f"model expects a different spatial resolution or frame "
                    f"count — verify that all image inputs (and any reference "
                    f"images / masks / keyframes) share the same H/W, and that "
                    f"the frame count meets the VAE's temporal stride "
                    f"requirement (e.g. Wan 2.2 expects frame counts compatible "
                    f"with its stride-of-4 latent)."
                ) from exc
            raise

        debug = (
            f"RadianceVAEEncode (Koolook v2.3.3) | input shape "
            f"{tuple(img_rgb.shape)} | source_space={source_space} | "
            f"exposure={exposure:+.2f} | hdr_mode={hdr_mode} | "
            f"alpha_handling={alpha_handling}"
        )
        return (latent, debug)


# ───────────────────────────────────────────────────────────────────────
# RadianceVAEDecode (Koolook v2.3.3)
# ───────────────────────────────────────────────────────────────────────


class RadianceVAEDecode:
    """
    Slim, video-friendly Radiance-style VAE decoder.

    See module docstring for the rationale and the upstream attribution.
    """

    TARGET_SPACES = ["Linear", "sRGB", "ACEScg", "Raw"]

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "target_space": (
                    cls.TARGET_SPACES,
                    {
                        "default": "Linear",
                        "tooltip": "Color space to convert the decoded image into.",
                    },
                ),
            },
            "optional": {
                "exposure_adjust": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Exposure adjustment in stops "
                        "(applied in linear space for correctness).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "debug_info")
    FUNCTION = "decode"
    CATEGORY = "Koolook/VFX"
    DESCRIPTION = (
        "Decode VAE latents (image or video) directly to 32-bit "
        "Linear/ACEScg/sRGB images. Rank-agnostic — works with whatever "
        "tensor shape the underlying VAE returns."
    )

    def decode(
        self,
        samples: Dict[str, Any],
        vae: Any,
        target_space: str = "Linear",
        exposure_adjust: float = 0.0,
    ) -> Tuple[torch.Tensor, str]:
        try:
            img = vae.decode(samples["samples"])
        except RuntimeError as exc:
            msg = str(exc)
            if "must match" in msg and "non-singleton dimension" in msg:
                raise RuntimeError(
                    f"RadianceVAEDecode (Koolook v2.3.3): vae.decode() failed "
                    f"with a tensor-shape mismatch ({msg}). Latent shape was "
                    f"{tuple(samples['samples'].shape)}. Verify the latent was "
                    f"produced by the same VAE you're decoding with."
                ) from exc
            raise

        img = img.float()

        if target_space == "Raw":
            if exposure_adjust != 0.0:
                img = img * (2.0 ** exposure_adjust)

        elif target_space == "sRGB":
            if exposure_adjust != 0.0:
                # Correctness: linearize → expose → re-gamma.
                linear = _tensor_srgb_to_linear(img)
                linear = linear * (2.0 ** exposure_adjust)
                img = _tensor_linear_to_srgb(linear)

        elif target_space == "Linear":
            img = _tensor_srgb_to_linear(img)
            if exposure_adjust != 0.0:
                img = img * (2.0 ** exposure_adjust)

        elif target_space == "ACEScg":
            linear = _tensor_srgb_to_linear(img)
            if exposure_adjust != 0.0:
                linear = linear * (2.0 ** exposure_adjust)
            matrix = _REC709_TO_AP1.to(dtype=linear.dtype, device=linear.device)
            img = _apply_3x3_color_matrix(linear, matrix)

        debug = (
            f"RadianceVAEDecode (Koolook v2.3.3) | output shape "
            f"{tuple(img.shape)} | target_space={target_space} | "
            f"exposure_adjust={exposure_adjust:+.2f}"
        )
        return (img, debug)


# ───────────────────────────────────────────────────────────────────────
# Mappings consumed by the wrapper __init__.py (which applies the
# __koolook_v2_3_3 namespace suffix).
# ───────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "RadianceVAEEncode": RadianceVAEEncode,
    "RadianceVAEDecode": RadianceVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceVAEEncode": "◆ Radiance VAE Encode (Video-friendly)",
    "RadianceVAEDecode": "◆ Radiance VAE Decode (Video-friendly)",
}
