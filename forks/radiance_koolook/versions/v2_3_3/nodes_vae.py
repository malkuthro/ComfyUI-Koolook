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
# Modified by ComfyUI-Koolook 2026-05-04: ported the 12-color-space surface,
# `Compress (Log)` HDR mode (with soft-shoulder + log-highlight denoise),
# `latent_sampling` (sample / mean / mode), and a real alpha output, while
# keeping the 4K tile engine, .rhdr export, and inverse-tonemap dropped.
# Implementation lives mostly in `color_helpers.py`; this file is the node
# wiring + sequence-aware dispatch.

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

  This Koolook variant keeps upstream's color-management surface
  (12 source/target spaces including the 6 cinema log curves,
  `Compress (Log)` HDR mode with soft-shoulder + highlight denoise,
  `sample` / `mean` / `mode` posterior sampling, real alpha roundtrip)
  but routes the prepped tensor straight into `vae.encode()`, letting
  the underlying VAE deal with whatever rank / temporal layout it needs.

What changed from upstream's interface:
  - Dropped: tile_size, overlap, processing_mode (no tiling).
  - Dropped: inverse_tonemap, target_stops, .rhdr export, crop_padding
    (niche features tied to the upstream tiler / Radiance Viewer).
  - Kept: pixels, vae, source_space, exposure, alpha_handling, hdr_mode,
    latent_sampling.
  - Added: rank dispatch for 5-D video tensors (frame iteration with
    temporal-stack on encode, per-frame decode + concat on decode);
    5-D → 4-D output normalization mirroring stock VAEDecode; alpha
    surfaced as a real third IMAGE return.
"""

from typing import Any, Dict, Tuple

import torch

from .color_helpers import (
    EXTENDED_SOURCE_SPACES,
    EXTENDED_TARGET_SPACES,
    EXTENDED_LOG_SPACES,
    LATENT_SAMPLING_MODES,
    LOG_PROFILE_HDR_PARAMS,
    LOG_PROFILE_HDR_DEFAULT,
    to_linear,
    from_linear,
    tensor_srgb_to_linear,
    tensor_linear_to_srgb,
    soft_log_shoulder,
    denoise_log_highlights,
    encode_with_sampling_mode,
)


HDR_MODES = ["Clip (SDR)", "Soft Clip", "Compress (Log)", "Passthrough"]


# ───────────────────────────────────────────────────────────────────────
# Easy_hdr_VAE_encode (Koolook v2.3.3)
# ───────────────────────────────────────────────────────────────────────


class Easy_hdr_VAE_encode:
    """
    Slim, video-friendly Radiance-style VAE encoder.

    See module docstring for the rationale and the upstream attribution.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
                "source_space": (
                    EXTENDED_SOURCE_SPACES,
                    {
                        "default": "Linear",
                        "tooltip": "Color space of the input pixels. "
                        "Auto-converted to scene-linear Rec.709 before exposure "
                        "and HDR processing. Log curves (LogC3, LogC4, S-Log3, "
                        "V-Log, DaVinci Intermediate, Log3G10) and ACES variants "
                        "(ACEScg, ACES 2065-1) are decoded properly.",
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
                        "tooltip": "Exposure adjustment in stops, applied in "
                        "linear space (correct semantics regardless of input "
                        "color space).",
                    },
                ),
                "alpha_handling": (
                    ["Preserve", "Ignore"],
                    {
                        "default": "Preserve",
                        "tooltip": "Preserve: surface input alpha as the third "
                        "IMAGE output (zeros if input had no alpha). "
                        "Ignore: always emit zeros for alpha.",
                    },
                ),
                "hdr_mode": (
                    HDR_MODES,
                    {
                        "default": "Soft Clip",
                        "tooltip": "How to handle values outside [0,1] before "
                        "encoding. Clip (SDR): hard clamp. Soft Clip: tanh "
                        "rolloff above 0.85. Compress (Log): encode through a "
                        "cinema log curve (uses source_space if it is a log "
                        "space, otherwise ARRI LogC4) — pairs with the same "
                        "hdr_mode on decode for HDR-clean roundtrips. "
                        "Passthrough: no clamp.",
                    },
                ),
                "latent_sampling": (
                    LATENT_SAMPLING_MODES,
                    {
                        "default": "sample",
                        "tooltip": "How to sample from the VAE posterior. "
                        "'sample' is ComfyUI's default (random). 'mean' or "
                        "'mode' are deterministic — best for img2img where "
                        "you want minimum reconstruction noise.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING", "IMAGE")
    RETURN_NAMES = ("samples", "debug_info", "alpha")
    FUNCTION = "encode"
    CATEGORY = "Koolook/VAE"
    DESCRIPTION = (
        "Encode 32-bit Linear / ACEScg / cinema-log images or image "
        "sequences (video) to VAE latents. 12 source spaces, 4 HDR "
        "modes, deterministic sampling option, real alpha roundtrip. "
        "Skips upstream Radiance's 4K tile engine for compatibility "
        "with video VAEs (Wan 2.2, Hunyuan, CogVideoX, LTX)."
    )

    def encode(
        self,
        pixels: torch.Tensor,
        vae: Any,
        source_space: str = "Linear",
        exposure: float = 0.0,
        alpha_handling: str = "Preserve",
        hdr_mode: str = "Soft Clip",
        latent_sampling: str = "sample",
    ) -> Tuple[Dict[str, Any], str, torch.Tensor]:
        img = pixels.clone().float()

        if img.shape[-1] < 3:
            raise ValueError(
                f"Easy_hdr_VAE_encode (Koolook v2.3.3): input tensor must have "
                f">= 3 channels in the last dimension; got shape "
                f"{tuple(img.shape)}."
            )

        # Surface alpha as a separate IMAGE output. Real alpha if the input
        # had a 4th channel and the user opted to preserve it; otherwise a
        # zeros tensor of compatible shape so downstream wiring never fails.
        if alpha_handling == "Preserve" and img.shape[-1] >= 4:
            alpha_out = img[..., 3:4].clone()
        else:
            alpha_shape = img.shape[:-1] + (1,)
            alpha_out = torch.zeros(alpha_shape, dtype=img.dtype, device=img.device)

        # Raw source bypasses all color and HDR processing — the user knows
        # what they're feeding the VAE. Exposure still applies as a
        # straightforward multiplication on whatever the input bytes are.
        if source_space == "Raw":
            if exposure != 0.0:
                img = img * (2.0 ** exposure)
            img_for_vae = img
        else:
            # 1. Convert source → linear Rec.709 (rank-agnostic helper handles
            #    every supported space, including the 6 log curves and ACES
            #    variants, plus 4-channel inputs by routing only the first 3
            #    channels through the matrix / log decode).
            img = to_linear(img, source_space)

            # 2. Exposure in linear domain (correct semantics for "stops" no
            #    matter what the source space was).
            if exposure != 0.0:
                img = img * (2.0 ** exposure)

            # 3. HDR mode — produces the tensor that goes into vae.encode().
            #    Compress (Log) re-encodes through a cinema log curve and
            #    deliberately skips the linear→sRGB-gamma step (the VAE will
            #    learn to round-trip log-coded data, paired with the same
            #    hdr_mode on the decode side). The other three modes
            #    (Clip / Soft Clip / Passthrough) operate on sRGB-gamma,
            #    which is what most image VAEs are trained on.
            if hdr_mode == "Compress (Log)":
                log_space = (
                    source_space if source_space in EXTENDED_LOG_SPACES
                    else "ARRI LogC4"
                )
                img_for_vae = from_linear(img, log_space)
            else:
                img_gamma = tensor_linear_to_srgb(img)
                if hdr_mode == "Clip (SDR)":
                    img_for_vae = torch.clamp(img_gamma, 0.0, 1.0)
                elif hdr_mode == "Soft Clip":
                    knee = 0.85
                    over = (img_gamma - knee).clamp(min=0.0)
                    img_for_vae = torch.where(
                        img_gamma > knee,
                        knee + torch.tanh(over) * (1.0 - knee),
                        img_gamma,
                    )
                    img_for_vae = torch.clamp(img_for_vae, 0.0, 1.0)
                elif hdr_mode == "Passthrough":
                    img_for_vae = img_gamma  # no clamp — extended sRGB range
                else:
                    raise ValueError(f"Unknown hdr_mode: {hdr_mode!r}")

        # 4. Drop alpha for the VAE itself; alpha is surfaced separately above.
        img_rgb = img_for_vae[..., :3]

        # 5. Sequence-aware encode dispatch.
        # 3D-native video VAEs (Wan 2.2 / Hunyuan / CogVideoX / LTX) expose
        # `latent_dim == 3` and accept a 5-D (B, F, H, W, C) tensor directly.
        # 2D image VAEs given a 5-D tensor need per-frame iteration with a
        # temporal stack, matching upstream RadianceVAE4KEncode's behavior.
        is_video = img_rgb.ndim == 5
        is_3d_vae = (
            hasattr(vae, "latent_dim") and getattr(vae, "latent_dim") == 3
        )

        try:
            if is_video and not is_3d_vae:
                _, num_frames, _, _, _ = img_rgb.shape
                frame_latents = [
                    encode_with_sampling_mode(vae, img_rgb[:, fi, ...], latent_sampling)
                    for fi in range(num_frames)
                ]
                latent = torch.stack(frame_latents, dim=2)
            else:
                latent = encode_with_sampling_mode(vae, img_rgb, latent_sampling)
        except RuntimeError as exc:
            msg = str(exc)
            if "must match" in msg and "non-singleton dimension" in msg:
                raise RuntimeError(
                    f"Easy_hdr_VAE_encode (Koolook v2.3.3): vae.encode() failed "
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

        path = (
            "video-iter" if is_video and not is_3d_vae
            else "video-3d" if is_video
            else "image"
        )
        debug = (
            f"Easy_hdr_VAE_encode (Koolook v2.3.3) | input shape "
            f"{tuple(img_rgb.shape)} | path={path} | "
            f"source_space={source_space} | exposure={exposure:+.2f} | "
            f"hdr_mode={hdr_mode} | latent_sampling={latent_sampling} | "
            f"alpha_handling={alpha_handling}"
        )
        # Wrap in the standard ComfyUI LATENT dict so downstream samplers
        # (KSampler etc.) can do `latent["samples"]`.
        return ({"samples": latent}, debug, alpha_out)


# ───────────────────────────────────────────────────────────────────────
# Easy_hdr_VAE_decode (Koolook v2.3.3)
# ───────────────────────────────────────────────────────────────────────


class Easy_hdr_VAE_decode:
    """
    Slim, video-friendly Radiance-style VAE decoder.

    See module docstring for the rationale and the upstream attribution.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "target_space": (
                    EXTENDED_TARGET_SPACES,
                    {
                        "default": "Linear",
                        "tooltip": "Color space to convert the decoded image into. "
                        "Same 12 spaces as the encoder.",
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
                        "tooltip": "Exposure adjustment in stops, applied in "
                        "linear space for correctness.",
                    },
                ),
                "hdr_mode": (
                    HDR_MODES,
                    {
                        "default": "Clip (SDR)",
                        "tooltip": "Must match the encode setting. "
                        "Compress (Log): inverts log-curve encoding with "
                        "soft-shoulder + log-space highlight denoise for "
                        "clean HDR output. Soft Clip / Passthrough: extended "
                        "sRGB EOTF. Clip (SDR): standard sRGB EOTF.",
                    },
                ),
                "source_space": (
                    EXTENDED_SOURCE_SPACES,
                    {
                        "default": "Linear",
                        "tooltip": "Only used with hdr_mode='Compress (Log)' "
                        "to identify which log curve was used at encode. "
                        "Set this to whatever you set source_space to on "
                        "the encoder. Ignored for other hdr_modes.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "debug_info")
    FUNCTION = "decode"
    CATEGORY = "Koolook/VAE"
    DESCRIPTION = (
        "Decode VAE latents (image or video) directly to 32-bit "
        "Linear / ACEScg / cinema-log images. 12 target spaces, "
        "Compress (Log) HDR pipeline with soft-shoulder + log denoise. "
        "Rank-agnostic — works with whatever tensor shape the underlying "
        "VAE returns."
    )

    def decode(
        self,
        samples: Dict[str, Any],
        vae: Any,
        target_space: str = "Linear",
        exposure_adjust: float = 0.0,
        hdr_mode: str = "Clip (SDR)",
        source_space: str = "Linear",
    ) -> Tuple[torch.Tensor, str]:
        # Sequence dispatch mirrors the encode side: 5-D (B, C, F, latH, latW)
        # latents from a 2-D image VAE are decoded per-frame and concatenated
        # along batch so the output is a standard ComfyUI (B*F, H, W, C)
        # IMAGE batch — which is what SaveImage and friends expect to write
        # as `_00001.png`, `_00002.png`, …
        latent = samples["samples"]
        is_video = latent.ndim == 5
        is_3d_vae = (
            hasattr(vae, "latent_dim") and getattr(vae, "latent_dim") == 3
        )

        try:
            if is_video and not is_3d_vae:
                _, _, num_frames, _, _ = latent.shape
                frame_imgs = [
                    vae.decode(latent[:, :, fi, ...]) for fi in range(num_frames)
                ]
                img = torch.cat(frame_imgs, dim=0)
            else:
                img = vae.decode(latent)
        except RuntimeError as exc:
            msg = str(exc)
            if "must match" in msg and "non-singleton dimension" in msg:
                raise RuntimeError(
                    f"Easy_hdr_VAE_decode (Koolook v2.3.3): vae.decode() failed "
                    f"with a tensor-shape mismatch ({msg}). Latent shape was "
                    f"{tuple(latent.shape)}. Verify the latent was "
                    f"produced by the same VAE you're decoding with."
                ) from exc
            raise

        # Normalize 5-D `(B, F, H, W, C)` output to the standard ComfyUI 4-D
        # IMAGE shape `(B*F, H, W, C)`. 3-D-aware video VAEs (Wan 2.2,
        # Hunyuan, CogVideoX, LTX) return 5-D directly; without this reshape
        # the trailing IMAGE-typed nodes see a single weirdly-shaped image
        # (`count=1, height=F`) instead of an N-frame sequence. Mirrors the
        # stock VAEDecode node — see ComfyUI nodes.py:303-304.
        if img.ndim == 5:
            img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])

        img = img.float()

        # Color-space + HDR processing. "Raw" target_space bypasses everything
        # except a code-domain exposure tweak — useful for inspecting what
        # the VAE actually produced.
        if target_space == "Raw":
            if exposure_adjust != 0.0:
                img = img * (2.0 ** exposure_adjust)
        else:
            if hdr_mode == "Compress (Log)":
                # The VAE saw log-coded data; recover it cleanly with the
                # per-profile soft-shoulder + log-domain highlight denoise,
                # then decode log → linear with the same curve used at encode.
                log_space = (
                    source_space if source_space in EXTENDED_LOG_SPACES
                    else "ARRI LogC4"
                )
                params = LOG_PROFILE_HDR_PARAMS.get(log_space, LOG_PROFILE_HDR_DEFAULT)
                knee, ceiling, threshold, strength = params
                img = soft_log_shoulder(img, knee=knee, ceiling=ceiling)
                img = denoise_log_highlights(img, threshold=threshold, strength=strength)
                img = to_linear(img, log_space)
            else:
                # Standard pipeline: VAE output is in sRGB-gamma; linearize.
                img = tensor_srgb_to_linear(img)

            # Apply exposure in linear (correct semantics).
            if exposure_adjust != 0.0:
                img = img * (2.0 ** exposure_adjust)

            # Convert linear → target_space.
            img = from_linear(img, target_space)

        path = (
            "video-iter" if is_video and not is_3d_vae
            else "video-3d" if is_video
            else "image"
        )
        debug = (
            f"Easy_hdr_VAE_decode (Koolook v2.3.3) | output shape "
            f"{tuple(img.shape)} | path={path} | "
            f"target_space={target_space} | hdr_mode={hdr_mode} | "
            f"source_space={source_space} | "
            f"exposure_adjust={exposure_adjust:+.2f}"
        )
        return (img, debug)


# ───────────────────────────────────────────────────────────────────────
# Mappings consumed by the wrapper __init__.py (which applies the
# __koolook_v2_3_3 namespace suffix).
# ───────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "Easy_hdr_VAE_encode": Easy_hdr_VAE_encode,
    "Easy_hdr_VAE_decode": Easy_hdr_VAE_decode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Display names follow the same Title-Case "(Koolook)"-suffixed
    # convention as the rest of the pack so they surface together when
    # users type "koolook" in ComfyUI's node-add search.
    "Easy_hdr_VAE_encode": "Easy HDR VAE Encode (Koolook)",
    "Easy_hdr_VAE_decode": "Easy HDR VAE Decode (Koolook)",
}
