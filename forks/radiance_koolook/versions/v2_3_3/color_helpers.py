# SPDX-License-Identifier: GPL-3.0-or-later
#
# ComfyUI-Koolook — color-space helpers for the Radiance v2.3.3 VAE fork.
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
# Adapted from fxtdstudios/radiance v2.3.3 (commit
# f262f47ddfda01ece154bf80c22769b1e4cef795, GPL-3.0):
#   * `color_utils.py` — log-curve conversions (LogC3, LogC4, S-Log3, V-Log,
#     DaVinci Intermediate, Log3G10) plus their EI parameter tables.
#   * `hdr/vae.py` — `_soft_log_shoulder`, `_denoise_log_highlights`,
#     `_encode_with_sampling_mode`, the LOG_PROFILE_HDR_PARAMS table, and the
#     EXTENDED_* color-space enumerations.
# All copied math is preserved verbatim with bug-fix history retained inline
# (the original docstrings document multiple curve-discontinuity fixes that
# are GPL-3.0-derived from upstream and we don't re-derive).
#
# Modified by ComfyUI-Koolook on 2026-05-04: extracted into a single helper
# module sized for the slim Easy_hdr_VAE_encode/decode wrapper, with the
# upstream's tile-engine, .rhdr export, and inverse-tonemap features
# intentionally dropped (see forks/THIRD_PARTY.md for the rationale).

"""
Rank-agnostic color-space helpers consumed by `nodes_vae.py`. All functions
operate on `(..., 3)` or `(..., 4)` tensors of any rank — image, batch, or
5-D video — and never touch the temporal axis. Drop-in replacement for the
inline helpers that previously lived at the top of `nodes_vae.py`.
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
#                       Color-space enumerations
# ═══════════════════════════════════════════════════════════════════════════

EXTENDED_SOURCE_SPACES = [
    "Linear",
    "ACEScg",
    "ACES 2065-1",
    "Rec.2020 Linear",
    "sRGB",
    "Raw",
    "ARRI LogC3",
    "ARRI LogC4",
    "Sony S-Log3",
    "Panasonic V-Log",
    "DaVinci Intermediate",
    "RED Log3G10",
]

EXTENDED_TARGET_SPACES = list(EXTENDED_SOURCE_SPACES)

EXTENDED_LOG_SPACES = {
    "ARRI LogC3",
    "ARRI LogC4",
    "Sony S-Log3",
    "Panasonic V-Log",
    "DaVinci Intermediate",
    "RED Log3G10",
}

LATENT_SAMPLING_MODES = ["sample", "mean", "mode"]


# ═══════════════════════════════════════════════════════════════════════════
#                       Color matrices (precomputed)
# ═══════════════════════════════════════════════════════════════════════════
# Stored once at import time, transposed for right-multiply on `(..., 3)`.
# At use, each is moved to the input tensor's device/dtype with `.to(...)`.

_AP1_TO_REC709 = torch.tensor([
    [1.7050509, -0.6217921, -0.0832588],
    [-0.1302564, 1.1408047, -0.0105483],
    [-0.0240033, -0.1289690, 1.1529723],
], dtype=torch.float32).T

_AP0_TO_REC709 = torch.tensor([
    [2.5216494, -1.1368885, -0.3847609],
    [-0.2752136, 1.3697052, -0.0944916],
    [-0.0159027, -0.1478148, 1.1637175],
], dtype=torch.float32).T

_REC2020_TO_REC709 = torch.tensor([
    [1.6604910, -0.5876411, -0.0728499],
    [-0.1245505, 1.1328999, -0.0083494],
    [-0.0181508, -0.1005789, 1.1187297],
], dtype=torch.float32).T

_REC709_TO_AP1 = torch.tensor([
    [0.613097, 0.339523, 0.047379],
    [0.070194, 0.916354, 0.013452],
    [0.020616, 0.109570, 0.869815],
], dtype=torch.float32).T

_REC709_TO_AP0 = torch.tensor([
    [0.4339316, 0.3762584, 0.1898100],
    [0.0886227, 0.8131989, 0.0981784],
    [0.0177087, 0.1095613, 0.8727300],
], dtype=torch.float32).T

_REC709_TO_REC2020 = torch.tensor([
    [0.6274039, 0.3292830, 0.0433131],
    [0.0690973, 0.9195404, 0.0113623],
    [0.0163914, 0.0880132, 0.8955954],
], dtype=torch.float32).T


def apply_3x3_color_matrix(img: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply a 3x3 color matrix to the first 3 channels of `img` (last dim),
    for any tensor rank. Channels beyond the first 3 (alpha) pass through.
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


# ═══════════════════════════════════════════════════════════════════════════
#                       sRGB EOTF (extended-range)
# ═══════════════════════════════════════════════════════════════════════════

def tensor_srgb_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    """sRGB EOTF → linear. Rank-agnostic; sign-preserving for negatives.
    Handles values >1 via continuous power-function extension (no clamp)."""
    sign = torch.sign(tensor)
    abs_t = torch.abs(tensor)
    linear = torch.where(
        abs_t <= 0.04045,
        abs_t / 12.92,
        torch.pow((abs_t + 0.055) / 1.055, 2.4),
    )
    return sign * linear


def tensor_linear_to_srgb(tensor: torch.Tensor) -> torch.Tensor:
    """Linear → sRGB EOTF. Rank-agnostic; sign-preserving for negatives."""
    sign = torch.sign(tensor)
    abs_t = torch.abs(tensor)
    srgb = torch.where(
        abs_t <= 0.0031308,
        abs_t * 12.92,
        1.055 * torch.pow(abs_t, 1.0 / 2.4) - 0.055,
    )
    return sign * srgb


# ═══════════════════════════════════════════════════════════════════════════
#                       Log curves (6 cinema profiles)
# ═══════════════════════════════════════════════════════════════════════════
# All curves are GPU-friendly torch.where dispatches between the linear toe
# and the log shoulder. Functions and constants are copied verbatim from
# upstream Radiance v2.3.3's color_utils.py (GPL-3.0); the bug-fix
# annotations in the docstrings reference upstream's history and are kept
# unchanged so the math stays auditable against upstream.

# ARRI LogC3 EI parameter table (cut, a, b, c, d, e, f) — verbatim from
# upstream `color_utils.LOGC3_EI_PARAMS`.
LOGC3_EI_PARAMS: Dict[int, Tuple[float, float, float, float, float, float, float]] = {
    160: (0.005561, 5.061087, 0.089004, 0.269035, 0.391007, 6.332427, 0.108361),
    200: (0.006208, 5.168208, 0.076621, 0.265275, 0.391007, 5.842037, 0.099519),
    250: (0.006871, 5.282072, 0.065521, 0.261620, 0.391007, 5.397270, 0.091111),
    320: (0.007622, 5.399335, 0.055194, 0.257766, 0.391007, 4.969419, 0.083295),
    400: (0.008318, 5.510883, 0.046585, 0.254174, 0.391007, 4.606965, 0.076257),
    500: (0.009031, 5.618393, 0.039023, 0.250758, 0.391007, 4.282556, 0.069776),
    640: (0.009840, 5.737055, 0.031538, 0.247070, 0.391007, 3.946374, 0.063409),
    800: (0.010591, 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809),
    1000: (0.011361, 5.944966, 0.019018, 0.240020, 0.391007, 3.369506, 0.051759),
    1280: (0.012235, 6.056760, 0.013804, 0.236500, 0.391007, 3.088156, 0.046447),
    1600: (0.013047, 6.161541, 0.009677, 0.233182, 0.391007, 2.852200, 0.041773),
    2000: (0.013901, 6.260724, 0.006210, 0.230014, 0.391007, 2.643126, 0.037413),
    2560: (0.014842, 6.362496, 0.002995, 0.226764, 0.391007, 2.440085, 0.033198),
    3200: (0.015711, 6.456037, 0.000295, 0.223740, 0.391007, 2.265605, 0.029493),
}


def tensor_linear_to_logc3(tensor: torch.Tensor, ei: int = 800) -> torch.Tensor:
    """ARRI LogC3 encoding. EI 800 is standard (Alexa Classic / Mini / SXT / LF)."""
    cut, a, b, c, d, e, f = LOGC3_EI_PARAMS.get(ei, LOGC3_EI_PARAMS[800])
    log_val = c * torch.log10(a * tensor + b) + d
    lin_val = e * tensor + f
    return torch.where(tensor > cut, log_val, lin_val)


def tensor_logc3_to_linear(tensor: torch.Tensor, ei: int = 800) -> torch.Tensor:
    cut, a, b, c, d, e, f = LOGC3_EI_PARAMS.get(ei, LOGC3_EI_PARAMS[800])
    cut_encoded = e * cut + f
    log_val = (torch.pow(10.0, (tensor - d) / c) - b) / a
    lin_val = (tensor - f) / e
    return torch.where(tensor > cut_encoded, log_val, lin_val)


def tensor_linear_to_logc4(tensor: torch.Tensor) -> torch.Tensor:
    """ARRI LogC4 encoding (Alexa 35)."""
    A = 4296.65
    D = 11.593
    slope = A / (D * 0.693147 * 64.0)
    log_val = (torch.log2(A * tensor + 64.0) - 6.0) / D
    lin_val = tensor * slope
    return torch.where(tensor > 0, log_val, lin_val)


def tensor_logc4_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    A = 4296.65
    D = 11.593
    slope = A / (D * 0.693147 * 64.0)
    log_val = (torch.pow(2.0, tensor * D + 6.0) - 64.0) / A
    lin_val = tensor / slope
    return torch.where(tensor > 0, log_val, lin_val)


def tensor_linear_to_slog3(tensor: torch.Tensor) -> torch.Tensor:
    """Sony S-Log3 encoding. Spec: S-Log3 Technical Note v1.1.

    Note: upstream BUG-1 fix preserved — the divisor `0.19` belongs INSIDE
    the log argument: `log10((x + 0.01) / 0.19) * 261.5`, not
    `log10(x + 0.01) / 0.19 * 261.5`.
    """
    cut = 0.011250
    log_val = (420.0 + torch.log10((tensor + 0.01) / 0.19) * 261.5) / 1023.0
    lin_val = (tensor + 0.01) * (171.2102946929 - 95.0) / (0.01125 * 1023.0) + (95.0 / 1023.0)
    return torch.where(tensor >= cut, log_val, lin_val)


def tensor_slog3_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    cut_v = 171.2102946929 / 1023.0
    log_val = 0.19 * torch.pow(10.0, (tensor * 1023.0 - 420.0) / 261.5) - 0.01
    lin_val = (tensor * 1023.0 - 95.0) / (4.0 * 261.5) - 0.01
    return torch.where(tensor >= cut_v, log_val, lin_val)


def tensor_linear_to_vlog(tensor: torch.Tensor) -> torch.Tensor:
    """Panasonic V-Log encoding."""
    cut = 0.01
    b, c, d = 0.00873, 0.241514, 0.598206
    log_val = c * torch.log10(tensor + b) + d
    lin_val = 5.625 * tensor + 0.125
    return torch.where(tensor >= cut, log_val, lin_val)


def tensor_vlog_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    """Panasonic V-Log decoding.

    Note: upstream BUG-2 fix preserved — `cut_encoded` is the encoded value
    of the linear cut (~0.181), NOT `c * 0.33893` (~0.082). Using the wrong
    threshold produces a visible kink in the dark/midtone range.
    """
    b, c, d = 0.00873, 0.241514, 0.598206
    cut_encoded = 0.181000  # = c * log10(0.01 + b) + d
    log_val = torch.pow(10.0, (tensor - d) / c) - b
    lin_val = (tensor - 0.125) / 5.625
    return torch.where(tensor >= cut_encoded, log_val, lin_val)


def tensor_linear_to_davinci_intermediate(tensor: torch.Tensor) -> torch.Tensor:
    """Blackmagic DaVinci Intermediate (DWG) encoding.

    Note: upstream BUG-5 fix preserved — the linear toe uses C1-continuous
    slope (~3.144) and intercept (~0.346) so the curve is continuous at the
    cut point. Earlier versions used slope=7.0 / intercept=0.073 which
    produced a 0.262-magnitude discontinuity.
    """
    A, C = 0.0075, 0.07329248
    cut = 0.00262409
    _slope = 3.14403760
    _intercept = 0.34555736
    log_val = torch.log10(tensor + A) * C + 0.5
    lin_val = _slope * tensor + _intercept
    return torch.where(tensor >= cut, log_val, lin_val)


def tensor_davinci_intermediate_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    A, C = 0.0075, 0.07329248
    cut_v = 0.353808  # encode(0.00262409)
    _slope = 3.14403760
    _intercept = 0.34555736
    log_val = torch.pow(10.0, (tensor - 0.5) / C) - A
    lin_val = (tensor - _intercept) / _slope
    return torch.where(tensor > cut_v, log_val, lin_val)


def tensor_linear_to_log3g10(tensor: torch.Tensor) -> torch.Tensor:
    """RED Log3G10 encoding."""
    a, b, c = 0.224282, 155.975327, 0.01
    cut = 0.01
    cut_encoded = 0.101551  # = a * log10(b * cut + 1) + c
    log_val = a * torch.log10(b * tensor + 1.0) + c
    lin_val = (tensor / cut) * cut_encoded
    return torch.where(tensor > cut, log_val, lin_val)


def tensor_log3g10_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    a, b, c = 0.224282, 155.975327, 0.01
    cut = 0.01
    cut_encoded = 0.101551
    log_val = (torch.pow(10.0, (tensor - c) / a) - 1.0) / b
    lin_val = tensor / cut_encoded * cut
    return torch.where(tensor > cut_encoded, log_val, lin_val)


# Dispatch table for compactness in nodes_vae.py — selected by space name.
_LOG_LINEAR_TO_ENCODED = {
    "ARRI LogC3":           tensor_linear_to_logc3,
    "ARRI LogC4":           tensor_linear_to_logc4,
    "Sony S-Log3":          tensor_linear_to_slog3,
    "Panasonic V-Log":      tensor_linear_to_vlog,
    "DaVinci Intermediate": tensor_linear_to_davinci_intermediate,
    "RED Log3G10":          tensor_linear_to_log3g10,
}

_LOG_ENCODED_TO_LINEAR = {
    "ARRI LogC3":           tensor_logc3_to_linear,
    "ARRI LogC4":           tensor_logc4_to_linear,
    "Sony S-Log3":          tensor_slog3_to_linear,
    "Panasonic V-Log":      tensor_vlog_to_linear,
    "DaVinci Intermediate": tensor_davinci_intermediate_to_linear,
    "RED Log3G10":          tensor_log3g10_to_linear,
}


def to_linear(img: torch.Tensor, source_space: str) -> torch.Tensor:
    """Convert any supported color space to scene-linear Rec.709.
    `source_space` must be one of EXTENDED_SOURCE_SPACES.
    """
    if source_space == "Linear":
        return img
    if source_space == "Raw":
        return img
    if source_space == "sRGB":
        return tensor_srgb_to_linear(img)
    if source_space == "ACEScg":
        m = _AP1_TO_REC709.to(dtype=img.dtype, device=img.device)
        return apply_3x3_color_matrix(img, m)
    if source_space == "ACES 2065-1":
        m = _AP0_TO_REC709.to(dtype=img.dtype, device=img.device)
        return apply_3x3_color_matrix(img, m)
    if source_space == "Rec.2020 Linear":
        m = _REC2020_TO_REC709.to(dtype=img.dtype, device=img.device)
        return apply_3x3_color_matrix(img, m)
    if source_space in _LOG_ENCODED_TO_LINEAR:
        decoder = _LOG_ENCODED_TO_LINEAR[source_space]
        # Preserve alpha for 4-channel inputs by routing only first 3 channels.
        if img.shape[-1] > 3:
            rgb = decoder(img[..., :3])
            return torch.cat([rgb, img[..., 3:]], dim=-1)
        return decoder(img)
    raise ValueError(f"Unknown source_space: {source_space!r}")


def from_linear(img: torch.Tensor, target_space: str) -> torch.Tensor:
    """Convert scene-linear Rec.709 to any supported target space."""
    if target_space == "Linear":
        return img
    if target_space == "Raw":
        return img
    if target_space == "sRGB":
        return tensor_linear_to_srgb(img)
    if target_space == "ACEScg":
        m = _REC709_TO_AP1.to(dtype=img.dtype, device=img.device)
        return apply_3x3_color_matrix(img, m)
    if target_space == "ACES 2065-1":
        m = _REC709_TO_AP0.to(dtype=img.dtype, device=img.device)
        return apply_3x3_color_matrix(img, m)
    if target_space == "Rec.2020 Linear":
        m = _REC709_TO_REC2020.to(dtype=img.dtype, device=img.device)
        return apply_3x3_color_matrix(img, m)
    if target_space in _LOG_LINEAR_TO_ENCODED:
        encoder = _LOG_LINEAR_TO_ENCODED[target_space]
        if img.shape[-1] > 3:
            rgb = encoder(img[..., :3])
            return torch.cat([rgb, img[..., 3:]], dim=-1)
        return encoder(img)
    raise ValueError(f"Unknown target_space: {target_space!r}")


# ═══════════════════════════════════════════════════════════════════════════
#                       HDR helpers (soft shoulder + denoise)
# ═══════════════════════════════════════════════════════════════════════════

# Per-profile adaptive HDR parameters for `Compress (Log)` decode.
# Format: {space_name: (shoulder_knee, shoulder_ceiling, denoise_threshold, denoise_strength)}
# Tuned per-curve based on the slope of log→linear at code 1.0:
#   • Steeper curves amplify VAE noise more aggressively after decompression.
#   • So they get a lower knee, tighter ceiling, lower denoise threshold,
#     and higher denoise strength. See upstream commit notes for the table
#     derivation; verbatim from upstream Radiance v2.3.3.
LOG_PROFILE_HDR_PARAMS: Dict[str, Tuple[float, float, float, float]] = {
    "ARRI LogC4":           (0.96, 1.08, 0.80, 0.55),  # moderate-steep
    "Sony S-Log3":          (0.96, 1.08, 0.80, 0.55),  # moderate-steep
    "ARRI LogC3":           (0.95, 1.06, 0.78, 0.60),  # steep
    "Panasonic V-Log":      (0.95, 1.07, 0.78, 0.60),  # steep
    "DaVinci Intermediate": (0.93, 1.05, 0.75, 0.70),  # very steep
    "RED Log3G10":          (0.92, 1.04, 0.72, 0.75),  # extremely steep
}

LOG_PROFILE_HDR_DEFAULT = (0.96, 1.08, 0.80, 0.55)  # safe middle-ground (LogC4 numbers)


def soft_log_shoulder(
    img: torch.Tensor, knee: float = 0.96, ceiling: float = 1.08
) -> torch.Tensor:
    """Soft-shoulder compressor for log-coded VAE output.

    Replaces the naive hard clamp that destroys legitimate super-white signal.
    Values in [0, knee] pass through unchanged; values above knee are
    smoothly compressed toward `ceiling` via tanh; sub-zero values are
    floored at zero (no valid signal below 0 in log-coded space).
    """
    img = torch.clamp(img, min=0.0)
    above = img > knee
    if above.any():
        result = img.clone()
        rng = ceiling - knee
        excess = (img[above] - knee) / rng
        result[above] = knee + rng * torch.tanh(excess)
        return result
    return img


def denoise_log_highlights(
    img: torch.Tensor, threshold: float = 0.80, strength: float = 0.6
) -> torch.Tensor:
    """Spatially-adaptive 3x3 box smooth weighted by code-value proximity to
    the highlight region. Midtones pass through untouched; only the noisy
    highlight region gets smoothed. Operates on (B, H, W, C) — for 5-D
    video inputs the caller must collapse the temporal axis into batch
    before calling, or we error out gracefully on rank.
    """
    if img.ndim != 4 or img.shape[1] < 3 or img.shape[2] < 3:
        return img  # too small / wrong rank — skip
    ramp = torch.clamp((img - threshold) / (1.0 - threshold + 1e-8), 0.0, 1.0)
    blend_alpha = strength * ramp * ramp  # quadratic ramp for natural falloff
    x = img.permute(0, 3, 1, 2)  # BHWC → BCHW for F.avg_pool2d
    x_padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
    smoothed = F.avg_pool2d(x_padded, kernel_size=3, stride=1, padding=0)
    smoothed = smoothed.permute(0, 2, 3, 1)  # back to BHWC
    return img + blend_alpha * (smoothed - img)


# ═══════════════════════════════════════════════════════════════════════════
#                       Latent sampling mode (sample / mean / mode)
# ═══════════════════════════════════════════════════════════════════════════

def encode_with_sampling_mode(
    vae: Any, pixels: torch.Tensor, mode: str
) -> torch.Tensor:
    """Encode pixels with explicit posterior-sampling control.

    mode = "sample" — random sample from posterior (default ComfyUI).
    mode = "mean"   — use posterior mean (deterministic; lowest-noise for img2img).
    mode = "mode"   — same as mean for Gaussian posteriors.

    The "mean" / "mode" path replicates ComfyUI's encode preprocessing
    (BHWC → BCHW, [0,1] → [-1,1]) before reaching into `first_stage_model`
    so we get the same pre-scaling. Falls back to standard sample if the
    raw model interface isn't available — never crashes.
    """
    if mode == "sample":
        result = vae.encode(pixels)
        if isinstance(result, dict):
            return result["samples"]
        return result

    # Try ComfyUI's exposed raw-encode methods if present.
    for method_name in ("encode_raw", "encode_to_distribution"):
        fn = getattr(vae, method_name, None)
        if callable(fn):
            try:
                posterior = fn(pixels)
                if hasattr(posterior, "mean"):
                    return posterior.mean
                if hasattr(posterior, "mode"):
                    return posterior.mode()
                if isinstance(posterior, torch.Tensor):
                    return posterior
            except Exception:
                pass

    # Reach into first_stage_model with ComfyUI-equivalent preprocessing.
    try:
        fsm = vae.first_stage_model
        device = next(fsm.parameters()).device
        x = pixels.movedim(-1, 1).to(device=device, dtype=torch.float32)
        x = x * 2.0 - 1.0
        posterior = fsm.encode(x)
        if hasattr(posterior, "latent_dist"):
            posterior = posterior.latent_dist
        if hasattr(posterior, "mean"):
            latent = posterior.mean
        elif hasattr(posterior, "mode"):
            latent = posterior.mode() if callable(posterior.mode) else posterior.mode
        elif isinstance(posterior, torch.Tensor):
            latent = posterior
        else:
            raise TypeError(f"Unknown posterior type: {type(posterior)}")
        scale_factor = getattr(vae, "scale_factor", None)
        if scale_factor is None:
            config = getattr(vae, "config", None)
            if config is not None:
                scale_factor = getattr(config, "scaling_factor", None)
        if scale_factor is not None and isinstance(scale_factor, (int, float)):
            latent = latent * scale_factor
        return latent
    except Exception:
        # Fall back to standard sample — always works.
        result = vae.encode(pixels)
        if isinstance(result, dict):
            return result["samples"]
        return result
