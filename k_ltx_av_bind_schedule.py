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

That audio->video term makes the lips follow the audio — and also fights a hard
keyframe pose change when a transition lands on an audio peak (the jump). This
node scales ``a2v_out`` by a **gain** so we control where/when audio binds.

Two modes (selected by whether ``transition_frames`` is set):

* **Global ramp** (``transition_frames`` empty): gain ramps over the denoise
  from ``early_gain`` (high sigma — motion settles audio-blind) up to 1.0 (low
  sigma — lips bind). Decouples motion from lips, but lip-sync timing is also
  coarse motion, so holding audio out globally desyncs the mouth.

* **Transition windows** (``transition_frames`` set, e.g. "34,63,93" with
  ``total_frames``): audio binds at FULL strength everywhere (lips sync across
  the whole clip) EXCEPT in a ±``transition_window`` band around each keyframe
  transition, where the sigma ramp applies — so only those moments are
  motion-protected early and rejoined late. This breaks the motion/lip-sync
  trade-off: smooth transitions AND synced lips.

``early_gain=1.0`` (or no node) reproduces stock behavior.

NOTE: the pure ramp (`a2v_gain`) and per-frame mask (`windowed_frame_gains`) are
unit-tested; the model wiring is validated by rendering on a machine with the
LTX 2.3 AV model loaded (not in CI).
"""
from __future__ import annotations

import types
import logging

log = logging.getLogger(__name__)

_GAIN_KEY = "nghtdrp_a2v_gain"  # per-step sigma-ramp gain, set by the unet wrapper


def a2v_gain(sigma, sigma_max, bind_start=0.55, bind_end=0.75, early_gain=0.0):
    """Audio->video gain for the current ``sigma`` (the sigma ramp).

    Progress runs 0 at the start of denoise (``sigma == sigma_max``) to 1 at the
    end (``sigma -> 0``). Holds at ``early_gain`` until ``bind_start``, smoothsteps
    to 1.0 by ``bind_end`` (progress fractions in [0, 1]).
    """
    if sigma_max <= 0.0:
        return 1.0
    progress = 1.0 - (float(sigma) / float(sigma_max))
    progress = 0.0 if progress < 0.0 else (1.0 if progress > 1.0 else progress)
    if bind_end <= bind_start:
        return 1.0 if progress >= bind_end else early_gain
    if progress <= bind_start:
        return early_gain
    if progress >= bind_end:
        return 1.0
    t = (progress - bind_start) / (bind_end - bind_start)
    s = t * t * (3.0 - 2.0 * t)  # smoothstep
    return early_gain + (1.0 - early_gain) * s


def windowed_frame_gains(latent_frames, transition_idxs, window, sigma_gain):
    """Per-latent-frame a2v gain list.

    1.0 everywhere (full audio -> lips sync), dipping toward ``sigma_gain`` inside
    a ``+/-window`` band around each transition latent index, tapered smoothly so
    there's no hard edge. At a transition's center the gain is ``sigma_gain`` (so
    the sigma ramp still protects motion early / rejoins late there); at the band
    edge it's back to 1.0.
    """
    if latent_frames <= 0:
        return []
    window = max(0, int(window))
    gains = [1.0] * latent_frames
    if not transition_idxs:
        # no windows -> uniform sigma_gain (degenerates to the global ramp)
        return [float(sigma_gain)] * latent_frames
    for f in range(latent_frames):
        best = 1.0
        for ti in transition_idxs:
            dist = abs(f - int(ti))
            if dist <= window:
                if window == 0:
                    taper = 0.0
                else:
                    u = dist / window            # 0 center -> 1 edge
                    taper = u * u * (3.0 - 2.0 * u)  # smoothstep
                gv = sigma_gain + (1.0 - sigma_gain) * taper
                if gv < best:
                    best = gv
        gains[f] = best
    return gains


def _parse_frames(s):
    out = []
    for tok in str(s or "").replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def _make_a2v_scale_method(underlying, transition_fracs, window_frac, diag):
    """Bound-method wrapper around audio_to_video_attn.forward: scale its output
    by the per-step (and optionally per-frame) gain. ``diag`` is a shared 1-elem
    list used to log geometry once per render."""

    def wrapped(_self_module, *args, **kwargs):
        out = underlying(*args, **kwargs)
        to = kwargs.get("transformer_options", {}) or {}
        sigma_gain = float(to.get(_GAIN_KEY, 1.0))
        grid = to.get("grid_sizes", None)

        def _diag(applied):
            if not diag[0]:
                diag[0] = True
                log.info(
                    "[LTXAVBindSchedule] a2v fired: out.shape=%s grid_sizes=%s "
                    "transition_fracs=%s window_frac=%.4f -> %s",
                    tuple(getattr(out, "shape", ())), grid, transition_fracs,
                    window_frac, applied,
                )

        # Global mode (no transition windows): uniform scalar gain.
        if not transition_fracs:
            _diag("global scalar")
            return out if sigma_gain == 1.0 else out * sigma_gain

        # Windowed mode: need the latent grid to map tokens -> frames.
        try:
            latent_frames = int(grid[0])
            tpf = int(grid[1]) * int(grid[2])
        except Exception:
            _diag("FALLBACK scalar (no grid_sizes)")
            return out if sigma_gain == 1.0 else out * sigma_gain

        if latent_frames <= 0 or tpf <= 0 or out.shape[1] != latent_frames * tpf:
            _diag(f"FALLBACK scalar (shape mismatch: out[1]={out.shape[1]} vs "
                  f"{latent_frames}*{tpf}={latent_frames * tpf})")
            return out if sigma_gain == 1.0 else out * sigma_gain
        _diag(f"per-frame (latent_frames={latent_frames}, tpf={tpf})")

        ti = sorted({max(0, min(latent_frames - 1, round(fr * (latent_frames - 1))))
                     for fr in transition_fracs})
        wl = max(1, round(window_frac * latent_frames))
        fg = windowed_frame_gains(latent_frames, ti, wl, sigma_gain)
        if all(g == 1.0 for g in fg):
            return out
        # expand per-frame -> per-token (frame-major) and scale
        gain_tok = out.new_tensor(fg).repeat_interleave(tpf).view(1, -1, 1)
        return out * gain_tok

    return wrapped


class LTXAVBindSchedule:
    """Schedule LTX 2.3 audio->video binding across denoise and/or in time."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "LTX 2.3 audio-video model."}),
                "early_gain": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Audio->video strength in the early (high-sigma) steps "
                               "where big motion resolves. 0 = audio-blind motion; "
                               "1 = stock. In windowed mode this is the floor INSIDE "
                               "the transition bands only.",
                }),
                "bind_start": ("FLOAT", {
                    "default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoise progress (0=start,1=end) where audio begins "
                               "ramping in.",
                }),
                "bind_end": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoise progress where audio binding reaches full.",
                }),
            },
            "optional": {
                "transition_frames": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated keyframe transition frames (timeline "
                               "frames, e.g. '34,63,93'). Empty = global ramp (audio "
                               "gated everywhere). Set = WINDOWED: audio binds fully "
                               "everywhere except a band around these frames.",
                }),
                "total_frames": ("INT", {
                    "default": 0, "min": 0, "max": 100000, "step": 1,
                    "tooltip": "Clip length in timeline frames (e.g. 146). Required "
                               "for windowed mode to map frames -> latent grid.",
                }),
                "transition_window": ("INT", {
                    "default": 8, "min": 0, "max": 100000, "step": 1,
                    "tooltip": "Half-width (timeline frames) of the audio-protected "
                               "band around each transition.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Koolook/LTX"
    DESCRIPTION = (
        "Decouple big motion from lip-sync in LTX 2.3 by gating the audio->video "
        "cross-attention — globally by sigma, or (windowed) only around keyframe "
        "transitions so lips stay synced everywhere else. One pass, no re-noise."
    )

    def patch(self, model, early_gain, bind_start, bind_end,
              transition_frames="", total_frames=0, transition_window=8):
        m = model.clone()

        try:
            sigma_max = float(m.get_model_object("model_sampling").sigma_max)
        except Exception as e:  # pragma: no cover - live model
            log.warning("[LTXAVBindSchedule] no model_sampling.sigma_max: %s", e)
            sigma_max = 1.0

        # windowed-mode geometry (fractions are stride-independent)
        frames = _parse_frames(transition_frames)
        if frames and total_frames and total_frames > 0:
            transition_fracs = [min(1.0, max(0.0, f / float(total_frames))) for f in frames]
            window_frac = max(0.0, transition_window / float(total_frames))
            mode = f"windowed ({len(transition_fracs)} transitions, +/-{transition_window}f)"
        else:
            transition_fracs = []
            window_frac = 0.0
            mode = "global ramp"

        patched = 0
        diag = [False]  # shared one-shot diagnostic flag across blocks
        try:
            dm = m.get_model_object("diffusion_model")
            for idx, block in enumerate(getattr(dm, "transformer_blocks", None) or []):
                if getattr(block, "audio_to_video_attn", None) is None:
                    continue
                key = f"diffusion_model.transformer_blocks.{idx}.audio_to_video_attn.forward"
                underlying = m.get_model_object(key)
                wrapper = _make_a2v_scale_method(underlying, transition_fracs, window_frac, diag)
                m.add_object_patch(key, types.MethodType(wrapper, block.audio_to_video_attn))
                patched += 1
        except Exception as e:  # pragma: no cover - live model
            log.warning("[LTXAVBindSchedule] block patch failed: %s", e)

        if patched == 0:
            log.warning(
                "[LTXAVBindSchedule] no audio_to_video_attn blocks found — is this an "
                "LTX 2.3 audio-video model? Model passed through unchanged."
            )
            return (m,)

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
        log.info("[LTXAVBindSchedule] patched %d a2v blocks; mode=%s; early_gain=%.2f "
                 "bind=%.2f..%.2f", patched, mode, early_gain, bind_start, bind_end)
        return (m,)


NODE_CLASS_MAPPINGS = {"LTXAVBindSchedule": LTXAVBindSchedule}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXAVBindSchedule": "LTX A/V Bind Schedule (Koolook)"}

__all__ = [
    "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
    "a2v_gain", "windowed_frame_gains",
]
