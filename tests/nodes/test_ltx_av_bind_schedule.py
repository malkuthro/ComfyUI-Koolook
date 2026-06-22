"""Unit tests for the LTX A/V Bind Schedule gain ramp.

Only the pure `a2v_gain` ramp is tested here — it's stdlib-only. The model
wiring (object-patch on audio_to_video_attn + the unet wrapper) is validated by
rendering against a live LTX 2.3 AV model, not in CI. Loaded by file path to
avoid importing the package __init__ (which pulls in ComfyUI).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_MOD = (
    Path(__file__).resolve().parents[2] / "k_ltx_av_bind_schedule.py"
)
_spec = importlib.util.spec_from_file_location("k_ltx_av_bind_schedule_under_test", _MOD)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
gain = _m.a2v_gain

SMAX = 10.0


def test_early_steps_use_early_gain():
    # sigma at the top of the schedule -> progress 0 -> early_gain
    assert gain(SMAX, SMAX, 0.55, 0.75, 0.0) == 0.0
    assert gain(SMAX, SMAX, 0.55, 0.75, 0.2) == pytest.approx(0.2)


def test_late_steps_full_bind():
    # sigma -> 0 -> progress 1 -> full audio binding
    assert gain(0.0, SMAX, 0.55, 0.75, 0.0) == 1.0
    assert gain(0.01 * SMAX, SMAX, 0.55, 0.75, 0.0) == 1.0


def test_monotonic_non_decreasing_as_sigma_falls():
    # walk sigma from max down to 0; gain must never decrease
    prev = -1.0
    for i in range(101):
        sigma = SMAX * (1.0 - i / 100.0)  # SMAX -> 0
        g = gain(sigma, SMAX, 0.55, 0.75, 0.0)
        assert g >= prev - 1e-9
        prev = g


def test_holds_early_gain_before_bind_start():
    # progress 0.3 (sigma=0.7*SMAX) is below bind_start=0.55 -> early_gain
    assert gain(0.7 * SMAX, SMAX, 0.55, 0.75, 0.1) == pytest.approx(0.1)


def test_full_at_and_after_bind_end():
    # progress 0.75 (sigma=0.25*SMAX) == bind_end -> 1.0
    assert gain(0.25 * SMAX, SMAX, 0.55, 0.75, 0.0) == 1.0


def test_smoothstep_midpoint_between_bounds():
    # midpoint of the ramp: progress 0.65 -> smoothstep(0.5)=0.5 -> halfway
    g = gain((1.0 - 0.65) * SMAX, SMAX, 0.55, 0.75, 0.0)
    assert g == pytest.approx(0.5, abs=1e-6)


def test_zero_sigma_max_is_safe():
    assert gain(5.0, 0.0, 0.55, 0.75, 0.0) == 1.0


def test_degenerate_bounds_step():
    # bind_end <= bind_start -> hard step at bind_end (progress >= 0.6 -> 1.0)
    assert gain(0.5 * SMAX, SMAX, 0.6, 0.6, 0.0) == 0.0   # progress 0.5 < 0.6
    assert gain(0.3 * SMAX, SMAX, 0.6, 0.6, 0.0) == 1.0   # progress 0.7 >= 0.6


def test_early_gain_one_is_stock_behavior():
    # early_gain=1.0 -> gain is 1.0 everywhere (node becomes a no-op)
    for frac in (0.0, 0.3, 0.6, 1.0):
        assert gain((1.0 - frac) * SMAX, SMAX, 0.55, 0.75, 1.0) == pytest.approx(1.0)
