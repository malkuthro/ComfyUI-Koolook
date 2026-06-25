"""Unit tests for the LTX Keyframe Soften Schedule ramp.

The pure ``soften_gain`` / ``soften_mask_value`` helpers are stdlib-only and
always tested; the model wiring (the denoise_mask_function set on the patcher,
which softens the video keyframe pins early and restores them by the crossover)
is validated by rendering against a live LTX 2.3 model. Loaded by file path to
avoid importing the package __init__.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_MOD = Path(__file__).resolve().parents[2] / "k_ltx_keyframe_soften_schedule.py"
_spec = importlib.util.spec_from_file_location("k_ltx_keyframe_soften_schedule_under_test", _MOD)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
gain = _m.soften_gain
soften = _m.soften_mask_value

SMAX = 10.0


# --- soften_gain ramp (max early, 0 by crossover) --------------------------

def test_max_soften_at_denoise_start():
    # sigma == sigma_max -> progress 0 -> full max_soften
    assert gain(SMAX, SMAX, 0.5, 0.6) == pytest.approx(0.6)


def test_zero_after_crossover():
    # sigma -> 0 (progress 1) is past the crossover -> fully restored (0)
    assert gain(0.0, SMAX, 0.5, 0.6) == 0.0
    # exactly at the crossover -> 0
    assert gain(SMAX * 0.5, SMAX, 0.5, 0.6) == 0.0


def test_midpoint_of_ramp():
    # progress 0.25 is the midpoint of [0, 0.5] -> smoothstep 0.5 -> half max
    assert gain(SMAX * 0.75, SMAX, 0.5, 0.6) == pytest.approx(0.3)


def test_monotonic_non_increasing_as_sigma_falls():
    prev = 1.0
    for i in range(0, 21):
        s = SMAX * (1.0 - i / 20.0)
        g = gain(s, SMAX, 0.55, 0.6)
        assert g <= prev + 1e-9
        prev = g


def test_max_soften_zero_is_noop():
    assert gain(SMAX, SMAX, 0.55, 0.0) == 0.0


def test_crossover_zero_is_noop():
    assert gain(SMAX, SMAX, 0.0, 0.6) == 0.0


def test_degenerate_sigma_max():
    assert gain(1.0, 0.0, 0.5, 0.6) == 0.0


# --- soften_mask_value (raise a pin toward 1) ------------------------------

def test_gain_zero_keeps_exact_pin():
    assert soften(0.2, 0.0) == pytest.approx(0.2)


def test_gain_one_fully_frees():
    assert soften(0.2, 1.0) == pytest.approx(1.0)


def test_free_frame_stays_free():
    # a video frame already at 1.0 is unaffected by any gain
    assert soften(1.0, 0.7) == pytest.approx(1.0)


def test_partial_soften():
    # 0.2 pin, gain 0.5 -> 0.2 + 0.8*0.5 = 0.6
    assert soften(0.2, 0.5) == pytest.approx(0.6)
