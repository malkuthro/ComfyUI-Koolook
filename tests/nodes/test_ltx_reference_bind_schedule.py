"""Unit tests for the LTX Reference Bind Schedule ramp + entry boosting.

The pure `ref_gain` ramp and `_boosted_entries` helper are stdlib-only and
always tested; the model wiring (unet wrapper + object-patch on
_build_guide_self_attention_mask) is validated by rendering against a live LTX
2.3 model. Loaded by file path to avoid importing the package __init__.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_MOD = Path(__file__).resolve().parents[2] / "k_ltx_reference_bind_schedule.py"
_spec = importlib.util.spec_from_file_location("k_ltx_reference_bind_schedule_under_test", _MOD)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
gain = _m.ref_gain
boost = _m._boosted_entries

SMAX = 10.0


# --- ref_gain ramp ---------------------------------------------------------

def test_early_steps_neutral():
    # high sigma == start of denoise -> progress 0 -> neutral 1.0
    assert gain(SMAX, SMAX, 0.5, 0.9, 2.0) == 1.0
    # still before ramp_start
    assert gain(SMAX * 0.6, SMAX, 0.5, 0.9, 2.0) == 1.0  # progress 0.4 < 0.5


def test_late_steps_reach_peak():
    # sigma -> 0 -> progress 1 -> peak
    assert gain(0.0, SMAX, 0.5, 0.9, 2.0) == 2.0
    assert gain(0.05 * SMAX, SMAX, 0.5, 0.9, 2.0) == 2.0  # progress 0.95 >= 0.9


def test_midpoint_between_one_and_peak():
    # progress 0.7 is the midpoint of [0.5, 0.9] -> smoothstep 0.5 -> 1.5
    g = gain(SMAX * 0.3, SMAX, 0.5, 0.9, 2.0)
    assert g == pytest.approx(1.5)


def test_monotonic_non_decreasing_as_sigma_falls():
    prev = 0.0
    for i in range(0, 21):
        s = SMAX * (1.0 - i / 20.0)
        g = gain(s, SMAX, 0.5, 0.9, 2.0)
        assert g >= prev - 1e-9
        prev = g


def test_degenerate_ramp_window():
    # ramp_end <= ramp_start -> step at ramp_end
    assert gain(0.0, SMAX, 0.9, 0.9, 3.0) == 3.0
    assert gain(SMAX, SMAX, 0.9, 0.9, 3.0) == 1.0


# --- _boosted_entries ------------------------------------------------------

def test_boost_scales_only_trailing_refs():
    entries = [{"strength": 1.0}, {"strength": 1.0}, {"strength": 1.0}]  # 2 kf + 1 ref
    out = boost(entries, 1, 2.0)
    assert [e["strength"] for e in out] == [1.0, 1.0, 2.0]


def test_boost_multiplies_existing_strength():
    entries = [{"strength": 0.8}, {"strength": 0.5}]
    out = boost(entries, 1, 2.0)
    assert out[0]["strength"] == 0.8
    assert out[1]["strength"] == 1.0


def test_boost_noop_when_gain_one():
    entries = [{"strength": 1.0}, {"strength": 1.0}]
    assert boost(entries, 1, 1.0) is entries  # untouched object


def test_boost_noop_when_no_refs():
    entries = [{"strength": 1.0}]
    assert boost(entries, 0, 2.0) is entries


def test_boost_does_not_mutate_input():
    entries = [{"strength": 1.0}, {"strength": 1.0}]
    _ = boost(entries, 1, 3.0)
    assert entries[1]["strength"] == 1.0


def test_boost_count_capped_at_length():
    entries = [{"strength": 1.0}, {"strength": 1.0}]
    out = boost(entries, 9, 2.0)
    assert [e["strength"] for e in out] == [2.0, 2.0]
