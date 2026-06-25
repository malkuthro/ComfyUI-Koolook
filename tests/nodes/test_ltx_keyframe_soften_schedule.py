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
soften_mask = _m.soften_denoise_mask
resolve = _m.resolve_soften_window

SMAX = 10.0


class _FakeNested:
    """Minimal NestedTensor stub: holds (video, audio) parts, exposes is_nested + unbind."""
    is_nested = True

    def __init__(self, parts):
        self.parts = tuple(parts)

    def unbind(self):
        return self.parts


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


# --- soften_denoise_mask: audio-safety on the A/V nested mask ---------------

def test_nested_mask_softens_video_only_audio_untouched():
    # NestedTensor((video_pin=0.2, audio_pin=0.3)). Only the video half moves.
    nested = _FakeNested((0.2, 0.3))
    out = soften_mask(nested, 0.5, nested_ctor=_FakeNested)
    video, audio = out.unbind()
    assert video == pytest.approx(0.6)   # 0.2 -> 0.2 + 0.8*0.5
    assert audio == pytest.approx(0.3)   # audio mask untouched (lip-sync safe)


def test_video_only_mask_softened_whole():
    # plain (non-nested) mask -> softened directly
    assert soften_mask(0.2, 0.5) == pytest.approx(0.6)


def test_soften_mask_noop_when_gain_zero():
    nested = _FakeNested((0.2, 0.3))
    assert soften_mask(nested, 0.0, nested_ctor=_FakeNested) is nested


def test_soften_mask_noop_when_none():
    assert soften_mask(None, 0.5) is None


# --- resolve_soften_window: step-fraction → schedule, curve-independent --------

def test_resolve_half_of_steps():
    # 4 steps (5 sigmas incl final 0); protect 0.5 -> cutoff 2 -> threshold sigmas[2]
    cutoff, thr, n = resolve([1.0, 0.9, 0.6, 0.3, 0.0], 0.5)
    assert (cutoff, n) == (2, 4)
    assert thr == pytest.approx(0.6)


def test_resolve_zero_protects_nothing():
    # cutoff 0 -> threshold == sigma_max -> downstream crossover-progress 0 -> no soften
    cutoff, thr, n = resolve([1.0, 0.9, 0.0], 0.0)
    assert cutoff == 0 and thr == pytest.approx(1.0) and n == 2


def test_resolve_full_protects_all():
    cutoff, thr, n = resolve([1.0, 0.9, 0.0], 1.0)
    assert cutoff == n == 2 and thr == pytest.approx(0.0)


def test_resolve_same_fraction_scales_with_step_count():
    # The whole point: 0.5 means "half the steps" on ANY step count.
    c4, _, n4 = resolve([1.0, 0.9, 0.6, 0.3, 0.0], 0.5)       # 4 steps
    c16, _, n16 = resolve([1.0 - i / 16 for i in range(17)], 0.5)  # 16 steps
    assert (c4, n4) == (2, 4)
    assert (c16, n16) == (8, 16)


def test_resolve_front_loaded_curve_reads_actual_sigma():
    # flat-top 16-step curve: protect 0.4 -> step 6, which is still on the plateau
    sig = [1.0] * 8 + [0.9, 0.8, 0.65, 0.5, 0.35, 0.2, 0.1, 0.05, 0.0]  # 17 vals, 16 steps
    cutoff, thr, n = resolve(sig, 0.4)
    assert n == 16 and cutoff == 6
    assert thr == pytest.approx(1.0)   # step 6 is still sigma 1.0 on this curve


def test_resolve_short_or_empty():
    assert resolve([1.0], 0.5) == (0, 1.0, 0)
    assert resolve([], 0.5) == (0, 0.0, 0)
