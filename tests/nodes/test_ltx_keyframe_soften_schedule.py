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
steps = _m.steps_in_range

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


# --- steps_in_range: level (top % of sigma range) -> steps, for the info readout

def test_steps_top_range_counts_leading_steps():
    # sigmas start at 1.0; progress = 1 - sigma. top 25% = progress < 0.25.
    # 1.0(p0) 0.9(p.1) 0.8(p.2) 0.6(p.4)... -> first 3 steps are in the top 25%.
    count, n = steps([1.0, 0.9, 0.8, 0.6, 0.3, 0.0], 0.25)
    assert (count, n) == (3, 5)


def test_steps_zero_range_is_none():
    count, n = steps([1.0, 0.9, 0.0], 0.0)
    assert count == 0 and n == 2


def test_steps_full_range_covers_all():
    count, n = steps([1.0, 0.9, 0.5, 0.0], 1.0)
    assert count == n == 3


def test_steps_anchors_to_start_sigma_not_one():
    # schedule starts at 0.931 (not 1.0); progress is relative to that start.
    # top 20%: progress < 0.2 -> sigma > 0.931*0.8 = 0.745.
    count, n = steps([0.931, 0.811, 0.616, 0.346, 0.0], 0.20)
    assert (count, n) == (2, 4)   # 0.931, 0.811 in; 0.616 (p=0.34) out


def test_steps_flat_top_curve_spans_many_steps():
    # the 'top 3% covers ~7 steps' case: flat-ish front-loaded 16-step curve
    sig = [1.0, 0.999, 0.997, 0.994, 0.99, 0.985, 0.978, 0.969, 0.95,
           0.9, 0.8, 0.65, 0.5, 0.3, 0.15, 0.05, 0.0]
    count, n = steps(sig, 0.03)   # top 3% -> progress < 0.03 -> sigma > 0.97
    assert n == 16 and count == 7   # steps 0..6 have sigma >= 0.978; step 7 = 0.969 out


def test_steps_short_or_empty():
    assert steps([1.0], 0.5) == (0, 0)
    assert steps([], 0.5) == (0, 0)


# --- denoise_mask_function wiring: connected-SIGMAS fallback + honest coverage log


class _FakeModelPatcher:
    """Stub of the ModelPatcher surface patch() touches."""

    def __init__(self):
        self.model_options = {}
        self.mask_fn = None

    def clone(self):
        return self

    def set_model_denoise_mask_function(self, fn):
        self.mask_fn = fn


WIRED = [1.0, 0.9, 0.8, 0.6, 0.3, 0.0]   # 5 steps; top 25% covers the first 3


def _patched_fn(sigmas=WIRED):
    node = _m.LTXKeyframeSoftenSchedule()
    fake = _FakeModelPatcher()
    node.patch(fake, max_soften=0.5, soften_range=0.25, sigmas=sigmas)
    return fake.mask_fn


def test_runtime_sigmas_anchor_and_log_coverage(caplog):
    # a valid runtime schedule wins over the wired one, for anchor AND log
    fn = _patched_fn()
    with caplog.at_level("INFO", logger=_m.log.name):
        out = fn(1.0, 0.2, extra_options={"sigmas": [1.0, 0.5, 0.0]})
    assert out == pytest.approx(soften(0.2, 0.5))   # progress 0 -> full gain
    assert "first 1 of 2 steps" in caplog.text


def test_malformed_runtime_sigmas_fall_back_to_wired_for_anchor_and_log(caplog):
    # runtime sigmas present but empty: behavior anchors to the WIRED schedule,
    # and the coverage log must report the wired steps too — not "0 of 0".
    fn = _patched_fn()
    with caplog.at_level("INFO", logger=_m.log.name):
        out = fn(1.0, 0.2, extra_options={"sigmas": []})
    assert out == pytest.approx(soften(0.2, 0.5))
    assert "first 3 of 5 steps" in caplog.text
    assert "0 of 0" not in caplog.text


def test_zero_start_runtime_sigmas_fall_back_to_wired(caplog):
    # a degenerate runtime schedule (starts at 0) is unusable -> wired fallback
    fn = _patched_fn()
    with caplog.at_level("INFO", logger=_m.log.name):
        out = fn(1.0, 0.2, extra_options={"sigmas": [0.0, 0.0]})
    assert out == pytest.approx(soften(0.2, 0.5))
    assert "first 3 of 5 steps" in caplog.text


def test_no_usable_sigmas_anywhere_warns_once_and_passes_through(caplog):
    fn = _patched_fn(sigmas=None)
    with caplog.at_level("WARNING", logger=_m.log.name):
        out1 = fn(1.0, 0.2, extra_options={})
        out2 = fn(1.0, 0.2, extra_options={})
    assert out1 == 0.2 and out2 == 0.2   # mask untouched
    assert caplog.text.count("soften inactive") == 1
