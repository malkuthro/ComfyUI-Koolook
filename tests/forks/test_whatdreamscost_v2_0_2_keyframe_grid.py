"""Tests for the v2_0_2 latent-grid keyframe snapper (issue #258).

The snapper lives in
``forks/whatdreamscost_koolook/versions/v2_0_2/keyframe_grid.py`` and is
deliberately pure stdlib, so we load it by file path — the same isolation
trick the other fork tests use to avoid triggering the package
``__init__`` (which imports ``ltx_director`` -> ``comfy_api``, unavailable
inside pytest).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_MOD_PATH = (
    Path(__file__).resolve().parents[2]
    / "forks"
    / "whatdreamscost_koolook"
    / "versions"
    / "v2_0_2"
    / "keyframe_grid.py"
)
_spec = importlib.util.spec_from_file_location(
    "whatdreamscost_v2_0_2_keyframe_grid", _MOD_PATH
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

latent_index = _mod.latent_index
bucket_center = _mod.bucket_center
latent_count_for_duration = _mod.latent_count_for_duration
snap = _mod.snap_keyframes_to_grid

S = 8  # LTXV temporal stride


# --- latent_index -----------------------------------------------------------

@pytest.mark.parametrize(
    "frame,expected",
    [
        (0, 0),    # causal first frame
        (-5, 0),   # clamp negatives to the first frame
        (1, 1),    # first frame of bucket 1
        (8, 1),    # last frame of bucket 1
        (9, 2),    # first frame of bucket 2
        (16, 2),
        (33, 5),
        (40, 5),
    ],
)
def test_latent_index(frame, expected):
    assert latent_index(frame, S) == expected


# --- bucket_center round-trips ----------------------------------------------

@pytest.mark.parametrize("i", range(0, 20))
def test_bucket_center_round_trips(i):
    # Any pin placed at a bucket's center maps back to that exact bucket.
    assert latent_index(bucket_center(i, S), S) == i


def test_bucket_center_values():
    assert bucket_center(0, S) == 0
    assert bucket_center(1, S) == 4
    assert bucket_center(5, S) == 36


# --- snapping ---------------------------------------------------------------

def test_frame_zero_passthrough():
    snapped, warns = snap([0], S, latent_length=20)
    assert snapped == [0]
    assert warns == []


def test_boundary_pin_snaps_to_center_same_latent_frame():
    # A pin near a bucket edge keeps its latent frame but moves to center.
    (snapped,), warns = snap([33], S, latent_length=20)
    assert latent_index(33, S) == latent_index(snapped, S) == 5
    assert snapped == 36
    assert warns == []


def test_distinct_buckets_untouched_latent_indices():
    frames = [0, 39, 86]  # the known-good 00016 layout
    snapped, warns = snap(frames, S, latent_length=20)
    assert [latent_index(f, S) for f in snapped] == [0, 5, 11]
    assert warns == []


def test_output_order_matches_input_order():
    frames = [86, 0, 39]  # deliberately unsorted
    snapped, _ = snap(frames, S, latent_length=20)
    # position-for-position, each output is the snap of the same input
    assert latent_index(snapped[0], S) == latent_index(86, S)
    assert snapped[1] == 0
    assert latent_index(snapped[2], S) == latent_index(39, S)


def test_collision_bumps_later_pin_and_warns():
    # Two pins in the same bucket (latent 1): 4 and 6 both -> L1.
    frames = [4, 6]
    snapped, warns = snap(frames, S, latent_length=20)
    indices = sorted(latent_index(f, S) for f in snapped)
    assert indices == [1, 2]  # second bumped to the next bucket
    assert len(warns) == 1
    assert "bumped" in warns[0]


def test_collision_exhaustion_warns_when_no_free_bucket():
    # latent_length=2 -> only latent indices {0, 1} exist. Three pins all in
    # bucket 1 cannot all fit; the last has nowhere to go.
    frames = [2, 3, 4]
    snapped, warns = snap(frames, S, latent_length=2)
    # everything clamps to the max index (1); at least one exhaustion warning
    assert all(latent_index(f, S) <= 1 for f in snapped)
    assert any("no free bucket" in w for w in warns)


def test_invalid_stride_raises():
    with pytest.raises(ValueError):
        snap([0, 8], 0, latent_length=20)


def test_latent_count_for_duration_uses_ltxv_ceiling_rule():
    # 120 is not 8n+1. LTXV rounds it to 121 pixel frames, i.e. 16 latent
    # frames, so the final bucket (index 15) is valid for snapping.
    assert latent_count_for_duration(120, S) == 16
    snapped, warns = snap([119], S, latent_length=latent_count_for_duration(120, S))
    assert snapped == [bucket_center(15, S)]
    assert latent_index(snapped[0], S) == 15
    assert warns == []


def test_latent_count_rejects_invalid_stride():
    with pytest.raises(ValueError):
        latent_count_for_duration(120, 0)


def test_empty_input():
    snapped, warns = snap([], S, latent_length=20)
    assert snapped == []
    assert warns == []


# --- expand_keyframe_ease: strength-ramped neighbor clusters ---

ease = _mod.expand_keyframe_ease


def test_ease_zero_is_passthrough():
    idxs, frames, strs = ease([0, 40], [1.0, 1.0], 8, 0, 0.5, 145)
    assert idxs == [0, 1]
    assert frames == [0, 40]
    assert strs == [1.0, 1.0]


def test_ease_keeps_centers_exact_first():
    idxs, frames, strs = ease([40], [1.0], 8, 1, 0.5, 145)
    # center emitted first, exact
    assert idxs[0] == 0 and frames[0] == 40 and strs[0] == 1.0


def test_ease_adds_ramped_neighbors_one_bucket_apart():
    idxs, frames, strs = ease([40], [1.0], 8, 1, 0.5, 145)
    # neighbors at 32 and 48, strength 0.5, reusing image index 0
    pairs = sorted((f, s) for f, s in zip(frames[1:], strs[1:]))
    assert pairs == [(32, 0.5), (48, 0.5)]
    assert all(i == 0 for i in idxs)


def test_ease_two_levels_falloff():
    idxs, frames, strs = ease([40], [1.0], 8, 2, 0.5, 145)
    m = {f: s for f, s in zip(frames, strs)}
    assert m[40] == 1.0
    assert m[32] == 0.5 and m[48] == 0.5     # k=1
    assert m[24] == 0.25 and m[56] == 0.25   # k=2


def test_ease_skips_out_of_range():
    # center at 0: left neighbor (-8) dropped, right (8) kept
    idxs, frames, strs = ease([0], [1.0], 8, 1, 0.5, 145)
    assert sorted(frames) == [0, 8]


def test_ease_skips_collision_with_other_center():
    # centers 40 and 48 (one bucket apart): 40's +8 == 48 (a center) -> skipped
    idxs, frames, strs = ease([40, 48], [1.0, 1.0], 8, 1, 0.5, 145)
    # both centers present
    assert frames.count(40) == 1 and frames.count(48) == 1
    # no duplicate frame entries
    assert len(frames) == len(set(frames))
