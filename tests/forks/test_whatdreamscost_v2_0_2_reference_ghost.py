"""Tests for the v2_0_2 Ghost Mask reference geometry helpers.

Lives in
``forks/whatdreamscost_koolook/versions/v2_0_2/reference_ghost.py`` and is
pure stdlib, so we load it by file path — the same isolation trick the other
fork tests use to avoid triggering the package ``__init__`` (which imports
``ltx_director`` -> ``comfy_api``, unavailable inside pytest).
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
    / "reference_ghost.py"
)
_spec = importlib.util.spec_from_file_location(
    "whatdreamscost_v2_0_2_reference_ghost", _MOD_PATH
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ghost_insert_frames = _mod.ghost_insert_frames
ghost_total_latent_frames = _mod.ghost_total_latent_frames
clean_slice_bounds = _mod.clean_slice_bounds


# --- ghost_insert_frames ---------------------------------------------------

def test_insert_frames_pin_trailing_latent_slots():
    # clean region is 10 latent frames; two refs -> latent slots 10 and 11.
    # pixel = slot * 8, and L(pixel) = 1 + (pixel-1)//8 must equal the slot.
    frames = ghost_insert_frames(10, 2)
    assert frames == [80, 88]
    for i, f in enumerate(frames):
        latent_idx = 1 + (f - 1) // 8
        assert latent_idx == 10 + i


def test_insert_frames_empty_when_no_refs():
    assert ghost_insert_frames(10, 0) == []


def test_insert_frames_custom_stride():
    assert ghost_insert_frames(3, 2, stride=4) == [12, 16]


def test_insert_frames_rejects_negative():
    with pytest.raises(ValueError):
        ghost_insert_frames(-1, 1)
    with pytest.raises(ValueError):
        ghost_insert_frames(1, -1)
    with pytest.raises(ValueError):
        ghost_insert_frames(1, 1, stride=0)


# --- ghost_total_latent_frames ---------------------------------------------

def test_total_latent_frames_adds_refs():
    assert ghost_total_latent_frames(10, 0) == 10
    assert ghost_total_latent_frames(10, 3) == 13


def test_total_latent_frames_rejects_negative():
    with pytest.raises(ValueError):
        ghost_total_latent_frames(-1, 0)
    with pytest.raises(ValueError):
        ghost_total_latent_frames(0, -1)


# --- clean_slice_bounds ----------------------------------------------------

def test_slice_bounds_drops_trailing_refs():
    # total 13 frames (10 clean + 3 refs); keep [0, 10).
    assert clean_slice_bounds(13, 0, 10) == (0, 10)


def test_slice_bounds_clamps_overlong_length():
    assert clean_slice_bounds(10, 0, 999) == (0, 10)


def test_slice_bounds_clamps_start_into_range():
    assert clean_slice_bounds(10, 50, 5) == (9, 1)


def test_slice_bounds_empty_tensor():
    assert clean_slice_bounds(0, 0, 5) == (0, 0)


def test_slice_bounds_negative_inputs_floored():
    assert clean_slice_bounds(10, -3, -2) == (0, 0)
