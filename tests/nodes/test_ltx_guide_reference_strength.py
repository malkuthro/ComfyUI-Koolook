"""Unit tests for the LTX Guide Reference Strength helper.

Pure stdlib. Loaded by file path to avoid importing the package __init__
(which pulls in ComfyUI).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_MOD = Path(__file__).resolve().parents[2] / "k_ltx_guide_reference_strength.py"
_spec = importlib.util.spec_from_file_location("k_ltx_guide_reference_strength_under_test", _MOD)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
scale = _m.scaled_reference_strengths


def test_absolute_rewrites_only_trailing_refs():
    # 4 keyframes (0.8) + 2 refs (0.5); boost refs to 0.9, keyframes untouched.
    s = [0.8, 0.8, 0.8, 0.8, 0.5, 0.5]
    assert scale(s, 2, 0.9, "absolute") == [0.8, 0.8, 0.8, 0.8, 0.9, 0.9]


def test_multiply_scales_trailing_refs():
    s = [0.8, 0.8, 0.4, 0.4]
    assert scale(s, 2, 2.0, "multiply") == [0.8, 0.8, 0.8, 0.8]


def test_multiply_clamps_to_one():
    s = [0.8, 0.5]
    assert scale(s, 1, 5.0, "multiply") == [0.8, 1.0]


def test_zero_reference_count_is_noop():
    s = [0.8, 0.8, 0.8]
    assert scale(s, 0, 0.9, "absolute") == [0.8, 0.8, 0.8]


def test_reference_count_capped_at_length():
    s = [0.5, 0.5]
    assert scale(s, 9, 0.9, "absolute") == [0.9, 0.9]


def test_does_not_mutate_input():
    s = [0.8, 0.5]
    _ = scale(s, 1, 0.9, "absolute")
    assert s == [0.8, 0.5]


def test_node_passthrough_when_no_refs():
    node = _m.LTXGuideReferenceStrength()
    gd = {"strengths": [0.8, 0.8], "reference_count": 0}
    (out,) = node.apply(gd, 0.9, "absolute")
    assert out["strengths"] == [0.8, 0.8]


def test_node_boosts_refs_and_copies():
    node = _m.LTXGuideReferenceStrength()
    gd = {"strengths": [0.8, 0.5], "reference_count": 1, "images": ["x", "y"]}
    (out,) = node.apply(gd, 0.9, "absolute")
    assert out["strengths"] == [0.8, 0.9]
    assert gd["strengths"] == [0.8, 0.5]  # original untouched
    assert out["images"] == ["x", "y"]   # other keys preserved
