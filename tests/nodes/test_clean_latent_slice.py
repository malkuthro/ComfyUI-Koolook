"""Unit tests for the Clean Latent Slice node.

The clamp math (`_clamp_slice`) is stdlib and always tested. The torch.narrow
slice is exercised only when torch is importable (skipped in the stdlib CI
env, validated locally / on a torch machine). Loaded by file path to avoid
importing the package __init__ (which pulls in ComfyUI).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_MOD = Path(__file__).resolve().parents[2] / "k_clean_latent_slice.py"
_spec = importlib.util.spec_from_file_location("k_clean_latent_slice_under_test", _MOD)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
clamp = _m._clamp_slice


def _real_torch():
    """Skip unless a real torch is installed.

    Other node tests inject a bare ``types.ModuleType("torch")`` into
    ``sys.modules`` that persists for the session, so ``importorskip`` alone
    would find that hollow stub. Real torch carries ``__version__``; the stub
    does not.
    """
    torch = pytest.importorskip("torch")
    if not getattr(torch, "__version__", None):
        pytest.skip("real torch not installed (hollow stub present)")
    return torch


# --- _clamp_slice ----------------------------------------------------------

def test_clamp_keeps_clean_region():
    # 13 frames (10 clean + 3 refs), drop the refs.
    assert clamp(13, 0, 10) == (0, 10)


def test_clamp_overlong_length():
    assert clamp(10, 0, 999) == (0, 10)


def test_clamp_start_past_end():
    assert clamp(10, 50, 5) == (9, 1)


def test_clamp_empty_axis():
    assert clamp(0, 0, 5) == (0, 0)


def test_clamp_negatives_floored():
    assert clamp(10, -3, -2) == (0, 0)


# --- slice_latent (torch) --------------------------------------------------

def test_slice_latent_drops_trailing_frames():
    torch = _real_torch()
    node = _m.CleanLatentSlice()
    # [B, C, F, H, W] with F = 5 (3 clean + 2 refs)
    samples = torch.arange(5).float().reshape(1, 1, 5, 1, 1)
    (out,) = node.slice_latent({"samples": samples}, 0, 3)
    assert out["samples"].shape[2] == 3
    assert out["samples"].flatten().tolist() == [0.0, 1.0, 2.0]


def test_slice_latent_skips_broadcast_noise_mask():
    torch = _real_torch()
    node = _m.CleanLatentSlice()
    samples = torch.zeros(1, 1, 5, 1, 1)
    bmask = torch.ones(1, 1, 1, 1, 1)  # broadcasted, size-1 temporal axis
    (out,) = node.slice_latent({"samples": samples, "noise_mask": bmask}, 0, 3)
    assert out["samples"].shape[2] == 3
    assert out["noise_mask"].shape[2] == 1  # untouched
