"""Tests for Easy_VideoCombine's absolute-path discrimination.

The full VHS encode pipeline needs ffmpeg + torch + a live ComfyUI
process to exercise meaningfully, so we don't drive ``combine_video()``
end-to-end here. The behavior worth pinning at unit-test scope is the
relative-vs-absolute branch and the ``create_path_if_missing`` toggle —
both live in the pure helper :func:`_resolve_abs_target`, which doesn't
need VHS to be installed.

Importing ``k_video_combine`` works regardless of VHS availability: the
module's top-level try/except shape leaves ``_resolve_abs_target``
defined either way (only the class registration is gated).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from k_video_combine import _resolve_abs_target


def test_relative_prefix_returns_none() -> None:
    """Relative prefixes are upstream's domain — helper signals passthrough."""
    assert _resolve_abs_target("AnimateDiff", create_path_if_missing=False) is None
    assert _resolve_abs_target("renders/clip01", create_path_if_missing=True) is None
    # A path with `..` is still relative — sandbox stays in force.
    assert _resolve_abs_target("../escape", create_path_if_missing=False) is None


def test_absolute_prefix_existing_dir(tmp_path: Path) -> None:
    """Absolute prefix with existing parent returns (dir, basename)."""
    prefix = str(tmp_path / "clipname")
    result = _resolve_abs_target(prefix, create_path_if_missing=False)

    assert result is not None
    abs_dir, abs_base = result
    assert Path(abs_dir) == tmp_path
    assert abs_base == "clipname"


def test_absolute_prefix_missing_dir_raises_when_create_off(tmp_path: Path) -> None:
    """Missing parent + create_path_if_missing=False -> FileNotFoundError.

    A typo in a render path should fail loudly, not silently spawn
    directories the user didn't intend.
    """
    prefix = str(tmp_path / "nested" / "deeper" / "clip")

    with pytest.raises(FileNotFoundError, match="Directory does not exist"):
        _resolve_abs_target(prefix, create_path_if_missing=False)

    assert not (tmp_path / "nested").exists()  # No silent mkdir.


def test_absolute_prefix_missing_dir_created_when_toggled(tmp_path: Path) -> None:
    """Missing parent + create_path_if_missing=True -> mkdir + resolve."""
    parent = tmp_path / "nested" / "deeper"
    prefix = str(parent / "clip")

    result = _resolve_abs_target(prefix, create_path_if_missing=True)

    assert result is not None
    abs_dir, abs_base = result
    assert parent.exists()
    assert Path(abs_dir) == parent
    assert abs_base == "clip"


def test_absolute_prefix_with_trailing_separator(tmp_path: Path) -> None:
    """Trailing separator is normalized; dirname becomes the basename.

    Matches spacepxl's SaveEXR convention — typing
    ``"E:/renders/shot01/"`` is interpreted as "use 'shot01' as the
    file root inside 'E:/renders/'."
    """
    renders_dir = tmp_path / "renders"
    renders_dir.mkdir()
    prefix = str(renders_dir) + os.sep

    result = _resolve_abs_target(prefix, create_path_if_missing=False)

    assert result is not None
    abs_dir, abs_base = result
    assert Path(abs_dir) == tmp_path
    assert abs_base == "renders"


def test_absolute_prefix_with_no_filename_component_raises(tmp_path: Path) -> None:
    """An absolute path with empty basename has no filename root to
    append the counter to — surface as a clear ValueError instead of
    silently writing files named ``_00001.mp4``."""
    root = tmp_path.anchor  # "/" on POSIX, "C:\\" on Windows.
    assert os.path.isabs(root)

    with pytest.raises(ValueError, match="no filename component"):
        _resolve_abs_target(root, create_path_if_missing=False)
