"""Tests for Easy_LoadVideo's split input path composition."""
from __future__ import annotations

from pathlib import Path

import pytest

from k_video_load import _compose_input_video_path, _normalize_text_input


def test_empty_input_path_passes_video_through() -> None:
    assert _compose_input_video_path("E:/plates/shot01.mp4", "") == "E:/plates/shot01.mp4"
    assert _compose_input_video_path("https://example.test/clip.mp4", "") == "https://example.test/clip.mp4"


def test_empty_input_path_accepts_directory_and_filename_lines() -> None:
    root = Path.cwd().anchor
    absolute_dir = str(Path(root) / "projects" / "shot01")
    composed = _compose_input_video_path(
        f"{absolute_dir}\nplate_main.mp4",
        "",
    )
    assert Path(composed) == Path(absolute_dir) / "plate_main.mp4"


def test_absolute_input_path_joins_filename() -> None:
    # Use a real tmp_path so the absolute-path branch is platform-neutral.
    root = Path.cwd().anchor
    absolute_dir = str(Path(root) / "projects" / "shot01")
    composed = _compose_input_video_path(
        "plate_main.mp4",
        absolute_dir,
    )
    assert Path(composed) == Path(absolute_dir) / "plate_main.mp4"


def test_relative_input_path_roots_under_comfy_input_dir(tmp_path: Path) -> None:
    composed = _compose_input_video_path(
        "plate_main.mp4",
        "shots/shot01",
        input_root=str(tmp_path),
    )
    assert Path(composed) == tmp_path / "shots" / "shot01" / "plate_main.mp4"


def test_input_path_strips_video_path_components(tmp_path: Path) -> None:
    composed = _compose_input_video_path(
        "accidental/subdir/plate_main.mp4",
        "shots/shot01",
        input_root=str(tmp_path),
    )
    assert Path(composed) == tmp_path / "shots" / "shot01" / "plate_main.mp4"


def test_input_path_strips_windows_video_path_components(tmp_path: Path) -> None:
    composed = _compose_input_video_path(
        "accidental\\subdir\\plate_main.mp4",
        "shots/shot01",
        input_root=str(tmp_path),
    )
    assert Path(composed) == tmp_path / "shots" / "shot01" / "plate_main.mp4"


def test_input_path_requires_filename() -> None:
    with pytest.raises(ValueError, match="video must include a filename"):
        _compose_input_video_path("", "/projects/shot01")


def test_normalize_text_input_handles_frontend_sentinels() -> None:
    assert _normalize_text_input("undefined") == ""
    assert _normalize_text_input("None") == ""
    assert _normalize_text_input("  plate_main.mp4\nundefined") == "plate_main.mp4"
