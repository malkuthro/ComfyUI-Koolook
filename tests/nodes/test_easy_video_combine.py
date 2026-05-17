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

from k_video_combine import (
    _add_metadata_json_sidecar,
    _compose_prefix,
    _display_format_name,
    _metadata_sidecar_path,
    _normalize_bool_input,
    _normalize_text_input,
    _remove_audio_suffix_from_result,
    _resolve_abs_target,
    _runtime_format_name,
)


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
    """Existing directory paths work with a final slash."""
    renders_dir = tmp_path / "renders"
    renders_dir.mkdir()
    prefix = str(renders_dir) + os.sep

    result = _resolve_abs_target(prefix, create_path_if_missing=False)

    assert result is not None
    abs_dir, abs_base = result
    assert Path(abs_dir) == renders_dir
    assert abs_base == "renders"


def test_absolute_existing_directory_without_trailing_separator(tmp_path: Path) -> None:
    """Pasted directory paths work with or without a final slash."""
    renders_dir = tmp_path / "renders"
    renders_dir.mkdir()

    result = _resolve_abs_target(str(renders_dir), create_path_if_missing=False)

    assert result is not None
    abs_dir, abs_base = result
    assert Path(abs_dir) == renders_dir
    assert abs_base == "renders"


def test_absolute_prefix_with_no_filename_component_raises(tmp_path: Path) -> None:
    """An absolute path with empty basename has no filename root to
    append the counter to — surface as a clear ValueError instead of
    silently writing files named ``_00001.mp4``."""
    root = tmp_path.anchor  # "/" on POSIX, "C:\\" on Windows.
    assert os.path.isabs(root)

    with pytest.raises(ValueError, match="no usable folder name"):
        _resolve_abs_target(root, create_path_if_missing=False)


# =============================================================================
# _compose_prefix: split-mode output_directory + filename_prefix.
# =============================================================================

def test_compose_empty_output_directory_passes_through() -> None:
    """No output_directory -> filename_prefix returned unchanged."""
    assert _compose_prefix("AnimateDiff", "") == "AnimateDiff"
    assert _compose_prefix("E:/renders/clip", "") == "E:/renders/clip"


def test_compose_absolute_directory_with_name() -> None:
    """Absolute output_directory + name -> joined absolute prefix."""
    composed = _compose_prefix("clip_v003", "E:/renders/shot01")
    assert composed == "E:/renders/shot01" + os.sep + "clip_v003"
    assert os.path.isabs(composed)


def test_compose_relative_directory_with_name() -> None:
    """Relative output_directory + name -> joined relative prefix.

    Stays relative so the downstream isabs discrimination passes it
    through to upstream's sandbox.
    """
    composed = _compose_prefix("clip", "shots/v003")
    assert composed == "shots/v003" + os.sep + "clip"
    assert not os.path.isabs(composed)


def test_compose_strips_path_components_from_filename_prefix() -> None:
    """When output_directory is set, only the basename of filename_prefix
    becomes the file root — users can change the name without retyping
    the directory."""
    composed = _compose_prefix("subdir/clip_v003", "E:/renders/shot01")
    assert composed == "E:/renders/shot01" + os.sep + "clip_v003"


def test_compose_trims_trailing_separator_from_directory() -> None:
    """Trailing slashes on output_directory don't double-up in the join."""
    composed_unix = _compose_prefix("clip", "E:/renders/shot01/")
    composed_win = _compose_prefix("clip", "E:\\renders\\shot01\\")
    assert composed_unix == "E:/renders/shot01" + os.sep + "clip"
    assert composed_win == "E:\\renders\\shot01" + os.sep + "clip"


def test_compose_empty_name_falls_back_to_default() -> None:
    """If filename_prefix's basename is empty (e.g. user typed just '/'),
    fall back to AnimateDiff rather than producing a path ending in
    just a separator."""
    composed = _compose_prefix("/", "E:/renders/shot01")
    assert composed == "E:/renders/shot01" + os.sep + "AnimateDiff"


# =============================================================================
# _normalize_text_input: guard against the ComfyUI frontend "undefined" quirk.
# A STRING widget left untouched can arrive at the backend as the literal
# string "undefined" (or "null" / "None") instead of an empty value.
# Without normalization that would slip into _compose_prefix and produce a
# nonsensical `undefined/` subdirectory on disk.
# =============================================================================

def test_normalize_empty_inputs() -> None:
    assert _normalize_text_input("") == ""
    assert _normalize_text_input("   ") == ""
    assert _normalize_text_input(None) == ""


def test_normalize_undefined_sentinel_strings() -> None:
    """All case variants of the frontend-quirk strings become empty."""
    assert _normalize_text_input("undefined") == ""
    assert _normalize_text_input("Undefined") == ""
    assert _normalize_text_input("UNDEFINED") == ""
    assert _normalize_text_input("null") == ""
    assert _normalize_text_input("None") == ""
    assert _normalize_text_input("  undefined  ") == ""


def test_normalize_passes_through_real_values() -> None:
    assert _normalize_text_input("E:/renders/shot01") == "E:/renders/shot01"
    assert _normalize_text_input("clip_v003") == "clip_v003"
    # Substring matches don't trigger — only exact match (post-strip).
    assert _normalize_text_input("undefined_path") == "undefined_path"
    assert _normalize_text_input("my_null_clip") == "my_null_clip"


def test_normalize_prevents_undefined_subdir_in_compose() -> None:
    """Regression test for the bug: output_directory='undefined' was
    being treated as a real directory, producing an `undefined/`
    subfolder when the user typed an absolute path in filename_prefix."""
    normalized_dir = _normalize_text_input("undefined")
    composed = _compose_prefix("E:/renders/clip_v003", normalized_dir)
    # With normalization, no output_directory -> filename_prefix passes through.
    assert composed == "E:/renders/clip_v003"
    assert "undefined" not in composed


# =============================================================================
# _normalize_bool_input: guard against dynamic widget restore drift.
# If format-specific widgets are restored by position instead of by name, a
# value like "hq" can land in pingpong. Python truthiness would treat that as
# True and VHS would double the video by ping-ponging the frame sequence.
# =============================================================================

def test_normalize_bool_input_known_values() -> None:
    assert _normalize_bool_input(True) is True
    assert _normalize_bool_input(False) is False
    assert _normalize_bool_input("true") is True
    assert _normalize_bool_input("1") is True
    assert _normalize_bool_input("on") is True
    assert _normalize_bool_input("false") is False
    assert _normalize_bool_input("0") is False
    assert _normalize_bool_input("off") is False
    assert _normalize_bool_input("") is False


def test_normalize_bool_input_unknown_string_uses_default() -> None:
    assert _normalize_bool_input("hq", default=False) is False
    assert _normalize_bool_input("hq", default=True) is True


def test_metadata_sidecar_path_replaces_vhs_png_sidecar(tmp_path: Path) -> None:
    png = tmp_path / "clip_00001.png"
    assert _metadata_sidecar_path([str(png)], save_metadata_png=False) == str(
        tmp_path / "clip_00001.json"
    )


def test_add_metadata_json_sidecar_replaces_disabled_png_entry(tmp_path: Path) -> None:
    png = tmp_path / "clip_00001.png"
    mp4 = tmp_path / "clip_00001.mp4"
    result = {
        "ui": {"gifs": [{"workflow": png.name}]},
        "result": ((True, [str(png), str(mp4)]),),
    }

    out = _add_metadata_json_sidecar(
        result,
        {"prompt": {"1": {"class_type": "Example"}}},
        save_metadata_json=True,
        save_metadata_png=False,
    )

    json_path = tmp_path / "clip_00001.json"
    assert json_path.exists()
    assert out["result"][0][1] == [str(json_path), str(mp4)]
    assert out["ui"]["gifs"][0]["workflow"] == json_path.name


def test_remove_audio_suffix_renames_final_and_drops_missing_intermediate(tmp_path: Path) -> None:
    png = tmp_path / "clip_00001.png"
    silent = tmp_path / "clip_00001.mp4"
    audio = tmp_path / "clip_00001-audio.mp4"
    png.write_text("png")
    audio.write_text("video")
    result = {
        "ui": {"gifs": [{"filename": audio.name, "fullpath": str(audio)}]},
        "result": ((True, [str(png), str(silent), str(audio)]),),
    }

    out = _remove_audio_suffix_from_result(result, keep_silent_intermediate=False)

    clean = tmp_path / "clip_00001.mp4"
    assert clean.exists()
    assert not audio.exists()
    assert out["result"][0][1] == [str(png), str(clean)]
    assert out["ui"]["gifs"][0]["filename"] == clean.name
    assert out["ui"]["gifs"][0]["fullpath"] == str(clean)


def test_koolook_format_display_hides_json_suffix() -> None:
    assert _display_format_name("video/koolook-ASTRA-h264.json") == "video/koolook-ASTRA-h264"
    assert _display_format_name("video/ProRes") == "video/ProRes"


def test_koolook_format_runtime_restores_json_suffix() -> None:
    assert _runtime_format_name("video/koolook-ASTRA-h264") == "video/koolook-ASTRA-h264.json"
    assert _runtime_format_name("video/ProRes") == "video/ProRes"
