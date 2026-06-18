"""Tests for filesystem-aware output version auto-detection."""
from __future__ import annotations

from pathlib import Path

from koolook_versioning import next_version_token


def test_missing_directory_returns_start(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    assert next_version_token(str(missing), "bearMask") == "v001"


def test_empty_directory_returns_start(tmp_path: Path) -> None:
    assert next_version_token(str(tmp_path), "bearMask") == "v001"


def test_next_after_existing_versioned_files(tmp_path: Path) -> None:
    (tmp_path / "bearMask_v001.png").write_bytes(b"x")
    (tmp_path / "bearMask_v002.png").write_bytes(b"x")
    assert next_version_token(str(tmp_path), "bearMask") == "v003"


def test_next_ignores_other_names(tmp_path: Path) -> None:
    (tmp_path / "bearMask_v005.png").write_bytes(b"x")
    (tmp_path / "otherShot_v009.png").write_bytes(b"x")
    # Exact base-name match only -- otherShot must not bump bearMask.
    assert next_version_token(str(tmp_path), "bearMask") == "v006"


def test_next_matches_versioned_subfolders(tmp_path: Path) -> None:
    (tmp_path / "bearMask_v003").mkdir()
    (tmp_path / "bearMask_v007").mkdir()
    assert next_version_token(str(tmp_path), "bearMask") == "v008"


def test_next_handles_exr_sequence_names(tmp_path: Path) -> None:
    (tmp_path / "bearMask_v002.0001.exr").write_bytes(b"x")
    (tmp_path / "bearMask_v002.0002.exr").write_bytes(b"x")
    assert next_version_token(str(tmp_path), "bearMask") == "v003"


def test_padding_and_prefix_are_configurable(tmp_path: Path) -> None:
    (tmp_path / "clip_take2.mov").write_bytes(b"x")
    assert (
        next_version_token(str(tmp_path), "clip", version_prefix="take", padding=1)
        == "take3"
    )


def test_empty_name_matches_bare_version_tokens(tmp_path: Path) -> None:
    (tmp_path / "v004.png").write_bytes(b"x")
    assert next_version_token(str(tmp_path), "") == "v005"


def test_is_auto_version_detects_keywords() -> None:
    from koolook_versioning import is_auto_version

    for yes in ("auto", "AUTO", " next ", "Next", '"auto"'):
        assert is_auto_version(yes) is True
    for no in ("", "v001", "3", "final", "automatic", None):
        assert is_auto_version(no) is False


def test_ai_pipeline_auto_version_picks_next(tmp_path: Path) -> None:
    from k_ai_pipeline import build_pipeline_outputs

    # Existing version folders v001/v002 under the base (no_subfolders mode).
    (tmp_path / "v001").mkdir()
    (tmp_path / "v002").mkdir()
    out = build_pipeline_outputs(
        81, 1, "", str(tmp_path), ".png", "bearMask", "", "auto",
        False, False, True,
        create_directory=False, check_overwrite=False,
    )
    file_path, name, version_str = out[0], out[1], out[2]
    assert version_str == "v003"
    assert name == "bearMask_v003.png"
    assert file_path.endswith("/v003/bearMask_v003.png")


def test_ai_pipeline_auto_version_first_run_is_v001(tmp_path: Path) -> None:
    from k_ai_pipeline import build_pipeline_outputs

    out = build_pipeline_outputs(
        81, 1, "", str(tmp_path), ".png", "bearMask", "", "auto",
        False, False, True,
        create_directory=False, check_overwrite=False,
    )
    assert out[2] == "v001"


