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


def test_node_registered_and_runs(tmp_path: Path) -> None:
    from k_publish_contract import NODE_CLASS_MAPPINGS, Koolook_NextVersion

    assert NODE_CLASS_MAPPINGS["Koolook_NextVersion"] is Koolook_NextVersion
    (tmp_path / "shot_v001.mov").write_bytes(b"x")
    (tmp_path / "shot_v002.mov").write_bytes(b"x")
    assert Koolook_NextVersion().run(str(tmp_path), "shot") == ("v003",)
    # Empty folder -> start token.
    assert Koolook_NextVersion().run(str(tmp_path / "nope"), "shot") == ("v001",)
