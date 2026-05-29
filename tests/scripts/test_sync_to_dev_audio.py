"""Safety coverage for scripts/sync_to_dev_audio.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "sync_to_dev_audio.py"
_spec = importlib.util.spec_from_file_location("sync_to_dev_audio", _SCRIPT)
assert _spec is not None and _spec.loader is not None
sync_to_dev_audio = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync_to_dev_audio)


def _stale_path(root: Path) -> Path:
    return root / "web" / "whatdreamscost_koolook_v1_3_2"


def test_remove_stale_paths_removes_directory(tmp_path: Path) -> None:
    stale = _stale_path(tmp_path)
    stale.mkdir(parents=True)
    (stale / "ltx_director.js").write_text("old extension\n", encoding="utf-8")

    removed = sync_to_dev_audio.remove_stale_paths(
        tmp_path,
        dry_run=False,
        verbose=False,
    )

    assert removed == 1
    assert not stale.exists()


def test_remove_stale_paths_dry_run_preserves_directory(tmp_path: Path) -> None:
    stale = _stale_path(tmp_path)
    stale.mkdir(parents=True)

    removed = sync_to_dev_audio.remove_stale_paths(
        tmp_path,
        dry_run=True,
        verbose=False,
    )

    assert removed == 1
    assert stale.exists()


def test_remove_stale_paths_idempotent_when_absent(tmp_path: Path) -> None:
    removed = sync_to_dev_audio.remove_stale_paths(
        tmp_path,
        dry_run=False,
        verbose=False,
    )

    assert removed == 0


def test_remove_stale_paths_unlinks_symlink_without_touching_target(tmp_path: Path) -> None:
    current = tmp_path / "web" / "whatdreamscost_koolook"
    current.mkdir(parents=True)
    keep = current / "ltx_director.js"
    keep.write_text("current extension\n", encoding="utf-8")
    stale = _stale_path(tmp_path)
    try:
        stale.symlink_to(current, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    removed = sync_to_dev_audio.remove_stale_paths(
        tmp_path,
        dry_run=False,
        verbose=False,
    )

    assert removed == 1
    assert not stale.exists()
    assert keep.read_text(encoding="utf-8") == "current extension\n"


def test_target_is_repo_root_detects_source_repo() -> None:
    assert sync_to_dev_audio.target_is_repo_root(sync_to_dev_audio._dev.REPO_ROOT)
