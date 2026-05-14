"""Tests for the listing row-augment that surfaces the newest recovery
mtime per named preset.

The Load dialog uses the augmented mtime to decide whether to flag a row
as "newer auto-save available". Before issue #137 the augmentation only
considered ``periodic.json``; pre-load files were hidden in the Recovery
disclosure even when newer. After the fix the helper returns the max
mtime across ``periodic.json`` and every ``pre_load_*.json`` so the row
flag reflects the freshest recovery file regardless of which capture
mechanism wrote it.
"""
from __future__ import annotations

import os
from pathlib import Path

from koolook_routes import _latest_autosave_mtime


def _touch(path: Path, mtime: float) -> None:
    """Create ``path`` with empty contents and pin its mtime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    os.utime(path, (mtime, mtime))


def test_returns_periodic_mtime_when_only_periodic_is_newer(tmp_path: Path) -> None:
    autosave_dir = tmp_path / "foo_autosave"
    _touch(autosave_dir / "periodic.json", mtime=2000.0)

    assert _latest_autosave_mtime(autosave_dir, named_mtime=1000.0) == 2000.0


def test_returns_newest_pre_load_mtime_when_only_pre_loads_exist(tmp_path: Path) -> None:
    """When no ``periodic.json`` exists but ``pre_load_*.json`` files do, the
    helper returns the mtime of the newest pre_load file (if newer than the
    named save). Pre_loads are written on every Load and rotated up to 5
    deep, so any of them may be the freshest recovery point."""
    autosave_dir = tmp_path / "foo_autosave"
    _touch(autosave_dir / "pre_load_2024-01-01T10-00-00.json", mtime=1500.0)
    _touch(autosave_dir / "pre_load_2024-01-02T10-00-00.json", mtime=3000.0)
    _touch(autosave_dir / "pre_load_2024-01-01T11-00-00.json", mtime=2000.0)

    assert _latest_autosave_mtime(autosave_dir, named_mtime=1000.0) == 3000.0


def test_returns_max_when_both_periodic_and_pre_loads_exist(tmp_path: Path) -> None:
    """When both file kinds exist, the helper returns the newest mtime across
    them regardless of which kind produced it. Issue #137's whole point:
    the row flag must reflect the freshest recovery point even when a
    pre_load is newer than the latest periodic snapshot.
    """
    autosave_dir = tmp_path / "foo_autosave"
    _touch(autosave_dir / "periodic.json", mtime=2000.0)
    _touch(autosave_dir / "pre_load_2024-01-02T10-00-00.json", mtime=4000.0)
    _touch(autosave_dir / "pre_load_2024-01-01T10-00-00.json", mtime=1800.0)

    assert _latest_autosave_mtime(autosave_dir, named_mtime=1000.0) == 4000.0

    # And the symmetric case: periodic wins when newer than every pre_load.
    _touch(autosave_dir / "periodic.json", mtime=5000.0)
    assert _latest_autosave_mtime(autosave_dir, named_mtime=1000.0) == 5000.0


def test_returns_none_when_no_autosave_file_is_newer_than_named(tmp_path: Path) -> None:
    """Strictly-newer rule: when every autosave file's mtime is ``<=`` the
    named save's mtime, the helper returns ``None`` so the row stays in
    its normal state (no "newer auto-save" affordance fires).
    """
    autosave_dir = tmp_path / "foo_autosave"
    _touch(autosave_dir / "periodic.json", mtime=900.0)
    _touch(autosave_dir / "pre_load_2024-01-01T10-00-00.json", mtime=800.0)
    _touch(autosave_dir / "pre_load_2024-01-02T10-00-00.json", mtime=1000.0)

    assert _latest_autosave_mtime(autosave_dir, named_mtime=1000.0) is None


def test_returns_none_when_autosave_dir_is_missing(tmp_path: Path) -> None:
    """No autosave dir at all (the common case for freshly-created presets):
    the helper returns ``None`` without raising. Powers the library-root
    listing pass where most rows won't have a sibling autosave dir.
    """
    missing = tmp_path / "foo_autosave"
    assert _latest_autosave_mtime(missing, named_mtime=1000.0) is None
