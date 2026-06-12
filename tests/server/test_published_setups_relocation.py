from __future__ import annotations

import koolook_routes
from koolook_setups import DEFAULT_SETUPS_FILENAME, DEFAULT_SETUPS_SUBDIR


def test_published_setups_path_sits_beside_snapshot_library(monkeypatch, tmp_path) -> None:
    """Published setups live as a sibling of the configured snapshot library
    (so they follow libraryPath / KFORGELABS_PRESETS) instead of a fixed
    user-dir folder (issue #227)."""
    library = tmp_path / "kforge" / "koolook-presets"
    monkeypatch.setattr(koolook_routes, "_configured_dir", lambda: (library, "settings"))

    assert koolook_routes._published_setups_path() == (
        tmp_path / "kforge" / DEFAULT_SETUPS_SUBDIR / DEFAULT_SETUPS_FILENAME
    )


def test_migrate_legacy_published_setups_copies_when_target_absent(tmp_path) -> None:
    """A pre-relocation registry at the old user-dir path is copied to the new
    location when the new one has none yet — non-destructively (old file kept)."""
    legacy = tmp_path / "user" / "koolook-published-setups" / "setups.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"setups": [{"id": "carried-over"}]}', encoding="utf-8")
    target = tmp_path / "kforge" / "koolook-published-setups" / "setups.json"

    koolook_routes._migrate_legacy_published_setups(legacy, target)

    assert target.is_file()
    assert target.read_text(encoding="utf-8") == legacy.read_text(encoding="utf-8")
    assert legacy.is_file()  # non-destructive: original is left in place


def test_migrate_legacy_published_setups_noop_when_target_present(tmp_path) -> None:
    """Migration never overwrites an existing registry at the new location."""
    legacy = tmp_path / "user" / "koolook-published-setups" / "setups.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"setups": [{"id": "old"}]}', encoding="utf-8")
    target = tmp_path / "kforge" / "koolook-published-setups" / "setups.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"setups": [{"id": "current"}]}', encoding="utf-8")

    koolook_routes._migrate_legacy_published_setups(legacy, target)

    assert '"current"' in target.read_text(encoding="utf-8")


def test_migrate_legacy_published_setups_noop_when_no_legacy(tmp_path) -> None:
    """Fresh install (no old file) migrates nothing and does not create the
    target — the registry's own fallback handles an empty location."""
    legacy = tmp_path / "user" / "koolook-published-setups" / "setups.json"
    target = tmp_path / "kforge" / "koolook-published-setups" / "setups.json"

    koolook_routes._migrate_legacy_published_setups(legacy, target)

    assert not target.exists()


def _point_registry(monkeypatch, tmp_path):
    """Wire the default registry factory at tmp paths and return (legacy, target)."""
    library = tmp_path / "lib" / "koolook-presets"
    monkeypatch.setattr(koolook_routes, "_configured_dir", lambda: (library, "settings"))
    legacy = tmp_path / "user" / "koolook-published-setups" / "setups.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"setups": []}', encoding="utf-8")
    monkeypatch.setattr(koolook_routes, "default_storage_path", lambda: legacy)
    target = tmp_path / "lib" / "koolook-published-setups" / "setups.json"
    return legacy, target


def test_default_registry_keeps_legacy_data_when_migration_fails(monkeypatch, tmp_path) -> None:
    """If the legacy->new copy fails, the registry keeps reading the real legacy
    file rather than masking it with the bundled sample (PR #239 review, HIGH)."""
    legacy, target = _point_registry(monkeypatch, tmp_path)

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(koolook_routes.shutil, "copy2", boom)

    registry = koolook_routes._default_published_setup_registry()

    # Migration could not produce the new file, so the primary storage stays on
    # the legacy file (real data) — not the new path that would sample-fallback.
    assert not target.exists()
    assert registry.storage_path == legacy


def test_default_registry_uses_new_path_after_successful_migration(monkeypatch, tmp_path) -> None:
    """On a clean migration the registry uses the relocated path."""
    legacy, target = _point_registry(monkeypatch, tmp_path)

    registry = koolook_routes._default_published_setup_registry()

    assert target.is_file()
    assert registry.storage_path == target
