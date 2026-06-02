"""Tests for the duplicate-install guard (#162) — see
``koolook_install_guard.py`` for the module under test.

Each test materialises a simulated ``custom_nodes/`` layout under
``tmp_path`` and runs the pure-function helpers directly. No ComfyUI,
no aiohttp, no relative imports — the guard is deliberately decoupled
from the runtime so it can be exercised by stdlib pytest.
"""
from __future__ import annotations

from pathlib import Path

from koolook_install_guard import (
    build_duplicate_report,
    detect_duplicate_koolook_installs,
    pick_winning_install,
    read_pyproject_version,
)


def _make_install(parent: Path, name: str, version: str | None = "0.3.7") -> Path:
    """Materialise a fake Koolook install — a directory with the
    ``koolook_routes.py`` marker file and an optional ``pyproject.toml``
    with the given version. Returns the install directory.
    """
    install_dir = parent / name
    install_dir.mkdir()
    (install_dir / "koolook_routes.py").write_text("# marker", encoding="utf-8")
    if version is not None:
        (install_dir / "pyproject.toml").write_text(
            f'[project]\nname = "koolook"\nversion = "{version}"\n',
            encoding="utf-8",
        )
    return install_dir


# =============================================================================
# detect_duplicate_koolook_installs
# =============================================================================


def test_detect_no_siblings_returns_empty(tmp_path: Path) -> None:
    """Single install — no other Koolook folders. Sibling list is empty
    and the guard short-circuits to "normal startup"."""
    here = _make_install(tmp_path, "koolook")
    assert detect_duplicate_koolook_installs(here) == []


def test_detect_finds_sibling_with_marker(tmp_path: Path) -> None:
    """Two installs side by side under ``custom_nodes/`` — the sibling
    is found via its ``koolook_routes.py`` marker. This is the canonical
    failure mode the guard exists to catch."""
    here = _make_install(tmp_path, "koolook")
    other = _make_install(tmp_path, "ComfyUI-Koolook")
    assert detect_duplicate_koolook_installs(here) == [other]


def test_detect_ignores_dirs_without_marker(tmp_path: Path) -> None:
    """A non-Koolook custom node sitting in the same ``custom_nodes/``
    folder must not be flagged as a sibling — only directories carrying
    the ``koolook_routes.py`` marker file are duplicates of us."""
    here = _make_install(tmp_path, "koolook")
    bystander = tmp_path / "ComfyUI-Manager"
    bystander.mkdir()
    (bystander / "__init__.py").write_text("# unrelated plugin", encoding="utf-8")
    assert detect_duplicate_koolook_installs(here) == []


def test_detect_ignores_loose_files(tmp_path: Path) -> None:
    """A loose ``koolook_routes.py`` left in ``custom_nodes/`` itself
    (not inside a directory) doesn't count — ComfyUI only loads custom
    nodes from subdirectories, so a misplaced file can't compete with
    our install for route registration."""
    here = _make_install(tmp_path, "koolook")
    (tmp_path / "koolook_routes.py").write_text("# loose", encoding="utf-8")
    assert detect_duplicate_koolook_installs(here) == []


def test_detect_orders_siblings_alphabetically(tmp_path: Path) -> None:
    """Two+ siblings come back in case-insensitive name order so the
    critical log reads consistently across both copies (and matches
    the alphabetical winner selection in ``pick_winning_install``)."""
    here = _make_install(tmp_path, "mid-koolook")
    later = _make_install(tmp_path, "z-koolook")
    earlier = _make_install(tmp_path, "a-koolook")
    siblings = detect_duplicate_koolook_installs(here)
    assert [s.name for s in siblings] == ["a-koolook", "z-koolook"]
    # Mid-koolook is ``here`` and excluded from the sibling list — the
    # caller computes the full alphabetical order by passing both.
    assert earlier in siblings
    assert later in siblings


def test_detect_skips_unreadable_sibling(tmp_path: Path, monkeypatch) -> None:
    """A sibling directory we can't stat into (restrictive perms / broken
    mount) must be skipped, not crash the scan. Regression: an unguarded
    ``is_file()`` raised ``PermissionError`` up into ``__init__.py`` and the
    whole plugin failed to load when ``custom_nodes/`` held an unreadable
    neighbour (e.g. a root-owned service dir)."""
    here = _make_install(tmp_path, "koolook")
    blocked = tmp_path / "blocked-node"
    blocked.mkdir()
    real_is_file = Path.is_file

    def fake_is_file(self: Path) -> bool:
        if self.name == "koolook_routes.py" and self.parent == blocked:
            raise PermissionError(13, "Permission denied")
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    # Must not raise; the unreadable sibling is simply skipped.
    assert detect_duplicate_koolook_installs(here) == []


# =============================================================================
# read_pyproject_version
# =============================================================================


def test_read_version_extracts_quoted_value(tmp_path: Path) -> None:
    install = _make_install(tmp_path, "koolook", version="0.3.8")
    assert read_pyproject_version(install) == "0.3.8"


def test_read_version_handles_single_quotes(tmp_path: Path) -> None:
    """Some pyprojects use single quotes — our manual scan should
    handle both. The duplicate-install log doesn't care which the user
    wrote, just that we can extract it."""
    install = tmp_path / "koolook"
    install.mkdir()
    (install / "koolook_routes.py").write_text("# marker", encoding="utf-8")
    (install / "pyproject.toml").write_text(
        "[project]\nname = 'koolook'\nversion = '0.3.9'\n",
        encoding="utf-8",
    )
    assert read_pyproject_version(install) == "0.3.9"


def test_read_version_returns_qmark_when_missing(tmp_path: Path) -> None:
    """No pyproject.toml — fall back to ``"?"`` so the critical log
    still names the path."""
    install = _make_install(tmp_path, "koolook", version=None)
    assert read_pyproject_version(install) == "?"


def test_read_version_ignores_versioning_near_collision(tmp_path: Path) -> None:
    """A ``[tool.koolook] versioning = "strict"`` line shouldn't trip
    the version extractor — only the literal ``version = ...`` key."""
    install = tmp_path / "koolook"
    install.mkdir()
    (install / "koolook_routes.py").write_text("# marker", encoding="utf-8")
    (install / "pyproject.toml").write_text(
        '[project]\nname = "koolook"\nversioning = "strict"\nversion = "0.3.10"\n',
        encoding="utf-8",
    )
    assert read_pyproject_version(install) == "0.3.10"


def test_read_version_strips_inline_comment(tmp_path: Path) -> None:
    """Tolerate ``version = "0.3.11"  # bumped 2026-05-29``."""
    install = tmp_path / "koolook"
    install.mkdir()
    (install / "koolook_routes.py").write_text("# marker", encoding="utf-8")
    (install / "pyproject.toml").write_text(
        '[project]\nversion = "0.3.11"  # bumped\n',
        encoding="utf-8",
    )
    assert read_pyproject_version(install) == "0.3.11"


def test_read_version_handles_non_utf8_pyproject(tmp_path: Path) -> None:
    """A non-UTF-8 / binary ``pyproject.toml`` must not raise
    ``UnicodeDecodeError`` — the version is simply unknown (``"?"``).
    Regression: ``read_text(encoding="utf-8")`` raised a ``ValueError``
    that escaped the ``except OSError`` and crashed the plugin import."""
    install = tmp_path / "koolook"
    install.mkdir()
    (install / "koolook_routes.py").write_text("# marker", encoding="utf-8")
    (install / "pyproject.toml").write_bytes(b"\xff\xfe version = broken \x00")
    assert read_pyproject_version(install) == "?"


# =============================================================================
# pick_winning_install + build_duplicate_report
# =============================================================================


def test_pick_winning_install_is_alphabetical(tmp_path: Path) -> None:
    """Resolution is alphabetical-by-folder-name (case-insensitive) so
    both copies arrive at the same winner without runtime coordination."""
    here = _make_install(tmp_path, "koolook")
    other = _make_install(tmp_path, "ComfyUI-Koolook")
    # "ComfyUI-Koolook" < "koolook" case-insensitively → other wins.
    assert pick_winning_install(here, [other]) == other


def test_build_report_marks_winner_correctly(tmp_path: Path) -> None:
    """The ``custom_nodes/koolook/`` install (Registry-derived name) is
    the alphabetical loser when shadowed by a ``ComfyUI-Koolook/`` clone.
    The Manager-installed copy is the one users see in the Manager UI,
    so the winner is the clone — but the guard reports both paths
    regardless so the user can choose which to remove."""
    here = _make_install(tmp_path, "koolook", version="0.3.7")
    other = _make_install(tmp_path, "ComfyUI-Koolook", version="0.3.8")
    is_winning, message = build_duplicate_report(here, [other])
    assert is_winning is False
    assert str(here) in message
    assert str(other) in message
    assert "0.3.7" in message
    assert "0.3.8" in message
    assert "non-winning duplicate" in message
    assert "CRITICAL" in message


def test_build_report_winner_skips_skip_message(tmp_path: Path) -> None:
    """When ``here`` IS the winner, the critical message names both
    paths but does NOT add the "skipping registration" trailer — the
    winner runs unchanged."""
    here = _make_install(tmp_path, "ComfyUI-Koolook", version="0.3.8")
    other = _make_install(tmp_path, "koolook", version="0.3.7")
    is_winning, message = build_duplicate_report(here, [other])
    assert is_winning is True
    assert "CRITICAL" in message
    assert str(here) in message
    assert str(other) in message
    assert "non-winning duplicate" not in message


def test_build_report_handles_three_way_collision(tmp_path: Path) -> None:
    """Edge case: three Koolook installs in the same ``custom_nodes/``.
    The alphabetical winner is the lone survivor; the other two each
    individually compute "I'm a loser" and skip."""
    a = _make_install(tmp_path, "a-koolook", version="0.3.6")
    b = _make_install(tmp_path, "koolook", version="0.3.7")
    c = _make_install(tmp_path, "z-koolook", version="0.3.8")
    # From ``b``'s perspective, both ``a`` and ``c`` are siblings.
    siblings_from_b = detect_duplicate_koolook_installs(b)
    assert siblings_from_b == [a, c]
    is_winning_b, _ = build_duplicate_report(b, siblings_from_b)
    assert is_winning_b is False
    # From ``a``'s perspective, ``a`` wins.
    siblings_from_a = detect_duplicate_koolook_installs(a)
    is_winning_a, _ = build_duplicate_report(a, siblings_from_a)
    assert is_winning_a is True


def test_build_report_survives_non_utf8_sibling_pyproject(tmp_path: Path) -> None:
    """The full report path must not raise on a duplicate whose
    ``pyproject.toml`` is non-UTF-8 — it degrades that version to ``"?"``
    and still names both paths. Regression: this raised ``UnicodeDecodeError``
    up into ``__init__.py`` and took the whole plugin offline (both copies)."""
    here = _make_install(tmp_path, "koolook", version="0.3.7")
    other = tmp_path / "ComfyUI-Koolook"
    other.mkdir()
    (other / "koolook_routes.py").write_text("# marker", encoding="utf-8")
    (other / "pyproject.toml").write_bytes(b"\xffname = broken")
    is_winning, message = build_duplicate_report(here, [other])
    assert is_winning is False  # "ComfyUI-Koolook" < "koolook" case-insensitively
    assert str(other) in message
    assert "0.3.7" in message  # here's own version is still extracted
