"""Tests for ``_list_child_dirs`` / ``_list_child_files`` — the listing
helpers that power the folder picker (issue #137, mockup section 6).

Folders are listed by name, sorted case-insensitively, with ``_autosave``
subfolders hidden (those are koolook-internal recovery dirs, not user-
selectable library locations). Files are listed JSON-only, also sorted
case-insensitively. Both helpers swallow per-entry ``OSError`` so a
broken symlink or unreadable dotfile in the middle of the listing
doesn't sink the whole response.
"""
from __future__ import annotations

from pathlib import Path

from koolook_routes import _list_child_dirs, _list_child_files


def test_list_child_dirs_returns_sorted_dir_entries(tmp_path: Path) -> None:
    (tmp_path / "zebra").mkdir()
    (tmp_path / "Alpha").mkdir()
    (tmp_path / "mango").mkdir()

    rows = _list_child_dirs(tmp_path)

    assert [r["name"] for r in rows] == ["Alpha", "mango", "zebra"]
    for row in rows:
        assert set(row) == {"name", "path"}
        assert row["path"].endswith(row["name"])


def test_list_child_dirs_hides_autosave_subfolders(tmp_path: Path) -> None:
    """Per-preset autosave dirs live under the library, but the picker
    must not let users pick them as a library location — they're
    koolook-internal recovery state. ``_unsaved_autosave`` (no preset
    name prefix) is also hidden by the same rule."""
    (tmp_path / "user-pick").mkdir()
    (tmp_path / "MyPreset_autosave").mkdir()
    (tmp_path / "_unsaved_autosave").mkdir()
    (tmp_path / "other-pick").mkdir()

    rows = _list_child_dirs(tmp_path)

    assert [r["name"] for r in rows] == ["other-pick", "user-pick"]


def test_list_child_dirs_ignores_regular_files(tmp_path: Path) -> None:
    (tmp_path / "subdir").mkdir()
    (tmp_path / "note.txt").write_text("ignored")
    (tmp_path / "preset.json").write_text("{}")

    rows = _list_child_dirs(tmp_path)

    assert [r["name"] for r in rows] == ["subdir"]


def test_list_child_files_returns_only_json_sorted(tmp_path: Path) -> None:
    """Files-for-context affordance restricts to ``*.json`` because the
    picker is exclusively used to choose a snapshot library folder;
    surfacing non-JSON files would mislead the user."""
    (tmp_path / "Koolook_v03.json").write_text("{}")
    (tmp_path / "starter.json").write_text("{}")
    (tmp_path / "readme.md").write_text("ignored")
    (tmp_path / "notes.txt").write_text("ignored")
    # Subdirs are not files — should be ignored even if named *.json.
    (tmp_path / "looks_like_a_file.json").mkdir()

    rows = _list_child_files(tmp_path)

    assert [r["name"] for r in rows] == ["Koolook_v03.json", "starter.json"]
    for row in rows:
        assert set(row) == {"name"}, "files-for-context entries carry name only"


def test_list_child_files_returns_empty_for_no_json(tmp_path: Path) -> None:
    (tmp_path / "readme.md").write_text("ignored")
    (tmp_path / "subdir").mkdir()

    assert _list_child_files(tmp_path) == []


def test_list_child_files_is_case_insensitive_on_extension(tmp_path: Path) -> None:
    """Spelling of ``.json`` shouldn't matter on a case-insensitive FS;
    surface upper-case copies too so a user-renamed ``PRESET.JSON``
    doesn't disappear from the affordance."""
    (tmp_path / "lower.json").write_text("{}")
    (tmp_path / "UPPER.JSON").write_text("{}")

    rows = _list_child_files(tmp_path)

    assert sorted(r["name"] for r in rows) == ["UPPER.JSON", "lower.json"]
