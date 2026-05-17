"""Path-input hardening for ``EasyAIPipeline.generate_pipeline``.

The base directory widget accepts whatever the user pastes, and real users
paste in wildly different shapes: Windows Explorer's address bar ends paths
in ``\\``, ``Shift+Right-click → Copy as path`` wraps the result in ``"..."``,
Linux/Mac habits append ``/``, sloppy copy-paste leaves stray whitespace,
and so on.

Every variation has to land in the same canonical ``output_directory``. If
a trailing separator leaks through, some downstream save nodes split on
``/``, stringify the empty tail as ``"undefined"``, and write into
``…/undefined/…`` on disk — a phantom folder that the user never asked
for. These tests pin both the pure ``_normalize_base_path`` helper and the
end-to-end ``generate_pipeline`` behavior so the bug can't regress.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from k_ai_pipeline import EasyAIPipeline, _normalize_base_path


# ---------------------------------------------------------------------------
# _normalize_base_path — pure helper, no filesystem dependency
# ---------------------------------------------------------------------------


class TestNormalizeBasePath:
    def test_clean_input_unchanged(self):
        assert _normalize_base_path("n:\\foo\\bar") == "n:\\foo\\bar"
        assert _normalize_base_path("/home/user/proj") == "/home/user/proj"

    def test_trailing_backslash_stripped(self):
        assert _normalize_base_path("n:\\foo\\bar\\") == "n:\\foo\\bar"

    def test_trailing_forward_slash_stripped(self):
        assert _normalize_base_path("/n/foo/bar/") == "/n/foo/bar"

    def test_multiple_trailing_separators_stripped(self):
        assert _normalize_base_path("n:\\foo\\bar\\\\") == "n:\\foo\\bar"
        assert _normalize_base_path("/n/foo/bar///") == "/n/foo/bar"

    def test_mixed_trailing_separators_stripped(self):
        """A path can end in a mix of forward and backward slashes if the
        user copy-pasted from two different sources."""
        assert _normalize_base_path("n:/foo/bar\\/\\") == "n:/foo/bar"

    def test_surrounding_whitespace_stripped(self):
        assert _normalize_base_path("  n:\\foo\\bar  ") == "n:\\foo\\bar"

    def test_surrounding_double_quotes_stripped(self):
        """``Shift+Right-click → Copy as path`` in Windows Explorer wraps
        the result in double quotes."""
        assert _normalize_base_path('"n:\\foo\\bar"') == "n:\\foo\\bar"

    def test_surrounding_single_quotes_stripped(self):
        assert _normalize_base_path("'n:/foo/bar'") == "n:/foo/bar"

    def test_quotes_and_inner_trailing_separator_combined(self):
        """A user might paste a quote-wrapped path that also carries the
        Explorer trailing backslash — both layers should peel off."""
        assert _normalize_base_path('"n:\\foo\\bar\\"') == "n:\\foo\\bar"
        assert _normalize_base_path('  "n:/foo/bar/"  ') == "n:/foo/bar"

    def test_drive_root_preserved(self):
        """``C:\\`` (drive root) must NOT collapse to ``C:`` — the latter
        means "current directory on C: drive", not the drive root itself.
        Same for ``n:/``."""
        assert _normalize_base_path("C:\\") == "C:\\"
        assert _normalize_base_path("n:/") == "n:/"

    def test_unmatched_quotes_passed_through(self):
        """Only strip quotes when BOTH ends match. A single dangling quote
        is probably part of the path, not a wrapper."""
        assert _normalize_base_path('"n:\\foo') == '"n:\\foo'
        assert _normalize_base_path("n:\\foo'") == "n:\\foo'"
        assert _normalize_base_path("\"n:\\foo'") == "\"n:\\foo'"

    def test_empty_string_passes_through(self):
        assert _normalize_base_path("") == ""
        assert _normalize_base_path("   ") == ""


# ---------------------------------------------------------------------------
# generate_pipeline — end-to-end behavior with realistic inputs
# ---------------------------------------------------------------------------


def _run(base_directory_path: str, **overrides):
    """Run ``EasyAIPipeline.generate_pipeline`` with sensible defaults and
    return ``(file_path, output_name, version_string, output_directory)``,
    which is what the bug-area assertions actually care about."""
    defaults = dict(
        shot_duration=81,
        seed_value=453453453,
        instruction="(ignored)",
        base_directory_path=base_directory_path,
        extension=".%04d.exr",
        shot_name="LTX-Director-2k_base",
        ai_method="",
        version=1,
        disable_versioning=True,
        enable_overwrite=False,
        no_subfolders=False,
    )
    defaults.update(overrides)
    result = EasyAIPipeline().generate_pipeline(**defaults)
    file_path, output_name, version_string, output_directory = result[:4]
    return file_path, output_name, version_string, output_directory


# Variations the user might realistically paste into the widget. Each must
# canonicalise to the same output_directory as the bare ``{base}`` form.
PASTE_VARIANTS = [
    "{base}",                # clean
    "{base}\\",              # Explorer address-bar trailing backslash
    "{base}/",               # POSIX trailing slash
    "{base}\\\\",            # accidental double backslash
    "{base}//",              # accidental double forward slash
    "{base}\\/",             # mixed trailing separators
    "  {base}  ",            # sloppy whitespace
    "  {base}\\  ",          # whitespace + trailing backslash
    '"{base}"',              # Shift+Right-click → Copy as path
    '"{base}\\"',            # quoted AND trailing backslash
    "'{base}'",              # single-quoted
]


@pytest.mark.parametrize("variant", PASTE_VARIANTS)
def test_output_directory_canonical_across_paste_variations(tmp_path: Path, variant: str):
    """No matter how the user pastes the base path, ``output_directory``
    must be identical to the clean-input form. This is the contract that
    prevents downstream save nodes from receiving subtly different paths
    based on irrelevant whitespace/quoting/trailing-slash differences."""
    base = tmp_path / "bear"
    base.mkdir()
    canonical = str(base).replace("\\", "/")

    _, _, _, out = _run(variant.format(base=base))

    expected = f"{canonical}/LTX-Director-2k_base"
    assert out == expected, f"variant={variant!r} produced {out!r}, expected {expected!r}"


@pytest.mark.parametrize("variant", PASTE_VARIANTS)
def test_output_directory_never_has_trailing_separator(tmp_path: Path, variant: str):
    """Belt-and-suspenders: regardless of which variation the user pastes,
    ``output_directory`` (slot 3 of the returns tuple) must never end with
    ``/`` or ``\\``. A trailing separator here is exactly what triggers the
    downstream ``undefined/`` phantom-folder bug."""
    base = tmp_path / "bear"
    base.mkdir()

    _, _, _, out = _run(variant.format(base=base))

    assert not out.endswith(("/", "\\")), f"variant={variant!r} → {out!r}"


@pytest.mark.parametrize("variant", PASTE_VARIANTS)
def test_write_file_path_clean_across_paste_variations(tmp_path: Path, variant: str):
    """``WRITE_file_path`` (slot 0) is the string downstream savers actually
    consume. It must never contain ``//``, ``undefined``, or a trailing
    separator regardless of how the base was pasted."""
    base = tmp_path / "bear"
    base.mkdir()

    file_path, _, _, _ = _run(variant.format(base=base))

    assert "//" not in file_path, f"variant={variant!r} → {file_path!r}"
    assert "undefined" not in file_path, f"variant={variant!r} → {file_path!r}"
    assert not file_path.endswith(("/", "\\")), f"variant={variant!r} → {file_path!r}"
    assert file_path.endswith(".%04d.exr"), f"variant={variant!r} → {file_path!r}"


def test_no_subfolders_also_canonicalises_trailing(tmp_path: Path):
    """With ``no_subfolders=True``, ``output_directory`` IS the (normalised)
    base path. The trailing-sep strip must still apply, otherwise no-subfolders
    mode would still leak ``undefined/`` downstream."""
    base = tmp_path / "bear"
    base.mkdir()
    canonical = str(base).replace("\\", "/")

    _, _, _, with_slash = _run(f"{base}\\", no_subfolders=True)
    _, _, _, without_slash = _run(str(base), no_subfolders=True)

    assert with_slash == without_slash == canonical


def test_blank_ai_method_yields_no_dangling_underscore(tmp_path: Path):
    """Companion regression for the same session's fix — verifies the
    empty-piece filter so ``ai_method=""`` doesn't produce ``shot_.exr``."""
    base = tmp_path / "bear"
    base.mkdir()

    _, name, _, _ = _run(str(base), shot_name="shot", ai_method="", extension=".exr")

    assert name == "shot.exr"


def test_blank_ai_method_with_versioning_skips_double_underscore(tmp_path: Path):
    """Same rule with versioning on: a blank ``ai_method`` between
    ``shot_name`` and ``vNNN`` must not produce ``shot__v003.exr``."""
    base = tmp_path / "bear"
    base.mkdir()

    _, name, _, _ = _run(
        str(base), shot_name="shot", ai_method="",
        version=3, disable_versioning=False, extension=".exr",
    )

    assert name == "shot_v003.exr"


def test_blank_ai_method_does_not_create_phantom_subfolder(tmp_path: Path):
    """With ``ai_method`` blank and subfolders enabled, the directory chain
    is just ``base/shot_name`` — no empty middle level."""
    base = tmp_path / "bear"
    base.mkdir()
    canonical = str(base).replace("\\", "/")

    _, _, _, out = _run(str(base), shot_name="shot", ai_method="")

    assert out == f"{canonical}/shot"
