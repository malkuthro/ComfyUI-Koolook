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

from k_ai_pipeline import (
    EasyAIPipeline,
    _normalize_base_path,
    _sanitize_segment,
    _strip_control_chars,
)


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


# ---------------------------------------------------------------------------
# _sanitize_segment — path-escape regression coverage
# ---------------------------------------------------------------------------
#
# os.path.join('n:/safe', '/oops') returns '/oops' (or 'C:/oops' on Windows) —
# its 'last absolute path wins' rule means an absolute-looking shot_name or
# ai_method could escape the user's intended base_directory_path entirely.
# A reviewer caught this regression in PR #156 (the original lstrip('/') was
# dropped during the refactor). These tests pin the sanitizer so it can't
# come back.


class TestSanitizeSegment:
    def test_clean_input_unchanged(self):
        assert _sanitize_segment("shot_v1") == "shot_v1"
        assert _sanitize_segment("RTX-upscale") == "RTX-upscale"

    def test_leading_forward_slash_stripped(self):
        assert _sanitize_segment("/oops") == "oops"

    def test_leading_backslash_stripped(self):
        assert _sanitize_segment("\\oops") == "oops"

    def test_multiple_leading_separators_stripped(self):
        assert _sanitize_segment("///oops") == "oops"
        assert _sanitize_segment("\\\\\\oops") == "oops"
        assert _sanitize_segment("/\\/oops") == "oops"

    def test_drive_prefix_handled(self):
        """On Windows ``os.path.splitdrive`` peels ``C:`` off; on POSIX it's a
        no-op (no drive concept), so the colon survives — that's fine because
        ``os.path.join`` on POSIX has no special handling for ``C:`` and
        treats it as a regular folder name, which can't escape the base."""
        import os as _os
        result = _sanitize_segment("C:/Windows")
        if _os.name == "nt":
            assert result == "Windows"
        else:
            # POSIX: drive not recognised, but no escape risk either.
            assert result == "C:/Windows"

    def test_empty_string_passes_through(self):
        assert _sanitize_segment("") == ""

    def test_only_separators_collapses_to_empty(self):
        assert _sanitize_segment("///") == ""
        assert _sanitize_segment("\\") == ""


@pytest.mark.parametrize("malicious", [
    "/oops",
    "\\oops",
    "C:/Windows/junk",
    "/etc/passwd",
])
def test_absolute_shot_name_cannot_escape_base(tmp_path: Path, malicious: str):
    """The reviewer-flagged blocker in PR #156: absolute ``shot_name`` values
    used to leak through ``os.path.join`` and replace the base path entirely.
    After ``_sanitize_segment`` they get stripped down to relative form, so
    ``output_directory`` stays under the user's intended base."""
    base = tmp_path / "safe"
    base.mkdir()
    canonical_base = str(base).replace("\\", "/")

    _, _, _, out = _run(str(base), shot_name=malicious)

    assert out.startswith(canonical_base), (
        f"shot_name={malicious!r} escaped base — out={out!r}, base={canonical_base!r}"
    )


@pytest.mark.parametrize("malicious", [
    "/oops",
    "\\oops",
    "C:/Windows/junk",
    "/etc/passwd",
])
def test_absolute_ai_method_cannot_escape_base(tmp_path: Path, malicious: str):
    """Same escape vector through ``ai_method`` (the middle segment). The
    sanitizer applies symmetrically."""
    base = tmp_path / "safe"
    base.mkdir()
    canonical_base = str(base).replace("\\", "/")

    _, _, _, out = _run(str(base), shot_name="shot", ai_method=malicious)

    assert out.startswith(canonical_base), (
        f"ai_method={malicious!r} escaped base — out={out!r}, base={canonical_base!r}"
    )


def test_absolute_shot_name_does_not_leak_into_filename(tmp_path: Path):
    """The filename builder uses the sanitized segments too, so a leading
    separator can't produce ``/shot_v001.exr`` (which on some platforms is
    interpreted as a path, on others is just an invalid filename)."""
    base = tmp_path / "safe"
    base.mkdir()

    file_path, name, _, _ = _run(str(base), shot_name="/oops")

    assert not name.startswith(("/", "\\")), f"filename leaked separator: {name!r}"
    assert "/oops" not in name, f"filename retained absolute prefix: {name!r}"


def test_no_subfolders_flattens_embedded_separators_in_shot_name(tmp_path: Path):
    """Real-world maintainer report: chained EasyAIPipeline workflow where the
    upstream node feeds ``shot_name = "job/shot"``. With ``no_subfolders=true``,
    embedded separators used to silently re-create subfolders via the filename
    concat — output was ``base/job/shot.ext`` instead of the flat ``base/<flat>.ext``
    the toggle promises. After fix: separators in shot_name flatten to ``_`` in
    the filename, while the directory stays at ``base`` (toggle honored)."""
    base = tmp_path / "bear"
    base.mkdir()
    canonical = str(base).replace("\\", "/")

    file_path, name, _, output_directory = _run(
        str(base),
        shot_name="ComfyUI-working-folder/LTX-Director-2K",
        no_subfolders=True,
    )

    assert output_directory == canonical, f"no_subfolders=True should keep dir=base; got {output_directory!r}"
    assert name == "ComfyUI-working-folder_LTX-Director-2K.%04d.exr", f"unexpected name: {name!r}"
    assert "/" not in name and "\\" not in name, f"separator in filename: {name!r}"
    assert file_path == f"{canonical}/ComfyUI-working-folder_LTX-Director-2K.%04d.exr"


def test_filename_never_contains_path_separators(tmp_path: Path):
    """Belt-and-suspenders: regardless of toggle state or input, the filename
    portion of WRITE_file_path must never contain `/` or `\\` — those are
    invalid in filenames on every OS. Subfolder semantics live in the
    directory portion, not the filename."""
    base = tmp_path / "safe"
    base.mkdir()

    for cfg in [
        dict(shot_name="Project/v1", ai_method="", no_subfolders=True),
        dict(shot_name="Project/v1", ai_method="", no_subfolders=False),
        dict(shot_name="a/b/c", ai_method="d/e", no_subfolders=True),
        dict(shot_name="a/b/c", ai_method="d/e", no_subfolders=False),
        dict(shot_name="back\\slash", ai_method="", no_subfolders=True),
    ]:
        file_path, name, _, _ = _run(str(base), **cfg)
        assert "/" not in name and "\\" not in name, (
            f"separator leaked into filename for cfg={cfg!r}: name={name!r}"
        )


def test_no_subfolders_false_still_uses_slashes_as_subfolders(tmp_path: Path):
    """With ``no_subfolders=False`` (default), embedded slashes in shot_name
    intentionally become nested subfolders — this is how users organize
    output into project/shot hierarchies. The filename stays flat regardless."""
    base = tmp_path / "bear"
    base.mkdir()
    canonical = str(base).replace("\\", "/")

    _, name, _, output_directory = _run(
        str(base),
        shot_name="Project/v1",
        ai_method="upscale",
        version=3, disable_versioning=False,
        no_subfolders=False,
    )

    assert output_directory == f"{canonical}/Project/v1/upscale/v003"
    assert name == "Project_v1_upscale_v003.%04d.exr"


def test_no_subfolders_true_with_versioning_keeps_version_folder(tmp_path: Path):
    """Maintainer-clarified behavior for ``no_subfolders=True``: shot_name and
    ai_method drop out of the directory path entirely (they only appear in
    the filename), but the version folder (``v###``) IS still added when
    ``disable_versioning`` is off. Versioned outputs stay organised under
    ``base/v###/`` even with the toggle on."""
    base = tmp_path / "bear"
    base.mkdir()
    canonical = str(base).replace("\\", "/")

    file_path, name, _, output_directory = _run(
        str(base),
        shot_name="shot",
        ai_method="upscale",
        version=5, disable_versioning=False,
        no_subfolders=True,
    )

    assert output_directory == f"{canonical}/v005", (
        f"Expected base/v005 (only version folder); got {output_directory!r}"
    )
    # Filename still gets all the pieces joined with underscores
    assert name == "shot_upscale_v005.%04d.exr"
    assert file_path == f"{canonical}/v005/shot_upscale_v005.%04d.exr"


def test_no_subfolders_true_without_versioning_is_flat_base(tmp_path: Path):
    """Companion to the above: ``no_subfolders=True`` + ``disable_versioning=True``
    means truly flat output — just ``base/<filename>``, no subfolders at all."""
    base = tmp_path / "bear"
    base.mkdir()
    canonical = str(base).replace("\\", "/")

    _, name, _, output_directory = _run(
        str(base),
        shot_name="shot",
        ai_method="upscale",
        version=5, disable_versioning=True,
        no_subfolders=True,
    )

    assert output_directory == canonical
    assert name == "shot_upscale.%04d.exr"


# ---------------------------------------------------------------------------
# Control-char stripping — newlines from upstream Text Multiline widgets
# ---------------------------------------------------------------------------
#
# An upstream node (e.g. WAS's `Text Multiline`) can feed strings that contain
# embedded newlines / tabs / carriage returns. Those can't legally appear in
# any filesystem path — they'd silently break the actual file save, and the
# preview widget rendered them as visually broken multi-line output. Strip
# at the helper level so they never reach the path build.


class TestStripControlChars:
    def test_strips_newline(self):
        assert _strip_control_chars("foo\nbar") == "foobar"

    def test_strips_carriage_return(self):
        assert _strip_control_chars("foo\rbar") == "foobar"

    def test_strips_tab(self):
        assert _strip_control_chars("foo\tbar") == "foobar"

    def test_strips_all_mixed(self):
        assert _strip_control_chars("foo\r\n\tbar\nbaz") == "foobarbaz"

    def test_preserves_spaces(self):
        """Spaces are valid in path segments; only newline/CR/tab get stripped."""
        assert _strip_control_chars("foo bar") == "foo bar"

    def test_empty_passes_through(self):
        assert _strip_control_chars("") == ""


def test_shot_name_with_embedded_newline_does_not_break_path(tmp_path: Path):
    """The maintainer-reported case: upstream feeds shot_name with a stray
    newline (e.g. paragraph break in a Text Multiline). Pre-fix, this leaked
    through to the preview AND would have broken the save with a literal
    line-break in the path. Fix: _sanitize_segment strips control chars at
    the source."""
    base = tmp_path / "bear"
    base.mkdir()

    file_path, name, _, output_directory = _run(
        str(base),
        shot_name="ComfyUI-working-folder/LTX-Director-2K\n_0",
        no_subfolders=True,
    )

    assert "\n" not in file_path, f"newline leaked into file_path: {file_path!r}"
    assert "\n" not in name, f"newline leaked into name: {name!r}"
    assert "\n" not in output_directory, f"newline leaked into output_directory: {output_directory!r}"


def test_base_path_with_embedded_newline_normalised(tmp_path: Path):
    """Same defense at the base_directory_path layer — Text Multiline
    sometimes appends a trailing \\n that survived the previous fix."""
    base = tmp_path / "bear"
    base.mkdir()
    canonical = str(base).replace("\\", "/")

    _, _, _, output_directory = _run(
        f"{base}\n",
        shot_name="shot",
        no_subfolders=True,
    )

    assert "\n" not in output_directory
    assert output_directory == canonical


def test_extension_with_embedded_newline_stripped(tmp_path: Path):
    """Extension is the third path-bound string input; stray newlines there
    were also leaking into the filename."""
    base = tmp_path / "bear"
    base.mkdir()

    _, name, _, _ = _run(
        str(base),
        shot_name="shot",
        extension=".exr\n",
    )

    assert "\n" not in name
    assert name == "shot.exr"


@pytest.mark.parametrize("dirty,expected", [
    (".%04d.exr ",   ".%04d.exr"),   # trailing space
    (" .%04d.exr",   ".%04d.exr"),   # leading space
    (" .%04d.exr ",  ".%04d.exr"),   # both
    (".%04d .exr",   ".%04d.exr"),   # internal space
    (".%04d.exr\t",  ".%04d.exr"),   # trailing tab
    (".%04d.exr\xa0", ".%04d.exr"),  # non-breaking space (common copy-paste invisible char)
])
def test_extension_whitespace_stripped(tmp_path: Path, dirty: str, expected: str):
    """Maintainer-reported: ComfyUI-HQ-Image-Save's ``save_images`` validates
    via ``os.path.splitext(filepath)[1].lower() != '.exr'``. A single trailing
    space on the extension widget made splitext return ``.exr `` (with space),
    which failed the strict equality and aborted the save. Extension must have
    NO whitespace at all (internal or surrounding) — it's a strict suffix
    field, not a label."""
    base = tmp_path / "bear"
    base.mkdir()

    _, name, _, _ = _run(str(base), shot_name="shot", extension=dirty)

    assert name == f"shot{expected}", f"dirty={dirty!r}: got name={name!r}"
    # And the HQ-Image-Save check would pass:
    import os
    assert os.path.splitext(name)[1].lower() == ".exr"


@pytest.mark.parametrize("dirty,clean", [
    ("  shot  ", "shot"),       # surrounding spaces
    ("shot\t",   "shot"),       # trailing tab
    ("\tshot",   "shot"),       # leading tab
])
def test_shot_name_surrounding_whitespace_stripped(tmp_path: Path, dirty: str, clean: str):
    """Surrounding whitespace on shot_name (from sloppy upstream feeds) used
    to leak into the path and filename. ``_sanitize_segment`` now strips
    surrounding whitespace as well as control chars."""
    base = tmp_path / "safe"
    base.mkdir()
    canonical = str(base).replace("\\", "/")

    _, name, _, output_directory = _run(str(base), shot_name=dirty, no_subfolders=False)

    assert name == f"{clean}.%04d.exr"
    assert output_directory == f"{canonical}/{clean}"
