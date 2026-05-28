"""Unit tests for the shared ``koolook_versioning`` helpers.

These pin the single source of truth for the ``vNNN`` convention shared by
``EasyAIPipeline`` and ``Easy_VideoCombine``. The load-bearing contract is
that a wired/typed version STRING wins verbatim over the legacy INT widget
and over ``disable_versioning`` — that's what lets one "global version"
source drive every node consistently.
"""
from __future__ import annotations

from koolook_versioning import normalize_version_token, resolve_version_token


class TestNormalizeVersionToken:
    def test_empty_and_none(self):
        assert normalize_version_token("") == ""
        assert normalize_version_token("   ") == ""
        assert normalize_version_token(None) == ""

    def test_sentinels_become_empty(self):
        assert normalize_version_token("undefined") == ""
        assert normalize_version_token("Undefined") == ""
        assert normalize_version_token("NULL") == ""
        assert normalize_version_token("None") == ""

    def test_verbatim_values_pass_through(self):
        assert normalize_version_token("v001") == "v001"
        assert normalize_version_token("final") == "final"
        assert normalize_version_token("take_3") == "take_3"
        # A substring match of a sentinel must NOT trigger.
        assert normalize_version_token("v_undefined_3") == "v_undefined_3"

    def test_surrounding_whitespace_and_quotes_stripped(self):
        assert normalize_version_token("  v002  ") == "v002"
        assert normalize_version_token('"v002"') == "v002"
        assert normalize_version_token("'v002'") == "v002"

    def test_control_chars_stripped(self):
        assert normalize_version_token("v0\n01") == "v001"
        assert normalize_version_token("v001\t") == "v001"

    def test_path_separators_flattened(self):
        assert normalize_version_token("v001/extra") == "v001_extra"
        assert normalize_version_token("a\\b") == "a_b"


class TestResolveVersionToken:
    def test_disable_versioning_is_master_off(self):
        assert resolve_version_token("v007", disable_versioning=True) == ""
        assert resolve_version_token("5", disable_versioning=True) == ""

    def test_empty_or_sentinel_returns_empty(self):
        assert resolve_version_token("") == ""
        assert resolve_version_token("   ") == ""
        assert resolve_version_token("undefined") == ""

    def test_bare_number_becomes_vnnn(self):
        # Back-compat with the old INT widget + convenience when typing.
        assert resolve_version_token("3") == "v003"
        assert resolve_version_token("12") == "v012"
        # A saved workflow's stored INT arrives as a number, not a string.
        assert resolve_version_token(5) == "v005"

    def test_token_used_verbatim(self):
        assert resolve_version_token("v001") == "v001"
        assert resolve_version_token("final") == "final"
        assert resolve_version_token("v2_take3") == "v2_take3"
