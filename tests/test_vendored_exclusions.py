"""Guards for vendored-upstream exclusions.

Only the WhatDreamsCost 2.0.2 files added as upstream replicas should be hidden
from GitHub language stats / CodeQL. Older fork versions carry Koolook-authored
customizations, so broad ``versions/*`` exclusions would hide real review
surface.
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_whatdreamscost_vendored_exclusions_are_version_scoped() -> None:
    gitattributes = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    codeql_config = (
        REPO_ROOT / ".github" / "codeql" / "codeql-config.yml"
    ).read_text(encoding="utf-8")

    for text in (gitattributes, codeql_config):
        assert "versions/*/prompt_relay.py" not in text
        assert "versions/*/patches.py" not in text
        assert "versions/v2_0_2/prompt_relay.py" in text
        assert "versions/v2_0_2/patches.py" in text
