"""Guards for Cursor discovery mirrors of repo-local skills.

`.claude/skills/<name>/SKILL.md` is canonical. When a matching
`.cursor/skills/<name>/SKILL.md` discovery mirror is maintained for Cursor
skill discovery, the two files must stay byte-identical so edits cannot
drift silently.
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Skills that intentionally keep a Cursor discovery mirror in sync with the
# Claude canonical copy. Older thin/divergent mirrors (e.g. add-external-fork)
# are out of scope until they opt into this contract.
BYTE_IDENTICAL_CURSOR_MIRRORS = (
    "review-pr-comfy-cursor",
    "review-pr-fast-comfy-cursor",
)


def test_cursor_discovery_mirrors_match_claude_canonical() -> None:
    for name in BYTE_IDENTICAL_CURSOR_MIRRORS:
        canonical = REPO_ROOT / ".claude" / "skills" / name / "SKILL.md"
        mirror = REPO_ROOT / ".cursor" / "skills" / name / "SKILL.md"
        assert canonical.is_file(), f"missing canonical skill: {canonical}"
        assert mirror.is_file(), f"missing Cursor mirror: {mirror}"
        assert canonical.read_bytes() == mirror.read_bytes(), (
            f"Cursor mirror drifted from canonical for {name}. "
            f"Copy .claude/skills/{name}/SKILL.md → "
            f".cursor/skills/{name}/SKILL.md"
        )
