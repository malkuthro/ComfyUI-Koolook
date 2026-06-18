from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TREE_JS = REPO_ROOT / "web" / "sidebar" / "tree.js"


def test_compare_panels_disable_update_footer() -> None:
    text = TREE_JS.read_text(encoding="utf-8")

    assert "showUpdateFooter: false" in text
    assert "if (showUpdateFooter) {" in text
