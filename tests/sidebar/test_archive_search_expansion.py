from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_node_scenario(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", "--input-type=module"],
        input=source,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_archive_folder_is_not_forced_open_during_search() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { resolveFolderExpanded } from "./web/sidebar/tree_expansion.js";

        assert.equal(
          resolveFolderExpanded({
            forceExpanded: true,
            iconKind: "archive",
            startExpanded: false,
          }),
          false,
        );

        assert.equal(
          resolveFolderExpanded({
            forceExpanded: true,
            iconKind: "folder",
            startExpanded: false,
          }),
          true,
        );

        assert.equal(
          resolveFolderExpanded({
            forceExpanded: true,
            iconKind: "archive",
            isPinned: true,
            startExpanded: false,
          }),
          true,
        );
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr
