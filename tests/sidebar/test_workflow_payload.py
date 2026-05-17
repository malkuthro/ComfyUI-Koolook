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


def test_sidebar_workflow_load_clone_gets_fresh_id_without_mutating_store_copy() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { cloneWorkflowForTemporaryLoad } from "./web/sidebar/workflow_payload.js";

        const stored = {
          id: "original-workflow-id",
          nodes: [{ id: 1, type: "KSampler", widgets_values: [42] }],
          links: [],
          extra: { ds: { scale: 1, offset: [0, 0] } },
          version: 0.4,
        };

        const loaded = cloneWorkflowForTemporaryLoad(stored);

        assert.notEqual(loaded, stored);
        assert.equal(stored.id, "original-workflow-id");
        assert.notEqual(loaded.id, stored.id);
        assert.match(loaded.id, /^[0-9a-f-]{36}$/i);
        assert.deepEqual(loaded.nodes, stored.nodes);

        loaded.nodes[0].widgets_values[0] = 100;
        assert.equal(stored.nodes[0].widgets_values[0], 42);
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr
