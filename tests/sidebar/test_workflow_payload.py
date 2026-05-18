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


def test_sidebar_workflow_load_clone_can_use_stable_temp_id() -> None:
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

        const first = cloneWorkflowForTemporaryLoad(stored, "Shots\\u0000Bear\\u0000Render");
        const second = cloneWorkflowForTemporaryLoad(stored, "Shots\\u0000Bear\\u0000Render");
        const other = cloneWorkflowForTemporaryLoad(stored, "Shots\\u0000Bear\\u0000Archived");

        assert.equal(first.id, second.id);
        assert.notEqual(first.id, other.id);
        assert.notEqual(first.id, stored.id);
        assert.match(first.id, /^[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-8[0-9a-f]{3}-[0-9a-f]{12}$/i);
        assert.equal(stored.id, "original-workflow-id");
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


def test_sidebar_workflow_load_clone_treats_empty_key_as_stable() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { cloneWorkflowForTemporaryLoad } from "./web/sidebar/workflow_payload.js";

        const first = cloneWorkflowForTemporaryLoad({ id: "original" }, "");
        const second = cloneWorkflowForTemporaryLoad({ id: "original" }, "");
        const random = cloneWorkflowForTemporaryLoad({ id: "original" });

        assert.equal(first.id, second.id);
        assert.notEqual(first.id, random.id);
        assert.match(first.id, /^[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-8[0-9a-f]{3}-[0-9a-f]{12}$/i);
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr
