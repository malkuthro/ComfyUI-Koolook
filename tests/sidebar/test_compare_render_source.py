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


def test_picks_render_source_override_is_read_only() -> None:
    # Compare mode feeds the comparison panel a snapshot's picks via a
    # render-source override. It must SERVE the override to loadUserPicks but
    # never persist it — clearing returns to live picks (empty under Node,
    # which has no localStorage). Proves the non-destructive guarantee.
    proc = run_node_scenario(
        textwrap.dedent(
            """
            import assert from "node:assert/strict";
            import {
                setPicksRenderSource,
                clearPicksRenderSource,
                loadUserPicks,
            } from "./web/sidebar/picks_store.js";

            assert.deepEqual(loadUserPicks(), []);              // live (no localStorage in Node)
            setPicksRenderSource(["NodeA", "NodeB"]);
            assert.deepEqual(loadUserPicks(), ["NodeA", "NodeB"]);  // override served
            clearPicksRenderSource();
            assert.deepEqual(loadUserPicks(), []);              // live again, untouched
            """
        )
    )
    assert proc.returncode == 0, proc.stderr


def test_workflows_render_source_override_does_not_mutate_live() -> None:
    proc = run_node_scenario(
        textwrap.dedent(
            """
            import assert from "node:assert/strict";
            import {
                setWorkflowsRenderSource,
                clearWorkflowsRenderSource,
                dirOf,
                getAllWorkflowsForExport,
            } from "./web/sidebar/workflows_store.js";

            const liveBefore = JSON.stringify(getAllWorkflowsForExport());
            setWorkflowsRenderSource({
                directories: { Cmp: { workflows: { w: { graph: {} } }, directories: {} } },
            });
            assert.ok(dirOf(["Cmp"]));                          // reads resolve against the override
            clearWorkflowsRenderSource();
            // The live export (workflowsCache) is never touched by the override.
            assert.equal(JSON.stringify(getAllWorkflowsForExport()), liveBefore);
            assert.equal(dirOf(["Cmp"]), undefined);            // override gone -> not in live store
            """
        )
    )
    assert proc.returncode == 0, proc.stderr
