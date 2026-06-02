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


def test_diff_picks_partitions_working_comparison_and_shared() -> None:
    proc = run_node_scenario(
        textwrap.dedent(
            """
            import assert from "node:assert/strict";
            import { diffPicks } from "./web/sidebar/snapshot_diff.js";

            const r = diffPicks(["x", "y"], ["y", "z"]);
            assert.deepEqual(r.onlyWorking, ["x"]);
            assert.deepEqual(r.onlyComparison, ["z"]);
            assert.deepEqual(r.shared, ["y"]);
            """
        )
    )
    assert proc.returncode == 0, proc.stderr


def test_diff_workflows_classifies_comparison_workflows_by_presence() -> None:
    proc = run_node_scenario(
        textwrap.dedent(
            """
            import assert from "node:assert/strict";
            import { diffWorkflows } from "./web/sidebar/snapshot_diff.js";

            const working = {
                directories: {
                    Basics: { workflows: { txt2img: { graph: { a: 1 } } }, directories: {} },
                },
            };
            const comparison = {
                directories: {
                    Basics: {
                        workflows: {
                            txt2img: { graph: { a: 1 } },
                            img2img: { graph: { b: 2 } },
                        },
                        directories: {},
                    },
                },
            };

            const status = diffWorkflows(working, comparison);
            // present only in the comparison -> green/new
            assert.equal(status["Basics/img2img"], "new");
            // present in both, identical graph -> plain/same
            assert.equal(status["Basics/txt2img"], "same");
            """
        )
    )
    assert proc.returncode == 0, proc.stderr


def test_diff_workflows_compares_graph_ignoring_saved_at_and_key_order() -> None:
    proc = run_node_scenario(
        textwrap.dedent(
            """
            import assert from "node:assert/strict";
            import { diffWorkflows } from "./web/sidebar/snapshot_diff.js";

            const working = { directories: { D: { workflows: {
                changed:   { graph: { nodes: [1, 2] }, savedAt: "2026-01-01T00:00:00Z" },
                unchanged: { graph: { a: 1, b: 2 },    savedAt: "2026-01-01T00:00:00Z" },
            }, directories: {} } } };
            const comparison = { directories: { D: { workflows: {
                changed:   { graph: { nodes: [1, 2, 3] }, savedAt: "2026-05-29T00:00:00Z" },
                unchanged: { graph: { b: 2, a: 1 },       savedAt: "2026-05-29T12:34:56Z" },
            }, directories: {} } } };

            const status = diffWorkflows(working, comparison);
            // graph content differs -> red/diff
            assert.equal(status["D/changed"], "diff");
            // only savedAt + key order differ -> plain/same (savedAt is volatile)
            assert.equal(status["D/unchanged"], "same");
            """
        )
    )
    assert proc.returncode == 0, proc.stderr
