from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_compare_load_path_returns_before_applysnapshot() -> None:
    # Spec #181 hard constraint: "no applySnapshot() call on the compare path
    # (static guard test)." The Load dialog doubles as the read-only compare
    # chooser via `onChoose`; both load functions MUST short-circuit (return) in
    # onChoose mode BEFORE reaching the destructive applySnapshot. Pin that
    # ordering so a future refactor can't silently reintroduce a destructive
    # load on the compare path.
    src = (REPO_ROOT / "web" / "sidebar" / "modals.js").read_text(encoding="utf-8")
    for fn in ("doNamedLoad", "doAutosaveRestore"):
        fn_start = src.index(f"async function {fn}(")
        choose_idx = src.index("if (onChoose)", fn_start)
        # Match the actual call (with paren), not the word in a comment.
        apply_idx = src.index("applySnapshot(", fn_start)
        assert choose_idx < apply_idx, (
            f"{fn}: the onChoose compare-mode guard must precede applySnapshot"
        )
        assert "return;" in src[choose_idx:apply_idx], (
            f"{fn}: the onChoose guard must return before reaching applySnapshot"
        )


def test_enter_compare_mode_uses_readonly_chooser() -> None:
    # The compare ENTRY path opens the existing Load dialog in read-only choose
    # mode (onChoose), never the destructive apply path.
    src = (REPO_ROOT / "web" / "sidebar" / "tree.js").read_text(encoding="utf-8")
    start = src.index("function enterCompareMode(")
    body = src[start:start + 1400]
    assert "showLoadSnapshotDialog(" in body
    assert "onChoose:" in body, (
        "enterCompareMode must open the Load dialog in read-only choose mode"
    )


def _compare_guard_block(src: str) -> str:
    # The read-only capture-guard block lives between the NAV_ALLOW declaration
    # and the "Forward declarations" marker that follows the `if (compare)` block.
    start = src.index("const NAV_ALLOW =")
    return src[start:src.index("// Forward declarations", start)]


def test_compare_panel_guard_blocks_drag_drop_fully() -> None:
    # #181 read-only guarantee: the comparison panel must neutralize cross-panel
    # drag-and-drop, not just dragstart. A drag begun in the live panel and
    # dropped on a comparison folder row would otherwise persist a live
    # move/archive (handleDndDrop -> persistMutation on the live store).
    guard = _compare_guard_block(
        (REPO_ROOT / "web" / "sidebar" / "tree.js").read_text(encoding="utf-8")
    )
    for evt in ("click", "contextmenu", "dragstart", "dragover", "drop"):
        assert f'addEventListener("{evt}"' in guard, (
            f"compare read-only guard is missing a capture handler for '{evt}'"
        )
    # Every guard listener must be capture-phase (third arg `true`).
    assert guard.count(", true)") >= 5, (
        "all compare-guard listeners must be registered capture-phase"
    )


def test_compare_nav_allow_excludes_group_mode_toggle() -> None:
    # The grouping mode toggle writes a shared global key (GROUP_MODE_KEY), so it
    # must NOT be allow-listed in the read-only comparison panel.
    src = (REPO_ROOT / "web" / "sidebar" / "tree.js").read_text(encoding="utf-8")
    start = src.index("const NAV_ALLOW =")
    nav = src[start:src.index(";", start)]
    assert "koolook-mode-toggle" not in nav, (
        "grouping mode toggle writes a shared global key; drop it from NAV_ALLOW"
    )


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
