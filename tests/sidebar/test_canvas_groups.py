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


def test_selection_groups_keep_groups_that_cover_selected_nodes() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { groupsForSelectedNodes } from "./web/sidebar/canvas_groups.js";

        const fullGraph = {
          nodes: [
            { id: 1, pos: [20, 20], size: [120, 80] },
            { id: 2, pos: [320, 20], size: [120, 80] },
            { id: 3, pos: [700, 20], size: [120, 80] },
          ],
          groups: [
            { title: "Koolook Input", bounding: [0, 0, 220, 140], color: "#335" },
            { title: "Koolook Output", pos: [300, 0], size: [220, 140], font_size: 24 },
            { title: "Unselected", bounding: [680, 0, 220, 140] },
          ],
        };

        const groups = groupsForSelectedNodes(fullGraph, new Set(["1", "2"]));

        assert.deepEqual(groups, [
          { title: "Koolook Input", bounding: [0, 0, 220, 140], color: "#335" },
          { title: "Koolook Output", pos: [300, 0], size: [220, 140], font_size: 24 },
        ]);

        groups[0].title = "mutated";
        assert.equal(fullGraph.groups[0].title, "Koolook Input");
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


def test_translate_groups_moves_serialized_group_rectangles() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { translateGroups } from "./web/sidebar/canvas_groups.js";

        const original = [
          { title: "Koolook Input", bounding: [10, 20, 220, 140] },
          { title: "Koolook Output", pos: [300, 400], size: [220, 140] },
        ];

        const translated = translateGroups(original, 50, -10);

        assert.deepEqual(translated, [
          { title: "Koolook Input", bounding: [60, 10, 220, 140] },
          { title: "Koolook Output", pos: [350, 390], size: [220, 140] },
        ]);
        assert.deepEqual(original, [
          { title: "Koolook Input", bounding: [10, 20, 220, 140] },
          { title: "Koolook Output", pos: [300, 400], size: [220, 140] },
        ]);
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr
