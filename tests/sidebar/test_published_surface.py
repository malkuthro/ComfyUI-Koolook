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


def test_infer_setup_surface_from_koolook_groups() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { inferSetupSurface } from "./web/sidebar/published_surface.js";

        const surface = inferSetupSurface({
          nodes: [
            { id: 12, type: "Load Image", title: "Source image", pos: [40, 40], size: [180, 80] },
            { id: 20, type: "Preview Image", title: "Preview", pos: [420, 40], size: [180, 80] },
            { id: 30, type: "Note", title: "Outside", pos: [800, 40], size: [180, 80] },
          ],
          groups: [
            { title: "Koolook Input", bounding: [20, 20, 240, 140] },
            { title: "Koolook Output", pos: [400, 20], size: [240, 140] },
          ],
        });

        assert.deepEqual(surface, {
          sourceInputs: [{
            group: "Koolook Input",
            nodes: [{ id: "12", type: "Load Image", title: "Source image" }],
          }],
          outputs: [{
            group: "Koolook Output",
            nodes: [{ id: "20", type: "Preview Image", title: "Preview" }],
          }],
          controls: [],
        });
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


def test_infer_setup_surface_ignores_malformed_group_geometry() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { inferSetupSurface } from "./web/sidebar/published_surface.js";

        const surface = inferSetupSurface({
          nodes: [
            { id: 12, type: "Load Image", title: "Source image", pos: [40, 40], size: [180, 80] },
          ],
          groups: [
            { title: "Koolook Input", bounding: ["bad", 20, 240, 140] },
          ],
        });

        assert.deepEqual(surface.sourceInputs, []);
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr
