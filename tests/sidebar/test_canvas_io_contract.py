from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_sidebar_workflow_load_does_not_bind_comfy_filename_namespace() -> None:
    source = (REPO_ROOT / "web/sidebar/canvas_io.js").read_text(encoding="utf-8")
    load_fn = source[source.index("export async function loadWorkflowOntoCanvas"):]
    load_fn = load_fn[:load_fn.index("export async function dropPlaceholdersForPacks")]

    assert "app.loadGraphData(graph, true, true);" in load_fn
    assert "app.loadGraphData(graph, true, true, wfName" not in load_fn
    assert "workflows/${wfName}.json" not in load_fn
