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


def test_sidebar_selection_serialization_preserves_selected_groups() -> None:
    source = (REPO_ROOT / "web/sidebar/canvas_io.js").read_text(encoding="utf-8")
    selection_fn = source[source.index("export function serializeSelection"):]
    selection_fn = selection_fn[:selection_fn.index("export function canvasIsNonEmpty")]

    assert "groupsForSelectedNodes" in source
    assert "groups: groupsForSelectedNodes(full, selectedKeys)" in selection_fn
    assert "groups: []" not in selection_fn


def test_sidebar_workflow_insert_restores_saved_groups() -> None:
    source = (REPO_ROOT / "web/sidebar/canvas_io.js").read_text(encoding="utf-8")
    insert_fn = source[source.index("export async function insertWorkflowOntoCanvas"):]
    insert_fn = insert_fn[:insert_fn.index("export function getSelectedNodeTypes")]

    assert "translateGroups" in source
    assert "addWorkflowGroupsToCanvas" in source
    assert "const placement = placeBboxAtCanvasCenter(nodesRaw);" in insert_fn
    assert "addWorkflowGroupsToCanvas(clone.groups, placement.dx, placement.dy);" in insert_fn


def test_api_prompt_capture_restore_failure_blocks_publish() -> None:
    source = (REPO_ROOT / "web/sidebar/canvas_io.js").read_text(encoding="utf-8")
    capture_fn = source[source.index("export async function captureWorkflowApiPrompt"):]
    capture_fn = capture_fn[:capture_fn.index("export async function loadWorkflowOntoCanvas")]

    assert "restoreError" in capture_fn
    assert "could not restore the current canvas" in capture_fn
    assert "console.warn(\"[Koolook] failed to restore canvas after API prompt capture:" not in capture_fn
