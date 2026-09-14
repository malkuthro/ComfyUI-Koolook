from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_sidebar_workflow_load_binds_deduped_tab_name() -> None:
    source = (REPO_ROOT / "web/sidebar/canvas_io.js").read_text(encoding="utf-8")
    load_fn = source[source.index("export async function loadWorkflowOntoCanvas"):]
    load_fn = load_fn[:load_fn.index("export async function dropPlaceholdersForPacks")]

    # The tab is bound to the *resolved* (de-duped) name, never the raw
    # sidebar workflow name. Binding raw wfName reintroduces the autosave
    # 409 Conflict against persisted workflows/<name>.json files and the
    # duplicate-graph-id crash across open tabs (#166).
    assert "const tabName = resolveLoadedTabName(store, wfName);" in load_fn
    assert "app.loadGraphData(graph, true, true, tabName);" in load_fn
    assert "app.loadGraphData(graph, true, true, wfName" not in load_fn
    assert "workflows/${wfName}.json" not in load_fn

    # The stable draft id stays folder-qualified (dirPath in the key) so
    # same-named workflows in different sidebar folders keep distinct draft
    # identities, and includes the resolved tab name so two open tabs can
    # never share a graph id.
    assert 'const loadKey = [...dirPath, wfName, tabName].join("\\u0000");' in load_fn
    assert "cloneWorkflowForTemporaryLoad(sourceGraph, loadKey)" in load_fn

    # The 409 guard: de-dupe must consider persisted (non-temporary)
    # workflow files, not just open tabs.
    assert "persistedWorkflowNames" in source
    assert "wf?.isTemporary === true" in source


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


def test_api_prompt_capture_closes_temporary_workflow_tabs() -> None:
    source = (REPO_ROOT / "web/sidebar/canvas_io.js").read_text(encoding="utf-8")
    capture_fn = source[source.index("export async function captureWorkflowApiPrompt"):]
    capture_fn = capture_fn[:capture_fn.index("export async function loadWorkflowOntoCanvas")]

    # The capture loads graphs onto the canvas, which spawns throwaway
    # "Unsaved Workflow" tabs. It must snapshot the open workflows before
    # capture and close whatever is new afterward so a publish leaves no stray
    # tabs behind.
    assert "openWorkflowsSnapshot(store)" in capture_fn
    assert "closeWorkflowsOpenedDuringCapture(store, openBefore, originalWorkflow)" in capture_fn

    # Cleanup must go through Comfy's workflow store, stay guarded (the API is
    # version-specific), and never warn the user about unsaved throwaway tabs.
    assert "app?.extensionManager?.workflow" in source
    assert "typeof store.closeWorkflow !== \"function\"" in source
    assert "warnIfUnsaved: false" in source


def test_api_prompt_capture_warns_when_workflow_discovery_fails() -> None:
    source = (REPO_ROOT / "web/sidebar/canvas_io.js").read_text(encoding="utf-8")

    assert "could not inspect Comfy workflow store for publish cleanup" in source
    assert "could not snapshot open workflows for publish cleanup" in source
    assert "could not read active workflow for publish cleanup" in source
