from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TREE_JS = REPO_ROOT / "web" / "sidebar" / "tree.js"
MODALS_JS = REPO_ROOT / "web" / "sidebar" / "modals.js"


def test_workflow_context_menu_exposes_publish_setup_flow() -> None:
    tree = TREE_JS.read_text(encoding="utf-8")
    modals = MODALS_JS.read_text(encoding="utf-8")

    assert "showPublishSetupModal," in tree
    assert "publishSavedWorkflowSetup(" in tree
    assert "captureWorkflowApiPrompt," in tree
    assert "captureApiPrompt: captureWorkflowApiPrompt," in tree
    assert "label: \"Publish setup…\"" in tree
    assert "export function showPublishSetupModal(" in modals
    assert "Test run" not in modals
    assert "onTestRun" not in modals


def test_publish_setup_modal_shows_inferred_group_surface() -> None:
    tree = TREE_JS.read_text(encoding="utf-8")
    modals = MODALS_JS.read_text(encoding="utf-8")

    assert 'import { inferSetupSurface } from "./published_surface.js";' in modals
    assert "visualGraph = null" in modals
    assert "Inferred app surface" in modals
    assert "Source breadcrumbs" in modals
    assert "Mode switch" in modals
    assert "Source fields" in modals
    assert "Output controls" in modals
    assert "Result fields" in modals
    assert "Koolook Input" in modals
    assert "Koolook Output" in modals
    assert "Advanced contract JSON" in modals
    assert "JSON.stringify({ inputs: [] }, null, 2)" in modals
    assert "JSON.stringify({ outputs: [] }, null, 2)" in modals
    assert "const visualGraph = getWorkflowGraph(dirPath, wfName);" in tree
    assert "visualGraph," in tree
