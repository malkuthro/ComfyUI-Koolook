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


def test_publish_saved_workflow_sends_registry_payload() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { publishSavedWorkflowSetup } from "./web/sidebar/published_setups.js";

        const graph = {
          nodes: [{ id: 12, inputs: [{ name: "text" }] }],
          links: [],
        };
        const apiPrompt = {
          "12": { class_type: "Text Multiline", inputs: { text: "published" } },
        };
        const calls = [];
        const result = await publishSavedWorkflowSetup({
          dirPath: ["Demos"],
          wfName: "Director",
          metadata: {
            id: "director-demo",
            title: "Director Demo",
            description: "A curated director workflow.",
            category: "Video",
            tags: "director, video",
            previewImage: "preview.png",
          },
          inputContract: {
            inputs: [{
              key: "prompt",
              label: "Prompt",
              type: "text",
              required: true,
              target: { node: "12", input: "text" },
            }],
          },
          outputContract: { outputs: [{ key: "preview", type: "image" }] },
          apiPrompt,
          getWorkflowGraph: (path, name) => {
            assert.deepEqual(path, ["Demos"]);
            assert.equal(name, "Director");
            return graph;
          },
          fetchImpl: async (url, options) => {
            calls.push({ url, options });
            return {
              ok: true,
              status: 200,
              async json() {
                return { ok: true, setup: { id: "director-demo" } };
              },
            };
          },
        });

        assert.equal(result.setup.id, "director-demo");
        assert.equal(calls.length, 1);
        assert.equal(calls[0].url, "/koolook/api/setups");
        assert.equal(calls[0].options.method, "POST");
        assert.equal(calls[0].options.headers["Content-Type"], "application/json");
        const payload = JSON.parse(calls[0].options.body);
        assert.deepEqual(payload.visualGraph, graph);
        assert.deepEqual(payload.apiPrompt, apiPrompt);
        assert.deepEqual(payload.source, {
          kind: "sidebar-workflow",
          path: "Demos/Director",
          inventoryPath: ["Demos"],
          name: "Director",
        });
        assert.deepEqual(payload.metadata.tags, ["director", "video"]);
        assert.equal(payload.metadata.previewImage, "preview.png");
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


def test_publish_saved_workflow_rejects_missing_source_before_fetch() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { publishSavedWorkflowSetup } from "./web/sidebar/published_setups.js";

        let fetchCalled = false;
        await assert.rejects(
          () => publishSavedWorkflowSetup({
            dirPath: ["Missing"],
            wfName: "Gone",
            metadata: { id: "gone", title: "Gone", description: "Gone" },
            inputContract: { inputs: [] },
            outputContract: { outputs: [] },
            getWorkflowGraph: () => null,
            fetchImpl: async () => {
              fetchCalled = true;
              return { ok: true, async json() { return {}; } };
            },
          }),
          /Saved workflow not found/
        );
        assert.equal(fetchCalled, false);
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


def test_publish_saved_workflow_can_submit_reviewed_visual_graph() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { publishSavedWorkflowSetup } from "./web/sidebar/published_setups.js";

        const reviewedGraph = {
          nodes: [{ id: 12, type: "Load Image" }],
          links: [],
          groups: [{ title: "Koolook Input", bounding: [0, 0, 100, 100] }],
        };
        const changedGraph = {
          nodes: [{ id: 99, type: "Different" }],
          links: [],
          groups: [],
        };
        const apiPrompt = {
          "12": { class_type: "Load Image", inputs: {} },
        };
        let payload;

        await publishSavedWorkflowSetup({
          dirPath: ["Demos"],
          wfName: "Director",
          visualGraph: reviewedGraph,
          apiPrompt,
          metadata: { id: "director-demo", title: "Director Demo", description: "Reviewed graph" },
          inputContract: { inputs: [] },
          outputContract: { outputs: [] },
          getWorkflowGraph: () => changedGraph,
          fetchImpl: async (_url, options) => {
            payload = JSON.parse(options.body);
            return {
              ok: true,
              status: 200,
              async json() {
                return { ok: true, setup: { id: "director-demo" } };
              },
            };
          },
        });

        assert.deepEqual(payload.visualGraph, reviewedGraph);
        assert.deepEqual(payload.apiPrompt, apiPrompt);
        assert.notDeepEqual(payload.visualGraph, changedGraph);
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


def test_publish_saved_workflow_can_submit_comfy_api_prompt() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { publishSavedWorkflowSetup } from "./web/sidebar/published_setups.js";

        const visualGraph = {
          nodes: [{ id: 12, type: "Text Multiline" }],
          links: [],
        };
        const apiPrompt = {
          "12": { class_type: "Text Multiline", inputs: { text: "comfy export" } },
        };
        let payload;

        await publishSavedWorkflowSetup({
          dirPath: ["Demos"],
          wfName: "Director",
          visualGraph,
          apiPrompt,
          metadata: { id: "director-demo", title: "Director Demo", description: "API prompt" },
          inputContract: { inputs: [] },
          outputContract: { outputs: [{ key: "preview", type: "image" }] },
          fetchImpl: async (_url, options) => {
            payload = JSON.parse(options.body);
            return {
              ok: true,
              status: 200,
              async json() {
                return { ok: true, setup: { id: "director-demo" } };
              },
            };
          },
        });

        assert.deepEqual(payload.visualGraph, visualGraph);
        assert.deepEqual(payload.apiPrompt, apiPrompt);
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


def test_publish_saved_workflow_captures_api_prompt_when_missing() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { publishSavedWorkflowSetup } from "./web/sidebar/published_setups.js";

        const visualGraph = {
          nodes: [{ id: 12, type: "Text Multiline" }],
          links: [],
        };
        const capturedPrompt = {
          "12": { class_type: "Text Multiline", inputs: { text: "captured" } },
        };
        const calls = [];

        await publishSavedWorkflowSetup({
          dirPath: ["Models", "RMGB"],
          wfName: "Publish v04",
          visualGraph,
          metadata: { id: "rmgb-v04", title: "RMGB v04", description: "Captured prompt" },
          inputContract: { inputs: [] },
          outputContract: { outputs: [] },
          captureApiPrompt: async (graph) => {
            calls.push({ kind: "capture", graph });
            return capturedPrompt;
          },
          fetchImpl: async (_url, options) => {
            calls.push({ kind: "fetch", payload: JSON.parse(options.body) });
            return {
              ok: true,
              status: 200,
              async json() {
                return { ok: true, setup: { id: "rmgb-v04" } };
              },
            };
          },
        });

        assert.equal(calls.length, 2);
        assert.equal(calls[0].kind, "capture");
        assert.deepEqual(calls[0].graph, visualGraph);
        assert.equal(calls[1].kind, "fetch");
        assert.deepEqual(calls[1].payload.apiPrompt, capturedPrompt);
        assert.deepEqual(calls[1].payload.visualGraph, visualGraph);
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


def test_publish_saved_workflow_requires_api_prompt_or_capture_before_fetch() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { publishSavedWorkflowSetup } from "./web/sidebar/published_setups.js";

        let fetchCalled = false;
        await assert.rejects(
          () => publishSavedWorkflowSetup({
            dirPath: ["Models"],
            wfName: "Needs Capture",
            visualGraph: { nodes: [], links: [] },
            metadata: { id: "needs-capture", title: "Needs Capture", description: "No prompt" },
            inputContract: { inputs: [] },
            outputContract: { outputs: [] },
            fetchImpl: async () => {
              fetchCalled = true;
              return { ok: true, async json() { return {}; } };
            },
          }),
          /API prompt capture is required/
        );
        assert.equal(fetchCalled, false);
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


def test_publish_saved_workflow_rejects_failed_api_prompt_capture_before_fetch() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { publishSavedWorkflowSetup } from "./web/sidebar/published_setups.js";

        let fetchCalled = false;
        await assert.rejects(
          () => publishSavedWorkflowSetup({
            dirPath: ["Models"],
            wfName: "Broken",
            visualGraph: { nodes: [], links: [] },
            metadata: { id: "broken", title: "Broken", description: "No prompt" },
            inputContract: { inputs: [] },
            outputContract: { outputs: [] },
            captureApiPrompt: async () => null,
            fetchImpl: async () => {
              fetchCalled = true;
              return { ok: true, async json() { return {}; } };
            },
          }),
          /API prompt capture failed/
        );
        assert.equal(fetchCalled, false);
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr
