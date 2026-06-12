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


def browser_stubs() -> str:
    return textwrap.dedent(
        """
        function setupBrowserStubs() {
          const storage = new Map();
          globalThis.localStorage = {
            getItem: (key) => storage.has(key) ? storage.get(key) : null,
            setItem: (key, value) => storage.set(key, String(value)),
            removeItem: (key) => storage.delete(key),
          };
          globalThis.CustomEvent = class CustomEvent { constructor(type) { this.type = type; } };
          globalThis.window = { dispatchEvent() {} };
          Object.defineProperty(globalThis, "navigator", { value: {}, configurable: true });
        }
        function okResponse(body) {
          return { ok: true, status: 200, text: async () => body, json: async () => JSON.parse(body || "{}") };
        }
        """
    )


def test_workflow_has_tag_reflects_added_published_tag() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import {
          loadWorkflowsStore,
          addTag,
          workflowHasTag,
        } from "./web/sidebar/workflows_store.js";

        setupBrowserStubs();
        const serverStore = {
          directories: {
            "Demos": {
              workflows: {
                "Director": { savedAt: "2026-01-01T00:00:00.000Z", graph: { nodes: [] } },
              },
              directories: {},
            },
          },
        };
        globalThis.fetch = async (url) => {
          if (String(url).startsWith("/userdata/koolook_workflows.json")) {
            return okResponse(JSON.stringify(serverStore));
          }
          return { ok: false, status: 404, text: async () => "", json: async () => ({}) };
        };

        await loadWorkflowsStore();

        // Untagged -> false; missing workflow -> false (never throws).
        assert.equal(workflowHasTag(["Demos"], "Director", "published"), false);
        assert.equal(workflowHasTag(["Demos"], "Missing", "published"), false);

        addTag(["Demos"], "Director", "published");
        assert.equal(workflowHasTag(["Demos"], "Director", "published"), true);
        // Case-sensitive, like the rest of the tag system.
        assert.equal(workflowHasTag(["Demos"], "Director", "Published"), false);
        """
    )

    result = run_node_scenario(script + browser_stubs())
    assert result.returncode == 0, result.stderr
