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


def test_workflow_save_survives_reload_when_server_write_succeeds() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import {
          loadWorkflowsStore,
          persistMutation,
          saveWorkflowEntry,
          getAllWorkflowsForExport,
        } from "./web/sidebar/workflows_store.js";

        setupBrowserStubs();

        let serverStore = {
          directories: {
            "_WIP": {
              workflows: {},
              directories: {
                "TEST_setups": {
                  workflows: {
                    "LTX-2.3_Director_4K_v01": {
                      savedAt: "2026-05-17T10:00:00.000Z",
                      graph: { nodes: [{ id: 1, mode: "4K" }] },
                    },
                  },
                  directories: {},
                },
              },
            },
          },
        };

        globalThis.fetch = async (url, init = {}) => {
          const textUrl = String(url);
          if (init.method === "POST") {
            serverStore = JSON.parse(init.body);
            return okResponse("");
          }
          if (textUrl.startsWith("/userdata/koolook_workflows.json")) {
            return okResponse(JSON.stringify(serverStore));
          }
          return { ok: false, status: 404, text: async () => "", json: async () => ({}) };
        };

        await loadWorkflowsStore();
        const saved = await persistMutation({
          mutate: () => saveWorkflowEntry(
            ["_WIP", "TEST_setups"],
            "LTX-2.3_Director_2K_v01",
            { nodes: [{ id: 1, mode: "2K" }] },
          ),
        });
        assert.equal(saved, true);

        await loadWorkflowsStore();
        const exported = getAllWorkflowsForExport();
        const workflows = exported.directories._WIP.directories.TEST_setups.workflows;
        assert.ok(workflows["LTX-2.3_Director_2K_v01"]);
        assert.ok(workflows["LTX-2.3_Director_4K_v01"]);
        assert.equal(workflows["LTX-2.3_Director_2K_v01"].graph.nodes[0].mode, "2K");

        function okResponse(body) {
          return { ok: true, status: 200, text: async () => body, json: async () => JSON.parse(body || "{}") };
        }

        function setupBrowserStubs() {
          const storage = new Map();
          globalThis.localStorage = {
            getItem: (key) => storage.has(key) ? storage.get(key) : null,
            setItem: (key, value) => storage.set(key, String(value)),
            removeItem: (key) => storage.delete(key),
          };
          globalThis.CustomEvent = class CustomEvent { constructor(type) { this.type = type; } };
          globalThis.window = { dispatchEvent() {} };
          Object.defineProperty(globalThis, "navigator", {
            value: {},
            configurable: true,
          });
          const makeEl = () => ({
            className: "",
            textContent: "",
            style: {},
            classList: { add() {}, remove() {} },
            appendChild() {},
            remove() {},
            addEventListener() {},
            setAttribute() {},
          });
          globalThis.document = {
            createElement: makeEl,
            body: { appendChild() {} },
            head: { appendChild() {} },
            querySelectorAll: () => [],
            getElementById: () => null,
          };
        }
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


def test_workflow_startup_read_bypasses_browser_cache() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { loadWorkflowsStore } from "./web/sidebar/workflows_store.js";

        setupBrowserStubs();

        const serverStore = { directories: {} };
        const calls = [];
        globalThis.fetch = async (url, init = {}) => {
          calls.push({ url: String(url), cache: init.cache || null });
          if (String(url).startsWith("/userdata/koolook_workflows.json")) {
            return okResponse(JSON.stringify(serverStore));
          }
          return { ok: false, status: 404, text: async () => "", json: async () => ({}) };
        };

        await loadWorkflowsStore();

        assert.equal(calls.length, 1);
        assert.match(calls[0].url, /^\\/userdata\\/koolook_workflows\\.json\\?_/);
        assert.equal(calls[0].cache, "no-store");

        function okResponse(body) {
          return { ok: true, status: 200, text: async () => body, json: async () => JSON.parse(body || "{}") };
        }

        function setupBrowserStubs() {
          const storage = new Map();
          globalThis.localStorage = {
            getItem: (key) => storage.has(key) ? storage.get(key) : null,
            setItem: (key, value) => storage.set(key, String(value)),
            removeItem: (key) => storage.delete(key),
          };
          globalThis.CustomEvent = class CustomEvent { constructor(type) { this.type = type; } };
          globalThis.window = { dispatchEvent() {} };
          Object.defineProperty(globalThis, "navigator", {
            value: {},
            configurable: true,
          });
          const makeEl = () => ({
            className: "",
            textContent: "",
            style: {},
            classList: { add() {}, remove() {} },
            appendChild() {},
            remove() {},
            addEventListener() {},
            setAttribute() {},
          });
          globalThis.document = {
            createElement: makeEl,
            body: { appendChild() {} },
            head: { appendChild() {} },
            querySelectorAll: () => [],
            getElementById: () => null,
          };
        }
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


def test_fallback_only_workflow_save_becomes_live_again_after_reload() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import {
          loadWorkflowsStore,
          persistMutation,
          saveWorkflowEntry,
          getAllWorkflowsForExport,
        } from "./web/sidebar/workflows_store.js";

        setupBrowserStubs();

        const staleServerStore = {
          directories: {
            "_WIP": {
              workflows: {},
              directories: {
                "TEST_setups": {
                  workflows: {
                    "LTX-2.3_Director_4K_v01": {
                      savedAt: "2026-05-17T10:00:00.000Z",
                      graph: { nodes: [{ id: 1, mode: "4K" }] },
                    },
                  },
                  directories: {},
                },
              },
            },
          },
        };

        let rejectPosts = true;
        globalThis.fetch = async (url, init = {}) => {
          const textUrl = String(url);
          if (init.method === "POST") {
            if (rejectPosts) return { ok: false, status: 500, text: async () => "" };
            return okResponse("");
          }
          if (textUrl.startsWith("/userdata/koolook_workflows.json")) {
            return okResponse(JSON.stringify(staleServerStore));
          }
          return { ok: false, status: 404, text: async () => "", json: async () => ({}) };
        };

        await loadWorkflowsStore();
        const saved = await persistMutation({
          mutate: () => saveWorkflowEntry(
            ["_WIP", "TEST_setups"],
            "LTX-2.3_Director_2K_v01",
            { nodes: [{ id: 1, mode: "2K" }] },
          ),
        });
        assert.equal(saved, true);
        assert.ok(getAllWorkflowsForExport().directories._WIP.directories.TEST_setups.workflows[
          "LTX-2.3_Director_2K_v01"
        ]);

        rejectPosts = false;
        await loadWorkflowsStore();
        const afterReload = getAllWorkflowsForExport();
        const workflows = afterReload.directories._WIP.directories.TEST_setups.workflows;
        assert.ok(
          workflows["LTX-2.3_Director_2K_v01"],
          "new fallback-only workflow should be recovered as live state on reload",
        );
        assert.equal(workflows["LTX-2.3_Director_2K_v01"].graph.nodes[0].mode, "2K");

        function okResponse(body) {
          return { ok: true, status: 200, text: async () => body, json: async () => JSON.parse(body || "{}") };
        }

        function setupBrowserStubs() {
          const storage = new Map();
          globalThis.localStorage = {
            getItem: (key) => storage.has(key) ? storage.get(key) : null,
            setItem: (key, value) => storage.set(key, String(value)),
            removeItem: (key) => storage.delete(key),
          };
          globalThis.CustomEvent = class CustomEvent { constructor(type) { this.type = type; } };
          globalThis.window = { dispatchEvent() {} };
          Object.defineProperty(globalThis, "navigator", {
            value: {},
            configurable: true,
          });
          const makeEl = () => ({
            className: "",
            textContent: "",
            style: {},
            classList: { add() {}, remove() {} },
            appendChild() {},
            remove() {},
            addEventListener() {},
            setAttribute() {},
          });
          globalThis.document = {
            createElement: makeEl,
            body: { appendChild() {} },
            head: { appendChild() {} },
            querySelectorAll: () => [],
            getElementById: () => null,
          };
        }
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


def test_fallback_recovery_preserves_server_only_workflows() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import {
          loadWorkflowsStore,
          getAllWorkflowsForExport,
        } from "./web/sidebar/workflows_store.js";

        setupBrowserStubs();

        let serverStore = {
          directories: {
            "_WIP": {
              workflows: {},
              directories: {
                "TEST_setups": {
                  workflows: {
                    "LTX-2.3_Director_4K_v01": {
                      savedAt: "2026-05-17T10:00:00.000Z",
                      graph: { nodes: [{ id: 1, mode: "server-4K" }] },
                    },
                    "LTX-2.3_Director_8K_v01": {
                      savedAt: "2026-05-17T10:05:00.000Z",
                      graph: { nodes: [{ id: 2, mode: "server-only-8K" }] },
                    },
                  },
                  directories: {},
                },
              },
            },
          },
        };
        const fallbackStore = {
          directories: {
            "_WIP": {
              workflows: {},
              directories: {
                "TEST_setups": {
                  workflows: {
                    "LTX-2.3_Director_4K_v01": {
                      savedAt: "2026-05-17T10:10:00.000Z",
                      graph: { nodes: [{ id: 1, mode: "fallback-newer-4K" }] },
                    },
                  },
                  directories: {},
                },
              },
            },
          },
        };

        localStorage.setItem("koolook.workflows.fallback.v1", JSON.stringify(fallbackStore));

        let posts = 0;
        globalThis.fetch = async (url, init = {}) => {
          const textUrl = String(url);
          if (init.method === "POST") {
            posts += 1;
            serverStore = JSON.parse(init.body);
            return okResponse("");
          }
          if (textUrl.startsWith("/userdata/koolook_workflows.json")) {
            return okResponse(JSON.stringify(serverStore));
          }
          return { ok: false, status: 404, text: async () => "", json: async () => ({}) };
        };

        await loadWorkflowsStore();

        assert.equal(posts, 1);
        const exported = getAllWorkflowsForExport();
        const workflows = exported.directories._WIP.directories.TEST_setups.workflows;
        assert.equal(workflows["LTX-2.3_Director_4K_v01"].graph.nodes[0].mode, "fallback-newer-4K");
        assert.equal(workflows["LTX-2.3_Director_8K_v01"].graph.nodes[0].mode, "server-only-8K");
        assert.equal(
          serverStore.directories._WIP.directories.TEST_setups.workflows[
            "LTX-2.3_Director_8K_v01"
          ].graph.nodes[0].mode,
          "server-only-8K",
        );
        assert.equal(localStorage.getItem("koolook.workflows.fallback.v1"), null);

        function okResponse(body) {
          return { ok: true, status: 200, text: async () => body, json: async () => JSON.parse(body || "{}") };
        }

        function setupBrowserStubs() {
          const storage = new Map();
          globalThis.localStorage = {
            getItem: (key) => storage.has(key) ? storage.get(key) : null,
            setItem: (key, value) => storage.set(key, String(value)),
            removeItem: (key) => storage.delete(key),
          };
          globalThis.CustomEvent = class CustomEvent { constructor(type) { this.type = type; } };
          globalThis.window = { dispatchEvent() {} };
          Object.defineProperty(globalThis, "navigator", {
            value: {},
            configurable: true,
          });
          const makeEl = () => ({
            className: "",
            textContent: "",
            style: {},
            classList: { add() {}, remove() {} },
            appendChild() {},
            remove() {},
            addEventListener() {},
            setAttribute() {},
          });
          globalThis.document = {
            createElement: makeEl,
            body: { appendChild() {} },
            head: { appendChild() {} },
            querySelectorAll: () => [],
            getElementById: () => null,
          };
        }
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


def test_redundant_fallback_is_cleared_after_server_catches_up() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import {
          loadWorkflowsStore,
          getAllWorkflowsForExport,
        } from "./web/sidebar/workflows_store.js";

        setupBrowserStubs();

        const serverStore = {
          directories: {
            "_WIP": {
              directories: {
                "TEST_setups": {
                  directories: {},
                  workflows: {
                    "LTX-2.3_Director_4K_v01": {
                      graph: { nodes: [{ id: 1, mode: "4K" }] },
                      savedAt: "2026-05-17T10:00:00.000Z",
                    },
                  },
                },
              },
              workflows: {},
            },
          },
        };
        const fallbackStoreWithDifferentKeyOrder = {
          directories: {
            "_WIP": {
              workflows: {},
              directories: {
                "TEST_setups": {
                  workflows: {
                    "LTX-2.3_Director_4K_v01": {
                      savedAt: "2026-05-17T10:00:00.000Z",
                      graph: { nodes: [{ mode: "4K", id: 1 }] },
                    },
                  },
                  directories: {},
                },
              },
            },
          },
        };

        localStorage.setItem(
          "koolook.workflows.fallback.v1",
          JSON.stringify(fallbackStoreWithDifferentKeyOrder),
        );

        let posts = 0;
        globalThis.fetch = async (url, init = {}) => {
          const textUrl = String(url);
          if (init.method === "POST") {
            posts += 1;
            return okResponse("");
          }
          if (textUrl.startsWith("/userdata/koolook_workflows.json")) {
            return okResponse(JSON.stringify(serverStore));
          }
          return { ok: false, status: 404, text: async () => "", json: async () => ({}) };
        };

        const loadResult = await loadWorkflowsStore();

        assert.equal(loadResult.fallbackBlob, undefined);
        assert.equal(posts, 0);
        assert.equal(localStorage.getItem("koolook.workflows.fallback.v1"), null);
        const workflows = getAllWorkflowsForExport().directories._WIP.directories.TEST_setups.workflows;
        assert.equal(workflows["LTX-2.3_Director_4K_v01"].graph.nodes[0].mode, "4K");

        function okResponse(body) {
          return { ok: true, status: 200, text: async () => body, json: async () => JSON.parse(body || "{}") };
        }

        function setupBrowserStubs() {
          const storage = new Map();
          globalThis.localStorage = {
            getItem: (key) => storage.has(key) ? storage.get(key) : null,
            setItem: (key, value) => storage.set(key, String(value)),
            removeItem: (key) => storage.delete(key),
          };
          globalThis.CustomEvent = class CustomEvent { constructor(type) { this.type = type; } };
          globalThis.window = { dispatchEvent() {} };
          Object.defineProperty(globalThis, "navigator", {
            value: {},
            configurable: true,
          });
          const makeEl = () => ({
            className: "",
            textContent: "",
            style: {},
            classList: { add() {}, remove() {} },
            appendChild() {},
            remove() {},
            addEventListener() {},
            setAttribute() {},
          });
          globalThis.document = {
            createElement: makeEl,
            body: { appendChild() {} },
            head: { appendChild() {} },
            querySelectorAll: () => [],
            getElementById: () => null,
          };
        }
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr
