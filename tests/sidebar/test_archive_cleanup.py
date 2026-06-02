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


def test_archive_cleanup_groups_by_base_name_and_keeps_recent_windows() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import {
          loadWorkflowsStore,
          getArchiveCleanupPlan,
          cleanUpArchive,
          getAllWorkflowsForExport,
        } from "./web/sidebar/workflows_store.js";

        setupBrowserStubs();

        const serverStore = {
          directories: {
            Shows: {
              workflows: {
                "ShotA": {
                  savedAt: "2026-06-02T12:00:00.000Z",
                  graph: { nodes: [{ id: "active-a" }] },
                },
                "ShotA (archived 2026-06-02 11:57:00)": {
                  archived: true,
                  archivedAt: "2026-06-02T11:57:00.000Z",
                  savedAt: "2026-06-02T11:57:00.000Z",
                  graph: { nodes: [{ id: "a-3m" }] },
                },
                "ShotA (archived 2026-06-02 11:30:00)": {
                  archived: true,
                  archivedAt: "2026-06-02T11:30:00.000Z",
                  savedAt: "2026-06-02T11:30:00.000Z",
                  graph: { nodes: [{ id: "a-30m" }] },
                },
                "ShotA (archived 2026-06-02 09:00:00)": {
                  archived: true,
                  archivedAt: "2026-06-02T09:00:00.000Z",
                  savedAt: "2026-06-02T09:00:00.000Z",
                  graph: { nodes: [{ id: "a-3h" }] },
                },
                "ShotA (archived 2026-05-31 12:00:00)": {
                  archived: true,
                  archivedAt: "2026-05-31T12:00:00.000Z",
                  savedAt: "2026-05-31T12:00:00.000Z",
                  graph: { nodes: [{ id: "a-old" }] },
                },
                "ShotB (archived 2026-05-30 12:00:00)": {
                  archived: true,
                  archivedAt: "2026-05-30T12:00:00.000Z",
                  savedAt: "2026-05-30T12:00:00.000Z",
                  graph: { nodes: [{ id: "b-only-old" }] },
                },
              },
              directories: {},
            },
          },
        };

        globalThis.fetch = async (url, init = {}) => {
          if (init.method === "POST") return okResponse("");
          if (String(url).startsWith("/userdata/koolook_workflows.json")) {
            return okResponse(JSON.stringify(serverStore));
          }
          return { ok: false, status: 404, text: async () => "", json: async () => ({}) };
        };

        await loadWorkflowsStore();
        const now = new Date("2026-06-02T12:00:00.000Z");
        const plan = getArchiveCleanupPlan(["Shows"], now);

        assert.equal(plan.keepCount, 4);
        assert.equal(plan.deleteCount, 1);
        assert.deepEqual(plan.deleteNames.sort(), [
            "ShotA (archived 2026-05-31 12:00:00)",
        ]);
        assert.ok(plan.keepNames.includes("ShotB (archived 2026-05-30 12:00:00)"));

        const result = cleanUpArchive(["Shows"], now);
        assert.equal(result.keepCount, 4);
        assert.equal(result.deleteCount, 1);

        const workflows = getAllWorkflowsForExport().directories.Shows.workflows;
        assert.ok(workflows.ShotA, "active workflow must survive cleanup");
        assert.ok(workflows["ShotA (archived 2026-06-02 11:57:00)"]);
        assert.ok(workflows["ShotA (archived 2026-06-02 11:30:00)"]);
        assert.ok(workflows["ShotA (archived 2026-06-02 09:00:00)"]);
        assert.ok(workflows["ShotB (archived 2026-05-30 12:00:00)"]);
        assert.equal(workflows["ShotA (archived 2026-05-31 12:00:00)"], undefined);

        function okResponse(body) {
          return { ok: true, status: 200, text: async () => body, json: async () => JSON.parse(body || "{}") };
        }
        """
    )

    result = run_node_scenario(script + browser_stubs())
    assert result.returncode == 0, result.stderr


def test_archive_cleanup_triages_multiple_setup_names_independently() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import {
          loadWorkflowsStore,
          getArchiveCleanupPlan,
          cleanUpArchive,
          getAllWorkflowsForExport,
        } from "./web/sidebar/workflows_store.js";

        setupBrowserStubs();

        const serverStore = {
          directories: {
            Shows: {
              workflows: {
                "Script cleanup version 01": {
                  savedAt: "2026-06-02T12:00:00.000Z",
                  graph: { nodes: [{ id: "active-v01" }] },
                },
                "Script cleanup version 01 (archived 2026-06-02 11:57:00)": {
                  archived: true,
                  archivedAt: "2026-06-02T11:57:00.000Z",
                  savedAt: "2026-06-02T11:57:00.000Z",
                  graph: { nodes: [{ id: "v01-5min" }] },
                },
                "Script cleanup version 01 (archived 2026-06-02 11:30:00)": {
                  archived: true,
                  archivedAt: "2026-06-02T11:30:00.000Z",
                  savedAt: "2026-06-02T11:30:00.000Z",
                  graph: { nodes: [{ id: "v01-hour" }] },
                },
                "Script cleanup version 01 (archived 2026-05-31 12:00:00)": {
                  archived: true,
                  archivedAt: "2026-05-31T12:00:00.000Z",
                  savedAt: "2026-05-31T12:00:00.000Z",
                  graph: { nodes: [{ id: "v01-old-delete" }] },
                },
                "Script cleanup version 02": {
                  savedAt: "2026-06-02T12:00:00.000Z",
                  graph: { nodes: [{ id: "active-v02" }] },
                },
                "Script cleanup version 02 (archived 2026-06-02 11:58:00)": {
                  archived: true,
                  archivedAt: "2026-06-02T11:58:00.000Z",
                  savedAt: "2026-06-02T11:58:00.000Z",
                  graph: { nodes: [{ id: "v02-5min" }] },
                },
                "Script cleanup version 02 (archived 2026-06-02 10:30:00)": {
                  archived: true,
                  archivedAt: "2026-06-02T10:30:00.000Z",
                  savedAt: "2026-06-02T10:30:00.000Z",
                  graph: { nodes: [{ id: "v02-day" }] },
                },
                "Script cleanup version 02 (archived 2026-05-30 12:00:00)": {
                  archived: true,
                  archivedAt: "2026-05-30T12:00:00.000Z",
                  savedAt: "2026-05-30T12:00:00.000Z",
                  graph: { nodes: [{ id: "v02-old-delete" }] },
                },
              },
              directories: {},
            },
          },
        };

        globalThis.fetch = async (url, init = {}) => {
          if (init.method === "POST") return okResponse("");
          if (String(url).startsWith("/userdata/koolook_workflows.json")) {
            return okResponse(JSON.stringify(serverStore));
          }
          return { ok: false, status: 404, text: async () => "", json: async () => ({}) };
        };

        await loadWorkflowsStore();
        const plan = getArchiveCleanupPlan(["Shows"], new Date("2026-06-02T12:00:00.000Z"));

        assert.deepEqual(plan.deleteNames.sort(), [
          "Script cleanup version 01 (archived 2026-05-31 12:00:00)",
          "Script cleanup version 02 (archived 2026-05-30 12:00:00)",
        ]);
        assert.ok(plan.keepNames.includes("Script cleanup version 01 (archived 2026-06-02 11:57:00)"));
        assert.ok(plan.keepNames.includes("Script cleanup version 01 (archived 2026-06-02 11:30:00)"));
        assert.ok(plan.keepNames.includes("Script cleanup version 02 (archived 2026-06-02 11:58:00)"));
        assert.ok(plan.keepNames.includes("Script cleanup version 02 (archived 2026-06-02 10:30:00)"));

        cleanUpArchive(["Shows"], plan);
        const workflows = getAllWorkflowsForExport().directories.Shows.workflows;
        assert.ok(workflows["Script cleanup version 01"], "active v01 must survive cleanup");
        assert.ok(workflows["Script cleanup version 02"], "active v02 must survive cleanup");
        assert.ok(workflows["Script cleanup version 01 (archived 2026-06-02 11:57:00)"]);
        assert.ok(workflows["Script cleanup version 01 (archived 2026-06-02 11:30:00)"]);
        assert.ok(workflows["Script cleanup version 02 (archived 2026-06-02 11:58:00)"]);
        assert.ok(workflows["Script cleanup version 02 (archived 2026-06-02 10:30:00)"]);
        assert.equal(workflows["Script cleanup version 01 (archived 2026-05-31 12:00:00)"], undefined);
        assert.equal(workflows["Script cleanup version 02 (archived 2026-05-30 12:00:00)"], undefined);

        function okResponse(body) {
          return { ok: true, status: 200, text: async () => body, json: async () => JSON.parse(body || "{}") };
        }
        """
    )

    result = run_node_scenario(script + browser_stubs())
    assert result.returncode == 0, result.stderr


def test_archive_metadata_uses_name_timestamp_fallback_and_archived_at_on_new_archives() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import {
          loadWorkflowsStore,
          persistMutation,
          saveWorkflowEntry,
          getArchiveDisplayInfo,
          getAllWorkflowsForExport,
        } from "./web/sidebar/workflows_store.js";

        setupBrowserStubs();

        let serverStore = {
          directories: {
            Shows: {
              workflows: {
                "Legacy Name (archived 2026-06-02 10:15:00)": {
                  archived: true,
                  savedAt: "2026-01-01T00:00:00.000Z",
                  graph: { nodes: [{ id: "legacy" }] },
                },
                "Current": {
                  savedAt: "2026-06-02T11:00:00.000Z",
                  graph: { nodes: [{ id: "old-current" }] },
                },
              },
              directories: {},
            },
          },
        };

        globalThis.fetch = async (url, init = {}) => {
          if (init.method === "POST") {
            serverStore = JSON.parse(init.body);
            return okResponse("");
          }
          if (String(url).startsWith("/userdata/koolook_workflows.json")) {
            return okResponse(JSON.stringify(serverStore));
          }
          return { ok: false, status: 404, text: async () => "", json: async () => ({}) };
        };

        await loadWorkflowsStore();
        const info = getArchiveDisplayInfo(["Shows"], "Legacy Name (archived 2026-06-02 10:15:00)", new Date("2026-06-02T12:00:00.000Z"));
        assert.equal(info.label, "Legacy Name");
        assert.equal(info.timestampMs, Date.parse("2026-06-02T10:15:00.000Z"));
        assert.match(info.meta, /^archived today \\d{2}:15$/);

        await persistMutation({
          mutate: () => saveWorkflowEntry(
            ["Shows"],
            "Current",
            { nodes: [{ id: "new-current" }] },
            { now: new Date("2026-06-02T12:34:56.000Z") },
          ),
        });

        const workflows = getAllWorkflowsForExport().directories.Shows.workflows;
        const archivedName = "Current (archived 2026-06-02 12:34:56)";
        assert.equal(workflows[archivedName].archived, true);
        assert.equal(workflows[archivedName].archivedAt, "2026-06-02T12:34:56.000Z");

        function okResponse(body) {
          return { ok: true, status: 200, text: async () => body, json: async () => JSON.parse(body || "{}") };
        }
        """
    )

    result = run_node_scenario(script + browser_stubs())
    assert result.returncode == 0, result.stderr


def test_archive_cleanup_uses_frozen_plan_for_confirmed_delete_set() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import {
          loadWorkflowsStore,
          getArchiveCleanupPlan,
          cleanUpArchive,
          getAllWorkflowsForExport,
        } from "./web/sidebar/workflows_store.js";

        setupBrowserStubs();

        const serverStore = {
          directories: {
            Shows: {
              workflows: {
                "ShotA (archived 2026-06-02 11:56:00)": {
                  archived: true,
                  archivedAt: "2026-06-02T11:56:00.000Z",
                  savedAt: "2026-06-02T11:56:00.000Z",
                  graph: { nodes: [{ id: "within-preview-window" }] },
                },
              },
              directories: {},
            },
          },
        };

        globalThis.fetch = async (url, init = {}) => {
          if (init.method === "POST") return okResponse("");
          if (String(url).startsWith("/userdata/koolook_workflows.json")) {
            return okResponse(JSON.stringify(serverStore));
          }
          return { ok: false, status: 404, text: async () => "", json: async () => ({}) };
        };

        await loadWorkflowsStore();
        const previewPlan = getArchiveCleanupPlan(["Shows"], new Date("2026-06-02T12:00:00.000Z"));
        assert.equal(previewPlan.deleteCount, 0);

        const result = cleanUpArchive(["Shows"], previewPlan);
        assert.equal(result, false);
        assert.ok(
          getAllWorkflowsForExport().directories.Shows.workflows["ShotA (archived 2026-06-02 11:56:00)"],
          "entry must not be deleted just because it crossed a window after preview",
        );

        function okResponse(body) {
          return { ok: true, status: 200, text: async () => body, json: async () => JSON.parse(body || "{}") };
        }
        """
    )

    result = run_node_scenario(script + browser_stubs())
    assert result.returncode == 0, result.stderr


def test_archive_cleanup_keeps_future_dated_newest_archive() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import {
          loadWorkflowsStore,
          getArchiveCleanupPlan,
          cleanUpArchive,
          getAllWorkflowsForExport,
        } from "./web/sidebar/workflows_store.js";

        setupBrowserStubs();

        const serverStore = {
          directories: {
            Shows: {
              workflows: {
                "ShotA (archived 2026-06-02 12:02:00)": {
                  archived: true,
                  archivedAt: "2026-06-02T12:02:00.000Z",
                  savedAt: "2026-06-02T12:02:00.000Z",
                  graph: { nodes: [{ id: "future-newest" }] },
                },
                "ShotA (archived 2026-06-02 11:57:00)": {
                  archived: true,
                  archivedAt: "2026-06-02T11:57:00.000Z",
                  savedAt: "2026-06-02T11:57:00.000Z",
                  graph: { nodes: [{ id: "recent" }] },
                },
                "ShotA (archived 2026-06-02 10:00:00)": {
                  archived: true,
                  archivedAt: "2026-06-02T10:00:00.000Z",
                  savedAt: "2026-06-02T10:00:00.000Z",
                  graph: { nodes: [{ id: "older" }] },
                },
              },
              directories: {},
            },
          },
        };

        globalThis.fetch = async (url, init = {}) => {
          if (init.method === "POST") return okResponse("");
          if (String(url).startsWith("/userdata/koolook_workflows.json")) {
            return okResponse(JSON.stringify(serverStore));
          }
          return { ok: false, status: 404, text: async () => "", json: async () => ({}) };
        };

        await loadWorkflowsStore();
        const plan = getArchiveCleanupPlan(["Shows"], new Date("2026-06-02T12:00:00.000Z"));
        assert.ok(plan.keepNames.includes("ShotA (archived 2026-06-02 12:02:00)"));
        assert.ok(!plan.deleteNames.includes("ShotA (archived 2026-06-02 12:02:00)"));

        cleanUpArchive(["Shows"], plan);
        const workflows = getAllWorkflowsForExport().directories.Shows.workflows;
        assert.ok(workflows["ShotA (archived 2026-06-02 12:02:00)"]);

        function okResponse(body) {
          return { ok: true, status: 200, text: async () => body, json: async () => JSON.parse(body || "{}") };
        }
        """
    )

    result = run_node_scenario(script + browser_stubs())
    assert result.returncode == 0, result.stderr


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
          Object.defineProperty(globalThis, "navigator", {
            value: {},
            configurable: true,
          });
          const makeEl = () => ({
            className: "",
            textContent: "",
            title: "",
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
