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


def test_simulator_lists_setup_and_polls_public_run_routes() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import {
          listPublishedSetups,
          getPublishedSetup,
          runAndPollPublishedSetup,
          formatRun,
        } from "./web/setup_runner_simulator.js";

        const calls = [];
        const responses = [
          { ok: true, status: 200, body: { ok: true, setups: [{ id: "director-demo", title: "Director Demo" }] } },
          { ok: true, status: 200, body: { id: "director-demo", metadata: { title: "Director Demo" } } },
          { ok: true, status: 200, body: { ok: true, run: { runId: "run-000001", promptId: "prompt-1", status: "queued" } } },
          { ok: true, status: 200, body: { ok: true, run: { runId: "run-000001", setupId: "director-demo", promptId: "prompt-1", status: "running", outputs: [] } } },
          { ok: true, status: 200, body: { ok: true, run: { runId: "run-000001", setupId: "director-demo", promptId: "prompt-1", status: "succeeded", comfyStatus: { completed: true }, outputs: [{ key: "video" }] } } },
        ];
        const fetchImpl = async (url, options = {}) => {
          calls.push({ url, options });
          const response = responses.shift();
          return {
            ok: response.ok,
            status: response.status,
            async json() { return response.body; },
          };
        };

        const setups = await listPublishedSetups({ baseUrl: "http://127.0.0.1:8188/", fetchImpl });
        assert.deepEqual(setups, [{ id: "director-demo", title: "Director Demo" }]);
        const setup = await getPublishedSetup({ setupId: "director-demo", baseUrl: "http://127.0.0.1:8188/", fetchImpl });
        assert.equal(setup.id, "director-demo");
        const updates = [];
        const finalRun = await runAndPollPublishedSetup({
          setupId: "director-demo",
          inputs: { prompt: "A close-up" },
          baseUrl: "http://127.0.0.1:8188/",
          intervalMs: 1,
          timeoutMs: 1000,
          sleepImpl: async () => {},
          onUpdate: run => updates.push(run.status),
          fetchImpl,
        });

        assert.equal(finalRun.status, "succeeded");
        assert.deepEqual(updates, ["queued", "running", "succeeded"]);
        assert.equal(calls[0].url, "http://127.0.0.1:8188/koolook/api/setups");
        assert.equal(calls[1].url, "http://127.0.0.1:8188/koolook/api/setups/director-demo");
        assert.equal(calls[2].url, "http://127.0.0.1:8188/koolook/api/setups/director-demo/run");
        assert.deepEqual(JSON.parse(calls[2].options.body), { inputs: { prompt: "A close-up" } });
        assert.equal(calls[3].url, "http://127.0.0.1:8188/koolook/api/runs/run-000001");
        assert.equal(calls[4].url, "http://127.0.0.1:8188/koolook/api/runs/run-000001");
        assert.match(formatRun(finalRun), /Details:/);
        assert.match(formatRun(finalRun), /comfyStatus/);
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


def test_simulator_accepts_bare_array_catalog_response() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { listPublishedSetups } from "./web/setup_runner_simulator.js";

        const setups = await listPublishedSetups({
          fetchImpl: async () => ({
            ok: true,
            status: 200,
            async json() {
              return [{ id: "mask_rmgb", metadata: { title: "Mask_RMGB" } }];
            },
          }),
        });

        assert.deepEqual(setups, [{ id: "mask_rmgb", metadata: { title: "Mask_RMGB" } }]);
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


def test_simulator_preserves_error_payload_for_debugging() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { runPublishedSetup, formatError } from "./web/setup_runner_simulator.js";

        await assert.rejects(
          () => runPublishedSetup({
            setupId: "director-demo",
            fetchImpl: async () => ({
              ok: false,
              status: 400,
              async json() {
                return { ok: false, errors: ["required input 'prompt' is missing"] };
              },
            }),
          }),
          error => {
            assert.equal(error.message, "required input 'prompt' is missing");
            assert.equal(error.status, 400);
            assert.deepEqual(error.payload, {
              ok: false,
              errors: ["required input 'prompt' is missing"],
            });
            assert.match(formatError(error), /required input/);
            return true;
          }
        );
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr
