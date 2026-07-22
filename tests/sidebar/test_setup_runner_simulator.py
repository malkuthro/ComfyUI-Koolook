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


def test_simulator_supports_independent_output_switch() -> None:
    """An app surface with an outputSwitch submits an output_switch value. The
    'Same as input' sentinel (-1) follows the input switch; a concrete choice
    overrides it — so an EXR source can be routed to a QT writer."""
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import {
          defaultRunInputsFromSetup,
          runInputsFromAppValues,
        } from "./web/setup_runner_simulator.js";

        const setup = {
          setupSurface: {
            app: {
              switch: {
                key: "switch",
                default: 0,
                options: [
                  { value: 0, label: "EXR", input: "sequence_folder", visible: true },
                  { value: 1, label: "QT", input: "qt_file", visible: true },
                  { value: 2, label: "Img", input: "single_file", visible: true },
                ],
              },
              outputSwitch: {
                key: "output_switch",
                default: -1,
                sameAsInput: true,
                options: [
                  { value: 0, label: "EXR", visible: true },
                  { value: 1, label: "QT", visible: true },
                  { value: 2, label: "Img", visible: true },
                ],
              },
              inputs: [{ key: "sequence_folder", default: "/plates/demo" }],
              outputs: [{ key: "folder", default: "/renders/demo" }],
            },
          },
        };

        // Default: output follows input (EXR=0) via the "Same as input" sentinel.
        assert.deepEqual(defaultRunInputsFromSetup(setup), {
          switch: 0,
          sequence_folder: "/plates/demo",
          folder: "/renders/demo",
          output_switch: 0,
        });

        // Explicit override: EXR source (switch 0), QT writer (output_switch 1).
        assert.deepEqual(runInputsFromAppValues(setup, {
          switch: 0,
          output_switch: 1,
          sequence_folder: "/plates/demo",
          folder: "/renders/demo",
        }), {
          switch: 0,
          sequence_folder: "/plates/demo",
          folder: "/renders/demo",
          output_switch: 1,
        });

        // "Same as input" (-1) submitted -> resolves to the chosen input (QT=1).
        assert.deepEqual(runInputsFromAppValues(setup, {
          switch: 1,
          output_switch: -1,
          sequence_folder: "/plates/demo",
          folder: "/renders/demo",
        }), {
          switch: 1,
          sequence_folder: "/plates/demo",
          folder: "/renders/demo",
          output_switch: 1,
        });
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


def test_simulator_times_out_when_active_status_never_progresses() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { runAndPollPublishedSetup } from "./web/setup_runner_simulator.js";

        let now = 0;
        const originalNow = Date.now;
        Date.now = () => now;
        try {
          const fetchImpl = async (url, options = {}) => {
            const isRunRequest = options.method === "POST";
            return {
              ok: true,
              status: 200,
              async json() {
                return isRunRequest
                  ? { ok: true, run: { runId: "run-000001", promptId: "prompt-1", status: "queued" } }
                  : { ok: true, run: { runId: "run-000001", promptId: "prompt-1", status: "running", outputs: [] } };
              },
            };
          };

          await assert.rejects(
            () => runAndPollPublishedSetup({
              setupId: "director-demo",
              fetchImpl,
              intervalMs: 1,
              timeoutMs: 5,
              sleepImpl: async ms => { now += ms; },
            }),
            /Timed out waiting for run/
          );
        } finally {
          Date.now = originalNow;
        }
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


def test_simulator_accepts_canonical_setup_file_shapes() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { setupsFromPublishedSetupFile } from "./web/setup_runner_simulator.js";

        const setup = {
          schemaVersion: 1,
          id: "rmgb-publish-v04",
          metadata: { title: "RMGB Publish v04" },
          visualGraph: { nodes: [] },
          apiPrompt: { "12": { class_type: "Text Multiline", inputs: { text: "demo" } } },
          inputContract: { inputs: [] },
          outputContract: { outputs: [] },
          setupSurface: { app: { inputs: [], outputs: [], results: [] } },
          source: { kind: "sidebar-workflow", path: "Models/RMGB/rmgb-publish-v04", inventoryPath: ["Models", "RMGB"], name: "rmgb-publish-v04" },
          validation: { status: "valid", diagnostics: [] },
        };

        assert.deepEqual(setupsFromPublishedSetupFile(setup), [setup]);
        assert.deepEqual(setupsFromPublishedSetupFile({ setups: [setup] }), [setup]);
        assert.deepEqual(setupsFromPublishedSetupFile([setup]), [setup]);
        assert.throws(
          () => setupsFromPublishedSetupFile({ nope: [] }),
          /published setup record/
        );
        assert.throws(
          () => setupsFromPublishedSetupFile({ ...setup, apiPrompt: null }),
          /apiPrompt/
        );
        assert.throws(
          () => setupsFromPublishedSetupFile({ ...setup, apiPrompt: [] }),
          /apiPrompt/
        );
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


def test_simulator_marks_local_setup_files_as_not_runnable() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import {
          localSetupFileRecords,
          setupCanRunFromRegistry,
          localSetupRunMessage,
        } from "./web/setup_runner_simulator.js";

        const setup = {
          schemaVersion: 1,
          id: "file-only",
          metadata: { title: "File Only" },
          visualGraph: { nodes: [] },
          apiPrompt: { "1": { class_type: "Example", inputs: {} } },
          inputContract: { inputs: [] },
          outputContract: { outputs: [] },
          setupSurface: { app: { inputs: [], outputs: [], results: [] } },
          source: { kind: "sidebar-workflow", path: "File/file-only", inventoryPath: ["File"], name: "file-only" },
          validation: { status: "valid", diagnostics: [] },
        };

        assert.equal(setupCanRunFromRegistry(setup), true);
        const [local] = localSetupFileRecords({ setups: [setup] });
        assert.equal(setupCanRunFromRegistry(local), false);
        assert.match(localSetupRunMessage(local), /loaded from a file/i);
        assert.deepEqual(setupCanRunFromRegistry(setup), true);
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


def test_simulator_rejects_malformed_success_responses() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import {
          listPublishedSetups,
          runPublishedSetup,
          getPublishedSetupRun,
        } from "./web/setup_runner_simulator.js";

        await assert.rejects(
          () => listPublishedSetups({
            fetchImpl: async () => ({
              ok: true,
              status: 200,
              async json() { throw new Error("invalid json"); },
            }),
          }),
          /Response was not valid JSON/
        );

        await assert.rejects(
          () => listPublishedSetups({
            fetchImpl: async () => ({
              ok: true,
              status: 200,
              async json() { return { ok: true }; },
            }),
          }),
          /setups array/
        );

        await assert.rejects(
          () => runPublishedSetup({
            setupId: "director-demo",
            fetchImpl: async () => ({
              ok: true,
              status: 200,
              async json() { return { ok: true }; },
            }),
          }),
          /run object/
        );

        await assert.rejects(
          () => runPublishedSetup({
            setupId: "director-demo",
            fetchImpl: async () => ({
              ok: true,
              status: 200,
              async json() { return { ok: true, run: {} }; },
            }),
          }),
          /Missing runId/
        );

        await assert.rejects(
          () => runPublishedSetup({
            setupId: "director-demo",
            fetchImpl: async () => ({
              ok: true,
              status: 200,
              async json() { return { ok: true, run: { runId: "run-000001", status: "queued" } }; },
            }),
          }),
          /Missing promptId/
        );

        await assert.rejects(
          () => getPublishedSetupRun({
            runId: "run-000001",
            fetchImpl: async () => ({
              ok: true,
              status: 200,
              async json() { return { ok: true }; },
            }),
          }),
          /run object/
        );

        await assert.rejects(
          () => getPublishedSetupRun({
            runId: "run-000001",
            fetchImpl: async () => ({
              ok: true,
              status: 200,
              async json() { return { ok: true, run: {} }; },
            }),
          }),
          /Missing runId/
        );
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


def test_simulator_prefers_catalog_metadata_title() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { setupDisplayTitle } from "./web/setup_runner_simulator.js";

        assert.equal(setupDisplayTitle({
          id: "rmgb-publish-v04",
          metadata: { title: "RMGB Publish v04" },
        }), "RMGB Publish v04");
        assert.equal(setupDisplayTitle({ id: "fallback-id", title: "Legacy Title" }), "Legacy Title");
        assert.equal(setupDisplayTitle({ id: "fallback-id" }), "fallback-id");
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


def test_simulator_builds_external_app_inputs_from_setup_surface() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import {
          defaultRunInputsFromSetup,
          runInputsFromAppValues,
          setupHasAppSurface,
          summarizeRunResultLines,
        } from "./web/setup_runner_simulator.js";

        const setup = {
          setupSurface: {
            app: {
              switch: {
                key: "switch",
                default: 2,
                options: [
                  { value: 0, label: "EXR", input: "sequence_folder", visible: true },
                  { value: 1, label: "QT", input: "qt_file", visible: true },
                  { value: 2, label: "Img", input: "single_file", visible: true },
                  { value: 3, label: "Prompt", input: "prompt", visible: false },
                ],
              },
              inputs: [
                { key: "sequence_folder", default: "/plates/demo" },
                { key: "qt_file", default: "/plates/demo/source.mov" },
                { key: "single_file", default: "/plates/demo/source.png" },
                { key: "prompt", visible: false, default: "hidden prompt" },
              ],
              outputs: [
                { key: "folder", default: "/renders/demo" },
                { key: "name", default: "demo" },
                { key: "version", default: "1" },
              ],
            },
          },
        };

        assert.equal(setupHasAppSurface(setup), true);
        assert.deepEqual(defaultRunInputsFromSetup(setup), {
          switch: 2,
          single_file: "/plates/demo/source.png",
          folder: "/renders/demo",
          name: "demo",
          version: "1",
        });
        assert.deepEqual(runInputsFromAppValues(setup, {
          switch: 1,
          qt_file: "/plates/demo/shot_010.mov",
          folder: "/custom/out",
        }), {
          switch: 1,
          qt_file: "/plates/demo/shot_010.mov",
          folder: "/custom/out",
          name: "demo",
          version: "1",
        });
        assert.deepEqual(summarizeRunResultLines({
          outputs: [
            {
              key: "result",
              label: "Result",
              items: [{ nodeId: "300", kind: "text", value: "/custom/out/demo_v001.mov" }],
            },
          ],
        }), ["Result: /custom/out/demo_v001.mov"]);
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


def test_simulator_submits_app_builder_param_fields_with_typed_values() -> None:
    """App-builder param picks are standalone declared fields: their defaults
    ride the payload, submitted overrides pass through, and form-string values
    coerce back to the JSON types the runner injects verbatim."""
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import {
          defaultRunInputsFromSetup,
          runInputsFromAppValues,
          appFieldValueTypes,
          coerceFieldValue,
        } from "./web/setup_runner_simulator.js";

        const setup = {
          setupSurface: {
            app: {
              inputs: [
                {
                  key: "param_30_steps",
                  label: "Main sampler: steps",
                  visible: true,
                  standalone: true,
                  appParam: true,
                  valueType: "int",
                  target: { node: "30", input: "steps" },
                  default: 8,
                },
                {
                  key: "param_30_enabled",
                  label: "Main sampler: enabled",
                  visible: true,
                  standalone: true,
                  appParam: true,
                  valueType: "boolean",
                  target: { node: "30", input: "enabled" },
                  default: true,
                },
              ],
              outputs: [],
              results: [],
            },
          },
        };

        assert.deepEqual(defaultRunInputsFromSetup(setup), {
          param_30_steps: 8,
          param_30_enabled: true,
        });
        assert.deepEqual(
          runInputsFromAppValues(setup, { param_30_steps: 12, param_30_enabled: false }),
          { param_30_steps: 12, param_30_enabled: false },
        );
        assert.deepEqual(appFieldValueTypes(setup), {
          param_30_steps: "int",
          param_30_enabled: "boolean",
        });
        assert.equal(coerceFieldValue("12", "int"), 12);
        assert.equal(coerceFieldValue("1.5", "float"), 1.5);
        assert.equal(coerceFieldValue("true", "boolean"), true);
        assert.equal(coerceFieldValue(false, "boolean"), false);
        assert.equal(coerceFieldValue("not-a-number", "int"), "not-a-number");
        assert.equal(coerceFieldValue("plain", undefined), "plain");
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr
