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


def test_infer_setup_surface_from_koolook_groups() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { inferSetupSurface } from "./web/sidebar/published_surface.js";

        const surface = inferSetupSurface({
          nodes: [
            { id: 12, type: "Load Image", title: "Source image", pos: [40, 40], size: [180, 80] },
            { id: 20, type: "Preview Image", title: "Preview", pos: [420, 40], size: [180, 80] },
            { id: 30, type: "Note", title: "Outside", pos: [800, 40], size: [180, 80] },
          ],
          groups: [
            { title: "Koolook Input", bounding: [20, 20, 240, 140] },
            { title: "Koolook Output", pos: [400, 20], size: [240, 140] },
          ],
        });

        assert.deepEqual(surface, {
          sourceInputs: [{
            group: "Koolook Input",
            nodes: [{ id: "12", type: "Load Image", title: "Source image" }],
          }],
          outputs: [{
            group: "Koolook Output",
            nodes: [{ id: "20", type: "Preview Image", title: "Preview" }],
          }],
          controls: [],
          app: { inputs: [], outputs: [], results: [] },
        });
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


def test_infer_setup_surface_app_contract_from_publish_nodes() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { inferSetupSurface } from "./web/sidebar/published_surface.js";

        const surface = inferSetupSurface({
          nodes: [
            {
              id: 100,
              type: "Koolook_PublishInput",
              title: "Koolook Publish Input",
              pos: [40, 60],
              size: [360, 320],
              widgets_values: [
                "Img",
                "/shots/example/frames",
                "/shots/example/movie.mov",
                "/shots/example/image.png",
                "unused prompt",
              ],
            },
            {
              id: 200,
              type: "Koolook_PublishOutput",
              title: "Koolook Publish Output",
              pos: [520, 60],
              size: [360, 240],
              widgets_values: [
                "/shots/example/output",
                "publish-OUT",
                "1",
              ],
            },
            {
              id: 300,
              type: "Koolook_PublishResult",
              title: "Koolook Publish Result",
              pos: [520, 340],
              size: [360, 160],
              widgets_values: ["/shots/example/output/publish-OUT_v001.mov"],
            },
          ],
          groups: [
            { title: "Koolook Input", bounding: [20, 20, 440, 420] },
            { title: "Koolook Output", bounding: [500, 20, 360, 560] },
          ],
        });

        assert.deepEqual(surface.app, {
          inputs: [
            { key: "sequence_folder", label: "Sequence folder", visible: true, target: { node: "100", input: "sequence_folder" }, default: "/shots/example/frames" },
            { key: "qt_file", label: "QT file", visible: true, target: { node: "100", input: "qt_file" }, default: "/shots/example/movie.mov" },
            { key: "single_file", label: "Single file", visible: true, target: { node: "100", input: "single_file" }, default: "/shots/example/image.png" },
            { key: "prompt", label: "Prompt", visible: true, standalone: true, multiline: true, target: { node: "100", input: "prompt" }, default: "", placeholder: "unused prompt", help: "Describe the shot in one simple line: subject + action + setting." },
          ],
          outputs: [
            { key: "folder", label: "Output folder", visible: true, target: { node: "200", input: "folder" }, default: "/shots/example/output" },
            { key: "name", label: "Output name", visible: true, target: { node: "200", input: "name" }, default: "publish-OUT" },
            { key: "version", label: "Version", visible: true, target: { node: "200", input: "version" }, default: "1" },
          ],
          results: [
            { key: "result", label: "Result", visible: true, target: { node: "300", input: "result" }, default: "/shots/example/output/publish-OUT_v001.mov" },
          ],
          switch: {
            key: "switch",
            label: "Input type",
            visible: true,
            target: { node: "100", input: "mode" },
            default: 2,
            options: [
              { value: 0, label: "EXR", visible: true, input: "sequence_folder" },
              { value: 1, label: "QT", visible: true, input: "qt_file" },
              { value: 2, label: "Img", visible: true, input: "single_file" },
              { value: 3, label: "Prompt", visible: false, input: "prompt" },
            ],
          },
        });
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


def test_prompt_is_an_always_on_field_not_a_source_mode() -> None:
    # The prompt must render independently of the EXR/QT/Img source switch:
    # visible + standalone, with the author's prompt-widget text surfaced as a
    # placeholder hint (not a submitted default), and excluded from the switch's
    # source options.
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { inferSetupSurface } from "./web/sidebar/published_surface.js";

        const surface = inferSetupSurface({
          nodes: [
            {
              id: 100,
              type: "Koolook_PublishInput",
              title: "Koolook Publish Input",
              pos: [40, 60],
              size: [360, 320],
              widgets_values: [
                "EXR",
                "/shots/example/frames",
                "/shots/example/movie.mov",
                "/shots/example/image.png",
                "a bear talking to the camera in a sunny forest",
              ],
            },
          ],
          groups: [
            { title: "Koolook Input", bounding: [20, 20, 440, 420] },
          ],
        });

        const prompt = surface.app.inputs.find(field => field.key === "prompt");
        assert.ok(prompt, "prompt field is present");
        assert.equal(prompt.visible, true);
        assert.equal(prompt.standalone, true);
        assert.equal(prompt.multiline, true);
        assert.equal(prompt.default, "");
        assert.equal(prompt.placeholder, "a bear talking to the camera in a sunny forest");
        assert.ok(prompt.help && prompt.help.length > 0, "prompt carries a help hint");

        // The prompt is never offered as a source mode in the switch.
        const promptOption = surface.app.switch.options.find(o => o.input === "prompt");
        assert.equal(promptOption.visible, false);
        const visibleSources = surface.app.switch.options.filter(o => o.visible !== false).map(o => o.input);
        assert.deepEqual(visibleSources, ["sequence_folder", "qt_file", "single_file"]);
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


def test_infer_setup_surface_ignores_malformed_group_geometry() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { inferSetupSurface } from "./web/sidebar/published_surface.js";

        const surface = inferSetupSurface({
          nodes: [
            { id: 12, type: "Load Image", title: "Source image", pos: [40, 40], size: [180, 80] },
          ],
          groups: [
            { title: "Koolook Input", bounding: ["bad", 20, 240, 140] },
          ],
        });

        assert.deepEqual(surface.sourceInputs, []);
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr
