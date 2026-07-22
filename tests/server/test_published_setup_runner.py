from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from koolook_setup_runner import PublishedSetupRunner, SetupRunError
from koolook_setups import PublishedSetupRegistry, StaticSetupStorage
from tests.server.test_published_setup_registry import _valid_setup


class FakeComfyClient:
    def __init__(self, prompt_response=None, *, history=None, queue=None, queue_error=None) -> None:
        self.prompt_response = prompt_response or {"prompt_id": "comfy-prompt-1"}
        self.submitted_prompts: list[dict] = []
        self.history = history or {}
        self.queue = queue or {"queue_running": [], "queue_pending": []}
        self.queue_error = queue_error

    async def queue_prompt(self, prompt: dict) -> dict:
        if self.queue_error is not None:
            raise self.queue_error
        self.submitted_prompts.append(prompt)
        return self.prompt_response

    async def get_history(self, prompt_id: str) -> dict:
        return {prompt_id: self.history[prompt_id]} if prompt_id in self.history else {}

    async def get_queue(self) -> dict:
        return self.queue


def test_run_setup_injects_declared_inputs_into_cloned_prompt() -> None:
    async def exercise() -> None:
        setup = _valid_setup()
        original_prompt = deepcopy(setup["apiPrompt"])
        registry = PublishedSetupRegistry(StaticSetupStorage([setup]))
        comfy = FakeComfyClient()
        runner = PublishedSetupRunner(registry, comfy)

        result = await runner.runSetup("ltx-director-demo", {"prompt": "external prompt"})

        assert result == {
            "runId": "run-000001",
            "promptId": "comfy-prompt-1",
            "status": "queued",
        }
        assert comfy.submitted_prompts == [
            {"12": {"class_type": "Text Multiline", "inputs": {"text": "external prompt"}}}
        ]
        assert setup["apiPrompt"] == original_prompt

    asyncio.run(exercise())


def test_run_setup_rejects_inputs_not_declared_by_contract() -> None:
    async def exercise() -> None:
        registry = PublishedSetupRegistry(StaticSetupStorage([_valid_setup()]))
        runner = PublishedSetupRunner(registry, FakeComfyClient())

        with pytest.raises(SetupRunError) as exc_info:
            await runner.runSetup("ltx-director-demo", {"prompt": "ok", "seed": 123})

        assert exc_info.value.status_code == 400
        assert exc_info.value.errors == ["input 'seed' is not declared by this setup"]

    asyncio.run(exercise())


def test_run_setup_rejects_non_callable_setup() -> None:
    async def exercise() -> None:
        setup = _valid_setup()
        setup["apiPrompt"] = None
        setup["validation"] = {"status": "draft", "diagnostics": ["API prompt conversion pending."]}
        registry = PublishedSetupRegistry(StaticSetupStorage([setup]))
        runner = PublishedSetupRunner(registry, FakeComfyClient())

        with pytest.raises(SetupRunError) as exc_info:
            await runner.runSetup("ltx-director-demo", {"prompt": "external prompt"})

        assert exc_info.value.status_code == 400
        assert exc_info.value.errors == ["Published setup 'ltx-director-demo' is not callable."]

    asyncio.run(exercise())


def test_run_setup_translates_comfy_queue_failures() -> None:
    async def exercise() -> None:
        registry = PublishedSetupRegistry(StaticSetupStorage([_valid_setup()]))
        runner = PublishedSetupRunner(registry, FakeComfyClient(queue_error=RuntimeError("server down")))

        with pytest.raises(SetupRunError) as exc_info:
            await runner.runSetup("ltx-director-demo", {"prompt": "external prompt"})

        assert exc_info.value.status_code == 502
        assert exc_info.value.errors == ["ComfyUI queue request failed: server down"]

    asyncio.run(exercise())


def test_get_run_reports_running_state_from_comfy_queue() -> None:
    async def exercise() -> None:
        registry = PublishedSetupRegistry(StaticSetupStorage([_valid_setup()]))
        runner = PublishedSetupRunner(
            registry,
            FakeComfyClient(queue={"queue_running": [[7, "comfy-prompt-1", {}]], "queue_pending": []}),
        )

        queued = await runner.runSetup("ltx-director-demo", {"prompt": "external prompt"})
        status = await runner.getRun(queued["runId"])

        assert status == {
            "runId": "run-000001",
            "setupId": "ltx-director-demo",
            "promptId": "comfy-prompt-1",
            "status": "running",
            "outputs": [],
        }

    asyncio.run(exercise())


def test_get_run_reports_lost_when_prompt_is_missing_from_history_and_queue() -> None:
    async def exercise() -> None:
        registry = PublishedSetupRegistry(StaticSetupStorage([_valid_setup()]))
        runner = PublishedSetupRunner(
            registry,
            FakeComfyClient(queue={"queue_running": [], "queue_pending": []}),
        )

        queued = await runner.runSetup("ltx-director-demo", {"prompt": "external prompt"})
        status = await runner.getRun(queued["runId"])

        assert status == {
            "runId": "run-000001",
            "setupId": "ltx-director-demo",
            "promptId": "comfy-prompt-1",
            "status": "lost",
            "outputs": [],
        }

    asyncio.run(exercise())


def test_get_run_preserves_comfy_history_status_payload() -> None:
    async def exercise() -> None:
        registry = PublishedSetupRegistry(StaticSetupStorage([_valid_setup()]))
        comfy_status = {
            "completed": False,
            "status_str": "error",
            "messages": [["execution_error", {"node_id": "12", "exception_message": "missing file"}]],
        }
        runner = PublishedSetupRunner(
            registry,
            FakeComfyClient(
                history={
                    "comfy-prompt-1": {
                        "status": comfy_status,
                        "outputs": {},
                    }
                }
            ),
        )

        queued = await runner.runSetup("ltx-director-demo", {"prompt": "external prompt"})
        status = await runner.getRun(queued["runId"])

        assert status == {
            "runId": "run-000001",
            "setupId": "ltx-director-demo",
            "promptId": "comfy-prompt-1",
            "status": "failed",
            "comfyStatus": comfy_status,
            "outputs": [{"key": "video", "label": "Video", "type": "video", "items": []}],
        }

    asyncio.run(exercise())


def test_run_setup_accepts_inputs_declared_by_app_surface_contract() -> None:
    async def exercise() -> None:
        setup = _valid_setup()
        setup["inputContract"] = {"inputs": []}
        setup["outputContract"] = {"outputs": []}
        setup["visualGraph"]["groups"] = [
            {"title": "Koolook Input", "bounding": [0, 0, 300, 200]},
            {"title": "Koolook Output", "bounding": [400, 0, 300, 200]},
        ]
        setup["visualGraph"]["nodes"][0]["pos"] = [20, 20]
        setup["setupSurface"] = {
            "sourceInputs": [
                {
                    "group": "Koolook Input",
                    "nodes": [{"id": "12", "type": "Text Multiline", "title": "Prompt"}],
                }
            ],
            "outputs": [
                {
                    "group": "Koolook Output",
                    "nodes": [{"id": "20", "type": "Preview Image", "title": "Preview"}],
                }
            ],
            "controls": [],
            "app": {
                "inputs": [
                    {
                        "key": "prompt",
                        "label": "Prompt",
                        "visible": True,
                        "target": {"node": "12", "input": "text"},
                        "default": "demo prompt",
                    }
                ],
                "outputs": [],
                "results": [],
            },
        }
        registry = PublishedSetupRegistry(StaticSetupStorage([setup]))
        comfy = FakeComfyClient()
        runner = PublishedSetupRunner(registry, comfy)

        await runner.runSetup("ltx-director-demo", {"prompt": "app contract prompt"})

        assert comfy.submitted_prompts == [
            {"12": {"class_type": "Text Multiline", "inputs": {"text": "app contract prompt"}}}
        ]

    asyncio.run(exercise())


def test_run_setup_accepts_output_controls_declared_by_app_surface() -> None:
    async def exercise() -> None:
        setup = _valid_setup()
        setup["inputContract"] = {"inputs": []}
        setup["outputContract"] = {"outputs": []}
        setup["visualGraph"] = {
            "nodes": [
                {
                    "id": 200,
                    "type": "Koolook_PublishOutput",
                    "title": "Koolook Publish Output",
                    "pos": [20, 20],
                    "size": [320, 160],
                    "inputs": [],
                    "widgets_values": ["/shots/example/output", "demo", "1"],
                }
            ],
            "links": [],
            "groups": [
                {"title": "Koolook Input", "bounding": [0, 0, 80, 80]},
                {"title": "Koolook Output", "bounding": [0, 0, 380, 220]},
            ],
        }
        setup["apiPrompt"] = {
            "200": {
                "class_type": "Koolook_PublishOutput",
                "inputs": {
                    "folder": "/shots/example/output",
                    "name": "demo",
                    "version": "1",
                },
            }
        }
        setup["setupSurface"] = {
            "sourceInputs": [{"group": "Koolook Input", "nodes": [{"id": "200", "type": "Koolook_PublishOutput", "title": "Koolook Publish Output"}]}],
            "outputs": [{"group": "Koolook Output", "nodes": [{"id": "200", "type": "Koolook_PublishOutput", "title": "Koolook Publish Output"}]}],
            "controls": [],
            "app": {
                "inputs": [],
                "outputs": [
                    {
                        "key": "folder",
                        "label": "Output folder",
                        "visible": True,
                        "target": {"node": "200", "input": "folder"},
                        "default": "/shots/example/output",
                    }
                ],
                "results": [],
            },
        }
        registry = PublishedSetupRegistry(StaticSetupStorage([setup]))
        comfy = FakeComfyClient()
        runner = PublishedSetupRunner(registry, comfy)

        await runner.runSetup("ltx-director-demo", {"folder": "/shots/custom-output"})

        assert comfy.submitted_prompts == [
            {
                "200": {
                    "class_type": "Koolook_PublishOutput",
                    "inputs": {
                        "folder": "/shots/custom-output",
                        "name": "demo",
                        "version": "1",
                    },
                }
            }
        ]

    asyncio.run(exercise())


def test_get_run_reports_app_surface_results_without_output_contract() -> None:
    async def exercise() -> None:
        setup = _valid_setup()
        setup["inputContract"] = {"inputs": []}
        setup["outputContract"] = {"outputs": []}
        setup["visualGraph"] = {
            "nodes": [
                {
                    "id": 100,
                    "type": "Koolook_PublishInput",
                    "title": "Koolook Publish Input",
                    "pos": [20, 20],
                    "size": [320, 240],
                    "inputs": [],
                    "widgets_values": ["Img", "", "", "", ""],
                },
                {
                    "id": 200,
                    "type": "Koolook_PublishOutput",
                    "title": "Koolook Publish Output",
                    "pos": [420, 20],
                    "size": [320, 160],
                    "inputs": [],
                    "widgets_values": ["/shots/example/output", "demo", "1"],
                },
                {
                    "id": 300,
                    "type": "Koolook_PublishResult",
                    "title": "Koolook Publish Result",
                    "pos": [420, 220],
                    "size": [320, 120],
                    "inputs": [],
                    "widgets_values": ["/shots/example/output/default.mov"],
                },
            ],
            "links": [],
            "groups": [
                {"title": "Koolook Input", "bounding": [0, 0, 360, 300]},
                {"title": "Koolook Output", "bounding": [400, 0, 380, 380]},
            ],
        }
        setup["apiPrompt"] = {
            "100": {
                "class_type": "Koolook_PublishInput",
                "inputs": {
                    "mode": "Img",
                    "sequence_folder": "",
                    "qt_file": "",
                    "single_file": "",
                    "prompt": "",
                },
            },
            "200": {
                "class_type": "Koolook_PublishOutput",
                "inputs": {
                    "folder": "/shots/example/output",
                    "name": "demo",
                    "version": "1",
                },
            },
            "300": {
                "class_type": "Koolook_PublishResult",
                "inputs": {"result": "/shots/example/output/default.mov"},
            },
        }
        setup["setupSurface"] = {
            "sourceInputs": [
                {
                    "group": "Koolook Input",
                    "nodes": [{"id": "100", "type": "Koolook_PublishInput", "title": "Koolook Publish Input"}],
                }
            ],
            "outputs": [
                {
                    "group": "Koolook Output",
                    "nodes": [
                        {"id": "200", "type": "Koolook_PublishOutput", "title": "Koolook Publish Output"},
                        {"id": "300", "type": "Koolook_PublishResult", "title": "Koolook Publish Result"},
                    ],
                }
            ],
            "controls": [],
            "app": {
                "inputs": [],
                "outputs": [
                    {
                        "key": "folder",
                        "label": "Output folder",
                        "visible": True,
                        "target": {"node": "200", "input": "folder"},
                        "default": "/shots/example/output",
                    }
                ],
                "results": [
                    {
                        "key": "result",
                        "label": "Result",
                        "visible": True,
                        "target": {"node": "300", "input": "result"},
                        "default": "/shots/example/output/default.mov",
                    }
                ],
            },
        }
        registry = PublishedSetupRegistry(StaticSetupStorage([setup]))
        runner = PublishedSetupRunner(
            registry,
            FakeComfyClient(
                history={
                    "comfy-prompt-1": {
                        "status": {"completed": True, "status_str": "success"},
                        "outputs": {
                            "300": {
                                "text": ["/shots/example/output/generated.mov"]
                            }
                        },
                    }
                }
            ),
        )

        queued = await runner.runSetup("ltx-director-demo", {})
        status = await runner.getRun(queued["runId"])

        assert status["outputs"] == [
            {
                "key": "folder",
                "label": "Output folder",
                "type": "output",
                "visible": True,
                "target": {"node": "200", "input": "folder"},
                "default": "/shots/example/output",
                "items": [],
            },
            {
                "key": "result",
                "label": "Result",
                "type": "result",
                "visible": True,
                "target": {"node": "300", "input": "result"},
                "default": "/shots/example/output/default.mov",
                "items": [
                    {
                        "nodeId": "300",
                        "kind": "text",
                        "value": "/shots/example/output/generated.mov",
                    }
                ],
            },
        ]

    asyncio.run(exercise())


def test_run_setup_uses_execution_map_router_to_keep_selected_writer() -> None:
    async def exercise() -> None:
        setup = _valid_setup()
        setup["inputContract"] = {"inputs": []}
        setup["outputContract"] = {"outputs": []}
        setup["visualGraph"] = {
            "nodes": [
                {
                    "id": 100,
                    "type": "Koolook_PublishInput",
                    "inputs": [
                        {"name": "mode", "widget": {"name": "mode"}},
                        {"name": "single_file", "widget": {"name": "single_file"}},
                    ],
                },
                {"id": 200, "type": "Koolook_PublishOutput", "inputs": []},
                {"id": 300, "type": "RMBG", "inputs": [{"name": "image", "link": 1}]},
                {"id": 400, "type": "Koolook_PublishRouter", "inputs": [{"name": "selector"}, {"name": "payload"}]},
                {"id": 301, "type": "EasyAIPipeline", "inputs": []},
                {"id": 302, "type": "EasyAIPipeline", "inputs": []},
                {"id": 303, "type": "EasyAIPipeline", "inputs": []},
                {"id": 311, "type": "SaveEXRFrames", "inputs": []},
                {"id": 312, "type": "Easy_VideoCombine", "inputs": []},
                {"id": 313, "type": "SaveImageAndPromptExact", "inputs": [{"name": "filepath", "widget": {"name": "filepath"}}]},
            ],
            "links": [],
            "groups": [],
        }
        setup["apiPrompt"] = {
            "100": {
                "class_type": "Koolook_PublishInput",
                "inputs": {
                    "mode": "Img",
                    "sequence_folder": "",
                    "qt_file": "",
                    "single_file": "/shots/source/input.png",
                    "prompt": "",
                },
            },
            "200": {
                "class_type": "Koolook_PublishOutput",
                "inputs": {"folder": "/shots/output", "name": "publish-OUT", "version": "2"},
            },
            "300": {"class_type": "RMBG", "inputs": {"image": ["100", 2]}},
            "301": {"class_type": "EasyAIPipeline", "inputs": {"base_directory_path": ["200", 0], "extension": ".exr"}},
            "302": {"class_type": "EasyAIPipeline", "inputs": {"base_directory_path": ["200", 0], "extension": ".mov"}},
            "303": {"class_type": "EasyAIPipeline", "inputs": {"base_directory_path": ["200", 0], "extension": ".png"}},
            "400": {"class_type": "Koolook_PublishRouter", "inputs": {"selector": ["100", 4], "payload": ["300", 0]}},
            "311": {"class_type": "SaveEXRFrames", "inputs": {"filepath": ["301", 0], "images": ["400", 0]}},
            "312": {"class_type": "Easy_VideoCombine", "inputs": {"filepath": ["302", 0], "images": ["400", 1]}},
            "313": {"class_type": "SaveImageAndPromptExact", "inputs": {"filepath": ["303", 0], "image": ["400", 2]}},
        }
        setup["setupSurface"] = {
            "sourceInputs": [
                {
                    "group": "Koolook Input",
                    "nodes": [{"id": "100", "type": "Koolook_PublishInput", "title": "Koolook Publish Input"}],
                }
            ],
            "outputs": [
                {
                    "group": "Koolook Output",
                    "nodes": [
                        {"id": "200", "type": "Koolook_PublishOutput", "title": "Koolook Publish Output"},
                        {"id": "400", "type": "Koolook_PublishRouter", "title": "Koolook Publish Router"},
                        {"id": "311", "type": "SaveEXRFrames", "title": "EXR writer"},
                        {"id": "312", "type": "Easy_VideoCombine", "title": "QT writer"},
                        {"id": "313", "type": "SaveImageAndPromptExact", "title": "Image writer"},
                    ],
                }
            ],
            "controls": [],
            "app": {
                "inputs": [
                    {
                        "key": "single_file",
                        "label": "Single file",
                        "visible": True,
                        "target": {"node": "100", "input": "single_file"},
                        "default": "/shots/source/input.png",
                    }
                ],
                "outputs": [
                    {
                        "key": "folder",
                        "label": "Output folder",
                        "visible": True,
                        "target": {"node": "200", "input": "folder"},
                        "default": "/shots/output",
                    },
                    {
                        "key": "name",
                        "label": "Output name",
                        "visible": True,
                        "target": {"node": "200", "input": "name"},
                        "default": "publish-OUT",
                    },
                    {
                        "key": "version",
                        "label": "Version",
                        "visible": True,
                        "target": {"node": "200", "input": "version"},
                        "default": "2",
                    },
                ],
                "results": [],
                "switch": {
                    "key": "switch",
                    "label": "Input type",
                    "visible": True,
                    "target": {"node": "100", "input": "mode"},
                    "default": 2,
                    "options": [
                        {"value": 0, "label": "EXR", "visible": False, "input": "single_file"},
                        {"value": 1, "label": "QT", "visible": False, "input": "single_file"},
                        {"value": 2, "label": "Img", "visible": True, "input": "single_file"},
                        {"value": 3, "label": "Prompt", "visible": False, "input": "single_file"},
                    ],
                },
            },
        }
        setup["executionMap"] = {
            "version": 1,
            "routers": [
                {
                    "node": "400",
                    "switchKey": "switch",
                    "selector": {"node": "100", "output": 4},
                    "payload": {"node": "300", "output": 0},
                    "branches": {
                        "0": {"label": "EXR", "output": 0, "writerNodes": ["311"]},
                        "1": {"label": "QT", "output": 1, "writerNodes": ["312"]},
                        "2": {"label": "Img", "output": 2, "writerNodes": ["313"]},
                        "3": {"label": "Prompt", "output": 3, "writerNodes": []},
                    },
                }
            ],
        }
        registry = PublishedSetupRegistry(StaticSetupStorage([setup]))
        comfy = FakeComfyClient()
        runner = PublishedSetupRunner(registry, comfy)

        await runner.runSetup(
            "ltx-director-demo",
            {
                "switch": 2,
                "single_file": "/shots/source/input.png",
                "folder": "/shots/output",
                "name": "external",
                "version": "7",
            },
        )

        assert set(comfy.submitted_prompts[0]) == {"100", "200", "300", "303", "400", "313"}
        assert comfy.submitted_prompts[0]["100"]["inputs"]["mode"] == "Img"
        assert comfy.submitted_prompts[0]["313"]["inputs"] == {
            "filepath": ["303", 0],
            "image": ["400", 2],
        }

    asyncio.run(exercise())


def test_get_run_result_path_follows_output_switch_not_input() -> None:
    """When a mode-switched Koolook_PublishResult is driven by the OUTPUT switch,
    the reported result path follows the chosen output type. An EXR source
    (input switch 0) with QT output (output_switch 1) must report the QT movie
    path from the result index-switch's value1 branch, not the EXR value0."""
    async def exercise() -> None:
        setup = _valid_setup()
        setup["inputContract"] = {"inputs": []}
        setup["outputContract"] = {"outputs": []}
        setup["visualGraph"] = {
            "nodes": [
                {"id": 188, "type": "Koolook_PublishInput", "inputs": [{"name": "mode", "widget": {"name": "mode"}}]},
                {"id": 200, "type": "Koolook_PublishOutput", "inputs": [{"name": "output_mode", "widget": {"name": "output_mode"}}]},
                {"id": 205, "type": "Koolook_PublishResult", "inputs": [{"name": "result", "link": 900}]},
            ],
            "links": [[900, 179, 0, 205, 0, "STRING"]],
            "groups": [],
        }
        setup["apiPrompt"] = {
            "188": {
                "class_type": "Koolook_PublishInput",
                "inputs": {"mode": "EXR", "sequence_folder": "/shots/source/frames", "qt_file": "", "single_file": "", "prompt": ""},
            },
            "200": {
                "class_type": "Koolook_PublishOutput",
                "inputs": {"folder": "/shots/output", "name": "publish-OUT", "version": "2", "output_mode": "QT", "input_switch": ["188", 4]},
            },
            # Result index-switch driven by the OUTPUT switch (node 200, slot 3).
            "179": {
                "class_type": "Switch any [Crystools]",
                "inputs": {"index": ["200", 3], "value0": ["301", 0], "value1": ["302", 0], "value2": ["303", 0]},
            },
            "301": {"class_type": "EasyAIPipeline", "inputs": {"WRITE_file_path": "/shots/output/publish-OUT_v002.exr"}},
            "302": {"class_type": "Easy_VideoCombine", "inputs": {"video_path": "/shots/output/publish-OUT_v002.mov"}},
            "303": {"class_type": "EasyAIPipeline", "inputs": {"WRITE_file_path": "/shots/output/publish-OUT_v002.png"}},
            "205": {"class_type": "Koolook_PublishResult", "inputs": {"result": ["179", 0]}},
        }
        setup["setupSurface"] = {
            "sourceInputs": [
                {"group": "Koolook Input", "nodes": [{"id": "188", "type": "Koolook_PublishInput", "title": "In"}]}
            ],
            "outputs": [
                {
                    "group": "Koolook Output",
                    "nodes": [
                        {"id": "200", "type": "Koolook_PublishOutput", "title": "Out"},
                        {"id": "205", "type": "Koolook_PublishResult", "title": "Result"},
                        {"id": "301", "type": "EasyAIPipeline", "title": "EXR path"},
                        {"id": "302", "type": "Easy_VideoCombine", "title": "QT path"},
                        {"id": "303", "type": "EasyAIPipeline", "title": "Img path"},
                    ],
                }
            ],
            "controls": [],
            "app": {
                "inputs": [
                    {"key": "sequence_folder", "label": "Sequence folder", "visible": True, "target": {"node": "188", "input": "sequence_folder"}, "default": "/shots/source/frames"}
                ],
                "outputs": [],
                "results": [
                    {"key": "result", "label": "Result", "visible": True, "target": {"node": "205", "input": "result"}, "default": ""}
                ],
                "switch": {
                    "key": "switch",
                    "label": "Input type",
                    "visible": True,
                    "target": {"node": "188", "input": "mode"},
                    "default": 0,
                    "options": [
                        {"value": 0, "label": "EXR", "visible": True, "input": "sequence_folder"},
                        {"value": 1, "label": "QT", "visible": True, "input": "sequence_folder"},
                        {"value": 2, "label": "Img", "visible": True, "input": "sequence_folder"},
                    ],
                },
                "outputSwitch": {
                    "key": "output_switch",
                    "label": "Output type",
                    "visible": True,
                    "sameAsInput": True,
                    "target": {"node": "200", "input": "output_mode"},
                    "default": -1,
                    "options": [
                        {"value": 0, "label": "EXR", "visible": True},
                        {"value": 1, "label": "QT", "visible": True},
                        {"value": 2, "label": "Img", "visible": True},
                    ],
                },
            },
        }
        registry = PublishedSetupRegistry(StaticSetupStorage([setup]))
        comfy = FakeComfyClient(
            history={
                "comfy-prompt-1": {
                    "status": {"completed": True, "status_str": "success"},
                    # History carries paths for both branch nodes; the runner must
                    # pick the QT one because output_switch selected it.
                    "outputs": {
                        "301": {"text": ["/shots/output/publish-OUT_v002.exr"]},
                        "302": {"text": ["/shots/output/publish-OUT_v002.mov"]},
                    },
                }
            }
        )
        runner = PublishedSetupRunner(registry, comfy)

        queued = await runner.runSetup("ltx-director-demo", {"switch": 0, "output_switch": 1})
        status = await runner.getRun(queued["runId"])

        result_output = next(o for o in status["outputs"] if o["key"] == "result")
        values = [item["value"] for item in result_output["items"]]
        assert values == ["/shots/output/publish-OUT_v002.mov"], (
            f"result should follow output_switch (QT .mov), got {values!r}"
        )

    asyncio.run(exercise())


def _divergent_output_setup(
    *,
    output_default: int = 1,
    output_options: list[dict[str, object]] | None = None,
    same_as_input: bool = True,
) -> dict:
    """Setup with an EXR/QT/Img writer router driven by the output switch."""
    setup = _valid_setup()
    setup["inputContract"] = {"inputs": []}
    setup["outputContract"] = {"outputs": []}
    setup["visualGraph"] = {
        "nodes": [
            {"id": 100, "type": "Koolook_PublishInput", "inputs": [{"name": "mode", "widget": {"name": "mode"}}]},
            {"id": 200, "type": "Koolook_PublishOutput", "inputs": [{"name": "output_mode", "widget": {"name": "output_mode"}}]},
        ],
        "links": [],
        "groups": [],
    }
    setup["apiPrompt"] = {
        "100": {
            "class_type": "Koolook_PublishInput",
            "inputs": {
                "mode": "EXR",
                "sequence_folder": "/shots/source/frames",
                "qt_file": "",
                "single_file": "",
                "prompt": "",
            },
        },
        "200": {
            "class_type": "Koolook_PublishOutput",
            "inputs": {"folder": "/shots/output", "name": "publish-OUT", "version": "2", "output_mode": "QT", "input_switch": ["100", 4]},
        },
        "300": {"class_type": "RMBG", "inputs": {"image": ["100", 0]}},
        "301": {"class_type": "EasyAIPipeline", "inputs": {"base_directory_path": ["200", 0], "extension": ".exr"}},
        "302": {"class_type": "EasyAIPipeline", "inputs": {"base_directory_path": ["200", 0], "extension": ".mov"}},
        "303": {"class_type": "EasyAIPipeline", "inputs": {"base_directory_path": ["200", 0], "extension": ".png"}},
        "400": {"class_type": "Koolook_PublishRouter", "inputs": {"selector": ["200", 3], "payload": ["300", 0]}},
        "311": {"class_type": "SaveEXRFrames", "inputs": {"filepath": ["301", 0], "images": ["400", 0]}},
        "312": {"class_type": "Easy_VideoCombine", "inputs": {"filepath": ["302", 0], "images": ["400", 1]}},
        "313": {"class_type": "SaveImageAndPromptExact", "inputs": {"filepath": ["303", 0], "image": ["400", 2]}},
    }
    setup["setupSurface"] = {
        "sourceInputs": [
            {"group": "Koolook Input", "nodes": [{"id": "100", "type": "Koolook_PublishInput", "title": "In"}]}
        ],
        "outputs": [
            {
                "group": "Koolook Output",
                "nodes": [
                    {"id": "200", "type": "Koolook_PublishOutput", "title": "Out"},
                    {"id": "400", "type": "Koolook_PublishRouter", "title": "Router"},
                    {"id": "311", "type": "SaveEXRFrames", "title": "EXR writer"},
                    {"id": "312", "type": "Easy_VideoCombine", "title": "QT writer"},
                    {"id": "313", "type": "SaveImageAndPromptExact", "title": "Image writer"},
                ],
            }
        ],
        "controls": [],
        "app": {
            "inputs": [
                {"key": "sequence_folder", "label": "Sequence folder", "visible": True, "target": {"node": "100", "input": "sequence_folder"}, "default": "/shots/source/frames"}
            ],
            "outputs": [
                {"key": "folder", "label": "Output folder", "visible": True, "target": {"node": "200", "input": "folder"}, "default": "/shots/output"},
            ],
            "results": [],
            "switch": {
                "key": "switch",
                "label": "Input type",
                "visible": True,
                "target": {"node": "100", "input": "mode"},
                "default": 0,
                "options": [
                    {"value": 0, "label": "EXR", "visible": True, "input": "sequence_folder"},
                    {"value": 1, "label": "QT", "visible": False, "input": "sequence_folder"},
                    {"value": 2, "label": "Img", "visible": False, "input": "sequence_folder"},
                ],
            },
            "outputSwitch": {
                "key": "output_switch",
                "label": "Output type",
                "visible": True,
                "sameAsInput": same_as_input,
                "target": {"node": "200", "input": "output_mode"},
                "default": output_default,
                "options": output_options
                if output_options is not None
                else [
                    {"value": 0, "label": "EXR", "visible": True},
                    {"value": 1, "label": "QT", "visible": True},
                    {"value": 2, "label": "Img", "visible": True},
                ],
            },
        },
    }
    setup["executionMap"] = {
        "version": 1,
        "routers": [
            {
                "node": "400",
                "switchKey": "output_switch",
                "selector": {"node": "200", "output": 3},
                "payload": {"node": "300", "output": 0},
                "branches": {
                    "0": {"label": "EXR", "output": 0, "writerNodes": ["311"]},
                    "1": {"label": "QT", "output": 1, "writerNodes": ["312"]},
                    "2": {"label": "Img", "output": 2, "writerNodes": ["313"]},
                },
            }
        ],
    }
    return setup


def test_run_setup_output_switch_selects_writer_independently_of_input() -> None:
    """Decoupled output type: the writer router is driven by the OUTPUT switch,
    so an EXR source (input switch 0) can still write a QT movie (output switch
    1). The runner must keep the QT writer and prune the EXR/Img writers, proving
    the writer branch follows output_switch, not the input switch."""
    async def exercise() -> None:
        setup = _divergent_output_setup()
        registry = PublishedSetupRegistry(StaticSetupStorage([setup]))
        comfy = FakeComfyClient()
        runner = PublishedSetupRunner(registry, comfy)

        # EXR in, QT out — the two switches disagree on purpose.
        await runner.runSetup(
            "ltx-director-demo",
            {"switch": 0, "output_switch": 1, "sequence_folder": "/shots/source/frames", "folder": "/shots/output"},
        )

        kept = set(comfy.submitted_prompts[0])
        assert "312" in kept, "QT writer (selected by output_switch) must be kept"
        assert "311" not in kept and "313" not in kept, "non-selected writers must be pruned"

    asyncio.run(exercise())


def test_run_setup_rejects_hidden_output_switch_option() -> None:
    """An output type marked visible: False (no wired writer branch) is not
    selectable through the API either — a direct caller submitting it gets a
    400 listing only the visible choices, and nothing is queued."""
    async def exercise() -> None:
        setup = _divergent_output_setup(
            output_default=1,
            same_as_input=False,
            output_options=[
                {"value": 0, "label": "EXR", "visible": False},
                {"value": 1, "label": "QT", "visible": True},
                {"value": 2, "label": "Img", "visible": False},
            ],
        )
        registry = PublishedSetupRegistry(StaticSetupStorage([setup]))
        comfy = FakeComfyClient()
        runner = PublishedSetupRunner(registry, comfy)

        with pytest.raises(SetupRunError) as exc_info:
            await runner.runSetup(
                "ltx-director-demo",
                {"switch": 0, "output_switch": 0, "sequence_folder": "/shots/source/frames"},
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.errors == [
            "input 'output_switch' must be one of: 1 (QT)"
        ]
        assert comfy.submitted_prompts == []

    asyncio.run(exercise())


def test_run_setup_omitted_output_switch_follows_input_when_default_is_same_as_input() -> None:
    """When the registry leaves outputSwitch.default at the -1 "Same as input"
    sentinel, a direct API caller that omits output_switch must still get a
    valid run: the runner resolves the sentinel to the input switch's value
    and keeps that writer branch (the browser simulator pre-resolves this;
    the server must handle it too)."""
    async def exercise() -> None:
        setup = _divergent_output_setup(output_default=-1, same_as_input=True)
        registry = PublishedSetupRegistry(StaticSetupStorage([setup]))
        comfy = FakeComfyClient()
        runner = PublishedSetupRunner(registry, comfy)

        # EXR in, no output_switch submitted — output follows input.
        await runner.runSetup(
            "ltx-director-demo",
            {"switch": 0, "sequence_folder": "/shots/source/frames", "folder": "/shots/output"},
        )

        kept = set(comfy.submitted_prompts[0])
        assert "311" in kept, "EXR writer (output follows input) must be kept"
        assert "312" not in kept and "313" not in kept, "non-selected writers must be pruned"

    asyncio.run(exercise())


@pytest.mark.parametrize("switch_value", [True, 99, 0, 1])
def test_run_setup_rejects_invalid_switch_values_before_queueing(switch_value: object) -> None:
    """Out-of-range values AND hidden options (0/1 are visible: False here) are
    rejected before queueing; the error offers only the selectable choices."""
    async def exercise() -> None:
        setup = _valid_setup()
        setup["inputContract"] = {"inputs": []}
        setup["outputContract"] = {"outputs": []}
        setup["apiPrompt"] = {
            "100": {
                "class_type": "Koolook_PublishInput",
                "inputs": {
                    "mode": "Img",
                    "sequence_folder": "",
                    "qt_file": "",
                    "single_file": "/shots/source/input.png",
                    "prompt": "",
                },
            },
            "300": {"class_type": "RMBG", "inputs": {"image": ["100", 2]}},
            "400": {"class_type": "Koolook_PublishRouter", "inputs": {"selector": ["100", 4], "payload": ["300", 0]}},
            "313": {"class_type": "SaveImageAndPromptExact", "inputs": {"image": ["400", 2]}},
        }
        setup["setupSurface"] = {
            "sourceInputs": [],
            "outputs": [],
            "controls": [],
            "app": {
                "inputs": [],
                "outputs": [],
                "results": [],
                "switch": {
                    "key": "switch",
                    "label": "Input type",
                    "visible": True,
                    "target": {"node": "100", "input": "mode"},
                    "default": 2,
                    "options": [
                        {"value": 0, "label": "EXR", "visible": False, "input": "single_file"},
                        {"value": 1, "label": "QT", "visible": False, "input": "single_file"},
                        {"value": 2, "label": "Img", "visible": True, "input": "single_file"},
                    ],
                },
            },
        }
        setup["executionMap"] = {
            "version": 1,
            "routers": [
                {
                    "node": "400",
                    "switchKey": "switch",
                    "selector": {"node": "100", "output": 4},
                    "payload": {"node": "300", "output": 0},
                    "branches": {
                        "0": {"label": "EXR", "output": 0, "writerNodes": []},
                        "1": {"label": "QT", "output": 1, "writerNodes": []},
                        "2": {"label": "Img", "output": 2, "writerNodes": ["313"]},
                    },
                }
            ],
        }
        comfy = FakeComfyClient()
        runner = PublishedSetupRunner(
            type("Registry", (), {"getSetup": lambda self, setup_id: setup if setup_id == "ltx-director-demo" else None})(),
            comfy,
        )

        with pytest.raises(SetupRunError) as exc_info:
            await runner.runSetup("ltx-director-demo", {"switch": switch_value})

        assert exc_info.value.status_code == 400
        assert exc_info.value.errors == [
            "input 'switch' must be one of: 2 (Img)"
        ]
        assert comfy.submitted_prompts == []

    asyncio.run(exercise())


def test_run_setup_rejects_execution_map_branch_with_stale_writer_nodes() -> None:
    async def exercise() -> None:
        setup = _valid_setup()
        setup["inputContract"] = {"inputs": []}
        setup["outputContract"] = {"outputs": []}
        setup["apiPrompt"] = {
            "100": {
                "class_type": "Koolook_PublishInput",
                "inputs": {
                    "mode": "Img",
                    "sequence_folder": "",
                    "qt_file": "",
                    "single_file": "/shots/source/input.png",
                    "prompt": "",
                },
            },
            "300": {"class_type": "RMBG", "inputs": {"image": ["100", 2]}},
            "400": {"class_type": "Koolook_PublishRouter", "inputs": {"selector": ["100", 4], "payload": ["300", 0]}},
            "313": {"class_type": "SaveImageAndPromptExact", "inputs": {"image": ["400", 2]}},
        }
        setup["setupSurface"] = {
            "sourceInputs": [],
            "outputs": [],
            "controls": [],
            "app": {
                "inputs": [],
                "outputs": [],
                "results": [],
                "switch": {
                    "key": "switch",
                    "label": "Input type",
                    "visible": True,
                    "target": {"node": "100", "input": "mode"},
                    "default": 2,
                    "options": [
                        {"value": 0, "label": "EXR", "visible": False, "input": "single_file"},
                        {"value": 1, "label": "QT", "visible": False, "input": "single_file"},
                        {"value": 2, "label": "Img", "visible": True, "input": "single_file"},
                    ],
                },
            },
        }
        setup["executionMap"] = {
            "version": 1,
            "routers": [
                {
                    "node": "400",
                    "switchKey": "switch",
                    "selector": {"node": "100", "output": 4},
                    "payload": {"node": "300", "output": 0},
                    "branches": {
                        "2": {"label": "Img", "output": 2, "writerNodes": ["999"]},
                    },
                }
            ],
        }
        comfy = FakeComfyClient()
        runner = PublishedSetupRunner(
            type("Registry", (), {"getSetup": lambda self, setup_id: setup if setup_id == "ltx-director-demo" else None})(),
            comfy,
        )

        with pytest.raises(SetupRunError) as exc_info:
            await runner.runSetup("ltx-director-demo", {"switch": 2})

        assert exc_info.value.status_code == 400
        assert exc_info.value.errors == [
            "execution map branch 2 for switch 'switch' references writer node(s) not present in the prompt: 999"
        ]
        assert comfy.submitted_prompts == []

    asyncio.run(exercise())


def test_get_run_reports_execution_map_writer_filepath_when_saver_history_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    async def exercise() -> None:
        import k_ai_pipeline

        makedirs_calls = []
        monkeypatch.setattr(
            k_ai_pipeline.os,
            "makedirs",
            lambda *args, **kwargs: makedirs_calls.append((args, kwargs)),
        )
        setup = _valid_setup()
        output_folder = tmp_path / "missing-output"
        setup["inputContract"] = {"inputs": []}
        setup["outputContract"] = {"outputs": []}
        setup["apiPrompt"] = {
            "100": {
                "class_type": "Koolook_PublishInput",
                "inputs": {
                    "mode": "Img",
                    "sequence_folder": "",
                    "qt_file": "",
                    "single_file": "/shots/source/input.png",
                    "prompt": "",
                },
            },
            "200": {
                "class_type": "Koolook_PublishOutput",
                "inputs": {"folder": str(output_folder), "name": "publish-OUT", "version": "2"},
            },
            "300": {"class_type": "RMBG", "inputs": {"image": ["100", 2]}},
            "303": {
                "class_type": "EasyAIPipeline",
                "inputs": {
                    "shot_duration": 81,
                    "seed_value": 1,
                    "instruction": "",
                    "base_directory_path": ["200", 0],
                    "extension": ".png",
                    "shot_name": ["200", 1],
                    "ai_method": "",
                    "version": ["200", 2],
                    "disable_versioning": False,
                    "enable_overwrite": True,
                    "no_subfolders": True,
                },
            },
            "400": {"class_type": "Koolook_PublishRouter", "inputs": {"selector": ["100", 4], "payload": ["300", 0]}},
            "313": {"class_type": "SaveImageAndPromptExact", "inputs": {"filepath": ["303", 0], "image": ["400", 2]}},
            "999": {"class_type": "Koolook_PublishResult", "inputs": {"result": ""}},
        }
        setup["setupSurface"] = {
            "sourceInputs": [
                {
                    "group": "Koolook Input",
                    "nodes": [{"id": "100", "type": "Koolook_PublishInput", "title": "Koolook Publish Input"}],
                }
            ],
            "outputs": [
                {
                    "group": "Koolook Output",
                    "nodes": [
                        {"id": "200", "type": "Koolook_PublishOutput", "title": "Koolook Publish Output"},
                        {"id": "400", "type": "Koolook_PublishRouter", "title": "Koolook Publish Router"},
                        {"id": "313", "type": "SaveImageAndPromptExact", "title": "Image writer"},
                        {"id": "999", "type": "Koolook_PublishResult", "title": "Koolook Publish Result"},
                    ],
                }
            ],
            "controls": [],
            "app": {
                "inputs": [
                    {
                        "key": "single_file",
                        "label": "Single file",
                        "visible": True,
                        "target": {"node": "100", "input": "single_file"},
                        "default": "/shots/source/input.png",
                    }
                ],
                "outputs": [
                    {"key": "folder", "label": "Output folder", "visible": True, "target": {"node": "200", "input": "folder"}, "default": str(output_folder)},
                    {"key": "name", "label": "Output name", "visible": True, "target": {"node": "200", "input": "name"}, "default": "publish-OUT"},
                    {"key": "version", "label": "Version", "visible": True, "target": {"node": "200", "input": "version"}, "default": "2"},
                ],
                "results": [
                    {"key": "result", "label": "Result", "visible": True, "target": {"node": "999", "input": "result"}, "default": ""}
                ],
                "switch": {
                    "key": "switch",
                    "label": "Input type",
                    "visible": True,
                    "target": {"node": "100", "input": "mode"},
                    "default": 2,
                    "options": [
                        {"value": 0, "label": "EXR", "visible": False, "input": "single_file"},
                        {"value": 1, "label": "QT", "visible": False, "input": "single_file"},
                        {"value": 2, "label": "Img", "visible": True, "input": "single_file"},
                        {"value": 3, "label": "Prompt", "visible": False, "input": "single_file"},
                    ],
                },
            },
        }
        setup["executionMap"] = {
            "version": 1,
            "routers": [
                {
                    "node": "400",
                    "switchKey": "switch",
                    "selector": {"node": "100", "output": 4},
                    "payload": {"node": "300", "output": 0},
                    "branches": {
                        "2": {"label": "Img", "output": 2, "writerNodes": ["313"]},
                    },
                }
            ],
        }
        comfy = FakeComfyClient()
        runner = PublishedSetupRunner(
            type("Registry", (), {"getSetup": lambda self, setup_id: setup if setup_id == "ltx-director-demo" else None})(),
            comfy,
        )

        queued = await runner.runSetup(
            "ltx-director-demo",
            {"switch": 2, "folder": str(output_folder), "name": "external", "version": "7"},
        )
        comfy.history[queued["promptId"]] = {
            "status": {"completed": True, "status_str": "success"},
            "prompt": [2, queued["promptId"], comfy.submitted_prompts[0], {}],
            "outputs": {"313": {"images": []}},
        }

        status = await runner.getRun(queued["runId"])

        # build_pipeline_outputs canonicalizes paths to forward slashes on
        # every platform, so the expected prefix must match — str(tmp_path)
        # yields backslashes on Windows.
        canonical_folder = str(output_folder).replace("\\", "/")
        assert status["outputs"][-1]["items"] == [
            {
                "nodeId": "313",
                "kind": "text",
                "value": f"{canonical_folder}/v007/external_v007.png",
            }
        ]
        assert makedirs_calls == []
        assert not output_folder.exists()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("switch_value", "mode_label", "branch_node", "source_node", "saver_node", "result_path"),
    [
        (0, "EXR", "301", "401", "311", "/shots/output/publish-OUT_v002.exr"),
        (1, "QT", "302", "402", "312", "/shots/output/publish-OUT_v002.mov"),
        (2, "Img", "303", "403", "313", "/shots/output/publish-OUT_v002.png"),
    ],
)
def test_get_run_resolves_selected_switch_result_when_result_node_history_is_missing(
    switch_value: int,
    mode_label: str,
    branch_node: str,
    source_node: str,
    saver_node: str,
    result_path: str,
) -> None:
    async def exercise() -> None:
        setup = _valid_setup()
        setup["inputContract"] = {"inputs": []}
        setup["outputContract"] = {"outputs": []}
        setup["visualGraph"] = {
            "nodes": [
                {
                    "id": 188,
                    "type": "Koolook_PublishInput",
                    "inputs": [
                        {"name": "mode", "widget": {"name": "mode"}},
                        {"name": "sequence_folder", "widget": {"name": "sequence_folder"}},
                        {"name": "qt_file", "widget": {"name": "qt_file"}},
                        {"name": "single_file", "widget": {"name": "single_file"}},
                    ],
                },
                {
                    "id": 205,
                    "type": "Koolook_PublishResult",
                    "inputs": [{"name": "result", "link": 900}],
                },
            ],
            "links": [[900, 179, 0, 205, 0, "STRING"]],
            "groups": [],
        }
        setup["apiPrompt"] = {
            "188": {
                "class_type": "Koolook_PublishInput",
                "inputs": {
                    "mode": "Img",
                    "sequence_folder": "",
                    "qt_file": "",
                    "single_file": "/shots/source/input.png",
                    "prompt": "",
                },
            },
            "179": {
                "class_type": "Switch any [Crystools]",
                "inputs": {
                    "index": ["188", 4],
                    "value0": ["301", 0],
                    "value1": ["302", 0],
                    "value2": ["303", 0],
                },
            },
            "180": {
                "class_type": "Switch any [Crystools]",
                "inputs": {
                    "index": ["188", 4],
                    "value0": ["401", 0],
                    "value1": ["402", 0],
                    "value2": ["403", 0],
                },
            },
            "301": {
                "class_type": "EasyAIPipeline",
                "inputs": {"WRITE_file_path": "/shots/output/publish-OUT_v002.exr"},
            },
            "302": {
                "class_type": "Easy_VideoCombine",
                "inputs": {"video_path": "/shots/output/publish-OUT_v002.mov"},
            },
            "303": {
                "class_type": "EasyAIPipeline",
                "inputs": {"WRITE_file_path": "/shots/output/publish-OUT_v002.png"},
            },
            "311": {
                "class_type": "SaveEXRFrames",
                "inputs": {"filepath": ["301", 0], "images": ["180", 0]},
            },
            "312": {
                "class_type": "Easy_VideoCombine",
                "inputs": {"filepath": ["302", 0], "images": ["180", 0]},
            },
            "313": {
                "class_type": "SaveImageAndPromptExact",
                "inputs": {"filepath": ["303", 0], "images": ["180", 0]},
            },
            "401": {
                "class_type": "LoadEXR",
                "inputs": {"path": "/shots/source/input.exr"},
            },
            "402": {
                "class_type": "VHS_LoadVideoPath",
                "inputs": {"video": "/shots/source/input.mov"},
            },
            "403": {
                "class_type": "VHS_LoadImagePath",
                "inputs": {"image": "/shots/source/input.png"},
            },
            "205": {
                "class_type": "Koolook_PublishResult",
                "inputs": {"result": ["179", 0]},
            },
            "900": {
                "class_type": "Preview Image",
                "inputs": {"images": ["301", 0]},
            },
        }
        setup["setupSurface"] = {
            "sourceInputs": [
                {
                    "group": "Koolook Input",
                    "nodes": [{"id": "188", "type": "Koolook_PublishInput", "title": "Koolook Publish Input"}],
                }
            ],
            "outputs": [
                {
                    "group": "Koolook Output",
                    "nodes": [
                        {"id": "205", "type": "Koolook_PublishResult", "title": "Koolook Publish Result"},
                        {"id": "301", "type": "EasyAIPipeline", "title": "EXR path"},
                        {"id": "302", "type": "EasyAIPipeline", "title": "QT path"},
                        {"id": "303", "type": "EasyAIPipeline", "title": "Img path"},
                        {"id": "311", "type": "SaveEXRFrames", "title": "EXR saver"},
                        {"id": "312", "type": "Easy_VideoCombine", "title": "QT saver"},
                        {"id": "313", "type": "SaveImageAndPromptExact", "title": "Image saver"},
                    ],
                }
            ],
            "controls": [],
            "app": {
                "inputs": [
                    {
                        "key": "sequence_folder",
                        "label": "Sequence folder",
                        "visible": True,
                        "target": {"node": "188", "input": "sequence_folder"},
                        "default": "",
                    },
                    {
                        "key": "qt_file",
                        "label": "QT file",
                        "visible": True,
                        "target": {"node": "188", "input": "qt_file"},
                        "default": "",
                    },
                    {
                        "key": "single_file",
                        "label": "Single file",
                        "visible": True,
                        "target": {"node": "188", "input": "single_file"},
                        "default": "/shots/source/input.png",
                    }
                ],
                "outputs": [],
                "results": [
                    {
                        "key": "result",
                        "label": "Result",
                        "visible": True,
                        "target": {"node": "205", "input": "result"},
                        "default": "",
                    }
                ],
                "switch": {
                    "key": "switch",
                    "label": "Input type",
                    "visible": True,
                    "target": {"node": "188", "input": "mode"},
                    "default": 2,
                    "options": [
                        {"value": 0, "label": "EXR", "visible": True, "input": "sequence_folder"},
                        {"value": 1, "label": "QT", "visible": True, "input": "qt_file"},
                        {"value": 2, "label": "Img", "visible": True, "input": "single_file"},
                    ],
                },
            },
        }
        registry = PublishedSetupRegistry(StaticSetupStorage([setup]))
        comfy = FakeComfyClient(
            history={
                "comfy-prompt-1": {
                    "status": {"completed": True, "status_str": "success"},
                    "outputs": {
                        branch_node: {
                            "text": [result_path],
                        }
                    },
                }
            }
        )
        runner = PublishedSetupRunner(
            registry,
            comfy,
        )

        queued = await runner.runSetup("ltx-director-demo", {"switch": switch_value})
        status = await runner.getRun(queued["runId"])

        assert set(comfy.submitted_prompts[0]) == {
            "188",
            "179",
            "180",
            "205",
            branch_node,
            source_node,
            saver_node,
        }
        assert comfy.submitted_prompts[0]["188"]["inputs"]["mode"] == mode_label
        assert comfy.submitted_prompts[0]["179"]["inputs"] == {
            "index": ["188", 4],
            f"value{switch_value}": [branch_node, 0],
        }
        assert comfy.submitted_prompts[0]["180"]["inputs"] == {
            "index": ["188", 4],
            f"value{switch_value}": [source_node, 0],
        }
        assert status["outputs"] == [
            {
                "key": "result",
                "label": "Result",
                "type": "result",
                "visible": True,
                "target": {"node": "205", "input": "result"},
                "default": "",
                "items": [
                    {
                        "nodeId": branch_node,
                        "kind": "text",
                        "value": result_path,
                    }
                ],
            }
        ]

    asyncio.run(exercise())


def _app_builder_param_setup() -> dict:
    """A callable setup whose app surface declares one App-builder param pick."""
    setup = _valid_setup()
    setup["id"] = "app-builder-runner"
    setup["visualGraph"]["nodes"].append(
        {"id": 30, "type": "KSampler", "title": "Main sampler", "inputs": []}
    )
    setup["apiPrompt"]["30"] = {"class_type": "KSampler", "inputs": {"steps": 8}}
    setup["setupSurface"] = {
        "sourceInputs": [],
        "outputs": [],
        "controls": [],
        "app": {
            "inputs": [
                {
                    "key": "param_30_steps",
                    "label": "Main sampler: steps",
                    "visible": True,
                    "standalone": True,
                    "appParam": True,
                    "valueType": "int",
                    "target": {"node": "30", "input": "steps"},
                    "default": 8,
                }
            ],
            "outputs": [],
            "results": [],
        },
    }
    return setup


def test_run_setup_injects_app_builder_param_into_cloned_prompt() -> None:
    async def exercise() -> None:
        registry = PublishedSetupRegistry(StaticSetupStorage([_app_builder_param_setup()]))
        comfy = FakeComfyClient()
        runner = PublishedSetupRunner(registry, comfy)

        await runner.runSetup(
            "app-builder-runner",
            {"prompt": "external prompt", "param_30_steps": 12},
        )

        assert comfy.submitted_prompts[0]["30"]["inputs"]["steps"] == 12
        assert comfy.submitted_prompts[0]["12"]["inputs"]["text"] == "external prompt"

    asyncio.run(exercise())


def test_run_setup_keeps_param_default_when_not_submitted() -> None:
    async def exercise() -> None:
        registry = PublishedSetupRegistry(StaticSetupStorage([_app_builder_param_setup()]))
        comfy = FakeComfyClient()
        runner = PublishedSetupRunner(registry, comfy)

        await runner.runSetup("app-builder-runner", {"prompt": "external prompt"})

        assert comfy.submitted_prompts[0]["30"]["inputs"]["steps"] == 8

    asyncio.run(exercise())
