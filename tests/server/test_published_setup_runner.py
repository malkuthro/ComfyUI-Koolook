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


def test_get_run_reports_execution_map_writer_filepath_when_saver_history_is_empty() -> None:
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
            "200": {
                "class_type": "Koolook_PublishOutput",
                "inputs": {"folder": "/shots/output", "name": "publish-OUT", "version": "2"},
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
                    {"key": "folder", "label": "Output folder", "visible": True, "target": {"node": "200", "input": "folder"}, "default": "/shots/output"},
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
            {"switch": 2, "folder": "/shots/output", "name": "external", "version": "7"},
        )
        comfy.history[queued["promptId"]] = {
            "status": {"completed": True, "status_str": "success"},
            "prompt": [2, queued["promptId"], comfy.submitted_prompts[0], {}],
            "outputs": {"313": {"images": []}},
        }

        status = await runner.getRun(queued["runId"])

        assert status["outputs"][-1]["items"] == [
            {
                "nodeId": "313",
                "kind": "text",
                "value": "/shots/output/v007/external_v007.png",
            }
        ]

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
