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
                                "strings": [
                                    {"value": "/shots/example/output/generated.mov"}
                                ]
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
                        "kind": "strings",
                        "value": "/shots/example/output/generated.mov",
                    }
                ],
            },
        ]

    asyncio.run(exercise())
