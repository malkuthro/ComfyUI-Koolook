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
