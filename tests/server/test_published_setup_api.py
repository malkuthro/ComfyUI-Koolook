from __future__ import annotations

import asyncio
import json
from copy import deepcopy

from aiohttp import web
from aiohttp.test_utils import make_mocked_request

import koolook_routes
from koolook_setup_runner import PublishedSetupRunner
from koolook_setups import PublishedSetupRegistry, StaticSetupStorage
from tests.server.test_published_setup_registry import _valid_setup


def test_catalog_routes_expose_list_and_detail_contracts() -> None:
    async def exercise() -> None:
        registry = PublishedSetupRegistry(StaticSetupStorage([_valid_setup()]))
        app = _app_with_registry(registry)

        list_response = await _handle(app, "GET", "/koolook/api/setups")
        detail_response = await _handle(app, "GET", "/koolook/api/setups/ltx-director-demo")
        missing_response = await _handle(app, "GET", "/koolook/api/setups/missing")

        assert list_response.status == 200
        assert detail_response.status == 200
        assert missing_response.status == 404

        rows = _json_body(list_response)
        detail = _json_body(detail_response)
        assert rows[0]["id"] == "ltx-director-demo"
        assert "visualGraph" not in rows[0]
        assert "apiPrompt" not in rows[0]
        assert detail["visualGraph"] == _valid_setup()["visualGraph"]
        assert detail["inputContract"] == _valid_setup()["inputContract"]

    asyncio.run(exercise())


def test_catalog_list_omits_invalid_setups() -> None:
    async def exercise() -> None:
        invalid = deepcopy(_valid_setup())
        invalid["id"] = "broken"
        invalid.pop("metadata")
        registry = PublishedSetupRegistry(StaticSetupStorage([invalid, _valid_setup()]))
        app = _app_with_registry(registry)

        response = await _handle(app, "GET", "/koolook/api/setups")
        rows = _json_body(response)

        assert response.status == 200
        assert [row["id"] for row in rows] == ["ltx-director-demo"]
        assert registry.diagnostics == ["broken: missing required field: metadata"]

    asyncio.run(exercise())


def test_publish_route_persists_setup_and_catalog_returns_it() -> None:
    async def exercise() -> None:
        registry = PublishedSetupRegistry(StaticSetupStorage([]))
        app = _app_with_registry(registry)
        payload = {
            "visualGraph": {
                "nodes": [
                    {
                        "id": 12,
                        "type": "Text Multiline",
                        "inputs": [{"name": "text", "widget": {"name": "text"}}],
                        "widgets_values": ["published prompt"],
                    }
                ],
                "links": [],
            },
            "metadata": {
                "id": "published-from-sidebar",
                "title": "Published From Sidebar",
                "description": "Published through the API.",
                "category": "Video",
                "tags": ["publish"],
                "previewImage": "",
            },
            "inputContract": {
                "inputs": [
                    {
                        "key": "prompt",
                        "label": "Prompt",
                        "type": "text",
                        "required": True,
                        "target": {"node": "12", "input": "text"},
                    }
                ]
            },
            "outputContract": {"outputs": [{"key": "preview", "type": "image"}]},
            "source": {"kind": "sidebar-workflow", "path": "Demos/Published From Sidebar"},
        }

        publish_response = await _handle_json(app, "POST", "/koolook/api/setups", payload)
        detail_response = await _handle(app, "GET", "/koolook/api/setups/published-from-sidebar")

        assert publish_response.status == 200
        body = _json_body(publish_response)
        assert body["ok"] is True
        assert body["setup"]["id"] == "published-from-sidebar"
        detail = _json_body(detail_response)
        assert detail["source"]["path"] == "Demos/Published From Sidebar"
        assert detail["apiPrompt"] == {
            "12": {"class_type": "Text Multiline", "inputs": {"text": "published prompt"}}
        }
        assert detail["validation"] == {"status": "valid", "diagnostics": []}

    asyncio.run(exercise())


def test_publish_route_rejects_invalid_contract_with_clear_error() -> None:
    async def exercise() -> None:
        registry = PublishedSetupRegistry(StaticSetupStorage([]))
        app = _app_with_registry(registry)

        response = await _handle_json(
            app,
            "POST",
            "/koolook/api/setups",
            {
                "visualGraph": {
                    "nodes": [
                        {
                            "id": 12,
                            "type": "Text Multiline",
                            "inputs": [{"name": "text", "widget": {"name": "text"}}],
                            "widgets_values": ["bad prompt"],
                        }
                    ],
                    "links": [],
                },
                "metadata": {"id": "bad", "title": "Bad", "description": "Bad"},
                "inputContract": {
                    "inputs": [{"key": "prompt", "type": "text", "target": {"node": "99", "input": "text"}}]
                },
                "outputContract": {"outputs": [{"key": "preview", "type": "image"}]},
                "source": {"kind": "sidebar-workflow", "path": "Demos/Bad"},
            },
        )

        assert response.status == 400
        assert "target.node not found" in _json_body(response)["errors"][0]
        assert registry.listSetups() == []

    asyncio.run(exercise())


def test_publish_route_rejects_missing_metadata_with_clear_error() -> None:
    async def exercise() -> None:
        registry = PublishedSetupRegistry(StaticSetupStorage([]))
        app = _app_with_registry(registry)

        response = await _handle_json(
            app,
            "POST",
            "/koolook/api/setups",
            {
                "visualGraph": {"nodes": []},
                "inputContract": {"inputs": []},
                "outputContract": {"outputs": []},
                "source": {"kind": "sidebar-workflow", "path": "Demos/Missing Metadata"},
            },
        )

        assert response.status == 400
        assert "metadata must be an object" in _json_body(response)["errors"][0]
        assert registry.listSetups() == []

    asyncio.run(exercise())


def test_run_route_queues_callable_setup_and_returns_stable_run_id() -> None:
    async def exercise() -> None:
        registry = PublishedSetupRegistry(StaticSetupStorage([_valid_setup()]))
        comfy = _FakeComfyClient()
        app = _app_with_registry(
            registry,
            setup_runner_factory=lambda: PublishedSetupRunner(registry, comfy),
        )

        response = await _handle_json(
            app,
            "POST",
            "/koolook/api/setups/ltx-director-demo/run",
            {"inputs": {"prompt": "external prompt"}},
        )

        assert response.status == 200
        assert _json_body(response) == {
            "ok": True,
            "run": {
                "runId": "run-000001",
                "promptId": "comfy-prompt-1",
                "status": "queued",
            },
        }
        assert comfy.submitted_prompts == [
            {"12": {"class_type": "Text Multiline", "inputs": {"text": "external prompt"}}}
        ]

    asyncio.run(exercise())


def test_run_status_route_reports_succeeded_outputs_from_comfy_history() -> None:
    async def exercise() -> None:
        registry = PublishedSetupRegistry(StaticSetupStorage([_valid_setup()]))
        comfy = _FakeComfyClient(
            history={
                "comfy-prompt-1": {
                    "status": {"completed": True, "status_str": "success"},
                    "outputs": {
                        "20": {
                            "videos": [
                                {
                                    "filename": "demo.mp4",
                                    "subfolder": "koolook",
                                    "type": "output",
                                }
                            ]
                        }
                    },
                }
            }
        )
        runner = PublishedSetupRunner(registry, comfy)
        app = _app_with_registry(registry, setup_runner_factory=lambda: runner)

        run_response = await _handle_json(
            app,
            "POST",
            "/koolook/api/setups/ltx-director-demo/run",
            {"inputs": {"prompt": "external prompt"}},
        )
        run_id = _json_body(run_response)["run"]["runId"]
        status_response = await _handle(app, "GET", f"/koolook/api/runs/{run_id}")

        assert status_response.status == 200
        assert _json_body(status_response) == {
            "ok": True,
            "run": {
                "runId": "run-000001",
                "setupId": "ltx-director-demo",
                "promptId": "comfy-prompt-1",
                "status": "succeeded",
                "outputs": [
                    {
                        "key": "video",
                        "label": "Video",
                        "type": "video",
                        "items": [
                            {
                                "nodeId": "20",
                                "kind": "videos",
                                "filename": "demo.mp4",
                                "subfolder": "koolook",
                                "type": "output",
                            }
                        ],
                    }
                ],
            },
        }

    asyncio.run(exercise())


def test_run_status_route_reports_failed_state_from_comfy_history() -> None:
    async def exercise() -> None:
        registry = PublishedSetupRegistry(StaticSetupStorage([_valid_setup()]))
        comfy = _FakeComfyClient(
            history={
                "comfy-prompt-1": {
                    "status": {"completed": True, "status_str": "error"},
                    "outputs": {},
                }
            }
        )
        runner = PublishedSetupRunner(registry, comfy)
        app = _app_with_registry(registry, setup_runner_factory=lambda: runner)

        run_response = await _handle_json(
            app,
            "POST",
            "/koolook/api/setups/ltx-director-demo/run",
            {"inputs": {"prompt": "external prompt"}},
        )
        run_id = _json_body(run_response)["run"]["runId"]
        status_response = await _handle(app, "GET", f"/koolook/api/runs/{run_id}")

        assert status_response.status == 200
        assert _json_body(status_response)["run"]["status"] == "failed"

    asyncio.run(exercise())


def test_run_route_returns_clear_errors_for_invalid_inputs() -> None:
    async def exercise() -> None:
        registry = PublishedSetupRegistry(StaticSetupStorage([_valid_setup()]))
        app = _app_with_registry(
            registry,
            setup_runner_factory=lambda: PublishedSetupRunner(registry, _FakeComfyClient()),
        )

        response = await _handle_json(
            app,
            "POST",
            "/koolook/api/setups/ltx-director-demo/run",
            {"inputs": {"prompt": "ok", "seed": 123}},
        )

        assert response.status == 400
        assert _json_body(response) == {
            "ok": False,
            "errors": ["input 'seed' is not declared by this setup"],
        }

    asyncio.run(exercise())


def test_run_route_returns_clear_error_for_non_callable_setup() -> None:
    async def exercise() -> None:
        setup = _valid_setup()
        setup["apiPrompt"] = None
        setup["validation"] = {"status": "draft", "diagnostics": ["API prompt conversion pending."]}
        registry = PublishedSetupRegistry(StaticSetupStorage([setup]))
        app = _app_with_registry(
            registry,
            setup_runner_factory=lambda: PublishedSetupRunner(registry, _FakeComfyClient()),
        )

        response = await _handle_json(
            app,
            "POST",
            "/koolook/api/setups/ltx-director-demo/run",
            {"inputs": {"prompt": "ok"}},
        )

        assert response.status == 400
        assert _json_body(response) == {
            "ok": False,
            "errors": ["Published setup 'ltx-director-demo' is not callable."],
        }

    asyncio.run(exercise())


def test_run_route_returns_clear_error_for_comfy_queue_failure() -> None:
    async def exercise() -> None:
        registry = PublishedSetupRegistry(StaticSetupStorage([_valid_setup()]))
        app = _app_with_registry(
            registry,
            setup_runner_factory=lambda: PublishedSetupRunner(
                registry,
                _FakeComfyClient(queue_error=RuntimeError("server down")),
            ),
        )

        response = await _handle_json(
            app,
            "POST",
            "/koolook/api/setups/ltx-director-demo/run",
            {"inputs": {"prompt": "ok"}},
        )

        assert response.status == 502
        assert _json_body(response) == {
            "ok": False,
            "errors": ["ComfyUI queue request failed: server down"],
        }

    asyncio.run(exercise())


def test_run_route_returns_clear_error_for_missing_setup() -> None:
    async def exercise() -> None:
        registry = PublishedSetupRegistry(StaticSetupStorage([_valid_setup()]))
        app = _app_with_registry(
            registry,
            setup_runner_factory=lambda: PublishedSetupRunner(registry, _FakeComfyClient()),
        )

        response = await _handle_json(
            app,
            "POST",
            "/koolook/api/setups/missing/run",
            {"inputs": {"prompt": "ok"}},
        )

        assert response.status == 404
        assert _json_body(response) == {
            "ok": False,
            "errors": ["Published setup 'missing' not found."],
        }

    asyncio.run(exercise())


def test_run_status_route_returns_clear_error_for_unknown_run_id() -> None:
    async def exercise() -> None:
        registry = PublishedSetupRegistry(StaticSetupStorage([_valid_setup()]))
        app = _app_with_registry(
            registry,
            setup_runner_factory=lambda: PublishedSetupRunner(registry, _FakeComfyClient()),
        )

        response = await _handle(app, "GET", "/koolook/api/runs/run-missing")

        assert response.status == 404
        assert _json_body(response) == {
            "ok": False,
            "errors": ["Koolook run 'run-missing' not found."],
        }

    asyncio.run(exercise())


class _FakeComfyClient:
    def __init__(self, *, history=None, queue=None, queue_error=None) -> None:
        self.submitted_prompts: list[dict] = []
        self.history = history or {}
        self.queue = queue or {"queue_running": [], "queue_pending": []}
        self.queue_error = queue_error

    async def queue_prompt(self, prompt: dict) -> dict:
        if self.queue_error is not None:
            raise self.queue_error
        self.submitted_prompts.append(prompt)
        return {"prompt_id": "comfy-prompt-1"}

    async def get_history(self, prompt_id: str) -> dict:
        return {prompt_id: self.history[prompt_id]} if prompt_id in self.history else {}

    async def get_queue(self) -> dict:
        return self.queue


def _app_with_registry(
    registry: PublishedSetupRegistry,
    *,
    setup_runner_factory=None,
) -> web.Application:
    routes = web.RouteTableDef()
    koolook_routes.register_routes(
        routes,
        setup_registry_factory=lambda: registry,
        setup_runner_factory=setup_runner_factory,
    )
    app = web.Application()
    app.add_routes(routes)
    return app


async def _handle(app: web.Application, method: str, path: str) -> web.Response:
    request = make_mocked_request(method, path, app=app)
    try:
        return await app._handle(request)
    except web.HTTPException as exc:
        return exc


async def _handle_json(app: web.Application, method: str, path: str, payload) -> web.Response:
    request = make_mocked_request(
        method,
        path,
        app=app,
        headers={"Content-Type": "application/json"},
    )
    request._read_bytes = json.dumps(payload).encode("utf-8")
    try:
        return await app._handle(request)
    except web.HTTPException as exc:
        return exc


def _json_body(response: web.Response):
    return json.loads(response.text or "")
