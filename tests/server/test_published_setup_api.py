from __future__ import annotations

import asyncio
import json
from copy import deepcopy

from aiohttp import web
from aiohttp.test_utils import make_mocked_request

import koolook_routes
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


def _app_with_registry(registry: PublishedSetupRegistry) -> web.Application:
    routes = web.RouteTableDef()
    koolook_routes.register_routes(routes, setup_registry_factory=lambda: registry)
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
