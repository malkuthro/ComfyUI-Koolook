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


def _json_body(response: web.Response):
    return json.loads(response.text or "")
