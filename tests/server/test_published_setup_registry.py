from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json

import pytest

from koolook_setups import (
    FileSetupStorage,
    PublishedSetupRegistry,
    StaticSetupStorage,
    validate_setup,
)


def _valid_setup() -> dict:
    return {
        "schemaVersion": 1,
        "id": "ltx-director-demo",
        "version": 1,
        "updatedAt": "2026-06-06T08:00:00Z",
        "metadata": {
            "title": "LTX Director Demo",
            "description": "A tiny published setup fixture.",
            "category": "Video",
            "tags": ["demo", "ltx"],
            "previewImage": "koolook://preview/ltx-director-demo.png",
        },
        "visualGraph": {"nodes": [{"id": 1, "type": "Text Multiline"}], "links": []},
        "apiPrompt": None,
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
        "outputContract": {
            "outputs": [
                {
                    "key": "video",
                    "label": "Video",
                    "type": "video",
                }
            ]
        },
        "source": {
            "kind": "sidebar-workflow",
            "path": "Demos/LTX Director Demo",
        },
        "validation": {
            "status": "valid",
            "diagnostics": [],
        },
    }


def test_validate_setup_accepts_complete_published_setup() -> None:
    result = validate_setup(_valid_setup())

    assert result.valid is True
    assert result.setup["id"] == "ltx-director-demo"
    assert result.diagnostics == []


@pytest.mark.parametrize("missing_key", ["id", "metadata", "visualGraph", "inputContract"])
def test_validate_setup_rejects_missing_required_fields(missing_key: str) -> None:
    setup = deepcopy(_valid_setup())
    setup.pop(missing_key)

    result = validate_setup(setup)

    assert result.valid is False
    assert any(missing_key in diagnostic for diagnostic in result.diagnostics)


@pytest.mark.parametrize(
    ("mutator", "expected"),
    [
        (lambda setup: setup["metadata"].pop("title"), "metadata.title"),
        (lambda setup: setup["inputContract"]["inputs"][0].pop("target"), "inputContract.inputs[0].target"),
        (lambda setup: setup["validation"].__setitem__("status", "unknown"), "validation.status"),
        (lambda setup: setup.__setitem__("updatedAt", "06/06/2026"), "updatedAt"),
        (lambda setup: setup.__setitem__("schemaVersion", 2), "unsupported schemaVersion: 2"),
    ],
)
def test_validate_setup_rejects_malformed_schema_fields(mutator, expected: str) -> None:
    setup = deepcopy(_valid_setup())
    mutator(setup)

    result = validate_setup(setup)

    assert result.valid is False
    assert any(expected in diagnostic for diagnostic in result.diagnostics)


def test_registry_lists_summaries_and_fetches_full_setup_without_invalid_records() -> None:
    invalid = deepcopy(_valid_setup())
    invalid["id"] = "broken"
    invalid.pop("metadata")
    registry = PublishedSetupRegistry(StaticSetupStorage([invalid, _valid_setup()]))

    rows = registry.listSetups()
    detail = registry.getSetup("ltx-director-demo")

    assert [row["id"] for row in rows] == ["ltx-director-demo"]
    assert rows[0] == {
        "id": "ltx-director-demo",
        "version": 1,
        "updatedAt": "2026-06-06T08:00:00Z",
        "metadata": {
            "title": "LTX Director Demo",
            "description": "A tiny published setup fixture.",
            "category": "Video",
            "tags": ["demo", "ltx"],
            "previewImage": "koolook://preview/ltx-director-demo.png",
        },
        "validation": {"status": "valid", "diagnostics": []},
        "inputSummary": [{"key": "prompt", "label": "Prompt", "type": "text", "required": True}],
        "outputSummary": [{"key": "video", "label": "Video", "type": "video"}],
    }
    assert detail["visualGraph"] == _valid_setup()["visualGraph"]
    assert detail["apiPrompt"] is None
    assert registry.getSetup("broken") is None
    assert registry.diagnostics == ["broken: missing required field: metadata"]


def test_file_storage_loads_setups_object_and_uses_sample_fallback(tmp_path) -> None:
    primary = tmp_path / "setups.json"
    fallback = tmp_path / "sample.json"
    fallback.write_text('{"setups": [%s]}' % _json(_valid_setup()), encoding="utf-8")

    storage = FileSetupStorage(primary, fallback_path=fallback)
    assert storage.load_setups()[0]["id"] == "ltx-director-demo"

    primary.write_text('{"setups": []}', encoding="utf-8")
    assert storage.load_setups() == []


def test_file_storage_reports_corrupt_primary_without_falling_back(tmp_path) -> None:
    primary = tmp_path / "setups.json"
    fallback = tmp_path / "sample.json"
    primary.write_text("{not json", encoding="utf-8")
    fallback.write_text('{"setups": [%s]}' % _json(_valid_setup()), encoding="utf-8")
    storage = FileSetupStorage(primary, fallback_path=fallback)

    registry = PublishedSetupRegistry(storage)

    assert registry.listSetups() == []
    assert len(registry.diagnostics) == 1
    assert str(primary) in registry.diagnostics[0]
    assert "could not read published setups" in registry.diagnostics[0]


def test_publish_setup_does_not_copy_sample_fallback_into_primary(tmp_path) -> None:
    primary = tmp_path / "user" / "setups.json"
    fallback = tmp_path / "sample.json"
    fallback.write_text(json.dumps({"setups": [_valid_setup()]}), encoding="utf-8")

    registry = PublishedSetupRegistry(FileSetupStorage(primary, fallback_path=fallback))
    result = registry.publishSetup(
        visualGraph={"nodes": [{"id": 12, "inputs": [{"name": "text"}]}]},
        metadata={
            "id": "first-curated-setup",
            "title": "First Curated Setup",
            "description": "The first real user-published setup.",
        },
        inputContract={
            "inputs": [
                {
                    "key": "prompt",
                    "type": "text",
                    "target": {"node": "12", "input": "text"},
                }
            ]
        },
        outputContract={"outputs": [{"key": "preview", "type": "image"}]},
        source={"kind": "sidebar-workflow", "path": "Demos/First"},
    )

    assert result.valid is True
    stored = json.loads(primary.read_text(encoding="utf-8"))
    assert [setup["id"] for setup in stored["setups"]] == ["first-curated-setup"]


def test_publish_setup_persists_draft_and_catalog_reflects_it() -> None:
    storage = StaticSetupStorage([])
    registry = PublishedSetupRegistry(storage)
    visual_graph = {
        "nodes": [
            {
                "id": 12,
                "type": "Text Multiline",
                "inputs": [{"name": "text", "type": "STRING"}],
            }
        ],
        "links": [],
    }

    result = registry.publishSetup(
        visualGraph=visual_graph,
        metadata={
            "id": "my-callable-flow",
            "title": "My Callable Flow",
            "description": "External users can run this curated setup.",
            "category": "Video",
            "tags": ["demo", "publish"],
            "previewImage": "koolook://preview/card.png",
        },
        inputContract={
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
        outputContract={"outputs": [{"key": "preview", "label": "Preview", "type": "image"}]},
        source={"kind": "sidebar-workflow", "path": "Demos/My Callable Flow"},
    )

    assert result.valid is True
    setup = registry.getSetup("my-callable-flow")
    assert setup is not None
    assert setup["visualGraph"] == visual_graph
    assert setup["apiPrompt"] is None
    assert setup["source"] == {"kind": "sidebar-workflow", "path": "Demos/My Callable Flow"}
    assert setup["updatedAt"].endswith("Z")
    assert "+00:00" in datetime.fromisoformat(setup["updatedAt"].replace("Z", "+00:00")).isoformat()
    assert setup["validation"] == {
        "status": "draft",
        "diagnostics": ["API prompt conversion pending."],
    }
    assert registry.listSetups()[0]["id"] == "my-callable-flow"


def test_publish_setup_rejects_missing_contract_target() -> None:
    registry = PublishedSetupRegistry(StaticSetupStorage([]))

    result = registry.publishSetup(
        visualGraph={"nodes": [{"id": 12, "inputs": [{"name": "text"}]}]},
        metadata={"id": "bad-flow", "title": "Bad", "description": "Bad target"},
        inputContract={
            "inputs": [
                {
                    "key": "prompt",
                    "type": "text",
                    "target": {"node": "99", "input": "text"},
                }
            ]
        },
        outputContract={"outputs": [{"key": "preview", "type": "image"}]},
        source={"kind": "sidebar-workflow", "path": "Demos/Bad"},
    )

    assert result.valid is False
    assert "inputContract.inputs[0].target.node not found in visualGraph" in result.diagnostics
    assert registry.listSetups() == []


def _json(value: dict) -> str:
    import json

    return json.dumps(value)
