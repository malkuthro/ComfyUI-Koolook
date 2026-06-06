from __future__ import annotations

from copy import deepcopy

import pytest

from koolook_setups import FileSetupStorage, PublishedSetupRegistry, StaticSetupStorage, validate_setup


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


def _json(value: dict) -> str:
    import json

    return json.dumps(value)
