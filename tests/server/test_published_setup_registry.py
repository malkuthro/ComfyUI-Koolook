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
        "visualGraph": {
            "nodes": [
                {
                    "id": 12,
                    "type": "Text Multiline",
                    "inputs": [{"name": "text", "type": "STRING", "widget": {"name": "text"}}],
                    "widgets_values": ["demo prompt"],
                }
            ],
            "links": [],
        },
        "apiPrompt": {
            "12": {
                "class_type": "Text Multiline",
                "inputs": {"text": "demo prompt"},
            }
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


def test_validate_setup_rejects_valid_status_without_api_prompt() -> None:
    setup = deepcopy(_valid_setup())
    setup["apiPrompt"] = None

    result = validate_setup(setup)

    assert result.valid is False
    assert "validation.status valid requires apiPrompt" in result.diagnostics


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
    assert detail["apiPrompt"] == _valid_setup()["apiPrompt"]
    assert registry.getSetup("broken") is None
    assert registry.diagnostics == ["broken: missing required field: metadata"]


def test_registry_preserves_supplied_api_prompt_that_differs_from_fallback_conversion() -> None:
    stale = deepcopy(_valid_setup())
    stale["apiPrompt"] = {
        "1": {
            "class_type": "Different Node",
            "inputs": {},
        }
    }
    registry = PublishedSetupRegistry(StaticSetupStorage([stale]))

    assert registry.listSetups()[0]["id"] == "ltx-director-demo"
    setup = registry.getSetup("ltx-director-demo")
    assert setup is not None
    assert setup["apiPrompt"] == stale["apiPrompt"]
    assert registry.diagnostics == []


def test_registry_normalizes_legacy_visual_artifacts_in_api_prompt() -> None:
    setup = deepcopy(_valid_setup())
    setup["apiPrompt"]["99"] = {"class_type": "Label (rgthree)", "inputs": {}}
    registry = PublishedSetupRegistry(StaticSetupStorage([setup]))

    normalized = registry.getSetup("ltx-director-demo")

    assert normalized is not None
    assert normalized["apiPrompt"] == _valid_setup()["apiPrompt"]
    assert registry.diagnostics == []


def test_registry_omits_group_first_setup_missing_persisted_surface() -> None:
    setup = deepcopy(_valid_setup())
    setup["id"] = "missing-surface"
    setup["inputContract"] = {"inputs": []}
    setup["outputContract"] = {"outputs": []}
    registry = PublishedSetupRegistry(StaticSetupStorage([setup]))

    assert registry.getSetup("missing-surface") is None
    assert registry.diagnostics == [
        "missing-surface: setupSurface must be a JSON object for group-authored setups"
    ]


def test_registry_omits_malformed_persisted_setup_surface() -> None:
    setup = deepcopy(_valid_setup())
    setup["setupSurface"] = {
        "sourceInputs": [{"group": "Koolook Input", "nodes": [{"id": "12"}]}],
        "outputs": "not-list",
        "controls": [],
    }
    registry = PublishedSetupRegistry(StaticSetupStorage([setup]))

    assert registry.getSetup("ltx-director-demo") is None
    assert "ltx-director-demo: setupSurface.outputs must be a list" in registry.diagnostics


def test_registry_omits_malformed_persisted_setup_surface_app() -> None:
    setup = deepcopy(_valid_setup())
    setup["inputContract"] = {"inputs": []}
    setup["outputContract"] = {"outputs": []}
    setup["setupSurface"] = {
        "sourceInputs": [
            {
                "group": "Koolook Input",
                "nodes": [{"id": "12", "type": "Koolook_PublishInput", "title": "Input"}],
            }
        ],
        "outputs": [
            {
                "group": "Koolook Output",
                "nodes": [{"id": "20", "type": "Koolook_PublishOutput", "title": "Output"}],
            }
        ],
        "controls": [],
        "app": "not-object",
    }
    registry = PublishedSetupRegistry(StaticSetupStorage([setup]))

    assert registry.getSetup("ltx-director-demo") is None
    assert "ltx-director-demo: setupSurface.app must be a JSON object" in registry.diagnostics


def test_registry_omits_malformed_persisted_setup_surface_app_fields() -> None:
    setup = deepcopy(_valid_setup())
    setup["inputContract"] = {"inputs": []}
    setup["outputContract"] = {"outputs": []}
    setup["setupSurface"] = {
        "sourceInputs": [
            {
                "group": "Koolook Input",
                "nodes": [{"id": "12", "type": "Koolook_PublishInput", "title": "Input"}],
            }
        ],
        "outputs": [
            {
                "group": "Koolook Output",
                "nodes": [{"id": "20", "type": "Koolook_PublishOutput", "title": "Output"}],
            }
        ],
        "controls": [],
        "app": {
            "inputs": [
                {
                    "key": "single_file",
                    "label": "Single file",
                    "visible": True,
                    "target": {"node": "12"},
                    "default": "/image.png",
                }
            ],
            "outputs": "not-list",
            "results": [],
            "switch": {
                "key": "switch",
                "label": "Input type",
                "visible": True,
                "target": {"node": "12", "input": "mode"},
                "default": 2,
                "options": [{"value": True, "label": "Img", "visible": True, "input": "single_file"}],
            },
        },
    }
    registry = PublishedSetupRegistry(StaticSetupStorage([setup]))

    assert registry.getSetup("ltx-director-demo") is None
    assert "ltx-director-demo: setupSurface.app.outputs must be a list" in registry.diagnostics
    assert (
        "ltx-director-demo: setupSurface.app.inputs[0].target.input must be non-empty text"
        in registry.diagnostics
    )
    assert (
        "ltx-director-demo: setupSurface.app.switch.options[0].value must be an integer"
        in registry.diagnostics
    )


def test_registry_omits_persisted_setup_surface_app_targets_missing_from_graph() -> None:
    setup = deepcopy(_valid_setup())
    setup["inputContract"] = {"inputs": []}
    setup["outputContract"] = {"outputs": []}
    setup["setupSurface"] = {
        "sourceInputs": [
            {
                "group": "Koolook Input",
                "nodes": [{"id": "12", "type": "Koolook_PublishInput", "title": "Input"}],
            }
        ],
        "outputs": [
            {
                "group": "Koolook Output",
                "nodes": [{"id": "20", "type": "Koolook_PublishOutput", "title": "Output"}],
            }
        ],
        "controls": [],
        "app": {
            "inputs": [
                {
                    "key": "single_file",
                    "label": "Single file",
                    "visible": True,
                    "target": {"node": "999", "input": "missing"},
                    "default": "/image.png",
                }
            ],
            "outputs": [],
            "results": [],
            "switch": {
                "key": "switch",
                "label": "Input type",
                "visible": True,
                "target": {"node": "12", "input": "mode"},
                "default": 2,
                "options": [{"value": 2, "label": "Img", "visible": True, "input": "missing_key"}],
            },
        },
    }
    registry = PublishedSetupRegistry(StaticSetupStorage([setup]))

    assert registry.getSetup("ltx-director-demo") is None
    assert (
        "ltx-director-demo: setupSurface.app.inputs[0].target.node not found in visualGraph"
        in registry.diagnostics
    )
    assert (
        "ltx-director-demo: setupSurface.app.switch.options[0].input must match a setupSurface.app.inputs key"
        in registry.diagnostics
    )


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
        visualGraph={
            "nodes": [
                {
                    "id": 12,
                    "type": "Text Multiline",
                    "inputs": [{"name": "text", "widget": {"name": "text"}}],
                    "widgets_values": ["first prompt"],
                }
            ],
            "links": [],
        },
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


def test_publish_setup_converts_supported_visual_graph_to_api_prompt() -> None:
    storage = StaticSetupStorage([])
    registry = PublishedSetupRegistry(storage)
    visual_graph = {
        "nodes": [
            {
                "id": 12,
                "type": "Text Multiline",
                "inputs": [{"name": "text", "type": "STRING", "widget": {"name": "text"}}],
                "widgets_values": ["default prompt"],
            },
            {
                "id": 20,
                "type": "Text Concatenate",
                "inputs": [
                    {"name": "text_a", "type": "STRING", "link": 101},
                    {"name": "delimiter", "type": "STRING", "widget": {"name": "delimiter"}},
                ],
                "widgets_values": ["_"],
            }
        ],
        "links": [[101, 12, 0, 20, 0, "STRING"]],
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
    assert setup["apiPrompt"] == {
        "12": {
            "class_type": "Text Multiline",
            "inputs": {"text": "default prompt"},
        },
        "20": {
            "class_type": "Text Concatenate",
            "inputs": {"text_a": ["12", 0], "delimiter": "_"},
        }
    }
    assert setup["source"] == {"kind": "sidebar-workflow", "path": "Demos/My Callable Flow"}
    assert setup["updatedAt"].endswith("Z")
    assert "+00:00" in datetime.fromisoformat(setup["updatedAt"].replace("Z", "+00:00")).isoformat()
    assert setup["validation"] == {"status": "valid", "diagnostics": []}
    assert registry.listSetups()[0]["id"] == "my-callable-flow"


def test_publish_setup_allows_empty_description() -> None:
    registry = PublishedSetupRegistry(StaticSetupStorage([]))

    result = registry.publishSetup(
        visualGraph={
            "nodes": [
                {
                    "id": 12,
                    "type": "Text Multiline",
                    "inputs": [{"name": "text", "type": "STRING", "widget": {"name": "text"}, "link": None}],
                    "widgets_values": ["default prompt"],
                }
            ],
            "links": [],
        },
        metadata={"id": "empty-description-flow", "title": "Empty Description", "description": ""},
        inputContract={"inputs": []},
        outputContract={"outputs": [{"key": "preview", "type": "text"}]},
        source={"kind": "sidebar-workflow", "path": "Demos/Empty Description"},
    )

    assert result.valid is True
    setup = registry.getSetup("empty-description-flow")
    assert setup is not None
    assert setup["metadata"]["description"] == ""


def test_publish_setup_converts_text_multiline_without_serialized_inputs() -> None:
    registry = PublishedSetupRegistry(StaticSetupStorage([]))
    visual_graph = {
        "nodes": [
            {
                "id": 3,
                "type": "Text Multiline",
                "pos": [40, 40],
                "size": [300, 100],
                "inputs": [],
                "widgets_values": ["W:/projects/example/frames"],
            },
            {
                "id": 1,
                "type": "SaveEXRFrames",
                "pos": [420, 40],
                "size": [240, 180],
                "inputs": [],
                "widgets_values": ["path/to/frame%04d.exr", "linear", 1001, False, "ui"],
            },
        ],
        "links": [],
        "groups": [
            {"title": "Koolook Input", "bounding": [20, 20, 340, 160]},
            {"title": "Koolook Output", "bounding": [400, 20, 300, 240]},
        ],
    }

    result = registry.publishSetup(
        visualGraph=visual_graph,
        metadata={
            "id": "text-multiline-simple-setup",
            "title": "Text Multiline Simple Setup",
            "description": "A simple setup with widget-only source text.",
        },
        inputContract={"inputs": []},
        outputContract={"outputs": []},
        source={"kind": "sidebar-workflow", "path": "Demos/Text Multiline"},
    )

    assert result.valid is True
    setup = registry.getSetup("text-multiline-simple-setup")
    assert setup is not None
    assert setup["apiPrompt"]["3"]["inputs"] == {"text": "W:/projects/example/frames"}


def test_publish_setup_rejects_text_multiline_without_widget_value() -> None:
    registry = PublishedSetupRegistry(StaticSetupStorage([]))

    result = registry.publishSetup(
        visualGraph={
            "nodes": [
                {
                    "id": 3,
                    "type": "Text Multiline",
                    "pos": [40, 40],
                    "size": [300, 100],
                    "inputs": [],
                },
                {"id": 1, "type": "Preview Image", "pos": [420, 40], "size": [240, 180], "inputs": []},
            ],
            "links": [],
            "groups": [
                {"title": "Koolook Input", "bounding": [20, 20, 340, 160]},
                {"title": "Koolook Output", "bounding": [400, 20, 300, 240]},
            ],
        },
        metadata={
            "id": "missing-text-widget",
            "title": "Missing Text Widget",
            "description": "Missing widget-backed text.",
        },
        inputContract={"inputs": []},
        outputContract={"outputs": []},
        source={"kind": "sidebar-workflow", "path": "Demos/Missing Text"},
    )

    assert result.valid is False
    assert (
        "visualGraph.nodes[0].widgets_values is missing a value for widget-only input text"
        in result.diagnostics
    )


def test_publish_setup_converts_easy_ai_pipeline_widget_only_values() -> None:
    registry = PublishedSetupRegistry(StaticSetupStorage([]))
    visual_graph = {
        "nodes": [
            {
                "id": 6,
                "type": "EasyAIPipeline",
                "pos": [40, 40],
                "size": [420, 700],
                "inputs": [],
                "widgets_values": [
                    81,
                    453453453,
                    "Place your base folder path in the FIELD below",
                    "W:/projects/example",
                    ".%04d.exr",
                    "msk",
                    "",
                    "1",
                    False,
                    False,
                    "W:/preview/path/frame.%04d.exr",
                    "",
                    None,
                    None,
                ],
            },
            {"id": 1, "type": "SaveEXRFrames", "pos": [620, 40], "size": [240, 180], "inputs": []},
        ],
        "links": [],
        "groups": [
            {"title": "Koolook Input", "bounding": [20, 20, 500, 760]},
            {"title": "Koolook Output", "bounding": [600, 20, 300, 240]},
        ],
    }

    result = registry.publishSetup(
        visualGraph=visual_graph,
        metadata={
            "id": "easy-ai-pipeline-simple-setup",
            "title": "Easy AI Pipeline Simple Setup",
            "description": "A simple setup with EasyAIPipeline widget-only values.",
        },
        inputContract={"inputs": []},
        outputContract={"outputs": []},
        source={"kind": "sidebar-workflow", "path": "Demos/EasyAIPipeline"},
    )

    assert result.valid is True
    setup = registry.getSetup("easy-ai-pipeline-simple-setup")
    assert setup is not None
    assert setup["apiPrompt"]["6"]["inputs"] == {
        "shot_duration": 81,
        "seed_value": 453453453,
        "instruction": "Place your base folder path in the FIELD below",
        "base_directory_path": "W:/projects/example",
        "extension": ".%04d.exr",
        "shot_name": "msk",
        "ai_method": "",
        "version": "1",
        "disable_versioning": False,
        "enable_overwrite": False,
        "no_subfolders": False,
    }


def test_publish_setup_rejects_partial_easy_ai_pipeline_widget_only_values() -> None:
    registry = PublishedSetupRegistry(StaticSetupStorage([]))

    result = registry.publishSetup(
        visualGraph={
            "nodes": [
                {
                    "id": 6,
                    "type": "EasyAIPipeline",
                    "pos": [40, 40],
                    "size": [420, 700],
                    "inputs": [],
                    "widgets_values": [81],
                },
                {"id": 1, "type": "SaveEXRFrames", "pos": [620, 40], "size": [240, 180], "inputs": []},
            ],
            "links": [],
            "groups": [
                {"title": "Koolook Input", "bounding": [20, 20, 500, 760]},
                {"title": "Koolook Output", "bounding": [600, 20, 300, 240]},
            ],
        },
        metadata={
            "id": "partial-easy-ai-pipeline",
            "title": "Partial Easy AI Pipeline",
            "description": "Missing widget-backed pipeline values.",
        },
        inputContract={"inputs": []},
        outputContract={"outputs": []},
        source={"kind": "sidebar-workflow", "path": "Demos/Partial EasyAIPipeline"},
    )

    assert result.valid is False
    assert (
        "visualGraph.nodes[0].widgets_values is missing a value for widget-only input seed_value"
        in result.diagnostics
    )
    assert (
        "visualGraph.nodes[0].widgets_values is missing a value for widget-only input base_directory_path"
        in result.diagnostics
    )
    assert (
        "visualGraph.nodes[0].widgets_values is missing a value for widget-only input no_subfolders"
        not in result.diagnostics
    )


def test_publish_setup_converts_subgraph_proxy_widget_values() -> None:
    registry = PublishedSetupRegistry(StaticSetupStorage([]))
    visual_graph = {
        "nodes": [
            {
                "id": 10,
                "type": "sam3-subgraph",
                "inputs": [
                    {"name": "text", "type": "STRING", "widget": {"name": "text"}, "link": None},
                    {"name": "threshold", "type": "FLOAT", "widget": {"name": "threshold"}, "link": None},
                    {"name": "image", "type": "IMAGE", "link": 42},
                ],
                "properties": {
                    "proxyWidgets": [
                        ["135", "text"],
                        ["133", "threshold"],
                    ],
                },
                "widgets_values": [],
            },
            {"id": 20, "type": "LoadImage", "inputs": []},
        ],
        "links": [
            [42, 20, 0, 10, 2, "IMAGE"],
        ],
        "definitions": {
            "subgraphs": [
                {
                    "id": "sam3-subgraph",
                    "inputs": [
                        {"name": "text", "type": "STRING"},
                        {"name": "threshold", "type": "FLOAT"},
                        {"name": "image", "type": "IMAGE"},
                    ],
                    "nodes": [
                        {
                            "id": 135,
                            "type": "CLIPTextEncode",
                            "inputs": [
                                {"name": "clip", "type": "CLIP", "link": None},
                                {"name": "text", "type": "STRING", "widget": {"name": "text"}, "link": 2},
                            ],
                            "widgets_values": ["bear"],
                        },
                        {
                            "id": 133,
                            "type": "SAM3_Detect",
                            "inputs": [
                                {"name": "image", "type": "IMAGE", "link": 3},
                                {"name": "threshold", "type": "FLOAT", "widget": {"name": "threshold"}, "link": 4},
                            ],
                            "widgets_values": [0.5],
                        },
                    ],
                    "links": [
                        {"id": 2, "origin_id": -10, "origin_slot": 0, "target_id": 135, "target_slot": 1},
                        {"id": 3, "origin_id": -10, "origin_slot": 2, "target_id": 133, "target_slot": 0},
                        {"id": 4, "origin_id": -10, "origin_slot": 1, "target_id": 133, "target_slot": 1},
                    ],
                },
            ],
        },
    }

    result = registry.publishSetup(
        visualGraph=visual_graph,
        metadata={
            "id": "sam3-proxy-widget-setup",
            "title": "SAM3 Proxy Widget Setup",
            "description": "A subgraph wrapper with proxy widget defaults.",
        },
        inputContract={"inputs": []},
        outputContract={"outputs": [{"key": "mask", "type": "mask"}]},
        source={"kind": "sidebar-workflow", "path": "Demos/SAM3"},
    )

    assert result.valid is True
    setup = registry.getSetup("sam3-proxy-widget-setup")
    assert setup is not None
    assert "10" not in setup["apiPrompt"]
    assert setup["apiPrompt"]["10:135"]["inputs"] == {"text": "bear"}
    assert setup["apiPrompt"]["10:133"]["inputs"] == {
        "threshold": 0.5,
        "image": ["20", 0],
    }


def test_publish_setup_converts_widget_values_saved_as_mapping() -> None:
    registry = PublishedSetupRegistry(StaticSetupStorage([]))
    visual_graph = {
        "nodes": [
            {
                "id": 142,
                "type": "VHS_LoadVideo",
                "inputs": [
                    {"name": "video", "type": "COMBO", "widget": {"name": "video"}, "link": None},
                    {"name": "force_rate", "type": "FLOAT", "widget": {"name": "force_rate"}, "link": None},
                    {"name": "frame_load_cap", "type": "INT", "widget": {"name": "frame_load_cap"}, "link": None},
                ],
                "widgets_values": {
                    "video": "example.mp4",
                    "force_rate": 0,
                    "frame_load_cap": 41,
                },
            },
        ],
        "links": [],
    }

    result = registry.publishSetup(
        visualGraph=visual_graph,
        metadata={
            "id": "mapping-widget-values",
            "title": "Mapping Widget Values",
            "description": "A node saved with named widget values.",
        },
        inputContract={"inputs": []},
        outputContract={"outputs": [{"key": "preview", "type": "video"}]},
        source={"kind": "sidebar-workflow", "path": "Demos/Mapping Widgets"},
    )

    assert result.valid is True
    setup = registry.getSetup("mapping-widget-values")
    assert setup is not None
    assert setup["apiPrompt"]["142"]["inputs"] == {
        "video": "example.mp4",
        "force_rate": 0,
        "frame_load_cap": 41,
    }


def test_publish_setup_converts_reroute_with_unnamed_input() -> None:
    registry = PublishedSetupRegistry(StaticSetupStorage([]))
    visual_graph = {
        "nodes": [
            {
                "id": 1,
                "type": "Text Multiline",
                "inputs": [],
                "outputs": [{"name": "STRING", "type": "STRING", "links": [10]}],
                "widgets_values": ["example"],
            },
            {
                "id": 170,
                "type": "Reroute",
                "inputs": [{"name": "", "type": "*", "link": 10}],
                "outputs": [{"name": "", "type": "*", "links": [11]}],
            },
            {
                "id": 2,
                "type": "Text Concatenate",
                "inputs": [{"name": "text_a", "type": "STRING", "link": 11}],
            },
        ],
        "links": [[10, 1, 0, 170, 0, "*"], [11, 170, 0, 2, 0, "*"]],
    }

    result = registry.publishSetup(
        visualGraph=visual_graph,
        metadata={
            "id": "reroute-flow",
            "title": "Reroute Flow",
            "description": "A workflow with an unnamed reroute input.",
        },
        inputContract={"inputs": []},
        outputContract={"outputs": [{"key": "preview", "type": "text"}]},
        source={"kind": "sidebar-workflow", "path": "Demos/Reroute"},
    )

    assert result.valid is True
    setup = registry.getSetup("reroute-flow")
    assert setup is not None
    assert "170" not in setup["apiPrompt"]
    assert setup["apiPrompt"]["2"]["inputs"] == {"text_a": ["1", 0]}


def test_publish_setup_infers_app_surface_from_koolook_groups() -> None:
    storage = StaticSetupStorage([])
    registry = PublishedSetupRegistry(storage)
    visual_graph = {
        "nodes": [
            {
                "id": 12,
                "type": "Load Image",
                "title": "Source image",
                "pos": [40, 40],
                "size": [180, 80],
                "inputs": [],
            },
            {
                "id": 20,
                "type": "Preview Image",
                "title": "Preview",
                "pos": [420, 40],
                "size": [180, 80],
                "inputs": [],
            },
        ],
        "links": [],
        "groups": [
            {"title": "Koolook Input", "bounding": [20, 20, 240, 140]},
            {"title": "Koolook Output", "pos": [400, 20], "size": [240, 140]},
        ],
    }

    result = registry.publishSetup(
        visualGraph=visual_graph,
        metadata={
            "id": "group-surface-flow",
            "title": "Group Surface Flow",
            "description": "A setup authored with ComfyUI groups.",
        },
        inputContract={"inputs": []},
        outputContract={"outputs": []},
        source={"kind": "sidebar-workflow", "path": "Demos/Group Surface"},
    )

    assert result.valid is True
    setup = registry.getSetup("group-surface-flow")
    assert setup is not None
    assert setup["setupSurface"] == {
        "sourceInputs": [
            {
                "group": "Koolook Input",
                "nodes": [{"id": "12", "type": "Load Image", "title": "Source image"}],
            }
        ],
        "outputs": [
            {
                "group": "Koolook Output",
                "nodes": [{"id": "20", "type": "Preview Image", "title": "Preview"}],
            }
        ],
        "controls": [],
        "app": {"inputs": [], "outputs": [], "results": []},
    }


def test_publish_setup_infers_app_contract_from_publish_nodes() -> None:
    storage = StaticSetupStorage([])
    registry = PublishedSetupRegistry(storage)
    visual_graph = {
        "nodes": [
            {
                "id": 100,
                "type": "Koolook_PublishInput",
                "title": "Koolook Publish Input",
                "pos": [40, 60],
                "size": [360, 320],
                "inputs": [],
                "widgets_values": [
                    "Img",
                    "/shots/example/frames",
                    "/shots/example/movie.mov",
                    "/shots/example/image.png",
                    "unused prompt",
                ],
                "outputs": [
                    {"name": "sequence_folder", "type": "STRING", "links": []},
                    {"name": "qt_file", "type": "STRING", "links": []},
                    {"name": "single_file", "type": "STRING", "links": []},
                    {"name": "prompt", "type": "STRING", "links": []},
                    {"name": "switch", "type": "INT", "links": []},
                ],
            },
            {
                "id": 200,
                "type": "Koolook_PublishOutput",
                "title": "Koolook Publish Output",
                "pos": [520, 60],
                "size": [360, 240],
                "inputs": [],
                "widgets_values": [
                    "/shots/example/output",
                    "publish-OUT",
                    "1",
                ],
                "outputs": [
                    {"name": "folder", "type": "STRING", "links": []},
                    {"name": "name", "type": "STRING", "links": []},
                    {"name": "version", "type": "STRING", "links": []},
                ],
            },
            {
                "id": 300,
                "type": "Koolook_PublishResult",
                "title": "Koolook Publish Result",
                "pos": [520, 340],
                "size": [360, 160],
                "inputs": [
                    {
                        "name": "result",
                        "type": "STRING",
                        "widget": {"name": "result"},
                        "link": None,
                    }
                ],
                "widgets_values": [
                    "/shots/example/output/publish-OUT_v001.mov",
                ],
                "outputs": [
                    {"name": "result", "type": "STRING", "links": []},
                ],
            },
        ],
        "links": [],
        "groups": [
            {"title": "Koolook Input", "bounding": [20, 20, 440, 420]},
            {"title": "Koolook Output", "bounding": [500, 20, 360, 560]},
        ],
    }

    result = registry.publishSetup(
        visualGraph=visual_graph,
        metadata={
            "id": "app-contract-flow",
            "title": "App Contract Flow",
            "description": "A setup authored with Koolook publish nodes.",
        },
        inputContract={"inputs": []},
        outputContract={"outputs": []},
        source={"kind": "sidebar-workflow", "path": "Demos/App Contract"},
    )

    assert result.valid is True
    setup = registry.getSetup("app-contract-flow")
    assert setup is not None
    assert setup["setupSurface"]["app"] == {
        "inputs": [
            {
                "key": "sequence_folder",
                "label": "Sequence folder",
                "visible": True,
                "target": {"node": "100", "input": "sequence_folder"},
                "default": "/shots/example/frames",
            },
            {
                "key": "qt_file",
                "label": "QT file",
                "visible": True,
                "target": {"node": "100", "input": "qt_file"},
                "default": "/shots/example/movie.mov",
            },
            {
                "key": "single_file",
                "label": "Single file",
                "visible": True,
                "target": {"node": "100", "input": "single_file"},
                "default": "/shots/example/image.png",
            },
            {
                "key": "prompt",
                "label": "Prompt",
                "visible": True,
                "standalone": True,
                "multiline": True,
                "target": {"node": "100", "input": "prompt"},
                "default": "",
                "placeholder": "unused prompt",
                "help": "Describe the shot in one simple line: subject + action + setting.",
            },
        ],
        "outputs": [
            {
                "key": "folder",
                "label": "Output folder",
                "visible": True,
                "target": {"node": "200", "input": "folder"},
                "default": "/shots/example/output",
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
                "default": "1",
            },
        ],
        "results": [
            {
                "key": "result",
                "label": "Result",
                "visible": True,
                "target": {"node": "300", "input": "result"},
                "default": "/shots/example/output/publish-OUT_v001.mov",
            },
        ],
        "switch": {
            "key": "switch",
            "label": "Input type",
            "visible": True,
            "target": {"node": "100", "input": "mode"},
            "default": 2,
            "options": [
                {"value": 0, "label": "EXR", "visible": True, "input": "sequence_folder"},
                {"value": 1, "label": "QT", "visible": True, "input": "qt_file"},
                {"value": 2, "label": "Img", "visible": True, "input": "single_file"},
                {"value": 3, "label": "Prompt", "visible": False, "input": "prompt"},
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
    }


def test_publish_setup_stores_execution_map_from_publish_router() -> None:
    storage = StaticSetupStorage([])
    registry = PublishedSetupRegistry(storage)
    visual_graph = {
        "nodes": [
            {
                "id": 100,
                "type": "Koolook_PublishInput",
                "title": "Koolook Publish Input",
                "pos": [40, 60],
                "size": [360, 320],
                "inputs": [],
                "widgets_values": ["Img", "/seq", "/movie.mov", "/image.png", "prompt"],
            },
            {
                "id": 200,
                "type": "Koolook_PublishOutput",
                "title": "Koolook Publish Output",
                "pos": [520, 60],
                "size": [360, 240],
                "inputs": [],
                "widgets_values": ["/out", "mask", "1", "Same as input"],
            },
            {
                "id": 400,
                "type": "Koolook_PublishRouter",
                "title": "Koolook Publish Router",
                "pos": [520, 340],
                "size": [360, 220],
                "inputs": [],
                "widgets_values": [],
            },
            {"id": 311, "type": "SaveEXRFrames", "title": "EXR writer", "pos": [920, 80], "size": [220, 120]},
            {"id": 312, "type": "Easy_VideoCombine", "title": "QT writer", "pos": [920, 260], "size": [220, 120]},
            {"id": 313, "type": "SaveImageAndPromptExact", "title": "Img writer", "pos": [920, 440], "size": [220, 120]},
        ],
        "links": [],
        "groups": [
            {"title": "Koolook Input", "bounding": [20, 20, 420, 420]},
            {"title": "Koolook Output", "bounding": [500, 20, 700, 620]},
        ],
    }

    result = registry.publishSetup(
        visualGraph=visual_graph,
        apiPrompt={
            "100": {
                "class_type": "Koolook_PublishInput",
                "inputs": {
                    "mode": "Img",
                    "sequence_folder": "/seq",
                    "qt_file": "/movie.mov",
                    "single_file": "/image.png",
                    "prompt": "prompt",
                },
            },
            "200": {"class_type": "Koolook_PublishOutput", "inputs": {"folder": "/out", "name": "mask", "version": "1", "output_mode": "Same as input", "input_switch": ["100", 4]}},
            "300": {"class_type": "RMBG", "inputs": {"image": ["100", 2]}},
            "301": {"class_type": "EasyAIPipeline", "inputs": {"base_directory_path": ["200", 0], "extension": ".exr"}},
            "302": {"class_type": "EasyAIPipeline", "inputs": {"base_directory_path": ["200", 0], "extension": ".mov"}},
            "303": {"class_type": "EasyAIPipeline", "inputs": {"base_directory_path": ["200", 0], "extension": ".png"}},
            "400": {"class_type": "Koolook_PublishRouter", "inputs": {"selector": ["100", 4], "payload": ["300", 0]}},
            "311": {"class_type": "SaveEXRFrames", "inputs": {"filepath": ["301", 0], "images": ["400", 0]}},
            "312": {"class_type": "Easy_VideoCombine", "inputs": {"filepath": ["302", 0], "images": ["400", 1]}},
            "313": {"class_type": "SaveImageAndPromptExact", "inputs": {"filepath": ["303", 0], "image": ["400", 2]}},
        },
        metadata={
            "id": "router-contract-flow",
            "title": "Router Contract Flow",
            "description": "A setup authored with a publish router.",
        },
        inputContract={"inputs": []},
        outputContract={"outputs": []},
        source={"kind": "sidebar-workflow", "path": "Demos/Router Contract"},
    )

    assert result.valid is True
    setup = registry.getSetup("router-contract-flow")
    assert setup is not None
    assert setup["executionMap"] == {
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


def test_publish_setup_execution_map_uses_output_switch_when_router_is_output_driven() -> None:
    """An EXR-in / QT-out setup wires the writer router from the OUTPUT switch
    (Koolook_PublishOutput.switch, slot 3) instead of the input switch. The
    executionMap must tag that router with switchKey 'output_switch' so the
    runner prunes writer branches by the independently-chosen output type."""
    storage = StaticSetupStorage([])
    registry = PublishedSetupRegistry(storage)
    visual_graph = {
        "nodes": [
            {
                "id": 100,
                "type": "Koolook_PublishInput",
                "title": "Koolook Publish Input",
                "pos": [40, 60],
                "size": [360, 320],
                "inputs": [],
                "widgets_values": ["EXR", "/seq", "/movie.mov", "/image.png", "prompt"],
            },
            {
                "id": 200,
                "type": "Koolook_PublishOutput",
                "title": "Koolook Publish Output",
                "pos": [520, 60],
                "size": [360, 240],
                "inputs": [],
                "widgets_values": ["/out", "mask", "1", "QT"],
            },
            {
                "id": 400,
                "type": "Koolook_PublishRouter",
                "title": "Koolook Publish Router",
                "pos": [520, 340],
                "size": [360, 220],
                "inputs": [],
                "widgets_values": [],
            },
            {"id": 311, "type": "SaveEXRFrames", "title": "EXR writer", "pos": [920, 80], "size": [220, 120]},
            {"id": 312, "type": "Easy_VideoCombine", "title": "QT writer", "pos": [920, 260], "size": [220, 120]},
            {"id": 313, "type": "SaveImageAndPromptExact", "title": "Img writer", "pos": [920, 440], "size": [220, 120]},
        ],
        "links": [],
        "groups": [
            {"title": "Koolook Input", "bounding": [20, 20, 420, 420]},
            {"title": "Koolook Output", "bounding": [500, 20, 700, 620]},
        ],
    }

    result = registry.publishSetup(
        visualGraph=visual_graph,
        apiPrompt={
            "100": {
                "class_type": "Koolook_PublishInput",
                "inputs": {
                    "mode": "EXR",
                    "sequence_folder": "/seq",
                    "qt_file": "/movie.mov",
                    "single_file": "/image.png",
                    "prompt": "prompt",
                },
            },
            "200": {
                "class_type": "Koolook_PublishOutput",
                # input_switch wired from the input switch so "Same as input"
                # setups still work; here output_mode overrides to QT.
                "inputs": {"folder": "/out", "name": "mask", "version": "1", "output_mode": "QT", "input_switch": ["100", 4]},
            },
            "300": {"class_type": "RMBG", "inputs": {"image": ["100", 2]}},
            "301": {"class_type": "EasyAIPipeline", "inputs": {"base_directory_path": ["200", 0], "extension": ".exr"}},
            "302": {"class_type": "EasyAIPipeline", "inputs": {"base_directory_path": ["200", 0], "extension": ".mov"}},
            "303": {"class_type": "EasyAIPipeline", "inputs": {"base_directory_path": ["200", 0], "extension": ".png"}},
            # Router selector wired from the OUTPUT switch (node 200, slot 3).
            "400": {"class_type": "Koolook_PublishRouter", "inputs": {"selector": ["200", 3], "payload": ["300", 0]}},
            "311": {"class_type": "SaveEXRFrames", "inputs": {"filepath": ["301", 0], "images": ["400", 0]}},
            "312": {"class_type": "Easy_VideoCombine", "inputs": {"filepath": ["302", 0], "images": ["400", 1]}},
            "313": {"class_type": "SaveImageAndPromptExact", "inputs": {"filepath": ["303", 0], "image": ["400", 2]}},
        },
        metadata={
            "id": "divergent-output-flow",
            "title": "Divergent Output Flow",
            "description": "EXR input, QT output via an output-driven router.",
        },
        inputContract={"inputs": []},
        outputContract={"outputs": []},
        source={"kind": "sidebar-workflow", "path": "Demos/Divergent Output"},
    )

    assert result.valid is True
    setup = registry.getSetup("divergent-output-flow")
    assert setup is not None
    # The independent output switch is exposed and defaults to QT (author's widget).
    assert setup["setupSurface"]["app"]["outputSwitch"]["default"] == 1
    router = setup["executionMap"]["routers"][0]
    assert router["switchKey"] == "output_switch"
    assert router["selector"] == {"node": "200", "output": 3}
    assert router["branches"]["1"]["writerNodes"] == ["312"]  # QT writer


@pytest.mark.parametrize(
    ("visual_graph", "expected"),
    [
        (
            {
                "nodes": [{"id": 12, "type": "Load Image", "pos": [40, 40], "inputs": []}],
                "links": [],
                "groups": [{"title": "Koolook Output", "bounding": [20, 20, 240, 140]}],
            },
            "setupSurface.sourceInputs requires a non-empty Koolook Input group",
        ),
        (
            {
                "nodes": [{"id": 12, "type": "Load Image", "pos": [400, 40], "inputs": []}],
                "links": [],
                "groups": [
                    {"title": "Koolook Input", "bounding": [20, 20, 240, 140]},
                    {"title": "Koolook Output", "bounding": [20, 20, 240, 140]},
                ],
            },
            "setupSurface.sourceInputs requires a non-empty Koolook Input group",
        ),
        (
            {
                "nodes": [{"id": 12, "type": "Load Image", "pos": [40, 40], "inputs": []}],
                "links": [],
                "groups": [{"title": "Koolook Input", "bounding": [20, 20, 240, 140]}],
            },
            "setupSurface.outputs requires a non-empty Koolook Output group",
        ),
    ],
)
def test_publish_setup_rejects_missing_or_empty_required_surface_groups(
    visual_graph: dict,
    expected: str,
) -> None:
    registry = PublishedSetupRegistry(StaticSetupStorage([]))

    result = registry.publishSetup(
        visualGraph=visual_graph,
        metadata={
            "id": "bad-group-surface",
            "title": "Bad Group Surface",
            "description": "Missing required setup groups.",
        },
        inputContract={"inputs": []},
        outputContract={"outputs": []},
        source={"kind": "sidebar-workflow", "path": "Demos/Bad Group"},
    )

    assert result.valid is False
    assert expected in result.diagnostics
    assert registry.getSetup("bad-group-surface") is None


@pytest.mark.parametrize(
    ("visual_graph", "expected"),
    [
        (
            {
                "nodes": [{"id": 12, "type": "Load Image", "pos": [40, 40], "inputs": []}],
                "links": [],
                "groups": [
                    {"title": "Koolook Input", "bounding": ["bad", 20, 240, 140]},
                    {"title": "Koolook Output", "bounding": [20, 20, 240, 140]},
                ],
            },
            "visualGraph.groups[0].bounding must contain numeric x, y, width, height",
        ),
        (
            {
                "nodes": [{"id": 12, "type": "Load Image", "pos": ["bad", 40], "inputs": []}],
                "links": [],
                "groups": [
                    {"title": "Koolook Input", "bounding": [20, 20, 240, 140]},
                    {"title": "Koolook Output", "bounding": [20, 20, 240, 140]},
                ],
            },
            "visualGraph.nodes[0].pos must contain numeric x and y for setup surface inference",
        ),
        (
            {
                "nodes": [{"id": 12, "type": "Load Image", "pos": [True, 40], "inputs": []}],
                "links": [],
                "groups": [
                    {"title": "Koolook Input", "bounding": [20, 20, 240, 140]},
                    {"title": "Koolook Output", "bounding": [20, 20, 240, 140]},
                ],
            },
            "visualGraph.nodes[0].pos must contain numeric x and y for setup surface inference",
        ),
        (
            {
                "nodes": [{"id": 12, "type": "Load Image", "pos": [40, 40], "inputs": []}],
                "links": [],
                "groups": [
                    {"title": "Koolook Input", "bounding": [20, 20, float("inf"), 140]},
                    {"title": "Koolook Output", "bounding": [20, 20, 240, 140]},
                ],
            },
            "visualGraph.groups[0].bounding must contain numeric x, y, width, height",
        ),
    ],
)
def test_publish_setup_rejects_malformed_surface_geometry(
    visual_graph: dict,
    expected: str,
) -> None:
    registry = PublishedSetupRegistry(StaticSetupStorage([]))

    result = registry.publishSetup(
        visualGraph=visual_graph,
        metadata={
            "id": "bad-geometry",
            "title": "Bad Geometry",
            "description": "Malformed setup surface geometry.",
        },
        inputContract={"inputs": []},
        outputContract={"outputs": []},
        source={"kind": "sidebar-workflow", "path": "Demos/Bad Geometry"},
    )

    assert result.valid is False
    assert expected in result.diagnostics
    assert registry.getSetup("bad-geometry") is None


def test_publish_setup_rejects_missing_contract_target() -> None:
    registry = PublishedSetupRegistry(StaticSetupStorage([]))

    result = registry.publishSetup(
        visualGraph={
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


def test_publish_setup_rejects_visual_node_without_type() -> None:
    registry = PublishedSetupRegistry(StaticSetupStorage([]))

    result = registry.publishSetup(
        visualGraph={"nodes": [{"id": 12, "inputs": [{"name": "text"}]}]},
        metadata={"id": "bad-flow", "title": "Bad", "description": "Missing node type"},
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
        source={"kind": "sidebar-workflow", "path": "Demos/Bad"},
    )

    assert result.valid is False
    assert "visualGraph.nodes[0].type must be non-empty text" in result.diagnostics
    assert registry.getSetup("bad-flow") is None


def test_publish_setup_rejects_contract_target_missing_from_generated_api_prompt() -> None:
    registry = PublishedSetupRegistry(StaticSetupStorage([]))

    result = registry.publishSetup(
        visualGraph={
            "nodes": [
                {
                    "id": 12,
                    "type": "Text Multiline",
                    "inputs": [{"name": "text", "type": "STRING"}],
                    "widgets_values": ["default prompt"],
                }
            ],
            "links": [],
        },
        metadata={"id": "bad-flow", "title": "Bad", "description": "Not injectable"},
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
        source={"kind": "sidebar-workflow", "path": "Demos/Bad"},
    )

    assert result.valid is False
    assert (
        "inputContract.inputs[0].target.input not found in generated apiPrompt"
        in result.diagnostics
    )
    assert registry.getSetup("bad-flow") is None


def test_publish_setup_rejects_partial_module_graph_links() -> None:
    registry = PublishedSetupRegistry(StaticSetupStorage([]))

    result = registry.publishSetup(
        visualGraph={
            "nodes": [
                {
                    "id": 12,
                    "type": "Text Multiline",
                    "inputs": [{"name": "text", "type": "STRING", "link": 101}],
                    "widgets_values": ["default prompt"],
                }
            ],
            "links": [[101, -10, 0, 12, 0, "STRING"]],
        },
        metadata={"id": "bad-flow", "title": "Bad", "description": "Partial module graph"},
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
        source={"kind": "sidebar-workflow", "path": "Demos/Bad"},
    )

    assert result.valid is False
    assert "visualGraph.links[101] uses unsupported module graph sentinel node" in result.diagnostics
    assert registry.getSetup("bad-flow") is None


def test_publish_setup_rejects_link_with_missing_origin_node() -> None:
    registry = PublishedSetupRegistry(StaticSetupStorage([]))

    result = registry.publishSetup(
        visualGraph={
            "nodes": [
                {
                    "id": 20,
                    "type": "Text Concatenate",
                    "inputs": [{"name": "text_a", "type": "STRING", "link": 101}],
                }
            ],
            "links": [[101, 12, 0, 20, 0, "STRING"]],
        },
        metadata={"id": "bad-flow", "title": "Bad", "description": "Missing origin"},
        inputContract={"inputs": []},
        outputContract={"outputs": [{"key": "preview", "type": "image"}]},
        source={"kind": "sidebar-workflow", "path": "Demos/Bad"},
    )

    assert result.valid is False
    assert "visualGraph.links[101].origin_id not found in visualGraph" in result.diagnostics
    assert registry.getSetup("bad-flow") is None


def test_publish_setup_rejects_link_target_that_does_not_match_input() -> None:
    registry = PublishedSetupRegistry(StaticSetupStorage([]))

    result = registry.publishSetup(
        visualGraph={
            "nodes": [
                {
                    "id": 12,
                    "type": "Text Multiline",
                    "inputs": [{"name": "text", "widget": {"name": "text"}}],
                    "widgets_values": ["default prompt"],
                },
                {
                    "id": 20,
                    "type": "Text Concatenate",
                    "inputs": [{"name": "text_a", "type": "STRING", "link": 101}],
                },
            ],
            "links": [[101, 12, 0, 999, 0, "STRING"]],
        },
        metadata={"id": "bad-flow", "title": "Bad", "description": "Mismatched link"},
        inputContract={"inputs": []},
        outputContract={"outputs": [{"key": "preview", "type": "image"}]},
        source={"kind": "sidebar-workflow", "path": "Demos/Bad"},
    )

    assert result.valid is False
    assert "visualGraph.links[101].target does not match visualGraph.nodes[1].inputs[0]" in result.diagnostics
    assert registry.getSetup("bad-flow") is None


def test_publish_setup_rejects_link_with_malformed_origin_slot() -> None:
    registry = PublishedSetupRegistry(StaticSetupStorage([]))

    result = registry.publishSetup(
        visualGraph={
            "nodes": [
                {"id": 12, "type": "Text Multiline", "inputs": [], "widgets_values": ["prompt"]},
                {
                    "id": 20,
                    "type": "Text Concatenate",
                    "inputs": [{"name": "text_a", "type": "STRING", "link": 101}],
                },
            ],
            "links": [[101, 12, "zero", 20, 0, "STRING"]],
        },
        metadata={"id": "bad-flow", "title": "Bad", "description": "Malformed slot"},
        inputContract={"inputs": []},
        outputContract={"outputs": [{"key": "preview", "type": "image"}]},
        source={"kind": "sidebar-workflow", "path": "Demos/Bad"},
    )

    assert result.valid is False
    assert "visualGraph.links[101].origin_slot must be a non-negative integer" in result.diagnostics
    assert registry.getSetup("bad-flow") is None


def test_publish_setup_converts_links_from_nodes_declared_later() -> None:
    registry = PublishedSetupRegistry(StaticSetupStorage([]))
    visual_graph = {
        "nodes": [
            {
                "id": 20,
                "type": "Text Concatenate",
                "inputs": [{"name": "text_a", "type": "STRING", "link": 101}],
            },
            {"id": 12, "type": "Text Multiline", "inputs": [], "widgets_values": ["prompt"]},
        ],
        "links": [[101, 12, 0, 20, 0, "STRING"]],
    }

    result = registry.publishSetup(
        visualGraph=visual_graph,
        metadata={"id": "good-flow", "title": "Good", "description": "Later origin"},
        inputContract={"inputs": []},
        outputContract={"outputs": [{"key": "preview", "type": "image"}]},
        source={"kind": "sidebar-workflow", "path": "Demos/Good"},
    )

    assert result.valid is True
    setup = registry.getSetup("good-flow")
    assert setup is not None
    assert setup["apiPrompt"]["20"]["inputs"]["text_a"] == ["12", 0]


def test_publish_setup_rejects_duplicate_node_ids() -> None:
    registry = PublishedSetupRegistry(StaticSetupStorage([]))

    result = registry.publishSetup(
        visualGraph={
            "nodes": [
                {"id": 20, "type": "Text Multiline", "inputs": []},
                {"id": 20, "type": "Text Concatenate", "inputs": []},
            ],
            "links": [],
        },
        metadata={"id": "bad-flow", "title": "Bad", "description": "Duplicate nodes"},
        inputContract={"inputs": []},
        outputContract={"outputs": [{"key": "preview", "type": "image"}]},
        source={"kind": "sidebar-workflow", "path": "Demos/Bad"},
    )

    assert result.valid is False
    assert "visualGraph.nodes[1].id duplicates visualGraph.nodes[0].id" in result.diagnostics
    assert registry.getSetup("bad-flow") is None


def test_publish_setup_rejects_duplicate_link_ids() -> None:
    registry = PublishedSetupRegistry(StaticSetupStorage([]))

    result = registry.publishSetup(
        visualGraph={
            "nodes": [
                {"id": 12, "type": "Text Multiline", "inputs": []},
                {
                    "id": 20,
                    "type": "Text Concatenate",
                    "inputs": [{"name": "text_a", "type": "STRING", "link": 101}],
                },
            ],
            "links": [
                [101, 12, 0, 20, 0, "STRING"],
                [101, 12, 0, 20, 0, "STRING"],
            ],
        },
        metadata={"id": "bad-flow", "title": "Bad", "description": "Duplicate links"},
        inputContract={"inputs": []},
        outputContract={"outputs": [{"key": "preview", "type": "image"}]},
        source={"kind": "sidebar-workflow", "path": "Demos/Bad"},
    )

    assert result.valid is False
    assert "visualGraph.links[1].id duplicates visualGraph.links[0].id" in result.diagnostics
    assert registry.getSetup("bad-flow") is None


def test_publish_setup_rejects_present_non_list_node_inputs() -> None:
    registry = PublishedSetupRegistry(StaticSetupStorage([]))

    result = registry.publishSetup(
        visualGraph={
            "nodes": [{"id": 12, "type": "Text Multiline", "inputs": "not-list"}],
            "links": [],
        },
        metadata={"id": "bad-flow", "title": "Bad", "description": "Malformed inputs"},
        inputContract={"inputs": []},
        outputContract={"outputs": [{"key": "preview", "type": "image"}]},
        source={"kind": "sidebar-workflow", "path": "Demos/Bad"},
    )

    assert result.valid is False
    assert "visualGraph.nodes[0].inputs must be a list when present" in result.diagnostics
    assert registry.getSetup("bad-flow") is None


def _json(value: dict) -> str:
    import json

    return json.dumps(value)
