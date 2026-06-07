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


def test_registry_omits_setup_with_stale_api_prompt() -> None:
    stale = deepcopy(_valid_setup())
    stale["apiPrompt"] = {
        "1": {
            "class_type": "Different Node",
            "inputs": {},
        }
    }
    registry = PublishedSetupRegistry(StaticSetupStorage([stale]))

    assert registry.listSetups() == []
    assert registry.getSetup("ltx-director-demo") is None
    assert registry.diagnostics == ["ltx-director-demo: apiPrompt is stale for visualGraph"]


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
                    "nodes": [
                        {
                            "id": 135,
                            "type": "CLIPTextEncode",
                            "inputs": [
                                {"name": "clip", "type": "CLIP", "link": 1},
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
        inputContract={"inputs": [{"key": "prompt", "type": "text", "target": {"node": "10", "input": "text"}}]},
        outputContract={"outputs": []},
        source={"kind": "sidebar-workflow", "path": "Demos/SAM3"},
    )

    assert result.valid is True
    setup = registry.getSetup("sam3-proxy-widget-setup")
    assert setup is not None
    assert setup["apiPrompt"]["10"]["inputs"] == {
        "text": "bear",
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
                "outputs": [{"name": "", "type": "*", "links": []}],
            },
        ],
        "links": [[10, 1, 0, 170, 0, "*"]],
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
    assert setup["apiPrompt"]["170"] == {
        "class_type": "Reroute",
        "inputs": {"": ["1", 0]},
    }


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
        "app": {"inputs": [], "outputs": []},
    }


def test_publish_setup_infers_app_contract_from_named_group_nodes() -> None:
    storage = StaticSetupStorage([])
    registry = PublishedSetupRegistry(storage)
    visual_graph = {
        "nodes": [
            {
                "id": 3,
                "type": "Text Multiline",
                "title": "App : INPUT [ sequence folder ]",
                "pos": [40, 60],
                "size": [280, 80],
                "inputs": [],
                "widgets_values": ["/shots/example/frames"],
            },
            {
                "id": 147,
                "type": "Text Multiline",
                "title": "App : INPUT [ QT file ]",
                "pos": [40, 180],
                "size": [280, 80],
                "inputs": [],
                "widgets_values": ["/shots/example/movie.mov"],
            },
            {
                "id": 148,
                "type": "Text Multiline",
                "title": "App : INPUT [ single file ]",
                "pos": [40, 300],
                "size": [280, 80],
                "inputs": [],
                "widgets_values": ["/shots/example/image.png"],
            },
            {
                "id": 149,
                "type": "Text Multiline",
                "title": "App : INPUT optional [ prompt ]",
                "pos": [40, 420],
                "size": [280, 80],
                "inputs": [],
                "widgets_values": ["unused prompt"],
            },
            {
                "id": 182,
                "type": "easy int",
                "title": "App : INPUT [ switch ] 0=EXR / 1=QT / 2=Img / 3=Prompt",
                "pos": [40, 540],
                "size": [280, 60],
                "inputs": [],
                "outputs": [{"name": "int", "type": "INT", "links": []}],
                "widgets_values": [2],
            },
            {
                "id": 163,
                "type": "Text Multiline",
                "title": "App : OUTPUT [ folder ]",
                "pos": [520, 60],
                "size": [280, 80],
                "inputs": [],
                "widgets_values": ["/shots/example/output"],
            },
            {
                "id": 168,
                "type": "Text Multiline",
                "title": "App : OUTPUT [ name ]",
                "pos": [520, 180],
                "size": [280, 80],
                "inputs": [],
                "widgets_values": ["publish-OUT"],
            },
            {
                "id": 169,
                "type": "Text Multiline",
                "title": "App : OUTPUT [ version ]",
                "pos": [520, 300],
                "size": [280, 80],
                "inputs": [],
                "widgets_values": ["1"],
            },
            {
                "id": 176,
                "type": "easy showAnything",
                "title": "App : OUTPUT [ result ]",
                "pos": [520, 420],
                "size": [280, 80],
                "inputs": [],
                "widgets_values": ["/shots/example/output/publish-OUT_v001.mov"],
            },
        ],
        "links": [],
        "groups": [
            {"title": "Koolook Input", "bounding": [20, 20, 360, 660]},
            {"title": "Koolook Output", "bounding": [500, 20, 360, 560]},
        ],
    }

    result = registry.publishSetup(
        visualGraph=visual_graph,
        metadata={
            "id": "app-contract-flow",
            "title": "App Contract Flow",
            "description": "A setup authored with App named controls.",
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
                "label": "sequence folder",
                "visible": True,
                "target": {"node": "3", "input": "text"},
                "default": "/shots/example/frames",
            },
            {
                "key": "qt_file",
                "label": "QT file",
                "visible": True,
                "target": {"node": "147", "input": "text"},
                "default": "/shots/example/movie.mov",
            },
            {
                "key": "single_file",
                "label": "single file",
                "visible": True,
                "target": {"node": "148", "input": "text"},
                "default": "/shots/example/image.png",
            },
            {
                "key": "prompt",
                "label": "prompt",
                "visible": False,
                "target": {"node": "149", "input": "text"},
                "default": "unused prompt",
            },
        ],
        "outputs": [
            {
                "key": "folder",
                "label": "folder",
                "visible": True,
                "target": {"node": "163", "input": "text"},
                "default": "/shots/example/output",
            },
            {
                "key": "name",
                "label": "name",
                "visible": True,
                "target": {"node": "168", "input": "text"},
                "default": "publish-OUT",
            },
            {
                "key": "version",
                "label": "version",
                "visible": True,
                "target": {"node": "169", "input": "text"},
                "default": "1",
            },
            {
                "key": "result",
                "label": "result",
                "visible": True,
                "target": {"node": "176", "input": "anything"},
                "default": "/shots/example/output/publish-OUT_v001.mov",
            },
        ],
        "switch": {
            "key": "switch",
            "label": "switch",
            "visible": True,
            "target": {"node": "182", "input": "value"},
            "default": 2,
            "options": [
                {"value": 0, "label": "EXR", "visible": True, "input": "sequence_folder"},
                {"value": 1, "label": "QT", "visible": True, "input": "qt_file"},
                {"value": 2, "label": "Img", "visible": True, "input": "single_file"},
                {"value": 3, "label": "Prompt", "visible": False, "input": "prompt"},
            ],
        },
    }


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
                {"id": 12, "type": "Text Multiline", "inputs": []},
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
            {"id": 12, "type": "Text Multiline", "inputs": []},
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
