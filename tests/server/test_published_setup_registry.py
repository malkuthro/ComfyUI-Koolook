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
