"""Published workflow setup registry boundary.

This module owns the server-side shape for externally callable Koolook
setups. Callers use this boundary instead of reading sidebar workflow or
snapshot storage directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any


DEFAULT_SETUPS_SUBDIR = "koolook-published-setups"
DEFAULT_SETUPS_FILENAME = "setups.json"
SAMPLE_SETUPS_PATH = (
    Path(__file__).resolve().parent / "web" / "published_setups_sample.json"
)
SUPPORTED_SCHEMA_VERSION = 1
SETUP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
VALIDATION_STATUSES = {"valid", "draft", "invalid"}
PUBLISH_INPUT_CLASS = "Koolook_PublishInput"
PUBLISH_OUTPUT_CLASS = "Koolook_PublishOutput"
PUBLISH_RESULT_CLASS = "Koolook_PublishResult"
PUBLISH_INPUT_FIELDS = (
    ("sequence_folder", "Sequence folder", True),
    ("qt_file", "QT file", True),
    ("single_file", "Single file", True),
    ("prompt", "Prompt", False),
)
PUBLISH_INPUT_MODES = (
    (0, "EXR", "sequence_folder"),
    (1, "QT", "qt_file"),
    (2, "Img", "single_file"),
    (3, "Prompt", "prompt"),
)
PUBLISH_OUTPUT_FIELDS = (
    ("folder", "Output folder", True),
    ("name", "Output name", True),
    ("version", "Version", True),
)
PUBLISH_RESULT_FIELDS = (
    ("result", "Result", True),
)
WIDGET_ONLY_INPUTS_BY_CLASS = {
    "Text Multiline": ("text",),
    "Koolook_PublishInput": ("mode", "sequence_folder", "qt_file", "single_file", "prompt"),
    "Koolook_PublishOutput": ("folder", "name", "version"),
    "Koolook_PublishResult": ("result",),
    "EasyAIPipeline": (
        "shot_duration",
        "seed_value",
        "instruction",
        "base_directory_path",
        "extension",
        "shot_name",
        "ai_method",
        "version",
        "disable_versioning",
        "enable_overwrite",
        "no_subfolders",
    ),
}
WIDGET_ONLY_INPUT_DEFAULTS = {
    "EasyAIPipeline": {
        "no_subfolders": False,
    },
}

REQUIRED_SETUP_KEYS = (
    "schemaVersion",
    "id",
    "version",
    "updatedAt",
    "metadata",
    "visualGraph",
    "apiPrompt",
    "inputContract",
    "outputContract",
    "source",
    "validation",
)


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    setup: dict[str, Any]
    diagnostics: list[str]


@dataclass(frozen=True)
class ApiPromptConversionResult:
    api_prompt: dict[str, Any] | None
    diagnostics: list[str]


class StaticSetupStorage:
    """In-memory setup storage adapter for tests and seeded fixtures."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records
        self.diagnostics: list[str] = []

    def load_setups(self) -> list[dict[str, Any]]:
        return list(self._records)

    def save_setups(self, records: list[dict[str, Any]]) -> None:
        self._records = list(records)


class FileSetupStorage:
    """JSON-file storage adapter for published setup records."""

    def __init__(self, path: Path, fallback_path: Path | None = None) -> None:
        self.path = path
        self.fallback_path = fallback_path
        self.diagnostics: list[str] = []

    def load_setups(self) -> list[dict[str, Any]]:
        self.diagnostics = []
        path = self.path
        if not path.is_file() and self.fallback_path is not None:
            path = self.fallback_path
            self.diagnostics.append(
                f"{self.path}: missing; using bundled sample {self.fallback_path}"
            )
        if not path.is_file():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.diagnostics.append(f"{path}: could not read published setups: {exc}")
            return []
        if isinstance(raw, dict):
            raw = raw.get("setups", [])
        if not isinstance(raw, list):
            self.diagnostics.append(
                f"{path}: published setup storage must be a list or {{setups: [...]}}"
            )
            return []
        return [item for item in raw if isinstance(item, dict)]

    def load_primary_setups(self) -> list[dict[str, Any]]:
        self.diagnostics = []
        if not self.path.is_file():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.diagnostics.append(f"{self.path}: could not read published setups: {exc}")
            return []
        if isinstance(raw, dict):
            raw = raw.get("setups", [])
        if not isinstance(raw, list):
            self.diagnostics.append(
                f"{self.path}: published setup storage must be a list or {{setups: [...]}}"
            )
            return []
        return [item for item in raw if isinstance(item, dict)]

    def save_setups(self, records: list[dict[str, Any]]) -> None:
        self.diagnostics = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"setups": records}
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f"{self.path.stem}.",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.write("\n")
            os.replace(tmp_name, self.path)
        except OSError:
            try:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
            except OSError:
                pass
            raise


class PublishedSetupRegistry:
    """Read-only registry facade for published setup catalog consumers."""

    def __init__(self, storage: StaticSetupStorage | FileSetupStorage) -> None:
        self._storage = storage
        self.diagnostics: list[str] = []

    def _valid_setups(self) -> list[dict[str, Any]]:
        self.diagnostics = []
        out: list[dict[str, Any]] = []
        for raw in self._storage.load_setups():
            result = validate_setup(raw)
            if result.valid:
                out.append(result.setup)
                continue
            setup_id = raw.get("id", "<unknown>") if isinstance(raw, dict) else "<unknown>"
            for diagnostic in result.diagnostics:
                self.diagnostics.append(f"{setup_id}: {diagnostic}")
        self.diagnostics.extend(getattr(self._storage, "diagnostics", []))
        out.sort(key=lambda setup: str(setup["metadata"]["title"]).lower())
        return out

    def listSetups(self) -> list[dict[str, Any]]:
        return [_catalog_summary(setup) for setup in self._valid_setups()]

    def getSetup(self, setup_id: str) -> dict[str, Any] | None:
        for setup in self._valid_setups():
            if setup["id"] == setup_id:
                return setup
        return None

    def publishSetup(
        self,
        *,
        visualGraph: Any,
        metadata: Any,
        inputContract: Any,
        outputContract: Any,
        source: Any,
    ) -> ValidationResult:
        diagnostics = []
        for name, value in (
            ("visualGraph", visualGraph),
            ("metadata", metadata),
            ("inputContract", inputContract),
            ("outputContract", outputContract),
            ("source", source),
        ):
            if not isinstance(value, dict):
                diagnostics.append(f"{name} must be an object")
        if diagnostics:
            return ValidationResult(False, {}, diagnostics)

        conversion = _convert_visual_graph_to_api_prompt(visualGraph)
        if conversion.diagnostics:
            return ValidationResult(False, {}, conversion.diagnostics)
        surface_geometry_diagnostics = []
        if _uses_group_authored_surface(inputContract, outputContract):
            surface_geometry_diagnostics = _validate_setup_surface_geometry(visualGraph)
        if surface_geometry_diagnostics:
            return ValidationResult(False, {}, surface_geometry_diagnostics)
        setup = _build_draft_setup(
            visualGraph=visualGraph,
            apiPrompt=conversion.api_prompt,
            metadata=metadata,
            inputContract=inputContract,
            outputContract=outputContract,
            source=source,
        )
        result = validate_setup(setup)
        diagnostics = list(result.diagnostics)
        diagnostics.extend(_validate_contract_targets(visualGraph, inputContract))
        diagnostics.extend(_validate_api_prompt_contract_targets(conversion.api_prompt, inputContract))
        diagnostics.extend(_validate_setup_surface(setup, inputContract, outputContract))
        if diagnostics:
            return ValidationResult(False, {}, diagnostics)

        load_for_write = getattr(self._storage, "load_primary_setups", self._storage.load_setups)
        records = [
            existing
            for existing in load_for_write()
            if existing.get("id") != setup["id"]
        ]
        records.append(setup)
        self._storage.save_setups(records)
        return ValidationResult(True, setup, [])


def validate_setup(value: Any) -> ValidationResult:
    """Validate one published setup object through the public schema boundary."""
    diagnostics: list[str] = []
    if not isinstance(value, dict):
        return ValidationResult(False, {}, ["setup must be a JSON object"])

    for key in REQUIRED_SETUP_KEYS:
        if key not in value:
            diagnostics.append(f"missing required field: {key}")

    if diagnostics:
        return ValidationResult(False, {}, diagnostics)

    _validate_scalar(value, "id", str, diagnostics)
    if isinstance(value.get("id"), str) and not SETUP_ID_RE.match(value["id"]):
        diagnostics.append("id must be stable URL-safe text")
    _validate_scalar(value, "schemaVersion", int, diagnostics)
    if value.get("schemaVersion") != SUPPORTED_SCHEMA_VERSION:
        diagnostics.append(
            f"unsupported schemaVersion: {value.get('schemaVersion')}; "
            f"expected {SUPPORTED_SCHEMA_VERSION}"
        )
    _validate_scalar(value, "version", (int, str), diagnostics)
    _validate_iso_timestamp(value.get("updatedAt"), diagnostics)
    _validate_metadata(value.get("metadata"), diagnostics)
    _validate_mapping(value.get("visualGraph"), "visualGraph", diagnostics)
    if value.get("apiPrompt") is not None:
        _validate_mapping(value.get("apiPrompt"), "apiPrompt", diagnostics)
        if isinstance(value.get("visualGraph"), dict) and isinstance(value.get("apiPrompt"), dict):
            conversion = _convert_visual_graph_to_api_prompt(value["visualGraph"])
            if conversion.diagnostics:
                diagnostics.extend(conversion.diagnostics)
            elif conversion.api_prompt != value["apiPrompt"]:
                diagnostics.append("apiPrompt is stale for visualGraph")
    _validate_contract(value.get("inputContract"), "inputContract", "inputs", diagnostics)
    _validate_contract(value.get("outputContract"), "outputContract", "outputs", diagnostics)
    _validate_persisted_setup_surface(value, diagnostics)
    _validate_source(value.get("source"), diagnostics)
    _validate_validation(value.get("validation"), diagnostics)
    if (
        isinstance(value.get("validation"), dict)
        and value["validation"].get("status") == "valid"
        and value.get("apiPrompt") is None
    ):
        diagnostics.append("validation.status valid requires apiPrompt")

    return ValidationResult(
        valid=not diagnostics,
        setup=value if not diagnostics else {},
        diagnostics=diagnostics,
    )


def _build_draft_setup(
    *,
    visualGraph: dict[str, Any],
    apiPrompt: dict[str, Any] | None,
    metadata: dict[str, Any],
    inputContract: dict[str, Any],
    outputContract: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    title = metadata.get("title", "")
    setup_id = metadata.get("id", "")
    setup_surface = _infer_setup_surface(visualGraph)
    return {
        "schemaVersion": SUPPORTED_SCHEMA_VERSION,
        "id": setup_id,
        "version": 1,
        "updatedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "metadata": {
            "title": title,
            "description": metadata.get("description", ""),
            "category": metadata.get("category", ""),
            "tags": metadata.get("tags", []),
            "previewImage": metadata.get("previewImage", ""),
        },
        "visualGraph": visualGraph,
        "apiPrompt": apiPrompt,
        "inputContract": inputContract,
        "outputContract": outputContract,
        "setupSurface": setup_surface,
        "source": source,
        "validation": {
            "status": "valid" if apiPrompt is not None else "draft",
            "diagnostics": [] if apiPrompt is not None else ["API prompt conversion pending."],
        },
    }


def _infer_setup_surface(visual_graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "sourceInputs": _nodes_in_named_groups(visual_graph, "Koolook Input"),
        "outputs": _nodes_in_named_groups(visual_graph, "Koolook Output"),
        "controls": [],
        "app": _infer_app_surface(visual_graph),
    }


def _infer_app_surface(visual_graph: dict[str, Any]) -> dict[str, Any]:
    input_nodes = _nodes_in_group(visual_graph, "Koolook Input")
    output_nodes = _nodes_in_group(visual_graph, "Koolook Output")
    input_node = _first_node_of_type(input_nodes, PUBLISH_INPUT_CLASS)
    output_node = _first_node_of_type(output_nodes, PUBLISH_OUTPUT_CLASS)
    result_node = _first_node_of_type(output_nodes, PUBLISH_RESULT_CLASS)
    inputs = _publish_input_fields(input_node)
    outputs = _publish_output_fields(output_node)
    app: dict[str, Any] = {
        "inputs": inputs,
        "outputs": outputs,
        "results": _publish_result_fields(result_node),
    }
    if input_node is not None:
        app["switch"] = _publish_input_switch(input_node, inputs)
    return app


def _first_node_of_type(
    nodes: list[dict[str, Any]],
    class_type: str,
) -> dict[str, Any] | None:
    for node in nodes:
        if node.get("type") == class_type:
            return node
    return None


def _publish_input_fields(node: dict[str, Any] | None) -> list[dict[str, Any]]:
    if node is None:
        return []
    return [
        {
            "key": key,
            "label": label,
            "visible": visible,
            "target": {
                "node": str(node.get("id")),
                "input": key,
            },
            "default": _app_widget_value(node, key),
        }
        for key, label, visible in PUBLISH_INPUT_FIELDS
    ]


def _publish_output_fields(node: dict[str, Any] | None) -> list[dict[str, Any]]:
    if node is None:
        return []
    return [
        {
            "key": key,
            "label": label,
            "visible": visible,
            "target": {
                "node": str(node.get("id")),
                "input": key,
            },
            "default": _app_widget_value(node, key),
        }
        for key, label, visible in PUBLISH_OUTPUT_FIELDS
    ]


def _publish_result_fields(node: dict[str, Any] | None) -> list[dict[str, Any]]:
    if node is None:
        return []
    return [
        {
            "key": key,
            "label": label,
            "visible": visible,
            "target": {
                "node": str(node.get("id")),
                "input": key,
            },
            "default": _app_widget_value(node, key),
        }
        for key, label, visible in PUBLISH_RESULT_FIELDS
    ]


def _publish_input_switch(
    node: dict[str, Any],
    inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    inputs_by_key = {item["key"]: item for item in inputs}
    return {
        "key": "switch",
        "label": "Input type",
        "visible": True,
        "target": {"node": str(node.get("id")), "input": "mode"},
        "default": _publish_input_mode_index(_app_widget_value(node, "mode")),
        "options": [
            {
                "value": value,
                "label": label,
                "visible": bool(inputs_by_key[input_key].get("visible")),
                "input": input_key,
            }
            for value, label, input_key in PUBLISH_INPUT_MODES
        ],
    }


def _publish_input_mode_index(value: Any) -> int:
    if isinstance(value, int):
        return value
    for mode_value, label, _input_key in PUBLISH_INPUT_MODES:
        if str(value).strip().lower() == label.lower():
            return mode_value
    return 2


def _app_widget_value(node: dict[str, Any], key: str) -> Any:
    widget_values = node.get("widgets_values")
    if isinstance(widget_values, dict):
        return widget_values.get(key)
    if isinstance(widget_values, list):
        widget_names = WIDGET_ONLY_INPUTS_BY_CLASS.get(str(node.get("type")), ())
        for index, name in enumerate(widget_names):
            if name == key and index < len(widget_values):
                return widget_values[index]
    return None


def _validate_setup_surface(
    setup: dict[str, Any],
    input_contract: dict[str, Any],
    output_contract: dict[str, Any],
) -> list[str]:
    if not _uses_group_authored_surface(input_contract, output_contract):
        return []
    surface = setup.get("setupSurface")
    diagnostics: list[str] = []
    if not isinstance(surface, dict) or not surface.get("sourceInputs"):
        diagnostics.append("setupSurface.sourceInputs requires a non-empty Koolook Input group")
    if not isinstance(surface, dict) or not surface.get("outputs"):
        diagnostics.append("setupSurface.outputs requires a non-empty Koolook Output group")
    return diagnostics


def _validate_persisted_setup_surface(
    setup: dict[str, Any],
    diagnostics: list[str],
) -> None:
    surface = setup.get("setupSurface")
    input_contract = setup.get("inputContract")
    output_contract = setup.get("outputContract")
    group_authored = (
        isinstance(input_contract, dict)
        and isinstance(output_contract, dict)
        and _uses_group_authored_surface(input_contract, output_contract)
    )
    if surface is None:
        if group_authored:
            diagnostics.append("setupSurface must be a JSON object for group-authored setups")
        return
    if not isinstance(surface, dict):
        diagnostics.append("setupSurface must be a JSON object")
        return
    for key in ("sourceInputs", "outputs", "controls"):
        if key in surface and not isinstance(surface[key], list):
            diagnostics.append(f"setupSurface.{key} must be a list")
    if group_authored:
        if not isinstance(surface.get("sourceInputs"), list) or not surface["sourceInputs"]:
            diagnostics.append("setupSurface.sourceInputs requires a non-empty Koolook Input group")
        if not isinstance(surface.get("outputs"), list) or not surface["outputs"]:
            diagnostics.append("setupSurface.outputs requires a non-empty Koolook Output group")
    for key in ("sourceInputs", "outputs", "controls"):
        items = surface.get(key)
        if isinstance(items, list):
            _validate_surface_entries(items, f"setupSurface.{key}", diagnostics)
    if "app" in surface:
        _validate_surface_app(
            surface["app"],
            "setupSurface.app",
            diagnostics,
            setup.get("visualGraph"),
            setup.get("apiPrompt"),
        )


def _validate_surface_entries(
    entries: list[Any],
    path: str,
    diagnostics: list[str],
) -> None:
    for entry_index, entry in enumerate(entries):
        entry_path = f"{path}[{entry_index}]"
        if not isinstance(entry, dict):
            diagnostics.append(f"{entry_path} must be a JSON object")
            continue
        if not isinstance(entry.get("group"), str) or not entry["group"].strip():
            diagnostics.append(f"{entry_path}.group must be non-empty text")
        nodes = entry.get("nodes")
        if not isinstance(nodes, list):
            diagnostics.append(f"{entry_path}.nodes must be a list")
            continue
        for node_index, node in enumerate(nodes):
            node_path = f"{entry_path}.nodes[{node_index}]"
            if not isinstance(node, dict):
                diagnostics.append(f"{node_path} must be a JSON object")
                continue
            for field in ("id", "type", "title"):
                if not isinstance(node.get(field), str) or not node[field].strip():
                    diagnostics.append(f"{node_path}.{field} must be non-empty text")


def _validate_surface_app(
    app: Any,
    path: str,
    diagnostics: list[str],
    visual_graph: Any,
    api_prompt: Any,
) -> None:
    if not isinstance(app, dict):
        diagnostics.append(f"{path} must be a JSON object")
        return
    node_by_id = _visual_node_index(visual_graph)
    api_prompt = api_prompt if isinstance(api_prompt, dict) else None
    input_keys: set[str] = set()
    for key in ("inputs", "outputs", "results"):
        value = app.get(key)
        if value is None:
            continue
        if not isinstance(value, list):
            diagnostics.append(f"{path}.{key} must be a list")
            continue
        keys = _validate_surface_app_fields(
            value,
            f"{path}.{key}",
            diagnostics,
            node_by_id,
            api_prompt,
        )
        if key == "inputs":
            input_keys = keys
    if "switch" in app:
        _validate_surface_app_switch(
            app["switch"],
            f"{path}.switch",
            diagnostics,
            node_by_id,
            api_prompt,
            input_keys,
        )


def _validate_surface_app_fields(
    fields: list[Any],
    path: str,
    diagnostics: list[str],
    node_by_id: dict[str, dict[str, Any]],
    api_prompt: dict[str, Any] | None,
) -> set[str]:
    keys: set[str] = set()
    for field_index, field in enumerate(fields):
        field_path = f"{path}[{field_index}]"
        if not isinstance(field, dict):
            diagnostics.append(f"{field_path} must be a JSON object")
            continue
        _validate_non_empty_text(field.get("key"), f"{field_path}.key", diagnostics)
        if isinstance(field.get("key"), str) and field["key"].strip():
            keys.add(field["key"])
        _validate_non_empty_text(field.get("label"), f"{field_path}.label", diagnostics)
        if "visible" in field and not isinstance(field["visible"], bool):
            diagnostics.append(f"{field_path}.visible must be a boolean")
        _validate_surface_app_target(
            field.get("target"),
            f"{field_path}.target",
            diagnostics,
            node_by_id,
            api_prompt,
        )
    return keys


def _validate_surface_app_switch(
    switch: Any,
    path: str,
    diagnostics: list[str],
    node_by_id: dict[str, dict[str, Any]],
    api_prompt: dict[str, Any] | None,
    input_keys: set[str],
) -> None:
    if not isinstance(switch, dict):
        diagnostics.append(f"{path} must be a JSON object")
        return
    _validate_non_empty_text(switch.get("key"), f"{path}.key", diagnostics)
    _validate_non_empty_text(switch.get("label"), f"{path}.label", diagnostics)
    if "visible" in switch and not isinstance(switch["visible"], bool):
        diagnostics.append(f"{path}.visible must be a boolean")
    _validate_surface_app_target(
        switch.get("target"),
        f"{path}.target",
        diagnostics,
        node_by_id,
        api_prompt,
    )
    default = switch.get("default")
    if default is not None and (isinstance(default, bool) or not isinstance(default, int)):
        diagnostics.append(f"{path}.default must be an integer")
    options = switch.get("options")
    if not isinstance(options, list):
        diagnostics.append(f"{path}.options must be a list")
        return
    for option_index, option in enumerate(options):
        option_path = f"{path}.options[{option_index}]"
        if not isinstance(option, dict):
            diagnostics.append(f"{option_path} must be a JSON object")
            continue
        if isinstance(option.get("value"), bool) or not isinstance(option.get("value"), int):
            diagnostics.append(f"{option_path}.value must be an integer")
        _validate_non_empty_text(option.get("label"), f"{option_path}.label", diagnostics)
        _validate_non_empty_text(option.get("input"), f"{option_path}.input", diagnostics)
        if (
            isinstance(option.get("input"), str)
            and option["input"].strip()
            and input_keys
            and option["input"] not in input_keys
        ):
            diagnostics.append(f"{option_path}.input must match a setupSurface.app.inputs key")
        if "visible" in option and not isinstance(option["visible"], bool):
            diagnostics.append(f"{option_path}.visible must be a boolean")


def _validate_surface_app_target(
    target: Any,
    path: str,
    diagnostics: list[str],
    node_by_id: dict[str, dict[str, Any]],
    api_prompt: dict[str, Any] | None,
) -> None:
    if not isinstance(target, dict):
        diagnostics.append(f"{path} must be a JSON object")
        return
    _validate_non_empty_text(target.get("node"), f"{path}.node", diagnostics)
    _validate_non_empty_text(target.get("input"), f"{path}.input", diagnostics)
    node_id = target.get("node")
    input_name = target.get("input")
    if not isinstance(node_id, str) or not node_id.strip():
        return
    if not isinstance(input_name, str) or not input_name.strip():
        return
    node = node_by_id.get(node_id)
    if node is None:
        diagnostics.append(f"{path}.node not found in visualGraph")
        return
    if not _visual_node_has_input(node, input_name):
        diagnostics.append(f"{path}.input not found in visualGraph node")
    if api_prompt is None:
        return
    api_node = api_prompt.get(node_id)
    if not isinstance(api_node, dict):
        diagnostics.append(f"{path}.node not found in apiPrompt")
        return
    api_inputs = api_node.get("inputs")
    if not isinstance(api_inputs, dict) or input_name not in api_inputs:
        diagnostics.append(f"{path}.input not found in apiPrompt")


def _validate_non_empty_text(value: Any, path: str, diagnostics: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        diagnostics.append(f"{path} must be non-empty text")


def _visual_node_index(visual_graph: Any) -> dict[str, dict[str, Any]]:
    nodes = visual_graph.get("nodes") if isinstance(visual_graph, dict) else None
    if not isinstance(nodes, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if node_id is not None:
            out[str(node_id)] = node
    return out


def _visual_node_has_input(node: dict[str, Any], input_name: str) -> bool:
    inputs = node.get("inputs")
    if isinstance(inputs, list) and any(
        isinstance(port, dict) and str(port.get("name", "")) == input_name
        for port in inputs
    ):
        return True
    class_type = node.get("type")
    widget_names = WIDGET_ONLY_INPUTS_BY_CLASS.get(str(class_type), ())
    return input_name in widget_names


def _uses_group_authored_surface(
    input_contract: dict[str, Any],
    output_contract: dict[str, Any],
) -> bool:
    inputs = input_contract.get("inputs")
    outputs = output_contract.get("outputs")
    return inputs == [] and outputs == []


def _nodes_in_named_groups(visual_graph: dict[str, Any], group_title: str) -> list[dict[str, Any]]:
    groups = _nodes_in_group_sections(visual_graph, group_title)
    return [
        {"group": group_title, "nodes": [_surface_node_summary(node) for node in group_nodes]}
        for group_nodes in groups
        if group_nodes
    ]


def _nodes_in_group(visual_graph: dict[str, Any], group_title: str) -> list[dict[str, Any]]:
    groups = _nodes_in_group_sections(visual_graph, group_title)
    return [node for group_nodes in groups for node in group_nodes]


def _nodes_in_group_sections(
    visual_graph: dict[str, Any],
    group_title: str,
) -> list[list[dict[str, Any]]]:
    groups = visual_graph.get("groups")
    nodes = visual_graph.get("nodes")
    if not isinstance(groups, list) or not isinstance(nodes, list):
        return []

    node_rects = [
        (node, _rect_from_node(node))
        for node in nodes
        if isinstance(node, dict)
    ]
    out: list[list[dict[str, Any]]] = []
    for group in groups:
        if not isinstance(group, dict) or group.get("title") != group_title:
            continue
        group_rect = _rect_from_group(group)
        group_nodes = [
            node
            for node, node_rect in node_rects
            if _rects_overlap(group_rect, node_rect)
        ]
        if group_nodes:
            out.append(sorted(group_nodes, key=lambda node: (_rect_from_node(node)["y"], _rect_from_node(node)["x"])))
    return out


def _validate_setup_surface_geometry(visual_graph: dict[str, Any]) -> list[str]:
    diagnostics: list[str] = []
    groups = visual_graph.get("groups")
    if isinstance(groups, list):
        for group_index, group in enumerate(groups):
            if not isinstance(group, dict) or group.get("title") not in {"Koolook Input", "Koolook Output"}:
                continue
            bounding = group.get("bounding")
            pos = group.get("pos")
            size = group.get("size")
            if bounding is not None:
                if (
                    not isinstance(bounding, list)
                    or len(bounding) < 4
                    or not all(_is_number(value) for value in bounding[:4])
                ):
                    diagnostics.append(
                        f"visualGraph.groups[{group_index}].bounding must contain numeric "
                        "x, y, width, height"
                    )
                continue
            if pos is not None or size is not None:
                if (
                    not isinstance(pos, list)
                    or len(pos) < 2
                    or not all(_is_number(value) for value in pos[:2])
                    or not isinstance(size, list)
                    or len(size) < 2
                    or not all(_is_number(value) for value in size[:2])
                ):
                    diagnostics.append(
                        f"visualGraph.groups[{group_index}].pos/size must contain numeric "
                        "x, y, width, height"
                    )
                continue
            diagnostics.append(
                f"visualGraph.groups[{group_index}] must define bounding or pos/size "
                "for setup surface inference"
            )
    nodes = visual_graph.get("nodes")
    if isinstance(nodes, list):
        for node_index, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue
            pos = node.get("pos")
            if pos is not None and (
                not isinstance(pos, list)
                or len(pos) < 2
                or not all(_is_number(value) for value in pos[:2])
            ):
                diagnostics.append(
                    f"visualGraph.nodes[{node_index}].pos must contain numeric x and y "
                    "for setup surface inference"
                )
            size = node.get("size")
            if size is not None and (
                not isinstance(size, list)
                or len(size) < 2
                or not all(_is_number(value) for value in size[:2])
            ):
                diagnostics.append(
                    f"visualGraph.nodes[{node_index}].size must contain numeric width and height "
                    "for setup surface inference"
                )
    return diagnostics


def _surface_node_summary(node: dict[str, Any]) -> dict[str, str]:
    node_type = node.get("type")
    node_title = node.get("title")
    return {
        "id": str(node.get("id")),
        "type": node_type if isinstance(node_type, str) else "",
        "title": node_title if isinstance(node_title, str) and node_title else (
            node_type if isinstance(node_type, str) else str(node.get("id"))
        ),
    }


def _rect_from_group(group: dict[str, Any]) -> dict[str, float] | None:
    bounding = group.get("bounding")
    if isinstance(bounding, list) and len(bounding) >= 4:
        return {
            "x": _number_or_default(bounding[0], 0),
            "y": _number_or_default(bounding[1], 0),
            "w": _number_or_default(bounding[2], 0),
            "h": _number_or_default(bounding[3], 0),
        }
    pos = group.get("pos")
    size = group.get("size")
    if isinstance(pos, list) and isinstance(size, list) and len(pos) >= 2 and len(size) >= 2:
        return {
            "x": _number_or_default(pos[0], 0),
            "y": _number_or_default(pos[1], 0),
            "w": _number_or_default(size[0], 0),
            "h": _number_or_default(size[1], 0),
        }
    return None


def _rect_from_node(node: dict[str, Any]) -> dict[str, float] | None:
    pos = node.get("pos")
    if not isinstance(pos, list) or len(pos) < 2:
        return None
    size = node.get("size")
    if not isinstance(size, list) or len(size) < 2:
        size = [200, 100]
    return {
        "x": _number_or_default(pos[0], 0),
        "y": _number_or_default(pos[1], 0),
        "w": _number_or_default(size[0], 200),
        "h": _number_or_default(size[1], 100),
    }


def _rects_overlap(a: dict[str, float] | None, b: dict[str, float] | None) -> bool:
    if not a or not b or a["w"] <= 0 or a["h"] <= 0 or b["w"] <= 0 or b["h"] <= 0:
        return False
    return (
        a["x"] < b["x"] + b["w"]
        and a["x"] + a["w"] > b["x"]
        and a["y"] < b["y"] + b["h"]
        and a["y"] + a["h"] > b["y"]
    )


def _number_or_default(value: Any, default: float) -> float:
    return value if isinstance(value, int | float) else default


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)


def _convert_visual_graph_to_api_prompt(visual_graph: dict[str, Any]) -> ApiPromptConversionResult:
    nodes = visual_graph.get("nodes")
    if not isinstance(nodes, list):
        return ApiPromptConversionResult(None, ["visualGraph.nodes must be a list"])
    links = _visual_link_index(visual_graph.get("links"))
    if links.diagnostics:
        return ApiPromptConversionResult(None, links.diagnostics)

    node_indexes_by_id: dict[str, int] = {}
    node_ids = set[str]()
    for node_index, node in enumerate(nodes):
        if not isinstance(node, dict):
            return ApiPromptConversionResult(None, [f"visualGraph.nodes[{node_index}] must be an object"])
        node_id = node.get("id")
        class_type = node.get("type")
        node_inputs = node.get("inputs")
        if node_id is None or not isinstance(class_type, str) or not class_type.strip():
            return ApiPromptConversionResult(
                None,
                [f"visualGraph.nodes[{node_index}].type must be non-empty text"],
            )
        node_id_key = str(node_id)
        previous_node_index = node_indexes_by_id.get(node_id_key)
        if previous_node_index is not None:
            return ApiPromptConversionResult(
                None,
                [
                    f"visualGraph.nodes[{node_index}].id duplicates "
                    f"visualGraph.nodes[{previous_node_index}].id"
                ],
            )
        node_indexes_by_id[node_id_key] = node_index
        node_ids.add(node_id_key)

    api_prompt: dict[str, Any] = {}
    for node_index, node in enumerate(nodes):
        node_id_key = str(node["id"])
        class_type = node["type"]
        node_inputs = node.get("inputs")
        if node_inputs is None:
            node_inputs = []
        elif not isinstance(node_inputs, list):
            return ApiPromptConversionResult(
                None,
                [f"visualGraph.nodes[{node_index}].inputs must be a list when present"],
            )

        raw_widget_values = node.get("widgets_values")
        widget_values_by_name = raw_widget_values if isinstance(raw_widget_values, dict) else {}
        widget_values = raw_widget_values if isinstance(raw_widget_values, list) else []
        if not widget_values and not widget_values_by_name:
            widget_values = _proxy_widget_values_from_subgraph(visual_graph, node)
        widget_index = 0
        api_inputs: dict[str, Any] = {}
        for input_index, input_port in enumerate(node_inputs):
            if not isinstance(input_port, dict):
                return ApiPromptConversionResult(
                    None,
                    [f"visualGraph.nodes[{node_index}].inputs must contain objects"],
                )
            name = input_port.get("name")
            if not isinstance(name, str) or not name.strip():
                if class_type != "Reroute" or not isinstance(name, str):
                    return ApiPromptConversionResult(
                        None,
                        [f"visualGraph.nodes[{node_index}].inputs must have named ports"],
                    )
            link_id = input_port.get("link")
            if link_id is not None:
                link = links.links.get(str(link_id))
                if link is None:
                    return ApiPromptConversionResult(
                        None,
                        [f"visualGraph.nodes[{node_index}].inputs.{name} references missing link"],
                    )
                if _is_module_sentinel_id(link["origin_id"]):
                    return ApiPromptConversionResult(
                        None,
                        [
                            f"visualGraph.links[{link_id}] uses unsupported module graph sentinel node"
                        ],
                    )
                if str(link["origin_id"]) not in node_ids:
                    return ApiPromptConversionResult(
                        None,
                        [f"visualGraph.links[{link_id}].origin_id not found in visualGraph"],
                    )
                if not _is_non_negative_int(link["origin_slot"]):
                    return ApiPromptConversionResult(
                        None,
                        [
                            f"visualGraph.links[{link_id}].origin_slot must be "
                            "a non-negative integer"
                        ],
                    )
                if link["target_id"] is not None and str(link["target_id"]) != node_id_key:
                    return ApiPromptConversionResult(
                        None,
                        [
                            f"visualGraph.links[{link_id}].target does not match "
                            f"visualGraph.nodes[{node_index}].inputs[{input_index}]"
                        ],
                    )
                if (
                    link["target_slot"] is not None
                    and (
                        not _is_non_negative_int(link["target_slot"])
                        or int(link["target_slot"]) != input_index
                    )
                ):
                    return ApiPromptConversionResult(
                        None,
                        [
                            f"visualGraph.links[{link_id}].target does not match "
                            f"visualGraph.nodes[{node_index}].inputs[{input_index}]"
                        ],
                    )
                api_inputs[name] = [str(link["origin_id"]), int(link["origin_slot"])]
                continue
            widget = input_port.get("widget")
            if isinstance(widget, dict):
                if name in widget_values_by_name:
                    api_inputs[name] = widget_values_by_name[name]
                    continue
                if widget_index >= len(widget_values):
                    return ApiPromptConversionResult(
                        None,
                        [
                            f"visualGraph.nodes[{node_index}].widgets_values is missing a value "
                            f"for input {name}"
                        ],
                    )
                api_inputs[name] = widget_values[widget_index]
                widget_index += 1

        if not node_inputs:
            widget_only_diagnostics = _apply_widget_only_inputs(
                class_type,
                widget_values,
                api_inputs,
                node_index,
            )
            if widget_only_diagnostics:
                return ApiPromptConversionResult(None, widget_only_diagnostics)
        api_prompt[node_id_key] = {"class_type": class_type, "inputs": api_inputs}

    return ApiPromptConversionResult(api_prompt, [])


def _apply_widget_only_inputs(
    class_type: str,
    widget_values: list[Any],
    api_inputs: dict[str, Any],
    node_index: int,
) -> list[str]:
    if api_inputs:
        return []
    widget_names = WIDGET_ONLY_INPUTS_BY_CLASS.get(class_type)
    if not widget_names:
        return []
    defaults = WIDGET_ONLY_INPUT_DEFAULTS.get(class_type, {})
    diagnostics: list[str] = []
    for index, name in enumerate(widget_names):
        if index < len(widget_values):
            value = widget_values[index]
            if name == "no_subfolders" and not isinstance(value, bool):
                value = defaults.get(name, False)
            api_inputs[name] = value
        elif name in defaults:
            api_inputs[name] = defaults[name]
        else:
            diagnostics.append(
                f"visualGraph.nodes[{node_index}].widgets_values is missing a value "
                f"for widget-only input {name}"
            )
    return diagnostics


def _proxy_widget_values_from_subgraph(
    visual_graph: dict[str, Any],
    node: dict[str, Any],
) -> list[Any]:
    properties = node.get("properties")
    if not isinstance(properties, dict):
        return []
    proxy_widgets = properties.get("proxyWidgets")
    if not isinstance(proxy_widgets, list):
        return []
    subgraph = _subgraph_definition(visual_graph, str(node.get("type", "")))
    if subgraph is None:
        return []
    subgraph_nodes = subgraph.get("nodes")
    if not isinstance(subgraph_nodes, list):
        return []

    nodes_by_id = {
        str(subgraph_node.get("id")): subgraph_node
        for subgraph_node in subgraph_nodes
        if isinstance(subgraph_node, dict) and subgraph_node.get("id") is not None
    }
    values: list[Any] = []
    for proxy_widget in proxy_widgets:
        if not isinstance(proxy_widget, list) or len(proxy_widget) < 2:
            continue
        source_node = nodes_by_id.get(str(proxy_widget[0]))
        widget_name = proxy_widget[1]
        if not isinstance(source_node, dict) or not isinstance(widget_name, str):
            continue
        found, value = _node_widget_value(source_node, widget_name)
        if found:
            values.append(value)
    return values


def _subgraph_definition(
    visual_graph: dict[str, Any],
    subgraph_id: str,
) -> dict[str, Any] | None:
    definitions = visual_graph.get("definitions")
    if not isinstance(definitions, dict):
        return None
    subgraphs = definitions.get("subgraphs")
    if not isinstance(subgraphs, list):
        return None
    for subgraph in subgraphs:
        if isinstance(subgraph, dict) and subgraph.get("id") == subgraph_id:
            return subgraph
    return None


def _node_widget_value(
    node: dict[str, Any],
    widget_name: str,
) -> tuple[bool, Any]:
    inputs = node.get("inputs")
    widget_values = node.get("widgets_values")
    if not isinstance(inputs, list) or not isinstance(widget_values, list):
        return False, None

    widget_index = 0
    for input_port in inputs:
        if not isinstance(input_port, dict):
            continue
        widget = input_port.get("widget")
        if not isinstance(widget, dict):
            continue
        if widget.get("name") == widget_name:
            if widget_index >= len(widget_values):
                return False, None
            return True, widget_values[widget_index]
        widget_index += 1
    return False, None


@dataclass(frozen=True)
class VisualLinkIndex:
    links: dict[str, dict[str, Any]]
    diagnostics: list[str]


def _visual_link_index(value: Any) -> VisualLinkIndex:
    links: dict[str, dict[str, Any]] = {}
    if not isinstance(value, list):
        return VisualLinkIndex(links, [])
    link_indexes_by_id: dict[str, int] = {}
    for link_index, item in enumerate(value):
        if isinstance(item, dict):
            link_id = item.get("id")
            origin_id = item.get("origin_id")
            origin_slot = item.get("origin_slot")
            target_id = item.get("target_id")
            target_slot = item.get("target_slot")
        elif isinstance(item, list) and len(item) >= 5:
            link_id = item[0]
            origin_id = item[1]
            origin_slot = item[2]
            target_id = item[3]
            target_slot = item[4]
        elif isinstance(item, list) and len(item) >= 3:
            link_id = item[0]
            origin_id = item[1]
            origin_slot = item[2]
            target_id = None
            target_slot = None
        else:
            continue
        if link_id is None or origin_id is None or origin_slot is None:
            continue
        link_id_key = str(link_id)
        previous_link_index = link_indexes_by_id.get(link_id_key)
        if previous_link_index is not None:
            return VisualLinkIndex(
                links,
                [
                    f"visualGraph.links[{link_index}].id duplicates "
                    f"visualGraph.links[{previous_link_index}].id"
                ],
            )
        link_indexes_by_id[link_id_key] = link_index
        links[link_id_key] = {
            "origin_id": origin_id,
            "origin_slot": origin_slot,
            "target_id": target_id,
            "target_slot": target_slot,
        }
    return VisualLinkIndex(links, [])


def _is_module_sentinel_id(value: Any) -> bool:
    try:
        return int(value) < 0
    except (TypeError, ValueError):
        return False


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and value >= 0


def _validate_contract_targets(
    visual_graph: dict[str, Any],
    input_contract: dict[str, Any],
) -> list[str]:
    diagnostics: list[str] = []
    nodes = visual_graph.get("nodes") if isinstance(visual_graph, dict) else None
    node_by_id: dict[str, dict[str, Any]] = {}
    if isinstance(nodes, list):
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = node.get("id")
            if node_id is not None:
                node_by_id[str(node_id)] = node
    inputs = input_contract.get("inputs") if isinstance(input_contract, dict) else None
    if not isinstance(inputs, list):
        return diagnostics
    for index, item in enumerate(inputs):
        if not isinstance(item, dict):
            continue
        target = item.get("target")
        if not isinstance(target, dict):
            continue
        node_id = str(target.get("node", ""))
        input_name = str(target.get("input", ""))
        node = node_by_id.get(node_id)
        if node is None:
            diagnostics.append(
                f"inputContract.inputs[{index}].target.node not found in visualGraph"
            )
            continue
        node_inputs = node.get("inputs")
        if not isinstance(node_inputs, list):
            continue
        if not any(
            isinstance(port, dict) and str(port.get("name", "")) == input_name
            for port in node_inputs
        ):
            diagnostics.append(
                f"inputContract.inputs[{index}].target.input not found in visualGraph node"
            )
    return diagnostics


def _validate_api_prompt_contract_targets(
    api_prompt: dict[str, Any] | None,
    input_contract: dict[str, Any],
) -> list[str]:
    diagnostics: list[str] = []
    if api_prompt is None:
        return diagnostics
    inputs = input_contract.get("inputs") if isinstance(input_contract, dict) else None
    if not isinstance(inputs, list):
        return diagnostics
    for index, item in enumerate(inputs):
        if not isinstance(item, dict):
            continue
        target = item.get("target")
        if not isinstance(target, dict):
            continue
        node_id = str(target.get("node", ""))
        input_name = str(target.get("input", ""))
        api_node = api_prompt.get(node_id)
        if not isinstance(api_node, dict):
            diagnostics.append(
                f"inputContract.inputs[{index}].target.node not found in generated apiPrompt"
            )
            continue
        api_inputs = api_node.get("inputs")
        if not isinstance(api_inputs, dict) or input_name not in api_inputs:
            diagnostics.append(
                f"inputContract.inputs[{index}].target.input not found in generated apiPrompt"
            )
    return diagnostics


def default_storage_path() -> Path:
    try:
        import folder_paths

        user_dir = folder_paths.get_user_directory()
        return Path(user_dir) / DEFAULT_SETUPS_SUBDIR / DEFAULT_SETUPS_FILENAME
    except Exception:  # pragma: no cover - fallback for non-Comfy contexts
        return Path.cwd() / "user" / "default" / DEFAULT_SETUPS_SUBDIR / DEFAULT_SETUPS_FILENAME


def default_registry() -> PublishedSetupRegistry:
    return PublishedSetupRegistry(
        FileSetupStorage(default_storage_path(), fallback_path=SAMPLE_SETUPS_PATH)
    )


def _validate_scalar(
    mapping: dict[str, Any],
    key: str,
    expected_type: type | tuple[type, ...],
    diagnostics: list[str],
) -> None:
    if not isinstance(mapping.get(key), expected_type):
        diagnostics.append(f"{key} has invalid type")


def _validate_mapping(value: Any, path: str, diagnostics: list[str]) -> None:
    if not isinstance(value, dict):
        diagnostics.append(f"{path} must be a JSON object")


def _validate_iso_timestamp(value: Any, diagnostics: list[str]) -> None:
    if not isinstance(value, str):
        diagnostics.append("updatedAt must be an ISO timestamp")
        return
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        datetime.fromisoformat(normalized)
    except ValueError:
        diagnostics.append("updatedAt must be an ISO timestamp")


def _validate_metadata(value: Any, diagnostics: list[str]) -> None:
    if not isinstance(value, dict):
        diagnostics.append("metadata must be a JSON object")
        return
    if not isinstance(value.get("title"), str) or not value["title"].strip():
        diagnostics.append("metadata.title must be non-empty text")
    if "description" in value and not isinstance(value["description"], str):
        diagnostics.append("metadata.description must be text")
    if "category" in value and not isinstance(value["category"], str):
        diagnostics.append("metadata.category must be text")
    if "tags" in value and not _is_string_list(value["tags"]):
        diagnostics.append("metadata.tags must be a list of text")
    if "previewImage" in value and not isinstance(value["previewImage"], str):
        diagnostics.append("metadata.previewImage must be text")


def _validate_contract(value: Any, path: str, items_key: str, diagnostics: list[str]) -> None:
    if not isinstance(value, dict):
        diagnostics.append(f"{path} must be a JSON object")
        return
    items = value.get(items_key)
    if not isinstance(items, list):
        diagnostics.append(f"{path}.{items_key} must be a list")
        return
    for index, item in enumerate(items):
        item_path = f"{path}.{items_key}[{index}]"
        if not isinstance(item, dict):
            diagnostics.append(f"{item_path} must be a JSON object")
            continue
        for key in ("key", "type"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                diagnostics.append(f"{item_path}.{key} must be non-empty text")
        if "label" in item and not isinstance(item["label"], str):
            diagnostics.append(f"{item_path}.label must be text")
        if "required" in item and not isinstance(item["required"], bool):
            diagnostics.append(f"{item_path}.required must be true or false")
        if path == "inputContract":
            target = item.get("target")
            if not isinstance(target, dict):
                diagnostics.append(f"{item_path}.target must be a JSON object")
            else:
                for key in ("node", "input"):
                    if not isinstance(target.get(key), str) or not target[key].strip():
                        diagnostics.append(f"{item_path}.target.{key} must be non-empty text")


def _validate_source(value: Any, diagnostics: list[str]) -> None:
    if not isinstance(value, dict):
        diagnostics.append("source must be a JSON object")
        return
    for key in ("kind", "path"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            diagnostics.append(f"source.{key} must be non-empty text")


def _validate_validation(value: Any, diagnostics: list[str]) -> None:
    if not isinstance(value, dict):
        diagnostics.append("validation must be a JSON object")
        return
    if value.get("status") not in VALIDATION_STATUSES:
        diagnostics.append("validation.status must be valid, draft, or invalid")
    if "diagnostics" in value and not _is_string_list(value["diagnostics"]):
        diagnostics.append("validation.diagnostics must be a list of text")


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _catalog_summary(setup: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": setup["id"],
        "version": setup["version"],
        "updatedAt": setup["updatedAt"],
        "metadata": setup["metadata"],
        "validation": setup["validation"],
        "inputSummary": [
            {
                "key": item["key"],
                "label": item.get("label", item["key"]),
                "type": item["type"],
                "required": bool(item.get("required", False)),
            }
            for item in setup["inputContract"].get("inputs", [])
        ],
        "outputSummary": [
            {
                "key": item["key"],
                "label": item.get("label", item["key"]),
                "type": item["type"],
            }
            for item in setup["outputContract"].get("outputs", [])
        ],
    }
