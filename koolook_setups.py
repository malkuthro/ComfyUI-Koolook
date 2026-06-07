"""Published workflow setup registry boundary.

This module owns the server-side shape for externally callable Koolook
setups. Callers use this boundary instead of reading sidebar workflow or
snapshot storage directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
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
        "source": source,
        "validation": {
            "status": "valid" if apiPrompt is not None else "draft",
            "diagnostics": [] if apiPrompt is not None else ["API prompt conversion pending."],
        },
    }


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

        widget_values = node.get("widgets_values")
        if not isinstance(widget_values, list):
            widget_values = []
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

        api_prompt[node_id_key] = {"class_type": class_type, "inputs": api_inputs}

    return ApiPromptConversionResult(api_prompt, [])


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
    for key in ("title", "description"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            diagnostics.append(f"metadata.{key} must be non-empty text")
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
