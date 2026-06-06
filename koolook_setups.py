"""Published workflow setup registry boundary.

This module owns the server-side shape for externally callable Koolook
setups. Callers use this boundary instead of reading sidebar workflow or
snapshot storage directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_SETUPS_SUBDIR = "koolook-published-setups"
DEFAULT_SETUPS_FILENAME = "setups.json"
SAMPLE_SETUPS_PATH = (
    Path(__file__).resolve().parent / "web" / "published_setups_sample.json"
)
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


class StaticSetupStorage:
    """In-memory setup storage adapter for tests and seeded fixtures."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    def load_setups(self) -> list[dict[str, Any]]:
        return list(self._records)


class FileSetupStorage:
    """JSON-file storage adapter for published setup records."""

    def __init__(self, path: Path, fallback_path: Path | None = None) -> None:
        self.path = path
        self.fallback_path = fallback_path

    def load_setups(self) -> list[dict[str, Any]]:
        path = self.path
        if not path.is_file() and self.fallback_path is not None:
            path = self.fallback_path
        if not path.is_file():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("setups", [])
        if not isinstance(raw, list):
            raise ValueError("published setup storage must be a list or {setups: [...]}")
        return [item for item in raw if isinstance(item, dict)]


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
        out.sort(key=lambda setup: str(setup["metadata"]["title"]).lower())
        return out

    def listSetups(self) -> list[dict[str, Any]]:
        return [_catalog_summary(setup) for setup in self._valid_setups()]

    def getSetup(self, setup_id: str) -> dict[str, Any] | None:
        for setup in self._valid_setups():
            if setup["id"] == setup_id:
                return setup
        return None


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
    _validate_scalar(value, "version", (int, str), diagnostics)
    _validate_iso_timestamp(value.get("updatedAt"), diagnostics)
    _validate_metadata(value.get("metadata"), diagnostics)
    _validate_mapping(value.get("visualGraph"), "visualGraph", diagnostics)
    if value.get("apiPrompt") is not None:
        _validate_mapping(value.get("apiPrompt"), "apiPrompt", diagnostics)
    _validate_contract(value.get("inputContract"), "inputContract", "inputs", diagnostics)
    _validate_contract(value.get("outputContract"), "outputContract", "outputs", diagnostics)
    _validate_source(value.get("source"), diagnostics)
    _validate_validation(value.get("validation"), diagnostics)

    return ValidationResult(
        valid=not diagnostics,
        setup=value if not diagnostics else {},
        diagnostics=diagnostics,
    )


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
