"""Published workflow setup registry boundary.

This module owns the server-side shape for externally callable Koolook
setups. Callers use this boundary instead of reading sidebar workflow or
snapshot storage directly.
"""
from __future__ import annotations

from copy import deepcopy
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
PUBLISH_ROUTER_CLASS = "Koolook_PublishRouter"
VISUAL_ONLY_API_NODE_TYPES = {"Label (rgthree)", "Note"}
SINK_ONLY_API_NODE_TYPES = {"SetNode"}
PUBLISH_INPUT_FIELDS = (
    ("sequence_folder", "Sequence folder", True),
    ("qt_file", "QT file", True),
    ("single_file", "Single file", True),
)
# The prompt is an always-on field, independent of the EXR/QT/Img source switch
# (an external user picks a source AND can describe the shot). The author's
# prompt-widget text becomes a placeholder hint; the submitted default is empty
# so an untouched hint is never sent as the real prompt. Mirror of the JS in
# web/sidebar/published_surface.js -- keep the two in sync.
PUBLISH_PROMPT_HELP = "Describe the shot in one simple line: subject + action + setting."
PUBLISH_INPUT_MODES = (
    (0, "EXR", "sequence_folder"),
    (1, "QT", "qt_file"),
    (2, "Img", "single_file"),
    (3, "Prompt", "prompt"),
)
# Output-type modes are a subset of the input modes (no Prompt — it isn't a
# real output format). The indices deliberately match PUBLISH_INPUT_MODES so a
# router slot means the same type on either switch. -1 is the "Same as input"
# sentinel the author leaves in place to keep output type following input type.
PUBLISH_OUTPUT_MODES = (
    (0, "EXR"),
    (1, "QT"),
    (2, "Img"),
)
PUBLISH_OUTPUT_SAME_AS_INPUT = -1
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
    "Koolook_PublishOutput": ("folder", "name", "version", "output_mode"),
    "Koolook_PublishResult": ("result",),
    "Koolook_PublishRouter": (),
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
    # Older published setups authored their Koolook_PublishOutput with only
    # folder/name/version widgets. Default the new output_mode to "Same as
    # input" so those setups keep validating and behave exactly as before.
    "Koolook_PublishOutput": {
        "output_mode": "Same as input",
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

    @property
    def storage_path(self) -> Path | None:
        """Filesystem path of the backing ``setups.json`` when the storage
        adapter is file-backed; ``None`` for in-memory storage. Lets callers
        surface where a publish landed without reaching into the adapter."""
        path = getattr(self._storage, "path", None)
        return path if isinstance(path, Path) else None

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
        apiPrompt: Any = None,
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

        if apiPrompt is None:
            conversion = _convert_visual_graph_to_api_prompt(visualGraph)
            if conversion.diagnostics:
                return ValidationResult(False, {}, conversion.diagnostics)
            api_prompt = conversion.api_prompt
        elif isinstance(apiPrompt, dict):
            api_prompt = deepcopy(apiPrompt)
        else:
            return ValidationResult(False, {}, ["apiPrompt must be an object when present"])
        surface_geometry_diagnostics = []
        if _uses_group_authored_surface(inputContract, outputContract):
            surface_geometry_diagnostics = _validate_setup_surface_geometry(visualGraph)
        if surface_geometry_diagnostics:
            return ValidationResult(False, {}, surface_geometry_diagnostics)
        setup = _build_draft_setup(
            visualGraph=visualGraph,
            apiPrompt=api_prompt,
            metadata=metadata,
            inputContract=inputContract,
            outputContract=outputContract,
            source=source,
        )
        result = validate_setup(setup)
        diagnostics = list(result.diagnostics)
        if result.valid:
            setup = result.setup
        diagnostics.extend(_validate_contract_targets(visualGraph, inputContract))
        diagnostics.extend(_validate_api_prompt_contract_targets(setup.get("apiPrompt"), inputContract))
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
    setup = deepcopy(value)

    for key in REQUIRED_SETUP_KEYS:
        if key not in value:
            diagnostics.append(f"missing required field: {key}")

    if diagnostics:
        return ValidationResult(False, {}, diagnostics)

    _validate_scalar(setup, "id", str, diagnostics)
    if isinstance(setup.get("id"), str) and not SETUP_ID_RE.match(setup["id"]):
        diagnostics.append("id must be stable URL-safe text")
    _validate_scalar(setup, "schemaVersion", int, diagnostics)
    if setup.get("schemaVersion") != SUPPORTED_SCHEMA_VERSION:
        diagnostics.append(
            f"unsupported schemaVersion: {setup.get('schemaVersion')}; "
            f"expected {SUPPORTED_SCHEMA_VERSION}"
        )
    _validate_scalar(setup, "version", (int, str), diagnostics)
    _validate_iso_timestamp(setup.get("updatedAt"), diagnostics)
    _validate_metadata(setup.get("metadata"), diagnostics)
    _validate_mapping(setup.get("visualGraph"), "visualGraph", diagnostics)
    if setup.get("apiPrompt") is not None:
        _validate_mapping(setup.get("apiPrompt"), "apiPrompt", diagnostics)
        if isinstance(setup.get("visualGraph"), dict) and isinstance(setup.get("apiPrompt"), dict):
            conversion = None
            if _api_prompt_needs_visual_normalization(setup["apiPrompt"], setup["visualGraph"]):
                conversion = _convert_visual_graph_to_api_prompt(setup["visualGraph"])
            if conversion is not None and conversion.diagnostics:
                diagnostics.extend(conversion.diagnostics)
            elif conversion is not None and conversion.api_prompt is not None:
                setup["apiPrompt"] = conversion.api_prompt
    _validate_contract(setup.get("inputContract"), "inputContract", "inputs", diagnostics)
    _validate_contract(setup.get("outputContract"), "outputContract", "outputs", diagnostics)
    _validate_persisted_setup_surface(setup, diagnostics)
    _validate_source(setup.get("source"), diagnostics)
    _validate_validation(setup.get("validation"), diagnostics)
    if (
        isinstance(setup.get("validation"), dict)
        and setup["validation"].get("status") == "valid"
        and setup.get("apiPrompt") is None
    ):
        diagnostics.append("validation.status valid requires apiPrompt")

    return ValidationResult(
        valid=not diagnostics,
        setup=setup if not diagnostics else {},
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
    setup = {
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
    execution_map = _build_execution_map(setup)
    if execution_map is not None:
        setup["executionMap"] = execution_map
        _reconcile_output_switch_visibility(setup, execution_map)
    return setup


def _reconcile_output_switch_visibility(
    setup: dict[str, Any],
    execution_map: dict[str, Any],
) -> None:
    """Hide output-type options that have no wired writer branch.

    The external app should only offer output types that will actually write a
    file. The execution map records ``writerNodes`` per branch for each router;
    an output type is available when an ``output_switch``-driven router has a
    non-empty writer branch for it. When no router is driven by the output
    switch, output type can only follow the input type, so every concrete option
    is hidden and the frontend is left with just the "Same as input" choice.
    """
    app = setup.get("setupSurface", {}).get("app")
    if not isinstance(app, dict):
        return
    output_switch = app.get("outputSwitch")
    if not isinstance(output_switch, dict):
        return
    options = output_switch.get("options")
    if not isinstance(options, list):
        return

    available: set[int] = set()
    has_output_router = False
    routers = execution_map.get("routers") if isinstance(execution_map, dict) else None
    if isinstance(routers, list):
        for router in routers:
            if not isinstance(router, dict) or router.get("switchKey") != "output_switch":
                continue
            has_output_router = True
            branches = router.get("branches")
            if not isinstance(branches, dict):
                continue
            for branch in branches.values():
                if not isinstance(branch, dict):
                    continue
                writer_nodes = branch.get("writerNodes")
                value = branch.get("output")
                if isinstance(writer_nodes, list) and writer_nodes and isinstance(value, int):
                    available.add(value)

    for option in options:
        if not isinstance(option, dict):
            continue
        value = option.get("value")
        option["visible"] = bool(has_output_router and isinstance(value, int) and value in available)


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
    prompt = _publish_prompt_field(input_node)
    if prompt is not None:
        inputs.append(prompt)
    outputs = _publish_output_fields(output_node)
    app: dict[str, Any] = {
        "inputs": inputs,
        "outputs": outputs,
        "results": _publish_result_fields(result_node),
    }
    if input_node is not None:
        app["switch"] = _publish_input_switch(input_node, inputs)
    if output_node is not None:
        app["outputSwitch"] = _publish_output_switch(output_node)
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


def _publish_prompt_field(node: dict[str, Any] | None) -> dict[str, Any] | None:
    if node is None:
        return None
    hint = _app_widget_value(node, "prompt")
    return {
        "key": "prompt",
        "label": "Prompt",
        "visible": True,
        "standalone": True,
        "multiline": True,
        "target": {"node": str(node.get("id")), "input": "prompt"},
        "default": "",
        "placeholder": hint if isinstance(hint, str) else "",
        "help": PUBLISH_PROMPT_HELP,
    }


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
                # Standalone fields (e.g. the always-on prompt) are not source
                # modes, so they never appear as a switch option.
                "visible": bool(inputs_by_key.get(input_key, {}).get("visible"))
                and not inputs_by_key.get(input_key, {}).get("standalone"),
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


def _publish_output_switch(node: dict[str, Any]) -> dict[str, Any]:
    """App-surface descriptor for the independent output-type switch.

    Mirrors ``_publish_input_switch`` but for ``Koolook_PublishOutput``. The
    ``target`` is the node's ``output_mode`` input (where the runner injects the
    submitted value). ``sameAsInput`` tells the external frontend to offer a
    "Same as input" choice that follows the input switch; ``default`` is
    ``-1`` when the author left the widget on "Same as input".
    """
    return {
        "key": "output_switch",
        "label": "Output type",
        "visible": True,
        "sameAsInput": True,
        "target": {"node": str(node.get("id")), "input": "output_mode"},
        "default": _publish_output_mode_index(_app_widget_value(node, "output_mode")),
        "options": [
            {"value": value, "label": label, "visible": True}
            for value, label in PUBLISH_OUTPUT_MODES
        ],
    }


def _publish_output_mode_index(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    for mode_value, label in PUBLISH_OUTPUT_MODES:
        if text == label.lower():
            return mode_value
    return PUBLISH_OUTPUT_SAME_AS_INPUT


def _build_execution_map(setup: dict[str, Any]) -> dict[str, Any] | None:
    api_prompt = setup.get("apiPrompt")
    if not isinstance(api_prompt, dict):
        return None
    app = setup.get("setupSurface", {}).get("app", {})
    # A router may be driven by the input switch (source type) OR the output
    # switch (writer type). Collect both so an EXR-in/QT-out setup wires its
    # writer router from Koolook_PublishOutput.switch and still gets a map.
    candidate_switches = [
        switch
        for switch in (
            app.get("switch") if isinstance(app, dict) else None,
            app.get("outputSwitch") if isinstance(app, dict) else None,
        )
        if isinstance(switch, dict) and isinstance(switch.get("target"), dict)
    ]
    if not candidate_switches:
        return None
    output_surface_ids = _setup_output_surface_node_ids(setup)
    if not output_surface_ids:
        return None
    routers: list[dict[str, Any]] = []
    for node_id, node in api_prompt.items():
        if not isinstance(node, dict) or node.get("class_type") != PUBLISH_ROUTER_CLASS:
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        selector_ref = inputs.get("selector")
        payload_ref = inputs.get("payload")
        matched_switch = next(
            (
                switch
                for switch in candidate_switches
                if _api_ref_matches_target_output(
                    selector_ref, switch["target"], api_prompt, "switch"
                )
            ),
            None,
        )
        if matched_switch is None:
            continue
        branches: dict[str, dict[str, Any]] = {}
        for value, label, _input_key in PUBLISH_INPUT_MODES:
            branches[str(value)] = {
                "label": label,
                "output": value,
                "writerNodes": _writer_nodes_for_router_output(
                    api_prompt,
                    str(node_id),
                    value,
                    output_surface_ids,
                ),
            }
        router: dict[str, Any] = {
            "node": str(node_id),
            "switchKey": str(matched_switch.get("key", "switch")),
            "selector": {"node": str(selector_ref[0]), "output": int(selector_ref[1])},
            "branches": branches,
        }
        if _is_api_ref(payload_ref):
            router["payload"] = {"node": str(payload_ref[0]), "output": int(payload_ref[1])}
        routers.append(router)
    if not routers:
        return None
    routers.sort(key=lambda item: item["node"])
    return {"version": 1, "routers": routers}


def _writer_nodes_for_router_output(
    api_prompt: dict[str, Any],
    router_node_id: str,
    router_output: int,
    output_surface_ids: set[str],
) -> list[str]:
    children = _prompt_children_by_ref(api_prompt)
    out: list[str] = []
    seen: set[str] = set()
    pending = list(children.get((router_node_id, router_output), []))
    while pending:
        node_id = pending.pop(0)
        if node_id in seen:
            continue
        seen.add(node_id)
        if node_id in output_surface_ids and node_id != router_node_id:
            out.append(node_id)
        for child_ids in _children_for_node(children, node_id):
            pending.extend(child_ids)
    return sorted(dict.fromkeys(out))


def _children_for_node(
    children: dict[tuple[str, int], list[str]],
    node_id: str,
) -> list[list[str]]:
    return [
        child_ids
        for (source_node_id, _source_slot), child_ids in children.items()
        if source_node_id == node_id
    ]


def _prompt_children_by_ref(api_prompt: dict[str, Any]) -> dict[tuple[str, int], list[str]]:
    children: dict[tuple[str, int], list[str]] = {}
    for node_id, node in api_prompt.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for value in inputs.values():
            for ref in _api_refs(value):
                children.setdefault((str(ref[0]), int(ref[1])), []).append(str(node_id))
    return children


def _setup_output_surface_node_ids(setup: dict[str, Any]) -> set[str]:
    surface = setup.get("setupSurface")
    if not isinstance(surface, dict) or not isinstance(surface.get("outputs"), list):
        return set()
    out: set[str] = set()
    for group in surface["outputs"]:
        if not isinstance(group, dict) or not isinstance(group.get("nodes"), list):
            continue
        for node in group["nodes"]:
            if isinstance(node, dict) and node.get("id") is not None:
                out.add(str(node["id"]))
    return out


def _api_ref_matches_target_output(
    value: Any,
    target: dict[str, Any],
    api_prompt: dict[str, Any],
    output_name: str,
) -> bool:
    if not _is_api_ref(value):
        return False
    node_id = str(value[0])
    if node_id != str(target.get("node")):
        return False
    node = api_prompt.get(node_id)
    if not isinstance(node, dict):
        return False
    return _publish_input_output_slots(node).get(output_name) == int(value[1])


def _publish_input_output_slots(api_node: dict[str, Any]) -> dict[str, int]:
    """Output-slot name -> index for a publish contract node.

    Both ``Koolook_PublishInput`` and ``Koolook_PublishOutput`` expose a
    ``switch`` output that a router's ``selector`` can be wired from; the
    executionMap matcher uses this to tie a router to whichever switch drives
    it.
    """
    class_type = api_node.get("class_type")
    if class_type == PUBLISH_INPUT_CLASS:
        return {
            "sequence_folder": 0,
            "qt_file": 1,
            "single_file": 2,
            "prompt": 3,
            "switch": 4,
        }
    if class_type == PUBLISH_OUTPUT_CLASS:
        return {
            "folder": 0,
            "name": 1,
            "version": 2,
            "switch": 3,
        }
    return {}


def _is_api_ref(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and not isinstance(value[1], bool)
        and isinstance(value[1], int)
    )


def _api_refs(value: Any) -> list[list[Any]]:
    if _is_api_ref(value):
        return [value]
    if isinstance(value, dict):
        refs: list[list[Any]] = []
        for child in value.values():
            refs.extend(_api_refs(child))
        return refs
    if isinstance(value, list):
        refs = []
        for child in value:
            refs.extend(_api_refs(child))
        return refs
    return []


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
    if "outputSwitch" in app:
        # The output switch selects a writer branch, not a source field, so its
        # options carry no `input` mapping (require_option_input=False).
        _validate_surface_app_switch(
            app["outputSwitch"],
            f"{path}.outputSwitch",
            diagnostics,
            node_by_id,
            api_prompt,
            input_keys,
            require_option_input=False,
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
    *,
    require_option_input: bool = True,
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
        if require_option_input:
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

    node_by_id = {str(node["id"]): node for node in nodes}
    subgraphs_by_id = _subgraph_definitions_by_id(visual_graph)
    api_prompt: dict[str, Any] = {}
    subgraph_output_refs: dict[str, dict[int, Any]] = {}
    skipped_node_ids: set[str] = set()
    expanded_node_ids: set[str] = set()
    diagnostics: list[str] = []

    def resolve_outer_link(link_id: Any, seen: set[str] | None = None) -> Any:
        link_id_key = str(link_id)
        seen = set() if seen is None else seen
        if link_id_key in seen:
            diagnostics.append(f"visualGraph.links[{link_id_key}] has a reroute cycle")
            return None
        seen.add(link_id_key)
        link = links.links.get(link_id_key)
        if link is None:
            diagnostics.append(f"visualGraph.links[{link_id_key}] is missing")
            return None
        origin_id = str(link["origin_id"])
        if _is_module_sentinel_id(link["origin_id"]):
            diagnostics.append(f"visualGraph.links[{link_id_key}] uses unsupported module graph sentinel node")
            return None
        if not _is_non_negative_int(link["origin_slot"]):
            diagnostics.append(
                f"visualGraph.links[{link_id_key}].origin_slot must be a non-negative integer"
            )
            return None
        origin_slot = int(link["origin_slot"])
        origin_node = node_by_id.get(origin_id)
        if origin_node is None:
            diagnostics.append(f"visualGraph.links[{link_id_key}].origin_id not found in visualGraph")
            return None
        if origin_node.get("type") == "Reroute":
            reroute_input = _first_linked_input(origin_node)
            if reroute_input is None:
                diagnostics.append(f"visualGraph.nodes[{node_indexes_by_id[origin_id]}] Reroute is unconnected")
                return None
            return resolve_outer_link(reroute_input, seen)
        if origin_id in subgraph_output_refs:
            ref = subgraph_output_refs[origin_id].get(origin_slot)
            if ref is None:
                diagnostics.append(
                    f"visualGraph.links[{link_id_key}] references missing subgraph output {origin_slot}"
                )
            return ref
        if origin_id in skipped_node_ids or origin_id in expanded_node_ids:
            diagnostics.append(
                f"visualGraph.links[{link_id_key}] references non-executable node {origin_id}"
            )
            return None
        return [origin_id, origin_slot]

    def add_node_api(
        *,
        node: dict[str, Any],
        node_index: int,
        node_id_key: str,
        class_type: str,
        link_resolver,
    ) -> None:
        node_inputs = node.get("inputs")
        if node_inputs is None:
            node_inputs = []
        elif not isinstance(node_inputs, list):
            diagnostics.append(f"visualGraph.nodes[{node_index}].inputs must be a list when present")
            return

        raw_widget_values = node.get("widgets_values")
        widget_values_by_name = raw_widget_values if isinstance(raw_widget_values, dict) else {}
        widget_values = raw_widget_values if isinstance(raw_widget_values, list) else []
        if not widget_values and not widget_values_by_name:
            widget_values = _proxy_widget_values_from_subgraph(visual_graph, node)
        widget_index = 0
        api_inputs: dict[str, Any] = {}
        for input_index, input_port in enumerate(node_inputs):
            if not isinstance(input_port, dict):
                diagnostics.append(f"visualGraph.nodes[{node_index}].inputs must contain objects")
                return
            name = input_port.get("name")
            if not isinstance(name, str) or not name.strip():
                if class_type != "Reroute" or not isinstance(name, str):
                    diagnostics.append(f"visualGraph.nodes[{node_index}].inputs must have named ports")
                    return
            link_id = input_port.get("link")
            if link_id is not None:
                link = links.links.get(str(link_id))
                if link is not None and link["target_id"] is not None and str(link["target_id"]) != node_id_key:
                    diagnostics.append(
                        f"visualGraph.links[{link_id}].target does not match "
                        f"visualGraph.nodes[{node_index}].inputs[{input_index}]"
                    )
                    return
                if (
                    link is not None
                    and link["target_slot"] is not None
                    and (
                        not _is_non_negative_int(link["target_slot"])
                        or int(link["target_slot"]) != input_index
                    )
                ):
                    diagnostics.append(
                        f"visualGraph.links[{link_id}].target does not match "
                        f"visualGraph.nodes[{node_index}].inputs[{input_index}]"
                    )
                    return
                resolved = link_resolver(link_id)
                if resolved is None:
                    return
                api_inputs[name] = resolved
                continue
            widget = input_port.get("widget")
            if isinstance(widget, dict):
                if name in widget_values_by_name:
                    api_inputs[name] = widget_values_by_name[name]
                    continue
                if widget_index >= len(widget_values):
                    diagnostics.append(
                        f"visualGraph.nodes[{node_index}].widgets_values is missing a value "
                        f"for input {name}"
                    )
                    return
                api_inputs[name] = widget_values[widget_index]
                widget_index += 1

        if not node_inputs:
            widget_only_diagnostics = _apply_widget_only_inputs(
                class_type,
                widget_values,
                api_inputs,
                node_index,
            )
            diagnostics.extend(widget_only_diagnostics)
            if widget_only_diagnostics:
                return
        api_prompt[node_id_key] = {"class_type": class_type, "inputs": api_inputs}

    def expand_subgraph_node(node: dict[str, Any], node_index: int) -> None:
        node_id_key = str(node["id"])
        subgraph = subgraphs_by_id.get(str(node["type"]))
        if subgraph is None:
            return
        subgraph_links = _visual_link_index(subgraph.get("links"))
        if subgraph_links.diagnostics:
            diagnostics.extend(subgraph_links.diagnostics)
            return
        subgraph_nodes = subgraph.get("nodes")
        if not isinstance(subgraph_nodes, list):
            diagnostics.append(f"visualGraph.definitions.subgraphs[{node['type']}].nodes must be a list")
            return
        subgraph_inputs = subgraph.get("inputs")
        if not isinstance(subgraph_inputs, list):
            subgraph_inputs = []
        subgraph_nodes_by_id = {
            str(subgraph_node.get("id")): subgraph_node
            for subgraph_node in subgraph_nodes
            if isinstance(subgraph_node, dict) and subgraph_node.get("id") is not None
        }
        wrapper_inputs = _api_inputs_for_wrapper_node(
            visual_graph,
            node,
            node_index,
            links,
            resolve_outer_link,
            diagnostics,
        )
        skipped_internal_ids: set[str] = set()
        for internal_node in subgraph_nodes:
            if not isinstance(internal_node, dict):
                continue
            internal_id = str(internal_node.get("id"))
            if _should_omit_api_node(internal_node):
                skipped_internal_ids.add(internal_id)

        def resolve_subgraph_link(link_id: Any, seen: set[str] | None = None) -> Any:
            link_id_key = str(link_id)
            seen = set() if seen is None else seen
            if link_id_key in seen:
                diagnostics.append(f"visualGraph.definitions.subgraphs[{node['type']}].links[{link_id_key}] has a reroute cycle")
                return None
            seen.add(link_id_key)
            link = subgraph_links.links.get(link_id_key)
            if link is None:
                diagnostics.append(f"visualGraph.definitions.subgraphs[{node['type']}].links[{link_id_key}] is missing")
                return None
            origin_id = link["origin_id"]
            if not _is_non_negative_int(link["origin_slot"]):
                diagnostics.append(
                    f"visualGraph.definitions.subgraphs[{node['type']}].links[{link_id_key}].origin_slot "
                    "must be a non-negative integer"
                )
                return None
            origin_slot = int(link["origin_slot"])
            if _is_module_sentinel_id(origin_id):
                if int(origin_id) != -10:
                    diagnostics.append(
                        f"visualGraph.definitions.subgraphs[{node['type']}].links[{link_id_key}] "
                        "uses unsupported module graph sentinel node"
                    )
                    return None
                if origin_slot >= len(subgraph_inputs):
                    diagnostics.append(
                        f"visualGraph.definitions.subgraphs[{node['type']}].links[{link_id_key}] "
                        "references missing subgraph input"
                    )
                    return None
                input_name = subgraph_inputs[origin_slot].get("name")
                if not isinstance(input_name, str) or input_name not in wrapper_inputs:
                    return None
                return wrapper_inputs[input_name]
            origin_id_key = str(origin_id)
            origin_node = subgraph_nodes_by_id.get(origin_id_key)
            if origin_node is None:
                diagnostics.append(
                    f"visualGraph.definitions.subgraphs[{node['type']}].links[{link_id_key}].origin_id not found"
                )
                return None
            if origin_node.get("type") == "Reroute":
                reroute_input = _first_linked_input(origin_node)
                if reroute_input is None:
                    diagnostics.append(
                        f"visualGraph.definitions.subgraphs[{node['type']}].nodes[{origin_id_key}] Reroute is unconnected"
                    )
                    return None
                return resolve_subgraph_link(reroute_input, seen)
            if origin_id_key in skipped_internal_ids:
                diagnostics.append(
                    f"visualGraph.definitions.subgraphs[{node['type']}].links[{link_id_key}] "
                    f"references non-executable node {origin_id_key}"
                )
                return None
            return [f"{node_id_key}:{origin_id_key}", origin_slot]

        for internal_index, internal_node in enumerate(subgraph_nodes):
            if not isinstance(internal_node, dict):
                diagnostics.append(
                    f"visualGraph.definitions.subgraphs[{node['type']}].nodes[{internal_index}] must be an object"
                )
                return
            internal_class_type = internal_node.get("type")
            internal_id = internal_node.get("id")
            if internal_id is None or not isinstance(internal_class_type, str) or not internal_class_type.strip():
                diagnostics.append(
                    f"visualGraph.definitions.subgraphs[{node['type']}].nodes[{internal_index}].type "
                    "must be non-empty text"
                )
                return
            if str(internal_class_type) in subgraphs_by_id:
                diagnostics.append(
                    f"visualGraph.definitions.subgraphs[{node['type']}].nodes[{internal_index}] "
                    "nested subgraphs are not supported"
                )
                return
            if str(internal_id) in skipped_internal_ids:
                continue
            add_node_api(
                node=internal_node,
                node_index=node_index,
                node_id_key=f"{node_id_key}:{internal_id}",
                class_type=internal_class_type,
                link_resolver=resolve_subgraph_link,
            )
            if diagnostics:
                return

        output_refs: dict[int, Any] = {}
        for link_id, link in subgraph_links.links.items():
            if not _is_module_sentinel_id(link["target_id"]) or int(link["target_id"]) != -20:
                continue
            target_slot = link.get("target_slot")
            if not _is_non_negative_int(target_slot):
                continue
            ref = resolve_subgraph_link(link_id)
            if ref is not None:
                output_refs[int(target_slot)] = ref
        subgraph_output_refs[node_id_key] = output_refs
        expanded_node_ids.add(node_id_key)

    for node_index, node in enumerate(nodes):
        node_id_key = str(node["id"])
        class_type = node["type"]
        if _should_omit_api_node(node):
            skipped_node_ids.add(node_id_key)
            continue
        if class_type == "Reroute":
            skipped_node_ids.add(node_id_key)
            continue
        if class_type in subgraphs_by_id:
            expand_subgraph_node(node, node_index)
            if diagnostics:
                return ApiPromptConversionResult(None, diagnostics)
            continue
        add_node_api(
            node=node,
            node_index=node_index,
            node_id_key=node_id_key,
            class_type=class_type,
            link_resolver=resolve_outer_link,
        )
        if diagnostics:
            return ApiPromptConversionResult(None, diagnostics)

    return ApiPromptConversionResult(api_prompt, diagnostics)


def _api_inputs_for_wrapper_node(
    visual_graph: dict[str, Any],
    node: dict[str, Any],
    node_index: int,
    links: VisualLinkIndex,
    link_resolver,
    diagnostics: list[str],
) -> dict[str, Any]:
    node_inputs = node.get("inputs")
    if node_inputs is None:
        return {}
    if not isinstance(node_inputs, list):
        diagnostics.append(f"visualGraph.nodes[{node_index}].inputs must be a list when present")
        return {}
    raw_widget_values = node.get("widgets_values")
    widget_values_by_name = raw_widget_values if isinstance(raw_widget_values, dict) else {}
    widget_values = raw_widget_values if isinstance(raw_widget_values, list) else []
    widget_index = 0
    values: dict[str, Any] = {}
    for input_index, input_port in enumerate(node_inputs):
        if not isinstance(input_port, dict):
            diagnostics.append(f"visualGraph.nodes[{node_index}].inputs must contain objects")
            return values
        name = input_port.get("name")
        if not isinstance(name, str) or not name.strip():
            diagnostics.append(f"visualGraph.nodes[{node_index}].inputs must have named ports")
            return values
        link_id = input_port.get("link")
        if link_id is not None:
            link = links.links.get(str(link_id))
            if link is not None and link["target_id"] is not None and str(link["target_id"]) != str(node.get("id")):
                diagnostics.append(
                    f"visualGraph.links[{link_id}].target does not match "
                    f"visualGraph.nodes[{node_index}].inputs[{input_index}]"
                )
                return values
            resolved = link_resolver(link_id)
            if resolved is not None:
                values[name] = resolved
            continue
        widget = input_port.get("widget")
        if not isinstance(widget, dict):
            continue
        if name in widget_values_by_name:
            values[name] = widget_values_by_name[name]
            continue
        if widget_index < len(widget_values):
            values[name] = widget_values[widget_index]
        widget_index += 1
    proxy_values = _proxy_widget_values_from_subgraph(visual_graph, node)
    proxy_index = 0
    for input_port in node_inputs:
        if proxy_index >= len(proxy_values) or not isinstance(input_port, dict):
            continue
        name = input_port.get("name")
        if isinstance(name, str) and name not in values and input_port.get("widget"):
            values[name] = proxy_values[proxy_index]
        if input_port.get("widget"):
            proxy_index += 1
    return values


def _api_prompt_needs_visual_normalization(
    api_prompt: dict[str, Any],
    visual_graph: dict[str, Any],
) -> bool:
    subgraph_ids = set(_subgraph_definitions_by_id(visual_graph))
    for node in api_prompt.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        if not isinstance(class_type, str):
            continue
        if class_type == "Reroute":
            return True
        if class_type in VISUAL_ONLY_API_NODE_TYPES:
            return True
        if class_type in SINK_ONLY_API_NODE_TYPES:
            return True
        if class_type in subgraph_ids:
            return True
    return False


def _subgraph_definitions_by_id(visual_graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    definitions = visual_graph.get("definitions")
    if not isinstance(definitions, dict):
        return {}
    subgraphs = definitions.get("subgraphs")
    if not isinstance(subgraphs, list):
        return {}
    return {
        str(subgraph.get("id")): subgraph
        for subgraph in subgraphs
        if isinstance(subgraph, dict) and subgraph.get("id") is not None
    }


def _should_omit_api_node(node: dict[str, Any]) -> bool:
    class_type = str(node.get("type", ""))
    if class_type in VISUAL_ONLY_API_NODE_TYPES:
        return True
    if class_type in SINK_ONLY_API_NODE_TYPES:
        return not _node_has_linked_outputs(node)
    return False


def _node_has_linked_outputs(node: dict[str, Any]) -> bool:
    outputs = node.get("outputs")
    if not isinstance(outputs, list):
        return False
    for output in outputs:
        if not isinstance(output, dict):
            continue
        links = output.get("links")
        if isinstance(links, list) and links:
            return True
    return False


def _first_linked_input(node: dict[str, Any]) -> Any | None:
    node_inputs = node.get("inputs")
    if not isinstance(node_inputs, list):
        return None
    for input_port in node_inputs:
        if not isinstance(input_port, dict):
            continue
        link_id = input_port.get("link")
        if link_id is not None:
            return link_id
    return None


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
