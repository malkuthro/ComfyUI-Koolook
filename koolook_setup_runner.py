"""Execution boundary for published Koolook workflow setups."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

try:
    from .koolook_setups import PublishedSetupRegistry
except ImportError:  # pragma: no cover - standalone test/import context
    from koolook_setups import PublishedSetupRegistry


class SetupRunError(Exception):
    """API-facing setup runner error with stable HTTP status semantics."""

    def __init__(self, status_code: int, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.status_code = status_code
        self.errors = errors


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    setup_id: str
    prompt_id: str
    status: str
    inputs: dict[str, Any]


class InMemorySetupRunStore:
    """Small run-id mapping store for a running ComfyUI process."""

    def __init__(self) -> None:
        self._next_id = 1
        self._records: dict[str, RunRecord] = {}

    def create(self, *, setup_id: str, prompt_id: str, inputs: dict[str, Any] | None = None) -> RunRecord:
        run_id = f"run-{self._next_id:06d}"
        self._next_id += 1
        record = RunRecord(
            run_id=run_id,
            setup_id=setup_id,
            prompt_id=prompt_id,
            status="queued",
            inputs=deepcopy(inputs) if isinstance(inputs, dict) else {},
        )
        self._records[run_id] = record
        return record

    def get(self, run_id: str) -> RunRecord | None:
        return self._records.get(run_id)


class AiohttpComfyClient:
    """HTTP adapter for the managed ComfyUI server."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    async def queue_prompt(self, prompt: dict) -> dict:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self._base_url}/prompt", json={"prompt": prompt}) as response:
                try:
                    payload = await response.json()
                except Exception:
                    payload = {"error": await response.text()}
                if response.status >= 400:
                    detail = payload.get("error") if isinstance(payload, dict) else payload
                    raise SetupRunError(
                        502,
                        [f"ComfyUI queue request failed with HTTP {response.status}: {detail}"],
                    )
                return payload if isinstance(payload, dict) else {}

    async def get_history(self, prompt_id: str) -> dict:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self._base_url}/history/{prompt_id}") as response:
                try:
                    payload = await response.json()
                except Exception:
                    payload = {"error": await response.text()}
                if response.status >= 400:
                    detail = payload.get("error") if isinstance(payload, dict) else payload
                    raise SetupRunError(
                        502,
                        [f"ComfyUI history request failed with HTTP {response.status}: {detail}"],
                    )
                return payload if isinstance(payload, dict) else {}

    async def get_queue(self) -> dict:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self._base_url}/queue") as response:
                try:
                    payload = await response.json()
                except Exception:
                    payload = {"error": await response.text()}
                if response.status >= 400:
                    detail = payload.get("error") if isinstance(payload, dict) else payload
                    raise SetupRunError(
                        502,
                        [f"ComfyUI queue status request failed with HTTP {response.status}: {detail}"],
                    )
                return payload if isinstance(payload, dict) else {}


class PublishedSetupRunner:
    """Runs callable published setups without exposing ComfyUI internals."""

    def __init__(
        self,
        registry: PublishedSetupRegistry,
        comfy_client: Any,
        run_store: InMemorySetupRunStore | None = None,
    ) -> None:
        self._registry = registry
        self._comfy_client = comfy_client
        self._run_store = run_store or InMemorySetupRunStore()

    async def runSetup(self, setup_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
        setup = self._registry.getSetup(setup_id)
        if setup is None:
            raise SetupRunError(404, [f"Published setup '{setup_id}' not found."])
        if not isinstance(inputs, dict):
            raise SetupRunError(400, ["inputs must be a JSON object"])

        api_prompt = setup.get("apiPrompt")
        if not isinstance(api_prompt, dict) or setup.get("validation", {}).get("status") != "valid":
            raise SetupRunError(400, [f"Published setup '{setup_id}' is not callable."])

        prompt = deepcopy(api_prompt)
        input_fields = _declared_input_fields(setup)
        errors = _validate_run_inputs(input_fields, inputs)
        errors.extend(_validate_execution_map_inputs(setup, inputs))
        if errors:
            raise SetupRunError(400, errors)

        for key, value in inputs.items():
            field = input_fields[key]
            target = field["target"]
            prompt[str(target["node"])]["inputs"][str(target["input"])] = _prompt_value_for_field(field, value)

        mapped_prompt = _prune_prompt_for_execution_map(setup, prompt, inputs)
        prompt = (
            mapped_prompt
            if mapped_prompt is not None
            else _prune_prompt_for_selected_app_results(setup, prompt, inputs)
        )

        try:
            queued = await self._comfy_client.queue_prompt(prompt)
        except SetupRunError:
            raise
        except Exception as exc:
            raise SetupRunError(502, [f"ComfyUI queue request failed: {exc}"]) from exc

        prompt_id = queued.get("prompt_id") if isinstance(queued, dict) else None
        if not isinstance(prompt_id, str) or not prompt_id:
            raise SetupRunError(502, ["ComfyUI queue response did not include prompt_id"])

        record = self._run_store.create(setup_id=setup_id, prompt_id=prompt_id, inputs=inputs)
        return {"runId": record.run_id, "promptId": record.prompt_id, "status": record.status}

    async def getRun(self, run_id: str) -> dict[str, Any]:
        record = self._run_store.get(run_id)
        if record is None:
            raise SetupRunError(404, [f"Koolook run '{run_id}' not found."])

        setup = self._registry.getSetup(record.setup_id)
        if setup is None:
            raise SetupRunError(404, [f"Published setup '{record.setup_id}' not found."])

        try:
            history = await self._comfy_client.get_history(record.prompt_id)
        except SetupRunError:
            raise
        except Exception as exc:
            raise SetupRunError(502, [f"ComfyUI history request failed: {exc}"]) from exc

        history_entry = history.get(record.prompt_id) if isinstance(history, dict) else None
        if isinstance(history_entry, dict):
            status = _status_from_history(history_entry)
            return {
                "runId": record.run_id,
                "setupId": record.setup_id,
                "promptId": record.prompt_id,
                "status": status,
                "comfyStatus": history_entry.get("status", {}),
                "outputs": _summarize_outputs(
                    setup,
                    history_entry.get("outputs", {}),
                    record.inputs,
                    _history_prompt(history_entry),
                ),
            }

        try:
            queue = await self._comfy_client.get_queue()
        except SetupRunError:
            raise
        except Exception as exc:
            raise SetupRunError(502, [f"ComfyUI queue status request failed: {exc}"]) from exc
        return {
            "runId": record.run_id,
            "setupId": record.setup_id,
            "promptId": record.prompt_id,
            "status": _status_from_queue(record.prompt_id, queue),
            "outputs": [],
        }


def _declared_input_fields(setup: dict[str, Any]) -> dict[str, dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = {}
    for field in setup.get("inputContract", {}).get("inputs", []):
        if not isinstance(field, dict):
            continue
        key = field.get("key")
        target = field.get("target")
        if isinstance(key, str) and key and isinstance(target, dict):
            fields[key] = field
    app = setup.get("setupSurface", {}).get("app", {})
    if isinstance(app, dict):
        for field in app.get("inputs", []):
            if not isinstance(field, dict):
                continue
            key = field.get("key")
            target = field.get("target")
            if isinstance(key, str) and key and isinstance(target, dict):
                fields.setdefault(key, field)
        for field in app.get("outputs", []):
            if not isinstance(field, dict):
                continue
            key = field.get("key")
            target = field.get("target")
            if isinstance(key, str) and key and isinstance(target, dict):
                fields.setdefault(key, field)
        switch = app.get("switch")
        if isinstance(switch, dict):
            key = switch.get("key")
            target = switch.get("target")
            if isinstance(key, str) and key and isinstance(target, dict):
                fields.setdefault(key, switch)
    return fields


def _validate_run_inputs(input_fields: dict[str, dict[str, Any]], inputs: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in inputs:
        if key not in input_fields:
            errors.append(f"input '{key}' is not declared by this setup")
            continue
        field = input_fields[key]
        if field.get("key") == "switch" and _switch_option_value(field, inputs[key]) is None:
            errors.append(_switch_value_error(key, field))
    for key, field in input_fields.items():
        if field.get("required") is True and key not in inputs:
            errors.append(f"required input '{key}' is missing")
    return errors


def _prompt_value_for_field(field: dict[str, Any], value: Any) -> Any:
    if field.get("key") != "switch":
        return value
    selected_value = _switch_option_value(field, value)
    for option in field.get("options", []):
        if not isinstance(option, dict):
            continue
        if option.get("value") == selected_value and isinstance(option.get("label"), str):
            return option["label"]
    return value


def _switch_option_value(switch: dict[str, Any], selected: Any) -> int | None:
    if isinstance(selected, bool):
        return None
    for option in switch.get("options", []):
        if not isinstance(option, dict):
            continue
        value = option.get("value")
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if selected == value:
            return value
        if isinstance(selected, str) and (selected == str(value) or selected == option.get("label")):
            return value
    return None


def _switch_value_error(key: str, switch: dict[str, Any]) -> str:
    choices: list[str] = []
    for option in switch.get("options", []):
        if not isinstance(option, dict):
            continue
        value = option.get("value")
        label = option.get("label")
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        choices.append(f"{value} ({label})" if isinstance(label, str) and label else str(value))
    if choices:
        return f"input '{key}' must be one of: {', '.join(choices)}"
    return f"input '{key}' has no valid switch options"


def _status_from_history(history_entry: dict[str, Any]) -> str:
    status = history_entry.get("status")
    if not isinstance(status, dict):
        return "succeeded"
    status_text = str(status.get("status_str", "")).lower()
    if status.get("completed") is True and status_text not in {"error", "failed", "failure"}:
        return "succeeded"
    return "failed"


def _status_from_queue(prompt_id: str, queue: Any) -> str:
    if not isinstance(queue, dict):
        return "lost"
    if _queue_contains_prompt(queue.get("queue_running", []), prompt_id):
        return "running"
    if _queue_contains_prompt(queue.get("queue_pending", []), prompt_id):
        return "queued"
    return "lost"


def _queue_contains_prompt(entries: Any, prompt_id: str) -> bool:
    if not isinstance(entries, list):
        return False
    for entry in entries:
        if isinstance(entry, dict) and entry.get("prompt_id") == prompt_id:
            return True
        if isinstance(entry, list) and prompt_id in entry:
            return True
    return False


def _summarize_outputs(
    setup: dict[str, Any],
    raw_outputs: Any,
    run_inputs: dict[str, Any] | None = None,
    history_prompt: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    output_items = _flatten_history_outputs(raw_outputs)
    summaries = []
    for output in setup.get("outputContract", {}).get("outputs", []):
        if not isinstance(output, dict):
            continue
        summaries.append(
            {
                "key": output.get("key", ""),
                "label": output.get("label", output.get("key", "")),
                "type": output.get("type", ""),
                "items": [
                    item
                    for item in output_items
                    if _history_output_matches_contract(item, str(output.get("type", "")))
                ],
            }
        )
    summaries.extend(_summarize_app_surface_outputs(setup, output_items, run_inputs or {}, history_prompt))
    return summaries


def _summarize_app_surface_outputs(
    setup: dict[str, Any],
    output_items: list[dict[str, Any]],
    run_inputs: dict[str, Any],
    history_prompt: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    app = setup.get("setupSurface", {}).get("app", {})
    if not isinstance(app, dict):
        return []
    summaries: list[dict[str, Any]] = []
    for field_type, fields in (("output", app.get("outputs")), ("result", app.get("results"))):
        if not isinstance(fields, list):
            continue
        for field in fields:
            if not isinstance(field, dict):
                continue
            target = field.get("target")
            target_node = str(target.get("node")) if isinstance(target, dict) else ""
            items = [
                item
                for item in output_items
                if target_node and item.get("nodeId") == target_node
            ]
            if field_type == "result" and not items:
                items = _selected_switch_result_items(setup, field, output_items, run_inputs)
            if field_type == "result" and not items and history_prompt is not None:
                items = _execution_map_writer_result_items(setup, history_prompt, run_inputs)
            summary = {
                "key": field.get("key", ""),
                "label": field.get("label", field.get("key", "")),
                "type": field_type,
                "visible": field.get("visible", True),
                "target": target if isinstance(target, dict) else {},
                "default": field.get("default"),
                "items": items,
            }
            summaries.append(summary)
    return summaries


def _history_prompt(history_entry: dict[str, Any]) -> dict[str, Any] | None:
    prompt_record = history_entry.get("prompt")
    if (
        isinstance(prompt_record, list)
        and len(prompt_record) >= 3
        and isinstance(prompt_record[2], dict)
    ):
        return prompt_record[2]
    return None


def _execution_map_writer_result_items(
    setup: dict[str, Any],
    prompt: dict[str, Any],
    run_inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    writer_nodes = _selected_execution_map_writer_nodes(setup, run_inputs)
    items: list[dict[str, Any]] = []
    for writer_node in writer_nodes:
        node = prompt.get(writer_node)
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        value = _resolve_prompt_value(prompt, inputs.get("filepath"))
        if isinstance(value, str) and value:
            items.append({"nodeId": writer_node, "kind": "text", "value": value})
    return items


def _selected_execution_map_writer_nodes(
    setup: dict[str, Any],
    run_inputs: dict[str, Any],
) -> list[str]:
    execution_map = setup.get("executionMap")
    if not isinstance(execution_map, dict) or execution_map.get("version") != 1:
        return []
    routers = execution_map.get("routers")
    if not isinstance(routers, list):
        return []
    out: list[str] = []
    for router in routers:
        if not isinstance(router, dict):
            continue
        selected_value = _execution_map_selected_value(setup, router, run_inputs)
        branches = router.get("branches")
        branch = branches.get(str(selected_value)) if isinstance(branches, dict) else None
        writer_nodes = branch.get("writerNodes") if isinstance(branch, dict) else None
        if isinstance(writer_nodes, list):
            out.extend(str(node_id) for node_id in writer_nodes)
    return list(dict.fromkeys(out))


def _selected_switch_result_items(
    setup: dict[str, Any],
    result_field: dict[str, Any],
    output_items: list[dict[str, Any]],
    run_inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    selected_branch = _selected_result_branch_ref(setup, result_field, run_inputs)
    if selected_branch is None:
        return []
    selected_node = str(selected_branch[0])
    return [item for item in output_items if item.get("nodeId") == selected_node]


def _prune_prompt_for_execution_map(
    setup: dict[str, Any],
    prompt: dict[str, Any],
    run_inputs: dict[str, Any],
) -> dict[str, Any] | None:
    execution_map = setup.get("executionMap")
    if not isinstance(execution_map, dict) or execution_map.get("version") != 1:
        return None
    routers = execution_map.get("routers")
    if not isinstance(routers, list):
        return None

    roots: set[str] = set()
    for router in routers:
        if not isinstance(router, dict):
            continue
        router_node = str(router.get("node", ""))
        selected_value = _execution_map_selected_value(setup, router, run_inputs)
        if selected_value is None:
            continue
        branches = router.get("branches")
        branch = branches.get(str(selected_value)) if isinstance(branches, dict) else None
        if not isinstance(branch, dict):
            continue
        writer_nodes = branch.get("writerNodes")
        if isinstance(writer_nodes, list):
            roots.update(str(node_id) for node_id in writer_nodes if str(node_id) in prompt)
        if not writer_nodes and router_node in prompt:
            roots.add(router_node)
    if not roots:
        return None

    return _prune_prompt_from_roots(
        prompt,
        roots,
        _selected_app_switches(setup, prompt, run_inputs),
    )


def _execution_map_selected_value(
    setup: dict[str, Any],
    router: dict[str, Any],
    run_inputs: dict[str, Any],
) -> int | None:
    key = router.get("switchKey")
    app = setup.get("setupSurface", {}).get("app", {})
    switch = app.get("switch") if isinstance(app, dict) else None
    if isinstance(switch, dict) and switch.get("key") == key:
        return _selected_switch_value(switch, run_inputs)
    selected = run_inputs.get(key) if isinstance(key, str) else None
    if isinstance(selected, bool):
        return None
    if isinstance(selected, int):
        return selected
    if isinstance(selected, str) and selected.isdigit():
        return int(selected)
    return None


def _validate_execution_map_inputs(setup: dict[str, Any], run_inputs: dict[str, Any]) -> list[str]:
    execution_map = setup.get("executionMap")
    if not isinstance(execution_map, dict) or execution_map.get("version") != 1:
        return []
    routers = execution_map.get("routers")
    if not isinstance(routers, list):
        return []
    app = setup.get("setupSurface", {}).get("app", {})
    switch = app.get("switch") if isinstance(app, dict) else None
    errors: list[str] = []
    for router in routers:
        if not isinstance(router, dict):
            continue
        selected_value = _execution_map_selected_value(setup, router, run_inputs)
        key = router.get("switchKey")
        if selected_value is None:
            if isinstance(switch, dict) and switch.get("key") == key and isinstance(key, str):
                if key not in run_inputs:
                    errors.append(_switch_value_error(key, switch))
            elif isinstance(key, str):
                errors.append(f"input '{key}' must select a valid execution branch")
            continue
        branches = router.get("branches")
        if not isinstance(branches, dict) or str(selected_value) not in branches:
            errors.append(f"input '{key}' selects branch {selected_value}, but this setup has no execution branch for it")
    return errors


def _prune_prompt_for_selected_app_results(
    setup: dict[str, Any],
    prompt: dict[str, Any],
    run_inputs: dict[str, Any],
) -> dict[str, Any]:
    result_switches = _selected_result_switches(setup, prompt, run_inputs)
    if not result_switches:
        return prompt
    selected_switches = _selected_app_switches(setup, prompt, run_inputs)

    roots = {
        node_id
        for node_id, _selected_value in result_switches.values()
        if node_id in prompt
    }
    for field in _app_result_fields(setup):
        target = field.get("target")
        if isinstance(target, dict) and str(target.get("node")) in prompt:
            roots.add(str(target["node"]))

    selected_branch_nodes = {
        str(prompt[switch_node_id]["inputs"][f"value{selected_value}"][0])
        for switch_node_id, (_node_id, selected_value) in result_switches.items()
        if (
            switch_node_id in prompt
            and isinstance(prompt[switch_node_id], dict)
            and isinstance(prompt[switch_node_id].get("inputs"), dict)
            and _is_api_ref(prompt[switch_node_id]["inputs"].get(f"value{selected_value}"))
        )
    }
    roots.update(
        _selected_output_descendant_roots(
            prompt,
            selected_branch_nodes,
            _setup_output_surface_node_ids(setup),
        )
    )
    if not roots:
        return prompt

    return _prune_prompt_from_roots(prompt, roots, selected_switches)


def _prune_prompt_from_roots(
    prompt: dict[str, Any],
    roots: set[str],
    selected_switches: dict[str, tuple[str, int]],
) -> dict[str, Any]:
    keep: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in keep or node_id not in prompt:
            return
        keep.add(node_id)
        node = prompt.get(node_id)
        if not isinstance(node, dict):
            return
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            return
        selected_value = selected_switches.get(node_id, (None, None))[1]
        for input_name, value in inputs.items():
            if (
                selected_value is not None
                and input_name.startswith("value")
                and input_name != f"value{selected_value}"
            ):
                continue
            for ref in _api_refs(value):
                visit(str(ref[0]))

    for root in roots:
        visit(root)
    if not keep:
        return prompt
    pruned = {node_id: deepcopy(node) for node_id, node in prompt.items() if node_id in keep}
    for node_id, (_switch_node_id, selected_value) in selected_switches.items():
        node = pruned.get(node_id)
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        node["inputs"] = {
            key: value
            for key, value in inputs.items()
            if not key.startswith("value") or key == f"value{selected_value}"
        }
    return pruned


def _selected_result_switches(
    setup: dict[str, Any],
    api_prompt: dict[str, Any],
    run_inputs: dict[str, Any],
) -> dict[str, tuple[str, int]]:
    app = setup.get("setupSurface", {}).get("app", {})
    if not isinstance(app, dict):
        return {}
    switch = app.get("switch")
    if not isinstance(switch, dict):
        return {}
    selected_value = _selected_switch_value(switch, run_inputs)
    if selected_value is None:
        return {}
    selected: dict[str, tuple[str, int]] = {}
    for field in _app_result_fields(setup):
        result_node_id = _result_switch_node_id(setup, field, run_inputs, api_prompt)
        if result_node_id is not None:
            selected[result_node_id] = (result_node_id, selected_value)
    return selected


def _selected_app_switches(
    setup: dict[str, Any],
    api_prompt: dict[str, Any],
    run_inputs: dict[str, Any],
) -> dict[str, tuple[str, int]]:
    app = setup.get("setupSurface", {}).get("app", {})
    if not isinstance(app, dict):
        return {}
    switch = app.get("switch")
    if not isinstance(switch, dict):
        return {}
    switch_target = switch.get("target")
    if not isinstance(switch_target, dict):
        return {}
    selected_value = _selected_switch_value(switch, run_inputs)
    if selected_value is None:
        return {}

    selected: dict[str, tuple[str, int]] = {}
    for node_id, node in api_prompt.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        selector_ref = _switch_selector_ref(inputs)
        if not _selector_matches_switch_target(selector_ref, switch_target, api_prompt):
            continue
        if _is_api_ref(inputs.get(f"value{selected_value}")):
            selected[str(node_id)] = (str(node_id), selected_value)
    return selected


def _selected_output_descendant_roots(
    api_prompt: dict[str, Any],
    selected_branch_nodes: set[str],
    output_surface_node_ids: set[str],
) -> set[str]:
    if not selected_branch_nodes or not output_surface_node_ids:
        return set()
    children_by_node = _prompt_children_by_node(api_prompt)
    roots: set[str] = set()
    seen: set[str] = set()
    pending = list(selected_branch_nodes)
    while pending:
        node_id = pending.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        for child_id in children_by_node.get(node_id, set()):
            if child_id in output_surface_node_ids and child_id not in selected_branch_nodes:
                roots.add(child_id)
            pending.append(child_id)
    return roots


def _prompt_children_by_node(api_prompt: dict[str, Any]) -> dict[str, set[str]]:
    children: dict[str, set[str]] = {}
    for node_id, node in api_prompt.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for value in inputs.values():
            for ref in _api_refs(value):
                children.setdefault(str(ref[0]), set()).add(str(node_id))
    return children


def _setup_output_surface_node_ids(setup: dict[str, Any]) -> set[str]:
    setup_surface = setup.get("setupSurface", {})
    if not isinstance(setup_surface, dict):
        return set()
    output_ids: set[str] = set()
    outputs = setup_surface.get("outputs")
    if not isinstance(outputs, list):
        return output_ids
    for group in outputs:
        if not isinstance(group, dict) or not isinstance(group.get("nodes"), list):
            continue
        for node in group["nodes"]:
            if isinstance(node, dict) and node.get("id") is not None:
                output_ids.add(str(node["id"]))
    return output_ids


def _app_result_fields(setup: dict[str, Any]) -> list[dict[str, Any]]:
    app = setup.get("setupSurface", {}).get("app", {})
    if not isinstance(app, dict) or not isinstance(app.get("results"), list):
        return []
    return [field for field in app["results"] if isinstance(field, dict)]


def _result_switch_node_id(
    setup: dict[str, Any],
    result_field: dict[str, Any],
    run_inputs: dict[str, Any],
    api_prompt: dict[str, Any] | None = None,
) -> str | None:
    app = setup.get("setupSurface", {}).get("app", {})
    if not isinstance(app, dict):
        return None
    switch = app.get("switch")
    if not isinstance(switch, dict):
        return None
    selected_value = _selected_switch_value(switch, run_inputs)
    if selected_value is None:
        return None

    api_prompt = api_prompt if isinstance(api_prompt, dict) else setup.get("apiPrompt")
    if not isinstance(api_prompt, dict):
        return None
    target = result_field.get("target")
    if not isinstance(target, dict):
        return None
    result_node = api_prompt.get(str(target.get("node")))
    if not isinstance(result_node, dict):
        return None
    result_inputs = result_node.get("inputs")
    if not isinstance(result_inputs, dict):
        return None
    result_ref = result_inputs.get(str(target.get("input")))
    if not _is_api_ref(result_ref):
        return None

    switch_node_id = str(result_ref[0])
    switch_node = api_prompt.get(switch_node_id)
    if not isinstance(switch_node, dict):
        return None
    switch_inputs = switch_node.get("inputs")
    if not isinstance(switch_inputs, dict):
        return None
    selector_ref = _switch_selector_ref(switch_inputs)
    switch_target = switch.get("target")
    if isinstance(switch_target, dict) and not _selector_matches_switch_target(
        selector_ref,
        switch_target,
        api_prompt,
    ):
        return None
    if not _is_api_ref(switch_inputs.get(f"value{selected_value}")):
        return None
    return switch_node_id


def _selected_result_branch_ref(
    setup: dict[str, Any],
    result_field: dict[str, Any],
    run_inputs: dict[str, Any],
) -> list[Any] | None:
    api_prompt = setup.get("apiPrompt")
    if not isinstance(api_prompt, dict):
        return None
    switch_node_id = _result_switch_node_id(setup, result_field, run_inputs, api_prompt)
    if switch_node_id is None:
        return None
    switch_node = api_prompt.get(switch_node_id)
    if not isinstance(switch_node, dict):
        return None
    switch_inputs = switch_node.get("inputs")
    if not isinstance(switch_inputs, dict):
        return None
    switch = setup.get("setupSurface", {}).get("app", {}).get("switch")
    selected_value = _selected_switch_value(switch, run_inputs) if isinstance(switch, dict) else None
    if selected_value is None:
        return None
    branch_ref = switch_inputs.get(f"value{selected_value}")
    return branch_ref if _is_api_ref(branch_ref) else None


def _selected_switch_value(switch: dict[str, Any], run_inputs: dict[str, Any]) -> int | None:
    key = switch.get("key")
    selected = run_inputs.get(key) if isinstance(key, str) and key in run_inputs else switch.get("default")
    return _switch_option_value(switch, selected)


def _selector_matches_switch_target(selector_ref: Any, switch_target: dict[str, Any], api_prompt: dict[str, Any]) -> bool:
    if not _is_api_ref(selector_ref):
        return False
    source_node_id = str(selector_ref[0])
    target_node_id = str(switch_target.get("node"))
    if source_node_id != target_node_id:
        return False
    source_node = api_prompt.get(source_node_id)
    if not isinstance(source_node, dict):
        return False
    outputs = _publish_input_output_slots(source_node)
    return outputs.get("switch") == int(selector_ref[1])


def _switch_selector_ref(switch_inputs: dict[str, Any]) -> Any:
    if "select" in switch_inputs:
        return switch_inputs.get("select")
    return switch_inputs.get("index")


def _publish_input_output_slots(api_node: dict[str, Any]) -> dict[str, int]:
    if api_node.get("class_type") != "Koolook_PublishInput":
        return {}
    return {
        "sequence_folder": 0,
        "qt_file": 1,
        "single_file": 2,
        "prompt": 3,
        "switch": 4,
    }


def _resolve_prompt_value(prompt: dict[str, Any], value: Any) -> Any:
    if not _is_api_ref(value):
        return value
    node_id = str(value[0])
    output_index = int(value[1])
    node = prompt.get(node_id)
    if not isinstance(node, dict):
        return None
    class_type = node.get("class_type")
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return None
    if class_type == "Koolook_PublishOutput":
        values = (
            inputs.get("folder"),
            inputs.get("name"),
            inputs.get("version"),
        )
        return values[output_index] if output_index < len(values) else None
    if class_type == "EasyAIPipeline":
        if "WRITE_file_path" in inputs:
            values = (
                inputs.get("WRITE_file_path"),
                inputs.get("output_name", ""),
                inputs.get("version_string", ""),
                inputs.get("output_directory", ""),
                inputs.get("shot_duration", 0),
                inputs.get("seed_value", 0),
                inputs.get("shot_name", ""),
            )
            return values[output_index] if output_index < len(values) else None
        resolved_inputs = {
            key: _resolve_prompt_value(prompt, input_value)
            for key, input_value in inputs.items()
        }
        try:
            try:
                from .k_ai_pipeline import build_pipeline_outputs
            except ImportError:  # pragma: no cover - standalone test/import context
                from k_ai_pipeline import build_pipeline_outputs

            values = build_pipeline_outputs(
                **resolved_inputs,
                create_directory=False,
                check_overwrite=False,
            )
        except Exception:
            return None
        return values[output_index] if output_index < len(values) else None
    return None


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


def _flatten_history_outputs(raw_outputs: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_outputs, dict):
        return []
    items: list[dict[str, Any]] = []
    for node_id, node_outputs in raw_outputs.items():
        if not isinstance(node_outputs, dict):
            continue
        for kind, values in node_outputs.items():
            if not isinstance(values, list):
                continue
            for value in values:
                item = {"nodeId": str(node_id), "kind": str(kind)}
                if isinstance(value, dict):
                    item.update(value)
                else:
                    item["value"] = value
                items.append(item)
    return items


def _history_output_matches_contract(item: dict[str, Any], contract_type: str) -> bool:
    kind = str(item.get("kind", "")).lower()
    if contract_type == "video":
        return kind in {"video", "videos", "animated"}
    if contract_type == "image":
        return kind in {"image", "images"}
    return True
