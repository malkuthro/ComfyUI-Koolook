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


class InMemorySetupRunStore:
    """Small run-id mapping store for a running ComfyUI process."""

    def __init__(self) -> None:
        self._next_id = 1
        self._records: dict[str, RunRecord] = {}

    def create(self, *, setup_id: str, prompt_id: str) -> RunRecord:
        run_id = f"run-{self._next_id:06d}"
        self._next_id += 1
        record = RunRecord(
            run_id=run_id,
            setup_id=setup_id,
            prompt_id=prompt_id,
            status="queued",
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
        if errors:
            raise SetupRunError(400, errors)

        for key, value in inputs.items():
            field = input_fields[key]
            target = field["target"]
            prompt[str(target["node"])]["inputs"][str(target["input"])] = value

        try:
            queued = await self._comfy_client.queue_prompt(prompt)
        except SetupRunError:
            raise
        except Exception as exc:
            raise SetupRunError(502, [f"ComfyUI queue request failed: {exc}"]) from exc

        prompt_id = queued.get("prompt_id") if isinstance(queued, dict) else None
        if not isinstance(prompt_id, str) or not prompt_id:
            raise SetupRunError(502, ["ComfyUI queue response did not include prompt_id"])

        record = self._run_store.create(setup_id=setup_id, prompt_id=prompt_id)
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
                "outputs": _summarize_outputs(setup, history_entry.get("outputs", {})),
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
    for key, field in input_fields.items():
        if field.get("required") is True and key not in inputs:
            errors.append(f"required input '{key}' is missing")
    return errors


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
        return "queued"
    if _queue_contains_prompt(queue.get("queue_running", []), prompt_id):
        return "running"
    if _queue_contains_prompt(queue.get("queue_pending", []), prompt_id):
        return "queued"
    return "queued"


def _queue_contains_prompt(entries: Any, prompt_id: str) -> bool:
    if not isinstance(entries, list):
        return False
    for entry in entries:
        if isinstance(entry, dict) and entry.get("prompt_id") == prompt_id:
            return True
        if isinstance(entry, list) and prompt_id in entry:
            return True
    return False


def _summarize_outputs(setup: dict[str, Any], raw_outputs: Any) -> list[dict[str, Any]]:
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
    summaries.extend(_summarize_app_surface_outputs(setup, output_items))
    return summaries


def _summarize_app_surface_outputs(
    setup: dict[str, Any],
    output_items: list[dict[str, Any]],
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
            summary = {
                "key": field.get("key", ""),
                "label": field.get("label", field.get("key", "")),
                "type": field_type,
                "visible": field.get("visible", True),
                "target": target if isinstance(target, dict) else {},
                "default": field.get("default"),
                "items": [
                    item
                    for item in output_items
                    if target_node and item.get("nodeId") == target_node
                ],
            }
            summaries.append(summary)
    return summaries


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
