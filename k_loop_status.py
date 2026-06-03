# SPDX-License-Identifier: GPL-3.0-or-later
#
# ComfyUI-Koolook - loop status utilities
# Copyright (C) 2026 ComfyUI-Koolook contributors (kforgelabs).

"""Loop-body status pass-through nodes."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
import re
import threading
import traceback
import urllib.error
import urllib.request


LOGGER = logging.getLogger(__name__)
MAX_AUTO_QUEUE_DEPTH = 1000
_ACTIVE_QUEUE_KEYS: set[str] = set()
_ACTIVE_QUEUE_KEYS_LOCK = threading.Lock()


class AnyType(str):
    """ComfyUI wildcard type that compares as compatible with any socket."""

    def __ne__(self, _other):
        return False


ANY_TYPE = AnyType("*")


def _format_write_path(filepath: str, frame: int) -> str:
    text = str(filepath or "").strip()
    return re.sub(r"%0?(\d*)d", lambda m: f"{frame:0{m.group(1) or '0'}d}", text)


def build_status(label: str, index: int, total: int, filepath: str = "") -> str:
    total = max(1, int(total))
    frame = int(index)
    position = max(1, min(total, frame + 1))
    label = str(label or "loop").strip() or "loop"
    path = _format_write_path(filepath, frame)
    if path:
        return f"{label}: {position}/{total} frame {frame} -> {path}"
    return f"{label}: {position}/{total} frame {frame}"


def infer_index_node_id(prompt: dict | None, unique_id) -> str:
    if not isinstance(prompt, dict) or unique_id is None:
        return ""
    node = prompt.get(str(unique_id))
    if not isinstance(node, dict):
        return ""
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return ""
    link = inputs.get("index")
    if isinstance(link, (list, tuple)) and link:
        return str(link[0])
    return ""


_infer_index_node_id = infer_index_node_id


def _get_json(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    return json.loads(body) if body else {}


def _probe_server(server_url: str) -> None:
    _get_json(f"{server_url.rstrip('/')}/system_stats", timeout=10)


def _post_prompt(server_url: str, prompt: dict) -> dict:
    data = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url.rstrip('/')}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        body = response.read().decode("utf-8", errors="replace")
    payload = json.loads(body) if body else {}
    if payload.get("error") or not payload.get("prompt_id"):
        raise RuntimeError(f"ComfyUI rejected child prompt: {payload}")
    print(f"[Koolook Loop Status] queued next prompt: {body}")
    return payload


def _abort_marker_path(filepath: str, frame: int) -> Path:
    expected = Path(_format_write_path(filepath, frame))
    parent = expected.parent if str(expected.parent) else Path.cwd()
    return parent / f"_loop_aborted_at_frame_{frame}.txt"


def _write_abort_marker(filepath: str, frame: int, exc: BaseException) -> None:
    marker = _abort_marker_path(filepath, frame)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        "Koolook Loop Status failed to queue the next prompt.\n\n"
        f"Frame: {frame}\n"
        f"Error: {exc}\n\n"
        f"{traceback.format_exc()}",
        encoding="utf-8",
    )
    print(f"[Koolook Loop Status] wrote abort marker: {marker}")


def _queue_next_prompt(
    *,
    prompt: dict,
    index_node_id: str,
    next_index: int,
    server_url: str,
    queue_key: str,
    filepath: str,
    remaining_auto_queue_depth: int,
) -> None:
    try:
        child = copy.deepcopy(prompt)
        node = child.get(str(index_node_id))
        if not isinstance(node, dict):
            raise RuntimeError(f"index node id {index_node_id!r} is not in prompt")
        inputs = node.setdefault("inputs", {})
        inputs["value"] = int(next_index)
        status_node = child.get(str(queue_key.split(":", 1)[0]))
        if isinstance(status_node, dict):
            status_inputs = status_node.setdefault("inputs", {})
            status_inputs["remaining_auto_queue_depth"] = int(remaining_auto_queue_depth)
        _post_prompt(server_url, child)
    except Exception as exc:
        LOGGER.exception("Koolook Loop Status failed to queue next prompt")
        print(f"[Koolook Loop Status] failed to queue next prompt: {exc}")
        _write_abort_marker(filepath, next_index, exc)
    finally:
        with _ACTIVE_QUEUE_KEYS_LOCK:
            _ACTIVE_QUEUE_KEYS.discard(queue_key)


class KoolookLoopStatus:
    """Print per-iteration loop progress and pass the value through."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (ANY_TYPE,),
                "index": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "total": ("INT", {"default": 1, "min": 1, "max": 100000}),
            },
            "optional": {
                "filepath": ("STRING", {"default": "", "multiline": True}),
                "label": ("STRING", {"default": "loop", "multiline": False}),
                "auto_queue_next": ("BOOLEAN", {"default": False}),
                "index_node_id": ("STRING", {"default": "", "multiline": False}),
                "server_url": (
                    "STRING",
                    {"default": "http://127.0.0.1:8188", "multiline": False},
                ),
                "max_auto_queue_depth": (
                    "INT",
                    {"default": 100, "min": 1, "max": MAX_AUTO_QUEUE_DEPTH},
                ),
                "remaining_auto_queue_depth": (
                    "INT",
                    {"default": -1, "min": -1, "max": MAX_AUTO_QUEUE_DEPTH},
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (ANY_TYPE, "STRING")
    RETURN_NAMES = ("value", "status")
    FUNCTION = "report"
    CATEGORY = "Koolook/Loop"

    def report(
        self,
        value,
        index,
        total,
        filepath="",
        label="loop",
        auto_queue_next=False,
        index_node_id="",
        server_url="http://127.0.0.1:8188",
        max_auto_queue_depth=100,
        remaining_auto_queue_depth=-1,
        prompt=None,
        unique_id=None,
    ):
        frame = int(index)
        total = max(1, int(total))
        next_index = frame + 1
        index_node_id = str(index_node_id or "").strip()
        label = str(label or "").strip()
        if not index_node_id and label.isdigit():
            index_node_id = label
            label = "EXR_SAFE"
            print(
                "[Koolook Loop Status] numeric label treated as index_node_id; "
                "using label EXR_SAFE"
            )
        if not index_node_id:
            index_node_id = infer_index_node_id(prompt, unique_id)
        max_depth = max(1, min(int(max_auto_queue_depth), MAX_AUTO_QUEUE_DEPTH))
        remaining_depth = int(remaining_auto_queue_depth)
        if remaining_depth < 0:
            remaining_depth = max_depth
        should_queue = bool(auto_queue_next) and next_index < total
        server_url = str(server_url or "").strip()
        if should_queue:
            if total - frame - 1 > max_depth:
                raise RuntimeError(
                    f"Refusing to auto-queue {total - frame - 1} remaining prompts; "
                    f"max_auto_queue_depth is {max_depth}."
                )
            if remaining_depth <= 0:
                raise RuntimeError("Auto-queue depth exhausted before loop completed.")
            if not isinstance(prompt, dict):
                raise RuntimeError("Koolook Loop Status needs hidden PROMPT data.")
            if unique_id is None:
                raise RuntimeError("Koolook Loop Status needs hidden UNIQUE_ID data.")
            if not index_node_id:
                raise RuntimeError("Set index_node_id to the easy int frame index node.")
            if not server_url:
                raise RuntimeError("Set server_url to the running ComfyUI server.")
            try:
                _probe_server(server_url)
            except (OSError, urllib.error.URLError, TimeoutError) as exc:
                raise RuntimeError(f"ComfyUI server is not reachable: {server_url}") from exc
        status = build_status(label or "loop", index, total, filepath)
        print(f"[Koolook Loop Status] {status}")
        if should_queue:
            queue_key = f"{unique_id or 'loop-status'}:{frame}->{next_index}"
            queued = False
            with _ACTIVE_QUEUE_KEYS_LOCK:
                if queue_key not in _ACTIVE_QUEUE_KEYS:
                    _ACTIVE_QUEUE_KEYS.add(queue_key)
                    queued = True
            if queued:
                thread = threading.Thread(
                    target=_queue_next_prompt,
                    kwargs={
                        "prompt": prompt,
                        "index_node_id": index_node_id,
                        "next_index": next_index,
                        "server_url": server_url,
                        "queue_key": queue_key,
                        "filepath": str(filepath or ""),
                        "remaining_auto_queue_depth": remaining_depth - 1,
                    },
                    daemon=True,
                )
                thread.start()
                print(f"[Koolook Loop Status] queued next index {next_index}/{total - 1}")
        return (value, status)


NODE_CLASS_MAPPINGS = {
    "Koolook_LoopStatus": KoolookLoopStatus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koolook_LoopStatus": "Koolook Loop Status",
}
