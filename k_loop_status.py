# SPDX-License-Identifier: GPL-3.0-or-later
#
# ComfyUI-Koolook - loop status utilities
# Copyright (C) 2026 ComfyUI-Koolook contributors (kforgelabs).

"""Loop-body status pass-through nodes."""

from __future__ import annotations

import copy
import json
import threading
import time
import urllib.request


_ACTIVE_QUEUE_KEYS: set[str] = set()


def _format_write_path(filepath: str, frame: int) -> str:
    text = str(filepath or "").strip()
    if "%04d" in text:
        return text.replace("%04d", f"{frame:04d}")
    return text


def build_status(label: str, index: int, total: int, filepath: str = "") -> str:
    total = max(1, int(total))
    frame = int(index)
    position = max(1, min(total, frame + 1))
    label = str(label or "loop").strip() or "loop"
    path = _format_write_path(filepath, frame)
    if path:
        return f"{label}: {position}/{total} frame {frame} -> {path}"
    return f"{label}: {position}/{total} frame {frame}"


def _infer_index_node_id(prompt: dict | None, unique_id) -> str:
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


def _post_prompt(server_url: str, prompt: dict) -> None:
    data = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url.rstrip('/')}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        body = response.read().decode("utf-8", errors="replace")
    print(f"[Koolook Loop Status] queued next prompt: {body}")


def _queue_next_prompt(
    *,
    prompt: dict,
    index_node_id: str,
    next_index: int,
    server_url: str,
    queue_key: str,
) -> None:
    try:
        child = copy.deepcopy(prompt)
        node = child.get(str(index_node_id))
        if not isinstance(node, dict):
            raise RuntimeError(f"index node id {index_node_id!r} is not in prompt")
        inputs = node.setdefault("inputs", {})
        inputs["value"] = int(next_index)
        time.sleep(0.05)
        _post_prompt(server_url, child)
    except Exception as exc:
        print(f"[Koolook Loop Status] failed to queue next prompt: {exc}")
    finally:
        _ACTIVE_QUEUE_KEYS.discard(queue_key)


class KoolookLoopStatus:
    """Print per-iteration loop progress and pass the value through."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*",),
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
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("*", "STRING")
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
        if not index_node_id:
            index_node_id = _infer_index_node_id(prompt, unique_id)
        status = build_status(label or "loop", index, total, filepath)
        print(f"[Koolook Loop Status] {status}")
        if bool(auto_queue_next) and next_index < total:
            if not isinstance(prompt, dict):
                raise RuntimeError("Koolook Loop Status needs hidden PROMPT data.")
            if not index_node_id:
                raise RuntimeError("Set index_node_id to the easy int frame index node.")
            queue_key = f"{unique_id or 'loop-status'}:{frame}->{next_index}"
            if queue_key not in _ACTIVE_QUEUE_KEYS:
                _ACTIVE_QUEUE_KEYS.add(queue_key)
                thread = threading.Thread(
                    target=_queue_next_prompt,
                    kwargs={
                        "prompt": prompt,
                        "index_node_id": index_node_id,
                        "next_index": next_index,
                        "server_url": str(server_url or "http://127.0.0.1:8188"),
                        "queue_key": queue_key,
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
