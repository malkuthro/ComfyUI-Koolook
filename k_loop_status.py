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
import urllib.parse
import urllib.request


LOGGER = logging.getLogger(__name__)
MAX_AUTO_QUEUE_DEPTH = 1000
DEFAULT_SERVER_URL = "http://127.0.0.1:8188"
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


def _prompt_has_node(prompt: dict | None, node_id: str) -> bool:
    return isinstance(prompt, dict) and isinstance(prompt.get(str(node_id)), dict)


def _describe_prompt_node(prompt: dict | None, node_id: str) -> str:
    if not isinstance(prompt, dict):
        return f"node {node_id}"
    node = prompt.get(str(node_id))
    if not isinstance(node, dict):
        return f"node {node_id}"
    class_type = str(node.get("class_type") or node.get("type") or "").strip()
    title = str(node.get("_meta", {}).get("title") or node.get("title") or "").strip()
    if title and title != class_type:
        prefix = f"{title} ({class_type})" if class_type else title
    else:
        prefix = class_type
    return f"{prefix} node {node_id}" if prefix else f"node {node_id}"


def resolve_index_node_id(
    prompt: dict | None,
    unique_id,
    configured_index_node_id: str,
) -> tuple[str, str]:
    """Pick the frame-index node to advance and a human note about the choice.

    Prefers a configured id that actually exists in the prompt; otherwise falls
    back to the node feeding the connected ``index`` input (self-healing a stale
    or shifted manual id). Returns ``("", "")`` when nothing resolves so the
    caller can raise synchronously with an actionable message.
    """
    configured = str(configured_index_node_id or "").strip()
    inferred = infer_index_node_id(prompt, unique_id)
    if configured and _prompt_has_node(prompt, configured):
        return configured, f"using configured {_describe_prompt_node(prompt, configured)}"
    if inferred:
        if configured and configured != inferred:
            return (
                inferred,
                "configured index node "
                f"{configured!r} is not in this prompt; using connected "
                f"{_describe_prompt_node(prompt, inferred)}",
            )
        return inferred, f"using connected {_describe_prompt_node(prompt, inferred)}"
    if configured:
        return configured, f"using configured node {configured}"
    return "", ""


def _as_bool(value) -> bool:
    """Coerce saved widget values (incl. string booleans) to ``bool``.

    ComfyUI can persist a boolean widget as the string ``"true"``/``"false"``;
    ``bool("false")`` is truthy, so a naive cast would auto-queue when the user
    saved the toggle off.
    """
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _get_json(url: str, timeout: float = 10) -> dict:
    _validate_http_url(url)
    with urllib.request.urlopen(url, timeout=timeout) as response:  # nosec B310
        body = response.read().decode("utf-8", errors="replace")
    return json.loads(body) if body else {}


def _validate_http_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(str(url or ""))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(f"Only http(s) ComfyUI server URLs are allowed: {url!r}")


def _compose_server_url(host: str | None, port) -> str:
    """Join a detected host/port into a connectable ``http://`` URL.

    ComfyUI's bare ``--listen`` binds to all IPv4 and IPv6 interfaces as a
    comma-joined host value, so the host is split and any bind-all member means
    we connect over loopback. IPv6 literals are bracketed so ``urllib`` can parse
    the ``host:port`` netloc (``::1`` -> ``http://[::1]:port``).
    """
    members = [member.strip() for member in str(host or "").split(",") if member.strip()]
    bind_all = {"0.0.0.0", "::", "*"}  # nosec B104
    if any(member in bind_all for member in members):
        host = "127.0.0.1"
    else:
        host = members[0] if members else "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def _detect_local_server_url() -> str | None:
    """Best-effort URL of the ComfyUI server hosting this node.

    The node default points at ComfyUI's default port (8188), but installs
    launched with ``--port`` (or ``--listen``) bind elsewhere — probing the
    stale default is actively refused and aborts the loop. Prefer the address
    the running server actually bound to. Returns ``None`` when neither the
    parsed CLI args nor the running ``PromptServer`` can be inspected (e.g. a
    headless unit test), so the caller can fall back to the literal default.
    """
    port = None
    host = None
    try:
        from comfy.cli_args import args  # type: ignore[import-not-found]

        port = getattr(args, "port", None)
        host = getattr(args, "listen", None)
    except Exception:  # pragma: no cover - depends on ComfyUI runtime
        LOGGER.debug("comfy.cli_args server detection failed", exc_info=True)
    if not port:
        try:
            from server import PromptServer  # type: ignore[import-not-found]

            instance = PromptServer.instance
            port = port or getattr(instance, "port", None)
            host = host or getattr(instance, "address", None)
        except Exception:  # pragma: no cover - depends on ComfyUI runtime
            LOGGER.debug("PromptServer server detection failed", exc_info=True)
    if not port:
        return None
    return _compose_server_url(host, port)


def _resolve_server_url(server_url: str) -> str:
    """Return the URL to queue against, auto-detecting when left at default.

    A custom URL is respected verbatim; an empty value or the baked-in
    default is replaced with the running server's real address when it can
    be detected, so workflows saved on the default port still queue against
    an install launched with ``--port``.
    """
    server_url = str(server_url or "").strip()
    if server_url and server_url != DEFAULT_SERVER_URL:
        return server_url
    detected = _detect_local_server_url()
    if detected and detected != server_url:
        print(f"[Koolook Loop Status] resolved server_url to {detected}")
    return detected or DEFAULT_SERVER_URL


def _probe_server(server_url: str) -> None:
    _get_json(f"{server_url.rstrip('/')}/system_stats", timeout=10)


def _post_prompt(server_url: str, prompt: dict) -> dict:
    _validate_http_url(server_url)
    data = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url.rstrip('/')}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as response:  # nosec B310
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
                    {"default": DEFAULT_SERVER_URL, "multiline": False},
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
        server_url=DEFAULT_SERVER_URL,
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
        # Resolve the frame-index node to advance: prefer a configured id that
        # exists, else the node feeding the connected `index` input — so a stale
        # or mis-shifted index_node_id self-heals instead of aborting the loop in
        # the background queue thread. `index_note` records which node was used.
        index_node_id, index_note = resolve_index_node_id(prompt, unique_id, index_node_id)
        max_depth = max(1, min(int(max_auto_queue_depth), MAX_AUTO_QUEUE_DEPTH))
        remaining_depth = int(remaining_auto_queue_depth)
        if remaining_depth < 0:
            remaining_depth = max_depth
        should_queue = _as_bool(auto_queue_next) and next_index < total
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
            if index_node_id not in prompt:
                raise RuntimeError(
                    f"index_node_id {index_node_id!r} is not a node in this workflow. "
                    "Connect the loop status node's index input to your easy int "
                    "frame-index node, or set index_node_id to that node's id."
                )
            server_url = _resolve_server_url(server_url)
            if not server_url:
                raise RuntimeError("Set server_url to the running ComfyUI server.")
            try:
                _probe_server(server_url)
            except (OSError, urllib.error.URLError, TimeoutError) as exc:
                raise RuntimeError(f"ComfyUI server is not reachable: {server_url}") from exc
        status = build_status(label or "loop", index, total, filepath)
        print(f"[Koolook Loop Status] {status}")
        if should_queue and index_note:
            print(f"[Koolook Loop Status] {index_note}")
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
