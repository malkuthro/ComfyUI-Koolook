# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the loop status pass-through node."""

from __future__ import annotations

import json
import urllib.parse

import pytest

import k_loop_status
from k_loop_status import (
    DEFAULT_SERVER_URL,
    KoolookLoopStatus,
    _post_prompt,
    _resolve_server_url,
    build_status,
    infer_index_node_id,
)


@pytest.fixture(autouse=True)
def _clear_active_queue_keys():
    """Reset the module-global auto-queue dedup set between tests."""
    k_loop_status._ACTIVE_QUEUE_KEYS.clear()
    yield
    k_loop_status._ACTIVE_QUEUE_KEYS.clear()


def test_build_status_formats_one_based_position_and_frame_path():
    assert (
        build_status("write", 2, 4, "N:/out/frame.%04d.exr")
        == "write: 3/4 frame 2 -> N:/out/frame.0002.exr"
    )


def test_build_status_formats_other_padded_frame_patterns():
    assert (
        build_status("write", 12, 20, "N:/out/frame.%05d.exr")
        == "write: 13/20 frame 12 -> N:/out/frame.00012.exr"
    )


def test_report_prints_status_and_passes_value_through(capsys):
    value = object()
    node = KoolookLoopStatus()

    out_value, status = node.report(
        value,
        index=0,
        total=4,
        filepath="N:/out/frame.%04d.exr",
        label="EXR_SAFE",
    )

    assert out_value is value
    assert status == "EXR_SAFE: 1/4 frame 0 -> N:/out/frame.0000.exr"
    assert "[Koolook Loop Status] EXR_SAFE: 1/4" in capsys.readouterr().out


def test_registration_exports():
    from k_loop_status import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    assert NODE_CLASS_MAPPINGS["Koolook_LoopStatus"] is KoolookLoopStatus
    assert NODE_DISPLAY_NAME_MAPPINGS["Koolook_LoopStatus"] == "Koolook Loop Status"


def test_input_types_include_queue_controller_settings():
    optional = KoolookLoopStatus.INPUT_TYPES()["optional"]
    hidden = KoolookLoopStatus.INPUT_TYPES()["hidden"]

    assert "auto_queue_next" in optional
    assert "index_node_id" in optional
    assert "server_url" in optional
    assert "max_auto_queue_depth" in optional
    assert "remaining_auto_queue_depth" in optional
    assert hidden["prompt"] == "PROMPT"


def test_infers_index_node_id_from_connected_index_input():
    prompt = {"21": {"inputs": {"index": ["22", 0]}}}

    assert infer_index_node_id(prompt, "21") == "22"


def test_numeric_label_is_treated_as_shifted_index_node_id(capsys):
    node = KoolookLoopStatus()

    _value, status = node.report(
        "image",
        0,
        4,
        label="22",
        auto_queue_next=False,
    )

    assert status == "EXR_SAFE: 1/4 frame 0"
    assert "[Koolook Loop Status] EXR_SAFE: 1/4" in capsys.readouterr().out


def test_depth_guard_raises_before_status_print(capsys):
    node = KoolookLoopStatus()

    with pytest.raises(RuntimeError, match="Refusing to auto-queue"):
        node.report(
            "image",
            0,
            5,
            auto_queue_next=True,
            index_node_id="22",
            max_auto_queue_depth=2,
            prompt={"21": {"inputs": {}}},
        )

    assert "[Koolook Loop Status]" not in capsys.readouterr().out


def test_resolve_server_url_keeps_custom_value(monkeypatch):
    monkeypatch.setattr(
        k_loop_status, "_detect_local_server_url", lambda: "http://127.0.0.1:9999"
    )

    assert _resolve_server_url("http://10.0.0.5:7000") == "http://10.0.0.5:7000"


def test_resolve_server_url_overrides_default_with_detected_port(monkeypatch):
    monkeypatch.setattr(
        k_loop_status, "_detect_local_server_url", lambda: "http://127.0.0.1:8000"
    )

    assert _resolve_server_url(DEFAULT_SERVER_URL) == "http://127.0.0.1:8000"
    assert _resolve_server_url("") == "http://127.0.0.1:8000"


def test_resolve_server_url_falls_back_to_default_when_undetectable(monkeypatch):
    monkeypatch.setattr(k_loop_status, "_detect_local_server_url", lambda: None)

    assert _resolve_server_url(DEFAULT_SERVER_URL) == DEFAULT_SERVER_URL


def test_resolve_server_url_blank_falls_back_to_default_when_undetectable(monkeypatch):
    """Blank input must behave like the default when detection is unavailable."""
    monkeypatch.setattr(k_loop_status, "_detect_local_server_url", lambda: None)

    assert _resolve_server_url("") == DEFAULT_SERVER_URL


def test_compose_server_url_brackets_ipv6_literal():
    assert k_loop_status._compose_server_url("::1", 8000) == "http://[::1]:8000"
    assert (
        k_loop_status._compose_server_url("2001:db8::1", 8000)
        == "http://[2001:db8::1]:8000"
    )
    # Idempotent: an already-bracketed literal must not be double-bracketed.
    assert k_loop_status._compose_server_url("[::1]", 8000) == "http://[::1]:8000"


def test_compose_server_url_ipv6_result_passes_validation():
    """A bracketed IPv6 URL must parse through the pre-queue validator."""
    url = k_loop_status._compose_server_url("::1", 8000)

    k_loop_status._validate_http_url(url)  # must not raise
    assert urllib.parse.urlsplit(url).port == 8000


def test_compose_server_url_remaps_bind_all_to_localhost():
    assert k_loop_status._compose_server_url("0.0.0.0", 8000) == "http://127.0.0.1:8000"  # nosec B104
    assert k_loop_status._compose_server_url("::", 8000) == "http://127.0.0.1:8000"
    # ComfyUI's bare `--listen` is the comma-joined "all IPv4 and IPv6" value.
    assert (
        k_loop_status._compose_server_url("0.0.0.0,::", 8000)  # nosec B104
        == "http://127.0.0.1:8000"
    )
    # Order-independent: a bind-all member anywhere in the list routes to loopback.
    assert (
        k_loop_status._compose_server_url("::,0.0.0.0", 8000)  # nosec B104
        == "http://127.0.0.1:8000"
    )
    # No bind-all member: the first concrete host is used verbatim.
    assert (
        k_loop_status._compose_server_url("10.0.0.5,192.168.1.9", 8000)
        == "http://10.0.0.5:8000"
    )


def test_compose_server_url_keeps_ipv4_hostname_and_defaults_blank():
    assert k_loop_status._compose_server_url("127.0.0.1", 8000) == "http://127.0.0.1:8000"
    assert k_loop_status._compose_server_url("localhost", 8000) == "http://localhost:8000"
    assert k_loop_status._compose_server_url("", 8000) == "http://127.0.0.1:8000"


def test_probe_uses_detected_port_not_stale_default(monkeypatch):
    """Auto-queue on a non-default port must probe the running server."""
    monkeypatch.setattr(
        k_loop_status, "_detect_local_server_url", lambda: "http://127.0.0.1:8000"
    )
    probed = {}
    monkeypatch.setattr(
        k_loop_status, "_probe_server", lambda url: probed.update(url=url)
    )
    # Don't actually spawn the queue thread; we only care about the probe target.
    class _NoopThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(k_loop_status.threading, "Thread", _NoopThread)

    node = KoolookLoopStatus()
    node.report(
        "image",
        0,
        4,
        auto_queue_next=True,
        index_node_id="22",
        server_url=DEFAULT_SERVER_URL,
        prompt={"21": {"inputs": {}}, "22": {"inputs": {}}},
        unique_id="21",
    )

    assert probed["url"] == "http://127.0.0.1:8000"


def test_status_only_does_not_resolve_server_url(monkeypatch):
    """With auto-queue off, the node must not detect/probe the server URL."""
    called = {"detect": False}

    def _flag_detect():
        called["detect"] = True
        return None

    monkeypatch.setattr(k_loop_status, "_detect_local_server_url", _flag_detect)

    KoolookLoopStatus().report("image", 0, 4, auto_queue_next=False)

    assert called["detect"] is False


def test_stale_index_node_id_self_heals_from_connected_index(monkeypatch):
    """A shifted index_node_id like '0' falls back to the connected index node."""
    monkeypatch.setattr(
        k_loop_status, "_detect_local_server_url", lambda: "http://127.0.0.1:8000"
    )
    monkeypatch.setattr(k_loop_status, "_probe_server", lambda url: None)
    captured = {}

    class _NoopThread:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs.get("kwargs", {}))

        def start(self):
            pass

    monkeypatch.setattr(k_loop_status.threading, "Thread", _NoopThread)

    prompt = {
        "21": {"inputs": {"index": ["22", 0]}},
        "22": {"inputs": {"value": 0}},
    }
    _value, status = KoolookLoopStatus().report(
        "image",
        0,
        4,
        auto_queue_next=True,
        index_node_id="0",
        prompt=prompt,
        unique_id="21",
    )

    assert status.startswith("loop: 1/4")
    assert captured["index_node_id"] == "22"


def test_unknown_index_node_id_raises_synchronously():
    """An index_node_id with no matching node and no connection fails up front."""
    node = KoolookLoopStatus()

    with pytest.raises(RuntimeError, match="is not a node in this workflow"):
        node.report(
            "image",
            0,
            4,
            auto_queue_next=True,
            index_node_id="0",
            prompt={"21": {"inputs": {}}},
            unique_id="21",
        )


def test_as_bool_coerces_saved_string_booleans():
    assert k_loop_status._as_bool("true") is True
    assert k_loop_status._as_bool(" True ") is True
    assert k_loop_status._as_bool("1") is True
    assert k_loop_status._as_bool("yes") is True
    assert k_loop_status._as_bool("on") is True
    assert k_loop_status._as_bool("false") is False
    assert k_loop_status._as_bool("no") is False
    assert k_loop_status._as_bool("off") is False
    assert k_loop_status._as_bool("0") is False
    assert k_loop_status._as_bool("") is False
    assert k_loop_status._as_bool(True) is True
    assert k_loop_status._as_bool(0) is False


def test_resolve_index_node_id_prefers_configured_when_present():
    prompt = {"543": {"class_type": "easy int", "inputs": {"value": 0}}}

    node_id, note = k_loop_status.resolve_index_node_id(prompt, "21", "543")

    assert node_id == "543"
    assert "using configured easy int node 543" in note


def test_resolve_index_node_id_falls_back_from_stale_manual_id():
    prompt = {
        "21": {"inputs": {"index": ["543", 0]}},
        "543": {"class_type": "easy int", "inputs": {"value": 0}},
    }

    node_id, note = k_loop_status.resolve_index_node_id(prompt, "21", "22")

    assert node_id == "543"
    assert "configured index node '22' is not in this prompt" in note
    assert "easy int node 543" in note


def test_resolve_index_node_id_infers_when_blank():
    prompt = {"21": {"inputs": {"index": ["543", 0]}}, "543": {"inputs": {}}}

    node_id, note = k_loop_status.resolve_index_node_id(prompt, "21", "")

    assert node_id == "543"
    assert note.startswith("using connected")


def test_resolve_index_node_id_returns_empty_when_unresolvable():
    assert k_loop_status.resolve_index_node_id({"21": {"inputs": {}}}, "21", "") == ("", "")


def test_describe_prompt_node_without_class_type_is_not_doubled():
    assert k_loop_status._describe_prompt_node({"22": {"inputs": {}}}, "22") == "node 22"
    assert (
        k_loop_status._describe_prompt_node({"22": {"_meta": {"title": "Frame"}}}, "22")
        == "Frame node 22"
    )


def test_resolve_index_node_id_fallback_does_not_override_connected():
    """A recovered numeric label is last-resort; the connected wire still wins."""
    prompt = {
        "21": {"inputs": {"index": ["543", 0]}},
        "543": {"class_type": "easy int", "inputs": {"value": 0}},
        "22": {"class_type": "easy int", "inputs": {"value": 0}},
    }

    node_id, note = k_loop_status.resolve_index_node_id(prompt, "21", "", fallback_id="22")

    assert node_id == "543"
    assert "connected easy int node 543" in note


def test_resolve_index_node_id_uses_fallback_when_nothing_else_resolves():
    node_id, note = k_loop_status.resolve_index_node_id(None, None, "", fallback_id="22")

    assert node_id == "22"
    assert "recovered node 22" in note


def test_numeric_label_does_not_override_connected_index(monkeypatch):
    """User scenario: a node id stuck in `label` must not beat the wired index."""
    monkeypatch.setattr(k_loop_status, "_detect_local_server_url", lambda: None)
    monkeypatch.setattr(k_loop_status, "_probe_server", lambda url: None)
    captured = {}

    class _NoopThread:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs.get("kwargs", {}))

        def start(self):
            pass

    monkeypatch.setattr(k_loop_status.threading, "Thread", _NoopThread)

    prompt = {
        "21": {"inputs": {"index": ["543", 0]}},
        "543": {"class_type": "easy int", "inputs": {"value": 0}},
        "22": {"class_type": "easy int", "inputs": {"value": 0}},
    }
    _value, status = KoolookLoopStatus().report(
        "image",
        0,
        2,
        label="22",
        auto_queue_next=True,
        prompt=prompt,
        unique_id="21",
    )

    assert captured["index_node_id"] == "543"
    assert status.startswith("EXR_SAFE: 1/2")


def test_resolve_index_node_id_explicit_override_beats_connected_wire():
    """An explicit, valid index_node_id is a power-user override and wins over the wire."""
    prompt = {
        "21": {"inputs": {"index": ["543", 0]}},
        "543": {"class_type": "easy int", "inputs": {"value": 0}},
        "99": {"class_type": "easy int", "inputs": {"value": 0}},
    }

    node_id, note = k_loop_status.resolve_index_node_id(prompt, "21", "99")

    assert node_id == "99"
    assert "using configured easy int node 99" in note


def test_string_false_auto_queue_does_not_queue():
    """A saved 'false' string must not auto-queue (bool('false') is truthy)."""
    node = KoolookLoopStatus()

    _value, status = node.report(
        "image",
        0,
        4,
        auto_queue_next="false",
        index_node_id="22",
        prompt={"21": {"inputs": {}}},
        unique_id="21",
    )

    assert status == "loop: 1/4 frame 0"


def test_string_true_auto_queue_logs_detected_index_node(monkeypatch, capsys):
    """Saved 'true' enables auto-queue; the chosen index node class/id is logged."""
    monkeypatch.setattr(k_loop_status, "_detect_local_server_url", lambda: None)
    monkeypatch.setattr(k_loop_status, "_probe_server", lambda url: None)
    captured = {}

    class _NoopThread:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs.get("kwargs", {}))

        def start(self):
            pass

    monkeypatch.setattr(k_loop_status.threading, "Thread", _NoopThread)

    prompt = {
        "21": {"inputs": {"index": ["543", 0]}},
        "543": {"class_type": "easy int", "inputs": {"value": 0}},
    }
    _value, status = KoolookLoopStatus().report(
        "image",
        0,
        2,
        auto_queue_next="true",
        index_node_id="22",
        prompt=prompt,
        unique_id="21",
    )

    assert status.startswith("loop: 1/2")
    assert captured["index_node_id"] == "543"
    assert "easy int node 543" in capsys.readouterr().out


def test_post_prompt_rejects_error_payload(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"error": "bad prompt"}).encode("utf-8")

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())

    with pytest.raises(RuntimeError, match="rejected child prompt"):
        _post_prompt("http://127.0.0.1:8188", {})


def test_post_prompt_rejects_non_http_url():
    with pytest.raises(RuntimeError, match=r"Only http\(s\)"):
        _post_prompt("file:///tmp/comfy.sock", {})
