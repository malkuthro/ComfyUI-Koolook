# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the loop status pass-through node."""

from __future__ import annotations

import json

import pytest

from k_loop_status import KoolookLoopStatus, _post_prompt, build_status, infer_index_node_id


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
