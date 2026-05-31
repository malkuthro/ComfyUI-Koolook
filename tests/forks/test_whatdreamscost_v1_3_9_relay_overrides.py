"""Tests for the LTXDirector ``relay_overrides`` widget parser.

These guard the failure modes flagged in the PR #185 verified review
(H1 — values were never type-checked; H2 — malformed JSON silently fell
back to upstream defaults and polluted A/B iteration logs). The parser
lives in
``forks/whatdreamscost_koolook/versions/v1_3_9/_relay_overrides.py`` —
deliberately stdlib-only so we can import it here without pulling in
ComfyUI's ``comfy_api`` runtime (which the rest of ``ltx_director.py``
needs and which can't run inside pytest).
"""
from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path

import pytest


# Load the parser module by file path. We avoid the normal package import
# (``from forks.whatdreamscost_koolook... import ...``) because walking
# into that package triggers ``forks/whatdreamscost_koolook/versions/
# v1_3_9/__init__.py``, which imports ``ltx_director`` → ``comfy_api`` —
# ComfyUI-only and unavailable inside pytest. The parser module itself
# is pure stdlib + logging, so loading it in isolation is fine.
_RELAY_PATH = (
    Path(__file__).resolve().parents[2]
    / "forks"
    / "whatdreamscost_koolook"
    / "versions"
    / "v1_3_9"
    / "_relay_overrides.py"
)
_spec = importlib.util.spec_from_file_location(
    "whatdreamscost_v1_3_9_relay_overrides", _RELAY_PATH
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

RELAY_OVERRIDE_KEYS = _mod.RELAY_OVERRIDE_KEYS
parse_relay_overrides = _mod.parse_relay_overrides


# ---------------------------------------------------------------------------
# Silent fallback: only the empty/whitespace case should return None.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("blank", ["", "   ", "\n", "\t \n"])
def test_empty_or_whitespace_means_use_upstream_defaults(blank):
    assert parse_relay_overrides(blank) is None


# ---------------------------------------------------------------------------
# Happy path: known knobs with numeric values get coerced to float.
# ---------------------------------------------------------------------------


def test_single_known_knob():
    out = parse_relay_overrides('{"video_strength": 10.0}')
    assert out == {"video_strength": 10.0}
    assert isinstance(out["video_strength"], float)


def test_all_known_knobs():
    payload = {
        "video_strength": 2.5,
        "video_window_scale": 0.7,
        "audio_strength": 0.5,
        "audio_window_scale": 1.2,
        "audio_epsilon": 0.001,
    }
    assert parse_relay_overrides(json.dumps(payload)) == payload


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"video_strength": 10}', 10.0),         # int coerced to float
        ('{"video_strength": "10"}', 10.0),       # numeric string coerced
        ('{"video_strength": "10.5"}', 10.5),     # decimal string coerced
        ('{"video_strength": 1e2}', 100.0),       # scientific notation
        ('{"video_strength": -1.5}', -1.5),       # negatives flow through
    ],
)
def test_numeric_values_are_coerced_to_float(raw, expected):
    out = parse_relay_overrides(raw)
    assert out == {"video_strength": expected}
    assert isinstance(out["video_strength"], float)


@pytest.mark.parametrize(
    "raw",
    [
        "video_strength: 10.0",
        '"video_strength": 10.0',
        "video_strength = 10.0",
        "{\nvideo_strength: 10.0,\nvideo_window_scale: 0.75\n}",
    ],
)
def test_text_multiline_key_value_blocks_are_accepted(raw):
    out = parse_relay_overrides(raw)
    assert out["video_strength"] == 10.0


def test_stacked_json_objects_are_merged():
    raw = '{"video_strength": 10.0}\n{"video_window_scale": 0.75}'
    assert parse_relay_overrides(raw) == {
        "video_strength": 10.0,
        "video_window_scale": 0.75,
    }


# ---------------------------------------------------------------------------
# Comment keys: underscore-prefixed entries are silently dropped.
# ---------------------------------------------------------------------------


def test_underscore_prefixed_keys_are_dropped_silently(caplog):
    raw = '{"_doc": "trial 3", "_owner": "daisy", "video_strength": 5}'
    with caplog.at_level(logging.WARNING):
        out = parse_relay_overrides(raw)
    assert out == {"video_strength": 5.0}
    # No warning was emitted for the comment keys.
    assert "unknown keys" not in caplog.text


def test_only_comment_keys_returns_none():
    assert parse_relay_overrides('{"_doc": "no overrides yet"}') is None


# ---------------------------------------------------------------------------
# Unknown keys: warn-and-drop, don't silently misroute the operator's
# typo into the default-render bucket.
# ---------------------------------------------------------------------------


def test_unknown_key_logs_warning_and_drops_it(caplog):
    raw = '{"viedo_strength": 10, "video_strength": 2.5}'
    with caplog.at_level(logging.WARNING):
        out = parse_relay_overrides(raw)
    assert out == {"video_strength": 2.5}
    assert "unknown keys" in caplog.text.lower()
    assert "viedo_strength" in caplog.text


def test_only_unknown_keys_returns_none(caplog):
    with caplog.at_level(logging.WARNING):
        out = parse_relay_overrides('{"bogus": 1, "also_bogus": 2}')
    assert out is None
    assert "unknown keys" in caplog.text.lower()


# ---------------------------------------------------------------------------
# H2 — malformed JSON / non-object JSON must raise (not silently use
# upstream defaults, which previously polluted A/B comparisons).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        '{"video_strength":10.0',     # missing closing brace
        "not json at all",
    ],
)
def test_malformed_json_raises(raw):
    with pytest.raises(ValueError, match="not valid JSON or a supported"):
        parse_relay_overrides(raw)


@pytest.mark.parametrize(
    "raw",
    [
        '[1, 2, 3]',                  # array
        '"video_strength"',           # string
        "42",                         # number
        "true",                       # bool
        "null",                       # null
    ],
)
def test_non_object_json_raises(raw):
    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_relay_overrides(raw)


# ---------------------------------------------------------------------------
# H1 — a known knob with a non-numeric value must raise at parse time,
# not blow up deep inside the sampler arithmetic.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        '{"video_strength": "ten"}',
        '{"video_strength": true}',
        '{"video_strength": null}',
        '{"video_strength": [1, 2]}',
        '{"video_strength": {"nested": 1}}',
        '{"audio_epsilon": "very small"}',
    ],
)
def test_non_numeric_known_knob_raises(raw):
    with pytest.raises(ValueError, match="must be a number"):
        parse_relay_overrides(raw)


def test_error_message_names_offending_knob_and_shows_example():
    """The error message must let the operator fix the widget without
    consulting upstream Prompt-Relay source."""
    with pytest.raises(ValueError) as exc:
        parse_relay_overrides('{"video_window_scale": "narrow"}')
    msg = str(exc.value)
    assert "video_window_scale" in msg
    assert "Example:" in msg


# ---------------------------------------------------------------------------
# Allowlist sanity: documented keys match implementation.
# ---------------------------------------------------------------------------


def test_documented_knob_set_matches_implementation():
    """If anyone adds a knob in build_segments without updating the
    allowlist, this test fails — preventing silent typos-disguised-as-
    knobs that the H1 fix is meant to catch."""
    assert RELAY_OVERRIDE_KEYS == {
        "video_strength": float,
        "video_window_scale": float,
        "audio_strength": float,
        "audio_window_scale": float,
        "audio_epsilon": float,
    }
