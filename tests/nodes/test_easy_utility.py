# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Easy_Utility — the dispatchable converter node."""
from __future__ import annotations

import pytest

from k_easy_utility import Easy_Utility


@pytest.fixture
def node():
    return Easy_Utility()


@pytest.mark.parametrize(
    "int_value,prefix,pad_width,expected",
    [
        (1, "v", 3, "v001"),
        (0, "v", 3, "v000"),
        (42, "v", 3, "v042"),
        (999, "v", 3, "v999"),
        (1000, "v", 3, "v1000"),  # widens past pad_width when needed
        (5, "", 2, "05"),  # no prefix
        (7, "shot_", 4, "shot_0007"),
        (1, "v", 0, "v1"),  # pad_width=0 → no padding
    ],
)
def test_int_to_padded_string(node, int_value, prefix, pad_width, expected):
    (out,) = node.run("int_to_padded_string", int_value, prefix, pad_width)
    assert out == expected


def test_unknown_mode_returns_stringified_int(node):
    (out,) = node.run("nonexistent_mode", 42, "v", 3)
    assert out == "42"


def test_registration_exports():
    from k_easy_utility import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    assert "Easy_Utility" in NODE_CLASS_MAPPINGS
    assert NODE_CLASS_MAPPINGS["Easy_Utility"] is Easy_Utility
    assert NODE_DISPLAY_NAME_MAPPINGS["Easy_Utility"] == "Easy Utility (Koolook)"


def test_widget_defaults_produce_bare_padded_digits():
    """Default widget values should yield ``001`` (no prefix) — the
    EasyAIPipeline / Easy_VideoCombine consumers add ``v`` themselves
    via ``koolook_versioning.resolve_version_token``."""
    spec = Easy_Utility.INPUT_TYPES()["required"]
    defaults = {name: opts[1]["default"] for name, opts in spec.items()}
    n = Easy_Utility()
    (out,) = n.run(**defaults)
    assert out == "001"
