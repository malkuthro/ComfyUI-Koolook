# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for ``easy_ImageBatch`` offset mode.

The node imports ``torch`` at module load, but the offset *planner*
(``plan_offset_placements``) is pure Python and never touches torch — it
just computes which incoming frame lands at which output index. CI installs
only ``pytest`` / ``aiohttp`` / ``Pillow`` (no torch), so we stub the module
to import the node torch-free and test the placement logic directly.
"""
from __future__ import annotations

import sys
import types

# Must run before importing the node module (its top-level ``import torch``).
# ``setdefault`` keeps a real torch if one happens to be installed.
sys.modules.setdefault("torch", types.ModuleType("torch"))

from k_easy_image_batch import (  # noqa: E402  (import after stub)
    easy_ImageBatch,
    plan_offset_placements,
)


def _frames(plan):
    """VFX frame numbers placed, in order."""
    return [vfx for _src, _out, vfx in plan.placements]


def test_basic_offset_maps_packed_frames_to_listed_positions():
    # 5 processed frames, list 1,10,19,30,40, full 41-frame window (cut=1).
    plan = plan_offset_placements([1, 10, 19, 30, 40], source_count=5, total_frames=41, cut_start_frame=1)
    assert plan.placements == [
        (0, 0, 1),
        (1, 9, 10),
        (2, 18, 19),
        (3, 29, 30),
        (4, 39, 40),
    ]
    assert plan.outside_cut == []
    assert plan.unused_source == 0
    assert plan.missing_positions == []


def test_pairing_is_by_ascending_position_not_list_order():
    # Out-of-order list still pairs source[0] -> lowest position, etc., so it
    # round-trips with selected_frames (which is emitted ascending).
    plan = plan_offset_placements([40, 1, 19], source_count=3, total_frames=41, cut_start_frame=1)
    assert _frames(plan) == [1, 19, 40]
    assert [src for src, _out, _vfx in plan.placements] == [0, 1, 2]


def test_more_source_frames_than_positions_reports_unused():
    plan = plan_offset_placements([1, 10, 19], source_count=5, total_frames=41, cut_start_frame=1)
    assert _frames(plan) == [1, 10, 19]
    assert plan.unused_source == 2
    assert plan.missing_positions == []


def test_fewer_source_frames_than_positions_reports_missing():
    plan = plan_offset_placements([1, 10, 19, 30], source_count=2, total_frames=41, cut_start_frame=1)
    assert _frames(plan) == [1, 10]
    assert plan.unused_source == 0
    assert plan.missing_positions == [19, 30]


def test_positions_outside_cut_window_are_dropped():
    # output index = vfx - cut_start_frame; window [0, 20).
    # frame 1 -> -9 (out), 15 -> 5 (in), 40 -> 30 (out).
    plan = plan_offset_placements([1, 15, 40], source_count=3, total_frames=20, cut_start_frame=10)
    assert _frames(plan) == [15]
    assert plan.outside_cut == [1, 40]


def test_duplicate_positions_are_deduped():
    plan = plan_offset_placements([10, 10, 19], source_count=3, total_frames=41, cut_start_frame=1)
    assert _frames(plan) == [10, 19]


def test_empty_position_list_places_nothing():
    plan = plan_offset_placements([], source_count=5, total_frames=41, cut_start_frame=1)
    assert plan.placements == []
    assert plan.unused_source == 5


def test_cut_start_offsets_output_indices():
    # cut_start_frame=41: frame 41 -> idx 0, 63 -> idx 22.
    plan = plan_offset_placements([41, 63], source_count=2, total_frames=81, cut_start_frame=41)
    assert plan.placements == [(0, 0, 41), (1, 22, 63)]


def test_input_types_exposes_keyframe_batch_with_tooltip():
    spec = easy_ImageBatch.INPUT_TYPES()
    assert "keyframe_batch" in spec["optional"], "new offset-mode input must be declared"
    entry = spec["optional"]["keyframe_batch"]
    assert entry[0] == "IMAGE"
    tooltip = entry[1]["tooltip"].lower()
    assert "offset" in tooltip
    assert "frame list" in tooltip


def test_registration_exports_unchanged():
    from k_easy_image_batch import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    assert NODE_CLASS_MAPPINGS["easy_ImageBatch"] is easy_ImageBatch
    assert NODE_DISPLAY_NAME_MAPPINGS["easy_ImageBatch"] == "Easy Image Batch (Koolook)"
