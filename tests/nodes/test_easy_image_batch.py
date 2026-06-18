# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for ``easy_ImageBatch`` insert modes and schema.

The node imports ``torch`` at module load, but its *planners*
(``plan_offset_placements`` for insert placement, ``plan_source_base_fill``
for the insert-over-source background) are pure Python and never touch torch
— they just compute which frame lands at which output index. CI installs
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
    plan_source_base_fill,
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


def test_source_base_fill_full_window_maps_one_to_one():
    # cut_start=1: output i -> source i (source_batch[0] is VFX frame 1).
    placements, fallback = plan_source_base_fill(total_frames=5, cut_start_frame=1, base_len=10)
    assert placements == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    assert fallback == 0


def test_source_base_fill_offsets_with_cut_start():
    # cut_start=41: output 0 -> source 40 (VFX frame 41), etc.
    placements, fallback = plan_source_base_fill(total_frames=3, cut_start_frame=41, base_len=121)
    assert placements == [(0, 40), (1, 41), (2, 42)]
    assert fallback == 0


def test_source_base_fill_beyond_source_falls_back_to_placeholder():
    # base only covers 2 frames; the remaining 3 cut frames stay placeholder.
    placements, fallback = plan_source_base_fill(total_frames=5, cut_start_frame=1, base_len=2)
    assert placements == [(0, 0), (1, 1)]
    assert fallback == 3


def test_source_base_fill_cut_starts_past_source_end():
    # Cut window starts beyond the source entirely → all placeholder.
    placements, fallback = plan_source_base_fill(total_frames=4, cut_start_frame=50, base_len=10)
    assert placements == []
    assert fallback == 4


def test_input_types_exposes_keyframes_insert_with_tooltip():
    spec = easy_ImageBatch.INPUT_TYPES()
    assert "keyframes_insert" in spec["optional"], "insert-mode input must be declared"
    # The pre-rename name must be gone so the rename is unambiguous.
    assert "keyframe_batch" not in spec["optional"]
    entry = spec["optional"]["keyframes_insert"]
    assert entry[0] == "IMAGE"
    tooltip = entry[1]["tooltip"].lower()
    assert "insert" in tooltip
    assert "source_frames" in tooltip


def test_input_types_exposes_width_height_fallback_at_end():
    spec = easy_ImageBatch.INPUT_TYPES()
    # Fallback-only and declared OPTIONAL so they append to the END of the
    # widget list — keeping positional widgets_values of older saved workflows
    # aligned (inserting them earlier shifts placeholder_color/source_frames).
    for key in ("width", "height"):
        assert key in spec["optional"], f"{key} fallback widget must be optional"
        assert key not in spec["required"]
        assert spec["optional"][key][0] == "INT"
        assert spec["optional"][key][1]["default"] == 512
    # They must be the last two optional keys (= last two widgets).
    assert list(spec["optional"])[-2:] == ["width", "height"]


def test_registration_exports_unchanged():
    from k_easy_image_batch import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    assert NODE_CLASS_MAPPINGS["easy_ImageBatch"] is easy_ImageBatch
    assert NODE_DISPLAY_NAME_MAPPINGS["easy_ImageBatch"] == "Easy Image Batch (Koolook)"
