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

import pytest

# Must run before importing the node module (its top-level ``import torch``).
# ``setdefault`` keeps a real torch if one happens to be installed.
sys.modules.setdefault("torch", types.ModuleType("torch"))

# Enrich the stub with the handful of constructors ``create_batch`` uses so the
# *behaviour* (layer compositing, tie-break, passthrough) can be exercised
# torch-free — not just the pure planners. ``_FakeTensor`` wraps a flat list of
# frame-identity tags, so a test can assert *which* frame landed at each output
# index. Skipped automatically if a real torch is installed.
_torch = sys.modules["torch"]
_FAKE_TORCH = not hasattr(_torch, "__version__")
if _FAKE_TORCH:
    class _FakeTensor:
        """Minimal stand-in: a list of opaque per-frame tags + an H/W/C shape."""

        def __init__(self, data, hwc=(2, 2, 3)):
            self._data = list(data)
            self._hwc = tuple(hwc)
            self.device = "cpu"
            self.dtype = "float32"

        @property
        def shape(self):
            return (len(self._data),) + self._hwc

        def __len__(self):
            return len(self._data)

        def __getitem__(self, key):
            if isinstance(key, int):
                return self._data[key]
            if isinstance(key, slice):
                return _FakeTensor(self._data[key], self._hwc)
            if isinstance(key, (list, tuple)):
                return _FakeTensor([self._data[i] for i in key], self._hwc)
            raise TypeError(key)

        def __setitem__(self, key, value):
            if isinstance(key, int):
                self._data[key] = value
            else:
                raise TypeError(key)

        def __rsub__(self, other):  # 1.0 - alpha; identity is fine for these tests
            return self

    def _fake_full(shape, fill_value, device=None, dtype=None):
        return _FakeTensor([("placeholder", fill_value)] * shape[0], shape[1:])

    def _fake_ones(shape, device=None, dtype=None):
        return _FakeTensor([1.0] * shape[0], shape[1:])

    _torch.float32 = "float32"
    _torch.full = _fake_full
    _torch.ones = _fake_ones
else:  # pragma: no cover - real torch present, skip the integration tests
    _FakeTensor = None

_GRAY = ("placeholder", 0.5)  # what _fake_full tags an unfilled frame with


def _source(n):
    """A fake source_batch whose frame k (0-based) is the tag ``f"src{k+1}"``."""
    return _FakeTensor([f"src{k + 1}" for k in range(n)])


requires_fake_torch = pytest.mark.skipif(
    not _FAKE_TORCH, reason="behaviour tests rely on the fake-tensor stub"
)

from k_easy_image_batch import (  # noqa: E402  (import after stub)
    easy_ImageBatch,
    plan_offset_placements,
    plan_slot_overwrites,
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


def test_slot_overwrites_basic_placement():
    # image1 @ VFX frame 4, cut_start 1 -> output index 3.
    placements, outside = plan_slot_overwrites([(0, 4)], total_frames=24, cut_start_frame=1)
    assert placements == [(0, 3, 4)]
    assert outside == []


def test_slot_overwrites_higher_slot_wins_collision():
    # image1 (idx 0) and image3 (idx 2) both target VFX frame 10 -> the
    # higher-numbered slot wins, one winner per output index.
    placements, outside = plan_slot_overwrites(
        [(0, 10), (2, 10)], total_frames=24, cut_start_frame=1
    )
    assert placements == [(2, 9, 10)]
    assert outside == []


def test_slot_overwrites_higher_slot_wins_regardless_of_input_order():
    # Same collision, supplied highest-first: result must still pick image3.
    placements, _ = plan_slot_overwrites(
        [(2, 10), (0, 10)], total_frames=24, cut_start_frame=1
    )
    assert placements == [(2, 9, 10)]


def test_slot_overwrites_outside_cut_reported():
    # cut_start 10: frame 2 -> output -8 (outside); frame 15 -> 5 (in).
    placements, outside = plan_slot_overwrites(
        [(0, 2), (1, 15)], total_frames=20, cut_start_frame=10
    )
    assert placements == [(1, 5, 15)]
    assert outside == [2]


def test_slot_overwrites_cut_start_offsets_output_indices():
    placements, _ = plan_slot_overwrites(
        [(0, 41), (1, 63)], total_frames=81, cut_start_frame=41
    )
    assert placements == [(0, 0, 41), (1, 22, 63)]


def test_slot_overwrites_sorted_by_output_index():
    # Distinct frames, ascending by output index regardless of slot order.
    placements, _ = plan_slot_overwrites(
        [(0, 20), (1, 5)], total_frames=24, cut_start_frame=1
    )
    assert [out for _s, out, _v in placements] == [4, 19]


def test_slot_overwrites_empty_input_places_nothing():
    placements, outside = plan_slot_overwrites([], total_frames=10, cut_start_frame=1)
    assert placements == []
    assert outside == []


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


# --- Behaviour (layer compositing) tests, run against the fake-tensor stub ---


@requires_fake_torch
def test_select_slot_overwrites_list_pick_at_same_frame():
    # source pick at frame 5 AND image1 wired at frame 5 -> the slot wins.
    node = easy_ImageBatch()
    image_batch, _alpha, selected, frames = node.create_batch(
        total_frames=24, cut_start_frame=1, placeholder_color="Gray",
        invert_alpha=False, source_frames="5",
        image1_frame=5, source_batch=_source(24), image1=_FakeTensor(["slotA"]),
    )
    assert image_batch[4] == "slotA"   # output index 4 == VFX frame 5
    assert frames == "5"
    assert len(selected) == 1


@requires_fake_torch
def test_select_higher_slot_wins_at_same_frame():
    # image1 and image2 both target frame 10 -> the higher-numbered slot wins.
    node = easy_ImageBatch()
    image_batch, _alpha, _sel, _frames = node.create_batch(
        total_frames=24, cut_start_frame=1, placeholder_color="Gray",
        invert_alpha=False, source_frames="",
        image1_frame=10, image1=_FakeTensor(["slot1"]),
        image2=_FakeTensor(["slot2"]), image2_frame=10,
    )
    assert image_batch[9] == "slot2"


@requires_fake_torch
def test_insert_slot_composites_on_top_of_inserts():
    # Inserts ins1/ins2/ins3 land at frames 5/7/20; image1 @ frame 5 wins there.
    node = easy_ImageBatch()
    image_batch, _alpha, _sel, _frames = node.create_batch(
        total_frames=24, cut_start_frame=1, placeholder_color="Gray",
        invert_alpha=False, source_frames="5 7 20",
        image1_frame=5, keyframes_insert=_FakeTensor(["ins1", "ins2", "ins3"]),
        image1=_FakeTensor(["slotA"]),
    )
    assert image_batch[4] == "slotA"   # slot beats the insert at frame 5
    assert image_batch[6] == "ins2"    # other inserts untouched
    assert image_batch[19] == "ins3"


@requires_fake_torch
def test_select_empty_list_passes_source_through_with_no_phantom_picks():
    # EXE 2: source connected, empty list, no slots wired (defaults 5/9/13/17).
    # Whole batch is the source passthrough; nothing is "placed".
    node = easy_ImageBatch()
    image_batch, _alpha, selected, frames = node.create_batch(
        total_frames=24, cut_start_frame=1, placeholder_color="Gray",
        invert_alpha=False, source_frames="",
        image1_frame=5, source_batch=_source(24),
        image2_frame=9, image3_frame=13, image4_frame=17,
    )
    assert [image_batch[i] for i in range(24)] == [f"src{k + 1}" for k in range(24)]
    assert frames == ""        # background is never counted as placed
    assert len(selected) == 0


@requires_fake_torch
def test_select_nonempty_list_ignores_unwired_slot_defaults():
    # EXE 3: only the listed picks are placed; the unwired slots' default
    # imageN_frame values (9/13/17) do NOT pull phantom frames from source.
    node = easy_ImageBatch()
    image_batch, _alpha, _sel, frames = node.create_batch(
        total_frames=24, cut_start_frame=1, placeholder_color="Gray",
        invert_alpha=False, source_frames="5 8 20",
        image1_frame=5, source_batch=_source(24),
        image2_frame=9, image3_frame=13, image4_frame=17,
    )
    assert frames == "5, 8, 20"
    assert image_batch[4] == "src5"
    assert image_batch[7] == "src8"
    assert image_batch[19] == "src20"
    assert image_batch[8] == _GRAY    # frame 9 stays placeholder, no phantom pick
