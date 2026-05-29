# SPDX-License-Identifier: GPL-3.0-or-later
#
# ComfyUI-Koolook - standalone timeline editor
"""Lightweight timeline-data node that reuses the Koolook Director UI."""
from __future__ import annotations


class KoolookTimelineEditor:
    """Expose the Director timeline editor without applying LTX conditioning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration_frames": ("INT", {"default": 120, "min": 1, "max": 10000, "step": 1}),
                "duration_seconds": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 1000.0, "step": 0.01}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 1.0}),
                "display_mode": (["frames", "seconds"], {"default": "seconds"}),
                "timeline_data": ("STRING", {"default": "{}", "multiline": True}),
                "local_prompts": ("STRING", {"default": "", "multiline": True}),
                "segment_lengths": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "FLOAT", "INT")
    RETURN_NAMES = (
        "timeline_data",
        "local_prompts",
        "segment_lengths",
        "frame_rate",
        "duration_frames",
    )
    CATEGORY = "Koolook/Timeline"
    FUNCTION = "run"

    def run(
        self,
        duration_frames: int,
        duration_seconds: float,
        frame_rate: float,
        display_mode: str,
        timeline_data: str,
        local_prompts: str,
        segment_lengths: str,
    ):
        _ = duration_seconds, display_mode
        return (
            timeline_data or "{}",
            local_prompts or "",
            segment_lengths or "",
            float(frame_rate),
            int(duration_frames),
        )


NODE_CLASS_MAPPINGS = {"KoolookTimelineEditor": KoolookTimelineEditor}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KoolookTimelineEditor": "Koolook Timeline Editor"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
