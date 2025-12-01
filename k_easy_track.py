from __future__ import annotations

from pathlib import Path
from typing import List

COLUMN_COUNT = 19
DEFAULT_POSE_WIDTH = 1280.0
DEFAULT_POSE_HEIGHT = 720.0


def _parse_pose_line(line: str, line_number: int) -> List[float]:
    tokens = line.strip().split()
    if len(tokens) != COLUMN_COUNT:
        raise ValueError(
            f"Line {line_number} expected {COLUMN_COUNT} values but found {len(tokens)}."
        )
    try:
        return [float(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(f"Line {line_number} contains a non-numeric value.") from exc


class KoolookLoadCameraPosesAbsolute:
    """
    Loads RealEstate10k-style camera pose TXT files from any absolute path without ComfyUI's
    built-in strip_path restrictions. The file format matches the AnimateDiff CameraCtrl loader:
    line 1 is metadata, and each following line contains 19 space-separated floats describing
    timestamp, intrinsics, original width/height placeholders, and the flattened 3x4 pose.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": (
                    "STRING",
                    {
                        "default": "C:/projects/shot01/camera/test_camera_track.txt",
                        "multiline": False,
                    },
                ),
            },
            "optional": {
                "pose_width": (
                    "FLOAT",
                    {"default": DEFAULT_POSE_WIDTH, "min": 1.0, "max": 16384.0, "step": 1.0},
                ),
                "pose_height": (
                    "FLOAT",
                    {"default": DEFAULT_POSE_HEIGHT, "min": 1.0, "max": 16384.0, "step": 1.0},
                ),
            },
        }

    RETURN_TYPES = ("CAMERACTRL_POSES",)
    FUNCTION = "load_camera_poses"
    CATEGORY = "Koolook/Camera"
    DESCRIPTION = "Load CameraCtrl poses from a RealEstate10k-style TXT file located anywhere on disk."

    def load_camera_poses(self, file_path: str, pose_width: float = DEFAULT_POSE_WIDTH, pose_height: float = DEFAULT_POSE_HEIGHT):
        normalized_path = Path(file_path).expanduser()
        if not normalized_path.is_file():
            raise FileNotFoundError(f"Pose file not found: {normalized_path}")

        with normalized_path.open("r", encoding="utf-8") as fh:
            lines = fh.read().splitlines()

        if len(lines) < 2:
            raise ValueError("Pose file is missing data rows below the header/metadata line.")

        pose_rows = []
        for idx, raw_line in enumerate(lines[1:], start=2):
            stripped = raw_line.strip()
            if not stripped:
                continue  # allow blank separator lines
            values = _parse_pose_line(stripped, idx)
            values[5] = pose_width
            values[6] = pose_height
            pose_rows.append(values)

        if not pose_rows:
            raise ValueError("Pose file did not contain any valid pose rows.")

        return (pose_rows,)


NODE_CLASS_MAPPINGS = {
    "KoolookLoadCameraPosesAbsolute": KoolookLoadCameraPosesAbsolute,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KoolookLoadCameraPosesAbsolute": "Koolook Load Camera Poses (Absolute Path)",
}
