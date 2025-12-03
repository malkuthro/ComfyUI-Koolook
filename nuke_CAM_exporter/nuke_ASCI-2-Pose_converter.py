#!/usr/bin/env python3
"""Convert Nuke ASCI camera exports into RealEstate10k-style pose TXT for ComfyUI."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

DEFAULT_FPS = 25.0
DEFAULT_FX = 0.5
DEFAULT_FY = 0.5
DEFAULT_CX = 0.5
DEFAULT_CY = 0.5
DEFAULT_WIDTH = 1280.0
DEFAULT_HEIGHT = 720.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("inputs/camTrack_v01.asci"),
        help="Path to Nuke ASCI export",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("outputs/camTrack_v01_converted.txt"),
        help="Destination RealEstate-style TXT file",
    )
    parser.add_argument("--config", type=Path, help="Optional JSON pipeline config")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Frame rate for timestamps")
    parser.add_argument("--fx", type=float, default=DEFAULT_FX, help="Normalized focal length x")
    parser.add_argument("--fy", type=float, default=DEFAULT_FY, help="Normalized focal length y")
    parser.add_argument("--cx", type=float, default=DEFAULT_CX, help="Normalized principal point x")
    parser.add_argument("--cy", type=float, default=DEFAULT_CY, help="Normalized principal point y")
    parser.add_argument("--width", type=float, default=DEFAULT_WIDTH, help="Pose width placeholder")
    parser.add_argument("--height", type=float, default=DEFAULT_HEIGHT, help="Pose height placeholder")
    parser.add_argument("--unit-scale", type=float, default=1.0, help="Scale factor to convert translations into meters")
    parser.add_argument("--start-frame", type=int, default=None, help="First ASCI row (0-based) to include")
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive ASCI row to stop at")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap total frames exported after slicing")
    parser.add_argument("--focal-length-mm", type=float, help="Physical focal length in millimeters")
    parser.add_argument("--sensor-width-mm", type=float, help="Horizontal aperture / filmback in millimeters")
    parser.add_argument("--sensor-height-mm", type=float, help="Vertical aperture / filmback in millimeters")
    parser.add_argument(
        "--principal-offset-x-mm",
        type=float,
        help="Principal point offset from filmback center along X (mm, default 0)",
    )
    parser.add_argument(
        "--principal-offset-y-mm",
        type=float,
        help="Principal point offset from filmback center along Y (mm, default 0)",
    )
    return parser.parse_args()


@dataclass
class ConverterSettings:
    input_path: Path
    output_path: Path
    fps: float
    fx: float
    fy: float
    cx: float
    cy: float
    width: float
    height: float
    unit_scale: float
    metadata: dict
    frame_start: int
    frame_end: int | None
    max_frames: int | None


def load_config(path: Path | None) -> dict:
    if not path:
        return {}
    data = json.loads(path.read_text())
    data["_config_dir"] = path.parent
    return data


def resolve_path(config: dict, key: str, fallback: Path) -> Path:
    raw = config.get(key, fallback)
    if isinstance(raw, str):
        candidate = Path(raw)
    else:
        candidate = Path(raw)
    if not candidate.is_absolute() and "_config_dir" in config:
        candidate = config["_config_dir"] / candidate
    return candidate


def compute_intrinsics(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    args: argparse.Namespace,
    config: dict,
) -> tuple[float, float, float, float]:
    intrinsics_cfg = config.get("intrinsics", {})
    fx = intrinsics_cfg.get("fx", fx)
    fy = intrinsics_cfg.get("fy", fy)
    cx = intrinsics_cfg.get("cx", cx)
    cy = intrinsics_cfg.get("cy", cy)

    lens_cfg = config.get("lens", {})
    focal_mm = args.focal_length_mm or lens_cfg.get("focal_length_mm")
    sensor_w_mm = args.sensor_width_mm or lens_cfg.get("sensor_width_mm")
    sensor_h_mm = args.sensor_height_mm or lens_cfg.get("sensor_height_mm")
    offset_x_mm = args.principal_offset_x_mm
    if offset_x_mm is None:
        offset_x_mm = lens_cfg.get("principal_point_offset_x_mm", 0.0)
    offset_y_mm = args.principal_offset_y_mm
    if offset_y_mm is None:
        offset_y_mm = lens_cfg.get("principal_point_offset_y_mm", 0.0)

    if focal_mm and sensor_w_mm and sensor_h_mm:
        fx = focal_mm / sensor_w_mm
        fy = focal_mm / sensor_h_mm
        cx = 0.5 + (offset_x_mm / sensor_w_mm)
        cy = 0.5 + (offset_y_mm / sensor_h_mm)

    return fx, fy, cx, cy


def resolve_settings(args: argparse.Namespace) -> ConverterSettings:
    config = load_config(args.config)

    input_path = resolve_path(config, "input", args.input)
    output_path = resolve_path(config, "output", args.output)
    fps = config.get("fps", args.fps)

    width = config.get("pose_width", config.get("width", args.width))
    height = config.get("pose_height", config.get("height", args.height))
    resolution_cfg = config.get("resolution", {})
    width = resolution_cfg.get("width_px", width)
    height = resolution_cfg.get("height_px", height)

    unit_scale = config.get("unit_scale", args.unit_scale)
    fx, fy, cx, cy = compute_intrinsics(args.fx, args.fy, args.cx, args.cy, args, config)

    frame_cfg = config.get("frame_range", {})
    frame_start = args.start_frame if args.start_frame is not None else frame_cfg.get("start", 0)
    frame_end = args.end_frame if args.end_frame is not None else frame_cfg.get("end")
    max_frames = args.max_frames if args.max_frames is not None else frame_cfg.get("max_frames")

    metadata = {
        "config": str(args.config) if args.config else None,
        "lens": config.get("lens"),
        "resolution": resolution_cfg or {"width_px": width, "height_px": height},
        "unit_scale": unit_scale,
        "frame_range": {
            "start": frame_start,
            "end": frame_end,
            "max_frames": max_frames,
        },
    }

    return ConverterSettings(
        input_path=input_path,
        output_path=output_path,
        fps=fps,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        unit_scale=unit_scale,
        metadata=metadata,
        frame_start=frame_start,
        frame_end=frame_end,
        max_frames=max_frames,
    )


def euler_zxy_to_matrix(rx: float, ry: float, rz: float) -> List[List[float]]:
    """Convert ZXY Euler angles to rotation matrix in Nuke's coordinate system (Y-up)."""
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)

    Rx = (
        (1.0, 0.0, 0.0),
        (0.0, cx, -sx),
        (0.0, sx, cx),
    )
    Ry = (
        (cy, 0.0, sy),
        (0.0, 1.0, 0.0),
        (-sy, 0.0, cy),
    )
    Rz = (
        (cz, -sz, 0.0),
        (sz, cz, 0.0),
        (0.0, 0.0, 1.0),
    )

    return mat_mul(Ry, mat_mul(Rx, Rz))


def convert_yup_to_ydown(rotation: List[List[float]]) -> List[List[float]]:
    """
    Convert rotation matrix from Nuke's Y-up coordinate system to OpenCV's Y-down.
    
    Nuke: +X right, +Y up, +Z forward
    OpenCV: +X right, +Y down, +Z forward
    
    Transformation: flip Y axis by negating Y row and Y column of rotation matrix.
    This is equivalent to: R_opencv = T * R_nuke * T, where T = diag(1, -1, 1)
    """
    r = rotation
    # Negate Y row and Y column: R'[i,j] = T[i,i] * R[i,j] * T[j,j]
    # where T = diag(1, -1, 1)
    return [
        [r[0][0], -r[0][1], r[0][2]],  # X row: keep X, negate Y, keep Z
        [-r[1][0], r[1][1], -r[1][2]],  # Y row: negate X, keep Y, negate Z
        [r[2][0], -r[2][1], r[2][2]],  # Z row: keep X, negate Y, keep Z
    ]


def mat_mul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    return [
        [a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c] for c in range(3)]
        for r in range(3)
    ]


def iter_asci_poses(path: Path) -> Iterator[Tuple[List[List[float]], Tuple[float, float, float]]]:
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        values = stripped.split()
        if len(values) != 6:
            raise ValueError(f"Expected 6 columns per line, got {len(values)}: {line}")
        # Exporter columns: [translate.x, rotate.x, translate.y, rotate.y, translate.z, rotate.z]
        tx = float(values[0])
        rx = math.radians(float(values[1]))
        ty = float(values[2])
        ry = math.radians(float(values[3]))
        tz = float(values[4])
        rz = math.radians(float(values[5]))
        yield euler_zxy_to_matrix(rx, ry, rz), (tx, ty, tz)


def camera_center_to_translation(
    rotation: Sequence[Sequence[float]], camera_center: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    """
    Convert camera center (world position) to world-to-camera translation.
    
    Formula: t = -R * C, where C is camera center in world space.
    This gives us the translation component of the [R|t] matrix.
    """
    cx, cy, cz = camera_center
    r = rotation
    return (
        -(r[0][0] * cx + r[0][1] * cy + r[0][2] * cz),
        -(r[1][0] * cx + r[1][1] * cy + r[1][2] * cz),
        -(r[2][0] * cx + r[2][1] * cy + r[2][2] * cz),
    )


def format_pose_rows(
    rotation: List[List[float]], translation: Tuple[float, float, float]
) -> List[float]:
    r = rotation
    t = translation
    return [
        r[0][0], r[0][1], r[0][2], t[0],
        r[1][0], r[1][1], r[1][2], t[1],
        r[2][0], r[2][1], r[2][2], t[2],
    ]


def main() -> None:
    args = parse_args()
    settings = resolve_settings(args)
    poses = list(iter_asci_poses(settings.input_path))
    total_available = len(poses)
    start_idx = max(0, settings.frame_start)
    end_idx = settings.frame_end if settings.frame_end is not None else total_available
    if end_idx is not None:
        end_idx = max(start_idx, min(end_idx, total_available))
    poses = poses[start_idx:end_idx]
    if settings.max_frames is not None:
        poses = poses[: settings.max_frames]
    if not poses:
        raise ValueError("No poses left after applying frame_range/max_frames filters.")
    frame_time_us = int(round(1_000_000 / settings.fps))

    header = (
        f"source:{settings.input_path.as_posix()} fps:{settings.fps} fx:{settings.fx} fy:{settings.fy} "
        f"cx:{settings.cx} cy:{settings.cy} width:{settings.width} height:{settings.height} unit_scale:{settings.unit_scale}"
    )

    settings.output_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.output_path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for idx, (rotation, camera_center) in enumerate(poses):
            # Convert from Nuke's Y-up to OpenCV's Y-down coordinate system
            rotation_opencv = convert_yup_to_ydown(rotation)
            
            # Scale camera center to meters and convert to world-to-camera translation
            scaled_center = tuple(coord * settings.unit_scale for coord in camera_center)
            translation = camera_center_to_translation(rotation_opencv, scaled_center)
            
            row = [
                idx * frame_time_us,
                settings.fx,
                settings.fy,
                settings.cx,
                settings.cy,
                settings.width,
                settings.height,
            ]
            row.extend(format_pose_rows(rotation_opencv, translation))
            f.write(" ".join(f"{val:.12f}" for val in row) + "\n")

    print(
        f"Converted {len(poses)} poses "
        f"(frames {start_idx}–{start_idx + len(poses) - 1} of {total_available}) -> {settings.output_path}"
    )


if __name__ == "__main__":
    main()
