#!/usr/bin/env python3
"""Convert Nuke ASCI camera exports into RealEstate10k-style pose TXT for ComfyUI."""

from __future__ import annotations

import argparse
import math
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
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Frame rate for timestamps")
    parser.add_argument("--fx", type=float, default=DEFAULT_FX, help="Normalized focal length x")
    parser.add_argument("--fy", type=float, default=DEFAULT_FY, help="Normalized focal length y")
    parser.add_argument("--cx", type=float, default=DEFAULT_CX, help="Normalized principal point x")
    parser.add_argument("--cy", type=float, default=DEFAULT_CY, help="Normalized principal point y")
    parser.add_argument("--width", type=float, default=DEFAULT_WIDTH, help="Pose width placeholder")
    parser.add_argument("--height", type=float, default=DEFAULT_HEIGHT, help="Pose height placeholder")
    return parser.parse_args()


def euler_zxy_to_matrix(rx: float, ry: float, rz: float) -> List[List[float]]:
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
        # Exporter columns: [translate.x, rotate.x, translate.y, rotate.y, rotate.z, translate.z]
        tx = float(values[0])
        rx = math.radians(float(values[1]))
        ty = float(values[2])
        ry = math.radians(float(values[3]))
        rz = math.radians(float(values[4]))
        tz = float(values[5])
        yield euler_zxy_to_matrix(rx, ry, rz), (tx, ty, tz)


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
    poses = list(iter_asci_poses(args.input))
    frame_time_us = int(round(1_000_000 / args.fps))

    header = (
        f"source:{args.input.as_posix()} fps:{args.fps} fx:{args.fx} fy:{args.fy} "
        f"cx:{args.cx} cy:{args.cy} width:{args.width} height:{args.height}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for idx, (rotation, translation) in enumerate(poses):
            row = [
                idx * frame_time_us,
                args.fx,
                args.fy,
                args.cx,
                args.cy,
                args.width,
                args.height,
            ]
            row.extend(format_pose_rows(rotation, translation))
            f.write(" ".join(f"{val:.12f}" for val in row) + "\n")

    print(f"Converted {len(poses)} poses -> {args.output}")


if __name__ == "__main__":
    main()
