#!/usr/bin/env python3
import math
from pathlib import Path
from typing import Sequence

ASCI_PATH = Path(__file__).resolve().parents[1] / "inputs" / "nukeCam_unit-1_v2.asci"
POSE_PATH = Path(__file__).resolve().parents[1] / "outputs" / "nukeCam_unit-1_v4_converted.txt"


def mat_mul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    return [
        [a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c] for c in range(3)]
        for r in range(3)
    ]


def euler_zxy(rx: float, ry: float, rz: float) -> list[list[float]]:
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


def convert_yup_to_ydown(R: list[list[float]]) -> list[list[float]]:
    # T = diag(1, -1, 1)
    return [
        [R[0][0], -R[0][1], R[0][2]],
        [-R[1][0], R[1][1], -R[1][2]],
        [R[2][0], -R[2][1], R[2][2]],
    ]


def mat_vec_mul(R: Sequence[Sequence[float]], v: Sequence[float]) -> list[float]:
    return [
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
    ]


def read_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Skip header/metadata lines (e.g. "source:…") that are not numeric.
        first = stripped[0]
        if first not in "0123456789-.":
            continue
        rows.append(list(map(float, stripped.split())))
    return rows


def compare_rows(indices: list[int]) -> None:
    asci_rows = read_rows(ASCI_PATH)
    pose_rows = read_rows(POSE_PATH)

    print(f"Frames in ASCI: {len(asci_rows)}")
    print(f"Frames in TXT:  {len(pose_rows)}")

    for idx in indices:
        if idx >= len(pose_rows):
            continue
        tx, rx, ty, ry, tz, rz = asci_rows[idx]
        R_nuke = euler_zxy(math.radians(rx), math.radians(ry), math.radians(rz))
        R_cv = convert_yup_to_ydown(R_nuke)
        t = [-val for val in mat_vec_mul(R_cv, (tx, ty, tz))]

        pose_vals = pose_rows[idx]
        R_export = [
            pose_vals[7:11],
            pose_vals[11:15],
            pose_vals[15:19],
        ]

        rot_diff = max(abs(R_export[r][c] - R_cv[r][c]) for r in range(3) for c in range(3))

        print(f"\nFrame {idx}")
        print("Nuke  Tx Ty Tz :", (tx, ty, tz))
        print("Export t       :", tuple(round(pose_vals[10 + i * 4], 6) for i in range(3)))
        print("Expected t     :", tuple(round(val, 6) for val in t))
        print("Rotation diff  :", rot_diff)
        print("R_nuke→OpenCV  :")
        for row in R_cv:
            print("   ", " ".join(f"{val: .6f}" for val in row))
        print("R_export       :")
        for row in R_export:
            print("   ", " ".join(f"{val: .6f}" for val in row))


if __name__ == "__main__":
    compare_rows([0, 79])

