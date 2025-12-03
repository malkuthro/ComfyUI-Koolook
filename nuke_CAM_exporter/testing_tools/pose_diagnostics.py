#!/usr/bin/env python3
"""
Pose diagnostics CLI.

Replaces the previous individual helpers (check_mapping, validate_conversion,
debug_rotation) with a single entry point that exposes three subcommands:

    - validate: compare ASCI vs. converted TXT in detail.
    - check:    probe specific frames for translation/rotation parity.
    - rotations:inspect raw ASCI rotations or run a 90° sanity test.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Shared math helpers
# ---------------------------------------------------------------------------

def euler_zxy_to_matrix(rx: float, ry: float, rz: float) -> List[List[float]]:
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)

    Rx = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]]
    Ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
    Rz = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]

    return (np.dot(np.dot(Ry, Rx), Rz)).tolist()


def convert_yup_to_ydown(rotation: Sequence[Sequence[float]]) -> List[List[float]]:
    r = rotation
    return [
        [r[0][0], -r[0][1], r[0][2]],
        [-r[1][0], r[1][1], -r[1][2]],
        [r[2][0], -r[2][1], r[2][2]],
    ]


def mat_vec_mul(R: Sequence[Sequence[float]], v: Sequence[float]) -> List[float]:
    return [
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
    ]


def read_numeric_rows(path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[0] not in "0123456789-.":
            continue
        rows.append(list(map(float, stripped.split())))
    return rows


# ---------------------------------------------------------------------------
# validate command
# ---------------------------------------------------------------------------

def parse_pose_line(line: str) -> Tuple[float, List[float], List[List[float]]]:
    values = line.strip().split()
    if len(values) != 19:
        raise ValueError(f"Expected 19 columns, got {len(values)}")
    timestamp = float(values[0])
    intrinsics = [float(v) for v in values[1:7]]
    matrix_values = [float(v) for v in values[7:19]]
    R = [
        matrix_values[0:4],
        matrix_values[4:8],
        matrix_values[8:12],
    ]
    return timestamp, intrinsics, R


def validate_rotation_matrix(R: Sequence[Sequence[float]], name: str) -> List[str]:
    issues: List[str] = []
    R_np = np.array(R)
    det = np.linalg.det(R_np)
    if abs(det - 1.0) > 0.01:
        issues.append(f"{name}: determinant {det:.6f} (expected 1.0)")
    should_be_identity = np.dot(R_np, R_np.T)
    identity = np.eye(3)
    max_diff = np.abs(should_be_identity - identity).max()
    if max_diff > 0.01:
        issues.append(f"{name}: not orthogonal (max diff {max_diff:.6f})")
    return issues


def cmd_validate(args: argparse.Namespace) -> None:
    asci_path = args.asci
    pose_path = args.pose
    unit_scale = args.unit_scale

    print(f"Reading ASCI: {asci_path}")
    print(f"Reading poses: {pose_path}")
    print(f"Using unit_scale: {unit_scale}")
    print()

    asci_lines = [l for l in asci_path.read_text().splitlines() if l.strip()]
    pose_lines = pose_path.read_text().splitlines()
    header = pose_lines[0] if pose_lines else ""
    pose_data = [parse_pose_line(l) for l in pose_lines[1:] if l.strip()]

    print(f"Found {len(asci_lines)} frames in ASCI")
    print(f"Found {len(pose_data)} frames in pose file")
    print(f"Header: {header}")
    if len(asci_lines) != len(pose_data):
        print(
            f"WARNING: Frame count mismatch! ASCI: {len(asci_lines)}, Poses: {len(pose_data)}"
        )
    print()

    num_check = min(args.frames, len(asci_lines), len(pose_data))
    print(f"Detailed comparison of first {num_check} frames:")
    print("-" * 80)

    all_issues: List[str] = []
    for i in range(num_check):
        tx, rx_deg, ty, ry_deg, tz, rz_deg = map(float, asci_lines[i].split())
        rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)

        print(f"\nFrame {i}:")
        print(f"  ASCI: T=({tx:.3f}, {ty:.3f}, {tz:.3f}), R=({rx_deg:.2f}°, {ry_deg:.2f}°, {rz_deg:.2f}°)")

        R_asci = euler_zxy_to_matrix(rx, ry, rz)
        R_pose = [row[:3] for row in pose_data[i][2]]
        t_pose = [row[3] for row in pose_data[i][2]]
        print(f"  Pose t: ({t_pose[0]:.6f}, {t_pose[1]:.6f}, {t_pose[2]:.6f})")

        all_issues.extend(validate_rotation_matrix(R_asci, f"ASCI_R_{i}"))
        all_issues.extend(validate_rotation_matrix(R_pose, f"Pose_R_{i}"))

        # Translation check
        camera_center_scaled = np.array([tx * unit_scale, ty * unit_scale, tz * unit_scale])
        expected_t = -np.array(R_pose) @ camera_center_scaled
        t_diff = np.abs(np.array(t_pose) - expected_t)
        max_t_diff = t_diff.max()
        if max_t_diff > 0.01:
            print(f"  WARNING: translation mismatch (max diff {max_t_diff:.6f})")
            all_issues.append(f"Frame {i}: translation mismatch ({max_t_diff:.6f})")
        else:
            print(f"  ✓ translation matches (max diff {max_t_diff:.6f})")

    print("\n" + "=" * 80)
    if all_issues:
        print("SUMMARY: issues detected:")
        for issue in all_issues:
            print(" -", issue)
    else:
        print("SUMMARY: All checks passed ✓")


# ---------------------------------------------------------------------------
# check command (specific frame parity)
# ---------------------------------------------------------------------------

def cmd_check(args: argparse.Namespace) -> None:
    asci_rows = read_numeric_rows(args.asci)
    pose_rows = read_numeric_rows(args.pose)[1:]  # skip header already filtered

    print(f"Frames in ASCI: {len(asci_rows)}")
    print(f"Frames in TXT : {len(pose_rows)}")

    indices = args.frames or [0, len(pose_rows) - 1]

    for idx in indices:
        actual_idx = idx if idx >= 0 else len(pose_rows) + idx
        if actual_idx < 0 or actual_idx >= len(pose_rows):
            continue
        tx, rx, ty, ry, tz, rz = asci_rows[actual_idx]
        R_nuke = euler_zxy_to_matrix(math.radians(rx), math.radians(ry), math.radians(rz))
        R_cv = convert_yup_to_ydown(R_nuke)
        t_expected = [-val for val in mat_vec_mul(R_cv, (tx, ty, tz))]

        pose_vals = pose_rows[actual_idx]
        R_export = [
            pose_vals[7:11],
            pose_vals[11:15],
            pose_vals[15:19],
        ]

        rot_diff = max(abs(R_export[r][c] - R_cv[r][c]) for r in range(3) for c in range(3))

        print(f"\nFrame {actual_idx}")
        print("Nuke  Tx Ty Tz :", (tx, ty, tz))
        print("Export t       :", tuple(round(pose_vals[10 + i * 4], 6) for i in range(3)))
        print("Expected t     :", tuple(round(val, 6) for val in t_expected))
        print("Rotation diff  :", rot_diff)
        print("R_nuke→OpenCV  :")
        for row in R_cv:
            print("   ", " ".join(f"{val: .6f}" for val in row))
        print("R_export       :")
        for row in R_export:
            print("   ", " ".join(f"{val: .6f}" for val in row))


# ---------------------------------------------------------------------------
# rotations command (ASCI analysis)
# ---------------------------------------------------------------------------

def analyze_asci_rotations(path: Path, num_frames: int = 5) -> None:
    print(f"Analyzing rotations in: {path}")
    print("=" * 80)

    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    num_frames = min(num_frames, len(lines))

    for i in range(num_frames):
        values = list(map(float, lines[i].split()))
        tx, rx_deg, ty, ry_deg, tz, rz_deg = values

        print(f"\nFrame {i}: T=({tx:.3f}, {ty:.3f}, {tz:.3f})  R=({rx_deg:.2f}°, {ry_deg:.2f}°, {rz_deg:.2f}°)")
        R_nuke = np.array(euler_zxy_to_matrix(math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)))
        R_cv = np.array(convert_yup_to_ydown(R_nuke))
        print(f"  det(Nuke)={np.linalg.det(R_nuke):.6f} det(OpenCV)={np.linalg.det(R_cv):.6f}")


def test_90deg_rotation() -> None:
    print("=" * 80)
    print("Testing 90° Y-axis rotation conversion")
    ry = math.radians(90.0)
    R_nuke = np.array(euler_zxy_to_matrix(0.0, ry, 0.0))
    R_cv = np.array(convert_yup_to_ydown(R_nuke))
    print("Nuke rotation:\n", R_nuke)
    print("OpenCV rotation:\n", R_cv)
    print(f"det Nuke={np.linalg.det(R_nuke):.6f}, det OpenCV={np.linalg.det(R_cv):.6f}")


def cmd_rotations(args: argparse.Namespace) -> None:
    if args.test_90deg:
        test_90deg_rotation()
    analyze_asci_rotations(args.asci, args.frames)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pose diagnostics helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="Cross-check ASCI vs converted TXT (detailed)"
    )
    validate_parser.add_argument("asci", type=Path)
    validate_parser.add_argument("pose", type=Path)
    validate_parser.add_argument("--unit-scale", type=float, default=1.0)
    validate_parser.add_argument(
        "--frames", type=int, default=5, help="Number of initial frames to inspect"
    )
    validate_parser.set_defaults(func=cmd_validate)

    check_parser = subparsers.add_parser(
        "check", help="Inspect specific frame indices for rotation/translation parity"
    )
    check_parser.add_argument("asci", type=Path)
    check_parser.add_argument("pose", type=Path)
    check_parser.add_argument(
        "--frames",
        type=int,
        nargs="*",
        default=None,
        help="Frame indices to inspect (use negative indices for reverse order)",
    )
    check_parser.set_defaults(func=cmd_check)

    rot_parser = subparsers.add_parser(
        "rotations", help="Inspect rotations inside an ASCI export"
    )
    rot_parser.add_argument("asci", type=Path)
    rot_parser.add_argument("--frames", type=int, default=5)
    rot_parser.add_argument("--test-90deg", action="store_true")
    rot_parser.set_defaults(func=cmd_rotations)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


