#!/usr/bin/env python3
"""
Validation script to check camera pose conversion accuracy.

This script:
1. Reads the original ASCI file
2. Reads the converted pose file
3. Validates the transformations
4. Reports any issues with coordinate systems, scales, etc.
"""

import argparse
import math
from pathlib import Path
from typing import List, Tuple
import numpy as np


def parse_asci_line(line: str) -> Tuple[float, float, float, float, float, float]:
    """Parse one line of ASCI file: [Tx, Rx, Ty, Ry, Rz, Tz]"""
    values = line.strip().split()
    if len(values) != 6:
        raise ValueError(f"Expected 6 columns, got {len(values)}")
    return tuple(float(v) for v in values)


def parse_pose_line(line: str) -> Tuple[float, List[float], List[float]]:
    """Parse one line of pose file: timestamp, intrinsics, [R|t] matrix"""
    values = line.strip().split()
    if len(values) != 19:
        raise ValueError(f"Expected 19 columns, got {len(values)}")
    
    timestamp = float(values[0])
    intrinsics = [float(v) for v in values[1:7]]  # fx, fy, cx, cy, width, height
    
    # Extract 3x4 [R|t] matrix (row-major)
    matrix_values = [float(v) for v in values[7:19]]
    R = [
        matrix_values[0:4],  # Row 0: r00, r01, r02, t0
        matrix_values[4:8],  # Row 1: r10, r11, r12, t1
        matrix_values[8:12],  # Row 2: r20, r21, r22, t2
    ]
    
    return timestamp, intrinsics, R


def euler_zxy_to_matrix(rx: float, ry: float, rz: float) -> List[List[float]]:
    """Convert ZXY Euler angles to rotation matrix."""
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)

    Rx = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]]
    Ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
    Rz = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]

    # R = Ry * Rx * Rz
    R_temp = np.dot(Ry, Rx)
    R = np.dot(R_temp, Rz)
    return R.tolist()


def validate_rotation_matrix(R: List[List[float]], name: str = "R") -> List[str]:
    """Check if rotation matrix is valid (orthogonal, determinant = 1)."""
    issues = []
    R_np = np.array(R)
    
    # Check determinant (should be ~1.0 for rotation matrix)
    det = np.linalg.det(R_np)
    if abs(det - 1.0) > 0.01:
        issues.append(f"{name}: determinant = {det:.6f} (expected ~1.0)")
    
    # Check orthogonality (R * R^T should be identity)
    should_be_identity = np.dot(R_np, R_np.T)
    identity = np.eye(3)
    max_diff = np.abs(should_be_identity - identity).max()
    if max_diff > 0.01:
        issues.append(f"{name}: not orthogonal (max diff: {max_diff:.6f})")
    
    return issues


def compare_poses(asci_path: Path, pose_path: Path, unit_scale: float = 1.0) -> None:
    """Compare ASCI input with converted pose output."""
    print(f"Reading ASCI: {asci_path}")
    print(f"Reading poses: {pose_path}")
    print(f"Using unit_scale: {unit_scale}")
    print()
    
    # Read ASCI
    asci_lines = [l for l in asci_path.read_text().splitlines() if l.strip()]
    print(f"Found {len(asci_lines)} frames in ASCI")
    
    # Read pose header
    pose_lines = pose_path.read_text().splitlines()
    header = pose_lines[0]
    pose_data = [parse_pose_line(l) for l in pose_lines[1:] if l.strip()]
    print(f"Found {len(pose_data)} frames in pose file")
    print(f"Header: {header}")
    print()
    
    if len(asci_lines) != len(pose_data):
        print(f"WARNING: Frame count mismatch! ASCI: {len(asci_lines)}, Poses: {len(pose_data)}")
        print()
    
    # Compare first few frames in detail
    num_check = min(5, len(asci_lines), len(pose_data))
    print(f"Detailed comparison of first {num_check} frames:")
    print("-" * 80)
    
    all_issues = []
    
    for i in range(num_check):
        print(f"\nFrame {i}:")
        
        # Parse ASCI
        tx, rx_deg, ty, ry_deg, rz_deg, tz = parse_asci_line(asci_lines[i])
        rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
        
        print(f"  ASCI: T=({tx:.3f}, {ty:.3f}, {tz:.3f}), R=({rx_deg:.1f}°, {ry_deg:.1f}°, {rz_deg:.1f}°)")
        
        # Reconstruct rotation matrix from ASCI
        R_asci = euler_zxy_to_matrix(rx, ry, rz)
        R_asci_np = np.array(R_asci)
        
        # Get pose data
        timestamp, intrinsics, R_pose_row = parse_pose_line(pose_data[i])
        R_pose = [row[:3] for row in R_pose_row]  # Extract rotation part
        t_pose = [row[3] for row in R_pose_row]  # Extract translation part
        R_pose_np = np.array(R_pose)
        
        print(f"  Pose: timestamp={timestamp:.0f}µs, fx={intrinsics[0]:.6f}, fy={intrinsics[1]:.6f}")
        print(f"  Pose R[0]: [{R_pose[0][0]:.6f}, {R_pose[0][1]:.6f}, {R_pose[0][2]:.6f}]")
        print(f"  Pose R[1]: [{R_pose[1][0]:.6f}, {R_pose[1][1]:.6f}, {R_pose[1][2]:.6f}]")
        print(f"  Pose t: ({t_pose[0]:.6f}, {t_pose[1]:.6f}, {t_pose[2]:.6f})")
        
        # Validate rotation matrices
        issues_asci = validate_rotation_matrix(R_asci, f"ASCI_R_frame{i}")
        issues_pose = validate_rotation_matrix(R_pose, f"Pose_R_frame{i}")
        
        if issues_asci:
            print(f"  ISSUES with ASCI rotation matrix:")
            for issue in issues_asci:
                print(f"    - {issue}")
            all_issues.extend(issues_asci)
        
        if issues_pose:
            print(f"  ISSUES with Pose rotation matrix:")
            for issue in issues_pose:
                print(f"    - {issue}")
            all_issues.extend(issues_pose)
        
        # Check coordinate system conversion
        # In Y-up to Y-down conversion, Y row and Y column should be negated
        # Check if this pattern exists
        y_row_flipped = (
            abs(R_pose[1][0] + R_asci[1][0]) < 0.01 and
            abs(R_pose[1][1] - R_asci[1][1]) < 0.01 and
            abs(R_pose[1][2] + R_asci[1][2]) < 0.01
        )
        y_col_flipped = (
            abs(R_pose[0][1] + R_asci[0][1]) < 0.01 and
            abs(R_pose[2][1] + R_asci[2][1]) < 0.01
        )
        
        if not y_row_flipped or not y_col_flipped:
            print(f"  WARNING: Coordinate system conversion may be incorrect!")
            print(f"    Y-row flipped: {y_row_flipped}, Y-col flipped: {y_col_flipped}")
            all_issues.append(f"Frame {i}: Coordinate system conversion issue")
        
        # Check translation conversion
        # Expected: t = -R_opencv * C_scaled
        camera_center_scaled = np.array([tx * unit_scale, ty * unit_scale, tz * unit_scale])
        expected_t = -R_pose_np @ camera_center_scaled
        
        t_diff = np.abs(np.array(t_pose) - expected_t)
        max_t_diff = t_diff.max()
        if max_t_diff > 0.01:
            print(f"  WARNING: Translation mismatch!")
            print(f"    Expected t: ({expected_t[0]:.6f}, {expected_t[1]:.6f}, {expected_t[2]:.6f})")
            print(f"    Actual t:    ({t_pose[0]:.6f}, {t_pose[1]:.6f}, {t_pose[2]:.6f})")
            print(f"    Max diff: {max_t_diff:.6f}")
            all_issues.append(f"Frame {i}: Translation conversion issue (max diff: {max_t_diff:.6f})")
        else:
            print(f"  ✓ Translation conversion looks correct (max diff: {max_t_diff:.6f})")
    
    print("\n" + "=" * 80)
    if all_issues:
        print(f"SUMMARY: Found {len(all_issues)} issues:")
        for issue in all_issues:
            print(f"  - {issue}")
    else:
        print("SUMMARY: All checks passed! ✓")


def main():
    parser = argparse.ArgumentParser(description="Validate camera pose conversion")
    parser.add_argument("asci", type=Path, help="Input ASCI file")
    parser.add_argument("pose", type=Path, help="Converted pose file")
    parser.add_argument("--unit-scale", type=float, default=1.0, help="Unit scale used in conversion")
    args = parser.parse_args()
    
    compare_poses(args.asci, args.pose, args.unit_scale)


if __name__ == "__main__":
    main()

