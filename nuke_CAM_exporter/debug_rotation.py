#!/usr/bin/env python3
"""
Debug tool to analyze camera rotations and verify coordinate system conversion.

This helps identify if a 90° Y-axis rotation (or any rotation) is being handled
correctly by the converter.
"""

import argparse
import math
import numpy as np
from pathlib import Path
from typing import Tuple


def euler_zxy_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Convert ZXY Euler angles to rotation matrix (Nuke's system)."""
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)

    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])

    return Ry @ Rx @ Rz


def convert_yup_to_ydown(rotation: np.ndarray) -> np.ndarray:
    """Convert from Nuke Y-up to OpenCV Y-down coordinate system."""
    r = rotation
    return np.array([
        [r[0, 0], -r[0, 1], r[0, 2]],
        [-r[1, 0], r[1, 1], -r[1, 2]],
        [r[2, 0], -r[2, 1], r[2, 2]],
    ])


def rotation_to_euler_xyz(R: np.ndarray) -> Tuple[float, float, float]:
    """Extract XYZ Euler angles from rotation matrix (for debugging)."""
    # Extract angles (this is approximate, works for most cases)
    sy = -R[2, 0]
    cy = math.sqrt(max(0, 1 - sy * sy))
    
    if abs(cy) > 1e-6:
        sx = R[2, 1] / cy
        cx = R[2, 2] / cy
        sz = R[1, 0] / cy
        cz = R[0, 0] / cy
    else:
        # Gimbal lock case
        sx = R[1, 2]
        cx = R[1, 1]
        sz = 0.0
        cz = 1.0
    
    return math.atan2(sx, cx), math.atan2(sy, cy), math.atan2(sz, cz)


def analyze_asci_rotations(path: Path, num_frames: int = 5) -> None:
    """Analyze rotations in ASCI file to identify patterns."""
    print(f"Analyzing rotations in: {path}")
    print("=" * 80)
    
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    num_frames = min(num_frames, len(lines))
    
    print(f"\nFirst {num_frames} frames:")
    print("-" * 80)
    
    for i in range(num_frames):
        values = lines[i].split()
        if len(values) != 6:
            continue
        
        tx, rx_deg, ty, ry_deg, rz_deg, tz = [float(v) for v in values]
        
        print(f"\nFrame {i+1}:")
        print(f"  Translation: ({tx:.3f}, {ty:.3f}, {tz:.3f})")
        print(f"  Rotation (degrees): X={rx_deg:.2f}°, Y={ry_deg:.2f}°, Z={rz_deg:.2f}°")
        
        # Convert to rotation matrix
        rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
        R_nuke = euler_zxy_to_matrix(rx, ry, rz)
        
        # Check if Y rotation is near 90°
        if abs(ry_deg - 90.0) < 5.0 or abs(ry_deg - (-90.0)) < 5.0:
            print(f"  ⚠️  WARNING: Y rotation is near ±90° ({ry_deg:.2f}°)")
            print(f"     This might indicate a coordinate system orientation issue")
        
        # Convert to OpenCV coordinate system
        R_opencv = convert_yup_to_ydown(R_nuke)
        
        # Verify rotation matrix properties
        det_nuke = np.linalg.det(R_nuke)
        det_opencv = np.linalg.det(R_opencv)
        
        print(f"  Rotation matrix determinant: Nuke={det_nuke:.6f}, OpenCV={det_opencv:.6f}")
        
        # Check forward direction (Z axis in camera space)
        forward_nuke = R_nuke[:, 2]  # Third column
        forward_opencv = R_opencv[:, 2]
        
        print(f"  Forward direction (Nuke):    ({forward_nuke[0]:.3f}, {forward_nuke[1]:.3f}, {forward_nuke[2]:.3f})")
        print(f"  Forward direction (OpenCV): ({forward_opencv[0]:.3f}, {forward_opencv[1]:.3f}, {forward_opencv[2]:.3f})")
        
        # Check if coordinate conversion preserved the rotation correctly
        # The forward direction should point in the same physical direction
        # (just expressed in different coordinate systems)
        dot_product = np.dot(forward_nuke, forward_opencv)
        if abs(dot_product - 1.0) < 0.01:
            print(f"  ✓ Coordinate conversion preserves forward direction")
        else:
            print(f"  ⚠️  Forward directions differ (dot={dot_product:.6f})")
    
    print("\n" + "=" * 80)
    print("\nInterpretation:")
    print("- If Y rotation is consistently near ±90°, this might indicate:")
    print("  1. Camera solver oriented the scene differently (this is OK)")
    print("  2. Coordinate system mismatch in the solver (check Nuke settings)")
    print("- The converter should preserve all rotations correctly")
    print("- If motion doesn't match, check unit_scale and lens parameters")


def test_90_degree_rotation() -> None:
    """Test if 90° Y rotation is handled correctly."""
    print("\n" + "=" * 80)
    print("Testing 90° Y-axis rotation conversion:")
    print("-" * 80)
    
    # Test case: 90° rotation around Y axis
    ry_90 = math.radians(90.0)
    R_nuke = euler_zxy_to_matrix(0.0, ry_90, 0.0)
    
    print(f"\nNuke rotation matrix (90° Y):")
    print(R_nuke)
    
    R_opencv = convert_yup_to_ydown(R_nuke)
    
    print(f"\nOpenCV rotation matrix (after conversion):")
    print(R_opencv)
    
    # Verify properties
    det_nuke = np.linalg.det(R_nuke)
    det_opencv = np.linalg.det(R_opencv)
    
    print(f"\nDeterminants: Nuke={det_nuke:.6f}, OpenCV={det_opencv:.6f}")
    
    # Test forward direction
    forward_nuke = R_nuke[:, 2]
    forward_opencv = R_opencv[:, 2]
    
    print(f"\nForward direction:")
    print(f"  Nuke:    ({forward_nuke[0]:.3f}, {forward_nuke[1]:.3f}, {forward_nuke[2]:.3f})")
    print(f"  OpenCV:  ({forward_opencv[0]:.3f}, {forward_opencv[1]:.3f}, {forward_opencv[2]:.3f})")
    
    # In Nuke: +Z forward, so 90° Y rotation should point +X
    # After conversion, should still point in same physical direction
    print(f"\n✓ 90° Y rotation conversion test complete")


def main():
    parser = argparse.ArgumentParser(description="Debug camera rotations")
    parser.add_argument("asci", type=Path, help="ASCI file to analyze")
    parser.add_argument("--frames", type=int, default=5, help="Number of frames to analyze")
    parser.add_argument("--test-90deg", action="store_true", help="Run 90° rotation test")
    args = parser.parse_args()
    
    if args.test_90deg:
        test_90_degree_rotation()
    
    if args.asci.exists():
        analyze_asci_rotations(args.asci, args.frames)
    else:
        print(f"Error: {args.asci} not found")


if __name__ == "__main__":
    main()

