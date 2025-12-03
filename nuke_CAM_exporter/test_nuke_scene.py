#!/usr/bin/env python3
"""
Nuke script generator for testing camera export pipeline.

This script generates a Nuke script (.nk) that creates a simple test scene with:
- A camera with known, predictable motion
- Simple geometry (a grid/card) for reference
- Camera animation that can be easily validated

Usage:
    python test_nuke_scene.py --output test_scene.nk
"""

import argparse
from pathlib import Path


def generate_nuke_script(output_path: Path, num_frames: int = 50) -> None:
    """Generate a Nuke script with a test camera animation."""
    
    # Create a simple camera animation:
    # - Start at (0, 0, -5) looking at origin
    # - Move forward along Z axis (dolly in)
    # - Slight pan left
    # - Known focal length and filmback for easy validation
    
    script_lines = [
        "set cut_paste_input [stack 0]",
        "version 15.0 v1",
        "push $cut_paste_input",
        "",
        "# Test Camera - Simple forward dolly with pan",
        "Camera2 {",
        "    inputs 0",
        f"    translate {{curve x {0.0} {num_frames-1} {0.0}; curve y {0.0} {num_frames-1} {0.0}; curve z {-5.0} {num_frames-1} {-3.0}}}",
        f"    rotate {{curve x {0.0} {num_frames-1} {0.0}; curve y {-10.0} {num_frames-1} {-15.0}; curve z {0.0} {num_frames-1} {0.0}}}",
        "    focal {35.0}",
        "    haperture {36.0}",
        "    vaperture {24.0}",
        "    near {0.1}",
        "    far {1000.0}",
        "    win_translate {{0.5 0.5}}",
        "    win_scale {{1 1}}",
        "    winroll {0}",
        "    fov {54.432}",
        "    f_stop {2.8}",
        "    focus_distance {5.0}",
        "    lens {35.0}",
        "    name Camera2",
        "    selected true",
        "    xpos 0",
        "    ypos 0",
        "}",
        "",
        "# Reference Grid - helps visualize camera motion",
        "Grid2 {",
        "    inputs 0",
        "    display {luminance}",
        "    name Grid2",
        "    selected false",
        "    xpos 0",
        "    ypos -100",
        "}",
        "",
        "# Render node for validation",
        "Write1 {",
        "    file {test_render_%04d.exr}",
        "    file_type exr",
        "    datatype {32 bit float}",
        "    compression {Zip (1 scanline)}",
        "    name Write1",
        "    selected false",
        "    xpos 0",
        "    ypos -200",
        "}",
        "",
        "end_group",
    ]
    
    output_path.write_text("\n".join(script_lines))
    print(f"Generated Nuke script: {output_path}")
    print(f"  - {num_frames} frames")
    print(f"  - Camera starts at (0, 0, -5), ends at (0, 0, -3)")
    print(f"  - Camera rotates Y from -10° to -15°")
    print(f"  - Focal: 35mm, Filmback: 36x24mm")


def main():
    parser = argparse.ArgumentParser(description="Generate Nuke test scene")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_scene.nk"),
        help="Output Nuke script path",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=50,
        help="Number of frames in animation",
    )
    args = parser.parse_args()
    
    generate_nuke_script(args.output, args.frames)


if __name__ == "__main__":
    main()

