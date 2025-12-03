#!/bin/bash
# Test pipeline script for validating the camera conversion workflow
# 
# This script:
# 1. Generates a test Nuke scene
# 2. (Manual step) Export camera from Nuke as ASCI
# 3. Converts ASCI to pose format
# 4. Validates the conversion
#
# Usage: ./test_pipeline.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Step 1: Generate Nuke test scene ==="
python3 test_nuke_scene.py --output test_scene.nk --frames 50
echo "✓ Generated test_scene.nk"
echo ""
echo "=== Step 2: Manual step - Export camera from Nuke ==="
echo "1. Open test_scene.nk in Nuke"
echo "2. Select Camera2 node"
echo "3. Export Ascii with columns: [translate.x, rotate.x, translate.y, rotate.y, rotate.z, translate.z]"
echo "4. Save as: inputs/test_camera.asci"
echo ""
read -p "Press Enter when you've exported the ASCI file..."

if [ ! -f "inputs/test_camera.asci" ]; then
    echo "ERROR: inputs/test_camera.asci not found!"
    exit 1
fi

echo ""
echo "=== Step 3: Convert ASCI to pose format ==="
python3 nuke_ASCI-2-Pose_converter.py \
    inputs/test_camera.asci \
    outputs/test_camera_converted.txt \
    --fps 25.0 \
    --focal-length-mm 35.0 \
    --sensor-width-mm 36.0 \
    --sensor-height-mm 24.0 \
    --width 1920 \
    --height 1080 \
    --unit-scale 1.0

echo ""
echo "=== Step 4: Validate conversion ==="
python3 validate_conversion.py \
    inputs/test_camera.asci \
    outputs/test_camera_converted.txt \
    --unit-scale 1.0

echo ""
echo "=== Pipeline test complete! ==="


