# Testing the Camera Conversion Pipeline

This document explains how to test and validate the Nuke → ComfyUI camera conversion pipeline.

## Quick Test Workflow

### 1. Generate Test Scene

```bash
python test_nuke_scene.py --output test_scene.nk --frames 81
```

This creates a Nuke script with:
- A camera that moves forward (dollies in) from Z=-5 to Z=-3
- A slight pan (Y rotation from -10° to -15°)
- Known lens parameters (35mm focal, 36x24mm filmback)
- A reference grid for visualization

### 2. Export Camera from Nuke

1. Open `test_scene.nk` in Nuke
2. Select the `Camera2` node
3. Right-click → **Export Ascii**
4. Configure export:
   - **Columns**: `translate.x`, `rotate.x`, `translate.y`, `rotate.y`, `rotate.z`, `translate.z`
   - **Rotation order**: ZXY (Nuke's default)
   - **Rotation units**: Degrees
5. Save as `inputs/test_camera.asci`

### 3. Convert to Pose Format

```bash
python nuke_ASCI-2-Pose_converter.py \
    inputs/test_camera.asci \
    outputs/test_camera_converted.txt \
    --fps 25.0 \
    --focal-length-mm 35.0 \
    --sensor-width-mm 36.0 \
    --sensor-height-mm 24.0 \
    --width 1920 \
    --height 1080 \
    --unit-scale 1.0
```

Or use a config file:

```bash
python nuke_ASCI-2-Pose_converter.py --config configs/test_camera.json
```

### 4. Validate Conversion

```bash
python validate_conversion.py \
    inputs/test_camera.asci \
    outputs/test_camera_converted.txt \
    --unit-scale 1.0
```

This script checks:
- Rotation matrix validity (orthogonality, determinant = 1)
- Coordinate system conversion (Y-up → Y-down)
- Translation conversion (camera center → world-to-camera translation)
- Frame count consistency

## Common Issues and Fixes

### Issue: Camera motion doesn't match original plate

**Possible causes:**

1. **Missing coordinate system conversion**
   - ✅ Fixed: The converter now includes `convert_yup_to_ydown()` function
   - Check validation output for coordinate system warnings

2. **Camera solver rotated 90° (or any large rotation)**
   - **This is normal!** Nuke's camera solver may orient the camera with large rotations (e.g., 90° Y-axis)
   - The converter preserves all rotations correctly, including 90° rotations
   - Use `python debug_rotation.py inputs/your_camera.asci` to analyze rotations
   - If motion still doesn't match, check `unit_scale` and lens parameters (see below)

3. **Incorrect unit_scale**
   - Measure real-world distance traveled by camera
   - Calculate: `unit_scale = real_distance_meters / asci_delta`
   - Example: Camera moves 1.7m, ASCI Tz delta is 17.0 → `unit_scale = 0.1`

4. **Wrong lens parameters**
   - Verify focal length matches Nuke camera settings
   - Check sensor size (filmback) matches camera node
   - Use `--focal-length-mm` and `--sensor-width-mm` / `--sensor-height-mm` flags

5. **Frame range mismatch**
   - Ensure ASCI export includes all frames you want
   - Use `--start-frame` and `--end-frame` to slice if needed

### Issue: Validation shows coordinate system errors

If `validate_conversion.py` reports coordinate system issues:

1. Check that `convert_yup_to_ydown()` is being called in the converter
2. Verify the Y-row and Y-column negation pattern:
   - `R_opencv[1,0] ≈ -R_nuke[1,0]`
   - `R_opencv[1,2] ≈ -R_nuke[1,2]`
   - `R_opencv[0,1] ≈ -R_nuke[0,1]`
   - `R_opencv[2,1] ≈ -R_nuke[2,1]`

### Issue: Translation values look wrong

The converter uses `t = -R * C` where:
- `R` is the rotation matrix (after Y-up→Y-down conversion)
- `C` is the camera center scaled by `unit_scale`

If translations are off:
1. Verify `unit_scale` is correct
2. Check that camera center values from ASCI are being read correctly
3. Ensure rotation matrix is valid (orthogonal, det=1)

## Testing with Real Production Data

For production shots:

1. **Measure reference distance**
   - Use CameraTracker's "Set Reference Distance" feature
   - Or measure a known object size in the scene
   - Calculate `unit_scale` from this measurement

2. **Verify lens metadata**
   - Get focal length from camera report or EXIF data
   - Get sensor size from camera specs (e.g., Full Frame = 36x24mm)

3. **Export with correct frame range**
   - Match the frame range you'll use in ComfyUI
   - Use `--start-frame` and `--end-frame` to match exactly

4. **Validate before using in ComfyUI**
   - Run `validate_conversion.py` to catch issues early
   - Check that rotation matrices are valid
   - Verify coordinate system conversion

## Integration with ComfyUI

After conversion:

1. Load the pose file in ComfyUI using `Koolook Load Camera Poses`
2. Connect to `WanVideo FunCamera Embeds`
3. Set render resolution to match your pose file's width/height
4. Test with a simple prompt first to verify motion

If motion still doesn't match:
- Double-check `unit_scale` calculation
- Verify lens parameters match production data
- Check that ComfyUI render resolution matches pose file resolution
- Consider using camera projection in Nuke to validate the converted poses


