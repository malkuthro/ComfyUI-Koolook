# Camera Conversion Pipeline - Fixes and Improvements

## Summary

This document outlines the issues found and fixes applied to the Nuke → ComfyUI camera conversion pipeline.

## Critical Issue Found: Missing Coordinate System Conversion

### Problem
The converter was **not converting from Nuke's Y-up coordinate system to OpenCV's Y-down coordinate system**. This causes camera motion to appear incorrect in ComfyUI, as the coordinate axes don't match what the Wan model expects.

### Root Cause
- **Nuke uses**: +X right, +Y up, +Z forward
- **OpenCV/RealEstate10k uses**: +X right, +Y down, +Z forward
- The converter was building rotation matrices in Nuke's coordinate system but not transforming them to OpenCV's system

### Fix Applied
Added `convert_yup_to_ydown()` function that properly transforms rotation matrices:

```python
def convert_yup_to_ydown(rotation: List[List[float]]) -> List[List[float]]:
    """
    Convert rotation matrix from Nuke's Y-up coordinate system to OpenCV's Y-down.
    
    Transformation: R_opencv = T * R_nuke * T, where T = diag(1, -1, 1)
    This negates the Y row and Y column of the rotation matrix.
    """
    r = rotation
    return [
        [r[0][0], -r[0][1], r[0][2]],  # X row: keep X, negate Y, keep Z
        [-r[1][0], r[1][1], -r[1][2]],  # Y row: negate X, keep Y, negate Z
        [r[2][0], -r[2][1], r[2][2]],  # Z row: keep X, negate Y, keep Z
    ]
```

This function is now called in `main()` before computing translations and writing pose data.

### Impact
- **Before**: Camera motion would be flipped/mirrored on the Y axis
- **After**: Camera motion matches the original Nuke camera path

## Code Improvements

### 1. Added Documentation
- Added docstrings to key functions explaining coordinate system transformations
- Clarified the purpose of `camera_center_to_translation()` function

### 2. Improved Code Structure
- Separated coordinate system conversion into its own function
- Made the transformation explicit and testable

## Testing Tools Created

### 1. `test_nuke_scene.py`
Generates a Nuke script (.nk) with:
- A simple camera animation (forward dolly + pan)
- Known lens parameters (35mm focal, 36x24mm filmback)
- Reference geometry (grid) for visualization
- Easy to validate motion

**Usage:**
```bash
python test_nuke_scene.py --output test_scene.nk --frames 50
```

### 2. `validate_conversion.py`
Validates the conversion by:
- Reading both ASCI input and pose output
- Checking rotation matrix validity (orthogonality, determinant = 1)
- Verifying coordinate system conversion (Y-up → Y-down)
- Validating translation conversion (camera center → world-to-camera)
- Reporting any issues found

**Usage:**
```bash
python validate_conversion.py inputs/camera.asci outputs/camera_converted.txt --unit-scale 0.1
```

### 3. `test_pipeline.sh`
Automated test script that runs the full pipeline:
1. Generates test scene
2. (Manual) Export from Nuke
3. Converts ASCI to pose
4. Validates conversion

### 4. `README_TESTING.md`
Comprehensive testing guide with:
- Step-by-step workflow
- Common issues and fixes
- Integration instructions for ComfyUI

## How to Test the Fix

### Quick Test
1. **Generate test scene:**
   ```bash
   python test_nuke_scene.py --output test_scene.nk
   ```

2. **Export camera from Nuke:**
   - Open `test_scene.nk` in Nuke
   - Select Camera2 node
   - Export Ascii with columns: `[translate.x, rotate.x, translate.y, rotate.y, translate.z, rotate.z]`
   - Save as `inputs/test_camera.asci`

3. **Convert:**
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

4. **Validate:**
   ```bash
   python validate_conversion.py \
       inputs/test_camera.asci \
       outputs/test_camera_converted.txt \
       --unit-scale 1.0
   ```

### Test with Existing Data
To test with your existing `camTrack_1.7m_v02.asci`:

```bash
# Convert with fix
python nuke_ASCI-2-Pose_converter.py \
    inputs/camTrack_1.7m_v02.asci \
    outputs/camTrack_1.7m_v02_converted_FIXED.txt \
    --config configs/camera_config_v01.json

# Validate
python validate_conversion.py \
    inputs/camTrack_1.7m_v02.asci \
    outputs/camTrack_1.7m_v02_converted_FIXED.txt \
    --unit-scale 0.1

# Compare with old output
python validate_conversion.py \
    inputs/camTrack_1.7m_v02.asci \
    outputs/camTrack_1.7m_v02_converted.txt \
    --unit-scale 0.1
```

## Expected Results

### Before Fix
- Validation would show coordinate system conversion warnings
- Camera motion in ComfyUI would be flipped/mirrored on Y axis
- Rotation matrices would be in Nuke's coordinate system

### After Fix
- Validation should pass coordinate system checks
- Camera motion should match original Nuke camera path
- Rotation matrices properly converted to OpenCV coordinate system
- Translation values correctly computed using converted rotation matrix

## Additional Considerations

### Unit Scale
The `unit_scale` parameter is critical for correct motion:
- Measure real-world distance traveled by camera
- Calculate: `unit_scale = real_distance_meters / asci_delta`
- Example: Camera moves 1.7m, ASCI Tz delta is 17.0 → `unit_scale = 0.1`

### Lens Parameters
Ensure these match your Nuke camera:
- `--focal-length-mm`: Focal length in millimeters
- `--sensor-width-mm`: Horizontal filmback/aperture
- `--sensor-height-mm`: Vertical filmback/aperture

### Frame Range
Match the frame range between:
- Nuke ASCI export
- ComfyUI render
- Use `--start-frame` and `--end-frame` if needed

## Next Steps

1. **Test the fix** with your existing data
2. **Validate** using the validation script
3. **Compare** old vs new output in ComfyUI
4. **Adjust** `unit_scale` if motion scale is still off
5. **Verify** lens parameters match production data

## Files Modified

- `nuke_ASCI-2-Pose_converter.py`: Added coordinate system conversion
- `test_nuke_scene.py`: New test scene generator
- `validate_conversion.py`: New validation tool
- `test_pipeline.sh`: New automated test script
- `README_TESTING.md`: New testing documentation

## Questions?

If you encounter issues:
1. Run `validate_conversion.py` to identify problems
2. Check `README_TESTING.md` for common issues
3. Verify `unit_scale` calculation
4. Ensure lens parameters match Nuke camera settings


