# Understanding Camera Rotations and Coordinate Systems

## The 90° Y-Axis Rotation Question

If you notice that your solved camera has a **90° rotation on the Y-axis** (or any large rotation), this is **normal and expected**. Here's why:

### Why Nuke Camera Solver May Rotate the Camera

1. **Ambiguity in camera solving**: Camera tracking has inherent ambiguities - the solver can't always determine the absolute world orientation
2. **Coordinate system choice**: The solver picks a convenient coordinate system that may not align with your scene's "natural" orientation
3. **Plate orientation**: If your plate is rotated (e.g., portrait vs landscape), the solver may rotate the camera to match

### Does This Affect the Conversion?

**No!** The converter handles **any rotation correctly**, including:
- 90° rotations
- 180° rotations  
- Arbitrary rotations

The coordinate system conversion (`convert_yup_to_ydown`) is a **linear transformation** that preserves all rotations correctly. It only changes how the coordinate system is expressed (Y-up → Y-down), not the actual camera orientation.

### How to Verify Rotations Are Correct

Use the debug tool to analyze your camera rotations:

```bash
python debug_rotation.py inputs/your_camera.asci --frames 10
```

This will show:
- Rotation values for each frame
- Whether Y rotation is near ±90° (warns if so)
- Forward direction vectors (should be preserved after conversion)
- Rotation matrix determinants (should be ~1.0)

### Test 90° Rotation Handling

To verify the converter handles 90° rotations correctly:

```bash
python debug_rotation.py inputs/your_camera.asci --test-90deg
```

This runs a specific test case with a 90° Y rotation to ensure the conversion preserves it correctly.

### What If Motion Still Doesn't Match?

If your camera has a 90° rotation but motion still doesn't match in ComfyUI, check:

1. **Unit scale** - This is the most common issue
   ```bash
   # Measure real distance, calculate scale
   unit_scale = real_distance_meters / asci_translation_delta
   ```

2. **Lens parameters** - Must match Nuke camera exactly
   - Focal length (mm)
   - Sensor width/height (mm)

3. **Coordinate system** - The converter handles this, but verify:
   ```bash
   python validate_conversion.py inputs/camera.asci outputs/camera_converted.txt --unit-scale 0.1
   ```

4. **Frame range** - Ensure you're comparing the same frames

### Example: Analyzing Your Camera

If your camera has rotations like:
```
Frame 1: Y rotation = -102.4°
Frame 2: Y rotation = -102.6°
```

This is fine! The converter will:
1. Convert Euler angles → rotation matrix (preserving the -102° rotation)
2. Convert coordinate system (Y-up → Y-down)
3. Compute world-to-camera translation

The **-102° rotation is preserved** - it's just expressed in OpenCV's coordinate system instead of Nuke's.

### Key Takeaway

**A 90° (or any) rotation in the solved camera is NOT a problem.** The converter preserves all rotations correctly. If motion doesn't match, look at:
- `unit_scale` (most common)
- Lens parameters
- Frame range
- Coordinate system validation (use `validate_conversion.py`)

