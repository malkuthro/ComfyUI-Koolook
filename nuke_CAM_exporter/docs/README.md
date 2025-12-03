# Nuke ASCI Camera Export → ComfyUI Pose Pipeline

This document captures the end-to-end workflow we established for converting Nuke camera animations into RealEstate10k-compatible pose files that ComfyUI can ingest via the Koolook loader.

## 1. Source Data
- **Nuke camera**: standard `Camera` node with Translation / Rotation channels keyed over time.
- **Export step**: use Nuke's *Export Ascii* tool with the column mapping shown in `ASCI_source_data/AnimationChannels_and_values_firstFrame_plus-ASCI-window.png`.
- Column order: `[translate.x, rotate.x, translate.y, rotate.y, translate.z, rotate.z]`.
- Prefer to stick to the combined export unless you want to split the channels into two files. When splitting, export translations as `Tx Ty Tz` and rotations as `Rx Ry Rz` (degrees) and reference them via the config keys described below.
  - Rotations are exported in **degrees** using Nuke's **ZXY rotation order** (see `DataValidation-nuke.png`).
- **Raw file**: `nuke_CAM_exporter/inputs/camTrack_v01.asci` (162 lines covering frames 1001–1161 in this example).

```
Line 1 (frame 1001):
-1.0521379710 -0.7493053079 0.2045794427 -102.4243546000 4.2196946140 -0.2049491256
```

This line corresponds to the highlighted translation / rotation values keyed in Nuke (blue fields in the screenshot).

## 2. Conversion Script
File: `nuke_CAM_exporter/nuke_ASCI-2-Pose_converter.py`

Key behavior:
- Parses command-line arguments for input path, output path, fps, intrinsics, and placeholder pose width/height.
- Optionally ingests a JSON config (`--config`) that stores lens data, render resolution, file paths, unit scale, and frame caps per shot.
- Reads each ASCI row using the known column order and converts rotation degrees → radians.
- Builds a rotation matrix using Nuke's ZXY order (`euler_zxy_to_matrix`).
- Converts Nuke's camera-center translations into RealEstate10k-style world→camera translations via `t = -R * C`; without this step ComfyUI interprets movement on the wrong axis.
- Writes RealEstate10k-style rows:
  - Columns 1–7: timestamp (µs), fx, fy, cx, cy, pose_width, pose_height.
  - Columns 8–19: `[R|t]` flattened row-major (each row’s three rotation entries followed by that row’s translation component).
- Header line records all parameters for traceability.

Usage example:
```
cd nuke_CAM_exporter
python nuke_ASCI-2-Pose_converter.py \
    inputs/camTrack_v01.asci \
    outputs/camTrack_v01_converted.txt \
    --fps 25 --fx 0.5 --fy 0.5 --cx 0.5 --cy 0.5 --width 1280 --height 720
```

```
cd nuke_CAM_exporter
python nuke_ASCI-2-Pose_converter.py `
    inputs/camTrack_v01.asci `
    outputs/camTrack_v06_converted.txt `
    --fps 25 --fx 0.5 --fy 0.5 --cx 0.5 --cy 0.5 `
    --width 1248 --height 704 `
    --start-frame 10 --end-frame 70 --max-frames 48
```

### JSON-driven workflow
Use the template in `configs/camera_config_template.json` (copy + rename per shot):

```json
{
  "input": "../inputs/camTrack_v01.asci",
  "input_translation": null,
  "input_rotation": null,
  "output": "../outputs/camTrack_v06_converted.txt",
  "fps": 25.0,
  "unit_scale": 1.0,
  "lens": {
    "focal_length_mm": 33.5212349,
    "sensor_width_mm": 36.0,
    "sensor_height_mm": 24.0,
    "principal_point_offset_x_mm": 0.0,
    "principal_point_offset_y_mm": 0.0
  },
  "resolution": {
    "width_px": 1248,
    "height_px": 704
  },
  "frame_range": {
    "start": 10,
    "end": 170,
    "max_frames": 60
  }
}
```

Set `"input_translation"` / `"input_rotation"` when you export translations and rotations as two separate ASCI files. Leave them `null` (or omit entirely) when you rely on the combined `input` file.

Run:

```bash
cd nuke_CAM_exporter
python nuke_ASCI-2-Pose_converter.py --config configs/camera_config_v01.json
```

The script converts the physical lens info into RealEstate-style intrinsics:

- `fx = focal_length_mm / sensor_width_mm`
- `fy = focal_length_mm / sensor_height_mm`
- `cx = 0.5 + principal_offset_x_mm / sensor_width_mm`
- `cy = 0.5 + principal_offset_y_mm / sensor_height_mm`

### 🔑 Getting the unit scale right (don’t skip!)

ComfyUI expects world→camera translations in **meters**. If your Nuke track uses a different scale, the pose will look correct in direction but wrong in amplitude. Use this checklist every time:

1. **Measure a real distance** from the plate (dolly travel, two survey points, set dressing width). In Nuke you can:
   - Use CameraTracker’s *Set Reference Distance* or a custom Axis/Card pair with a known separation.
   - Enter the surveyed value (e.g. 6.3 m) directly into a distance expression.
2. **Read the raw ASCI delta** for the same span. Example: `translate.z` goes from `-0.215132` to `-0.105132` ⇒ raw delta = `0.11`.
3. **Compute the multiplier**:  
   `unit_scale = real_distance_meters / asci_delta`
   - If the real move is 3.5 m: `3.5 / 0.11 = 31.82` ⇒ set `"unit_scale": 31.82`.
   - If the scene is authored in centimeters (Nuke “World to Meters 100”), use `0.01`.
4. **Store the measurement** in your config so anyone regenerating the TXT knows where the number came from.

> **Tip:** When the camera tracker already scaled the scene (you set a reference distance), the raw ASCI deltas are in meters and `unit_scale` stays at `1.0`.

Once `unit_scale` is set, rerun the converter—the exported Z travel will match the original plate exactly.

Frame limiting options:
Frame limiting options:

- `frame_range.start` / `--start-frame`: zero-based index of the first ASCI row to keep.
- `frame_range.end` / `--end-frame`: exclusive stop index (omit or set `null` to use all remaining frames).
- `frame_range.max_frames` / `--max-frames`: hard cap after slicing (useful when a loader needs a specific length) / Wan22 = 80 (for Wan's 81 frames recomended workflows)

Snippet (core formatting function):
```python
def format_pose_rows(rotation, translation):
    r = rotation
    t = translation
    return [
        r[0][0], r[0][1], r[0][2], t[0],
        r[1][0], r[1][1], r[1][2], t[1],
        r[2][0], r[2][1], r[2][2], t[2],
    ]
```

## 3. Output
- Default target: `nuke_CAM_exporter/outputs/camTrack_v01_converted.txt`
- First rows look like:
```
source:inputs/camTrack_v01.asci fps:25.0 fx:0.5 fy:0.5 cx:0.5 cy:0.5 width:1280.0 height:720.0
0.000000000000 0.500000000000 ... 0.716894083480 0.478508303891 -0.507043071325 0.148742702111 ...
```
- Each row’s translation components (`... 0.1487427 ... -102.4243 ... -0.2049491`) now match the camera values from Nuke frame 1.

## 4. Documentation
- `docs/camTrack_v01_converted_README.txt` explains:
  - Source file, fps, intrinsics, and placeholder dimensions.
- Column mapping `[Tx, Rx, Ty, Ry, Tz, Rz]` and the ZXY rotation order.
  - Steps for regenerating the TXT and importing it into ComfyUI (`Koolook Load Camera Poses (Absolute Path)` + `CameraPoseVisualizer`).

## 5. Validation Checklist
1. In Nuke, verify the first keyframe values match the ASCI line (screenshot `DataValidation-nuke.png`).
2. After running the converter, open the TXT and confirm the translation entries in row 1 match the screenshot.
3. Load the TXT through the Koolook loader + CameraPoseVisualizer to visually confirm the path.

## 6. Using in ComfyUI:
1. Copy `outputs/camTrack_v01_converted.txt` into ComfyUI's input folder (or reference the absolute path).
2. Use the `Koolook Load Camera Poses (Absolute Path)` node to read it.
3. Preview with `CameraPoseVisualizer` and feed into camera/motion encoder nodes.

Keep this document handy as a “resume” of the pipeline so future adjustments reference the same assumptions and file locations.

## 7. Plotting the camera path (Nuke-style top view)
Whenever you need to sanity-check where the camera sits relative to the plate, use the helper script `testing_tools/plot_top_view.py`. It renders a quad view that matches Nuke’s “view from -Y” (horizontal axis = +Z frame left, vertical axis = +X frame down). Defaults are baked in—you don’t need to remember extra flags.

### Required files
- Config JSON for the shot (e.g. `nuke_CAM_exporter/configs/camera_config_v01.json`) with at least:
  - `"input"` *or* `"input_translation"` / `"input_rotation"` pointing at ASCI exports.
  - `"output"` pointing at the converted TXT (used to overlay world-to-camera translations).
  - `"unit_scale"` and `"frame_range"` set the same way you run the converter.
- Optional: override paths via CLI if you want to plot a different file set.

### Command
```
cd nuke_CAM_exporter
py -3 testing_tools/plot_top_view.py \
    --config configs/camera_config_v01.json \
    --output testing_tools/top_view.png
```

On Windows PowerShell:
```
cd nuke_CAM_exporter
py -3 testing_tools/plot_top_view.py `
    --config configs/camera_config_v01.json `
    --output testing_tools/top_view.png
```

What you get:
- Green curve = raw Nuke camera centers (after `unit_scale`).
- Purple curve = world→camera translations from the converted TXT.
- 1×1 m square centered on origin with the +Z / -X quadrant shaded (mirrors the Nuke viewport).
- Axis labels explicitly show frame directions (`+Z (frame left)`, `+X (frame down)`).
- Blue cones/arrows at the first and last Nuke keyframes show the camera’s facing direction within the same top view.

Use this plot to verify the camera lives in the expected quadrant or to explain offsets before re-rendering. The script respects the same config that drives the converter, so you only need to keep one JSON per shot.