# Nuke Camera Export → ComfyUI Pose Pipeline

This document captures the end-to-end workflow we established for converting Nuke camera animations into RealEstate10k-compatible pose files that ComfyUI can ingest via the Koolook loader.

## 1. Source Data
- **Nuke camera**: standard `Camera` node with Translation / Rotation channels keyed over time.
- **Export step**: use Nuke's *Export Ascii* tool with the column mapping shown in `ASCI_source_data/AnimationChannels_and_values_firstFrame_plus-ASCI-window.png`.
  - Column order: `[translate.x, rotate.x, translate.y, rotate.y, rotate.z, translate.z]`.
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
- Reads each ASCI row using the known column order and converts rotation degrees → radians.
- Builds a rotation matrix using Nuke's ZXY order (`euler_zxy_to_matrix`).
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
python nuke_ASCI-2-Pose_converter.py `
    inputs/camTrack_v01.asci `
    outputs/camTrack_v01_converted.txt `
    --fps 25 --fx 0.5 --fy 0.5 --cx 0.5 --cy 0.5 `
    --width 1248 --height 704
```

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
  - Column mapping `[Tx, Rx, Ty, Ry, Rz, Tz]` and the ZXY rotation order.
  - Steps for regenerating the TXT and importing it into ComfyUI (`Koolook Load Camera Poses (Absolute Path)` + `CameraPoseVisualizer`).

## 5. Validation Checklist
1. In Nuke, verify the first keyframe values match the ASCI line (screenshot `DataValidation-nuke.png`).
2. After running the converter, open the TXT and confirm the translation entries in row 1 match the screenshot.
3. Load the TXT through the Koolook loader + CameraPoseVisualizer to visually confirm the path.

Keep this document handy as a “resume” of the pipeline so future adjustments reference the same assumptions and file locations.
