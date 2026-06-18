# Koolook Load Camera Poses (Absolute Path)

| Aspect | Value |
|---|---|
| Display name | `Koolook Load Camera Poses (Absolute Path)` |
| Node ID | `KoolookLoadCameraPosesAbsolute` |
| ComfyUI category | `Koolook/Camera` |
| Source | Koolook-native helper |
| Source file | [`k_easy_track.py`](../../../../k_easy_track.py) |

## What it does

Loads a RealEstate10k-style CameraCtrl pose `.txt` file from a path you type or
wire into the node, then returns `CAMERACTRL_POSES`.

This is a narrow helper for workflows that need CameraCtrl poses stored outside
ComfyUI's normal input picker. It intentionally does not browse the filesystem
or rewrite pose timing; it just reads the file, validates each pose row, and
sets the width/height fields returned to downstream CameraCtrl nodes.

## Inputs

| Input | Type / default | Description |
|---|---|---|
| `file_path` | `STRING`, `input/poses/example_track.txt` | Path to the pose text file. Absolute paths and `~` user expansion are supported. |
| `pose_width` | `FLOAT`, `1280` | Width value written into each returned pose row. |
| `pose_height` | `FLOAT`, `720` | Height value written into each returned pose row. |

## Expected file format

The file should match the RealEstate10k/CameraCtrl text layout:

- The first line is metadata/header content and is skipped.
- Each following non-empty row must contain `19` space-separated numeric values.
- Rows with the wrong column count or non-numeric values raise a clear error.

The node replaces columns `5` and `6` in each pose row with `pose_width` and
`pose_height` before returning the pose list.

## Output

| Output | Type | Description |
|---|---|---|
| `CAMERACTRL_POSES` | `CAMERACTRL_POSES` | Parsed pose rows ready for CameraCtrl-compatible downstream nodes. |

## Tips

- Use this only when the pose file already exists on disk and you know its
  path. It is not a camera-track generator.
- Keep `pose_width` and `pose_height` aligned with the resolution expected by
  the CameraCtrl workflow consuming the poses.
- If a file fails to load, first check that it has at least one data row after
  the header and that every data row has exactly `19` numeric values.
