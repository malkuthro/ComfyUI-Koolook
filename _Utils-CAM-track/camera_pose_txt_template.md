# Camera Tracking TXT Template

This guide consolidates the RealEstate10k pose specification (`Dataset-Design_Real-Estate10k/DatasetDesign.txt`) with the behavior of the `LoadCameraPosesFromPath` node from AnimateDiff-Evolved (`Github-Repos/Comfyui-Animatedfiff-evolved/nodes_cameractrl.py`) and the `CameraPoseVisualizer` from KJNodes (`Github-Repos/Comfyui-kjnodes/nodes.py`). Follow it to author pose files that ComfyUI can ingest and preview using the workflow shown in `REF_screenshots/Desired-Workflow-Steps.png`.

## Format Constraints
- **Header line**: The loaders expect the first line to be a plain text identifier (typically the original dataset URL). The node skips this line, so anything can be placed there as long as it is a single line of text with no spaces-only entries.
- **Value separator**: Lines 2+ must be purely numeric values separated by single spaces. Do not include commas, tabs, comments, or blank lines or the parsing in `LoadCameraPosesFromPath` / `LoadCameraPosesFromFile` will fail.
- **Row length**: Every frame line contains exactly 19 columns: timestamp, 4 intrinsic terms, 2 reserved width/height slots, and a flattened 3×4 pose matrix (row-major). This matches the expectations of the CameraCtrl nodes and the `CameraPoseVisualizer`, which reshape indices 7–18 into a world-to-camera matrix.
- **Units & ranges**:
  - `timestamp`: integer microseconds since clip start; must be monotonically increasing.
  - `focal_length_x/y` (`fx`, `fy`): normalized focal lengths (multiply by pixel width/height to recover actual values as described in the dataset doc).
  - `principal_point_x/y` (`cx`, `cy`): normalized principal point coordinates in [0, 1].
  - Pose matrix: represents `[R|t]` mapping world coordinates to camera coordinates. Ensure rows/columns form a valid rigid transform to avoid warped trajectories when visualizing.
- **Width/height placeholders**: Columns 6–7 are not consumed by AnimateDiff directly. The loader overwrites them with `CAM.DEFAULT_POSE_WIDTH` (1280) and `CAM.DEFAULT_POSE_HEIGHT` (720), but you may pre-populate them if another tool downstream expects specific dimensions.

## Template Layout
1. **Line 1** – Metadata / source information. Example: `https://example.com/my_clip`. This line is ignored by the loader but retained for traceability.
2. **Lines 2+** – One frame per line, 19 space-separated values following the schema below.

| # | Column | Description |
|---|--------|-------------|
| 1 | `timestamp` | Integer microseconds since the clip begins. |
| 2 | `fx` | Normalized focal length along x; multiply by image width for pixel-space value. |
| 3 | `fy` | Normalized focal length along y; multiply by image height. |
| 4 | `cx` | Normalized principal point x coordinate (0–1). |
| 5 | `cy` | Normalized principal point y coordinate (0–1). |
| 6 | `pose_width` | Optional original image width in pixels (will be overwritten if left default). |
| 7 | `pose_height` | Optional original image height in pixels. |
| 8–11 | `r00 r01 r02 r03` | First row of the 3×4 `[R|t]` matrix (rotation + translation). |
| 12–15 | `r10 r11 r12 r13` | Second row of the matrix. |
| 16–19 | `r20 r21 r22 r23` | Third row of the matrix. |

### Sample Frame Line
```
0 0.62 0.62 0.5 0.5 1280 720 0.998 -0.010 0.060 0.150 0.012 0.995 -0.099 -1.200 -0.059 0.100 0.993 2.350
```
- Replace the timestamp, intrinsics, and matrix entries with your real measurements. The example encodes a gently moving camera with a translation of `(0.150, -1.200, 2.350)` and slight rotations.
- Ensure each subsequent line increments the timestamp and updates the matrix accordingly.

### Authoring Tips
- Generate the rotation matrix from yaw/pitch/roll or quarternions, then append translations in meters (or consistent world units) for the final column.
- Keep precision to at least 3 decimal places to avoid jitter.
- Validate the determinant of `R` stays close to 1 to confirm orthonormality.

## Workflow Integration Steps
1. **Place the TXT file** inside your ComfyUI `input` directory (e.g., `ComfyUI/input/my_camera_track.txt`).
2. **Provide the path** via a `Text Multiline` node or string field, mirroring step 1 in `Desired-Workflow-Steps.png`. Use absolute paths (preferred) to avoid ambiguity.
3. **Load poses** with `LoadCameraPoses (Path)` from AnimateDiff-Evolved. This node outputs `CAMERACTRL_POSES` suitable for downstream motion modules.
4. **Visual check**: Feed the pose output into `CameraPoseVisualizer` from KJNodes to verify the 3D trajectory. Adjust `base_xval`, `zval`, and `scale` if the model appears too small or large.
5. **Encode for video**: Connect the same pose output to nodes such as `WanVideo FunCamera Embeds` / `WanVideo FunCamera Wrapper` to supply camera embeddings before rendering, as indicated in the screenshot.

## Troubleshooting
- **"Could not parse pose file"**: Confirm each line has 19 space-separated numbers and no extra whitespace or tabs. An empty line at the end of the file can also cause this error.
- **Camera path looks skewed**: Double-check the `[R|t]` block ordering. `CameraPoseVisualizer` assumes the matrix is row-major world-to-camera; transposing rows/columns will flip axes.
- **Scale mismatch**: If previews look extremely zoomed, ensure `fx`/`fy` are normalized (0–1 range). If you are using pixel-space focal lengths, divide by width/height first.
- **Jumps between frames**: Verify timestamps increase monotonically and that `R` is continuous (small deltas). Abrupt changes often originate from incorrect unit conversions or mixing degrees/radians when generating the rotation matrices.

Following this template keeps your pose data aligned with AnimateDiff-Evolved CameraCtrl expectations and the visualization tooling in ComfyUI, reducing iteration time when building camera-aware workflows.
