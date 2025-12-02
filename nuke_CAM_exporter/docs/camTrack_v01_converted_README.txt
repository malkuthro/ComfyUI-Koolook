camTrack_v01_converted.txt
================================

Generated via `nuke_ASCI-2-Pose_converter.py` using:
- source: inputs/camTrack_v01.asci
- fps: 25.0 (timestamps spaced 40,000 µs)
- intrinsics: fx=fy=0.5, cx=cy=0.5
- pose size placeholders: width=1280, height=720
- ASCI layout: columns exported as [Tx, Rx, Ty, Ry, Rz, Tz]; rotations are degrees in Nuke's ZXY order and are converted to radians internally.
- rotation order: ZXY (Nuke default for this camera), translation units as stored in the ASCI export

To regenerate:
```
python nuke_ASCI-2-Pose_converter.py \
    inputs/camTrack_v01.asci \
    outputs/camTrack_v01_converted.txt \
    --fps 25 --fx 0.5 --fy 0.5 --cx 0.5 --cy 0.5 --width 1280 --height 720
```

Using in ComfyUI:
1. Copy `outputs/camTrack_v01_converted.txt` into ComfyUI's input folder (or reference the absolute path).
2. Use the `Koolook Load Camera Poses (Absolute Path)` node to read it.
3. Preview with `CameraPoseVisualizer` and feed into camera/motion encoder nodes.

Adjust intrinsics/fps parameters if your render setup differs; re-run the script to update the TXT.
