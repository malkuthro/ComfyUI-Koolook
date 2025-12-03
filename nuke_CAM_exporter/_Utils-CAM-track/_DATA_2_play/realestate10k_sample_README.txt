RealEstate10k sample pose file
================================

Source: `_Utils-CAM-track/Dataset-Design_Real-Estate10k/0bf152ef84195293.txt` (first 12 frames).
Adjustments: columns 6 and 7 (pose_width/pose_height) normalized to 1280 / 720 so ComfyUI loaders know the native resolution.

How to load in ComfyUI:
1. In the `Koolook Load Camera Poses (Absolute Path)` node, set `file_path` to the absolute path of `realestate10k_sample.txt`.
2. Optionally override pose width/height in the node inputs if the render resolution differs.
3. Pipe the resulting `CAMERACTRL_POSES` into `CameraPoseVisualizer` for validation and then into your motion encoder nodes.

Metadata:
- Frames included: 12 consecutive entries (timestamps 158692025–161795128 µs).
- Header line retains the original YouTube URL for provenance.
- Values remain RealEstate10k normalized intrinsics and world-to-camera matrices; only width/height placeholders were populated.
