# Easy Load Video (Koolook)

| Aspect | Value |
|---|---|
| Display name | `Easy Load Video (Koolook)` |
| Node ID | `Easy_LoadVideo` |
| ComfyUI category | `Koolook/Video` |
| Source | Koolook subclass of VHS `Load Video Path` (runtime composition) |
| Source file | [`k_video_load.py`](../../../../k_video_load.py) |
| Requires | [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) installed alongside Koolook |

## What it does

Loads video through Video Helper Suite, but splits the path into two fields:

| Input | Meaning |
|---|---|
| `input_path` | Folder containing the video. |
| `video` | Video filename. |

This mirrors `Easy Video Combine (Koolook)`, where the output side is split
into `output_directory` + `filename_prefix`.

## How it picks what to load

| `input_path` | `video` | What is loaded |
|---|---|---|
| `E:/projects/shot01/renders/v003` | `plate_main.mp4` | `E:/projects/shot01/renders/v003/plate_main.mp4` |
| `/mnt/projects/shot01/renders/v003` | `plate_main.mp4` | `/mnt/projects/shot01/renders/v003/plate_main.mp4` |
| `shots/shot01` | `plate_main.mp4` | `<ComfyUI>/input/shots/shot01/plate_main.mp4` |
| empty | `E:/projects/shot01/plate_main.mp4` | Passed through to VHS `Load Video Path` unchanged. |

When `input_path` is set, `video` is treated as a filename field. If you
accidentally paste a path into `video`, only the basename is used.

## Inputs and outputs

Same as VHS `Load Video Path`, with one Koolook split-path addition:

| Input | Type | Description |
|---|---|---|
| `input_path` | `STRING` | Absolute folder path, or relative folder under ComfyUI's `input/` directory. |
| `video` | `STRING` | Video filename. If `input_path` is empty, this can also be a full path or URL accepted by VHS. |

All frame-rate, frame-cap, skip, resize, batch-manager, and output behavior is
delegated to VHS unchanged.

## Why this is a wrapper

`Easy_LoadVideo` subclasses VHS `Load Video Path` and composes
`input_path + video` into the single path string upstream already supports.
No VHS source is copied, so upstream fixes continue to flow through.
