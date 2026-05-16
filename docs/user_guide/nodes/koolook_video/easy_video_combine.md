# Easy Video Combine (Koolook)

| Aspect | Value |
|---|---|
| Display name | `Easy Video Combine (Koolook)` |
| Node ID | `Easy_VideoCombine` |
| ComfyUI category | `Koolook/Video` |
| Source | Koolook subclass of VHS `VideoCombine` (runtime composition) |
| Source file | [`k_video_combine.py`](../../../../k_video_combine.py) |
| Requires | [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) installed alongside Koolook |

## What it does

Same as VHS's standard `Video Combine` — gif / webp / mp4 / ProRes / any
configured format from a stack of frames, optional audio mux, optional
batch-manager streaming for long sequences — **except** it can write the
output directly to an absolute path on disk instead of being confined to
ComfyUI's `output/` directory.

Designed for VFX / facility pipelines where renders need to land under a
project tree (`E:/projects/shot01/comp/v003/`) rather than in ComfyUI's
local sandbox.

## How it picks where to save

The decision is driven entirely by **the shape of the value** you type
into `filename_prefix` — no toggle, no extra pin:

| `filename_prefix` value | Where the file lands |
|---|---|
| `AnimateDiff`, `clip01`, `shots/v003` | `<ComfyUI>/output/<prefix>_<counter>.<ext>` (upstream behavior, sandboxed) |
| `E:/renders/shot01/v003` (Windows) | `E:/renders/shot01/v003_<counter>.<ext>` |
| `/mnt/projects/shot01/v003` (Linux/Mac) | `/mnt/projects/shot01/v003_<counter>.<ext>` |
| `E:/renders/shot01/v003/` (trailing slash) | `E:/renders/shot01/v003_<counter>.<ext>` (`v003` becomes the file root) |
| `E:/renders/shot01/` (only a dir) | `E:/renders/shot01_<counter>.<ext>` — *probably not what you want; append a file root* |

The basename of the absolute path is treated as the **filename root**
(not as a subdirectory). VHS's existing counter (`_00001`, `_00002`, …)
and extension are appended exactly as in upstream.

## Inputs

Same as VHS `Video Combine` plus one Koolook-specific toggle.

### Required widgets

| Input | Type | Default | Description |
|---|---|---|---|
| `images` | `IMAGE` or `LATENT` | — | Same as upstream. Latents need a connected `vae`. |
| `frame_rate` | `FLOAT` / `INT` | `8` | Same as upstream. |
| `loop_count` | `INT` | `0` | Same as upstream (gif/webp loop count). |
| `filename_prefix` | `STRING` | `AnimateDiff` | **Dual-mode.** Relative → ComfyUI's `output/`. Absolute → writes there directly. See the table above. |
| `format` | enum | `image/gif` | Same as upstream — gif, webp, or any ffmpeg-registered video format. |
| `pingpong` | `BOOLEAN` | `false` | Same as upstream. |
| `save_output` | `BOOLEAN` | `true` | Same as upstream when `filename_prefix` is relative. When `filename_prefix` is absolute, this toggle is ignored — the absolute path wins. |

### Optional inputs

| Input | Type | Description |
|---|---|---|
| `audio` | `AUDIO` | Same as upstream. |
| `meta_batch` | `VHS_BatchManager` | Same as upstream. |
| `vae` | `VAE` | Same as upstream — required when `images` is a `LATENT`. |
| `create_path_if_missing` | `BOOLEAN` (default `false`) | **Koolook-only.** When `filename_prefix` is absolute and the parent directory does not exist, auto-create it (recursively). Off by default so typos surface as errors instead of silently spawning directory trees. |

## Outputs

Identical to upstream — `Filenames` (`VHS_FILENAMES`) carrying the list
of files actually written. For absolute-path renders the entries are
fully-qualified paths.

## UI preview caveat

ComfyUI's frontend serves preview thumbnails through its `/view` HTTP
endpoint, which is locked to files under the local output directory. So:

- **Relative `filename_prefix`** — the workflow tile shows the standard
  inline video preview, same as upstream.
- **Absolute `filename_prefix`** — the file *is written* (you'll see it
  on disk and in the `Filenames` output), but the inline preview tile
  stays empty because Comfy can't fetch files outside its sandbox.

This is a Comfy frontend limitation, not a Koolook one. The render is
fine; just open the file with your favourite player.

## How it works under the hood

`Easy_VideoCombine` is a Python subclass of VHS's
`videohelpersuite.nodes.VideoCombine`. Its `combine_video()` does
exactly two things before delegating back to upstream:

1. Looks at `filename_prefix`. If it's relative, calls
   `super().combine_video(...)` unchanged — pure passthrough.
2. If it's absolute, scoped-patches `folder_paths.get_save_image_path`
   to return the absolute parent directory and basename, then calls
   `super().combine_video(...)`. A `try/finally` restores the original
   function whether the call succeeds or raises.

No VHS source is copied. Every encoder bug fix, new format, audio
improvement, or batch-manager enhancement that lands upstream flows
through automatically the next time you update VHS.

If VHS isn't installed at all, `Easy_VideoCombine` is simply not
registered in `NODE_CLASS_MAPPINGS` and a one-line message appears in
the ComfyUI console at startup explaining why.

## Why this isn't an upstream PR (yet)

It can be, eventually — and probably should be. The path-discrimination
pattern is borrowed from spacepxl's `ComfyUI-HQ-Image-Save` (`SaveEXR`)
which has carried it since 2023, so there's precedent for landing it
upstream. In the meantime, packaging it as a Koolook node means every
workstation that installs Koolook gets it immediately, without waiting
on upstream review and without per-workstation patches that
ComfyUI-Manager's auto-update would clobber.

See [`forks/THIRD_PARTY.md`](../../../../forks/THIRD_PARTY.md) for the
full attribution + license audit.
