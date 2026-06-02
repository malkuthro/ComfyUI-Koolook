# Easy AI Pipeline (Koolook)

| Aspect | Value |
|---|---|
| Display name | `Easy AI Pipeline (Koolook)` |
| Node ID | `EasyAIPipeline` |
| ComfyUI category | `Koolook/Pipeline` |
| Source | Koolook-native |
| Source file | [`k_ai_pipeline.py`](../../../../k_ai_pipeline.py) |

## What it does

Builds a VFX-style output path and filename from one base folder, shot name,
method tag, version token, extension, frame count, and seed. It is meant to be
the single path/version control node near the top of a workflow.

The node returns the final file path, filename, resolved version string, output
directory, frame count, seed, and shot name so downstream save/sampler nodes can
all use the same values.

## When to use it

Use it when you want a workflow to render into repeatable folders such as:

```text
<base>/<shot_name>/<ai_method>/v003/<shot_name>_<ai_method>_v003.%04d.exr
```

It is especially useful when many save nodes should share the same shot,
version, seed, and frame count without hand-editing each one.

## Inputs

| Input | Type / default | Description |
|---|---|---|
| `shot_duration` | `INT`, `81` | Pass-through frame count. Wire it into samplers, video savers, or other nodes that need total frames. |
| `seed_value` | `INT`, `453453453` | Pass-through master seed for the workflow. |
| `instruction` | `STRING` | Read-only hint text. The actual base path goes in `base_directory_path`. |
| `base_directory_path` | multiline `STRING` | Root output folder. Pasted paths are cleaned: surrounding quotes, whitespace, control characters, and trailing slashes are stripped where safe. |
| `extension` | `STRING`, `.%04d.exr` | Filename extension or sequence suffix. `%04d` is kept for downstream sequence savers. Whitespace is removed. |
| `shot_name` | `STRING` | Shot identifier. Used in the path and filename unless `no_subfolders` removes it from the path. |
| `ai_method` | `STRING` | Method tag such as `v2v`, `upscale`, or `denoise`. Blank is allowed and does not leave dangling underscores. |
| `version` | `STRING`, `v001` | Version token. Bare numbers like `2` become `v002`; existing strings like `v003` or `final` are preserved. |
| `disable_versioning` | `BOOLEAN`, `false` | Removes the version folder and version filename segment. |
| `enable_overwrite` | `BOOLEAN`, `false` | When off, errors if the exact final file already exists. Existing directories are OK. |
| `no_subfolders` | `BOOLEAN`, `false` | Drops `shot_name` and `ai_method` from the directory path, but keeps them in the filename. The version folder remains unless versioning is disabled. |

## Outputs

| Output | Type | Description |
|---|---|---|
| `WRITE_file_path` | `STRING` | Full path to the target file or sequence pattern. |
| `output_name` | `STRING` | Filename only, without the directory. |
| `version_string` | `STRING` | Resolved version token such as `v001`, or empty when versioning is disabled. |
| `output_directory` | `STRING` | Directory part of the output path. Created when missing if possible. |
| `shot_duration` | `INT` | Same value as the input. |
| `seed_value` | `INT` | Same value as the input. |
| `shot_name` | `STRING` | Raw shot name for labels or downstream metadata. |

## Recipes

### Standard versioned render

Set:

```text
base_directory_path = E:/jobs/project/renders
shot_name = sh010
ai_method = upscale
version = 3
extension = .%04d.exr
```

The version resolves to `v003`, and the output path becomes:

```text
E:/jobs/project/renders/sh010/upscale/v003/sh010_upscale_v003.%04d.exr
```

### Flat directory with version folder

Turn on `no_subfolders` to keep the filename descriptive while avoiding
`shot_name/ai_method` subfolders:

```text
E:/jobs/project/renders/v003/sh010_upscale_v003.%04d.exr
```

Turn on both `no_subfolders` and `disable_versioning` only when you truly want a
single flat output folder.

## Caveats

- `shot_name` and `ai_method` are sanitized before joining onto the base path, so
  accidental absolute-looking values cannot escape the base directory.
- Internal slashes in `shot_name` or `ai_method` can create nested directories
  when `no_subfolders` is off. They are flattened to underscores in filenames.
- The overwrite check is for the final file path only; it does not block an
  existing output directory.
