# Easy Utility (Koolook)

| Aspect | Value |
|---|---|
| Display name | `Easy Utility (Koolook)` |
| Node ID | `Easy_Utility` |
| ComfyUI category | `Koolook/Utility` |
| Source | Koolook-native |
| Source file | [`k_easy_utility.py`](../../../../k_easy_utility.py) |

## What it does

Converts small workflow control values into strings. The current mode turns an
integer into a zero-padded string, which is mainly useful for shared version
tokens.

## When to use it

Use it when one numeric version control should drive multiple path-aware nodes,
for example `EasyAIPipeline` and `Easy_VideoCombine`.

## Inputs

| Input | Type / default | Description |
|---|---|---|
| `mode` | `int_to_padded_string` | Conversion mode. More modes can be added later without changing saved workflows. |
| `int_value` | `INT`, `1` | Number to convert. |
| `prefix` | `STRING`, empty | Text prepended before the padded number. Leave empty when feeding Koolook version fields that already normalize bare digits to `v###`. |
| `pad_width` | `INT`, `3` | Minimum digit count. `1` with width `3` becomes `001`. |

## Outputs

| Output | Type | Description |
|---|---|---|
| `string` | `STRING` | Converted string such as `001`, `v001`, or `shot_001` depending on `prefix`. |

## Recipe

For a shared render version:

```text
int_value = 7
prefix = ""
pad_width = 3
```

Wire `string` into `EasyAIPipeline.version`. The pipeline resolves bare `007`
to `v007`, keeping the `v` rule in one downstream place.

Use `prefix = v` only when feeding a node that expects the literal `v007`
string and does not normalize version tokens itself.
