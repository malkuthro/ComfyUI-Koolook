# Wan 2.2 Easy Prompt (Koolook)

| Aspect | Value |
|---|---|
| Display name | `Wan 2.2 Easy Prompt (Koolook)` |
| Node ID | `EasyWan22Prompt` |
| ComfyUI category | `Koolook/Wan_Video` |
| Source | Koolook-native |
| Source file | [`k_easy_wan22_prompt.py`](../../../../k_easy_wan22_prompt.py) |
| Field config | [`config.json`](../../../../config.json) |

## What it does

Turns a set of cinematography/style dropdowns plus one free-text prompt body
into a Wan 2.2 prompt string. If a `CLIP` is connected, it also encodes that
combined text into a `CONDITIONING` output.

## When to use it

Use it as a compact positive-prompt builder when you want repeatable camera,
lighting, composition, motion, and style terms without writing them from memory
every time.

## Inputs

The dropdown inputs are loaded from `config.json`. Each field defaults to
`none`; `none` is skipped and does not appear in the output prompt.

| Input | Description |
|---|---|
| Cinematography/style dropdowns | Optional prompt fragments such as lighting, shot size, camera angle, composition, motion, emotion, camera movement, visual style, and visual effects. |
| `body` | Main free-text prompt. Appended after selected dropdown values. |
| `clip` | Optional `CLIP`. When connected, the node returns encoded conditioning for the combined prompt. |

## Outputs

| Output | Type | Description |
|---|---|---|
| `combined_prompt` | `STRING` | Selected dropdown values plus `body`, comma-separated. |
| `fields_only` | `STRING` | Selected dropdown values only, without `body`. |
| `conditioning` | `CONDITIONING` | Encoded combined prompt when `clip` is connected; otherwise empty. |

## Recipe

Pick only the fields you actually want to constrain. For example:

```text
light_source = Overcast lighting
shot_size = Medium wide shot
camera_angle = Low angle shot
visual_style = 2D anime style
body = a person walking across a rain-soaked bridge
```

`combined_prompt` becomes:

```text
Overcast lighting, Medium wide shot, Low angle shot, 2D anime style, a person walking across a rain-soaked bridge
```

## Caveats

- Dropdown order controls the order of terms in the generated prompt.
- The node hashes `config.json` for change detection, so edits to the field
  config should cause the node to re-run.
- Connecting `CLIP` is optional. If you only need prompt text, leave it
  unconnected.
