# Easy Pattern (Koolook)

| Aspect | Value |
|---|---|
| Display name | `Easy Pattern (Koolook)` |
| Node ID | `Easy_Pattern` |
| ComfyUI category | `Koolook/Testing` |
| Source | Koolook-native |
| Source file | [`k_easy_pattern.py`](../../../../k_easy_pattern.py) |

## What it does

Generates a batch of solid-color test images, optionally stamped with frame
numbers. It is a quick way to verify batch order, frame insertion, resizing,
video combine settings, and timeline wiring.

## When to use it

Use it before touching real media when you need to prove that a workflow keeps
frames in the expected order.

## Inputs

| Input | Type / default | Description |
|---|---|---|
| `batch_size` | `INT`, `81` | Number of generated images. |
| `width` / `height` | `INT`, `512` | Image dimensions. |
| `bg_color_mode` | enum, `Custom` | Background color preset, or `Custom` to read `bg_color`. |
| `show_text` | `BOOLEAN`, `true` | Draws the generated number text when on. |
| `text_color_mode` | enum, `White` | Text color preset, or `Custom` to read `text_color`. |
| `start_from` | `INT`, `1` | Number shown on the first frame. |
| `step` | `INT`, `1` | Amount added for each next frame. Negative steps count backward. |
| `font_size` | `INT`, `256` | Text size. If a system font cannot be loaded, Pillow's default font is used. |
| `position` | enum, `center` | Text placement. |
| `zero_pad` | `INT`, `0` | Minimum digit count. `7` with padding `3` becomes `007`. |
| `bg_color` | `STRING`, `#C71585` | Custom background hex color when `bg_color_mode = Custom`. |
| `text_color` | `STRING`, `#FFFFFF` | Custom text hex color when `text_color_mode = Custom`. |
| `prefix` / `suffix` | `STRING` | Optional text around the frame number. |

## Outputs

| Output | Type | Description |
|---|---|---|
| `images` | `IMAGE` | Generated image batch shaped `[batch_size, height, width, 3]`. |

## Recipes

### Check frame order

Set `batch_size = 121`, `start_from = 1`, `step = 1`, and `zero_pad = 3`.
After video combine or batch slicing, the visible numbers should still appear in
order.

### Label a cut range

Set `start_from = 41`, `batch_size = 81`, and `zero_pad = 3` to generate a
visible `041..121` range for testing an 81-frame VFX cut.

## Caveats

- Custom colors accept `#RRGGBB`, `RRGGBB`, or short `#RGB`. Invalid values
  fall back to black.
- Large `font_size` values can clip if the text is wider than the image.
