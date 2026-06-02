# Easy Resize (Koolook)

| Aspect | Value |
|---|---|
| Display name | `Easy Resize (Koolook)` |
| Node ID | `EasyResize_Koolook` |
| ComfyUI category | `Koolook/Image` |
| Source | Koolook reimplementation inspired by KJNodes `Resize Image V2` |
| Source file | [`k_easy_resize.py`](../../../../k_easy_resize.py) |

## What it does

Resizes an image to a target aspect ratio and model-friendly dimensions. It can
stretch, letterbox, or pillarbox, snap the final size to a divisibility multiple,
resize an optional mask, and return extra outputs for compositing and debugging.

## When to use it

Use this before image/video models that expect a specific aspect ratio or sizes
divisible by a number such as `32`. It is also useful when you need a resized
image and a matching resized mask from the same operation.

## Inputs

| Input | Type / default | Description |
|---|---|---|
| `image` | `IMAGE` | Image or image batch to resize. |
| `base_on` | `Width` / `Height` | Which dimension is controlled by `base_size`; the other is computed from `aspect_ratio`. |
| `base_size` | `INT`, `512` | Target width or height before snapping to `divisible_by`. |
| `aspect_ratio` | `STRING`, `16:9` | Target aspect ratio as `W:H`, for example `16:9`, `1:1`, or `9:16`. |
| `divisible_by` | `INT`, `32` | Snaps final dimensions to a multiple. Use the model's latent/grid requirement. |
| `upscale_method` | enum | Resampling method passed to ComfyUI upscale. |
| `keep_proportion` | `stretch` | `stretch` fills the target exactly; `letterbox` or `pillarbox` preserve image proportions and add bars. |
| `crop_position` | enum | Placement of the preserved image when bars are added. For `letterbox`, top/bottom matter; for `pillarbox`, left/right matter. |
| `pad_color_mode` | enum | Color used for letterbox/pillarbox bars. `Custom` reads `pad_color`. |
| `panel_color_mode` | enum | Color for the separate `COLOR_PANEL` output and masked composition background. |
| `device` | `cpu` / `cuda` | Device used for resize/composite tensors. |
| `mask` | optional `MASK` | Optional mask resized with the image. |
| `pad_color` | `STRING`, `0, 0, 0` | Custom RGB values in `0..1` when `pad_color_mode` is `Custom`. |
| `panel_color` | `STRING`, `0, 0, 0` | Custom RGB values in `0..1` when `panel_color_mode` is `Custom`. |
| `invert_composed_MASK` | `BOOLEAN`, `false` | Flips how the resized mask blends `IMAGE` with `COLOR_PANEL` for `composed_IMAGE`. |

## Outputs

| Output | Type | Description |
|---|---|---|
| `IMAGE` | `IMAGE` | Resized image. |
| `composed_IMAGE` | `IMAGE` | Resized image composited over `COLOR_PANEL` when a mask is connected; otherwise same as `IMAGE`. |
| `MASK` | `MASK` | Resized mask, or empty when no mask is connected. |
| `inverted_MASK` | `MASK` | `1.0 - MASK`, or empty when no mask is connected. |
| `width` / `height` | `INT` | Final snapped output dimensions. |
| `COLOR_PANEL` | `IMAGE` | Solid color image matching output size. |
| `original_width` / `original_height` | `INT` | Input image dimensions. |
| `original_aspect_ratio` | `STRING` | Simplified source ratio such as `16:9`. |

## Recipes

### Prepare 16:9 frames for a video model

Set `base_on = Width`, `base_size = 1280`, `aspect_ratio = 16:9`, and
`divisible_by = 32`. The node computes a model-friendly height and returns both
final dimensions for downstream nodes.

### Preserve a portrait image inside a landscape canvas

Set `keep_proportion = pillarbox`, choose a `pad_color_mode`, and use
`crop_position` to anchor the image left, right, or center.

## Caveats

- `letterbox` and `pillarbox` preserve image proportions; they do not crop image
  content.
- `pad_color` and `panel_color` expect comma-separated float RGB values such as
  `0, 0, 0` or `1, 0.5, 0`.
- The legacy node ID `EasyResize` is still registered only for saved-workflow
  compatibility. New workflows should use `EasyResize_Koolook`.
