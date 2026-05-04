# Easy HDR VAE Decode (Koolook)

| Aspect | Value |
|---|---|
| Display name | `Easy HDR VAE Decode (Koolook)` |
| Node ID | `Easy_hdr_VAE_decode` |
| ComfyUI category | `Koolook/VAE` |
| Source | Koolook fork of fxtdstudios/radiance v2.3.3 (slim port — drops upstream's 4K cosine-blend tile engine) |
| Source file | [`forks/radiance_koolook/versions/v2_3_3/nodes_vae.py`](../../../../forks/radiance_koolook/versions/v2_3_3/nodes_vae.py) |
| Pairs with | [`Easy_hdr_VAE_encode`](easy_hdr_vae_encode.md) |
| Cross-cutting | [HDR modes](hdr_modes.md), [Encode/decode pairing](encode_decode_pairing.md) |

## What it does

Decodes a `LATENT` back into pixel space, with cinema-grade colour-space
output and HDR awareness. Mirrors `Easy_hdr_VAE_encode` and accepts both
4-D image latents and 5-D video latents from 3-D-aware VAEs (Wan 2.2,
Hunyuan, CogVideoX, LTX).

For 3-D-aware VAEs that return a 5-D `(B, F, H, W, C)` tensor, the
decoder unconditionally reshapes it to ComfyUI's standard 4-D
`(B*F, H, W, C)` IMAGE batch — so `SaveImage` and other downstream
IMAGE-typed nodes see one IMAGE per frame and write `_00001.png`,
`_00002.png`, … as expected.

## Inputs

### Required

| Input | Type | Description |
|---|---|---|
| `samples` | `LATENT` | The latent to decode. Same VAE that produced this latent must be wired to `vae`. |
| `vae` | `VAE` | The VAE to decode through. **Must match the encoder's VAE.** |
| `target_space` | enum (12 options) | The colour space the output IMAGE should be in. Same set of 12 values as the encoder's `source_space`. |

### Optional

| Input | Type | Default | Description |
|---|---|---|---|
| `exposure_adjust` | `FLOAT` | `0.0` | Post-decode exposure adjustment in stops, applied **in linear space**. Range: `-10.0` to `+10.0`. |
| `hdr_mode` | enum (4 options) | `Clip (SDR)` | How to interpret the VAE's output before linearising. **Must match the encoder's `hdr_mode`** for clean roundtrips. `Compress (Log)` triggers the recovery pipeline (soft-shoulder + log-domain highlight denoise + log-curve decompression). Other modes treat the VAE output as sRGB-gamma data. See [HDR modes deep dive](hdr_modes.md). |
| `source_space` | enum (12 options) | `Linear` | **Only relevant when `hdr_mode = Compress (Log)`.** Tells the decoder which log curve was used at encode time, so it knows how to reverse the encoding. Set this to whatever you set `source_space` to on the encoder. **Ignored for all other `hdr_mode` values.** |

## Outputs

| Output | Type | Description |
|---|---|---|
| `image` | `IMAGE` | Decoded image in the requested `target_space`. Always 4-D `(B, H, W, C)` — for 5-D video latents, the temporal dimension is collapsed into the batch dimension. |
| `debug_info` | `STRING` | Per-run diagnostic — output shape, dispatch path (`image` / `video-iter` / `video-3d`), and parameter values. |

## Valid values for `target_space` and `source_space`

Same 12 colour spaces as the encoder's `source_space`. See
[`easy_hdr_vae_encode.md`](easy_hdr_vae_encode.md#valid-values-for-source_space)
for the full list.

## Internal dispatch (what the wrapper actually does)

Pseudocode, ignoring error handling:

```
latent = samples["samples"]
is_video = latent.ndim == 5
is_3d_vae = hasattr(vae, "latent_dim") and vae.latent_dim == 3

# Sequence-aware decode dispatch
if is_video and not is_3d_vae:
    img = cat([vae.decode(latent[:, :, fi, ...]) for fi in range(F)], dim=0)
else:
    img = vae.decode(latent)

# Normalize 5-D output to 4-D (B*F, H, W, C) for ComfyUI IMAGE compatibility
if img.ndim == 5:
    img = img.reshape(-1, H, W, C)

if target_space == "Raw":
    img *= 2 ** exposure_adjust
else:
    if hdr_mode == "Compress (Log)":
        # Recovery pipeline for log-coded VAE output
        log_space = source_space if is_log_space(source_space) else "ARRI LogC4"
        knee, ceiling, threshold, strength = LOG_PROFILE_HDR_PARAMS[log_space]
        img = soft_log_shoulder(img, knee=knee, ceiling=ceiling)
        img = denoise_log_highlights(img, threshold=threshold, strength=strength)
        img = to_linear(img, log_space)             # log → linear Rec.709
    else:
        img = srgb_to_linear(img)                   # sRGB-gamma → linear Rec.709

    img *= 2 ** exposure_adjust                     # exposure in linear
    img = from_linear(img, target_space)            # linear → target colour space

return (img, debug_info)
```

The full implementation lives at [`nodes_vae.py:Easy_hdr_VAE_decode.decode`](../../../../forks/radiance_koolook/versions/v2_3_3/nodes_vae.py).

## Common mistakes

- **Mismatched `hdr_mode` between encoder and decoder.** Most common is the encoder using `Compress (Log)` while the decoder uses `Clip (SDR)` (the default), or vice versa. Result: very wrong colour. The encoder's `debug_info` will say `hdr_mode=Compress (Log)`; the decoder's will say `hdr_mode=Clip (SDR)`. They must agree.
- **Forgetting to set `source_space` on the decoder when using `Compress (Log)`.** The decoder defaults `source_space = Linear`, which means "if Compress (Log) is on, use the default ARRI LogC4 log curve" — fine if the encoder also fell back to LogC4, *not fine* if you encoded with a different log curve. Set the decoder's `source_space` equal to the encoder's `source_space`.
- **Expecting a 5-D output from a 3-D-aware video VAE.** The decoder always returns a 4-D `(B*F, H, W, C)` IMAGE — that's ComfyUI convention for video sequences. If you need temporal grouping, that's downstream of this node.

## See also

- [HDR modes deep dive](hdr_modes.md)
- [Encode/decode pairing](encode_decode_pairing.md)
- [`Easy_hdr_VAE_encode`](easy_hdr_vae_encode.md)
