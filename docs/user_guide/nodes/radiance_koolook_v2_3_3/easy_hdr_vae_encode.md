# Easy HDR VAE Encode (Koolook)

| Aspect | Value |
|---|---|
| Display name | `Easy HDR VAE Encode (Koolook)` |
| Node ID | `Easy_hdr_VAE_encode` |
| ComfyUI category | `Koolook/VAE` |
| Source | Koolook fork of fxtdstudios/radiance v2.3.3 (slim port — drops upstream's 4K cosine-blend tile engine) |
| Source file | [`forks/radiance_koolook/versions/v2_3_3/nodes_vae.py`](../../../../forks/radiance_koolook/versions/v2_3_3/nodes_vae.py) |
| Pairs with | [`Easy_hdr_VAE_decode`](easy_hdr_vae_decode.md) |
| Cross-cutting | [HDR modes](hdr_modes.md), [Encode/decode pairing](encode_decode_pairing.md) |

## What it does

Encodes a 4-D `(B, H, W, C)` image batch *or* a 5-D `(B, F, H, W, C)`
video tensor into a VAE latent, with cinema-grade colour-space handling
and HDR awareness. Works with image VAEs (SD 1.5 / SDXL / Flux) **and**
3-D-aware video VAEs (Wan 2.2 / Hunyuan / CogVideoX / LTX) using a
single rank-aware code path.

## Inputs

### Required

| Input | Type | Description |
|---|---|---|
| `pixels` | `IMAGE` | The image batch or video tensor to encode. Accepts 3-channel (RGB) or 4-channel (RGBA). Rank can be 4-D or 5-D. |
| `vae` | `VAE` | The VAE to encode through. Same VAE must be wired to the matching decoder. |
| `source_space` | enum (12 options) | What colour space the input pixels are in. Drives a linearisation step before exposure / HDR processing. See [valid values](#valid-values-for-source_space) below. |

### Optional

| Input | Type | Default | Description |
|---|---|---|---|
| `exposure` | `FLOAT` | `0.0` | Exposure adjustment in stops, applied **in linear space** (correct semantics regardless of `source_space`). Range: `-10.0` to `+10.0`. |
| `alpha_handling` | `Preserve` / `Ignore` | `Preserve` | If `Preserve` and the input has 4 channels, the alpha channel is surfaced as the third output (a separate `IMAGE`). If `Ignore` (or input has only 3 channels), the alpha output is a zeros tensor of compatible shape. The VAE itself only ever encodes the first 3 channels regardless. |
| `hdr_mode` | enum (4 options) | `Soft Clip` | What to do with linearised values just before they hit `vae.encode()`. **Must match the decoder's `hdr_mode`** for clean roundtrips. See [HDR modes deep dive](hdr_modes.md). |
| `latent_sampling` | `sample` / `mean` / `mode` | `sample` | How to sample from the VAE posterior distribution. `sample` is ComfyUI's default (random per-run). `mean` and `mode` are deterministic — pick the posterior mean for img2img-style minimum-noise reconstruction. Falls back to `sample` if the underlying VAE class doesn't expose the raw `first_stage_model` for direct mean access. |

## Outputs

| Output | Type | Description |
|---|---|---|
| `samples` | `LATENT` | Standard ComfyUI LATENT dict `{"samples": tensor}`. For 5-D video paths through a 3-D-aware VAE, the inner tensor is also 5-D `(B, C, F, latH, latW)`. |
| `debug_info` | `STRING` | Per-run diagnostic — input shape, dispatch path (`image` / `video-iter` / `video-3d`), and parameter values. Wire to a Show Text / Display Any node to verify which dispatch branch fired. |
| `alpha` | `IMAGE` | Alpha channel surfaced separately so it can be routed *around* the VAE (which can't encode alpha) and re-composited with the decoded image downstream. Zeros tensor of compatible shape if the input had no alpha or `alpha_handling = Ignore`. |

## Valid values for `source_space`

The full set of 12 colour spaces:

| Group | Values |
|---|---|
| Common | `Linear`, `sRGB`, `Raw` |
| ACES family | `ACEScg`, `ACES 2065-1` |
| Other linear gamuts | `Rec.2020 Linear` |
| Cinema log curves | `ARRI LogC3`, `ARRI LogC4`, `Sony S-Log3`, `Panasonic V-Log`, `DaVinci Intermediate`, `RED Log3G10` |

`Raw` bypasses all colour processing — the input pixels go straight to the VAE unchanged (after exposure, which still applies as a raw multiplication). All other values trigger the appropriate to-linear-Rec.709 conversion before exposure and HDR processing.

## Internal dispatch (what the wrapper actually does)

Pseudocode, ignoring error handling:

```
if source_space == "Raw":
    img *= 2 ** exposure          # raw-domain multiplication
    img_for_vae = img             # no colour or HDR processing
else:
    img = to_linear(img, source_space)         # source → linear Rec.709
    img *= 2 ** exposure                       # exposure in linear
    if hdr_mode == "Compress (Log)":
        log_space = source_space if is_log_space(source_space) else "ARRI LogC4"
        img_for_vae = from_linear(img, log_space)
    else:
        img_gamma = linear_to_srgb(img)
        if hdr_mode == "Clip (SDR)":   img_for_vae = clamp(img_gamma, 0, 1)
        elif hdr_mode == "Soft Clip":  img_for_vae = soft_clip(img_gamma, knee=0.85)
        elif hdr_mode == "Passthrough": img_for_vae = img_gamma  # no clamp

# Sequence-aware encode dispatch
is_video = img_for_vae.ndim == 5
is_3d_vae = hasattr(vae, "latent_dim") and vae.latent_dim == 3

if is_video and not is_3d_vae:
    # Iterate frames, stack on temporal dim
    latent = stack([encode_with_sampling_mode(vae, img_for_vae[:, fi, ...], latent_sampling)
                    for fi in range(F)], dim=2)
else:
    latent = encode_with_sampling_mode(vae, img_for_vae, latent_sampling)

return ({"samples": latent}, debug_info, alpha_out)
```

The full implementation lives at [`nodes_vae.py:Easy_hdr_VAE_encode.encode`](../../../../forks/radiance_koolook/versions/v2_3_3/nodes_vae.py).

## Common mistakes

- **Setting `source_space = Linear` on a typical ComfyUI sRGB-gamma input.** Most ComfyUI image-loading nodes produce sRGB-gamma data. If you set `source_space = Linear`, the wrapper will linearise data that's already linear (apply sRGB EOTF on already-EOTF'd data). Set to `sRGB` for typical ComfyUI inputs.
- **Using `Passthrough` as a "do nothing" mode for HDR roundtrip.** It actually does the worst job for HDR fidelity — the VAE was trained on `[0, 1]` data and produces noise on out-of-domain inputs. Use `Compress (Log)` for HDR or `Clip (SDR)` for SDR; reserve `Passthrough` for diagnostics.
- **Forgetting to match `hdr_mode` on the decoder.** The decoder's `hdr_mode` determines how it interprets the VAE's output; mismatched modes give wrong colour by construction. See [encode/decode pairing](encode_decode_pairing.md).
- **Wiring the `alpha` output back into a 4-channel-expecting node downstream.** Most ComfyUI IMAGE-typed nodes assume 3-channel; the `alpha` output is a separate single-channel IMAGE meant for explicit re-compositing.

## See also

- [HDR modes deep dive](hdr_modes.md)
- [Encode/decode pairing](encode_decode_pairing.md)
- [`Easy_hdr_VAE_decode`](easy_hdr_vae_decode.md)
