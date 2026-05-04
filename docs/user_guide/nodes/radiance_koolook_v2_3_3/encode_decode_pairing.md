# `Easy_hdr_VAE_encode` ↔ `Easy_hdr_VAE_decode` — pairing guide

Quick reference for which fields on the decoder must match the encoder
(and which are independent) to get a clean roundtrip.

## At a glance — pairing rules

| Encoder field | Pairs with decoder field | Rule |
|---|---|---|
| `pixels` (IMAGE in) | — | Connect your image / video stream here |
| `vae` (VAE) | `vae` (VAE) | **Must be the same VAE on both sides** — otherwise the latent is meaningless |
| `source_space` | `target_space` (decoder) | Independent. Set them to match if you want the same colour space out as in (typical roundtrip). Use a different `target_space` to convert colour space *as part of the VAE round-trip*. |
| `source_space` | `source_space` (decoder) | **Only matters when `hdr_mode = Compress (Log)`.** When in Log mode, the decoder's `source_space` tells it which log curve was used at encode time so it knows how to decompress. **Set the decoder's `source_space` equal to the encoder's `source_space`** when both are in Compress (Log) mode. Ignored otherwise. |
| `exposure` | `exposure_adjust` | Independent. Both apply in linear space, both default to 0. Use them as creative exposure controls — the encoder bumps before VAE, the decoder bumps after VAE. They don't need to be inverses unless you want the roundtrip exposure-neutral. |
| `alpha_handling` | (none) | Encode-only flag. The encoder's `alpha` IMAGE output carries the alpha **around** the VAE; the VAE itself never sees alpha. Re-composite the alpha with the decoded image downstream if you need 4-channel output. |
| `hdr_mode` | `hdr_mode` | **Must match** for clean roundtrips, especially for `Compress (Log)`. Mismatched modes will produce wrong colours by design — the decoder will misinterpret what the VAE saw. |
| `latent_sampling` | (none) | Encode-only. `sample` is non-deterministic (default). `mean` / `mode` are deterministic — best for img2img where you want minimum reconstruction noise. |
| `samples` (LATENT out) | `samples` (LATENT in) | Direct wire. KSampler / other nodes can sit between encode and decode here. |
| `debug_info` (STRING out) | `debug_info` (STRING out) | Independent diagnostics. Wire to a Show Text / Display Any node to see the actual dispatch path the wrapper took (e.g. `path=video-3d` for a 3-D-aware video VAE). |

## Worked example #1 — pure SDR roundtrip (cleanest reconstruction)

Use case: standard sRGB image input, want it back out matching the input as closely as possible.

```
ENCODER                              DECODER
────────────────────                 ────────────────────
source_space    = sRGB               target_space    = sRGB
exposure        = 0.0                exposure_adjust = 0.0
alpha_handling  = Preserve           hdr_mode        = Clip (SDR)
hdr_mode        = Clip (SDR)         source_space    = (irrelevant — not in Log mode)
latent_sampling = mean               
```

Why these choices:

- **`source_space`/`target_space` both sRGB** — input is gamma-coded sRGB and output should match.
- **`hdr_mode = Clip (SDR)` on both sides** — VAE was trained on sRGB-gamma in `[0, 1]`, this gives it exactly that.
- **`latent_sampling = mean`** — deterministic encode → reproducible test runs and minimum reconstruction noise.

## Worked example #2 — ACEScg HDR roundtrip (preserves highlights and wide gamut)

Use case: scene-linear ACEScg input from a 3-D render or EXR sequence with HDR highlights and wide-gamut colour, want it back out the same way.

```
ENCODER                                  DECODER
─────────────────────                    ─────────────────────
source_space    = ACEScg                 target_space    = ACEScg
exposure        = 0.0                    exposure_adjust = 0.0
alpha_handling  = Preserve               hdr_mode        = Compress (Log)  ← matches encoder
hdr_mode        = Compress (Log)         source_space    = ACEScg          ← tells decoder which log curve was used
latent_sampling = mean                   
```

Why these choices:

- **`hdr_mode = Compress (Log)` on both sides** — encodes through a cinema log curve (defaults to ARRI LogC4 because `ACEScg` is not itself a log space), preserves the wide dynamic range through the VAE, and the decoder side knows to apply the soft-shoulder + log-domain highlight denoise that this pipeline relies on.
- **Decoder's `source_space = ACEScg`** — the encoder picks ARRI LogC4 as the log curve when the source is ACEScg; the decoder needs to know what was picked, and matching `source_space` to the encoder is the way to communicate it. *This is the field most commonly forgotten when people debug "Compress (Log) gave me weird colour."*
- **Both `target_space` and `source_space` set to ACEScg on the decoder** — `target_space` is "what colour space do I want the output in"; `source_space` is "what colour space did encode see, so I know which log curve was used."

## Worked example #3 — cinema log workflow (LogC4 in, sRGB out for preview)

Use case: ARRI LogC4 source footage, viewing on a standard sRGB monitor.

```
ENCODER                                  DECODER
─────────────────────                    ─────────────────────
source_space    = ARRI LogC4             target_space    = sRGB
exposure        = 0.0                    exposure_adjust = 0.0
alpha_handling  = Ignore                 hdr_mode        = Compress (Log)
hdr_mode        = Compress (Log)         source_space    = ARRI LogC4   ← tells decoder this is LogC4
latent_sampling = mean
```

Why these choices:

- **Encoder `source_space = ARRI LogC4`** — input is LogC4-coded; we want the VAE to see exactly that (the log-to-log "round-trip" path is exact for `exposure = 0` because `to_linear` followed by `from_linear` to the same log space cancels out).
- **Decoder `target_space = sRGB`** — output goes to a standard monitor; the decoder converts log-decoded linear back to sRGB-gamma for display.
- **Decoder `source_space = ARRI LogC4`** — *required* so the decoder knows which log curve to use when reversing the VAE's log-coded output.

## Worked example #4 — Wan 2.2 (or other 3-D-native video VAE)

Use case: image sequence input through a video VAE that handles temporal natively.

```
ENCODER                              DECODER
──────────────────                   ──────────────────
source_space    = Linear             target_space    = Linear
exposure        = 0.0                exposure_adjust = 0.0
alpha_handling  = Ignore             hdr_mode        = Soft Clip
hdr_mode        = Soft Clip          source_space    = (irrelevant — not in Log mode)
latent_sampling = sample
```

What you'll see in `debug_info`:

```
Easy_hdr_VAE_encode (Koolook v2.3.3) | input shape (1, 41, 1056, 1872, 3) |
  path=video-3d | source_space=Linear | exposure=+0.00 |
  hdr_mode=Soft Clip | latent_sampling=sample | alpha_handling=Ignore
```

The `path=video-3d` confirms the wrapper detected a 5-D video tensor + a 3-D-aware VAE (the `vae.latent_dim == 3` check) and passed it through directly to Wan VAE's native temporal handling.

The decoder will produce a 4-D `(B*F, H, W, C)` IMAGE batch (i.e. one IMAGE per frame, ready for `SaveImage` to write `_00001.png`, `_00002.png`, …) — that 5-D-to-4-D normalization happens unconditionally on the decode side, so SaveImage gets the layout it expects.

## "I'm getting wrong colour" — quick diagnostic checklist

Before suspecting the VAE itself:

1. **Same `vae` connected to both nodes?** Mismatched VAE = garbage latent.
2. **`hdr_mode` matches between encoder and decoder?** Different modes = wrong colour by design.
3. **If using `Compress (Log)`, does the decoder's `source_space` match the encoder's `source_space`?** This is the single most common mistake.
4. **Is your input actually in the colour space you set for `source_space`?** A common gotcha: ComfyUI nodes default to passing sRGB-gamma data; if a node upstream produces sRGB but you set `source_space = Linear`, the wrapper will linearise data that's already linear. Set `source_space = sRGB` for typical ComfyUI image inputs.
5. **`debug_info` STRING output wired to a Show Text node?** Look at the `path=…` field — if it shows `video-iter` when you expected `video-3d`, your VAE doesn't have the `latent_dim == 3` attribute (likely not a video VAE), and the wrapper is iterating frames manually. That's still correct, just slower.
6. **For HDR / wide-gamut roundtrip with visible shift:** try switching `hdr_mode` from `Passthrough` to `Compress (Log)` on both sides. `Passthrough` is the worst HDR mode for fidelity because it lets out-of-domain values into the VAE.

If after all that the colour is still wrong, that's a real bug worth filing — the wrapper internals are covered by the dispatch tests in the test suite, but real-world VAE behaviour on edge cases can surface issues. Wire `debug_info` on both nodes, capture both strings, and open an issue with the input + output and both debug strings attached.

## Reference connectivity diagram

For a typical SDR roundtrip with KSampler in between (text-to-image, img2img, etc.):

```
   IMAGE input
     │
     │           ┌────────────┐
     │           │  Load VAE  │
     │           └────┬───────┘
     │                │
     ▼                ▼
   ┌──────────────────────────────────────┐
   │ Easy_hdr_VAE_encode                  │
   │   pixels        ●─────                │
   │   vae           ●─────                │
   │   source_space  = sRGB                │
   │   exposure      = 0.0                 │
   │   alpha_handling= Preserve            │
   │   hdr_mode      = Clip (SDR)          │  ← these three (source_space,
   │   latent_sampling = mean              │     hdr_mode, and on log workflows
   │                       samples ●─────  │     the source_space) drive what
   │                       debug_info ●──  │     the decoder needs to mirror
   │                       alpha    ●────  │
   └──────────────────────────────────────┘
                   │ samples
                   ▼
            ┌──────────────┐
            │   KSampler   │  (denoise = 0 for a "passthrough" roundtrip
            │   (or none)  │   test — bypasses diffusion)
            └──────┬───────┘
                   │ samples
                   ▼
   ┌──────────────────────────────────────┐
   │ Easy_hdr_VAE_decode                  │
   │   samples       ●─────                │
   │   vae           ●─────                │
   │   target_space    = sRGB              │
   │   exposure_adjust = 0.0               │
   │   hdr_mode        = Clip (SDR)        │  ← matches encoder's hdr_mode
   │   source_space    = (n/a — not Log)   │  ← only matters in Compress (Log)
   │                       image      ●────│
   │                       debug_info ●────│
   └──────────────────────────────────────┘
                   │ image
                   ▼
              ┌──────────┐
              │SaveImage │
              └──────────┘
```

For the alpha path, route the encoder's `alpha` IMAGE around the VAE chain and re-composite with the decoded `image` downstream (e.g. via an `ImageCompositeMasked` or similar). The VAE itself only ever sees the first 3 channels; alpha bypasses it.

## See also

- [`hdr_modes.md`](hdr_modes.md) — what each HDR mode does to your pixels (per-pixel worked examples).
- [`easy_hdr_vae_encode.md`](easy_hdr_vae_encode.md) — encoder per-input reference.
- [`easy_hdr_vae_decode.md`](easy_hdr_vae_decode.md) — decoder per-input reference.
