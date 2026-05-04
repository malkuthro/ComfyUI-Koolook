# Koolook/VAE — User Guide

Cinema-grade VAE encode/decode with HDR / wide-gamut color awareness.
Designed to work with image VAEs (SD1.5, SDXL, Flux) **and** video VAEs
(Wan 2.2, Hunyuan, CogVideoX, LTX) using a single rank-aware code path.

## Pages in this folder

| Page | Use it to … |
|---|---|
| [`easy_hdr_vae_encode.md`](easy_hdr_vae_encode.md) | Look up what each input on the encoder does, and how its outputs are wired |
| [`easy_hdr_vae_decode.md`](easy_hdr_vae_decode.md) | Same, for the decoder |
| [`encode_decode_pairing.md`](encode_decode_pairing.md) | Get the connectivity right between encoder and decoder for a clean roundtrip — which fields must match, which are independent, with worked examples for SDR / HDR / cinema-log / Wan 2.2 video |
| [`hdr_modes.md`](hdr_modes.md) | Understand the four HDR modes (`Clip (SDR)`, `Soft Clip`, `Compress (Log)`, `Passthrough`) — what each one does to pixels before hitting the VAE, with per-pixel worked examples |

## Quick mental model

```
   pixels (any of 12 color spaces, any rank)
     │
     ▼
 ┌──────────────────────────┐
 │ Easy_hdr_VAE_encode      │  source_space → linear → exposure
 │                          │  → hdr_mode-dependent gamma/log encoding
 │                          │  → vae.encode (with sample/mean/mode)
 └──────────────────────────┘
     │ samples (LATENT)
     │ debug_info (STRING — per-run diagnostic)
     │ alpha (IMAGE — for round-tripping alpha around the VAE)
     │
     ▼
 KSampler (or whatever) — optional
     │
     ▼
 ┌──────────────────────────┐
 │ Easy_hdr_VAE_decode      │  vae.decode → 5-D-to-4-D normalization
 │                          │  → hdr_mode-dependent recovery (log
 │                          │    decompress + soft-shoulder if Log mode)
 │                          │  → exposure_adjust (linear)
 │                          │  → linear → target_space
 └──────────────────────────┘
     │ image (IMAGE)
     │ debug_info (STRING)
```

## Source

| Aspect | Value |
|---|---|
| Origin | Koolook fork of [fxtdstudios/radiance](https://github.com/fxtdstudios/radiance) v2.3.3 (commit `f262f47`) |
| Slim port | Drops upstream's 4K cosine-blend tile engine that was incompatible with video VAEs |
| Local path | [`forks/radiance_koolook/versions/v2_3_3/`](../../../../forks/radiance_koolook/versions/v2_3_3/) |
| Attribution + change notes | [`forks/THIRD_PARTY.md`](../../../../forks/THIRD_PARTY.md) |
| License | GPL-3.0 (inherited from upstream Radiance) |
