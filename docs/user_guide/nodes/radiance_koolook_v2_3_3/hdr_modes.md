# HDR Modes — what each one does to your pixels

`hdr_mode` is a parameter on both `Easy_hdr_VAE_encode` and
`Easy_hdr_VAE_decode`. It controls **what happens to linearised pixel
values just before they enter `vae.encode()`** (and, on the decode side,
how the VAE's output is interpreted on the way back to linear).

For roundtrip workflows the encoder's `hdr_mode` and the decoder's
`hdr_mode` should **match**. See [`encode_decode_pairing.md`](encode_decode_pairing.md)
for the cross-node pairing rules.

## The four modes side by side (encode side)

| Mode | What it does to values | What VAE sees | What's lost |
|---|---|---|---|
| **Clip (SDR)** | Linear → sRGB-gamma → **hard clamp to [0, 1]** | Standard SDR-range data — what most VAEs were trained on | Anything above `1.0` (HDR highlights) → cut to white. Anything below `0` (out-of-gamut negatives from wide-gamut sources like ACEScg) → cut to black. |
| **Soft Clip** | Linear → sRGB-gamma → **tanh rolloff above 0.85**, then clamp to [0, 1] | SDR-range data with smoothly curved highlights instead of a hard cliff | Highlights above ~0.85 progressively compressed; very bright pixels still end up near 1.0 but with no harsh clipped-white line. Negatives still clamped to 0. |
| **Compress (Log)** | Linear → **cinema log curve** (ARRI LogC4 by default, or matches `source_space` if `source_space` is itself a log space). No clamp. | Log-coded HDR data — squashes a wide dynamic range into [0, ~1.08] | Almost nothing — the log curve is designed for this. Caveat: the VAE wasn't trained on log-coded data, so reconstruction has its own characteristic noise. The decoder addresses this with per-profile soft-shoulder + log-domain highlight denoise — but it only kicks in when the decoder also has `hdr_mode = Compress (Log)`. |
| **Passthrough** | Linear → sRGB-gamma. **No clamp at all.** Negatives stay negative; values above 1.0 stay above 1.0. | Whatever the input gives it — including out-of-domain values | VAE behaviour on values it never saw in training (`>1` or `<0`) is undefined. It typically mashes them — clips highlights anyway, makes negatives weird. Output is often *worse* than `Clip (SDR)` in practice. |

## Concrete per-pixel examples

### SDR midtone — linear input value `0.4`

| Mode | sRGB-gamma stage | Final to VAE |
|---|---|---|
| Clip (SDR) | `0.665` | `0.665` (in [0, 1]) |
| Soft Clip | `0.665` | `0.665` (below 0.85 knee, passthrough) |
| Compress (Log) | (skips sRGB step entirely) | `~0.55` LogC4-coded |
| Passthrough | `0.665` | `0.665` |

All four modes preserve a normal SDR midtone identically — they only diverge at the extremes.

### Super-white HDR highlight — linear input value `5.0` (e.g. sun, explosion, VDB fire)

| Mode | sRGB-gamma stage | Final to VAE |
|---|---|---|
| Clip (SDR) | `1.94` | `1.0` ← **lost super-white** |
| Soft Clip | `1.94` | `~0.99` ← **lost most super-white**, but smoothly |
| Compress (Log) | (skips sRGB) | `~0.79` LogC4-coded ← **preserved** |
| Passthrough | `1.94` | `1.94` ← VAE will likely mash this anyway |

`Compress (Log)` is the only mode that actually preserves HDR highlights through the VAE.

### Out-of-gamut negative — linear input value `-0.05` (e.g. saturated cyan from ACEScg → Rec.709)

| Mode | Behaviour |
|---|---|
| Clip (SDR) | Clamped to `0.0` ← lost |
| Soft Clip | Clamped to `0.0` ← lost |
| Compress (Log) | Floored to `0.0` (no valid signal below 0 in log space) ← lost |
| Passthrough | Stays at `-0.05` ← VAE mashes it; usually worse than just clamping |

Wide-gamut colour preservation isn't really a job the VAE can do; if you need it, keep your wide-gamut data outside the VAE chain.

## When to use which

| Mode | Use when |
|---|---|
| **Clip (SDR)** | You have a normal SDR image / sequence (values fit in [0, 1] after colour-space conversion) and you want the cleanest VAE reconstruction in the SDR range. The VAE is in its native domain, so this gives the highest fidelity for non-HDR content. |
| **Soft Clip** | You have bright/HDR-ish content but you don't want to deal with log curves on both sides. The smooth highlight rolloff is more pleasant than hard clipping for things like rim-lit subjects, sky highlights, etc. |
| **Compress (Log)** | You're doing actual HDR / cinema / film work — ACES pipelines, ARRI / Sony / RED / Panasonic source material, scene-linear EXR sequences, etc. **Pair with the same mode on decode** plus `source_space` set to whichever log curve you encoded with. Best dynamic-range fidelity, at the cost of needing matched encode/decode settings. |
| **Passthrough** | Almost never the right answer for production. It's an "I know what I'm doing, give me the raw values" escape hatch. For roundtrip fidelity it's usually worse than `Clip (SDR)` because the VAE has nothing trained for out-of-domain inputs and produces characteristic noise. Useful for diagnosing what the VAE is actually receiving. |

## Trade-off matrix

|                  | SDR fidelity | HDR preservation | Highlight smoothness | Roundtrip cleanliness | Requires matched decode mode |
|---|---|---|---|---|---|
| Clip (SDR)       | ★★★★★ | ✗            | ✗ (hard cut)        | ★★★★★ (in-range)     | No — same path on decode |
| Soft Clip        | ★★★★ | ✗            | ★★★★                | ★★★★                  | No — same path on decode |
| Compress (Log)   | ★★★ | ★★★★★         | ★★★ (per-profile)   | ★★★★ (with matched decode) | **Yes** — decode also Compress (Log) + matching `source_space` |
| Passthrough      | ★★ | ★             | ✗                    | ★ (often worse than Clip)  | No — same path on decode |

## Behind the scenes — where each mode is implemented

- **Encode side:** [`forks/radiance_koolook/versions/v2_3_3/nodes_vae.py`](../../../../forks/radiance_koolook/versions/v2_3_3/nodes_vae.py),
  search for `if hdr_mode == "Compress (Log)"` to see the dispatch.
- **Log curves + soft-shoulder + log-domain denoise:**
  [`forks/radiance_koolook/versions/v2_3_3/color_helpers.py`](../../../../forks/radiance_koolook/versions/v2_3_3/color_helpers.py).
- **Per-profile HDR parameters** (`LOG_PROFILE_HDR_PARAMS`): tuned per
  curve based on each curve's slope at code 1.0. RED Log3G10 (the
  steepest, slope > 1100) gets the most conservative knee and most
  aggressive denoise; Sony S-Log3 / ARRI LogC4 (moderate-steep) get the
  loosest. See the docstring in `color_helpers.py` for the full table.
- **Why we don't just hard-clamp Compress (Log)**: VAE reconstruction
  noise is roughly uniform in log-coded space, but log→linear amplifies
  that noise exponentially in the highlights. A hard clamp at 1.0 would
  destroy legitimate super-white signal. The soft-shoulder + log-space
  denoise approach is the same principle DaVinci Resolve and Nuke use.
