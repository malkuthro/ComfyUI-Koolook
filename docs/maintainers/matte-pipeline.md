# Koolook Matte — pipeline internals, fidelity, and debugging

`matte/` is a first-party, MIT-spirit **clean reimplementation** of the published
**VideoMaMa** one-step, mask-guided SVD matting method. It loads (never bundles)
the SVD base VAE + the VideoMaMa-fine-tuned UNet and runs a single denoising step
at `timestep=1` with CLIP cross-attention zeroed, then decodes the latents to a
luminance alpha.

Because it is an independent reimplementation, the trap is **subtle numerical
divergence from the reference** — the pipeline can run end-to-end, produce a
correctly-shaped alpha, and still be wrong (black, or blurry). This doc records
the two bugs that actually bit us, how to tell them apart, and the fast loop that
catches them.

The upstream reference we validate against is okdalto's `ComfyUI-VideoMaMa`
(unlicensed — do **not** vendor it; it's a *comparison oracle* only). Our nodes
are a near-verbatim reimplementation, so a side-by-side A/B isolates any
regression to `matte/pipeline.py`.

---

## 1. Latent scaling — feed the UNet RAW latents (×1)

**This is the single most important invariant.** The VideoMaMa UNet was fine-tuned
on conditioning + mask latents at the **raw VAE scale** (`vae.encode(...).latent_dist.sample()`,
with **no** `scaling_factor` applied net). Feed it anything else and it goes
out-of-distribution.

The confusing part: okdalto ships **two** pipeline classes that disagree, and only
one matches the shipped weights:

| class | net cond/mask scale | result |
|---|---|---|
| `VideoInferencePipeline` (what the ComfyUI node uses) | `sample × s` then `/ s` → **×1 (raw)** | **correct / sharp** |
| `...OnestepWithMask` (a diffusers-style variant) | `sample` then `/ s` → **×5.49** | over-scaled |

`s = vae.config.scaling_factor = 0.18215`, so `1/s ≈ 5.49`.

**Our `_encode_latents` must return raw latents** (no `/ s`):

```python
lat = torch.cat(parts, dim=0)        # RAW — do NOT divide by scaling_factor
return lat.reshape(b, f, *lat.shape[1:])
```

Decode is unchanged and matches both references: divide the UNet output by `s`
before `vae.decode` (`pred = pred.squeeze(0) / s`).

### The regression this caused
While chasing a black output we diffed against the *wrong* okdalto class
(`OnestepWithMask`, ×5.49) and added `/ scaling_factor` to `_encode_latents`.
That over-scaled the conditioning ~5.5×, pushed the UNet OOD, and produced a
**systematically blurry** matte on every frame — subtle enough to look like a
"resolution" problem. Removing the division restored parity with the reference
(rated 5/5 vs the desktop/vanilla output on identical inputs).

**Rule of thumb:** cond/mask latents in, alpha out — if the alpha is *soft on
every frame at every seed*, suspect a scale/normalization divergence in
`pipeline.py` before you suspect resolution.

---

## 2. Black output ≠ scaling

An all-black matte is almost always an **empty guide mask**, not the pipeline:

- **`SAM3Propagate` `direction = "backward"` with `start_frame = 0`** — nothing to
  propagate *before* frame 0, so the track is empty on frames 1…N. Use `forward`
  (or `both`) from a seed frame.
- **Text prompt that doesn't match the subject** (e.g. `"goats"` on eagle footage)
  → SAM3 finds nothing → empty mask.

Confirm with the SAM3 output preview: white subject on black = good guide; all
black = fix the seg, not the matte. (The pixel test: our matte, the guide mask,
and the source should all be non-black before you blame the sampler.)

---

## 3. Fidelity levers (in priority order)

The matte decodes from an SVD VAE latent (8× downsample), so **effective pixels on
the subject** is the dominant detail lever.

1. **Processing resolution.** `KoolookMatteSampler.max_resolution` + `resize_mode`
   (`shortest_side` = desktop parity, never upscales). The sampler downscales the
   input to fit `max_resolution`; the decoded alpha is upscaled back. Keep this at
   or above the crop's native short side to avoid a needless downscale→upscale
   round-trip.
2. **OOM auto-backoff reduces *resolution*** (`target × 0.85`) to fit VRAM — which
   directly costs detail (we saw a `1408 → 1152` backoff soften the fur). For
   *detail* work, prefer to buy VRAM back from the **temporal window**
   (`window_size 16 → 8`) instead, so spatial resolution stays high. (The default
   backoff-resolution behaviour favours temporal stability over sharpness — a
   deliberate but detail-costing trade.)
3. **FocusCrop hero-box** (`KoolookMatteFocusCrop`) crops to the subject's union
   bbox so `max_resolution` applies to just the subject. It never upscales and
   keeps aspect (single scale factor); `max_long_side = 0` = native crop.
   Run it on the **native-res plate before any downscale** — a crop downstream of
   a shrink can't recover lost detail.
4. **Aspect-preserving `/64` alignment.** The sampler aligns W and H independently
   to `/64` (SVD requirement); this nudges the aspect each step, and the alpha is
   subtly stretched on stitch. Minimise resamples: crop native → process native
   (or one clean cap) → stitch, with rescaling only when OOM forces it.

---

## 4. Fast debugging loop

Full clips are 150–220s and can OOM. To iterate on fidelity:

1. **Run one frame.** Set the shot-length control to `1` (Custom). ~4s, ~9GB, no
   OOM backoff — so you see the *intended* processing resolution, not a degraded
   one. (A ComfyUI restart resets the shot-length widget; re-set it to 1.)
2. **Side-by-side the oracle.** Drop okdalto's `VideoMaMaPipelineLoader` +
   `VideoMaMaSampler` into the graph, wire them to the **same** FocusCrop crop and
   SAM3 mask, copy the identical widget values. Our node and the oracle then differ
   only in `pipeline.py`, so any gap is ours.
3. **Rate 1–5 by eye** (1 = the known-blurry state, 5 = matches the reference).
   Automated sharpness metrics on ComfyUI temp previews are fragile (temp files get
   recycled); the human rating is the reliable signal.
4. Change one lever, re-run the single frame, re-rate.

The vanilla nodes stay in the graph as a permanent control.

---

## 5. dev-sync & checkpoints

The SVD base model + VideoMaMa UNet (~10 GB) live **only in the live install**
under `matte/checkpoints/` — they're gitignored and downloaded at runtime, so the
repo has no copy.

`scripts/sync_to_dev.py` mirrors each runtime path (`rmtree` + `copytree`). Left
naive, that **deletes `matte/checkpoints/` on every matte sync** and forces a
multi-GB re-download on the next node load (this manifested as ComfyUI "stuck at
8%"). The sync now protects it via `PRESERVE_IN_DEST = {"checkpoints"}` and a
selective `_clean_dest` that keeps preserved subtrees while still mirroring code.

If you add another node pack with runtime-downloaded weights, add its data dir
name to `PRESERVE_IN_DEST`.

See also [`dev-iteration-loop.md`](dev-iteration-loop.md) for the general
push/publish gates.
