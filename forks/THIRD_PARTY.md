# Third-Party Code & Attribution

This file tracks all third-party code incorporated into `ComfyUI-Koolook`,
the upstream license, and what (if anything) was modified locally.

The package as a whole is licensed **GPL-3.0** (see [`LICENSE`](../LICENSE))
because it incorporates GPL-3.0 code from upstream Radiance per the entries
below — GPL-3.0 §5(c) requires the whole work to be GPL-3.0.

## Entries

### fxtdstudios/radiance — v1.0.1 baseline (Koolook fork)

- **Name:** Radiance (FXTD Studios)
- **Upstream repo URL:** https://github.com/fxtdstudios/radiance
- **Upstream commit/tag used:** `f1b8ae330848fa08aba24c9d3e355cb432d3515b`
  (closest public baseline; upstream `comfyui` branch tip when v1.0.1 was
  packaged. The `v1.0.1` tag itself was never published upstream.)
- **License:** GPL-3.0
- **Local path(s):** [`forks/radiance_koolook/versions/v1_0_1/`](radiance_koolook/versions/v1_0_1/)
  - `nodes_hdr.py`
  - `nodes_color_management.py`
  - `nodes_dna.py`
  - `__init__.py`
  - `UPSTREAM_PIN.yaml`
- **What changed locally:**
  - `RadianceOCIOColorTransformV2` is a Koolook-modified copy of upstream's
    `RadianceOCIOColorTransform`: added `_list_colorspaces()` helper, a
    second `STRING` debug output, config-file path sanitization, alpha-channel
    preservation, and a multi-frame batch processing path (flattening all
    frames into one `applyRGB` call).
  - `LoadImageEXRSequence` is a Koolook-original sequence loader (no upstream
    counterpart at this commit).
  - `RadianceVAEEncode` and `RadianceVAEDecode` are Koolook-original wrappers
    that integrate ComfyUI VAEs with HDR/ACEScg color-space handling. Upstream
    v1.0 did not ship VAE classes; v2.0+ added the much larger
    `RadianceVAE4KEncode` family.
  - All exposed node IDs are namespaced with the suffix `__koolook_v1_0_1` to
    avoid collisions with installed copies of upstream Radiance.
- **Why changed:** Adapt color-management nodes for VFX pipelines where image
  inputs are sequences (video frames) rather than single stills, and add HDR
  VAE wrappers tailored for Koolook's AI image/video generation workflows.
- **Last reviewed:** 2026-05-03

### fxtdstudios/radiance — v2.3.3 VAE subset (Koolook fork)

- **Name:** Radiance (FXTD Studios)
- **Upstream repo URL:** https://github.com/fxtdstudios/radiance
- **Upstream commit/tag used:** `f262f47ddfda01ece154bf80c22769b1e4cef795`
  (release commit "v2.3.3: Finalize release for GitHub and Comfy Registry").
- **License:** GPL-3.0
- **Local path(s):** [`forks/radiance_koolook/versions/v2_3_3/`](radiance_koolook/versions/v2_3_3/)
  *(scaffolded in v0.1.2 — see `feat(forks): port Radiance v2.3 VAE Encode/Decode`)*
- **What changed locally:**
  - Only the `RadianceVAE4KEncode` and `RadianceVAE4KDecode` *interface
    surfaces* are mirrored — Koolook ships a slimmer rank-agnostic
    reimplementation that supports image sequences (video frames) without
    the upstream's 4K cosine-blend tile engine. Upstream's tiling is
    redundant when the chained VAE (Wan 2.2, Hunyuan, CogVideoX, LTX, etc.)
    already handles temporal stitching internally.
  - All exposed node IDs are namespaced with the suffix `__koolook_v2_3_3`
    to avoid collisions with installed copies of upstream Radiance v2.3.3.
- **Why changed:** Upstream's `RadianceVAE4KEncode` errored with
  *"size of tensor a (192) must match the size of tensor b (132) at
  non-singleton dimension 4"* when used in Wan 2.2 video workflows — the
  4K tile-blender's spatial alignment doesn't agree with the video VAE's
  internal temporal-aware encoding. The Koolook version skips that
  pipeline and routes color-prepped frames directly to `vae.encode()`.
- **Last reviewed:** 2026-05-03

## Notes

- Keep this file updated when syncing upstream changes.
- If a node is heavily modified, document behavior differences clearly for users.
- For external fork variants, track exact upstream tag and pinned commit in
  [`forks_manifest.yaml`](forks_manifest.yaml).
- Freeze baseline first, then namespace node IDs before introducing newer
  upstream variants.
- Keep paths relative for portability (`../ComfyUI-Forks` by default).
- Keep only wrappers/manifests in MAIN; do not vendor large third-party trees here.
- Keep modified node source files in MAIN when you want full GitHub tracking
  and one-repo commits.
- **Run [`license-pre-check`](../.claude/skills/license-pre-check/) before
  adding any new third-party fork** — it catches license incompatibilities
  *before* you start copying code.
