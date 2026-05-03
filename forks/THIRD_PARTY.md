# Third-Party Code & Attribution

This file tracks all third-party code incorporated into `ComfyUI-Koolook`,
the upstream license, and what (if anything) was modified locally.

The package as a whole is licensed **GPL-3.0** (see [`LICENSE`](../LICENSE))
because it incorporates GPL-3.0 code from upstream Radiance per the entries
below — GPL-3.0 §5(c) requires the whole work to be GPL-3.0.

> **Note:** the v0.1.0–0.1.4 releases also incorporated a much larger
> Radiance v1.0 fork (~5,200 lines under
> `forks/radiance_koolook/versions/v1_0_1/`) which was removed in v0.1.5.
> The package's GPL-3.0 license persists from the v0.1.2 relicense and
> from the remaining v2_3_3 fork below; downgrading back to MIT would be
> confusing for downstream users and is not planned.

## Entries

### fxtdstudios/radiance — v1.0.1 baseline (REMOVED in v0.1.5)

- **Status:** **Removed** in ComfyUI-Koolook v0.1.5 (2026-05-03).
- **Name:** Radiance (FXTD Studios)
- **Upstream repo URL:** https://github.com/fxtdstudios/radiance
- **Upstream commit/tag that was used:** `f1b8ae330848fa08aba24c9d3e355cb432d3515b`
- **License:** GPL-3.0 (still applies to the historical commits in our git
  history that contain the derived code).
- **What was removed:** The entire `forks/radiance_koolook/versions/v1_0_1/`
  folder (~5,200 lines / 26 namespaced nodes). It wrapped 24 verbatim
  Radiance v1.0 classes plus 2 Koolook-modified ones
  (`RadianceOCIOColorTransformV2` with a multiframe + alpha-preserving
  variant; `LoadImageEXRSequence` as a Koolook-original sequence loader;
  `RadianceVAEEncode/Decode` as Koolook-original VAE wrappers).
- **Why removed:** The wrappers were vestigial — Koolook authors never
  used them, no internal workflow referenced the `__koolook_v1_0_1`
  namespaced IDs, and the VAE pair was superseded by `Easy_hdr_VAE_encode`
  / `Easy_hdr_VAE_decode` in the v2_3_3 fork. Users who want Radiance
  functionality should install upstream Radiance directly.
- **Recoverable via git history.** Last commit containing the v1_0_1 code:
  see CHANGELOG.md `[0.1.5]` for the SHA after the v0.1.5 release lands.

### kijai/ComfyUI-KJNodes — Resize Image V2 inspiration (Koolook reimplementation)

- **Name:** ComfyUI-KJNodes (Kijai)
- **Upstream repo URL:** https://github.com/kijai/ComfyUI-KJNodes
- **License:** GPL-3.0 (verified 2026-05-03 via raw `LICENSE` fetch)
- **Local path(s):** [`k_easy_resize.py`](../k_easy_resize.py)
  — exposed as the `EasyResize_Koolook` ComfyUI node ID (with `EasyResize`
  retained as a deprecated alias for workflow backward-compat).
- **What was inspired (not copied):** The general idea of a smarter
  resize node with aspect-ratio handling and stride-aware (divisible-by)
  sizing came from KJ Nodes' `Resize Image V2`. The Koolook implementation
  is a fresh write that materially expanded the surface:
  - Aspect-ratio string parser (`"16:9"`, `"9:16"`, etc.) with base-axis
    selection (Width/Height).
  - `keep_proportion` modes: stretch / letterbox / pillarbox.
  - Padding color + crop position controls for letterbox/pillarbox modes.
  - Per-call device selection (CPU / CUDA).
  - Mask + composed-image outputs in addition to the resized image.
  - Target W/H + original W/H/aspect-ratio reporting outputs.
  - Color-panel passthrough output.
- **Why renamed in v0.1.6:** The bare-name `EasyResize` collides with
  `ComfyUI-EasyFilePaths`, which also registers an `EasyResize` node ID.
  Bumping our canonical ID to `EasyResize_Koolook` removes the conflict
  while keeping the old ID as a deprecation alias so saved workflows
  still load. The KJ Nodes attribution stays in the file header and here.
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
