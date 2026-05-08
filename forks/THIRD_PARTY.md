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
  - The two currently-exposed node IDs (`Easy_hdr_VAE_encode`,
    `Easy_hdr_VAE_decode`) are Koolook-original names with no upstream
    counterpart, so they're exposed verbatim via the `SKIP_VERSION_SUFFIX`
    set in [`versions/v2_3_3/__init__.py`](radiance_koolook/versions/v2_3_3/__init__.py).
    The wrapper's `__koolook_v2_3_3` namespace suffix is reserved for
    future ports of upstream-named classes (e.g. `RadianceVAE4KEncode`),
    where the suffix would prevent collisions with an installed copy of
    upstream Radiance v2.3.3. See
    [`docs/reference/versioning.md`](../docs/reference/versioning.md) for
    the in-place vs. versioned-coexistence patterns.
- **Why changed:** Upstream's `RadianceVAE4KEncode` errored with
  *"size of tensor a (192) must match the size of tensor b (132) at
  non-singleton dimension 4"* when used in Wan 2.2 video workflows — the
  4K tile-blender's spatial alignment doesn't agree with the video VAE's
  internal temporal-aware encoding. The Koolook version skips that
  pipeline and routes color-prepped frames directly to `vae.encode()`.
- **Last reviewed:** 2026-05-03

## De-vendored upstream code (untracked in v0.1.4 / v0.1.5)

Six third-party trees were untracked from MAIN's git index in the
v0.1.4 → v0.1.5 registry-hygiene cleanup (`git rm -r --cached`, then
added to `.gitignore`). They had no runtime effect — nothing in MAIN
imports them — but the Comfy Registry's static scanner was picking up
`NODE_CLASS_MAPPINGS` from the vendored copies and counting them
against this pack (the misleading "44 nodes / 13 conflicts" badge in
ComfyUI-Manager). All six are listed here for the audit trail.

The corresponding files were either physically moved to the
maintainer's local `../ComfyUI-Forks-BK/` backup or remain on the
maintainer's local disk only (gitignored). None of them are part of
the published package — the early v0.1.0 / v0.1.1 / v0.1.3 push-to-main
publishes to the Comfy Registry all failed (verified via
`gh run list --workflow=publish.yml`), and the first successful
Registry publish was the `v0.1.4-prep` dispatch on 2026-05-03, by
which point all six trees had already been untracked. The historical
GitHub release source tarballs do still contain these files (git
history is permanent by design), but those are not on any user's
ComfyUI install path.

Each of the six is also registered in
[`forks_manifest.yaml`](forks_manifest.yaml) with `status: "removed"`,
`removed_in_release`, and best-effort `source_repo` / `local_paths` —
see the entries with the `_devendored` suffix.

Folder names below are reproduced verbatim from the v0.1.3 tree,
including any upstream-name typos and casing variants (e.g.
`Comfyui-Animatedfiff-evolved` and the lowercase `Comfyui-` prefixes
on two rows) — intentional historical record, not authoring
inconsistency.

| Former path in MAIN | Upstream | Notes |
|---|---|---|
| `upscaler_FIX/github_repos/ComfyUI-SuperUltimateVaceTools/` | third-party VACE-tools repo (upstream URL not pinned at vendor time) | 3 files: a `FIXED_code/nodes.py`, a `nodes_original.py` upstream copy, and a `link.txt` shortcut. License unverified. |
| `upscaler_FIX/github_repos/ComfyUI-multigpu/` | third-party MultiGPU node pack (upstream URL not pinned) | 3 files: a `FIXED_code/distorch_2.py` plus 2 upstream `_orig.py` copies. License unverified. |
| `upscaler_FIX/github_repos/debugg/` | n/a — local error dumps | 3 markdown files of debugging notes; no upstream. |
| `nuke_CAM_exporter/_Utils-CAM-track/Github-Repos/Comfyui-Animatedfiff-evolved/` | likely https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved (inferred from folder name) | `nodes_cameractrl.py` + `link.txt`. License unverified. |
| `nuke_CAM_exporter/_Utils-CAM-track/Github-Repos/Comfyui-kjnodes/` | https://github.com/kijai/ComfyUI-KJNodes | `nodes.py` + `link.txt`. License: GPL-3.0 (see KJNodes entry above for full attribution; this was a separate consumption point unrelated to the `EasyResize_Koolook` reimplementation). |
| `nuke_CAM_exporter/wrapper_FIX/` | https://github.com/kijai/ComfyUI-WanVideoWrapper | Contained both an upstream `fun_camera/nodes.py` copy and a Koolook-modified version exploring a token-count alignment fix between Wan 14B (21 tokens) and FunCamera (41 tokens). The work was incomplete (per the in-folder `todo_and_resume.md`) and the upstream pin was never recorded — if revisited, treat as a from-scratch port against current upstream. License: assume GPL-3.0 (consistent with kijai's other wrappers); verify before promoting to a tracked fork. |

If any of these is ever revived as a properly-tracked Koolook fork,
go through [`add-external-fork`](../.claude/skills/add-external-fork/)
to register it in [`forks_manifest.yaml`](forks_manifest.yaml) with a
real `pinned_commit` and a `license_verified_at` timestamp.

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
