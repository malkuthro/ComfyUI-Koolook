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

### Kosinkadink/ComfyUI-VideoHelperSuite — VideoCombine subclass (runtime composition)

- **Name:** ComfyUI-VideoHelperSuite (Kosinkadink)
- **Upstream repo URL:** https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
- **License:** GPL-3.0 (verified 2026-05-16 via raw `LICENSE` fetch)
- **Local path(s):** [`k_video_combine.py`](../k_video_combine.py)
  — exposed as the `Easy_VideoCombine` ComfyUI node ID (display name
  *"Easy Video Combine (Koolook)"*, category `Koolook/Video`).
- **What was incorporated (not copied):** A ~60-line subclass of
  `videohelpersuite.nodes.VideoCombine`. No VHS source code is copied
  into MAIN — the relationship is purely a Python runtime subclass that
  imports VHS at module load and delegates almost the entire body of
  `combine_video()` back to the parent class via `super()`. The only
  Koolook-side logic is:
  - An additional `create_path_if_missing` BOOLEAN input (optional,
    default `False`).
  - An `os.path.isabs(filename_prefix)` discriminator that bypasses
    ComfyUI's output-directory sandbox when the user types an absolute
    path.
  - A scoped monkey-patch of `folder_paths.get_save_image_path` for the
    duration of one `combine_video()` call so upstream's encoder /
    audio mux / batch manager run unchanged but write into the
    absolute target directory.
- **Pattern inspiration (separate from the runtime dependency on VHS):**
  The `os.path.isabs(filename_prefix)` overload-the-existing-field
  pattern is borrowed from
  [`spacepxl/ComfyUI-HQ-Image-Save`](https://github.com/spacepxl/ComfyUI-HQ-Image-Save)'s
  `SaveEXR` node (MIT, copyright 2023 spacepxl). MIT is GPL-3.0
  compatible. No spacepxl source code is copied — only the pattern.
  License verified 2026-05-16 via raw `LICENSE` fetch.
- **Why a subclass rather than a fork:** Patching the upstream VHS
  install on every workstation is fragile (ComfyUI-Manager auto-updates
  blow patches away); a global monkey-patch at Koolook import time
  changes behavior for every VHS node whether the user asked for it or
  not. A separate node opted into by placing it on the canvas is the
  additive, neighborly approach.
- **Runtime dependency:** Requires
  `Kosinkadink/ComfyUI-VideoHelperSuite` to be installed alongside
  Koolook. If VHS isn't importable at module load, `k_video_combine.py`
  prints a one-line `[Koolook] Easy_VideoCombine skipped: …` notice
  and returns empty NODE_CLASS_MAPPINGS — the rest of Koolook is
  unaffected.
- **Upstream PR plan:** the same `isabs` discrimination is a candidate
  to upstream into VHS proper. If/when accepted, `Easy_VideoCombine`
  can be deprecated in favour of `VHS_VideoCombine` directly.
- **Last reviewed:** 2026-05-16

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

### WhatDreamsCost/WhatDreamsCost-ComfyUI — v1.3.2 LTX Director subset (Koolook fork)

- **Name:** WhatDreamsCost-ComfyUI
- **Upstream repo URL:** https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI
- **Upstream commit/tag used:** `e81223a2add687555a6371e311b325880bf7c3c9`
  (v1.3.2; the latest README-only update on the release commit chain).
- **License:** GPL-3.0 (verified 2026-05-28 from the upstream `LICENSE`
  file — *GNU General Public License Version 3, 29 June 2007*).
- **Local path(s):** [`forks/whatdreamscost_koolook/versions/v1_3_2/`](whatdreamscost_koolook/versions/v1_3_2/)
- **What changed locally:**
  - **`ltx_director.py` (modified).** The `LTXDirector` node gains a new
    optional `relay_overrides` multiline-string input. The maintainer pastes
    a JSON dict of Prompt-Relay knobs (`video_strength`,
    `video_window_scale`, `audio_strength`, `audio_window_scale`,
    `audio_epsilon`) directly into the canvas widget; underscore-prefixed
    keys are ignored so the field can carry inline JSON comments. The
    parsed overrides flow through to `build_segments` per render — no
    disk file, no env var, the values live entirely inside the workflow
    JSON. Empty field → upstream Prompt-Relay defaults.
  - **`prompt_relay.py` (modified).** `build_segments` now computes σ
    per-segment using the Prompt-Relay paper formula
    `σ = (L − w_eff) / (2 · √ln(1/ε))` instead of the upstream's
    length-independent `σ = 1/ln(1/ε)` (≈ 0.1448 at ε=0.001). The new
    formula calibrates the penalty so it hits threshold ε exactly at
    the segment boundary regardless of segment length. A
    `SIGMA_FALLBACK = 0.1448` preserves the prior constant for the
    degenerate `L ≤ w_eff` corner. Per-segment σ is logged.
  - **`patches.py` (verbatim vendored).** `ltx_director.py` imports
    `detect_model_type` and `apply_patches` from this file. Carried
    along unmodified so the fork is self-contained — same precedent as
    `forks/radiance_koolook/versions/v2_3_3/color_helpers.py`.
- **What was *not* forked:** The companion `LTXDirectorGuide` node, plus
  `LTXKeyframer`, `MultiImageLoader`, `LTXSequencer`,
  `SpeechLengthCalculator`, `LoadAudioUI`, and `LoadVideoUI`. These are
  unmodified upstream nodes; users need an installed copy of upstream
  WhatDreamsCost-ComfyUI to use them. The Koolook variant of LTXDirector
  is namespaced (`LTXDirector__koolook_v1_3_2`) so the two coexist in
  the node picker.
- **Why namespaced (`__koolook_v1_3_2`):** per the convention in
  `forks/README.md`, ports of upstream-named classes carry the version
  suffix to avoid colliding with an installed copy of the same upstream
  package. Graphs wiring the Koolook variant pick
  `LTX Director (Koolook v1.3.2)` from the picker; the unsuffixed
  `LTX Director` continues to refer to the user's upstream install.
- **Why a partial fork rather than a subclass:** the two upstream
  modifications cross multiple call sites inside the `LTXDirector` body
  and inside the `build_segments` function in `prompt_relay.py`. A
  subclass override would have to either reach into upstream internals
  (brittle) or duplicate the bodies (more code than the verbatim fork).
  The partial fork was the smaller surface.
- **Maintenance loop:** the audio-lipsync iteration that drives changes
  here lives at
  [`../docs/automations/LTX-2.3/audio-lipsync/`](../docs/automations/LTX-2.3/audio-lipsync/).
  Patches were originally synced into the user's upstream install via
  `scripts/sync_investigation_patches.py`; with v1.3.2 promoted to a
  fork, the iteration uses the standard
  [`../scripts/sync_to_dev.py`](../scripts/sync_to_dev.py) flow and the
  workflow JSON references the Koolook-suffixed node ID.
- **Last reviewed:** 2026-05-28

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
