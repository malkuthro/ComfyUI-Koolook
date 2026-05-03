# Changelog

All notable changes to this project should be documented in this file.

The format is inspired by Keep a Changelog and SemVer.

## [Unreleased]

## [0.1.7] - 2026-05-03

### Changed
- All node display names now suffix `(Koolook)` so they surface together
  when users search "koolook" in the ComfyUI node-add menu. Previously
  only `KoolookLoadCameraPosesAbsolute` (and the v2.3.3 fork nodes via
  their version suffix) matched.
  - `Easy_Version` → `Easy Version (Koolook)`
  - `EasyAIPipeline` → `Easy AI Pipeline (Koolook)`
  - `easy_ImageBatch` → `Easy Image Batch (Koolook)`
  - `EasyWan22Prompt` → `Wan 2.2 Easy Prompt (Koolook)`
  - `Easy_Version` category moved from `VFX/Utils` to `Koolook/VFX`
    to match the rest of the pack.
- Class keys (`NODE_CLASS_MAPPINGS`) are unchanged — saved workflows keep
  loading and running unchanged.
- Carried in via PR #36, then bundled into this release alongside the
  documentation reorg below.

### Docs reorg (root cleanup)
- Moved out of repo root (preserving git history via `git mv`):
  - `Glossary.md` → [`docs/reference/glossary.md`](docs/reference/glossary.md)
  - `RELEASING.md` → [`docs/maintainers/releasing.md`](docs/maintainers/releasing.md)
- Root now keeps only the four files that *must* live there:
  `README.md` (GitHub repo home + Comfy Registry description),
  `LICENSE` (`pyproject.toml` reference + GitHub license badge),
  `CHANGELOG.md` (tooling convention),
  `CLAUDE.md` (Claude Code agent instructions).
- New `docs/` structure with three audience buckets:
  - [`docs/user_guide/`](docs/user_guide/) — end-user, per-node guides + screenshots (`img/`)
  - [`docs/reference/`](docs/reference/) — lookup material (glossary, node inventory)
  - [`docs/maintainers/`](docs/maintainers/) — project-internal procedures
- Each bucket gets a short index `README.md` so the structure is
  navigable from `docs/README.md` downwards without grep.

### Added (new maintainer docs)
- [`docs/maintainers/registry-api.md`](docs/maintainers/registry-api.md) —
  documents the **undocumented** Comfy Registry version-management API
  that we reverse-engineered while cleaning up v0.1.0–0.1.5
  (deprecate / undeprecate / yank endpoints, status enum values,
  auto-deprecation behavior, ready-to-paste curl recipes, and an
  optional `registry-mgmt.yml` GitHub Actions workflow for codified
  version management without ever exposing the token to a shell).
- [`docs/maintainers/node-versioning.md`](docs/maintainers/node-versioning.md) —
  codifies the rules from the v0.1.5 PR-review discussion for safely
  changing a node's `INPUT_TYPES` / `RETURN_TYPES` / class names without
  breaking saved user workflows. Five rules + the suffix-version pattern
  + the alias-then-deprecate migration path + concrete worked examples.
- Cross-references in `README.md`, `CLAUDE.md`,
  `.github/ISSUE_TEMPLATE/release_checklist.md`, and the
  `add-external-fork` skill all updated to point at the new paths.
- Imported the existing untracked `docs/` images that the maintainer
  had already started adding locally (node inventory screenshot under
  `reference/`, Easy Image Batch helper image under `user_guide/img/`).

## [0.1.6] - 2026-05-03

### Renamed (with back-compat alias)
- `EasyResize` is now exposed canonically as **`EasyResize_Koolook`**
  (display: `Easy Resize (Koolook)`) to resolve a node-ID collision with
  `ComfyUI-EasyFilePaths`, which also registers the bare name `EasyResize`.
  The old `EasyResize` ID is kept as a **deprecated alias** pointing at
  the same class, so saved workflows still load and run unchanged. The
  alias will be removed in a future major release once the deprecation
  has had time to propagate.

### Attribution
- Added a proper SPDX header + GPL-3.0 attribution block to
  `k_easy_resize.py`, crediting `kijai/ComfyUI-KJNodes` (`Resize Image V2`)
  as the original interface inspiration. The Koolook implementation is a
  fresh write that materially extended the surface (aspect-ratio parser,
  keep_proportion modes, mask + composed outputs, device selection,
  target/original W/H reporting, color-panel passthrough). KJ Nodes is
  GPL-3.0, same as our pack — no relicense required.
- `forks/THIRD_PARTY.md` and `forks/forks_manifest.yaml` upgraded the
  KJ Nodes entry from `license: "unknown"` /
  `sync_state: "needs-upstream-reference"` to verified GPL-3.0 with full
  per-feature change notes.

### Notes for users
- If a saved workflow uses the bare `EasyResize` ID, it still works but
  the node's display name now reads
  `Easy Resize (deprecated, use 'Easy Resize (Koolook)')` as a hint to
  swap. New workflows should pick `EasyResize_Koolook` from the node-add
  menu.

## [0.1.5] - 2026-05-03

### Removed (BREAKING for anyone using the v1_0_1 namespaced IDs)
- Dropped the entire `forks/radiance_koolook/versions/v1_0_1/` folder
  (~5,200 lines, 26 namespaced nodes). The wrappers were vestigial —
  Koolook authors never used them, no internal workflow referenced any
  `__koolook_v1_0_1` suffixed ID, and the VAE pair was already
  superseded by `Easy_hdr_VAE_encode` / `Easy_hdr_VAE_decode` in v2_3_3.
- IDs that no longer exist after this release:
  `ImageToFloat32__koolook_v1_0_1`, `Float32ColorCorrect__koolook_v1_0_1`,
  `HDRExpandDynamicRange__koolook_v1_0_1`, `HDRToneMap__koolook_v1_0_1`,
  `ColorSpaceConvert__koolook_v1_0_1`, `SaveImageEXR__koolook_v1_0_1`,
  `LoadImageEXR__koolook_v1_0_1`, `LoadImageEXRSequence__koolook_v1_0_1`,
  `SaveImage16bit__koolook_v1_0_1`, `HDRHistogram__koolook_v1_0_1`,
  `LogCurveEncode__koolook_v1_0_1`, `LogCurveDecode__koolook_v1_0_1`,
  `HDRExposureBlend__koolook_v1_0_1`,
  `HDRShadowHighlightRecovery__koolook_v1_0_1`,
  `OCIOColorTransform__koolook_v1_0_1`,
  `OCIOListColorspaces__koolook_v1_0_1`, `GPUTensorOps__koolook_v1_0_1`,
  `HDR360Generate__koolook_v1_0_1`, `SaveHDRI__koolook_v1_0_1`,
  `ACES2OutputTransform__koolook_v1_0_1`,
  `DaVinciWideGamut__koolook_v1_0_1`, `ARRIWideGamut4__koolook_v1_0_1`,
  `RadianceVAEEncode__koolook_v1_0_1`,
  `RadianceVAEDecode__koolook_v1_0_1`, `k_easy_OCIO_v101`
  (the short ID for `RadianceOCIOColorTransformV2`),
  `RadianceLogCurveDecode__koolook_v1_0_1`.
- Migration path: install upstream Radiance directly
  (https://github.com/fxtdstudios/radiance) for the HDR/EXR/OCIO
  functionality. Use `Easy_hdr_VAE_encode/decode` (already in v0.1.3+)
  for video VAE workflows.
- Source recoverable via `git checkout HEAD~1 -- forks/radiance_koolook/versions/v1_0_1/`
  if you ever realize you need any of those wrappers back.

### Added (registry hygiene from the original v0.1.4 plan)
- `.gitignore` now excludes `upscaler_FIX/` and `nuke_CAM_exporter/` —
  the maintainer's local dev workspaces accidentally committed in
  Dec 2025. These were never imported by the package's root
  `__init__.py`, so they had no runtime effect, but the Comfy Registry's
  static scanner picked up `NODE_CLASS_MAPPINGS` from vendored 3rd-party
  clones inside them and counted ~12 spurious nodes against this pack
  in ComfyUI-Manager (yielding the misleading "44 nodes / 13 conflicts"
  badge).

### Removed (registry hygiene)
- `git rm -r --cached upscaler_FIX nuke_CAM_exporter` — 70 files
  untracked from git, files stay on the maintainer's local disk for
  reference. ~3.7 MB of unrelated content out of the published archive.

### Net effect on Manager / registry
- Node count drops from 44 to **8** (6 root Koolook + 2 v2_3_3 VAE).
- Spurious "Conflict with `ComfyUI-SuperUltimateVaceTools`" warnings
  disappear.
- Published archive shrinks by ~9 MB total (3.7 MB dev workspaces +
  ~5.2 MB v1_0_1 fork code).

### Notes for the maintainer
- After merging this PR and `git pull`-ing main, your local working tree
  will lose the `upscaler_FIX/` and `nuke_CAM_exporter/` folders (git
  applies the deletion). Back up first or restore via
  `git checkout HEAD~1 -- ...`. The `upscaler_FIX/` folder has already
  been physically moved to `../ComfyUI-Forks-BK/`; the
  `_Utils-CAM-track/` subfolder of `nuke_CAM_exporter/` has been moved
  to `../ComfyUI-Tools-BK/nuke_CAM_exporter/`. The remainder of
  `nuke_CAM_exporter/` (your actual Nuke pipeline work) is still on
  disk in MAIN but no longer tracked by git.

## [0.1.4] - 2026-05-03 (test-published only, superseded by 0.1.5)

### Removed (registry hygiene)
- Untracked the maintainer's local dev workspaces from git: `upscaler_FIX/`
  and `nuke_CAM_exporter/`. These were never imported by the package's root
  `__init__.py`, so they had no effect on what ComfyUI loaded at runtime,
  but the Comfy Registry's static scanner picked up the `NODE_CLASS_MAPPINGS`
  dicts inside vendored 3rd-party clones and counted them as part of this
  pack — yielding a misleading "44 nodes / 13 conflicts" badge in
  ComfyUI-Manager (against `ComfyUI-SuperUltimateVaceTools`,
  `ComfyUI-multigpu`, etc.). Files removed from index via
  `git rm -r --cached` (still on the maintainer's local disk) and the two
  paths are now `.gitignore`d so they cannot leak again.
- Net effect: published archive shrinks by ~3.7 MB (70 files), Manager's
  node count drops from 44 to ~32 (the actual runtime registrations:
  6 root Koolook + ~26 namespaced fork variants), and the spurious
  conflict warnings disappear. Same class of issue as the v0.1.2 GPL-3.0
  relicense — vendored 3rd-party code in MAIN, this time without the
  license-compatibility risk because none of it was ever imported.

### Notes for the maintainer
- After merging this PR and `git pull`-ing main, your local working tree
  will lose the `upscaler_FIX/` and `nuke_CAM_exporter/` folders (git
  applies the deletion). If you want to keep working on those locally,
  back them up first (`cp -r upscaler_FIX nuke_CAM_exporter ~/backup/`)
  or restore from history afterwards
  (`git checkout HEAD~1 -- upscaler_FIX nuke_CAM_exporter`); they will
  then live as untracked files in your working tree, ignored by the new
  `.gitignore` rules.

## [0.1.3] - 2026-05-03

### Renamed
- The new v2_3_3 VAE nodes are now exposed as `Easy_hdr_VAE_encode` /
  `Easy_hdr_VAE_decode` (clean IDs and display names — no
  `__koolook_v2_3_3` suffix), to avoid visual collision with upstream
  Radiance v2.3.3's `RadianceVAEEncode` / `RadianceVAEDecode` aliases
  in the ComfyUI node-add search. The version is still tracked
  structurally (file lives in `versions/v2_3_3/`) and textually
  (file header + UPSTREAM_PIN.yaml + forks/THIRD_PARTY.md). Other
  Koolook nodes in the v2_3_3 set (none today, but possible in future)
  would still use the `__koolook_v2_3_3` suffix by default — opt out
  via the new `SKIP_VERSION_SUFFIX` set in `versions/v2_3_3/__init__.py`.
- 0.1.2 was test-published to the registry with the previous
  `RadianceVAEEncode__koolook_v2_3_3` IDs but never tagged or formally
  released; bumping to 0.1.3 publishes the renamed version cleanly.

## [0.1.2] - 2026-05-03 (test-published only, superseded by 0.1.3)

### License (BREAKING)
- **Relicensed entire package to GPL-3.0.** v0.1.0 and v0.1.1 shipped under
  a claimed MIT license while already incorporating GPL-3.0-derived code from
  [fxtdstudios/radiance](https://github.com/fxtdstudios/radiance) under
  `forks/radiance_koolook/`. GPL-3.0 §5(c) requires the entire combined work
  to be GPL-3.0; relicensing aligns the package with what we actually ship
  and matches the dominant license posture of the ComfyUI custom-node
  ecosystem. Downstream users incorporating, linking to, or deriving from
  ComfyUI-Koolook must now distribute under GPL-3.0 (or compatible).
- Added `LICENSE` file at repo root with the full GPL-3.0 text
  (`pyproject.toml` previously referenced a `LICENSE` file that did not
  exist on disk).
- README "License" section rewritten to declare GPL-3.0 and explain the
  §5(c) implication.

### Added
- **`forks/radiance_koolook/versions/v2_3_3/`** — slim, video-friendly
  re-implementation of `RadianceVAEEncode` / `RadianceVAEDecode` exposed
  under the namespace suffix `__koolook_v2_3_3`. Mirrors the *interface
  surface* of upstream `RadianceVAE4KEncode` / `RadianceVAE4KDecode`
  but skips the 4K cosine-blend tile engine, which conflicts with
  modern video VAEs (Wan 2.2, Hunyuan, CogVideoX, LTX) that already
  handle their own temporal/spatial stitching internally. Fixes the
  `"size of tensor a (192) must match the size of tensor b (132) at
  non-singleton dimension 4"` runtime error users hit when chaining
  upstream's VAE encoder into Wan 2.2 video workflows.
- `forks/THIRD_PARTY.md` — full attribution entries for the v1.0.1
  baseline (already in the repo) and the new v2.3.3 VAE subset, with
  per-class change notes.
- `.claude/skills/license-pre-check/` — blocking license-compatibility
  audit skill for Claude Code. Run **before** copying or porting any
  third-party code; refuses to proceed on incompatible combinations
  (e.g. GPL upstream into MIT downstream).
- `.claude/skills/add-external-fork/` — Claude Code port of the
  existing Cursor skill, with a mandatory Phase 0 ("run
  license-pre-check first") and a Phase 3c requirement to add GPL §5(a)
  modification headers on derived files.
- `RELEASING.md` — canonical, step-by-step release procedure (was previously
  ad-hoc; the gaps it closes are exactly what caused the `v0.1.0` orphan-tag
  and `CristianP` `PublisherId` incidents).
- README "Release & Stability" section now links `RELEASING.md` and the
  per-release checklist template.
- `CLAUDE.md` now references `RELEASING.md` for agents.

### Changed
- `forks/forks_manifest.yaml` — explicit `license: "GPL-3.0"` and
  `license_verified_at` fields on the existing radiance entry (was
  `to_verify_at_source_ref`); pre-registered the new v2.3.3 VAE entry
  with verified license metadata.
- `.github/ISSUE_TEMPLATE/release_checklist.md` rewritten to mirror
  `RELEASING.md`, including the registry-publisher validation step.

### Notes for users
- If your workflow currently references the bare-name `RadianceVAEEncode`
  or `RadianceVAEDecode` (which routes to upstream Radiance v2.3.3's
  `RadianceVAE4KEncode/Decode` via that package's alias), and you hit
  the 4D vs 5D tensor mismatch on video workflows, switch to the
  namespaced `RadianceVAEEncode__koolook_v2_3_3` /
  `RadianceVAEDecode__koolook_v2_3_3` (display name suffix
  *(Koolook v2.3.3)*).
- The existing `__koolook_v1_0_1` namespaced nodes are unchanged and
  remain available for backward compatibility with saved workflows.

## [0.1.1] - 2026-05-03

### Fixed
- Re-tag release at current `main` HEAD so `git describe --tags` resolves locally
  (the original `v0.1.0` tag points at an orphaned merge commit that is no longer
  in `main`'s ancestry, which caused ComfyUI-Manager to display the installed
  version as "unknown").
- Correct `[tool.comfy] PublisherId` from the placeholder `CristianP` to the
  real publisher `kforgelabs`, which was the actual cause of every
  `Publish to Comfy registry` workflow failure (registry returned a misleading
  `400 "Failed to validate token"` because the declared publisher did not exist).

### Changed
- Bumped `pyproject.toml` `version` to `0.1.1` to match the new tag.

### Chore
- `.gitignore`: fix syntax and exclude `__pycache__/`, `*.pyo`, `.DS_Store`
  (carried forward from commit `8fc28d1`).

## [0.1.0] - 2026-04-24

### Added
- Fork tracking moved to `forks/` with centralized workflow documentation.
- Versioned Radiance fork package layout under `forks/radiance_koolook/versions/v1_0_1`.
- `CLAUDE.md` and `Glossary.md` to keep workflow and naming conventions explicit.

### Changed
- Root node loader now imports the versioned Radiance fork entrypoint.
- Release/checklist/template references updated to current fork-based paths.
- Introduced compact node ID override for OCIO transform: `k_easy_OCIO_v101`.

### Removed
- Legacy `ACES_workflow/radiance` tree and duplicated tracking paths under `third_party/`.
- Deprecated docs/assets/workflow artifacts removed during repository cleanup.
