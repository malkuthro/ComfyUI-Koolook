# Changelog

All notable changes to this project should be documented in this file.

The format is inspired by Keep a Changelog and SemVer.

## [Unreleased]

### Internal (sidebar tidiness pass — no behavior change)
- **`makeToolbarButton({iconClass, title, onClick})`** factory in
  `web/sidebar/tree.js` replaces the four near-identical button-construction
  blocks in `renderPanel` (export, new-dir, save-canvas, save-selection).
  Each block was 5–7 lines of `createElement` / `className` / `innerHTML`
  / `title` / `addEventListener`; calls collapse to 4-line factory invocations.
  No behavior change. Closes the "extract `makeToolbarButton`" item on #47.
- **Dropped duplicate `.koolook-export-btn` CSS class.** It was identical to
  `.koolook-icon-btn` minus the `:disabled` state. The export button now
  uses only `.koolook-icon-btn` via `makeToolbarButton`. Closes the
  "drop duplicate CSS class" item on #47.
- **`buildFolder({path})` default removed.** `path` is now a required
  parameter; the previous `path = null` default was unreachable since every
  caller routes through `makeSectionCtx`, which always supplies a non-empty
  section-prefixed string. The runtime `if (path && …)` guard inside
  `buildFolder` is also dropped. Closes the "remove unreachable default"
  item on #47.
- **`workflowsCache` mutator invariants documented inline** above the public
  mutator block in `web/sidebar/workflows_store.js`. Four rules a future
  contributor needs before adding a new mutator: pair every mutate with a
  commit (or use `persistMutation`), return `false` for no-op vs. truthy for
  success, mutate in place, and never replace `workflowsCache` outside the
  seed/load paths. Closes the "document `workflowsCache` invariants" item
  on #47.

### Added
- **Recursive subdirectories under workflow directories.** Right-click any
  directory in the Workflows tree → "Create subdirectory…" to nest folders
  to arbitrary depth (e.g. `UP-scale / Type-A / Sharp`). Each nested
  directory behaves like a top-level one: it can hold workflows + an
  Archive subfolder + further subdirectories. The save modal directory
  picker is **cascading** (multi-step): pick a parent, then a child
  appears, then a grandchild, etc. — each child level has a `(save in
  "<path>")` option to stop drilling at that depth. The right-click
  workflow "Move to…" submenu lists every other path in the tree.
- **Delete archive (N) on the synthetic Archive folder.** Right-click the
  Archive folder under any directory → bulk-removes every archived
  workflow at that level in one confirm. Active workflows in the same
  directory are untouched.
- **Drag-and-drop in the workflows tree** (Tier 1 — moves only, no
  reordering). Drag a workflow onto a directory to move it. Drag a
  workflow onto an Archive folder to archive it (cross-directory drops
  move + archive in one go). Drag a directory onto another directory to
  nest it as a child. Cycle prevention rejects dropping a dir into
  itself or any of its descendants. Sort within a level stays
  alphabetical. (Custom ordering would be Tier 2.)
- **Schema is now recursive** — every directory node has a `workflows`
  object AND a `directories` object. Existing v0.2 stores load fine:
  `normalizeWorkflowsStore` treats a missing `directories` as `{}` and
  the rest of the code assumes it always exists post-normalization.

### Changed
- **Sidebar tab renamed:** "Curated Nodes" → **"Kforge Labs"**. Tooltip
  also updated. The tab id (`koolook.curatedNodes`) and the
  `app.registerExtension` name are unchanged so existing per-user tab
  state (pinning, ordering) is preserved.
- **Save modal — `Base on existing` candidates now walk leaf-up to root
  and dedupe (deepest wins).** Saving into an empty subdirectory like
  `UP-scale / seedvr2` no longer disables the existing-name actions —
  the modal pulls active workflows from every ancestor. Ancestor
  entries are labeled `<name>  ·  in <path>` so the candidate's
  source directory is unambiguous. The selected destination path is
  whatever the cascade resolves to — the ancestor source only seeds
  the workflow name; archive semantics still apply at the destination.
- **Save modal — `Action` dropdown is now hidden (not just disabled)
  when no base candidate exists** anywhere in the destination's
  ancestry. Disabled `<option>` elements render too subtly across
  browsers; the only useful path in that case is "type a fresh name
  and save," which the Workflow Name field below already provides.
  The underlying value is pinned to `new` so submit takes the
  by-name path.
- The right-click canvas-node menu item is now **"Add to Kforge Labs"**
  (was "Add to Curated Sidebar").
- Directory header counts in the workflows tree now show the total
  workflows in the **whole subtree** (active + archived + descendants),
  not just direct children. A parent with empty direct workflows but
  populated subdirectories no longer shows "0".

### Internal
- Workflow operations are now path-addressed: every mutator and lookup
  takes a `string[]` path (e.g. `["UP-scale", "Type-A"]`). The store's
  internal API: `addDirectory(parentPath, name)`,
  `renameDirectory(parentPath, old, new)`, `deleteDirectory(parentPath, name)`,
  `saveWorkflowEntry(path, wfName, graph)`, `archive/unarchive/rename/deleteWorkflow(path, wfName)`,
  `moveWorkflow(srcPath, wfName, dstPath)`, plus new helpers
  `listAllDirectoryPaths()` and `pathsEqual(a, b)`.
- Reserved-name check: subdirectory names cannot be `Archive`
  (case-insensitive) at any non-root level — collides with the synthetic
  Archive folder rendered for archived workflows.

### Documentation
- Closed out issue #28 (de-vendor upstream code under `upscaler_FIX/`
  and `nuke_CAM_exporter/`) by adding the audit-trail layer that the
  v0.1.4 / v0.1.5 registry-hygiene cleanup deferred:
  - `forks/THIRD_PARTY.md` gained a "De-vendored upstream code" section
    listing all six untracked trees with former path, upstream (where
    pinned) and per-tree provenance notes. Preamble now also documents
    the publish-history finding that none of these files were ever in
    a successful Comfy Registry publish.
  - `forks/forks_manifest.yaml` gained six entries with the `_devendored`
    suffix and `status: "removed"`, populating `source_repo` /
    `source_ref` / `local_paths` / `removed_in_release` / `license` per
    issue #28's acceptance criterion (2). Best-effort where upstream
    URLs were not pinned at vendor time. Closes #28.

## [0.2.0] - 2026-05-04

### Removed (BREAKING for any saved workflow using `Easy_Version`)
- Dropped `Easy_Version` (and the `k_easy_version.py` source) — the
  whole node was just a one-liner that turned an integer N into the
  string `vNNN` (zero-padded). Maintainer concluded it was the first
  trivial node they ever wrote and saw no real value. Anyone who needs
  this exact behavior can either:
  - inline the format string in their workflow downstream of an INT
    primitive (`f"v{N:03d}"`), or
  - use any of the more general string-format nodes in the ecosystem
    (KJ Nodes, ComfyUI-Custom-Scripts, etc.).
- Saved workflows that reference the `Easy_Version` ID will fail to
  load — same migration as v0.1.5's `__koolook_v1_0_1` cleanup.

### Fixed (carried over from PR #39, [Unreleased] in 0.1.8)
- `Easy_hdr_VAE_encode` (Koolook v2.3.3) now wraps the encoded tensor in
  the standard ComfyUI `LATENT` dict (`{"samples": t}`) instead of
  returning the raw tensor. Wiring this node into KSampler previously
  crashed with `IndexError: too many indices for tensor of dimension 5`
  on Wan 2.2 video workflows, because KSampler does
  `latent["samples"]` and the raw 5-D tensor doesn't support string
  indexing. The decoder side was already correct.
- `Easy_hdr_VAE_encode` / `Easy_hdr_VAE_decode` (Koolook v2.3.3) now
  produce proper N-frame sequences end-to-end on Wan 2.2 / Hunyuan /
  CogVideoX / LTX video workflows. Two combined fixes:
  1. **Rank-aware dispatch.** 5-D `(B, F, H, W, C)` video tensors paired
     with a 2-D image VAE are iterated frame-by-frame and stacked along
     a temporal axis on encode (and per-frame decoded then concatenated
     to a `(B*F, H, W, C)` ComfyUI IMAGE batch on decode). 5-D tensors
     with a 3-D-native VAE — identified via `vae.latent_dim == 3`,
     which is the actual attribute ComfyUI sets on its VAE class for
     video VAEs — pass through unchanged.
  2. **5-D output normalization on decode.** 3-D-aware VAEs return a
     5-D `(B, F, H, W, C)` tensor from `vae.decode()`. Without an
     explicit reshape, downstream IMAGE-typed nodes (Get Image Size,
     SaveImage) misread that as a single 4-D image with `count=1` and
     `height=F` — exactly the symptom we were debugging
     (`1056×41 count=1` after a 41-frame Wan 2.2 sequence). The fix
     mirrors the stock ComfyUI VAEDecode node
     ([nodes.py:303-304](https://github.com/comfyanonymous/ComfyUI/blob/master/nodes.py#L303-L304)):
     `if img.ndim == 5: img = img.reshape(-1, H, W, C)` — the leading
     `(B, F, …)` dims collapse into the standard ComfyUI batch axis,
     so SaveImage now writes `_00001.png … _0004N.png` instead of one
     squashed frame.

  The new `path=image | video-iter | video-3d` field in `debug_info`
  surfaces which dispatch branch fired per run, for live verification.

### Added (Easy_hdr_VAE_encode / decode feature parity with upstream)
- Restored the cinema-grade color-management surface from upstream
  Radiance v2.3.3's `RadianceVAE4KEncode/Decode`, while keeping the
  slim no-tile-engine code path that lets these nodes work with
  Wan 2.2 video VAEs (the original motivation for the Koolook fork —
  upstream's 4K cosine-blend tile engine errors with `"size of tensor a
  (192) must match the size of tensor b (132) at non-singleton dimension 4"`
  on video VAEs because the tiler's spatial alignment fights the video
  VAE's internal temporal-aware encoding).
- **12 source / target color spaces** (was 4): Linear, ACEScg,
  **ACES 2065-1**, **Rec.2020 Linear**, sRGB, Raw, plus six cinema log
  curves — **ARRI LogC3** (EI-aware, default 800), **ARRI LogC4**,
  **Sony S-Log3**, **Panasonic V-Log**, **DaVinci Intermediate**,
  **RED Log3G10**. Round-trip math (linearize → encode) and matrix
  constants ported verbatim from upstream's `color_utils.py` with the
  bug-fix history preserved in the docstrings. Conversions are
  rank-agnostic and 4-channel-aware (alpha passes through untouched).
- **`Compress (Log)` HDR mode** — the upstream HDR-clamp-free pipeline.
  On encode, the linear-space tensor is re-encoded through a cinema log
  curve (matches `source_space` if it's a log space, else ARRI LogC4)
  and goes into the VAE without a hard clamp. On decode, the VAE output
  is run through `_soft_log_shoulder` (tanh rolloff at the per-profile
  knee instead of a hard clamp at 1.0) and `_denoise_log_highlights`
  (3×3 box blur in log space, ramped quadratically toward the highlight
  region) before log→linear conversion. Per-profile parameters
  (`LOG_PROFILE_HDR_PARAMS` table) tune knee / ceiling / denoise
  threshold / strength to each curve's slope at code 1.0 — RED Log3G10
  gets the most conservative settings, Sony S-Log3 / ARRI LogC4 the
  loosest. Pair `hdr_mode="Compress (Log)"` on both encode and decode
  for HDR-clean roundtrips; mismatched modes produce garbage by design.
- **`latent_sampling`** parameter (`sample` / `mean` / `mode`).
  `sample` is ComfyUI's default random posterior sample. `mean` and
  `mode` use the posterior mean for deterministic, lowest-noise
  encoding — best for img2img where minimum reconstruction noise
  matters. The mean/mode path replicates ComfyUI's preprocessing
  (BHWC → BCHW → [-1, 1] scaling) before reaching `first_stage_model`,
  with a graceful fallback chain that never crashes.
- **Real alpha output.** `Easy_hdr_VAE_encode` now returns
  `(LATENT samples, STRING debug_info, IMAGE alpha)`. With
  `alpha_handling="Preserve"` and a 4-channel input, the alpha channel
  is surfaced as a separate IMAGE for downstream re-compositing
  post-decode (VAEs don't encode alpha, so it's routed around them).
  With `alpha_handling="Ignore"` or a 3-channel input, alpha is a
  zeros tensor of compatible shape — downstream wiring never fails.
  The previous `alpha_handling` flag was a documented no-op.
- **Exposure now applied in linear space** for all source spaces. The
  previous behavior multiplied raw input bytes by `2^exposure` even
  when the input was sRGB-gamma or log-coded, which produced visually
  wrong results for non-linear sources. Linear sources are unaffected
  (linear × 2^stops is the correct semantic). Raw source still does a
  raw-domain multiplication so users who know what they're feeding the
  VAE keep their bytes intact.
- New helper module
  [`forks/radiance_koolook/versions/v2_3_3/color_helpers.py`](forks/radiance_koolook/versions/v2_3_3/color_helpers.py)
  contains all log curves, soft-shoulder / log-denoise helpers,
  `encode_with_sampling_mode`, color-space matrices, and dispatch
  tables. `nodes_vae.py` keeps the node wiring + sequence dispatch.

### Intentionally NOT ported (upstream-only features)
- 4K cosine-blend tile engine and its `tile_size` / `overlap` /
  `processing_mode` knobs — this is the broken plumbing that motivated
  the slim fork in the first place.
- `inverse_tonemap` / `target_stops` (SDR→HDR expansion).
- `.rhdr` sidecar export and `rhdr_precision` (Radiance Viewer-specific).
- `crop_padding` (only relevant when tiling).

These can be added back in future patches if needed; the slim wrapper
stays at ~600 lines of node wiring + ~450 lines of color helpers,
~40% the size of upstream's `vae.py` (2,638 lines).

### Added (carried over from PR #40, also unreleased on main)
- `scripts/sync_to_dev.py` — pure-stdlib helper that copies the curated
  runtime files (`__init__.py`, `config.json`, top-level `k_*.py`,
  `forks/`, `web/`) into a live ComfyUI `custom_nodes/<pack>/` folder
  for fast local iteration without bumping a version. Reads the target
  path from the new `KOLOOK_COMFYUI_DEV_PATH` env var (auto-loads `.env`
  from repo root); errors out cleanly when unset.
- `.env.example` — `KOLOOK_COMFYUI_DEV_PATH=` entry with comment.
- `CLAUDE.md` — new "`dev-sync`" trigger-phrase section so the agent
  knows to run `scripts/sync_to_dev.py` when the maintainer says
  "dev-sync" / "sync dev" / "copy those files" mid-session.

### Net effect
- Node count drops from 8 to **7**.
- `Koolook/Pipeline` subfolder now has only 1 entry (`Easy AI Pipeline`)
  instead of 2.
- Faster local dev loop: a one-line tweak no longer requires cutting a
  full release just to test it inside ComfyUI.

## [0.1.8] - 2026-05-03

### Changed (categories — affects ComfyUI node-add menu)
- `Koolook/VFX` is gone. Three more granular subfolders replace it:
  - **`Koolook/Pipeline`** — `Easy_Version`, `EasyAIPipeline` (workflow
    setup nodes)
  - **`Koolook/Image`** — `EasyResize_Koolook`, `easy_ImageBatch` (joins
    `EasyResize_Koolook`; `easy_ImageBatch` was previously in VFX)
  - **`Koolook/VAE`** — `Easy_hdr_VAE_encode`, `Easy_hdr_VAE_decode`
    (now have their own subfolder; previously also in VFX)
- Workflows continue to load and run unchanged — `CATEGORY` is purely a
  UI-organization hint and only affects the node-add menu hierarchy.

### Fixed (search discoverability)
- `Easy_hdr_VAE_encode` / `Easy_hdr_VAE_decode` display names are now
  **"Easy HDR VAE Encode (Koolook)"** / **"Easy HDR VAE Decode (Koolook)"**.
  Previously the display name equalled the node ID (`Easy_hdr_VAE_encode`
  with no "Koolook" string), so typing `koolook` in ComfyUI's node-add
  search filter excluded them — they appeared registered but invisible.
  Now they group with the rest of the pack under that filter.
- Class IDs (`Easy_hdr_VAE_encode`, `Easy_hdr_VAE_decode`) are unchanged —
  saved workflows that reference these IDs continue to load.

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
