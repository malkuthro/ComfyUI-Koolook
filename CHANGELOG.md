# Changelog

All notable changes to this project should be documented in this file.

The format is inspired by Keep a Changelog and SemVer.

## [Unreleased]

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
