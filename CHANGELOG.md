# Changelog

All notable changes to this project should be documented in this file.

The format is inspired by Keep a Changelog and SemVer.

## [Unreleased]

### Added
- `RELEASING.md` — canonical, step-by-step release procedure (was previously
  ad-hoc; the gaps it closes are exactly what caused the `v0.1.0` orphan-tag
  and `CristianP` `PublisherId` incidents).
- README "Release & Stability" section now links `RELEASING.md` and the
  per-release checklist template.
- `CLAUDE.md` now references `RELEASING.md` for agents.

### Changed
- `.github/ISSUE_TEMPLATE/release_checklist.md` rewritten to mirror
  `RELEASING.md`, including the registry-publisher validation step.

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
