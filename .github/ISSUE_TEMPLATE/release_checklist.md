---
name: Release checklist
about: Track stable release readiness
title: "release: vX.Y.Z"
labels: ["release"]
assignees: []
---

> Canonical procedure lives in [`docs/maintainers/releasing.md`](../../docs/maintainers/releasing.md). This is
> the per-release tickbox mirror — keep both in sync.

## Pre-release validation

- [ ] Node import/startup passes on clean ComfyUI
- [ ] Core workflows smoke-tested
- [ ] `forks/forks_manifest.yaml` reviewed/updated
- [ ] `CHANGELOG.md` updated with `## [X.Y.Z] - YYYY-MM-DD` entry
- [ ] `pyproject.toml` `version` bumped to `X.Y.Z`
- [ ] Release PR opened from `release/vX.Y.Z-prep` → `main`

## Merge + tag (the tag drives publish + release)

- [ ] Squash-merge the PR (`gh pr merge --squash --delete-branch`)
- [ ] `git fetch origin`
- [ ] Annotated tag at the squash-merge SHA on `main` (`git tag -a vX.Y.Z origin/main`)
- [ ] Push tag (`git push origin vX.Y.Z`) — fires `release.yml`
- [ ] `git describe --tags origin/main` returns `vX.Y.Z`
- [ ] `release.yml` run is green: registry publish + GitHub Release both succeeded
- [ ] `curl https://api.comfy.org/nodes/koolook/versions` lists `X.Y.Z`

## Post-release

- [ ] ComfyUI-Manager UI displays `X.Y.Z` (not `unknown`)
- [ ] Registry version flips from `Pending` to `Active`
- [ ] Announce stable version to users
- [ ] Track regressions/bug reports
