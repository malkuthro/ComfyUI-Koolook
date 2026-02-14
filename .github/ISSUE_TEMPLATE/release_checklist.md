---
name: Release checklist
about: Track stable release readiness
title: "release: vX.Y.Z"
labels: ["release"]
assignees: []
---

## Pre-release validation

- [ ] Node import/startup passes on clean ComfyUI
- [ ] Core workflows smoke-tested
- [ ] `third_party/forks_manifest.yaml` reviewed/updated
- [ ] `CHANGELOG.md` updated
- [ ] Release notes drafted (`docs/RELEASE_NOTES_TEMPLATE.md`)

## Publish

- [ ] Create annotated tag (`vX.Y.Z`)
- [ ] Push tag
- [ ] Publish GitHub release with notes

## Post-release

- [ ] Announce stable version to users
- [ ] Track regressions/bug reports
