# Repo Audit — 2026-02-14

## Summary

The repository is functional but currently mixes multiple concerns (custom nodes, workflow assets, local environment artifacts, and experiments). This increases maintenance risk for production users.

## Findings

1. **Mixed-purpose layout**
   - Root contains node source, workflows, utility projects, docs, and assets.
   - Recommendation: gradually separate into clear areas (e.g. `nodes/`, `workflows/`, `runpod/`, `docs/`).

2. **Environment artifact present (`.venv/`)**
   - `.venv` is ignored by `.gitignore`, but still exists locally.
   - Recommendation: keep untracked; never commit virtual env artifacts.

3. **Release policy not explicit**
   - No formal SemVer/tag workflow documented.
   - Addressed by adding `docs/RELEASE_WORKFLOW.md` and `CHANGELOG.md`.

4. **Third-party modifications not centrally tracked**
   - Some code is derived/modified from external repos.
   - Addressed by adding `third_party/THIRD_PARTY.md` and `third_party/forks_manifest.yaml`.

## Immediate Next Steps

1. Fill `third_party/forks_manifest.yaml` for each external-derived node.
2. Publish first stable tag after verification.
3. Add CI release checks later (optional phase).
