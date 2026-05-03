# Maintainer docs

Project-internal procedures. If you're cutting a release, forking a node,
hitting the Comfy Registry API, or debugging CI — read these first.

These are docs for **humans** doing the work. The corresponding **agent**
workflows for Claude Code live in [`../../.claude/skills/`](../../.claude/skills/)
and should be kept in sync with these prose docs.

## Contents

| File | What it covers |
|---|---|
| [`releasing.md`](releasing.md) | Canonical step-by-step release procedure: branch → PR → publish-validate → squash-merge → tag → GitHub Release. Anti-patterns to avoid. The per-release tickbox mirror is [`.github/ISSUE_TEMPLATE/release_checklist.md`](../../.github/ISSUE_TEMPLATE/release_checklist.md). |
| [`registry-api.md`](registry-api.md) | The Comfy Registry's undocumented version-management endpoints (deprecate, delete, query) reverse-engineered while cleaning up v0.1.0–0.1.5. Includes ready-to-paste curl commands and an optional GitHub Actions workflow YAML for codified version management. |
| [`node-versioning.md`](node-versioning.md) | Hard-won rules for changing `INPUT_TYPES` / `RETURN_TYPES` / class names without breaking saved user workflows. The five rules + the suffix-version pattern (`MyNode` → `MyNodeV2`) + the alias-then-deprecate migration path. |

## Companion agent skills

| Skill (in `.claude/skills/`) | Mirrors this doc |
|---|---|
| `license-pre-check` | Runs *before* any third-party fork; refuses to proceed on incompatible licenses. The principles also apply when a human is doing it manually. |
| `add-external-fork` | The `forks/` workflow — clone outside MAIN, register in `forks_manifest.yaml`, namespace the IDs. Phase 0 calls `license-pre-check`. |
