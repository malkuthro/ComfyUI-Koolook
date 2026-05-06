# Maintainer docs

Project-internal procedures. If you're cutting a release, forking a node,
hitting the Comfy Registry API, or debugging CI — read these first.

These are docs for **humans** doing the work. The corresponding **agent**
workflows for Claude Code live in [`../../.claude/skills/`](../../.claude/skills/)
and should be kept in sync with these prose docs.

## Contents

| File | What it covers |
|---|---|
| [`conventions.md`](conventions.md) | Cross-cutting project conventions that don't fit any single topic doc — currently the Mermaid-for-diagrams rule. Add new cross-cutting rules here as they accrete. |
| [`releasing.md`](releasing.md) | Canonical step-by-step release procedure: branch → PR → publish-validate → squash-merge → tag → GitHub Release. Anti-patterns to avoid. The per-release tickbox mirror is [`.github/ISSUE_TEMPLATE/release_checklist.md`](../../.github/ISSUE_TEMPLATE/release_checklist.md). |
| [`registry-api.md`](registry-api.md) | The Comfy Registry's undocumented version-management endpoints (deprecate, delete, query) reverse-engineered while cleaning up v0.1.0–0.1.5. Includes ready-to-paste curl commands and an optional GitHub Actions workflow YAML for codified version management. |
| [`node-versioning.md`](node-versioning.md) | Hard-won rules for changing `INPUT_TYPES` / `RETURN_TYPES` / class names without breaking saved user workflows. The five rules + the suffix-version pattern (`MyNode` → `MyNodeV2`) + the alias-then-deprecate migration path. |
| [`curated-sidebar.md`](curated-sidebar.md) | The Kforge Labs sidebar starter-preset workflow. How to customize the bundled `web/starter_preset.json` shipped to fresh users (Snapshot → Load → starter). Storage keys, library directory resolution, reset procedure. |
| [`workflows-sidebar.md`](workflows-sidebar.md) | The "Workflows" section of the same sidebar. Save canvas / save selection / directories / archive on collision / load with named tab. Storage shape, action semantics, reset procedure. |
| [`dev-iteration-loop.md`](dev-iteration-loop.md) | The fast iteration pattern we used to build the sidebar — the dev-sync loop, agent trigger phrases (`go`, `dev-sync`, paste-JSON), what survives across re-syncs vs what doesn't, push/publish gates. Read this when returning to UI work after a break. |

> **Read this first if you're confused about version numbers:**
> [`../reference/versioning.md`](../reference/versioning.md) explains the three independent version axes in this codebase (pack version, fork wrapper version, upstream pinned commit) and which skill/doc owns each one. Worth a one-time read before starting any version-bump work.

## Companion agent skills

| Skill (in `.claude/skills/`) | Mirrors this doc |
|---|---|
| `license-pre-check` | Runs *before* any third-party fork; refuses to proceed on incompatible licenses. The principles also apply when a human is doing it manually. |
| `add-external-fork` | The `forks/` workflow — clone outside MAIN, register in `forks_manifest.yaml`, namespace the IDs. Phase 0 calls `license-pre-check`. |
| `docs-sync` | Find-and-replace a version tag (or any string token) across docs, code constants, and manifests in one controlled pass. Used during fork wrapper version bumps. |
| `preflight-release` | Runs the four release-gate checks from `tools/preflight_release.py` — static AST node-ID extraction, VAE wrapper rank/type dispatch roundtrip, ComfyUI-Manager extension-node-map drift (the issue #44 class of bug), and workflow-fixture stability. Run before opening any release PR — `releasing.md` step 1. |
| `menu-comfy` | Lists all project skills with one-liners — quick inventory when you can't remember what's available. Distinct from the global `/menu` skill so the two don't collide. |
