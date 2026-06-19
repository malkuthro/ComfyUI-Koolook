# Releasing

Canonical procedure for cutting a stable release of `ComfyUI-Koolook`.

This document is the source of truth. The
[`release_checklist.md`](../../.github/ISSUE_TEMPLATE/release_checklist.md) issue
template mirrors these steps in tickbox form for each release.

## Conventions

- Repo tag format: `vMAJOR.MINOR.PATCH` (semver). Example: `v0.1.1`.
- Fork code paths use the underscore variant: `v2_3_3` (current Radiance fork; the historical `v1_0_1` fork was removed in v0.1.5 — see `forks/THIRD_PARTY.md`).
- Comfy node ID namespace suffixes match fork paths: `__koolook_v2_3_3`.
- Always **squash-merge** release PRs. A clean single commit per release
  keeps `git describe --tags` honest and prevents the orphan-tag class of
  bugs (a re-merged PR can leave a tag pointing at a commit no longer in
  `main`'s ancestry, which makes ComfyUI-Manager show the installed
  version as `unknown`).

## Procedure

### 1. Pre-release validation

- Node import + startup pass on a clean ComfyUI install.
- Core workflows smoke-tested.
- `forks/forks_manifest.yaml` reviewed if any fork was touched.

### 2. Open the release PR

From a fresh checkout of `main`:

```bash
git checkout -b release/vX.Y.Z-prep
# Edit pyproject.toml: version = "X.Y.Z"
# Edit CHANGELOG.md: add `## [X.Y.Z] - YYYY-MM-DD` entry above [0.1.1]
git commit -am "Release vX.Y.Z: <one-line summary>"
git push -u origin release/vX.Y.Z-prep
gh pr create --base main --head release/vX.Y.Z-prep \
  --title "Release vX.Y.Z" --body "<see template below>"
```

PR body template:

```markdown
## Summary
- Bump version to X.Y.Z.
- <other release-relevant changes>

## Test plan
- [ ] CI green on the release PR.
- [ ] After merge + tag push: the `release.yml` run is green (registry publish + GitHub Release).
- [ ] curl https://api.comfy.org/nodes/koolook/versions lists the new version.
- [ ] git describe --tags origin/main returns vX.Y.Z.
```

### 3. Squash-merge

Always squash-merge (see [Conventions](#conventions) for why), then refresh
the local remote-tracking ref so the tag lands on the right commit:

```bash
gh pr merge <PR#> --squash --delete-branch
git fetch origin
```

### 4. Tag to release — the automation does the rest

Pushing a semver tag at the squash-merge commit is the **single release
trigger**. The [`release.yml`](../../.github/workflows/release.yml) workflow
fires on the tag and, in order:

1. **Publishes to the Comfy Registry** (`publish-node-action`), after checking
   the tag matches `pyproject.toml` `version` and a `CHANGELOG.md` section
   exists for it.
2. **Creates the GitHub Release** from that `CHANGELOG.md` section — only if
   the publish succeeded.

The tag must point at the squash-merge commit on `main`, **not** the release
branch tip (a tag on an orphaned commit makes ComfyUI-Manager report the
installed version as `unknown`):

```bash
git tag -a vX.Y.Z origin/main -m "Release vX.Y.Z

<short release summary>"
git push origin vX.Y.Z              # fires release.yml
git describe --tags origin/main     # must return vX.Y.Z
```

Then watch the run:

```bash
gh run watch "$(gh run list --workflow=release.yml --limit 1 \
  --json databaseId --jq '.[0].databaseId')" --exit-status
```

Because the registry only seals a version on a **successful** publish, a
failed run is safe to retry: fix the cause (usually the token — see the table
below) and **re-run the failed job**. No version bump, no double-publish —
this removes the class of bug that used to force a `vX.Y.(Z+1)` bump mid-cut
whenever a pre-merge publish was re-dispatched.

| Error | Likely cause | Fix |
|---|---|---|
| `400 "Failed to validate token"` | `[tool.comfy] PublisherId` does not match a real publisher on the Comfy Registry, **or** the secret `REGISTRY_ACCESS_TOKEN` is empty / corrupted | Verify with `curl https://api.comfy.org/publishers/<id>`. On Windows, never CTRL+V the token — right-click paste only (CTRL+V can append `\x16`). Re-run the failed job. |
| `409` / duplicate version | The version is already sealed on the registry and cannot be re-published | Cut the next patch (`vX.Y.(Z+1)`): bump `pyproject.toml` + `CHANGELOG.md`, re-merge, re-tag. |
| Auth 401/403 | Secret missing | `gh secret set REGISTRY_ACCESS_TOKEN --repo malkuthro/ComfyUI-Koolook`, then re-run the failed job. |

> **Manual fallback.** [`publish.yml`](../../.github/workflows/publish.yml)
> (`workflow_dispatch`) still exists for a one-off re-publish against a chosen
> ref — e.g. recovering a registry-side hiccup without cutting a new tag. The
> normal path is the tag; reach for the dispatch only when you specifically
> need to publish *without* a tag. Do **not** dispatch it during a normal
> release — publishing the real version before the tag re-introduces the
> double-publish/version-bump trap the tag flow exists to remove.

### 5. Post-release verification

- ComfyUI-Manager displays version `X.Y.Z` (no longer `unknown`).
- `https://api.comfy.org/nodes/koolook/versions` lists the new version
  with `status: NodeVersionStatusActive` (Pending will flip to Active
  after registry moderation).
- Track regressions / install reports.

## Anti-patterns to avoid

- **Do not move existing tags.** If you need to re-tag a release, cut a
  new patch version. Anyone who already pinned to the old tag must keep
  the bytes they installed.
- **Do not tag before `main` is correct.** Push the tag only after the
  squash-merge lands and `pyproject.toml` + `CHANGELOG.md` match the tag.
  `release.yml` hard-fails the publish on a mismatch, but a stray tag on
  `main` is still cleanup you don't want.
- **Do not dispatch `publish.yml` during a normal release.** Publishing the
  real version before the tag seals it on the Registry, so the tag's own
  publish then `409`s and you're back to bumping the patch number. Let the
  tag be the only publish.
- **`Failed to validate token` usually isn't the token.** It almost always
  means `[tool.comfy] PublisherId` doesn't match a real publisher. Confirm
  with `curl https://api.comfy.org/publishers/<id>` before touching the
  secret, then re-run the failed job.
