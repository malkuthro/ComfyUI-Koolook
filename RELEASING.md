# Releasing

Canonical procedure for cutting a stable release of `ComfyUI-Koolook`.

This document is the source of truth. The
[`release_checklist.md`](.github/ISSUE_TEMPLATE/release_checklist.md) issue
template mirrors these steps in tickbox form for each release.

## Conventions

- Repo tag format: `vMAJOR.MINOR.PATCH` (semver). Example: `v0.1.1`.
- Fork code paths use the underscore variant: `v1_0_1`.
- Comfy node ID namespace suffixes match fork paths: `__koolook_v1_0_1`.
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
- [ ] Manual workflow_dispatch of publish.yml against this branch succeeds.
- [ ] curl https://api.comfy.org/nodes/koolook returns the new version.
- [ ] After merge: tag pushed, GitHub Release created, git describe --tags
      origin/main returns vX.Y.Z.
```

### 3. Validate the registry publish *before* merging

Trigger the publish workflow against the PR branch:

```bash
gh workflow run publish.yml --ref release/vX.Y.Z-prep
gh run watch <run-id> --exit-status
curl -sS https://api.comfy.org/nodes/koolook | jq '.publisher.id, .id'
```

Expectations:

- Workflow run status: success.
- `publisher.id` returns `kforgelabs` (matches `[tool.comfy] PublisherId`).
- `GET /nodes/koolook/versions` includes the new version.

If publish fails:

| Error | Likely cause | Fix |
|---|---|---|
| `400 "Failed to validate token"` | `[tool.comfy] PublisherId` does not match a real publisher on the Comfy Registry, **or** the secret `REGISTRY_ACCESS_TOKEN` is empty / corrupted | Verify with `curl https://api.comfy.org/publishers/<id>`. On Windows, never CTRL+V the token — right-click paste only (CTRL+V can append `\x16`). |
| `409` / duplicate version | The version already exists in the registry | Bump `pyproject.toml` to a new version, update `CHANGELOG.md`, force-push the PR branch, re-validate. |
| Auth 401/403 | Secret missing | Set `REGISTRY_ACCESS_TOKEN` in repo Settings → Secrets, or via `gh secret set REGISTRY_ACCESS_TOKEN --repo malkuthro/ComfyUI-Koolook`. |

> **Trade-off:** validating before merge means the auto-triggered publish
> on the merge commit will 409 (duplicate version), leaving a cosmetic
> red badge on the merge. This is intentional — better to catch a broken
> publish in PR review than after release.

### 4. Squash-merge

```bash
gh pr merge <PR#> --squash --delete-branch
```

### 5. Tag and push

The tag must point at the squash-merge commit on `main`, **not** at the
release branch tip:

```bash
git fetch origin
git tag -a vX.Y.Z origin/main -m "Release vX.Y.Z

<short release summary>"
git push origin vX.Y.Z
git describe --tags origin/main   # must return vX.Y.Z
```

### 6. Create the GitHub Release

Use the matching `CHANGELOG.md` section as the release notes:

```bash
awk '/^## \[X\.Y\.Z\]/{flag=1;next} /^## \[/{flag=0} flag' CHANGELOG.md \
  > /tmp/vXYZ-notes.md
gh release create vX.Y.Z --title "vX.Y.Z" \
  --notes-file /tmp/vXYZ-notes.md --verify-tag
```

### 7. Post-release verification

- ComfyUI-Manager displays version `X.Y.Z` (no longer `unknown`).
- `https://api.comfy.org/nodes/koolook/versions` lists the new version
  with `status: NodeVersionStatusActive` (Pending will flip to Active
  after registry moderation).
- Track regressions / install reports.

## Anti-patterns to avoid

- **Do not move existing tags.** If you need to re-tag a release, cut a
  new patch version. Anyone who already pinned to the old tag must keep
  the bytes they installed.
- **Do not skip the registry validation step.** Every prior failure of
  `publish.yml` was masked by the misleading `Failed to validate token`
  error. The registry check (`curl /publishers/<id>`) is the only way to
  catch a `PublisherId` mismatch before users do.
- **Do not commit a release directly to `main`.** Release PRs trigger the
  publish workflow on push to `main`; doing it via PR keeps that auto-run
  associated with a reviewable change.
- **Do not edit `pyproject.toml` for non-release reasons in a release PR.**
  The publish workflow triggers on any `pyproject.toml` change. Keeping
  release PRs scoped to version + changelog avoids surprise publishes.
