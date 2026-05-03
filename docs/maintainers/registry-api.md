# Comfy Registry — version-management API

**Status:** undocumented at [docs.comfy.org](https://docs.comfy.org/), but
the endpoints are stable and we've used them in anger. This doc captures
what we discovered while cleaning up `koolook` v0.1.0–0.1.5.

## TL;DR

You can deprecate, undeprecate, or yank a published version of your node
pack via authenticated HTTP requests against `api.comfy.org`. The same
`REGISTRY_ACCESS_TOKEN` that publishes nodes also authorizes management.
docs.comfy.org doesn't mention this, but `OPTIONS` probes confirm the
methods exist and return the expected results when used.

## Endpoints

| Method | Path | Purpose | Auth |
|---|---|---|---|
| `GET`     | `/nodes/<node>/versions`                     | List published versions for a node (public) | none |
| `GET`     | `/nodes/<node>/versions/<v>`                 | Inspect one version (public) | none |
| `OPTIONS` | `/publishers/<id>/nodes/<node>/versions/<v>` | Discover allowed methods on the management endpoint | none |
| `PUT`     | `/publishers/<id>/nodes/<node>/versions/<v>` | Update fields on a version (currently `deprecated: bool`) | Bearer token |
| `DELETE`  | `/publishers/<id>/nodes/<node>/versions/<v>` | Yank a version (status flips to `NodeVersionStatusDeleted`) | Bearer token |

For `koolook`, that's:

- `<id>` = `kforgelabs`
- `<node>` = `koolook`

## Auth

Use the same personal access token that publishes nodes. We store it as a
GitHub Secret named `REGISTRY_ACCESS_TOKEN` in this repo. Header format:

```
Authorization: Bearer $REGISTRY_ACCESS_TOKEN
```

If you don't have the token locally and don't want to expose it to your
shell history, the cleanest path is the GitHub Actions workflow at the
bottom of this doc.

## Status enum values observed

| Value | Meaning |
|---|---|
| `NodeVersionStatusPending` | In moderation queue. Not yet installable, not yet listed publicly. |
| `NodeVersionStatusActive` | Live and installable via Manager / `comfy node install`. |
| `NodeVersionStatusFlagged` | Flagged by automated scanner. May still be visible but generally hidden from normal install flows. |
| `NodeVersionStatusDeleted` | Yanked. No longer installable. The historical record (download URL, metadata) often still resolves on `GET` for archival purposes. |

## Auto-deprecation observed

When a new version of a node publishes, **prior versions automatically get
`deprecated: true`** with no explicit action from the publisher. We saw
this happen across `koolook` 0.1.2–0.1.4 the moment 0.1.5 went up. So you
generally don't need to deprecate by hand — just publishing the next
version handles it.

You'd only need to manually deprecate or yank if:

- A version shipped with content that should never be installed fresh
  (license violation, security issue, broken metadata) and you want to
  pre-empt the auto-deprecation by yanking before the next release ships.
- You're cleaning up a long tail of old test-published versions.

## Recipes

### Inspect a version

```bash
curl -sS https://api.comfy.org/nodes/koolook/versions/0.1.5 | jq .
```

### Discover allowed methods

```bash
curl -sS -i -X OPTIONS \
  https://api.comfy.org/publishers/kforgelabs/nodes/koolook/versions/0.1.5 \
  | grep -i '^allow:'
# Expected: allow: OPTIONS, DELETE, PUT
```

### Deprecate a version

```bash
curl -sS -X PUT \
  -H "Authorization: Bearer $REGISTRY_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"deprecated": true}' \
  https://api.comfy.org/publishers/kforgelabs/nodes/koolook/versions/0.1.4
```

### Undeprecate (rare — only if you mistakenly deprecated something)

```bash
curl -sS -X PUT \
  -H "Authorization: Bearer $REGISTRY_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"deprecated": false}' \
  https://api.comfy.org/publishers/kforgelabs/nodes/koolook/versions/0.1.4
```

### Yank (delete) a version

⚠️ **Irreversible.** Verify the version first; verify it's not
the only published version (you don't want to leave the node with no
installable versions); verify nothing critical depends on it being
present.

```bash
curl -sS -X DELETE \
  -H "Authorization: Bearer $REGISTRY_ACCESS_TOKEN" \
  https://api.comfy.org/publishers/kforgelabs/nodes/koolook/versions/0.1.0
```

## Optional: a GitHub Actions workflow that keeps the token off your shell

Drop this into `.github/workflows/registry-mgmt.yml` if you want a
codified, repeatable way to deprecate/yank versions without ever pasting
the token into a terminal. Trigger via
`gh workflow run registry-mgmt.yml -f version=0.1.4 -f action=deprecate`
(or via the Actions UI).

```yaml
name: Comfy Registry — version management
on:
  workflow_dispatch:
    inputs:
      version:
        description: "Version to act on (e.g. 0.1.4)"
        required: true
        type: string
      action:
        description: "Action"
        required: true
        type: choice
        options: [deprecate, undeprecate, delete]

permissions: {}

jobs:
  manage:
    if: ${{ github.repository_owner == 'malkuthro' }}
    runs-on: ubuntu-latest
    steps:
      - name: Show current state
        run: |
          curl -sS \
            "https://api.comfy.org/nodes/koolook/versions/${{ inputs.version }}" \
            | jq .
      - name: Execute
        env:
          TOKEN: ${{ secrets.REGISTRY_ACCESS_TOKEN }}
          BASE: "https://api.comfy.org/publishers/kforgelabs/nodes/koolook/versions/${{ inputs.version }}"
        run: |
          case "${{ inputs.action }}" in
            deprecate)   BODY='{"deprecated": true}'  METHOD=PUT ;;
            undeprecate) BODY='{"deprecated": false}' METHOD=PUT ;;
            delete)      BODY=''                       METHOD=DELETE ;;
          esac
          if [ -n "$BODY" ]; then
            curl -sS -X "$METHOD" \
              -H "Authorization: Bearer $TOKEN" \
              -H "Content-Type: application/json" \
              -d "$BODY" "$BASE"
          else
            curl -sS -X "$METHOD" \
              -H "Authorization: Bearer $TOKEN" "$BASE"
          fi
      - name: Confirm new state
        run: |
          curl -sS \
            "https://api.comfy.org/nodes/koolook/versions/${{ inputs.version }}" \
            | jq .
```

This workflow file is **not committed today** — auto-deprecation has been
sufficient for our cleanup needs. Add it if and when manual yanks become
routine.

## What docs.comfy.org *does* document

For comparison, [the official publishing doc](https://docs.comfy.org/registry/publishing)
covers:

- Creating a publisher account and personal access token
- The required `pyproject.toml` `[project]` and `[tool.comfy]` fields
- The standard `publish.yml` GitHub Actions workflow that runs on push to `main`
- A note about the Windows CTRL+V `\x16` trap when pasting tokens

What it doesn't cover (and why this doc exists): version-management
endpoints, status enum values, auto-deprecation behavior, the OPTIONS
probe technique for discovering undocumented endpoints, programmatic
yank.

## Caveats

- These endpoints are undocumented. Comfy could change or remove them
  without warning. Re-probe with `OPTIONS` if a recipe stops working.
- The exact `PUT` body schema is empirically `{"deprecated": bool}` —
  there may be other writable fields we haven't discovered. Probe with a
  GET first to see the full record shape.
- "Auto-deprecation when a newer version publishes" is observed
  behavior, not documented promise. Don't rely on it for security-
  critical situations — yank explicitly if a version must not be
  installable.
