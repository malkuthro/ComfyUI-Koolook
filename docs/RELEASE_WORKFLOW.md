# Release Workflow (Stable + Dev)

This project supports production users, so releases should be predictable.

## Branches

- `main` → active development
- `release/stable` → tested commits only (optional but recommended)

If you do not want two long-lived branches yet, keep using `main` and only publish tested tags.

## Versioning (SemVer)

Use tags in the format `vMAJOR.MINOR.PATCH`:

- `v1.0.0` = first stable release
- `v1.0.1` = bugfixes only
- `v1.1.0` = new backward-compatible features
- `v2.0.0` = breaking changes

## Minimal Release Checklist

1. Confirm node load on a clean ComfyUI install
2. Confirm key workflows still run
3. Update `CHANGELOG.md`
4. Create and push annotated tag:

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

5. Publish GitHub release notes (what changed, tested matrix, known issues)

## Production Install Recommendation

Users in production should install by tag (not moving branch heads).

Example:

```bash
git clone --branch v1.0.0 https://github.com/malkuthro/ComfyUI-Koolook.git
```

Or in ComfyUI Manager, pin to a known stable tag/commit.
