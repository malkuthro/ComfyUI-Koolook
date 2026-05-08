# Forks Workflow

This folder is the single source for all fork tracking metadata inside MAIN.

## Purpose

- Keep modified forked code in MAIN under versioned paths (for GitHub tracking).
- Keep raw external checkouts outside MAIN under `../ComfyUI-Forks` (for upstream reference/sync).
- Keep all fork metadata in one place:
  - `forks/forks_manifest.yaml`
  - `forks/THIRD_PARTY.md`

## Folder Layout in MAIN

- `forks/radiance_koolook/__init__.py` -> package entrypoint used by root `__init__.py`
- `forks/radiance_koolook/versions/v2_3_3/` -> local modified node sources (currently the only active version)
- `forks/radiance_koolook/versions/<version>/UPSTREAM_PIN.yaml` -> exact upstream parity pin

The earlier `v1_0_1` fork was removed in v0.1.5 — see [`THIRD_PARTY.md`](THIRD_PARTY.md)
for the rationale and recovery instructions.

## External Folder Naming (outside MAIN)

Root: `../ComfyUI-Forks`

- Pinned baseline checkout:
  - `radiance-v2.3.3-koolook`
- Rolling upstream checkout:
  - `radiance-main-upstream`

These names are referenced from `forks/forks_manifest.yaml`.

## Referencing System

For each fork entry in `forks/forks_manifest.yaml`:

- `source_repo` = canonical upstream URL
- `source_ref` = release/tag (or baseline descriptor)
- `pinned_commit` = exact reproducible commit hash
- `external_checkout.relative_path_from_forks_root` = pinned raw reference folder
- `external_checkout.upstream_tracking_path_from_forks_root` = rolling upstream folder
- `local_paths` = files tracked in MAIN

## Tagging Policy (from now on)

- MAIN repo tags use SemVer: `vMAJOR.MINOR.PATCH`.
- Local fork version folders use underscore variant of the same version:
  - `v2_3_3`, `v2_4_0`, etc.
- Node mapping namespace suffixes must match folder versions:
  - `__koolook_v2_3_3`, `__koolook_v2_4_0`, etc.
- Some node IDs opt out of the suffix via `SKIP_VERSION_SUFFIX` (see
  [`forks/radiance_koolook/versions/v2_3_3/__init__.py`](radiance_koolook/versions/v2_3_3/__init__.py))
  to keep stable user-facing IDs across upgrades.
- Keep `source_ref` + `pinned_commit` in `forks/forks_manifest.yaml` and `UPSTREAM_PIN.yaml`.

## Upgrade Flow Example (Radiance)

Hypothetical bump from the current `v2_3_3` to a future `v2_4_0`:

1. Compare current local version (`forks/radiance_koolook/versions/v2_3_3`) with pinned raw baseline (`../ComfyUI-Forks/radiance-v2.3.3-koolook`).
2. Pull latest upstream in `../ComfyUI-Forks/radiance-main-upstream`.
3. Create a new local version folder in MAIN, e.g. `forks/radiance_koolook/versions/v2_4_0`.
4. Port/validate modifications.
5. Namespace node IDs for that version (or keep them stable via `SKIP_VERSION_SUFFIX` when upstream IDs are unchanged).
6. Add new entry/update metadata in `forks/forks_manifest.yaml`.

For the underlying mental model — three independent version axes (pack
version / fork wrapper version / upstream pinned commit) — see
[`docs/reference/versioning.md`](../docs/reference/versioning.md).

## Rule of Thumb

- If it is modified production code -> keep in MAIN under `forks/.../versions/...`
- If it is raw upstream reference -> keep outside MAIN under `../ComfyUI-Forks`
