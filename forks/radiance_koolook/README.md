# Radiance Koolook

Versioned local fork of Radiance nodes.

- Package entrypoint: `forks/radiance_koolook/__init__.py`
- Version folders: `forks/radiance_koolook/versions/<version>/`
- Each version folder keeps:
  - modified node files tracked in MAIN
  - `UPSTREAM_PIN.yaml` for parity with pinned external raw checkout

Current active version:
- `v2_3_3` — see [`forks/THIRD_PARTY.md`](../THIRD_PARTY.md) for upstream
  attribution and what changed locally. The earlier `v1_0_1` fork was removed
  in v0.1.5 (see [`CHANGELOG.md`](../../CHANGELOG.md) `[0.1.5]`).
