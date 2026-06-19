# Versioning — three numbers, three meanings

This codebase carries **three independent version numbers** that often get
confused. They mean different things, bump on different schedules, and live
in different places.

## At a glance

| Axis | What it is | Where it lives | Example |
|---|---|---|---|
| **Pack version** | The Koolook release that ships to GitHub + Comfy Registry. Semver. | [`pyproject.toml`](../../pyproject.toml) `version = "X.Y.Z"`, [`CHANGELOG.md`](../../CHANGELOG.md) `## [X.Y.Z]` | `0.2.0` |
| **Fork wrapper version** | A slot under `versions/` that holds Koolook's modified code for a particular upstream baseline. Underscored. | `forks/<fork>/versions/vX_Y_Z/` directory name + the `NAMESPACE_SUFFIX` constant inside it | `v2_3_3` |
| **Upstream pinned commit** | The exact commit of the upstream third-party repo that the fork wrapper is matched to. SHA. | [`forks/<fork>/versions/vX_Y_Z/UPSTREAM_PIN.yaml`](../../forks/radiance_koolook/versions/v2_3_3/UPSTREAM_PIN.yaml), [`forks/forks_manifest.yaml`](../../forks/forks_manifest.yaml), [`forks/THIRD_PARTY.md`](../../forks/THIRD_PARTY.md) | `f262f47ddfda01ece154bf80c22769b1e4cef795` |

A current snapshot reads: *"Koolook **0.2.0** ships the **v2_3_3** Radiance
fork wrapper, which is pinned to upstream Radiance commit **f262f47**."*
All three are unrelated numbers that happen to all describe the same
release.

## When each one bumps

### Pack version (`0.2.0` → `0.2.1` / `0.3.0` / `1.0.0`)

Bumps on **every release of the Koolook pack itself**. Drives the Comfy
Registry publish workflow. Follow [`docs/maintainers/releasing.md`](../maintainers/releasing.md)
for the canonical procedure.

- **Patch** (`0.2.0 → 0.2.1`): bug fixes, internal refactors, no surface change.
- **Minor** (`0.2.0 → 0.3.0`): new nodes, new node options, anything that adds surface but doesn't break existing workflows.
- **Major** (`0.2.0 → 1.0.0`): only after a v2-style breaking change has shipped (a removed-after-deprecation node ID, an INPUT_TYPES change a saved workflow can't tolerate, etc.).

### Fork wrapper version (`v2_3_3` → `v2_4_0` / `v2_3_4`)

Bumps when **upstream releases a new version we want to track** AND the
new upstream version requires meaningfully different wrapper code.

Two patterns, decided per-fork:

- **In-place upgrade** (used by Radiance Koolook today). All exposed node
  IDs are listed in `SKIP_VERSION_SUFFIX` so the user-facing IDs are
  stable across versions (e.g. `Easy_hdr_VAE_encode` stays as itself
  whether the wrapper is `v2_3_3` or `v2_4_0`). When upstream bumps, the
  whole `versions/v2_3_3/` directory gets renamed to `versions/v2_4_0/`
  in-place; there's no parallel coexistence because both versions would
  collide on the same ID. Saved workflows continue loading because the
  ID didn't change.
- **Versioned coexistence**. Node IDs carry the `__koolook_vX_Y_Z`
  suffix (the default behavior unless explicitly added to
  `SKIP_VERSION_SUFFIX`). When upstream bumps, the new `versions/v2_4_0/`
  is added alongside the old one. Both are registered simultaneously
  for one release cycle. Old workflows keep loading via the suffixed
  old ID; new workflows can adopt the new suffixed ID. The old version
  is removed in the next release.

### Upstream pinned commit (`f262f47` → some new SHA)

Bumps when **we re-sync the wrapper against a newer upstream commit**.
Recorded in:

- `forks/<fork>/versions/<vX_Y_Z>/UPSTREAM_PIN.yaml` (the per-version pin)
- `forks/forks_manifest.yaml` (the project-level fork inventory)
- `forks/THIRD_PARTY.md` (the human-readable attribution + change notes)

A new pinned commit may or may not require a new fork wrapper version —
small upstream patches can land at the same wrapper slot if our wrapper
code doesn't have to change. Bigger upstream releases usually warrant a
new wrapper version too.

## Public node-ID versioning (the `_v2` rule)

> **Scope.** This section is **mandatory for fork nodes** (anything under
> `forks/`, like the `Easy_hdr_VAE_*` example used below) — they keep full
> back-compat discipline by default. For **Koolook-created nodes** (the root
> `k_*.py` custom nodes) it is **opt-in**: by default a breaking change is made
> **in place** — rename or alter the existing ID, note it in the CHANGELOG,
> done — with no `_v2` suffix and no deprecation cycle, unless the maintainer
> asked to `check backward compatibility`. See [`CLAUDE.md`](../../CLAUDE.md) →
> *Change management*.

Independent of the three axes above, individual node IDs can carry a
public version suffix when a breaking change forces it:

- Original ID (no suffix): `Easy_hdr_VAE_encode`. Stays this forever.
- Bumped ID after a breaking change: `Easy_hdr_VAE_encode_v2`. Only
  introduced when the original would otherwise break saved workflows.
- Both IDs register simultaneously for **one full release cycle** (the
  old one displays as `Easy HDR VAE Encode (Koolook, v1 — please migrate)`
  during this window).
- Old ID is dropped one release later. Pack version goes major (`1.0.0`)
  to signal the breakage.

This is the rare path. Most version changes don't need it because either
the change isn't actually breaking, or the fork wrapper version mechanism
above already handles the back-compat.

## Which skill / doc owns which axis

| Axis | Skill | Doc |
|---|---|---|
| Pack version | (none, manual via release PR) | [`docs/maintainers/releasing.md`](../maintainers/releasing.md) |
| Fork wrapper version | [`docs-sync`](../../.claude/skills/docs-sync/SKILL.md) (string-sweep), [`add-external-fork`](../../.claude/skills/add-external-fork/SKILL.md) (initial setup) | [`forks/README.md`](../../forks/README.md), [`docs/maintainers/node-versioning.md`](../maintainers/node-versioning.md) |
| Upstream pinned commit | (none — manual edit to `UPSTREAM_PIN.yaml` + `forks_manifest.yaml`) | [`forks/THIRD_PARTY.md`](../../forks/THIRD_PARTY.md) |
| Node-ID `_v2` suffix | (none — manual code change with deprecation alias) | (this document) |

## Example — a hypothetical Radiance v2.4.0 release

Suppose upstream Radiance ships v2.4.0. We re-sync and the wrapper code
changes only slightly — same node IDs, same interface. The bumps:

- **Upstream pinned commit:** `f262f47…` → whatever the v2.4.0 release commit is. Update `UPSTREAM_PIN.yaml`, `forks_manifest.yaml`, `THIRD_PARTY.md`.
- **Fork wrapper version:** `v2_3_3` → `v2_4_0`. Rename `versions/v2_3_3/` to `versions/v2_4_0/`. Update `NAMESPACE_SUFFIX` and `DISPLAY_SUFFIX` constants inside. Update parent `__init__.py` import. Rename matching docs folder `docs/user_guide/nodes/radiance_koolook_v2_3_3/` → `radiance_koolook_v2_4_0/`. Update inline references via [`docs-sync v2_3_3 v2_4_0`](../../.claude/skills/docs-sync/SKILL.md).
- **Pack version:** `0.2.0` → `0.3.0`. Cut a release PR per [`releasing.md`](../maintainers/releasing.md).
- **Node IDs:** unchanged (stable IDs). No `_v2` suffix introduced. Saved workflows keep loading.

If the v2.4.0 wrapper required a *behaviorally* breaking change to one of
the node IDs — say `Easy_hdr_VAE_encode` now returns a different shape
that downstream samplers can't handle — only then would we add
`Easy_hdr_VAE_encode_v2` and start the deprecation cycle.

## See also

- [`docs/maintainers/releasing.md`](../maintainers/releasing.md) — pack version release procedure.
- [`docs/maintainers/node-versioning.md`](../maintainers/node-versioning.md) — the rules for safely changing `INPUT_TYPES` / `RETURN_TYPES` / class names without breaking saved workflows.
- [`forks/THIRD_PARTY.md`](../../forks/THIRD_PARTY.md) — upstream attribution and change notes for each fork.
- [`forks/forks_manifest.yaml`](../../forks/forks_manifest.yaml) — machine-readable fork inventory.
- [`.claude/skills/docs-sync/SKILL.md`](../../.claude/skills/docs-sync/SKILL.md) — the find-and-replace skill for fork-version bumps.
