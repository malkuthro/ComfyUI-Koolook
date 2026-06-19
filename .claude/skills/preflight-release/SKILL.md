---
name: preflight-release
description: Run the project's release-blocking sanity checks before merging or shipping. Verifies (1) all *.py files parse and the registered node-ID list can be statically extracted, (2) the VAE wrapper's rank/VAE-type dispatch branches still work via a stub-VAE roundtrip, (3) ComfyUI-Manager's upstream extension-node-map.json matches our actual node IDs (catches phantom-nodes drift like issue #44), and (4) every Koolook node ID referenced in tests/workflows/*.json fixtures is still registered (catches accidental ID rename / removal that would break saved workflows). Use before cutting a release, before merging a PR that touches node IDs or fork structure, or whenever the user mentions "preflight" / "release sanity check" / "are we good to ship". Wraps tools/preflight_release.py.
---

# preflight-release

Project-specific release-blocking sanity check. Runs four independent
checks via `tools/preflight_release.py`; any failure means "do not ship
yet, investigate first."

This skill is distinct from the user-level `preflight` skill (which is
about repo infrastructure — branch protection, labels, git state).
This one is about **code/metadata correctness**.

## When to use it

- **Before cutting a release.** Run it after the release-prep PR is
  open, before triggering the publish workflow.
- **Before merging any PR that touches node IDs or fork structure.** A
  changed `NODE_CLASS_MAPPINGS` or a renamed fork directory surfaces here
  so the workflow breakage is a *conscious* choice, not an accident.
  (Breaking a **Koolook** node is allowed by default; breaking a **fork**
  node is not — see the `workflows` check below for the split.)
- **After running [`docs-sync`](../docs-sync/SKILL.md) on a fork-version
  bump.** The string-sweep can leave references stale if anything is
  missed; pre-flight catches it before merge.
- **On demand** when the user asks "are we good to ship?" /
  "run preflight" / "release sanity check".

## How to invoke

Run the wrapped script directly:

```bash
python tools/preflight_release.py            # all four checks
python tools/preflight_release.py -v         # verbose per-file output
python tools/preflight_release.py --check static --check workflows
python tools/preflight_release.py --skip manager-meta   # offline mode
```

Exit code 0 = ship-clean. Exit code 1 = fix something. Exit code 2 =
invalid CLI usage.

## What each check actually does

### 1. `static` — AST extraction of NODE_CLASS_MAPPINGS

- Walks every `*.py` in the repo (excluding `__pycache__`, `.venv`,
  `upscaler_FIX/`, `nuke_CAM_exporter/`).
- AST-parses each one. **No imports** — works without torch / ComfyUI
  installed, which means CI doesn't need to install heavy deps.
- Collects the literal string keys of every `NODE_CLASS_MAPPINGS = {...}`
  literal-dict assignment found across the repo.
- Reports any parse errors with file:line.

**Limitation:** doesn't follow runtime-computed mappings (dict comprehensions,
`_namespace_mappings()` calls, etc.). For the current Radiance Koolook
v2_3_3 fork this happens to match reality because all exposed IDs are
in `SKIP_VERSION_SUFFIX`, so the source-dict keys equal the registered
keys. Future fork additions that rely on suffix mangling will need the
extraction extended.

### 2. `dispatch` — VAE rank/VAE-type branches

- Mocks `torch` with a `FakeTensor` stub that mimics just enough of the
  tensor API to drive the wrapper's dispatch logic (no real math).
- Loads `forks/radiance_koolook/versions/v2_3_3/{nodes_vae,color_helpers}.py`
  via `importlib`.
- Runs five test cases through `Easy_hdr_VAE_encode/decode`:
  1. 4-D image input + 2-D image VAE → `path=image`
  2. 5-D video input + 2-D image VAE → `path=video-iter` (frame iteration)
  3. 5-D video input + 3-D-aware video VAE → `path=video-3d` (passthrough)
  4. 5-D latent + 2-D VAE decode → frame iteration, 4-D output
  5. 4-D latent decode → standard path
- Asserts each case produces the expected shape + dispatch path string in
  `debug_info`.

Catches regressions to the rank-handling logic that the v0.2.0 release
introduced.

### 3. `manager-meta` — extension-node-map.json drift

- Fetches `https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/extension-node-map.json`.
- Looks up our entry (keyed on `https://github.com/malkuthro/ComfyUI-Koolook`).
- Diffs the upstream node-ID list against the AST-extracted list from check 1.
- Reports **phantom IDs** (in upstream but not in our code — Manager has
  stale data) and **missing IDs** (in our code but not in upstream — our
  code is ahead of the upstream metadata).

**Network-dependent.** Skipped automatically when offline (e.g. `--skip
manager-meta`, or in CI where it's currently always skipped per
`.github/workflows/ci.yml` because the drift is tracked separately as
issue #44 and fails will be noisy until the upstream PR lands).

When the upstream issue is resolved, remove the `--skip manager-meta`
from the CI job and let it become a release blocker.

### 4. `workflows` — fixture node-ID stability

- Walks `tests/workflows/*.json`.
- For each fixture, parses the JSON and extracts every node `type` (handles
  both ComfyUI canvas format with a top-level `nodes` array, and API
  format where each top-level dict value has a `class_type` key).
- Filters for IDs that look like Koolook IDs: matches `(?i)koolook`,
  `^Easy[A-Z_]`, or `^easy_[a-z]`.
- Asserts every Koolook ID found in the fixtures is still in the
  AST-extracted set from check 1.

If a fixture references a node ID we no longer register, this check fails
loudly. How to read the failure depends on which node it is (see
[`CLAUDE.md`](../../../CLAUDE.md) → *Change management*):

- **Koolook-created node** (root `k_*.py`, e.g. `easy_ImageBatch`): treat
  this as a **drift detector**, not a back-compat gate. Back-compat is
  opt-in for Koolook nodes, so an *intentional* rename is fine — the fix is
  to **update the fixture** to the new ID. The value here is catching an
  *unintended* rename you didn't mean to make.
- **Fork node** (anything under `forks/`, e.g. the `Easy_hdr_VAE_*` or a
  `*__koolook_vX_Y_Z` ID): this is a real **back-compat gate**. Fork IDs
  must stay stable — add the old ID back as an alias or version it; do
  **not** just edit the fixture.

Adding more fixtures over time grows coverage without code changes:
drop a new `*.json` into `tests/workflows/`, the script picks it up
automatically. See `tests/workflows/README.md` for the contribution
convention.

## How failures usually get fixed

| Check | Common failure | Usual fix |
|---|---|---|
| static | Syntax error in a `.py` file | Fix the syntax. ruff lint should also catch this. |
| static | "no NODE_CLASS_MAPPINGS literal dicts found" | Probably an accidental refactor that broke the registration; revert or restructure. |
| dispatch | Assertion fails on one of the five branches | Look at the changed file in `forks/radiance_koolook/versions/v2_3_3/`. Trace the dispatch logic in `nodes_vae.py:Easy_hdr_VAE_{encode,decode}`. The branch that broke tells you which rank/VAE combination regressed. |
| manager-meta (phantom IDs) | Upstream Manager metadata has IDs we don't register | File / update issue #44; not a code fix on our side. Skip the check. |
| manager-meta (missing IDs) | Our code has IDs upstream Manager doesn't list yet | Wait for the next upstream "update DB" run, or open an upstream PR. Skip the check until it propagates. |
| workflows | Fixture references a node ID no longer registered | **Koolook node (`k_*.py`):** update the fixture to the new ID — an intentional rename is fine. **Fork node (under `forks/`), or any node after `check backward compatibility`:** (a) add the old ID back as a `NODE_CLASS_MAPPINGS` alias pointing at the renamed class, OR (b) introduce a `_v2` suffix and start a deprecation cycle. |

## Companion docs

- `tools/preflight_release.py` — the actual implementation.
- `tests/workflows/README.md` — how to add new fixtures.
- `docs/maintainers/releasing.md` — the release procedure that this
  pre-flight feeds into.
- `docs/reference/versioning.md` — the three version axes that
  determine when each check matters.
- Issue #44 — the canonical example of the kind of drift `manager-meta`
  detects.
