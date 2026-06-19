---
name: add-external-fork
description: Set up an external upstream repository as a pinned reference checkout under ../ComfyUI-Forks and register it in MAIN's fork tracking files. Use when adding a new third-party fork, cloning the Radiance (or other) baseline on a fresh machine, registering an external repo for upstream sync work, or asks about the forks/ workflow.
---

# Add External Fork

Encodes the ComfyUI-Koolook fork workflow: external upstream code is **never vendored into MAIN**. Instead, raw clones live in the sibling folder `../ComfyUI-Forks/`, MAIN tracks them via `forks/forks_manifest.yaml`, and any modified node code lives in MAIN under `forks/<package>/versions/<vX_Y_Z>/` with namespaced node IDs.

Use this skill whenever the user wants to:
- clone an external repo for upstream comparison/sync work,
- set up the documented baseline on a fresh machine,
- add a brand-new fork (with or without a wrapper module in MAIN),
- bump a fork to a new pinned version (creates a new `vX_Y_Z` folder).

## Conventions (do not deviate)

- **Forks root:** `../ComfyUI-Forks` (relative to MAIN). Overridable via env var `KOLOOK_FORKS_DIR`. Never clone third-party repos *inside* MAIN.
- **External folder names:**
  - Pinned baseline → `<repo>-v<version>-koolook` (e.g. `radiance-v1.0.1-koolook`)
  - Rolling upstream → `<repo>-main-upstream` (e.g. `radiance-main-upstream`)
- **Local version paths in MAIN:** `forks/<package>/versions/v<MAJOR>_<MINOR>_<PATCH>/`
- **Node ID namespace suffix:** `__koolook_v<MAJOR>_<MINOR>_<PATCH>` (must match the version folder).
- **Single source of truth:** `forks/forks_manifest.yaml`. Every external repo must have an entry there.
- **Never rename existing namespaced node IDs** that already appear in saved workflows. New versions get a new suffix. Fork nodes keep full backward-compat discipline by **default** — the opt-in "clean rename" default in `CLAUDE.md` → *Change management* applies to Koolook-created nodes only, **not** forks.

## Phase 1 — Discover

Ask (or infer) before doing anything destructive:

1. **Repo URL** (e.g. `https://github.com/fxtdstudios/radiance.git`).
2. **Version label** (e.g. `1.0.1`). Used to name the folder and the namespace suffix.
3. **Pinned commit** (full hash). If unknown, offer to use the tip of the upstream default branch and record that hash.
4. **Source ref** (tag, branch, or descriptor — e.g. `comfyui (closest public baseline)`).
5. **Variants needed:**
   - pinned baseline only?
   - rolling upstream only?
   - both?
6. **Wrapper in MAIN?** (yes if Koolook will modify nodes; no if it's reference-only.)
7. **Modified files** (paths that will live in MAIN under the new `versions/<vX_Y_Z>/`).

If the request is "set up the baseline on this machine" and a manifest entry already exists, **read the manifest** to get all of the above instead of asking — that's the whole point of `forks_manifest.yaml`.

## Phase 2 — External clones (outside MAIN)

```bash
FORKS_ROOT="${KOLOOK_FORKS_DIR:-../ComfyUI-Forks}"
mkdir -p "$FORKS_ROOT"
cd "$FORKS_ROOT"
```

### Pinned baseline

```bash
PIN_DIR="<repo>-v<version>-koolook"
git clone <source_repo> "$PIN_DIR"
cd "$PIN_DIR"
git checkout <pinned_commit>
git rev-parse HEAD   # verify, must match pinned_commit
cd ..
```

### Rolling upstream (optional)

```bash
ROLL_DIR="<repo>-main-upstream"
git clone <source_repo> "$ROLL_DIR"
# leave on upstream default branch; do not check out a pin
```

If a clone already exists, do **not** re-clone. Instead `cd` in, fetch, and verify the commit/branch matches expectations. Report any drift.

## Phase 3 — Register in MAIN

### 3a. `forks/forks_manifest.yaml`

Append (or update) an entry. Mandatory fields:

```yaml
- id: "<package>_v<MAJOR>_<MINOR>_<PATCH>_baseline"
  name: "<Human readable name>"
  source_repo: "<URL>"
  source_ref: "<tag/branch descriptor>"
  pinned_commit: "<full sha>"
  external_checkout:
    relative_path_from_forks_root: "<repo>-v<version>-koolook"
    upstream_tracking_path_from_forks_root: "<repo>-main-upstream"
    default_branch: "<work branch name, optional>"
  local_paths:
    - "forks/<package>/versions/v<MAJOR>_<MINOR>_<PATCH>/__init__.py"
    - "forks/<package>/versions/v<MAJOR>_<MINOR>_<PATCH>/UPSTREAM_PIN.yaml"
    # plus every modified node source file
  status: "modified"          # or "reference_only"
  sync_state: "local-modified-from-upstream"
  license: "to_verify_at_source_ref"
  notes: "<what was modified and why>"
```

Bump `updated_at` at the top of the file.

### 3b. `forks/THIRD_PARTY.md`

Add a new entry using the template at the top of that file. Keep it human-readable.

### 3c. Wrapper module in MAIN (only if `status: modified`)

Create the version folder:

```
forks/<package>/versions/v<MAJOR>_<MINOR>_<PATCH>/
  __init__.py            # imports node mappings + applies namespace suffix
  UPSTREAM_PIN.yaml      # mirrors manifest pin for one-file lookup
  <copies of modified node source files>
```

`UPSTREAM_PIN.yaml` minimum:

```yaml
source_repo: "<URL>"
source_ref: "<descriptor>"
pinned_commit: "<full sha>"
external_checkout_relative_to_forks_root: "<repo>-v<version>-koolook"
upstream_tracking_relative_to_forks_root: "<repo>-main-upstream"
external_work_branch: "<branch>"
```

The version `__init__.py` must:
- import `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` from each modified node module,
- apply suffix `__koolook_v<MAJOR>_<MINOR>_<PATCH>` to every ID (and a display suffix like ` (Koolook v<version>)`),
- optionally register short-ID overrides (e.g. `k_easy_OCIO_v101`).

Then create/update the package entrypoint `forks/<package>/__init__.py` to merge all version mappings into a single `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`, and wire those into the **root `__init__.py`**.

### 3d. Documentation touchpoints

If the new fork changes user-visible behavior or adds/removes nodes:
- Add the new node IDs and categories to `README.md` (node table + categories list).
- Update `CHANGELOG.md` under `[Unreleased]`.
- Mention any new optional Python deps in the README "Compatibility" bullet.

## Phase 4 — Verify

Run from MAIN:

```bash
git status                               # MAIN should show only intended additions
python3 -c "import sys; sys.path.insert(0,'.'); import __init__ as m; print(sorted(m.NODE_CLASS_MAPPINGS))"  # smoke test
ls "$FORKS_ROOT"                          # confirms external clones present
cd "$FORKS_ROOT/<repo>-v<version>-koolook" && git rev-parse HEAD   # equals pinned_commit
```

If any of these fail, stop and report the discrepancy — do not paper over it.

## Phase 5 — Branch & commit

Per the repo's PR workflow, do MAIN-side changes on a feature branch, never on `main`:

```bash
git checkout -b feat/forks-add-<package>-v<MAJOR>_<MINOR>_<PATCH>
git add forks/ README.md CHANGELOG.md __init__.py
git commit -m "Add <package> v<version> fork wrapper and tracking."
```

External clones under `../ComfyUI-Forks/` are **not** committed to MAIN (they're outside the repo).

## Bumping an existing fork to a new pinned version

1. Phase 1: keep the existing entry; add a NEW entry with the new version.
2. Phase 2: clone a new pinned folder `<repo>-v<new_version>-koolook` (do not delete the old one — old workflows may need it for diff).
3. Phase 3: create a new `forks/<package>/versions/v<new>_<MAJOR>_<MINOR>_<PATCH>/` and a NEW namespace suffix. **Do not rename existing IDs.**
4. Update `forks_manifest.yaml`, `THIRD_PARTY.md`, README, CHANGELOG.
5. Verify both old and new namespaced IDs still resolve.

## Reference-only forks (no wrapper, no modifications)

Skip 3c entirely. Manifest entry uses `status: reference_only`, `local_paths: []`. External clone is still useful for inspection/docs.

## Anti-patterns

- ❌ Cloning third-party repos *inside* MAIN (e.g. `forks/<package>/upstream/`).
- ❌ Reusing an existing version folder for a different upstream commit.
- ❌ Renaming or removing namespace suffixes that already appear in saved workflows.
- ❌ Editing files inside `../ComfyUI-Forks/<repo>-v<version>-koolook/` and expecting MAIN to track them — that folder is reference-only.
- ❌ Updating `forks_manifest.yaml` without also updating the matching `UPSTREAM_PIN.yaml` (and vice-versa).
