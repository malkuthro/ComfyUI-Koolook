---
name: docs-sync
description: Find-and-replace a version tag (or any string token) across docs, code constants, and manifests in one controlled pass. Use when bumping a fork from one version to another (e.g. v2_3_3 -> v2_4_0), renaming a node ID across the repo, or generally propagating a string change everywhere it's referenced. Takes <old-tag> and <new-tag>; optional --scope flag (all / docs / code) and --dry-run. Always shows every match before applying and re-verifies post-edit.
---

# docs-sync

A controlled find-and-replace across the repo for keeping docs, manifests,
and code constants in sync when something tagged by a version string moves.

The skill does **not** rename folders or move files on its own — that would
be too easy to break things with. It only changes the **content of files**
that already exist. Folder renames (e.g. `versions/v2_3_3/` -> `versions/v2_4_0/`)
are a deliberate manual `git mv` step that this skill expects you to run
either before or after, and it'll re-grep to flag any stale references
after the move.

## When to use it

- Bumping a fork's version tag everywhere it appears (most common use).
- Renaming a node ID and propagating that to docs / changelog / manifests.
- Any string-level rename where you'd otherwise grep manually and worry
  about missing one.

## When NOT to use it

- Refactoring code structure (use a code-aware tool like an IDE refactor).
- Renaming things that have *non-textual* references (e.g. a JSON key that
  some tool computes by hash) — those need targeted edits, not a string
  sweep.

## Inputs

```
docs-sync <old-tag> <new-tag> [--scope all|docs|code|manifests] [--dry-run]
```

- `<old-tag>` — the string to find. Required. Can be a version like `v2_3_3`,
  a display version like `v2.3.3`, a node ID like `Easy_hdr_VAE_encode`, or
  a free-form tag.
- `<new-tag>` — the replacement string. Required.
- `--scope` — optional. Restrict where the find-and-replace applies:
  - `all` (default): docs, code, and manifests.
  - `docs`: only `docs/` and `README.md`.
  - `code`: only `*.py` source files.
  - `manifests`: only `pyproject.toml`, `forks/*.yaml`, `forks/*.md`.
- `--dry-run` — show every match without applying changes.

## Common usage patterns

### Bumping a fork version

When you're bumping `radiance_koolook` from v2.3.3 to v2.4.0, run the
skill **twice** to cover the two textual representations the codebase
uses:

```
docs-sync v2_3_3 v2_4_0     # underscored form (folder names, NAMESPACE_SUFFIX, docs paths)
docs-sync v2.3.3 v2.4.0     # dotted form (DISPLAY_SUFFIX, prose in docs and CHANGELOG)
```

You can also run with `--dry-run` first to see the scope of changes
before committing. The skill will not rename the actual `versions/v2_3_3/`
folder — do that manually with `git mv` after the string sweep is clean.

### Renaming a node ID after a v2 break

Before any tag change, every reference to the old ID gets updated:

```
docs-sync Easy_hdr_VAE_encode Easy_hdr_VAE_encode_v2 --scope docs
```

The `--scope docs` is important here — you almost certainly do *not*
want to update the Python `NODE_CLASS_MAPPINGS` key, because that's the
*old* ID that needs to remain registered for back-compat. Code edits
for new ID registration belong in a focused PR, not a sweep.

## Procedure

1. **Validate inputs.**
   - Ensure `<old-tag>` and `<new-tag>` are non-empty and distinct.
   - Reject if `<old-tag>` looks dangerous as a substring (very short
     literal like a single letter or common word). Ask the user to
     confirm if `<old-tag>` is shorter than 4 characters.

2. **Search.** Use ripgrep (via the agent's Grep tool):
   - Scope `all`: search the whole worktree, excluding `__pycache__/`,
     `.git/`, `node_modules/`, and binary files (default ripgrep behavior).
   - Scope `docs`: restrict to `docs/**/*.md`, `README.md`, top-level `*.md`.
   - Scope `code`: restrict to `**/*.py`.
   - Scope `manifests`: restrict to `pyproject.toml`, `forks/forks_manifest.yaml`,
     `forks/THIRD_PARTY.md`, `forks/**/UPSTREAM_PIN.yaml`.

3. **Categorize matches.** Group hits by file:
   - Code (`.py` files)
   - Manifests (`.yaml`, `.toml`, `THIRD_PARTY.md`)
   - Docs (`docs/**/*.md`, `*.md`)
   - Other (anything not in the above)

4. **Present the list.** Show every match with file path + line number +
   snippet of the line. Group by category. Total count at the bottom.

5. **Confirm.** Print: "Apply replacement of `<old>` -> `<new>` to N
   occurrences across M files? (y/N — or use --dry-run to preview only)"
   Wait for user confirmation. Skip this step if `--dry-run`.

6. **Apply.** For each file with matches, do an in-place text replacement.
   Use the agent's Edit tool with `replace_all=true` if multiple hits in
   one file, otherwise per-occurrence with enough context to be unique.

7. **Verify.** Re-grep for `<old-tag>` in the same scope. Report any
   remaining occurrences (these are likely intentional — e.g. CHANGELOG
   entries describing the *previous* version, which should reference the
   old tag literally and not get rewritten). Highlight them as
   "intentional remainders to keep" so the user can confirm.

8. **Report.** Final summary:
   - Files modified: N
   - Replacements applied: M
   - Remaining occurrences (if any): list
   - Suggested next steps (folder renames if applicable, commit message,
     dev-sync if relevant).

## Safety guardrails

- **Never delete or move files.** The skill only edits content within
  existing files.
- **Never auto-commit.** All changes stay in the working tree for the
  user to review with `git diff` before committing.
- **Word-boundary intent.** When `<old-tag>` is short or could be a
  substring of unrelated tokens (e.g. `v2` could match `v2.3.3` and
  `v2_pipeline_test`), prompt the user to use a more specific tag or
  add anchoring characters (e.g. `v2_3_3` rather than `v2`).
- **Skip changelog entries describing past releases.** When `--scope all`
  finds matches in `CHANGELOG.md` under sections that describe a
  *previous* release (e.g. `## [0.1.5]` or `## [0.2.0]`), default to
  *not* rewriting those — they're historical record. Surface them as
  "found but kept as historical" in the report.

## Examples

### Example 1 — bump radiance_koolook from v2.3.3 to v2.4.0

```
docs-sync v2_3_3 v2_4_0
```

Expected scope of changes:
- `forks/radiance_koolook/__init__.py` (import path)
- `forks/radiance_koolook/versions/v2_3_3/__init__.py` (NAMESPACE_SUFFIX
  constant) — but only after the folder is renamed via `git mv`
- `forks/THIRD_PARTY.md` (section heading + body references)
- `forks/forks_manifest.yaml` (notes field)
- `docs/user_guide/nodes/radiance_koolook_v2_3_3/` (path references in
  page bodies — but folder rename via `git mv` happens manually)
- `CHANGELOG.md` `[Unreleased]` section (new version note)

Then a separate run for the dotted form:

```
docs-sync v2.3.3 v2.4.0
```

Picks up:
- `DISPLAY_SUFFIX = " (Koolook v2.3.3)"` in the moved `__init__.py`
- Prose in docs that says "v2.3.3 baseline"
- THIRD_PARTY.md prose
- CLAUDE.md fork policy section

### Example 2 — preview a node-ID rename

```
docs-sync Easy_hdr_VAE_encode Easy_hdr_VAE_encode_v2 --scope docs --dry-run
```

Reports every doc page that mentions the old ID without making any
changes. Useful before deciding whether to do the rename in code first.

## Related skills

- `bump-fork-version` *(planned, not yet built)* — orchestrates the full
  fork bump including folder rename, constant updates, and CHANGELOG
  entry. Will internally invoke `docs-sync` for the string-sweep step.
- `add-external-fork` — for setting up a brand-new fork, not bumping an
  existing one.
