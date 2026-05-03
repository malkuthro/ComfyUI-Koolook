---
name: license-pre-check
description: Run a license-compatibility audit before incorporating any third-party code into ComfyUI-Koolook (or any project that mixes licenses). Use when the user wants to fork a node, vendor a library, port a class from another repo, or add an external dependency. Must run BEFORE any code is copied or written based on upstream. Catches GPL/AGPL contamination of permissively-licensed projects (and vice-versa) early.
---

# License Pre-Check

Run this **before** copying, porting, adapting, or writing anything based on
upstream code. The goal: catch license incompatibilities before they
contaminate the package.

This skill exists because ComfyUI-Koolook v0.1.0 and v0.1.1 shipped under a
claimed MIT license while already incorporating GPL-3.0-derived code from
Radiance. The cleanup retroactively relicensed the entire package to GPL-3.0
(see `forks/THIRD_PARTY.md`). Future forks must not repeat this mistake.

## When to run this skill

- Before cloning any external repo for "we want to use their code" purposes.
- Before porting a node, function, or class from another project.
- Before adding a Python package as a runtime dependency that ships with the
  package (not a normal pip install — anything that ends up in the
  distribution tarball).
- Before merging a PR that adds files under `forks/` or imports new modules.

If the upstream is going into the sibling `../ComfyUI-Forks/` for **read-only
inspection** with no code copying, the skill is informational rather than
blocking — but still run it so the license is recorded in the manifest.

## Phase 1 — Identify what's coming in

Get from the user (or infer from the repo URL):

1. **Upstream repo URL.**
2. **What scope** is being incorporated:
   - whole package (vendor),
   - a subset of files (partial vendor),
   - just patterns/interface (clean-room reimplementation),
   - or runtime-only dependency.
3. **The intent** — fork & modify? mirror? simply call into it?

## Phase 2 — Read upstream's license

```bash
FORKS_ROOT="${KOLOOK_FORKS_DIR:-../ComfyUI-Forks}"
# If a clone already exists, read it. Otherwise do a shallow clone purely
# to read the LICENSE; remove afterwards if not keeping it.

# 2a. Find the LICENSE file at the repo root.
ls "$FORKS_ROOT/<clone>/LICENSE"*

# 2b. Read the first ~30 lines — that's enough to identify standard licenses.
head -30 "$FORKS_ROOT/<clone>/LICENSE"

# 2c. Cross-check the README and pyproject.toml — sometimes they disagree.
grep -i "license" "$FORKS_ROOT/<clone>/README.md" | head -5
grep -i "license" "$FORKS_ROOT/<clone>/pyproject.toml"
```

Record the **detected license SPDX identifier** (e.g. `GPL-3.0-or-later`,
`AGPL-3.0-only`, `MIT`, `Apache-2.0`, `BSD-3-Clause`, `MPL-2.0`, `LGPL-2.1`,
`UNLICENSED`/proprietary, or `unknown`). If `unknown`, **stop and ask the
user** — never assume permissive.

## Phase 3 — Compare to ComfyUI-Koolook's license

Read our current license:

```bash
head -3 LICENSE
grep -i "license" pyproject.toml README.md | head -10
```

Use this compatibility matrix (downstream = ComfyUI-Koolook; upstream = the
code being incorporated). "OK" means we can incorporate without changing our
license. "→ X" means we must relicense to X to incorporate.

| Upstream → \\ Our license ↓ | MIT | Apache-2.0 | LGPL-3.0 | GPL-3.0 | AGPL-3.0 |
|---|---|---|---|---|---|
| **MIT / BSD / ISC** | OK | OK | OK | OK | OK |
| **Apache-2.0** | OK ¹ | OK | OK | OK | OK |
| **MPL-2.0** | file-level OK ² | file-level OK ² | file-level OK ² | OK | OK |
| **LGPL-2.1 / LGPL-3.0** | dynamic-link only ³ | dynamic-link only ³ | OK | OK | OK |
| **GPL-2.0** | → GPL-2.0+ | → GPL-2.0+ | → GPL-2.0+ | → GPL ⁴ | → AGPL ⁴ |
| **GPL-3.0** | → GPL-3.0 | → GPL-3.0 | → GPL-3.0 | OK | → AGPL-3.0 |
| **AGPL-3.0** | → AGPL-3.0 | → AGPL-3.0 | → AGPL-3.0 | → AGPL-3.0 | OK |
| **Proprietary / no license** | ❌ STOP | ❌ STOP | ❌ STOP | ❌ STOP | ❌ STOP |
| **CC-BY-NC / non-commercial** | ❌ STOP for code ⁵ | ❌ STOP for code ⁵ | ❌ STOP for code ⁵ | ❌ STOP for code ⁵ | ❌ STOP for code ⁵ |
| **unknown / no LICENSE file** | ❌ STOP | ❌ STOP | ❌ STOP | ❌ STOP | ❌ STOP |

Footnotes:

1. Apache-2.0 → MIT: redistributing Apache code under MIT requires keeping
   Apache's NOTICE file and per-file license headers; the *combined* work
   stays MIT but the Apache portions remain Apache.
2. MPL-2.0 is file-level copyleft: you must keep the MPL on those files,
   but the rest of the package can stay under your license.
3. LGPL allows linking from a non-LGPL host *only if* users can substitute
   a modified LGPL library. For a Python package this usually means the
   LGPL library is an external dep (pip-installable), not vendored.
4. GPL-2.0 and GPL-3.0 are not bidirectionally compatible without
   `GPL-2.0-or-later`. Watch the version carefully.
5. Creative Commons licenses are designed for prose/media, not software.
   CC-BY may be acceptable for asset files but never for code.

## Phase 4 — Report

Produce a single block to the user, blocking until they acknowledge:

```
LICENSE PRE-CHECK
─────────────────
Upstream:        <repo URL> @ <commit/tag>
Upstream license: <SPDX>
Our license:      <SPDX>
Scope of import: <whole | subset | clean-room | runtime-dep>

Compatibility:   <OK | requires-relicense-to-X | BLOCKED>

If RELICENSE: explain what the user must change before proceeding.
If BLOCKED:   stop. Recommend a clean-room alternative or a different
              upstream with a compatible license.

Required attribution actions:
- Add an entry to forks/THIRD_PARTY.md
- Update forks/forks_manifest.yaml (license + license_verified_at)
- If GPL/LGPL: keep upstream LICENSE in any folder containing copied files
- If Apache: keep NOTICE file
- For modified files: prepend a header "Modified from <upstream>, <date>"
  (GPL §5(a) requires this; Apache §4(b) requires a notice of changes)
```

## Phase 5 — Persist findings

Once compatibility is confirmed (or relicensing is decided), update:

1. `forks/forks_manifest.yaml` — `license:` and `license_verified_at:` fields
   on the manifest entry.
2. `forks/THIRD_PARTY.md` — full attribution entry (see template at top of
   that file).
3. If our license must change, that is its **own preceding chore PR** —
   never bundle "relicense the whole pack" with "add a new fork", because
   the legal change deserves a focused diff.

## Anti-patterns to refuse

- ❌ "I'll just copy the file and decide on the license later." Decide
  first.
- ❌ "It's only one function" — copying still triggers the same license.
- ❌ "The README says MIT but the LICENSE file is GPL." The LICENSE file
  controls (and you should report the discrepancy upstream).
- ❌ Adding a permissive shim that wraps GPL code while pretending the
  shim is MIT. The combined work is still GPL.
- ❌ Treating `KOLOOK_FORKS_DIR` (read-only inspection) as if no license
  check were needed. It's needed — to record the license in the manifest
  even if no copying happens.

## Hand-off

After this skill completes successfully, hand off to
[`add-external-fork`](../add-external-fork/) for the actual clone +
register + wrapper steps. `add-external-fork` will refuse to run if the
manifest entry it would create has no `license_verified_at` field.
