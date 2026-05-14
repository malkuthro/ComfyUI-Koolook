---
name: tdd
description: Test-driven development with red-green-refactor loop, customised for ComfyUI-Koolook (Python nodes + ComfyUI sidebar JS + docs/designs/ visual specs). Use when the maintainer wants to build features or fix bugs using TDD, mentions "red-green-refactor", wants integration tests, or asks for test-first development.
allowed-tools: Bash, Read, Glob, Grep, Edit
---

# Test-Driven Development — ComfyUI-Koolook

## Philosophy

Same as the global TDD skill: tests verify *behaviour through public
interfaces*, not implementation details. Code can change entirely; tests
shouldn't. Vertical slices (one test → one impl → repeat), never
horizontal (all tests then all code).

Generic principles — deep modules, mocking, interface design,
refactoring patterns, scope-check structure — live in the global skill
at `~/.claude/skills/tdd/`. The sections below are ComfyUI-Koolook-
specific and take precedence where they differ.

## Workflow

Standard TDD cycle:

1. **Plan** — agree with the maintainer on which behaviours to test (not
   implementation steps). Flag which assertions are CSS / layout /
   structural — those need visual verification (see below).
2. **Tracer bullet** — one test, one impl. Prove the end-to-end path
   works on this project's actual test surface (lint + smoke + preflight
   today; pytest / vitest only if you introduce them — see *Test
   surface* below).
3. **Loop** — RED → GREEN per behaviour. One test at a time. Minimal
   code to pass.
4. **Refactor** — only after GREEN. Run the gates after each step.

## Checklist per cycle

```
[ ] Test describes behaviour, not implementation
[ ] Test uses public interface only
[ ] Test would survive an internal refactor
[ ] Code is minimal for this test
[ ] No speculative features added
[ ] If the cycle's assertion is a CSS class, inline style, layout
    property, or wrapper structure: secondary visual verification done
    against the matching docs/designs/<feature>.html mockup
    (see visual-verification.md)
```

## When test green isn't enough

Test green is **necessary but not sufficient** for any cycle whose
assertion is:

- A CSS class string (`hidden`, `selected`, theme tokens)
- An inline `style={}` value
- A layout property (`position`, `padding`, `margin`, `z-index`,
  `top`/`bottom`/`left`/`right`)
- Section / wrapper structure ("recovery section auto-expands under the
  clicked preset", "badge on top, meta line below")
- Anything in the cycle's AC that says "looks like", "matches the
  mockup", "scoped to", or "above/below/beside"

For those, before declaring the cycle done, run the procedure in
[visual-verification.md](visual-verification.md): render in the real
ComfyUI shell via `dev-sync`, screenshot the affected state, diff
against the design source.

**Tiebreaker:** if a `docs/designs/<feature>.html` mockup contradicts
the AC's English description, the mockup wins. The English is a pointer;
the design page is the spec.

## ComfyUI-Koolook code surface

| Layer | Where | Test path today |
|---|---|---|
| Custom nodes (Python) | `k_*.py`, `koolook_routes.py` at repo root | lint (`ruff`), smoke (`compileall`), preflight (`tools/preflight_release.py`) |
| Forks / wrappers | `forks/*` | preflight `vae-dispatch` for the Radiance VAE wrapper rank/branch dispatch |
| Sidebar UI (JS) | `web/koolook_sidebar.js` + `web/sidebar/*.js` | visual verification only — no JS runner today |
| Design specs | `docs/designs/*.html` + `docs/designs/*-redesign.md` | mockup *is* the spec; iterate via Launch preview or `python3 -m http.server` |
| Live runtime | `KOLOOK_COMFYUI_DEV_PATH` (per-machine, set in `.env`) | `dev-sync` to deploy, restart ComfyUI to load |

## Test surface — what to run

Today's CI gates (see `.github/workflows/ci.yml`):

- **Lint:** `ruff check .`
- **Syntax smoke:** `python -m compileall -q -x '(\.venv|forks|upscaler_FIX/github_repos|nuke_CAM_exporter)' .`
- **Security:** `bandit -r . -x ./.venv,./forks,... -ll`
- **Pre-flight release:** `python tools/preflight_release.py --skip manager-meta -v`

No pytest or JS test runner is configured by default. Introduce one
**only when there's a behaviour that genuinely needs unit testing** —
not speculatively.

### Adding pytest (when first Python behaviour is testable)

- Add tests under `tests/test_<area>.py` next to `tests/workflows/`.
- Install into a local venv per [project supply-chain rules](../../../CLAUDE.md)
  in the global `~/.claude/CLAUDE.md`:
  ```bash
  python3 -m venv .venv
  .venv/bin/pip install pytest
  ```
- Add the `pytest` step to `ci.yml` in the **same PR** that introduces
  the first test. Don't ship pytest setup as a separate PR — that leaves
  a half-wired test runner with no tests behind it.

### Adding a JS runner (only if a pure-function helper is extracted)

No JS runner today. The sidebar runs inside ComfyUI's browser; full-UI
behaviour is verified via `dev-sync` + visual diff against the mockup.
A JS unit-test runner only earns its place when a non-trivial pure
function (parsing, sorting, schema validation) is extracted from the
sidebar modules. When that happens:

- Use vitest (lighter than jest, ESM-native).
- Keep `ignore-scripts=true` in `.npmrc` per project supply-chain rules.
- `--frozen-lockfile` in CI.
- Audit before installing: `npm audit` must be clean.

## The dev-sync iteration loop

Frontend implementation pairs with the dev-sync loop documented in
[dev-iteration-loop.md](../../../docs/maintainers/dev-iteration-loop.md):

1. Edit `web/koolook_sidebar.js` or any module under `web/sidebar/` in
   the worktree.
2. `python scripts/sync_to_dev.py --scope "<10-word summary>"` — copies
   the worktree's runtime files into `KOLOOK_COMFYUI_DEV_PATH`. The
   scope flag persists to `<target>/web/_dev_build.json` so the sidebar
   footer shows what's deployed.
3. Restart ComfyUI; verify visually.
4. **If a mockup exists** for what you implemented (look under
   `docs/designs/`), screenshot the result and diff against the
   matching mockup card. Don't request review until visual parity is
   reached or each deviation is documented (see
   [visual-verification.md](visual-verification.md)).
5. Iterate.

End-to-end: ~10–30 seconds per cycle.

**`dev-sync` is user-initiated only.** Never trigger it from an
automated flow (after a commit, after `/ship-pr`, at session end). The
maintainer typically runs multiple parallel sessions across worktrees;
an unsolicited sync silently overwrites what another session is
reviewing. See project CLAUDE.md for the full rule.

## High-value test targets

When you do introduce tests, prioritise:

- **Node registration / `NODE_CLASS_MAPPINGS`.** Preflight already
  statically extracts node IDs from every `*.py`; tests that exercise
  those IDs prevent silent name drift across saved workflows.
- **Server route input validation.** `koolook_routes.py` exposes
  filesystem operations behind aiohttp routes — verify name
  sanitisation, path-traversal rejection, and the `_resolve_target`
  boundary.
- **VAE wrapper rank / dispatch.**
  `forks/radiance_koolook/versions/v2_3_3/nodes_vae.py` has rank-based
  branches. Stub-VAE roundtrip tests already exist in preflight
  (`vae-dispatch`) — extend, don't duplicate.
- **Snapshot library row augmentation.** `_listing.json` rows resolve
  `latestAutosaveMtime` from `max(periodic.json, pre_load_*.json)` —
  the exact boundary that motivated the snapshot-dialogs redesign.

## Project drift patterns (scope-check signals)

When closing the loop on implementation, watch for these — they look
fine in code review but cause incidents in this codebase:

- **Vendored upstream code into MAIN** instead of into `forks/`. Project
  CLAUDE.md is explicit: third-party code stays outside MAIN; MAIN holds
  wrapper loaders + tracking docs only.
- **Renamed a namespaced node ID** that already appears in saved
  workflows. Node IDs are external API — never rename `Easy_*`,
  `*__koolook_v2_3_3`, or any other ID present in `tests/workflows/*.json`
  without a deprecation + alias.
- **Modified a `forks/<vendor>/versions/vX_Y_Z/`** file expecting upstream
  to also change — those are frozen baseline pins (per Radiance fork
  policy).
- **Frontend feature without dev-sync verification.** Lint + smoke
  prove nothing about the live sidebar.
- **Hardcoded absolute paths with usernames** in committed files —
  sibling-project references must go through `KOLOOK_*` env vars only;
  `.env.example` is the public template.
- **Renamed a node and ComfyUI-Manager's `extension-node-map.json` now
  drifts** — preflight catches this via `manager-meta`, but it's skipped
  on PRs (issue #44). Manually run
  `python tools/preflight_release.py manager-meta` before merging.
- **`dev-sync` triggered automatically** (after a commit, after
  `/ship-pr`, at session end, from a hook). Project CLAUDE.md forbids
  this; it's user-initiated only.

## Final scope alignment check

After all cycles are green and before declaring work ready for review,
run a fresh-context scope-check sub-agent against the diff + spec
(issue body, AC, applicable CLAUDE.md sections, parent tracker, the
drift patterns above). Routing rules match the global skill's
[`scope-check.md`](~/.claude/skills/tdd/scope-check.md):

- **Verdict: clean** — show the report, proceed. Ready for review.
- **Verdict: drift detected** — always show the report to the maintainer.
  Never revert anything on your own. Offer the maintainer the menu in
  `scope-check.md` (keep / revert / split / loop back) and act on their
  choice.

The report is part of "ready for review" even on a clean run — a
missing report is indistinguishable from a skipped check.
