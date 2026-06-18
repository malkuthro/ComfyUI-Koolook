# ComfyUI-Koolook Agent Workflow

## Source-of-truth model

- MAIN repo: `ComfyUI-Koolook`
- External forks root (default): `../ComfyUI-Forks` (relative to MAIN)
- External third-party code must remain outside MAIN.
- Sibling projects (read-only references that live outside MAIN) are referenced via `KOLOOK_*` env vars only. Never hardcode absolute paths with usernames in committed files. Real paths live in `.env` (gitignored); the committed `.env.example` is the public template. See README "Sibling Projects" and `docs/reference/glossary.md`.

## Hard rules

- Never delete, move, rename, or edit files outside this repository. External
  paths may be read only when the maintainer explicitly provides them for
  inspection or copying into the repository. Any destructive or modifying
  action outside the repo is forbidden; stop and ask the maintainer to perform
  it manually instead.
- If an instruction is unclear, contradictory, unusually destructive, or seems
  to conflict with the maintainer's broader intent, do not execute it
  literally. Stop, explain the ambiguity in plain language, and ask for
  confirmation before acting. This is especially important for filesystem
  operations, sync targets, workflow paths, and anything outside the repo.
- Do not vendor/copy full third-party repositories into MAIN.
- Install Python dev/test dependencies into a repo-local `.venv`. From a
  fresh checkout/worktree run the platform bootstrap script:
  `scripts\bootstrap_test_env.ps1` (Windows PowerShell) or
  `bash scripts/bootstrap_test_env.sh` (POSIX). Both create `.venv` and
  install the **locked** test set (`constraints-test.txt` — a pinned
  resolve of `.[test]` + its full transitive closure), then **audit** it
  with `pip-audit`; a known CVE fails the bootstrap. Idempotent (no-op if
  `.venv` exists) — pass `-Force` / `--force` to recreate, `-Relock` /
  `--relock` to re-resolve and rewrite the lock, `-NoAudit` / `--no-audit`
  to skip the audit (e.g. offline). Do not install into system Python or
  user-site packages. If `.venv` cannot be used, stop and explain the
  blocker before installing anywhere broader.
- Maintainer-facing dependency CVE response and relock instructions live in
  `docs/maintainers/dependency-security.md`.
- Keep MAIN limited to:
  - custom Koolook nodes
  - wrapper loaders under `forks/`
  - fork tracking docs under `forks/`
- For third-party integrations, update:
  - `forks/forks_manifest.yaml`
  - `forks/THIRD_PARTY.md`
  - `forks/README.md` (if workflow conventions change)

## Test env is pre-authorized (no "go" needed)

Running the bootstrap script above is **pre-approved** — run it
automatically whenever `.venv` is missing or `pytest` isn't importable,
without asking first. It is safe by construction: it installs only the
already-reviewed, pinned `constraints-test.txt` set and `pip-audit`-gates
the result, so it can never silently pull an unreviewed package or version.
This **supersedes the global `/warmup` skill's `MISSING_VENV` hard-stop**:
in this repo, when warmup finds `.venv` missing, silently run the bootstrap
and continue — do not surface it as a blocker.

Two things still need your attention, and only these:

- **Bootstrap exits non-zero** (pip-audit flagged a CVE, or no Python was
  found) — surface *that* as the blocker. A failed audit is the one signal
  that genuinely warrants a human.
- **Dependency changes** — editing the `[test]` extras in `pyproject.toml`
  or regenerating `constraints-test.txt` (`--relock`) — still go through a
  reviewed PR diff. The lock file *is* the review surface; never relock
  silently as a drive-by.

## Radiance fork policy (v2_3_3)

- Package entrypoint: `forks/radiance_koolook/__init__.py`
- Modified nodes live in MAIN and are GitHub-tracked:
  - `forks/radiance_koolook/versions/v2_3_3/nodes_vae.py`
- External sibling checkout (`../ComfyUI-Forks/radiance-v2.3.3-koolook`) is raw reference only.
- Current baseline pin for raw comparison: `f262f47ddfda01ece154bf80c22769b1e4cef795` (the v2.3.3 release commit; the cleanest tag pointing at it is the typo'd `radinace2.3.3`, so we pin by SHA).
- Koolook-original node IDs (`Easy_hdr_VAE_encode`, `Easy_hdr_VAE_decode`) are exposed verbatim via the `SKIP_VERSION_SUFFIX` set in `versions/v2_3_3/__init__.py`. Future Koolook nodes in this fork can opt out the same way; future ports of upstream-named classes should keep the `__koolook_v2_3_3` suffix to avoid colliding with installed copies of upstream Radiance.
- `forks/forks_manifest.yaml` must include `source_ref` + `pinned_commit` + `license` + `license_verified_at`.
- The historical `v1_0_1` fork was removed in v0.1.5 — see `forks/THIRD_PARTY.md` for the rationale and recovery instructions.

## Change management

- When upgrading external forks (v2, v3, etc.), add a new wrapper/version namespace.
- Never rename existing namespaced node IDs that already appear in saved workflows.

## `dev-sync` — copy runtime files into a live ComfyUI install

Fast iteration without cutting a release. Copies runtime files to
`KOLOOK_COMFYUI_DEV_PATH` (set in `.env`; see `.env.example`).

- `python scripts/sync_to_dev.py --scope "<short scope>"` — brief two-line summary; use `-v` for per-file output
- `--dry-run` — preview without touching the target

**Always pass `--scope`** with the same ≤10-word scope summary you'll
put on chat-report line 2. The flag persists the scope to
`<target>/web/_dev_build.json`, which the Kforge Labs sidebar reads on
load and renders as a discreet two-line footer (`dev <sha> · <time>` +
italic `<scope>` underneath). Same identifier the chat report quotes,
but visible in the running ComfyUI itself — closes the gap when the
maintainer can't tell at a glance whether the browser tab is showing
the latest sync.

**Trigger phrase: `dev-sync`** (or "copy those files", "sync dev", "push to dev").
The agent runs the script and reports completion. The script errors
cleanly when the env var is unset; never guess a path.

**User-initiated ONLY — never automatic.** Dev-sync overwrites the
maintainer's live ComfyUI install. Do NOT trigger it as part of any
automated flow:

  • after a commit
  • after a PR merge or `/ship-pr`
  • at session end / wrap-up
  • on hook completion or "task complete" cleanup
  • from any agent skill that doesn't explicitly require it

The maintainer typically runs multiple parallel sessions across worktrees;
an unsolicited sync from one session silently overwrites what another is
reviewing in ComfyUI, and the wrong code gets reviewed. Wait for the
explicit trigger phrase. If unsure whether the user wants a sync, ASK —
don't sync.

**Chat report format** — two lines, copied verbatim from the script's
own header line:

```
<short-sha> - <worktree-name>
<≤10 word summary of the PR / change scope to review>
```

Line 1 is the script's first output line — never invent it, just relay.
Line 2 is the *scope of what to review* — the change in this build, ≤10
words, no list of all files. Examples:

```
8dfb966 - dreamy-jones-40655e
Quick Save + status dot + recovery section

a1b2c3d - feat-install-missing
install-missing button on Nodes row
```

Broader iteration pattern (push/publish gates, what survives re-syncs)
is in [`docs/maintainers/dev-iteration-loop.md`](docs/maintainers/dev-iteration-loop.md).

### Per-module scoped variants — `dev-sync-<module>`

When an automation module under `docs/automations/` iterates on code that
lives in a single subtree (e.g. one fork under `forks/`), a *scoped*
dev-sync wrapper avoids churning the rest of the live install on every
iteration. Same `KOLOOK_COMFYUI_DEV_PATH` target and same chat-report
shape — just a smaller set of paths copied. Scoped sync wrappers only
copy files; restart ComfyUI manually when Python modules need to reload.

Current per-module variants:

- **`dev-sync-audio`** (chat phrase) → `python scripts/sync_to_dev_audio.py`.
  Copies `forks/whatdreamscost_koolook/`, `web/whatdreamscost_koolook/`,
  the global `web/koolook_draft_guard.js`, and root `__init__.py` only.
  It also removes the stale pre-v1.3.9
  `web/whatdreamscost_koolook_v1_3_2/` folder if present so legacy
  workflows do not load two timeline editors.
  Driven by the
  [`docs/automations/LTX-2.3/audio-lipsync/`](docs/automations/LTX-2.3/audio-lipsync/)
  iteration. Leaves `forks/radiance_koolook/` and the rest of the live
  install alone across iterations.

When a new code-touching automation module is added, follow the same
pattern: a `scripts/sync_to_dev_<module>.py` wrapper that imports the
shared infrastructure from `sync_to_dev` and supplies its own
`PATHS` tuple. The chat trigger is the file name minus `sync_to_dev_`
and minus `.py`, with the underscore between module segments turned
into a hyphen: `sync_to_dev_audio.py` → `dev-sync-audio`. Document the
new trigger in this section *and* in the module's `runs/LOOP.md`.

**All scoped variants follow the same user-initiated-only rule** as
plain `dev-sync` (the bullet list above applies verbatim). Never run any
`dev-sync-*` from an automated flow.

## Kforge Labs sidebar — starter preset distribution

The "Kforge Labs" ComfyUI sidebar tab (implemented in
`web/koolook_sidebar.js`) ships a single bundled snapshot file at
`web/starter_preset.json`. On first launch the seeder copies it into
the user's snapshot library directory (the same dir the Snapshot
Save/Load dialog reads) as `starter.json`. The user opens Snapshot →
Load, picks `starter`, clicks → picks + workflows + tags + archive
populated in one go.

To update the distributed starter preset:

1. Inside ComfyUI's Kforge Labs tab, dial in the state you want fresh
   users to see — picks via `+` / `×` / right-click; workflows via the
   Workflows section; tags via right-click → Tags…
2. In the **Tools** row (between Snapshot and the search field), click
   the ↓ button → full snapshot JSON copied to clipboard.
3. Paste into `web/starter_preset.json`, replacing the file contents.
4. Commit + push (does **not** trigger publish — only `pyproject.toml`
   changes merged to `main` do).

Full workflow including reset / verification steps:
[`docs/maintainers/curated-sidebar.md`](docs/maintainers/curated-sidebar.md).
The seeder runs exactly once per browser AND skips users with non-empty
existing picks (legacy curated_defaults.json users), so changing the
shipped starter doesn't disturb anyone's existing state — the new
starter only reaches fresh installs and users who clear browser data.

## Visual verification for design-driven implementation

When implementing a UI change that has a corresponding mockup in
[`docs/designs/`](docs/designs/) (e.g. `snapshot-dialogs.html`,
`sidebar-icon-proposals.html`), **the design file is the spec**. Before
asking the maintainer to review the implementation:

1. Render the implementation in a real browser. Three options, in order
   of preference for agent-driven iteration:
   - **In-tree visual harness** (`docs/designs/_harness/<feature>.html`)
     — mounts the live `web/sidebar/` modules in isolation against a
     static preview server (`python3 -m http.server` from the worktree
     root, port `8765` matches the `design-server` config in
     [`.claude/launch.json`](.claude/launch.json)). The agent can spin
     this up, screenshot, iterate, and tear down without touching the
     maintainer's live ComfyUI install. Use this for component-level
     visual diffing.
   - **`dev-sync` to a live ComfyUI install** — required for any
     integration-level verification (snapshot save round-trip, theme
     variables provided by Comfy's frontend, cross-dialog flows). User-
     initiated only per the dev-sync rule above; never run from an
     automated flow.
   - **Static server on `docs/designs/`** for iterating on the mockup
     itself (`python3 -m http.server` + open the `.html` file).
2. Screenshot the implemented UI at the same states the mockup shows
   (default, hover, expanded, selected, error, etc.).
3. Diff each screenshot against the corresponding card/section in the
   mockup. Either fix any deviation, or call it out explicitly with
   rationale.
4. Only then bring the work to the maintainer for review.

The maintainer should never be the first set of eyes on the rendered
result. Type checks and tests verify code correctness, not visual
fidelity to the design — if a mockup exists for the change you are
implementing, visual verification is a hard gate, not an optional polish
step.

Full workflow including how to add new harness pages and what the
agent-driven path does (and does not) replace:
[`docs/maintainers/visual-harness.md`](docs/maintainers/visual-harness.md).

## Releasing

- The canonical release procedure is `docs/maintainers/releasing.md`. Follow it for every
  `vX.Y.Z` cut — do not invent a one-off flow.
- The per-release tickbox mirror is `.github/ISSUE_TEMPLATE/release_checklist.md`.
  Open one as a tracking issue when starting a release.
- Always validate the Comfy Registry publish against the PR branch
  (`gh workflow run publish.yml --ref release/vX.Y.Z-prep`) before merging.
  A `Failed to validate token` error is almost always a `[tool.comfy] PublisherId`
  mismatch, not a token problem.
