# ComfyUI-Koolook Agent Workflow

## Source-of-truth model

- MAIN repo: `ComfyUI-Koolook`
- External forks root (default): `../ComfyUI-Forks` (relative to MAIN)
- External third-party code must remain outside MAIN.
- Sibling projects (read-only references that live outside MAIN) are referenced via `KOLOOK_*` env vars only. Never hardcode absolute paths with usernames in committed files. Real paths live in `.env` (gitignored); the committed `.env.example` is the public template. See README "Sibling Projects" and `docs/reference/glossary.md`.

## Hard rules

- Do not vendor/copy full third-party repositories into MAIN.
- Keep MAIN limited to:
  - custom Koolook nodes
  - wrapper loaders under `forks/`
  - fork tracking docs under `forks/`
- For third-party integrations, update:
  - `forks/forks_manifest.yaml`
  - `forks/THIRD_PARTY.md`
  - `forks/README.md` (if workflow conventions change)

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

- `python scripts/sync_to_dev.py` — brief one-line summary; use `-v` for per-file output
- `--dry-run` — preview without touching the target
- Restart ComfyUI to load the new code.

**Trigger phrase: `dev-sync`** (or "copy those files", "sync dev", "push to dev").
The agent runs the script and reports completion. The script errors
cleanly when the env var is unset; never guess a path.

**Chat report format** — always include the build id from the script's output:

```
dev-sync @ <short-sha> v<version> — <≤10 word summary>. Restart.
```

The summary names the *thing to look at* after restart, not a list of
all changes. Keep it ≤10 words. Examples:

```
dev-sync @ 8dfb966 v0.2.0 — Quick Save + status dot + recovery section. Restart.
dev-sync @ a1b2c3d v0.2.1 — install-missing button on Nodes row. Restart.
```

Broader iteration pattern (push/publish gates, what survives re-syncs)
is in [`docs/maintainers/dev-iteration-loop.md`](docs/maintainers/dev-iteration-loop.md).

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

## Releasing

- The canonical release procedure is `docs/maintainers/releasing.md`. Follow it for every
  `vX.Y.Z` cut — do not invent a one-off flow.
- The per-release tickbox mirror is `.github/ISSUE_TEMPLATE/release_checklist.md`.
  Open one as a tracking issue when starting a release.
- Always validate the Comfy Registry publish against the PR branch
  (`gh workflow run publish.yml --ref release/vX.Y.Z-prep`) before merging.
  A `Failed to validate token` error is almost always a `[tool.comfy] PublisherId`
  mismatch, not a token problem.
