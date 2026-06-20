# GitHub Copilot — repo orientation

Thin pointer. The canonical agent context lives in [`CLAUDE.md`](../CLAUDE.md)
at the repo root — read that first for source-of-truth conventions, dev-sync
rules, release rules, fork policy, and trigger phrases. This file only carries
the high-stability orientation that helps Copilot place a request before it
follows links.

## What this repo is

ComfyUI-Koolook is a custom-node pack for ComfyUI. Two shipped surfaces:

1. **Kforge Labs sidebar** — the main user-facing experience. A ComfyUI
   sidebar tab that saves favorite nodes, workflows (with subdirectories +
   archive + tags + reusable modules), and full-state snapshots to a
   configurable on-disk library.
   - Front-end entry: [`web/koolook_sidebar.js`](../web/koolook_sidebar.js);
     module bodies under [`web/sidebar/`](../web/sidebar/).
   - Back-end snapshot/preset routes: [`koolook_routes.py`](../koolook_routes.py)
     (registers `/koolook/presets/*` aiohttp routes on ComfyUI's PromptServer).
   - Bundled visual onboarding guide: [`web/guide/index.html`](../web/guide/index.html)
     (also opened in-app via the sidebar's Tools-row Help button).

2. **Koolook custom nodes** — Python node classes at the repo root, each
   exposing the standard ComfyUI node contract (`INPUT_TYPES`, `RETURN_TYPES`,
   `RETURN_NAMES`, `FUNCTION`, `CATEGORY`):
   - [`k_easy_wan22_prompt.py`](../k_easy_wan22_prompt.py) → `EasyWan22Prompt`
     (reads [`config.json`](../config.json) at import time — edits to
     `config.json` need a ComfyUI restart).
   - [`k_easy_resize.py`](../k_easy_resize.py) → `EasyResize_Koolook`
     (deprecated alias `EasyResize` still registered for back-compat).
   - [`k_ai_pipeline.py`](../k_ai_pipeline.py) → `EasyAIPipeline` (paired
     with front-end widgets in [`web/ai_pipeline.js`](../web/ai_pipeline.js)).
   - [`k_easy_image_batch.py`](../k_easy_image_batch.py) → `easy_ImageBatch`.
   - [`k_easy_track.py`](../k_easy_track.py) → `KoolookLoadCameraPosesAbsolute`.
   - [`k_video_combine.py`](../k_video_combine.py) → `Easy_VideoCombine`
     (runtime subclass of VHS `VideoCombine`; self-skips if VHS not installed).
   - Radiance VAE wrappers under
     [`forks/radiance_koolook/versions/v2_3_3/`](../forks/radiance_koolook/versions/v2_3_3/)
     → `Easy_hdr_VAE_encode`, `Easy_hdr_VAE_decode`.

[`__init__.py`](../__init__.py) imports each module's `NODE_CLASS_MAPPINGS` /
`NODE_DISPLAY_NAME_MAPPINGS` and exposes the merged set plus `WEB_DIRECTORY`.

## Things that bite if you don't know them

- **Backward compatibility is opt-in for Koolook-created nodes only** (the root
  `k_*.py` custom nodes). For those, make the cleanest change by default —
  rename/drop/reorder IDs, inputs, and outputs freely, no aliases or `_v2`
  suffixes — unless the maintainer says `check backward compatibility`.
  **Fork nodes are different: anything under `forks/` keeps full back-compat
  discipline by default** (never rename a fork ID in saved workflows, version
  via new suffix/namespace, keep the `__koolook_vX_Y_Z` collision suffix). See
  [`docs/maintainers/node-versioning.md`](../docs/maintainers/node-versioning.md)
  and [`CLAUDE.md`](../CLAUDE.md) → *Change management*.
- **Third-party code stays out of MAIN.** Modified fork code lives under
  `forks/<name>/versions/<vX_Y_Z>/`; raw upstream checkouts live in
  `../ComfyUI-Forks/`. See [`forks/README.md`](../forks/README.md) and
  [`forks/THIRD_PARTY.md`](../forks/THIRD_PARTY.md).
- **`dev-sync` is user-triggered only.** Never run it automatically (e.g.
  after a commit, after a merge, on session end). The maintainer runs
  parallel worktrees and an unsolicited sync overwrites whatever they're
  reviewing in their live ComfyUI. Wait for the explicit phrase. Full rules
  in [`CLAUDE.md`](../CLAUDE.md).
- **UI work needs visual QA.** See [`AGENTS.md`](../AGENTS.md) — open the
  changed page in a browser, take a screenshot, verify before claiming done.
- **Three independent version axes** (pack version / fork wrapper version /
  upstream pinned commit) — see [`docs/reference/versioning.md`](../docs/reference/versioning.md)
  before any version bump.

## Where to look next

- [`CLAUDE.md`](../CLAUDE.md) — full project conventions and protocols
- [`docs/maintainers/`](../docs/maintainers/) — release procedure, dev-sync
  loop, registry API, sidebar starter preset, workflows-sidebar reference
- [`docs/reference/`](../docs/reference/) — glossary, versioning model
- [`docs/user_guide/`](../docs/user_guide/) — per-node end-user docs
- [`README.md`](../README.md) — public landing page + install
- [`CHANGELOG.md`](../CHANGELOG.md) — what changed, in detail
