# Changelog

All notable changes to this project should be documented in this file.

The format is inspired by Keep a Changelog and SemVer.

## [Unreleased]

### Added
- **`Easy_VideoCombine`: loader-friendly video path outputs.** The node
  now keeps its original `Filenames` output and appends clean string
  outputs for the final video's full path, directory, filename, and JSON
  sidecar path, so a rendered clip can wire directly into `Easy Load Video
  (Koolook)` for a second workflow stage without parsing VHS's mixed
  JSON/video list.

### Fixed
- **Node registry survives one broken or missing module.** The root
  `__init__.py` now imports each node group independently and installs the
  Kforge Labs snapshot/preset routes (`/koolook/presets/*`) regardless of
  node-import results. Previously a single missing or failing node module
  raised at import time and zeroed *every* Koolook node **and** the snapshot
  routes at once (sidebar disk browse/save returning 404). A new guard test
  keeps every root `k_*.py` listed in dev-sync's `RUNTIME_PATHS`.
- **`loop-audio`: Director card reads execution-active values.** The
  audio-lipsync card now reads Director widgets from the saved Director
  widget order before falling back to generic input-name lookup, fixing
  false `model-gen` / unknown-fps cards when saved workflows include
  non-input widgets such as `epsilon`. `relay_overrides` is now reported
  only when the active Koolook Director's `relay_overrides` socket is
  actually wired; an unwired `RELAY_OVERRIDES` note no longer appears as
  an active knob on the card, log, metadata, or notes.
- **`Easy_LoadVideo`: direct `video_path` handoff.** The loader now
  accepts a complete video file path in `input_path` when `video` is
  empty, so `Easy_VideoCombine.video_path` can wire straight into
  `Easy_LoadVideo.input_path` without needing a separate filename field.
  Existing local video files bypass VHS's stricter path-wrapper
  validation and are passed directly to VHS's shared decode path. Wrapped
  multiline full paths from text nodes are joined back together before
  loading.
- **`Easy_LoadVideo`: two-line path/name input.** When `input_path` is
  empty, the loader now accepts a multiline `video` string whose first
  line is the directory and second line is the filename, matching the
  path/name text-block pattern used in automation workflows.
- **`dev-sync-audio`: stale Director web extension cleanup.** The scoped
  audio sync now removes the old `web/whatdreamscost_koolook_v1_3_2/`
  folder after the stable v1.3.9 web-extension rename, preventing legacy
  workflows from loading two identical timeline editors in dev installs.
- **LTX Director timeline images persist after canvas/setup switches.** The
  timeline editor keeps saved workflow payloads light by omitting preview-only
  image blobs, but now rebuilds image previews from the persisted `imageFile`
  path when the node is restored. It also strips legacy inline preview blobs
  from restored timeline data when a saved file path exists, preventing old
  image-heavy setups from re-triggering Comfy's draft-quota error.
- **Comfy workflow draft quota guard for large LTX timeline workflows.** The
  LTX Director frontend now strips preview-only media blobs from timeline
  serialization and catches browser quota failures on Comfy's own
  `Comfy.Workflow.Drafts` / `Comfy.Workflow.DraftOrder` writes, evicting only
  the oldest saved Comfy draft before retrying and showing a visible warning if
  the browser store is still full. This documents and extends the
  v0.3.6 stable-draft-ID lesson to large workflows imported directly from disk,
  which bypass the Koolook sidebar load path.
- **LTX Director transcript timing preserves image timelines.** The
  `audio_transcript_json` hook now rejects empty phrase lists and keeps existing
  timeline image segments intact, using the transcript only to update Prompt
  Relay `local_prompts` / `segment_lengths`. When no timeline image segments
  exist, it still builds speech/pause segments for older stripped-down flows.
- **Audio-lipsync run evidence no longer carries local machine paths.**
  Checked-in run workflow snapshots now replace user-home, working-folder, and
  project-drive paths with placeholders so the reproducibility evidence stays
  useful without publishing workstation-specific paths.
- **`loop-audio`: run numbering now respects the log.** The audio-lipsync
  loop now chooses the next run number from both existing `run-NNN_*`
  folders and `runs/log.md`, preventing a newly saved snapshot from reusing
  an already logged number when a folder is absent or a run was log-only.
- **`loop-audio`: fork-touching captures are always retained.** Every
  `loop-audio` invocation now writes the log row and matching
  `run-NNN_<label>/` evidence folder; scratch renders should be skipped
  explicitly with "no log" / "don't log this" before capture.
- **Snapshot drift on duplicate installs no longer silently corrupts the
  workflow store.** Closes
  [#161](https://github.com/malkuthro/ComfyUI-Koolook/issues/161) and
  [#162](https://github.com/malkuthro/ComfyUI-Koolook/issues/162) as a paired
  fix:
    - **Duplicate-install guard (#162).** When the Comfy Registry/Manager
      install (`custom_nodes/koolook/`) and a `git clone` checkout
      (`custom_nodes/ComfyUI-Koolook/`) co-exist, every ComfyUI boot would
      race two parallel plugins for the same `/koolook/presets/*` routes,
      the same Kforge Labs sidebar tab, and the same
      `/userdata/koolook_workflows.json` file — silently reverting workflows
      to a stale state on every restart. New
      [`koolook_install_guard.py`](koolook_install_guard.py) scans
      `custom_nodes/` siblings for `koolook_routes.py` markers at import
      time, prints a critical log naming both paths + their pyproject
      versions, and registers nothing (no nodes, no routes, no
      `WEB_DIRECTORY`) when this install is the alphabetical loser. The
      scan is fail-safe by construction — an unreadable sibling directory
      or a non-UTF-8 `pyproject.toml` is skipped rather than allowed to
      abort the plugin import, so the guard can never itself take Koolook
      offline. A client-side fallback in
      [`web/sidebar/extension_guard.js`](web/sidebar/extension_guard.js)
      catches the residual case where an older sibling without the backend
      guard still ships its `koolook_sidebar.js` to the browser, sets a
      `window` sentinel on first load, and skips the second
      `app.registerExtension` call with a critical toast.
      [`.env.example`](.env.example), the
      [dev-iteration loop doc](docs/maintainers/dev-iteration-loop.md), and
      [`scripts/sync_to_dev.py`](scripts/sync_to_dev.py) now point at
      `custom_nodes/koolook/` (the Registry-derived path) so dev-sync
      overwrites the Manager install in place instead of spawning a parallel
      `ComfyUI-Koolook/` folder.
    - **Boot-time tracked-snapshot drift guard (#161).** New
      `detectBootDrift()` in [`web/sidebar/snapshot.js`](web/sidebar/snapshot.js)
      reads the tracked named snapshot file at session start, fingerprints
      it the same way the live state is fingerprinted, and flags drift on
      mismatch. While drifted, `_autosaveSubdir()` routes periodic +
      pre-load autosaves to `_unsaved_autosave/` instead of
      `<preset>_autosave/` — so a corrupt live state can never masquerade
      as a "newer recovery" for the named snapshot. `getSnapshotStatus()`
      gains a `"drifted"` precedence (above `"saved"`) so the sidebar pill
      flips to a warning state (`· drifted (reload?)`) with a recovery-
      instructions tooltip. The flag clears automatically on the next
      `markStateSaved()` (Save / Quick Save / Load), so a deliberate
      realignment retires the warning without the user having to dismiss
      it.
- **`EasyAIPipeline`: preview resolves connected text-name builders.**
  The path preview buttons now evaluate common connected text nodes such as
  `Text Multiline` and `Text Concatenate` instead of reading the concatenate
  node's delimiter widget (`_`) as the whole shot name. This fixes previews
  like `.../_/v003/__v003.%04d.exr` when `shot_name` is wired, and the
  preview version formatter now matches Python: `003` becomes `v003`, while
  an existing `v003` stays `v003`.
- **`EasyAIPipeline`: path preview follows subgraph-routed version inputs.**
  The "Get output directory/file path" buttons no longer reuse a stale
  converted-widget `version` value when the live value is wired through
  KJ Set/Get nodes and a ComfyUI subgraph. The browser preview resolver
  now crosses subgraph outputs, maps subgraph inputs back to their host
  node widgets/links, evaluates `Easy_Utility`'s `int_to_padded_string`
  output, and reports "Cannot preview" instead of silently falling back
  to stale local widget state when a connected value is too complex.
- **`EasyAIPipeline` / `Easy_VideoCombine`: Run no-op when `version` widget
  is wired or carries a stale value.** PR #180 introduced a new STRING
  `version` widget in the same widget-slot the old INT lived in. Workflows
  saved before #180 replay an INT into the new slot; widgets converted to
  inputs leave the widget object with `value: undefined`; dict-serialized
  Easy_VideoCombine workflows simply omit the new key. Any of those states
  combined with ComfyUI-Custom-Scripts (pysssss) installed crashed
  `graphToPrompt` — pysssss's `presetText.js` patches every STRING widget's
  `serializeValue` to run `value.replace(...)` for `{variable}` substitution,
  and `.replace` on `undefined` aborts the whole queue ("Cannot read
  properties of undefined (reading 'replace')"; Run does nothing). Fixed by
  trapping `widget.value` reads/writes on the `version` widget via a
  property descriptor so the value is always coerced to a string
  ([`web/ai_pipeline.js`](web/ai_pipeline.js),
  [`web/easy_video_combine.js`](web/easy_video_combine.js)). Survives
  widget-to-input conversion, stale workflows, and downstream widget patches.

### Changed
- **Locked + audited test environment; bootstrap pre-authorized.**
  `scripts/bootstrap_test_env.{sh,ps1}` now install the `[test]` extras
  against a committed lock (`constraints-test.txt` — a pinned resolve of the
  extras + their full transitive closure) and gate the result with
  `pip-audit`; a known CVE fails the bootstrap. New flags: `--relock` /
  `-Relock` (re-resolve + rewrite the lock) and `--no-audit` / `-NoAudit`
  (skip the audit, e.g. offline). Because the bootstrap can now only install
  an already-reviewed, pinned, audited set, it is **pre-authorized to run
  automatically** — `CLAUDE.md` documents that this supersedes the global
  `/warmup` `MISSING_VENV` hard-stop, while dependency *changes* still land
  as a reviewed `pyproject.toml` / `constraints-test.txt` diff. The lock +
  audit govern local and agent bootstraps; CI still installs the minimal
  `pytest` + `aiohttp` subset it needs directly (not yet on the lock/audit
  path). New guard test
  [`tests/scripts/test_bootstrap_constraints.py`](tests/scripts/test_bootstrap_constraints.py)
  keeps the lock in sync with the declared extras.
- **Dev-sync scripts now only copy files.** `sync_to_dev.py` and the
  scoped `dev-sync-audio` wrapper no longer call the ComfyUI-Manager
  reboot endpoint and no longer expose `--no-restart` / `--restart-url`;
  restart ComfyUI manually after Python changes need to be re-imported.
- **`loop-audio` card redesigned around a strict two-source rule** (issue
  [#177](https://github.com/malkuthro/ComfyUI-Koolook/issues/177)). The
  audio-lipsync card now only reads (a) the five tracked
  `Text Multiline` nodes and (b) the `LTXDirector__koolook` node's
  own widget values + input wiring. The previous SAMPLER block
  (`BasicScheduler` / `KSamplerSelect` / `RandomNoise` / `CFGGuider`
  scrapes), FORK STATE block (`_dev_build.json` + `git status`), and
  INERT warning are gone — they were either widget scrapes from outside
  the two source families or duplicates of the Koolook Director's own
  state. New sections: `BASE · LOCKED` (epsilon · Audio src · Working
  folder, path-wrapped on separator boundaries) and `BASE · SCENE`
  (flat-left labels with one indented child — `Segments (N)` parent,
  spelled-out `1) 0 to 5 seconds` per segment, aggregated
  `Prompt / Audio / Keyframe` coverage rows). Audio src is derived
  structurally from the five-state machine (director presence ×
  audio_vae link × use_custom_audio × audioSegments count) — never from substring matches
  on prompt text. `notes.md` and `runs/log.md` follow the same source
  rule; the log's Phase 1 / Phase 2 / Custom audio columns are replaced
  by `Audio src` + `Segments`. `loop_audio.py`'s `.env` discovery now
  falls back to the main repo when running from a worktree (same as
  `scripts/make_card.py`). Fixed the upstream-LTXDirector widget index
  bug — the saved workflow preserves the legacy widget order even after
  the Koolook fork reordered the schema, so `DIRECTOR_WIDX` matches the
  serialised order, not the schema declaration.
- **Automations restructured around modules instead of models.**
  `docs/automations/` now pivots on one folder per *task* (each its own
  workflow + iteration loop + findings) rather than one folder per model.
  The existing `LTX-2.3/{README,findings,handoff-checklist}.md` content
  moved into `LTX-2.3/base-1step/` (single-stage render task) and the new
  sibling `LTX-2.3/audio-lipsync/` (audio-file lip-sync task) was created.
  `docs/automations/{README,CHEATSHEET}.md` updated to reflect the new
  layout; `docs/automations/CONVENTIONS.md` unchanged.
- **`docs/investigations/` removed.** Its two children migrated to their
  natural homes: narrative content into the new automation modules under
  `docs/automations/LTX-2.3/{base-1step,audio-lipsync}/`; the modified
  upstream Python code into the new `forks/whatdreamscost_koolook/`
  fork (see *Added*). The retired `scripts/sync_investigation_patches.py`
  and its `KOLOOK_WHATDREAMSCOST_PATH` env-var section in `.env.example`
  go with it — iteration now uses the standard `dev-sync` flow against
  the in-repo fork.
- **Glossary updated.** The *Investigation folder*, *Investigation patches*,
  *JSON file folder*, and `sync_investigation_patches.py` entries in
  [`docs/reference/glossary.md`](docs/reference/glossary.md) are replaced
  by *Automation module*, *Backstory*, *Working folder*, and *Card*; the
  *Loop* and *Run* entries are rewritten to point at the new structure.

### Added
- **Koolook Director accepts an audio transcript for speech timing.** The
  Koolook LTX Director gained an `audio_transcript_json` input that converts
  phrase-timed transcript JSON into Director `timeline_data`, `local_prompts`,
  and `segment_lengths` internally before Prompt Relay conditioning runs.
- **`forks/whatdreamscost_koolook/v1_3_9/` — updated Koolook LTX Director
  fork.** Upgrades the Koolook Director to upstream WhatDreamsCost-ComfyUI `3b65410`
  (pyproject/README version `1.3.9`; no `v1.3.9` git tag at fork time).
  It carries forward the Koolook `relay_overrides` multiline JSON input and
  per-segment Prompt Relay sigma patch, while keeping upstream's v1.3.9
  audio latent fixes. New workflows use stable node ID
  `LTXDirector__koolook`; existing workflows saved with
  `LTXDirector__koolook_v1_3_2` still load through a compatibility alias
  backed by v1.3.9. The original v1.3.2 source remains on disk for
  attribution, review, and rollback.
- **Easy Load Video (`Easy_LoadVideo`).** Path-aware sibling of VHS's
  `Load Video Path` node. It exposes a split `input_path` + `video`
  field layout so a workflow can keep the source folder fixed while
  changing only the clip name. Absolute `input_path` values load
  directly from disk; relative values resolve under ComfyUI's `input/`
  directory. Leaving `input_path` empty passes `video` through to VHS
  unchanged, preserving full-path/URL workflows. Implemented as a thin
  subclass in [`k_video_load.py`](k_video_load.py), with pure helper
  coverage in [`tests/nodes/test_easy_video_load.py`](tests/nodes/test_easy_video_load.py)
  and usage docs in
  [`docs/user_guide/nodes/koolook_video/easy_load_video.md`](docs/user_guide/nodes/koolook_video/easy_load_video.md).
- **Audio-lipsync reading-graph docs.** Three new files alongside
  [`docs/automations/LTX-2.3/audio-lipsync/`](docs/automations/LTX-2.3/audio-lipsync/):
  [`reading-graph.html`](docs/automations/LTX-2.3/audio-lipsync/reading-graph.html)
  is a visual flow diagram (dark theme, color-coded source bands) showing
  every value's path from canvas node to card section;
  [`reading-graph.schema.yaml`](docs/automations/LTX-2.3/audio-lipsync/reading-graph.schema.yaml)
  is the machine-readable version of the same graph (source families,
  widget indices, audio state machine, card-section field map,
  invariants); [`CHEATSHEET.md`](docs/automations/LTX-2.3/audio-lipsync/CHEATSHEET.md)
  refreshed against the new card design (chat triggers, required canvas
  shape, card source rule, the five Audio src states, snapshot folder
  layout, settings map).
- **`forks/whatdreamscost_koolook/v1_3_2/` — Koolook fork of WhatDreamsCost-ComfyUI's
  `LTXDirector`** (upstream `e81223a`, GPL-3.0). Two upstream files modified
  (`ltx_director.py` adds a `relay_overrides` multiline-JSON widget;
  `prompt_relay.py` uses the Prompt-Relay paper's per-segment σ formula
  instead of upstream's length-independent constant) plus the
  unmodified `patches.py` vendored verbatim because the modified file
  imports from it. Registered as
  `LTXDirector__koolook_v1_3_2` (display name *"LTX Director (Koolook
  v1.3.2)"*) so it coexists with an installed copy of upstream
  WhatDreamsCost-ComfyUI in the node picker. Drives the
  [`docs/automations/LTX-2.3/audio-lipsync/`](docs/automations/LTX-2.3/audio-lipsync/)
  iteration loop. License attribution + change log in
  [`forks/THIRD_PARTY.md`](forks/THIRD_PARTY.md); pin metadata in
  [`forks/whatdreamscost_koolook/versions/v1_3_2/UPSTREAM_PIN.yaml`](forks/whatdreamscost_koolook/versions/v1_3_2/UPSTREAM_PIN.yaml)
  and [`forks/forks_manifest.yaml`](forks/forks_manifest.yaml).

## [0.3.7] - 2026-05-23

- **Workflow validator (`scripts/validate_workflow.py`).** stdlib-only CLI
  that checks a ComfyUI workflow JSON for internal consistency: declared-
  vs-referenced links, slot index bounds and type matches, endpoint cross-
  references, `last_node_id`/`last_link_id` bounds, and duplicate IDs.
  Catches the class of bugs that come from hand-editing workflow JSONs
  (e.g. agent-driven node re-wiring when building a slim variant from a
  Director workflow). Defensive against malformed-but-valid JSON — every
  shape assumption is guarded so bad input produces problem lines, never
  an uncaught exception. Exit codes: `0` clean, `1` problems on stderr,
  `2` file missing or bad JSON. Covered by
  [`tests/scripts/test_validate_workflow.py`](tests/scripts/test_validate_workflow.py)
  (24 tests). See
  [`docs/automations/CONVENTIONS.md`](docs/automations/CONVENTIONS.md) §11
  for usage and triggers.

### Docs
- **User support and bug-reporting path.** The GitHub landing page now has
  troubleshooting and support sections, the in-app Help guide links to a
  support checklist, and GitHub issues include a bug-report template that
  asks for ComfyUI/Koolook versions, install method, logs, browser console
  errors, screenshots, and reproduction steps.
- **LTX 2.3 findings — spatial upscaler architecture & stage-2-only path.**
  Two new sections in
  [`docs/automations/LTX-2.3/findings.md`](docs/automations/LTX-2.3/findings.md):
  a locked-in *Spatial upscaler architecture* block (×2 hardcoded factor,
  brief enters only via `Director → guide_data → DirectorGuide`, attention
  scales `(W × H × T)²`) and an open *Stage-2-only refine path* block
  with the recipe, initial knob settings, and three hypotheses still
  pending validation across runs.

### Fixed
- **Workflow archive stays collapsed during search.** The sidebar search
  state now preserves the archive folder's collapsed/expanded intent instead
  of forcing archived workflows open while filtering. This keeps normal
  workflow search focused on active entries unless the archive was explicitly
  opened.
- **Snapshot Load list now reflects disk renames.** The snapshot library
  still validates each JSON file and reads its counts/metadata, but the
  Load dialog displays and sorts root snapshot rows by the current
  filename stem instead of the embedded `name` field. Renaming
  `Shot_A.json` to `Shot_B.json` in Finder/Explorer now shows `Shot_B`
  in the sidebar without editing the JSON contents.

## [0.3.6] - 2026-05-18

### Fixed
- **Kforge Labs workflow loads no longer churn Comfy draft ids.** Loading the
  same saved workflow from the sidebar now uses a stable temporary workflow id
  derived from its sidebar path/name, so Comfy's browser-side
  `Comfy.Workflow.Drafts` cache replaces the same draft entry instead of
  accumulating a new large draft on every load. This reduces the chance of
  hitting repeated `Failed to save workflow draft` toasts from browser storage
  quota exhaustion after restoring or loading archived workflows.

## [0.3.5] - 2026-05-18

### Added
- **Easy Video Combine (`Easy_VideoCombine`).** A path-aware sibling
  of Kosinkadink/ComfyUI-VideoHelperSuite's `Video Combine` node.
  When the `filename_prefix` field carries an absolute path
  (e.g. `E:/renders/shot01/v003`), the node bypasses ComfyUI's
  output-directory sandbox and writes the video, metadata PNG, and
  any audio mux directly to that location. Relative prefixes behave
  identically to upstream (sandboxed under ComfyUI's `output/`).
  Adds an optional `create_path_if_missing` BOOLEAN (default
  `False`) — when on, the parent directory is auto-created; when
  off, a missing parent surfaces as a clear error so typos don't
  silently spawn directories.

  Implemented as a thin subclass of VHS's `VideoCombine` (~60 lines
  of Python, no upstream code copied — see
  [`k_video_combine.py`](k_video_combine.py)) that scoped-patches
  `folder_paths.get_save_image_path` for the duration of one
  `combine_video()` call. All encoding, format widgets, audio mux,
  batch-manager, and progress-bar behavior comes from VHS unchanged.
  If VHS isn't installed the node simply doesn't register, with a
  one-line stderr notice at module load.

  The `os.path.isabs(filename_prefix)` discrimination pattern is
  borrowed from spacepxl/ComfyUI-HQ-Image-Save's `SaveEXR` (MIT) —
  overload the existing field for both modes rather than adding a
  new input pin. See
  [`forks/THIRD_PARTY.md`](forks/THIRD_PARTY.md) for attribution
  and [`docs/user_guide/nodes/koolook_video/easy_video_combine.md`](docs/user_guide/nodes/koolook_video/easy_video_combine.md)
  for the user-facing guide.

  Also adds an optional **`output_directory`** STRING input (default
  empty). When set, `filename_prefix` is treated as just the
  filename root and `output_directory` carries the path — so you
  can change the name without retyping the directory across many
  renders. Absolute `output_directory` writes there directly;
  relative `output_directory` joins under ComfyUI's `output/`. The
  overloaded-absolute-`filename_prefix` mode is preserved
  unchanged for users who prefer a single field.
- **Easy AI Pipeline (`EasyAIPipeline`): `no_subfolders` toggle.** When on,
  the node writes directly into `base_directory_path` instead of appending
  `shot_name/ai_method[/vNNN]` subfolders underneath. The base folder is
  still created on the fly if missing, so you can point it at a not-yet-
  existing directory. Default is off, so saved workflows render the same
  path as before.
- **Easy AI Pipeline: tooltips on every input.** Hovering any field in the
  node now explains what it controls and how it interacts with the other
  fields/toggles (e.g. that `ai_method` becomes both a filename segment
  and a subfolder, that `disable_versioning` drops the `vNNN` segment
  everywhere, that `enable_overwrite` only blocks an existing *file* —
  not an existing directory).
- **AI-assisted automation workflow framework.** Adds
  [`docs/automations/`](docs/automations/) as the canonical home for
  repeatable image/video iteration loops, with shared conventions,
  a quick-reference cheatsheet, an LTX 2.3 model-specific workflow,
  handoff checklist, promoted findings, and the companion
  [`docs/investigations/ltx-director-4k-transitions.md`](docs/investigations/ltx-director-4k-transitions.md)
  investigation. The new [`scripts/make_card.py`](scripts/make_card.py)
  and [`scripts/watch_cards.py`](scripts/watch_cards.py) helpers generate
  and monitor compact review cards from ComfyUI run metadata so experiment
  results can be compared across runs without re-reading full workflow JSON.
- **Project-local test bootstrap helpers.** Adds macOS/Linux and Windows
  scripts for creating the local test environment, plus `pip-audit` in the
  optional `test` dependency set and a repository line-ending policy. The
  POSIX helper accepts `PYTHON=...` and falls back to `python3` when `python`
  is not available on `PATH`.

### Changed
- **Workflow right-click menu shortened for large libraries.** The menu no
  longer lists every existing folder as a move destination, which could push
  useful actions off-screen when many projects/modules existed. Moving a
  workflow into an existing folder is now handled by drag-and-drop; the
  right-click menu still supports creating a new directory/subdirectory and
  moving the workflow there in one step.
- **Easy AI Pipeline: no more dangling underscores in filenames.** The
  filename builder now joins only the non-empty pieces of
  `shot_name` / `ai_method` / `vNNN` with `_`, so leaving `ai_method`
  blank yields `RTX-upscale.%04d.exr` instead of the previous
  `RTX-upscale_.%04d.exr`. Same rule applies to the subfolder build, so
  an empty `ai_method` no longer produces a phantom directory level. The
  preview shown by the `Get output file path` button now matches.

### Fixed
- **Kforge Labs snapshot/workflow restore no longer hydrates from stale browser cache.**
  The sidebar's startup read of `/userdata/koolook_workflows.json` now uses
  a cache-busted `no-store` request, and snapshot-library reads do the same
  for mutable preset/list/settings endpoints. This fixes the severe case
  where the browser returned an older 5-workflow `/userdata` response even
  though the on-disk file and a cache-busted fetch contained the correct
  59-workflow snapshot state; the status pill could appear to track the
  right snapshot while the workflow tree rendered stale data and periodic
  autosave captured that bad state.
- **Easy AI Pipeline: trailing-slash on base path no longer creates a
  phantom `undefined/` folder.** Typing `n:\foo\bar\` (trailing
  backslash) into `base_directory_path` used to leak a trailing `/` onto
  the `output_directory` output. Downstream save nodes split on `/`,
  stringified the resulting empty tail as `"undefined"`, and wrote into
  `…/bar/undefined/<shot_name>/…` on disk. The node now strips any
  trailing separator from `output_directory` (with a guard so drive
  roots like `n:/` survive intact), so `n:\foo\bar\` and `n:\foo\bar`
  resolve identically. The `Get output directory path` preview already
  showed the clean form — Python now matches. The same cleanup also treats
  literal frontend sentinels like `"undefined"` / `"null"` / `"None"` as
  empty text in path fields and strips those sentinel words when they arrive
  as whole path components from connected upstream values, matching
  `Easy_VideoCombine`'s defensive normalization.
- **Easy AI Pipeline: broader paste-input hardening on
  `base_directory_path`.** A new `_normalize_base_path` helper strips
  surrounding whitespace, one matched pair of surrounding quotes
  (`"..."` or `'...'` — `Shift+Right-click → Copy as path` in Windows
  Explorer wraps paths in double quotes), and any number of trailing
  separators (mixed `/` and `\` accepted). Drive roots like `C:\` and
  `n:/` are preserved. A new test suite at
  [`tests/nodes/test_easy_ai_pipeline.py`](tests/nodes/test_easy_ai_pipeline.py)
  pins both the helper and the end-to-end `generate_pipeline` behavior
  with parametrised paste-variant cases (49 tests covering 11 realistic
  paste shapes the maintainer has actually seen), so the `undefined/`
  phantom-folder bug can't regress.
- **Easy AI Pipeline: absolute `shot_name` / `ai_method` can no longer
  escape `base_directory_path`.** `os.path.join`'s "last absolute path
  wins" rule meant a stray leading `/` (typo) or a pasted full path
  (`C:/Windows/junk`) in either field would replace the user's intended
  base entirely — `n:/safe` + `shot_name="/oops"` was writing to
  `C:/oops` instead of `n:/safe/oops`. A new `_sanitize_segment` helper
  (lstrip → splitdrive → lstrip) strips drive prefix and leading
  separators so segments can only be joined onto the base, never replace
  it. Applied symmetrically to both directory build and filename build.
  Raw `shot_name` is preserved in the return tuple for downstream nodes
  that use it as a label. 16 new tests cover the helper and the escape
  vectors (`/oops`, `\oops`, `///oops`, `C:/Windows/junk`, `/etc/passwd`
  in both `shot_name` and `ai_method`). The JS preview's filename
  builder picked up the same sanitization plus an empty-base /
  `no_subfolders=true` corner-case fix (`/name.exr` → `name.exr`).
- **Easy AI Pipeline: JS preview and Python runtime now agree on
  drive-prefixed `shot_name` / `ai_method` segments on every host.**
  Segment sanitization now strips Windows-style drive prefixes with
  Windows path semantics even when ComfyUI is running on POSIX, so
  `C:/Windows` previews and executes as `Windows` consistently. Base
  directory drive roots such as `C:/` and `n:/` remain preserved.
- **Easy AI Pipeline: `no_subfolders=true` now truly produces flat
  output, even when `shot_name` contains embedded separators.** An
  upstream node feeding `shot_name="job/shot"` used to silently
  re-create subfolders via the filename concat — output was
  `base/job/shot.ext` instead of the flat `base/job_shot.ext` the
  toggle promises. The filename build now flattens any `/` or `\` in
  `shot_name` and `ai_method` to `_` (filesystems can't have separators
  in filenames anyway). Directory build with `no_subfolders=false`
  still uses the separators so users organizing into nested project
  hierarchies aren't affected.
- **Easy AI Pipeline: `no_subfolders=true` keeps the version folder.**
  Previously the toggle stripped *everything* under `base` —
  `shot_name`, `ai_method`, AND the `v###` version subfolder — which
  meant versioned outputs all collided in the same flat directory.
  New behavior: `no_subfolders` only drops `shot_name` and `ai_method`
  from the directory (they still appear in the filename). The version
  folder is added when `disable_versioning` is off, so versioned
  outputs stay organised under `base/v###/`. Truly flat output (no
  version folder either) requires `disable_versioning=true` AND
  `no_subfolders=true`, matching VFX convention. Tooltip on
  `no_subfolders` and `base_directory_path` rewritten to describe
  this. All `vNNN` references in tooltips updated to `v###` per
  Nuke-style convention.
- **Easy AI Pipeline: newlines / tabs from upstream Text Multiline
  widgets no longer leak into the path.** A new `_strip_control_chars`
  helper runs first in both `_normalize_base_path` and
  `_sanitize_segment`, plus explicitly on `extension` in
  `generate_pipeline`. Removes `\n`, `\r`, `\t` defensively — these
  can't legally appear in any filesystem path, but an upstream WAS
  `Text Multiline` node (or anything with a stray paragraph break)
  used to ship them through, silently breaking the save downstream
  while rendering the preview as visually broken multi-line output.
  Mirrored in the JS preview's `normalizeBasePath` / `sanitizeSegment`
  / extension handling.
- **Easy AI Pipeline: extension widget now strips ALL whitespace.**
  A single trailing space on `.%04d.exr` (easy to acquire via paste)
  made downstream save nodes that validate the suffix via
  `os.path.splitext(filepath)[1].lower() != ".exr"` (e.g. spacepxl's
  ComfyUI-HQ-Image-Save) fail with *"Filepath needs to end in .exr"* —
  because splitext returns `.exr ` with the trailing space, which
  doesn't equal `.exr`. The previous control-char strip only handled
  `\r\n\t`. Now `extension` runs through `"".join(extension.split())`
  which removes all whitespace (internal + surrounding, including
  unicode whitespace and non-breaking spaces). Mirrored in the JS
  preview's `cleanExtension`.
- **Easy AI Pipeline: `shot_name` / `ai_method` strip surrounding
  whitespace too.** `_sanitize_segment` now calls `.strip()` after the
  control-char strip, so an upstream feed with stray leading/trailing
  spaces (e.g. `"  shot_v1  "`) resolves to the same path as the
  clean form.
- **Easy AI Pipeline: preview buttons resolve EasyUse GET/SET tunnels.**
  `Get output directory path` and `Get output file path` now follow
  `easy getNode` back to its matching `easy setNode` input before reading
  the source widget value, so global path/name settings preview the same
  value that render-time execution receives instead of the GET key name.
- **Workflow sidebar persistence recovery.** Workflow library saves now
  preserve the full workflow payload shape and recover existing user data
  more defensively, with tests covering payload migration and store
  persistence.
- **Loaded workflows stay untitled in ComfyUI.** Loading from the Kforge
  Labs workflow sidebar no longer forces a saved workflow name into Comfy's
  tab/title state, keeping inserted or loaded workflow sessions aligned with
  ComfyUI's native untitled-workflow behavior.

## [0.3.2] - 2026-05-16

### Added
- **Snapshot dialogs redesign (issue #137).** The Save and Load dialogs
  now carry an inline "Saved to" / "Loaded from" library row at the top
  of the body (folder name + full path + "Open folder ↗" link) and a
  bottom command bar with `Save to…` / `Load from…` that opens a new
  navigate-into folder picker (matching the mockup in
  [`docs/designs/snapshot-dialogs.html`](docs/designs/snapshot-dialogs.html)
  §6). The picker reuses the existing `/koolook/presets/browse` endpoint
  plus an opt-in `?files=1` parameter that returns child JSON files
  alongside directories, rendered greyed for a "yes, this is the folder
  I expected" affordance.

  The Save dialog's primary button now cycles through the four-state
  pattern (Default / Dirty / In progress / Done) per mockup §5 — picking
  a different library via `Save to…` relabels Save to *"Save to new
  folder"* before the click commits.

  The Load dialog's auto-save discovery is now handled in one window:
  when a preset has a newer recovery file (server row-augment now
  considers `max(periodic.json, pre_load_*.json)` instead of just
  `periodic.json`), selecting that preset changes the title to
  *"Auto-save is newer than the saved version"*, opens the scoped
  Recovery auto-saves section, and offers explicit *"NO - load saved"*
  / *"YES - load latest"* buttons. Presets without a newer recovery file
  simply select the row and change the footer action to *"Load"*. Preset
  deletion is now inline too — clicking × outlines the row red and the
  bottom Close button transforms to *"Yes"*; Escape cancels. The delete
  buttons stay visually neutral until hover.

  The Settings cog is removed from the Snapshot toolbar (row is now
  status indicator · Load · Quick Save · Save); library-path
  management lives inside `Save to…` / `Load from…`. The first
  pytest run (`tests/server/`) ships with this change — 12 cases
  covering the server row-augment and the picker's filesystem
  helpers — and CI runs them via a new Pytest job.
- **Easy Image Batch (`easy_ImageBatch`): "White" placeholder + `source_batch`
  input.** The `placeholder_color` dropdown now offers `Black / Gray / White`
  (fill values `0.0 / 0.5 / 1.0`); previously only Black and Gray were
  exposed. A new optional `source_batch` (IMAGE) input accepts a pre-batched
  image stack (e.g. straight from Load Video). When `source_batch` is
  connected and a slot's individual `imageN` is *not* connected, the node
  pulls VFX frame `imageN_frame` from `source_batch[imageN_frame - 1]` — so
  `imageN_frame` controls both *which source frame to pick* and *where to
  place it on the output timeline*. This collapses the typical
  "Load Video → Get Images From Batch In Range × N → Easy Image Batch"
  helper chain into the node itself. Explicit `imageN` inputs still
  override the source-batch pick for that slot, so existing workflows keep
  working unchanged. `image1` is now optional too (you can drive the node
  entirely from `source_batch`); a clear error is raised if neither
  `image1` nor `source_batch` is connected.
- **Easy Image Batch unlimited keyframes via `source_frames` field.** A new
  optional multiline STRING input (positioned right above `image1_frame`)
  accepts a list of VFX-numbered frames separated by commas, newlines,
  spaces, or any mix (e.g. `"1, 27, 41, 63, 85, 120"` or one number per
  line) — for sequences needing more than 4 keyframes from a
  `source_batch`. Each number both picks `source_batch[N - 1]` and places it
  at output index `N - cut_start_frame` (same convention as the per-slot
  `imageN_frame` fields). Bad tokens warn and are skipped;
  out-of-range values warn and are skipped.

  **List-vs-slot interaction rule:** when `source_frames` is non-empty,
  the 4 manual slots contribute ONLY where `imageN` is explicitly
  connected — the `imageN_frame` defaults (5, 9, 13, 17) do NOT pick from
  `source_batch` in this mode. So the list fully controls the selection,
  with explicit `imageN` connections used solely to override individual
  positions. When `source_frames` is empty, the original 4-keyframe
  behaviour is preserved (unconnected slots fall back to
  `source_batch[imageN_frame - 1]`).

  Priority order (later wins): `source_frames` list → 4 manual slots.
  Output dedup is automatic (set-based tracking).
- **Easy Image Batch: two new outputs — `selected_image_batch` (IMAGE)
  and `selected_frames` (STRING).** `selected_image_batch` contains
  only the keyframes that actually landed, packed back-to-back in
  ascending timeline order with no placeholders (length 0 if nothing
  was placed). `selected_frames` is a comma-separated VFX-frame-number
  string of the placed keyframes (e.g. `"1, 27, 41, 63"`) — same
  format as the `source_frames` input, so the output round-trips
  cleanly into another `easy_ImageBatch` instance for chaining or
  re-using selections. Outputs are appended at slot positions 3 and 4,
  so existing wires on `image_batch` and `alpha_batch` are unaffected.
- **`Easy_Pattern` node** under category `Koolook/Testing`. Generates a
  batch of solid-color frames `[B, H, W, 3]` with optional per-frame
  index overlay; configurable `start_from` / `step` / `zero_pad` /
  `prefix` / `suffix` for the stamped text, and five overlay positions
  (`center` / `top-left` / `top-right` / `bottom-left` / `bottom-right`).
  Background and text colors each have a `*_color_mode` dropdown
  (`White` / `Black` / `Gray` / `Custom`) backed by an optional hex
  string consulted only when the mode is `Custom` — same idiom
  `EasyResize_Koolook` already uses for `pad_color_mode` and
  `panel_color_mode`, so "Gray" means the same `#808080` 50%-gray across
  nodes. Defaults out-of-box to an 81-frame magenta pattern with
  white centered text — useful for verifying sequence merging, frame
  insertion, batch indexing, and video-pipeline frame order at a glance.
  Drafted in an external AI chat, ingested in-tree, relicensed from the
  AI's MIT boilerplate to the project's GPL-3.0-or-later.

### Changed
- **Easy Image Batch second output renamed `mask_batch` → `alpha_batch`,
  with the same value convention as before.** The default behaviour is
  unchanged — selected (occupied) frames = `0.0` (black), missing
  frames = `1.0` (white) — so existing workflows wire up identically.
  A new `invert_alpha` boolean toggle (default `false`, labels *"inpaint"*
  / *"compositing"*) flips the alpha to compositing-style
  (`selected = 1.0, empty = 0.0`) when needed. The output slot keeps its
  position (second output) so wires re-attach automatically when the
  node reloads.
- **Easy Image Batch: `start_frame` renamed to `cut_start_frame` and
  reinterpreted as a built-in cut window.** The output now represents
  VFX frames `[cut_start_frame .. cut_start_frame + total_frames - 1]`.
  Frames placed outside this window are silently dropped from the output
  (a single end-of-run summary lists them, replacing the previous
  per-frame "out of range" warnings). Math:

      source pick index   = vfx_frame - 1                 (hardcoded VFX 1-based)
      output (cut) index  = vfx_frame - cut_start_frame   (cut window)

  The two were previously conflated under `start_frame`, which produced
  the wrong source pick whenever `start_frame` differed from `1`. With
  the new model each slot picks `source_batch[vfx-1]` and places it at
  output index `vfx - cut_start_frame` independently.

  Builds in the workflow that previously used an external
  `ImageFromBatch (batch_index, length)` to slice the Easy Image Batch
  output. Defaults: `cut_start_frame=1`, `total_frames=81` (was `16`),
  `imageN_frame` defaults bumped to `5, 9, 13, 17` (was `0, 4, 8, 12`)
  so they're valid VFX-1-based frame numbers.

  **Migration note:** saved workflows that had `start_frame` set will
  load with the field's stored value into `cut_start_frame` (same field
  position, same units). Behaviour is preserved when `cut_start_frame=1`
  is the default and the source_batch starts at VFX frame 1; if you had
  set `start_frame` to a non-1 value, re-check that your source pick
  result is still what you want — under the old code, `start_frame=N`
  also offset the source-batch index, which the new model intentionally
  does not.
- **`scripts/sync_to_dev.py` auto-restarts the live ComfyUI server** by
  hitting ComfyUI-Manager's `/manager/reboot` endpoint after a successful
  sync. Default-on because dev-sync is invoked precisely when the
  maintainer wants new code visible, and Python custom-node modules load
  once at server start — without a restart, `.py` changes stay invisible.
  Use `--no-restart` to stage files without disturbing the live session;
  `--restart-url` overrides the endpoint for non-default ComfyUI
  hosts/ports. Restart failures (Manager missing, server not running,
  etc.) print a diagnostic but do not fail the sync. The
  `RUNTIME_PATHS` manifest also gains `k_easy_pattern.py` so new
  top-level node files reach the dev install instead of being silently
  dropped — without this entry the modified `__init__.py` would import
  a module the live install doesn't have, aborting the whole pack's
  registration.
- **Release preflight no longer blocks on ComfyUI-Manager metadata drift.**
  `tools/preflight_release.py` still verifies local node registration,
  VAE dispatch, and workflow fixtures by default, while the upstream
  `extension-node-map.json` comparison remains available as
  `--check manager-meta` for advisory checks. Releases should not depend
  on third-party metadata updates landing on ComfyUI-Manager's schedule.

### Fixed
- **Snapshot Settings folder save no longer writes a snapshot.** Saving the
  library-path field in the Settings dialog no longer writes the in-memory
  snapshot. Folder-save is now path-only: the button reads "Save folder", the
  hint points to Save / Quick Save, and snapshot writes remain explicit actions.
  Resolves #129.

## [0.3.1] - 2026-05-10

### Changed
- Comfy Registry documentation metadata and README onboarding now point new
  users to the KForge Labs ComfyUI-Koolook spotlight as the public first-read
  overview, while keeping the bundled visual guide framed as the in-app
  operational guide.
- The bundled visual guide's six recipe examples now use the KForge spotlight
  style: wide divided showcase rows with explanatory copy paired to the
  relevant screenshot, instead of separate compact recipe cards.
- The bundled visual guide now includes a Tags showcase between Modules and
  Search, showing how saved workflows can appear under multiple labels and
  remain searchable without duplication.

## [0.3.0] - 2026-05-08

### Added
- **Visual sidebar guide now hosted on GitHub Pages.** The bundled guide at
  [`web/guide/index.html`](web/guide/index.html) is now also live at
  https://malkuthro.github.io/ComfyUI-Koolook/web/guide/ so users browsing
  the repo on github.com — and visitors landing on the package page in the
  Comfy Registry — can preview the onboarding without cloning. The in-app
  Tools → Help (`H`) button continues to open the bundled local copy via
  `import.meta.url`, so the guide stays available offline. README and
  `pyproject.toml`'s `Documentation` URL now point at the Pages site
  (replacing the unmaintained wiki link).
- **Load dialog asks YES/NO when an auto-save is newer than the saved
  version.** Click any preset row in Snapshot → Load and, if the
  matching `<preset>_autosave/periodic.json` is more recent than the
  named save's on-disk mtime, the confirm dialog upgrades from a single
  "Load preset" button to a YES/NO question: "Auto-save is NEWER than
  the saved version. Do you want to load the auto-saved version?". YES
  loads the auto-save, NO loads the older deliberate save (the row the
  user originally clicked), Esc cancels. Both paths run the standard
  pre-load auto-save protection so either choice is reversible.
  Replaces the earlier inline "↻ Newer auto-save" subline on the row
  itself — putting the question on the row asked the user to compare
  two timestamps before they had committed to loading anything; moving
  it to the post-click modal puts the choice exactly where the user has
  already decided to load and just needs to pick which version. The
  list rows render as before (one timestamp, one × button) so the
  library list stays scannable. Server-side: `/koolook/presets/list`
  augments root rows with `latestAutosaveMtime` via one extra `stat()`
  per row — no file read, cheap on large libraries. Aligns the system
  with the mental model: named saves stay deliberate (only updated on
  Save / Quick Save), and the auto-save is treated as a recovery copy
  the system surfaces at decision time rather than a parallel save
  destination the user has to remember to check.
- **Bundled visual onboarding guide for the Kforge Labs sidebar.** The
  existing Tools-row `H` help shortcut now opens `web/guide/index.html`, an
  offline visual guide that frames Snapshots as a reusable project kit for
  favorite nodes, cross-sidebar search, saved workflows, modules, tags,
  autosaves, and starter presets. Its static styling borrows the dark,
  token-driven card language from the ForgeFlow sibling project's design docs.
  The first version ships annotated mock images under
  `web/guide/img/` so they can be replaced by real ComfyUI screenshots later
  without changing the guide structure. Linked from the README and user guide
  index for users browsing docs outside ComfyUI.
- **Subgraph-aware module saves.** Selection saves now carry the
  transitively referenced `definitions.subgraphs` entries for selected
  ComfyUI subgraph wrapper nodes, so the saved workflow has the definitions
  available for native Load / later insert attempts.
- **Dev-sync build-tag footer at the bottom of the Kforge Labs sidebar.**
  `scripts/sync_to_dev.py --scope "<short scope>"` now writes a tiny
  `web/_dev_build.json` next to the sidebar JS. The sidebar fetches it
  on render and emits a discreet two-line footer: `dev <sha> · <time>`
  (line 1, SHA at 13px monospace) + the scope (line 2, italic
  proportional). Visible-in-app counterpart to the chat-report
  convention from #95 — when the maintainer is juggling multiple
  parallel worktree sessions, a glance at the footer confirms which
  one's code is actually live in the browser tab. Cache-busted fetch
  so a re-sync's new mtime reaches the browser without a hard refresh.
  Absent on registry installs (the JSON is dev-only and gitignored);
  the footer renders empty there. The `--scope` value is the same
  ≤10-word change-summary the agent quotes on chat-report line 2 —
  passing it once now feeds both surfaces.
- **`×` removability across both group modes for any visible favorite,
  including auto-pulled REPOS{select: "all"} nodes.** Adds a
  `koolook.autoPullHidden.v1` localStorage set: clicking `×` on a row
  that came from auto-pull adds the type to the hidden set so the
  next render skips it (without the set, the next gather pass would
  re-pull the type from the registry and the click would feel like a
  no-op). User picks still go through `removeFromMyPicks`. Re-adding
  via the toolbar `+` button or right-click → "Add to Kforge Labs"
  clears the hide as a side effect of `addToMyPicks`, so users don't
  have to think about which list a type lives in. Same uniform
  "remove this from my favorites / bring it back" UX whether the node
  is a Koolook auto-pull or a KJNodes user-pick.
- **Snapshot library path browser.** Snapshot Settings now has a Browse button
  backed by a new `/koolook/presets/browse` route, so users can navigate to
  the preset folder from inside Kforge Labs instead of copy-pasting a path from
  an external file manager. The browser hides internal auto-save folders,
  supports creating one child folder under the current location, and makes the
  selected folder name visually prominent before the user chooses it.

### Changed
- **Snapshot timestamps now use 24-hour format with relative-recent tier.**
  The Load dialog's "saved …" line and the snapshot status-dot tooltip
  previously formatted local time as 12-hour `en-US` (`May 8, 12:56 AM`),
  which read visually like noon for timestamps just past midnight — the
  classic US "12 AM = midnight" trap. Times now render as: `just now`
  (<1 min), `12 min ago` (<60 min), `today 14:32` / `yesterday 23:58`
  (same / previous calendar day), or `May 6 14:32` (older; year appended
  when not current). Centralized in a new `format_time.js` helper used
  by both the Load dialog rows and the status tooltip so future surfaces
  stay aligned.
- **Sidebar context menus now enforce a single open menu.** Opening a
  workflow, directory, archive, or Workflows-header right-click menu now closes
  any previous sidebar context menu first, so repeated right-clicks cannot
  stack multiple floating menus with stale actions.
- **"Create directory" moved off the Workflows toolbar onto right-click of
  the section header.** The 📂 (`pi-folder-open`) button used to live in the
  Workflows action bar between the section label and the Save buttons. Tree-
  structure operations (create / rename / delete a folder) already live on
  right-click of the row representing the structure being mutated — the per-
  folder menu offers "Create subdirectory…" / "Rename directory…" / "Delete
  directory" — but top-level dir creation was the lone outlier sitting in the
  toolbar. Moving it to right-click on the **"Workflows"** header row makes
  the action bar homogeneous (only save actions: Save canvas, Save selection)
  and unifies the mental model: every tree-structure operation lives on
  right-click of the structure node it acts on. Implemented via a new
  optional `rootContextMenu` field on section descriptors plumbed through
  `buildFolder`; only Workflows opts in for now (Nodes/Tags keep today's
  no-context-menu section header). Fresh-install path is unchanged — the
  Save modal's "+ New directory…" entry still creates the first directory
  when the workflow store is empty. Closes #104.
- **Kforge Labs sidebar action icons now use a first-pass ForgeFlow-style
  icon set.** Snapshot Load / Save, Tools export / install / help, the
  missing-pack canvas handoff, repo-mode grouping, and whole-canvas workflow
  save now use the approved compact letter / simple-glyph treatment from
  `docs/designs/sidebar-icon-proposals.html`, reducing repeated generic
  cloud / grid / floppy symbols while leaving all actions and tooltips
  unchanged. The existing Tools-row `H` help shortcut is now wired to the
  bundled visual onboarding guide.
- **Node help text tightened for scannability.** The Easy Resize description
  and HDR VAE encode/decode tooltips now use shorter, action-oriented text so
  the ComfyUI node help stays readable while detailed behavior remains in the
  user guide.
- **README refocused as a short product front door.** The repo landing page
  now leads with the Kforge Labs sidebar, Snapshots, search, workflows, and
  recovery, then links out to the deeper user/maintainer/fork docs instead of
  carrying long maintainer workflow and implementation sections inline.
- **Node-list hover previews now stay at the intended 300px mock-node width.**
  The preview card previously had only a `min-width` plus viewport `max-width`,
  so long node descriptions could set the card's intrinsic width and stretch the
  hover overlay across the canvas. The card now uses a fixed 300px preferred
  width clamped to the viewport, keeping labels and descriptions contained.
  Its surface colors also now track ComfyUI's native node charcoal grades more
  closely, so mock previews feel like lightweight nodes instead of black
  sidebar tooltips.
- **Theme mode now mirrors repo-mode population** — auto-pulled
  REPOS{select: "all"} nodes appear alongside user picks (deduped),
  filtered by the new auto-pull hidden set. Supersedes the earlier
  `[Unreleased]` "Picks-only (no REPOS-driven auto-pulled candidates)"
  intent below: the picks-only model dropped the entire Koolook pack
  from theme mode on a stock install, which silently violated the
  "my favorites" mental model (the user expected their visible
  favorites in repo mode to also show in theme mode, just regrouped).
  With the new × removability, the user can still curate exactly what
  they see — without the discoverability cost of dropping auto-pull
  entirely.
- **Pack-name badges restored on theme-mode leaves.** Small dim
  `Koolook` / `KJNodes` etc. on the right of each row, same opacity
  as the search-flatten breadcrumb. Memory aid for "where did this
  node come from?" — useful when the user is learning their favorites
  pack-by-pack. Reverts (for theme mode only) the earlier
  `[Unreleased]` removal further down; search-flatten still skips the
  badge since its breadcrumb prefix already conveys origin.
- **Snapshot status hover and path display tightened.** The status tooltip now
  shows only a central-time timestamp and the snapshot library location, and
  snapshot library path lines wrap to show the full path instead of truncating.
- **Snapshot Settings save state clarified.** After browsing to a new folder,
  the Save button now becomes an explicit dirty "Save update" action; once the
  path is persisted it turns back into a disabled gray "Saved" state. Load and
  Settings dialogs now show the active library folder name first, with the full
  path underneath; after changing folders, Settings warns that the current
  snapshot still needs Save / Quick Save to be written into the new location.
- **Sidebar second mode — "Theme" instead of "Category".** The sitemap-icon
  toggle in the Nodes action row now groups picks by **semantic theme**
  rather than by raw CATEGORY first-segment. The new algorithm strips the
  source pack's name from the front of each pick's CATEGORY (via the
  existing `findPackPathForType` precedence) before grouping, so
  `KJNodes/image/Get`, `Koolook/Image/EasyResize`, and `image/foo` all
  collapse to one top-level `image` folder. Each theme bucket renders as
  a flat sorted list — no further sub-folders. Picks-only (no
  REPOS-driven auto-pulled candidates) so the second mode genuinely
  reflects "my image-related favorites" rather than "every image-shaped
  node ComfyUI knows about." Synthetic buckets `(unresolved)` /
  `(uncategorized)` survive as before.
- **Pack-name badges removed from leaf rows.** The small dim labels
  (`Koolook`, `KJNodes`, `EasyUse`) that used to render at the right edge
  of every leaf in theme mode and in search-flatten are gone. The
  breadcrumb prefix on search-flatten rows already conveys origin in a
  cleaner form, and theme mode by definition doesn't care about source
  pack — the badge was redundant. `makeNodeLeafRow` no longer accepts a
  `packBadge` parameter; CSS rule `.koolook-pack-badge` removed from
  `constants.js`.

### Fixed
- **Cancel / Escape / click-outside on the "Replace current workflow?"
  placeholder-drop modal no longer leaks a pending Promise** (closes
  issue #84 part 2). `dropPlaceholdersForPacks` in
  `web/sidebar/canvas_io.js` wrapped `showConfirmModal` in
  `new Promise` but only resolved on the OK path — every cancel left
  an unresolved Promise behind. The Promise now resolves with
  `{ ok: false, reason: "cancelled" }` (matching the function's other
  failure shapes) on every non-OK dismissal. Fix is two-layered:
  `dropPlaceholdersForPacks` passes an `onCancel` resolver, AND
  `showConfirmModal` in `web/sidebar/modals.js` now fires `onCancel`
  for the Cancel button, Escape, AND overlay-click via an idempotent
  settle flag (previously only the Cancel button fired it). Closes a
  latent leak on the recovery toast's "Discard offline copy" path at
  `web/koolook_sidebar.js:130` for free — that caller had wired
  `onCancel` for the Cancel button only, so pressing Esc / clicking
  outside used to leave the toast button stuck in "Discarding…".
- **Snapshot status reads "auto-saved" — not "saved" — after restoring
  from an auto-save.** Previously, restoring `<preset>_autosave/
  periodic.json` (via the new YES path on the Load dialog OR the
  long-existing Recovery section) called `markStateSaved()`, which
  baselined the *named-save* fingerprint to the auto-save's content.
  The dot turned green even though `<preset>.json` on disk still held
  the OLDER deliberate save — blurring the deliberate-save model the
  Path C UX deliberately reinforces. New `markStateAutosaved()` export
  in `snapshot.js` baselines the *periodic* fingerprint instead, so the
  dot reads "auto-saved" (blue) until the user explicitly Quick Saves
  or Saves the restored state into the named file. Toast message also
  updated to nudge: `Restored auto-save of "Foo" — Quick Save to commit
  to the named file.` Fixes the regression flagged on PR #111 review.

### Changed
- **Load dialog header + Recovery group headers now share a layout.** The
  library section and each `<preset>_autosave` group header now use the
  same two-line format (title + full path) and the same `📂 Open` button
  styling — visual consistency across both surfaces. Concrete changes:
  - **Saved-time on every row.** Regular snapshot rows now show `saved
    May 7, 11:53 PM` (from the file's mtime, threaded through
    `loadPreview` from the listing endpoint) instead of just `exported
    07/05/2026`. Recovery rows already had this format; both row types
    now read consistently. Falls back to `exported <date>` if mtime
    isn't available (defensive — covers any caller that doesn't carry
    listing metadata).
  - **Recovery group headers now show the full subdir path** under the
    subdir name, mirroring the library header's `Library folder: X` /
    `<full path>` pair. Reuses `renderLibraryLocation` with a new
    `title` override so the same CSS styling applies without the
    "Library folder:" prefix.
  - **Library `📂 Open` button now uses `koolook-snapshot-row-btn`**
    (the smaller styling) to match the per-group Open button. Was
    `koolook-modal-button` (larger) before — visually mismatched.
  - `formatAutosaveMeta` collapsed back into `formatPreviewMeta` since
    the format is now identical across row types.

### Added
- **Offline-fallback banner now has Restore + Discard actions — no more
  permanent guilt trip.** The red `criticalToast` that fires when
  localStorage holds a stale outage-fallback blob used to be a one-way
  street: Copy details, Dismiss, repeat next page load forever. Two
  buttons close the loop:
  - **Restore as snapshot** wraps the offline workflows into a full
    snapshot envelope (live picks bundled in so a later Load wouldn't
    wipe them) and writes it as `recovered-<iso>.json` into the snapshot
    library, then clears the localStorage key. The user lands on a
    durable, in-UI-visible artifact they can inspect, Load, or
    selectively cherry-pick from. Banner self-extinguishes on success.
  - **Discard offline copy** asks for confirmation, then clears the
    localStorage key and dismisses the banner. For the case where the
    user has already verified nothing is missing.
  - The toast is now wired from `koolook_sidebar.js` (the entry point)
    rather than from `loadWorkflowsStore()` itself — the action handlers
    need `writePreset` / `loadUserPicks` / `showConfirmModal`, and
    pulling those into the workflows store would create a circular
    import.
  - `criticalToast` now accepts an `actions: [{label, onClick, primary?,
    busyLabel?}]` array; existing callers (`{ copyText }` and string-
    only) keep working unchanged.

### Added
- **Recovery section deep-links to the file system + per-row save time.**
  The Load dialog's Recovery auto-saves section now exposes everything
  a power user needs to navigate the autosave subfolders directly:
  - **`📂 Open` button at the top** of the Load dialog opens the
    snapshot library folder in the OS file manager.
  - **`📂 Open` button on each `<preset>_autosave` group header** opens
    that specific subfolder, so the user lands on the exact recovery
    files for that preset (with mtimes visible to Finder / Explorer
    sort).
  - **Per-row save time replaces "exported &lt;date&gt;".** Recovery
    rows now show `saved May 7, 2:34 PM` derived from the file's mtime
    — strictly more useful than the snapshot's self-reported
    `exportedAt` for telling "which pre-load is freshest." Mtime also
    survives out-of-band file copies in Finder.
  - **Bare filename instead of redundant displayName.** Each row now
    renders the actual filename (`pre_load_2026-05-07T14-32-…`,
    `periodic`) instead of the noisy
    `Periodic auto-save · <subdir> · <iso>` string that pushed the
    timestamp off-screen.
  - New server route `POST /koolook/presets/reveal[?dir=<subdir>]`
    powers both buttons via the platform-appropriate file-manager
    command (`open` on macOS, `explorer.exe` on Windows, `xdg-open`
    on Linux). Path-traversal grounded at the library base via the
    same `_resolve_target` check the file-write routes use.

### Fixed
- **Insert pre-flight toast now surfaces both pack misses AND subgraph
  misses when a saved workflow has both, in plain English.** Previously,
  the "all-misses-are-subgraph-UUIDs" early-return only fired when
  *every* missing type was a UUID; mixed cases fell through to a single
  "install the required pack(s)" message that listed UUIDs alongside
  pack types but mentioned nothing about the registration step the
  subgraph also needs. Users installed the pack, retried Insert, and
  hit a second unexplained "still missing" failure. Now the toast
  branches three ways — pure-pack / pure-subgraph / mixed — and the
  mixed case names both fixes upfront as a numbered two-step ("install
  pack(s) + right-click → Load it once"). Wording also de-jargoned —
  "subgraph definition not registered" was technically accurate but
  confused users who just wanted to know what to do. Engineer-speak
  stays in code comments and console.warn for devs; user-facing copy
  now talks in gestures and outcomes ("Right-click → Load it, then
  left-click Insert will work").
- **Atomic preset writes — server crash mid-save can no longer corrupt
  your snapshot library, and a hostile `.tmp` symlink can no longer
  redirect writes outside the library.** `POST /koolook/presets/file`
  now uses `tempfile.mkstemp` (with `O_CREAT | O_EXCL` semantics) to
  create a uniquely-named temp file inside the already-validated
  target directory, then `os.replace`s onto the final name. A dropped
  connection or process kill mid-write can no longer truncate the
  existing good file. An attacker who plants `<file>.json.tmp` as a
  symlink pointing outside the library can no longer trick the server
  into following it: `mkstemp` aborts with `O_EXCL` on any existing
  dirent at the chosen name, AND the name is randomized so the
  attacker can't pre-plant anything there. Covers every preset write
  path (named, periodic autosave, pre-load autosave, recovery
  snapshot) through the one route. The transient temp file is still
  invisible in the Load and Recovery dialogs (both list endpoints
  filter to `.json`).
- **`showConfirmModal` now exposes an `onCancel` callback.** Callers
  that wrap the modal in a Promise can now settle when the user
  cancels instead of leaking a pending Promise. Closes the second
  half of #84 (`dropPlaceholdersForPacks` was the original report);
  also removes the "Discarding…" stuck-button footgun on the
  offline-fallback recovery toast — Cancel now re-enables the toast
  button so the user can choose Restore / Copy / Dismiss instead of
  being trapped in a half-disabled state.
- **Snapshot status timestamps now follow the user's wall clock.**
  `formatCentralTime` was hardcoded to `timeZone: "America/Chicago"`
  and surfaced a `CDT` / `CST` stamp in the auto-saved hover tooltip
  regardless of where the user actually was. Renamed to
  `formatLocalTime` with the timezone option dropped — the browser
  formats with the user's system timezone. Also dropped
  `timeZoneName: "short"`: it just rendered noisy abbreviations (`GMT+2`,
  `CDT`, …) that didn't help anyone identify "their own time." Tooltip
  now reads as a clean wall-clock stamp like `May 7, 2026, 11:43 PM`.

### Added
- **Modules — splice a saved cluster into your live canvas instead of
  replacing it.** Tag any saved workflow with the literal tag `module`
  and the Kforge Labs sidebar starts treating it as a reusable building
  block: a green `pi pi-plus-circle` icon replaces the file glyph,
  left-click **inserts** the cluster at the viewport center (with
  internal links re-created and the new nodes left selected), and a
  distinct hover tooltip surfaces the changed semantics. The
  selection-save modal now ships a **`Save as module`** checkbox
  (pre-checked for selection saves; unchecked for whole-canvas saves),
  which adds the `module` tag inside the same `persistMutation` as the
  entry write so a commit failure rolls back both. The right-click
  menu on every workflow row gains an explicit **Insert into canvas**
  entry alongside the existing Load — Load still works on every row,
  module-tagged or not. New primitive `insertWorkflowOntoCanvas` in
  `web/sidebar/canvas_io.js` pre-flights every referenced node type
  against `LiteGraph.registered_node_types` and aborts cleanly when
  any are missing (a partial insert with stub nodes is worse than no
  insert), then deep-clones the saved graph, lets `app.graph.add()`
  allocate fresh node ids so nothing collides with the live canvas,
  and recreates internal connections via `originNode.connect(...)` so
  link ids are auto-allocated too — no manual link remap needed.
  Designed for setups like an EXR-output stack, a depth-only render
  cluster, or a Wan 2.2 prompt module the user wants to drop into many
  workflows. Full reference in
  [`docs/maintainers/workflows-sidebar.md`](docs/maintainers/workflows-sidebar.md#modules--splice-a-saved-cluster-into-your-live-canvas).
- **Spotlight effect on add.** Clicking the toolbar `+` (or the canvas
  right-click "Add to Kforge Labs") now collapses every Nodes-section
  pack folder, then auto-expands just the pack + subcategory the just-saved
  node lives in — a pedagogical aid that helps new users internalize
  which pack each node belongs to. Multi-select adds light up every hit
  pack simultaneously. Duplicate adds (already in picks) trigger the
  same spotlight, since the educational reminder still applies. Helper
  `findPackPathForType` mirrors the gather code's REPO-precedence-then-
  user-pick-fallback to locate any node ID's sidebar path; new exported
  `spotlightAddedPicks(typeNames)` from `web/sidebar/tree.js` does the
  collapse + pin sequence and is called from both add paths.
- **Group-by-category mode for the Nodes section.** Segmented toggle in
  the Nodes action row (📦 Repository / 🌐 Category — `pi-database` /
  `pi-sitemap`); choice persists in localStorage (`koolook.groupMode.v1`)
  and survives reloads, including cross-tab via the existing `storage`
  event listener. In repo mode (default) every pick lives under its
  pack; in category mode picks are regrouped by their node-class
  `CATEGORY` path, ignoring REPOS affiliation. Categories that look the
  same after canonical-key normalization
  (`lower().replace(/[\s_\-]+/g, "")`) collapse into one folder —
  `Loaders`/`loaders`, `Image/Upscaling`/`image/upscaling`, and
  `style_model`/`StyleModel`/`style model` all merge. Display label is
  the most-common original casing seen for that key (ties → first-seen,
  stable across renders because Map iteration follows insertion order).
  Ports of the same path under different repos all collapse together,
  so a Loader from Pack A and a Loader from Pack B share the
  `Loaders/` folder. Two synthetic top-level buckets surface the edge
  cases: `(unresolved)` for picks whose pack isn't currently loaded
  (repo mode silently dropped these — category mode keeps them visible
  as italic-dim rows), `(uncategorized)` for picks whose node class has
  no `CATEGORY`. Each leaf in category mode carries a small pack-name
  badge so users still see "where did this come from" — load-bearing
  for the existing `↓ Install missing` flow. Folder paths in
  `pathStates` use the canonical key (not the resolved display label),
  so an upstream casing change to the most-common label doesn't reset
  user expansions. Closes the "sort by topic, by what they do" piece
  of #46. (#73)
- **Flatten-on-search for the Nodes section.** When the search field is
  non-empty, the Nodes section drops its tree structure and renders a
  flat list of matching leaves, sorted by display name, each with a
  small dim-grey breadcrumb prefix (`Loaders › LoRA › LoraLoader` in
  category mode, `Pack › Subcategory › Display` in repo mode) so the
  spatial-origin signal survives. Workflows and Tags sections retain
  their tree-under-filter behavior — the change is scoped to Nodes
  only. The breadcrumb collapses redundant synthetic labels: the
  `(root)` subcategory (which adds no info beyond the pack name) and
  any subcategory whose label duplicates the pack label (the
  "(uncategorized)" double-up). Always-flatten on any non-empty query
  is the deliberate choice over a count-threshold gate; the spatial
  cue from a tree is mostly already lost once a search has narrowed
  the set, and a flat sorted list scans faster. (#73)
- **Hover preview card for sidebar leaf rows.** Mirrors ComfyUI's
  official Node Library preview ([NodePreview.vue][np-vue]): plain
  HTML/CSS card with a colored title bar (HSL hue hashed from the
  `CATEGORY` string), category breadcrumb, two columns of type-colored
  slot dots (inputs left, outputs right), widgets section with
  truncated defaults, optional description. Hover a leaf for ~250ms to
  show; mouseleave dismisses; only one card visible at a time
  (module-level singleton). Card positions to the right of the row by
  default (12px offset), flips to the left when it'd overflow the
  viewport, clamps inside an 8px viewport-padding gutter. Tall cards
  (a node with many inputs) cap at `calc(100vh - 16px)` and scroll
  internally rather than clip. `pointer-events: none` on the card so
  it never intercepts events from rows it floats over. Slot dot
  colors read ComfyUI's runtime palette
  (`app.canvas.default_connection_color_byType`, populated at canvas
  init) first, then fall back to LiteGraph's static
  `LGraphCanvas.link_type_colors` (mostly empty in stock LiteGraph),
  then a neutral grey. Widget classification mirrors ComfyUI's
  frontend rule: scalars (`INT`/`FLOAT`/`STRING`/`BOOLEAN`) and
  arrays-of-choices (`COMBO`) become widgets, everything else is a
  connection slot. `INPUT_TYPES()` is read defensively (some
  custom-node packs throw under unusual conditions); when one does
  throw, the failure is logged via `console.warn` so a broken pack is
  visible during debugging. Picks whose pack isn't loaded show a
  "Pack not loaded" stub card pointing at the Tools-row install
  button. The hover card additionally tears down on
  `visibilitychange` (page hidden) and `window.blur` so it doesn't
  leak across tab switches or focus loss. New module
  `web/sidebar/node_preview.js` holds the preview engine;
  `attachHoverPreview(row, type)` is wired inside `makeNodeLeafRow`
  so every leaf-emit site (repo tree, category tree, search-flat) gets
  the preview for free. The renderer calls `teardownPreview()` before
  each `treeEl.innerHTML = ""` — without it, an active card whose
  anchor row gets detached during a re-render would leak (the
  `pointer-events: none` card can't be dismissed by clicking it).
  (#73)

[np-vue]: https://github.com/Comfy-Org/ComfyUI_frontend/blob/main/src/components/node/NodePreview.vue

### Changed
- **Hover preview rebuilt to read like a real node mock.** The original
  layout (HSL-hashed title strip + two-column slot table + flat widget
  text list) read as a generic info card. Reworked to mirror upstream
  `NodePreview.vue` more faithfully: a flat header row (small colored
  dot + node display name) over a red **PREVIEW** badge, slot rows
  laid out as a 5-column grid (`[dot] [input-name] [spacer] [output-name] [dot]`)
  so inputs and outputs line up horizontally as parallel sockets, and
  widgets rendered as individual rounded pills with `◀ name [spacer]
  value ▶` chrome to mimic LiteGraph widget UI. **Bigger fix
  underneath:** `readSlots` and the title / category / description
  reads were looking at static `nodeClass.INPUT_TYPES` /
  `RETURN_TYPES` properties, which **don't exist** on
  ComfyUI-registered nodes — ComfyUI parks the original V1 def at
  `nodeClass.nodeData` (set by `litegraphService.ts`'s
  `registerNodeDef` right before `LiteGraph.registerNodeType`). Every
  card was rendering as just-the-header-and-badge because the slot
  arrays came back empty. Reading from `nodeData` first (with the old
  property names as a legacy fallback for non-ComfyUI registered
  nodes) fixes the empty-card class. (#78)

### Fixed
- **Module save/insert now preserves internal links more defensively.**
  Selection save and module insert both accept serialized link arrays and
  object-shaped LiteGraph `LLink` records, compare node/link ids by stable
  string keys, and strip stale saved link ids before recreating links via
  `node.connect(...)`. Saved module state is now first-class (`module: true`)
  while still honoring the legacy `module` tag, so left-click module insert
  and right-click Tags-based toggling stay in sync.
- **`pinExpanded` paths now expand on the immediate render** instead of
  being delayed by one. `buildFolder`'s expansion-resolution chain gained
  a `pinnedPaths` check between `forceExpanded` and `pathStates`. Phase 3
  of `renderTree` writes pins into `pathStates` AFTER Phase 2 has built
  the DOM, so without the new check the pin had no effect on the current
  render — only on the next one. Side-effect-free correctness improvement
  for the workflow-save pinning that already existed; load-bearing for
  the new spotlight feature above which depends on pinned paths
  expanding immediately.
- **Modal drag-out-of-input no longer dismisses the dialog.** The shared
  modal shell (`makeModalShell`) used by every Koolook dialog gained a
  mousedown / click-intent check: a click on the overlay now only fires
  the dismiss when the gesture *started* on the overlay too. Drag-
  selecting text inside an input field that releases in the overlay's
  dark area used to trigger `click` with `target=overlay` and dismiss
  the modal mid-edit; now it stays open. Affects all dialogs including
  Snapshot Save / Load / Settings, Install Missing, Save Workflow, Tags,
  Confirm, Input — load-bearing especially for the Snapshot Settings
  path field where the long absolute path encourages drag-selection.
- **Snapshot Settings path field shows full path on hover.** Added a
  `title` attribute synced to the saved library path on every refresh
  and save, so the user can read the full path without scrolling /
  selecting inside the narrow input.

### Added (continued)
- **"Install missing for picks" toolbar button** in the Nodes section
  (`pi-cloud-download` icon, next to Add and Export). Walks the user's
  picks against ComfyUI-Manager's `/customnode/getmappings` mapping,
  buckets into already-installed / will-install / unresolved, queues
  unique git URLs through `/customnode/install/git_url`, polls
  `/manager/queue/status` to drive a progress bar, and prompts to
  reboot. Works on any install with ComfyUI-Manager loaded; falls back
  to a clipboard URL list (with a `comfy node install` hint) if Manager
  isn't reachable. New module `web/sidebar/installer.js` holds the
  Manager-API client + resolver; the modal in `web/sidebar/modals.js`
  drives the four-phase UI (discovery → confirm → progress → result).
  403s from Manager's security gate are surfaced as actionable language
  ("your security level forbids git-URL installs") rather than raw HTTP.
- **Snapshot library** — save your full Kforge Labs state (curated picks
  + the entire workflows store including tags + archive) as a named
  preset to a configurable filesystem path. New top-level **Snapshot**
  action row above the search field with three icon buttons, each
  opening its own focused dialog:
  - **Save (cloud-up icon)** — one click. If a preset is currently
    loaded, prompts "Save over '<name>'?" with three options
    (**Save** overwrite / **Save as new…** rename / **Cancel**). If
    no preset is loaded yet, prompts for a name (default
    `preset YYYY-MM-DD`, fully editable). The "current preset" is
    tracked in localStorage and persists across reloads, so Save
    keeps doing the obvious thing across sessions.
  - **Load (cloud-down icon)** — single dialog. Lists every preset
    in the library with metadata (workflow count · pick count · export
    date). Click a row → confirm "Replace current state?" → restores
    picks + workflows. Each row has an **×** button for delete (with
    confirm). Header line shows the current library path so you can
    see where you're loading from.
  - **Settings (cog icon)** — single field for the library's
    filesystem path. Save writes to a per-install settings file the
    server reads. **Reset to default** clears the saved path so the
    server falls back to env-var or built-in default. A read-only
    line shows the currently-resolved path + source (settings panel
    / env var / built-in default), so what's in effect right now is
    always visible.
- **Configurable storage location.** Resolution chain (highest first):
  1. Path saved via the Settings dialog
     (`<comfyui-userdata>/koolook-settings.json`'s `libraryPath`).
  2. `KFORGELABS_PRESETS` env var (deployment / facility config).
  3. Built-in default `<comfyui-userdata>/koolook-presets/`.
  Use cases:
  - **Personal cross-machine sync:** point at a Dropbox/iCloud/Drive
    folder; the library follows your machines.
  - **Facility shared library:** point at an NFS/SMB mount writable
    by every workstation; all workstations save to + load from the
    same library natively, no symlinks.
  - **Read-only distribution:** point at a path the workstation can
    read but not write; save fails cleanly with the server reason in
    a toast, load works.
  Snapshot files carry a `kind: "koolook-snapshot"` discriminator +
  `version` field for future schema migrations. Closes the "save to
  a custom location, take it elsewhere" piece of #46.

### Internal
- New server-side module `koolook_routes.py` registers the
  `/koolook/presets/*` aiohttp routes on ComfyUI's PromptServer.
  Endpoints: `info`, `list`, `file` (GET/POST/DELETE on a single
  query-string-keyed endpoint, with the GET handler short-circuiting
  HEAD requests via aiohttp's auto-HEAD-from-GET so existence checks
  don't read the full file body), and `settings` (GET/POST). The
  HEAD short-circuit lives inside the GET handler rather than as a
  dedicated `@routes.head` registration because ComfyUI's
  mirror-to-`/api` code in `server.py` blindly forwards
  `RouteDef.kwargs` into a `RouteTableDef.route(...)` closure that
  rejects `allow_head=False`, so any kwarg-based opt-out crashes
  startup. Path-traversal protection at the route boundary via a
  strict filename whitelist regex; symlink protection via
  post-resolve `is_relative_to` check on every file op. Settings
  file is written atomically (`tmp + os.replace`) so an interrupted
  process can't truncate the user's saved library path. The library
  directory is auto-created on first save.
- `replaceAllWorkflows` now uses the same `snapshotCache` rollback
  primitive as `persistMutation` so a snapshot apply that fails to
  persist rolls the in-memory cache back to its pre-call state — the
  load is fully atomic. The `workflows_store.js` mutator-invariants
  doc block lists `replaceAllWorkflows` as the fourth legitimate
  rebind site.
- The Load dialog now gates the `currentPresetName` tracker on
  `picksOk && workflowsOk`. Partial-failure paths clear the tracker
  so the next Save forces a fresh name prompt rather than offering
  to overwrite the on-disk preset with corrupted half-state.
- The Load dialog clears the tracker if the user deletes the
  currently-loaded preset.
- Client-side `sanitizeName` now mirrors the server's filename
  whitelist regex — invalid characters collapse to `_` rather than
  hitting an opaque HTTP 400 from the server. The `presetExists`
  probe is now tri-state (true/false/null); the Save flow refuses
  to write when it can't reach the library to verify the name.
- Server error reasons are surfaced via `await resp.text()` rather
  than `resp.statusText`, since HTTP/2 (RFC 7540) strips reason
  phrases — behind any HTTP/2-terminating proxy `statusText` is
  empty and the server's helpful "read-only mount" / "invalid
  filename" / "parent missing" messages would otherwise be lost.

### Changed
- **Save selection toast distinguishes "no selection" from "selection
  points at deleted nodes."** Previously both produced the generic
  "Select one or more nodes on the canvas first." Now the deleted-node
  case (selection set non-empty but every id refers to a removed node,
  common after undo/redo) shows "Selected node(s) no longer exist.
  Click a node on the canvas to re-select." `serializeSelection`
  returns a discriminated `{ kind: "empty" | "stale" | "ok", graph? }`
  result so the caller can route messaging precisely.
- **`curated_defaults.json` retired in favor of `starter_preset.json`.**
  Fresh installs no longer get picks seeded directly into localStorage
  on first load. Instead the bundled `web/starter_preset.json` (full
  snapshot format — picks + workflows + tags + archive) is copied into
  the user's snapshot library directory as `starter.json`, and the user
  opens Snapshot → Load to apply it in one click. The change unifies
  fresh-install distribution with the per-user snapshot library
  (#68): the maintainer flow becomes "build state in ComfyUI → click
  the ↓ Tools-row button → paste over `web/starter_preset.json` →
  commit." Existing users with non-empty picks are explicitly skipped
  by the new seeder, so their state is untouched. Removed:
  `seedDefaultsIfNeeded`, `exportPicks`, `web/curated_defaults.json`,
  `SEEDED_KEY`, `DEFAULTS_URL`. Added: `seedStarterPresetIfNeeded`,
  `exportStarterPreset` (both in `web/sidebar/snapshot.js`),
  `STARTER_SEEDED_KEY`, `STARTER_URL`, `STARTER_PRESET_FILENAME`,
  `web/starter_preset.json`.
- **Sidebar toolbar: Tools row split out from Nodes row.** The Export
  button (now "Export starter preset") and the "Install missing for
  picks" button moved up out of the Nodes row into a new dedicated
  **Tools** row above the search field, alongside a new "Drop
  placeholders onto canvas" button. The Nodes row keeps only the
  everyday `+` Add button. Reasoning: Export and Install-missing are
  admin/advanced operations; segregating them keeps the daily flow
  uncluttered and makes it harder to fire them by accident. The new
  Drop-placeholders button is the `security_level=normal` escape hatch
  for "install missing" — it instantiates one placeholder per missing
  pack on a fresh canvas tab so ComfyUI/Manager's standard "Install
  Missing Custom Nodes" detection picks them up and routes the install
  through Manager's UI flow (which doesn't go through
  `/customnode/install/git_url`'s security gate the way our
  programmatic call does).

### Internal (sidebar tidiness pass — no behavior change)
- **Lifted save-modal action sentinels and the cascade picker
  sentinels into named constants** at the top of `showSaveWorkflowModal`
  in `web/sidebar/modals.js` (`ACTION_NEW`, `ACTION_USE_EXISTING`,
  `ACTION_MODIFY_EXISTING`, alongside the existing `NEW_TOP` /
  `SAVE_HERE`). Magic strings (`"new"`, `"use_existing"`,
  `"modify_existing"`) are gone from the function body.
- **Section-id constants** (`SECTION_ID_NODES`, `SECTION_ID_WORKFLOWS`,
  `SECTION_ID_TAGS`) lifted to module scope in `web/sidebar/tree.js`.
  The `pinExpanded` save-flow callsite now references the constant
  instead of the literal `"workflows"` string, so a future section-id
  rename only needs the constant updated.
- **`modalLabel(text)`** factory in `web/sidebar/modals.js` replaces
  the seven repetitions of the four-line `createElement("label")` +
  `className` + `textContent` + `appendChild` pattern.
- **`USERDATA_OVERWRITE_QUERY = "?overwrite=true"`** named constant in
  `web/sidebar/workflows_store.js`. ComfyUI's userdata API requires
  this flag to allow POST over an existing file; pinning it as a
  constant makes the contract visible at a glance.
- **`buildFolder` no longer round-trips folder-expansion state through
  `wrapper.dataset.expanded`.** The DOM dataset forces every value
  to a string (and led to the slightly awkward `!== "false"` read);
  expansion state now lives in a closure variable. `pathStates` is
  the canonical store for cross-render persistence — the dataset was
  just mirroring it for reads.
- **`subcategoryFor` fallback path now logs a `console.warn`** with
  the offending category and `categoryRoot`. Reachable only via a
  logic error (typically a stale `REPOS` entry whose `categoryRoot`
  no longer matches the upstream node category prefix); previously
  it silently rewrote the category, masking the misconfig.
- **Save-modal `onSave` callback contract renamed `dir` → `dirPath`**
  to match the codebase-wide convention (`dirPath` for arrays of
  segments, `dirName` for single segments, `dir` for resolved
  DirNode objects). Caller updates in `tree.js` paired.
- **`makeToolbarButton({iconClass, title, onClick})`** factory in
  `web/sidebar/tree.js` replaces the four near-identical button-construction
  blocks in `renderPanel` (export, new-dir, save-canvas, save-selection).
  Each block was 5–7 lines of `createElement` / `className` / `innerHTML`
  / `title` / `addEventListener`; calls collapse to 4-line factory invocations.
  No behavior change. Closes the "extract `makeToolbarButton`" item on #47.
- **`makeToolbarButton({iconClass, title, onClick})`** factory in
  `web/sidebar/tree.js` replaces the four near-identical button-construction
  blocks in `renderPanel` (export, new-dir, save-canvas, save-selection).
  Each block was 5–7 lines of `createElement` / `className` / `innerHTML`
  / `title` / `addEventListener`; calls collapse to 4-line factory invocations.
  No behavior change. Closes the "extract `makeToolbarButton`" item on #47.
- **Dropped duplicate `.koolook-export-btn` CSS class.** It was identical to
  `.koolook-icon-btn` minus the `:disabled` state. The export button now
  uses only `.koolook-icon-btn` via `makeToolbarButton`. Closes the
  "drop duplicate CSS class" item on #47.
- **`buildFolder({path})` default removed.** `path` is now a required
  parameter; the previous `path = null` default was unreachable since every
  caller routes through `makeSectionCtx`, which always supplies a non-empty
  section-prefixed string. The runtime `if (path && …)` guard inside
  `buildFolder` is also dropped. Closes the "remove unreachable default"
  item on #47.
- **`workflowsCache` mutator invariants documented inline** above the public
  mutator block in `web/sidebar/workflows_store.js`. Four rules a future
  contributor needs before adding a new mutator: pair every mutate with a
  commit (or use `persistMutation`), return `false` for no-op vs. truthy for
  success, mutate in place, and never replace `workflowsCache` outside the
  seed/load paths. Closes the "document `workflowsCache` invariants" item
  on #47.

### Added
- **Right-click "Duplicate…"** on any workflow row in the sidebar tree.
  Opens a name modal pre-filled with `<name> (copy)`; saving deep-clones
  the source graph into a new entry in the same directory. The duplicate
  inherits the source's tags so the user's categorization carries over.
  Same-name duplicates fall through to the existing archive-on-collision
  behavior in `saveWorkflowEntry`. (Closes #58.)
- **Right-click "Tags…"** on any workflow row. Opens a chip-style modal
  to view, add, and remove the workflow's tags one at a time. Each
  edit fires its own `persistMutation` so changes survive a mid-edit
  close. (Part of #56.)
- **Tags sidebar section.** A new section between Workflows lists every
  tag in use across the active workflow tree. Each tag becomes a folder
  whose entries are the tagged workflows (sorted A→Z); click loads the
  workflow from its real directory. Archived workflows are filtered
  out of the section so the active view stays clean — their tags are
  still preserved on the entry, so a restore from the Archive folder
  brings them back. Search matches tag name OR workflow name. (Part
  of #56.)
- **Right-click "+ New directory…" / "+ New subdirectory under <path>…"**
  in the workflow row's Move-to flow. Both create a fresh directory and
  move the workflow into it as one atomic mutation; if the move can't
  land, the new directory is rolled back so the cache never leaks an
  empty orphan. (Closes #57.)
- **Recursive subdirectories under workflow directories.** Right-click any
  directory in the Workflows tree → "Create subdirectory…" to nest folders
  to arbitrary depth (e.g. `UP-scale / Type-A / Sharp`). Each nested
  directory behaves like a top-level one: it can hold workflows + an
  Archive subfolder + further subdirectories. The save modal directory
  picker is **cascading** (multi-step): pick a parent, then a child
  appears, then a grandchild, etc. — each child level has a `(save in
  "<path>")` option to stop drilling at that depth. The right-click
  workflow "Move to…" submenu lists every other path in the tree.
- **Delete archive (N) on the synthetic Archive folder.** Right-click the
  Archive folder under any directory → bulk-removes every archived
  workflow at that level in one confirm. Active workflows in the same
  directory are untouched.
- **Drag-and-drop in the workflows tree** (Tier 1 — moves only, no
  reordering). Drag a workflow onto a directory to move it. Drag a
  workflow onto an Archive folder to archive it (cross-directory drops
  move + archive in one go). Drag a directory onto another directory to
  nest it as a child. Cycle prevention rejects dropping a dir into
  itself or any of its descendants. Sort within a level stays
  alphabetical. (Custom ordering would be Tier 2.)
- **Schema is now recursive** — every directory node has a `workflows`
  object AND a `directories` object. Existing v0.2 stores load fine:
  `normalizeWorkflowsStore` treats a missing `directories` as `{}` and
  the rest of the code assumes it always exists post-normalization.

### Changed
- **Sidebar tab renamed:** "Curated Nodes" → **"Kforge Labs"**. Tooltip
  also updated. The tab id (`koolook.curatedNodes`) and the
  `app.registerExtension` name are unchanged so existing per-user tab
  state (pinning, ordering) is preserved.
- **Save modal — `Base on existing` candidates now walk leaf-up to root
  and dedupe (deepest wins).** Saving into an empty subdirectory like
  `UP-scale / seedvr2` no longer disables the existing-name actions —
  the modal pulls active workflows from every ancestor. Ancestor
  entries are labeled `<name>  ·  in <path>` so the candidate's
  source directory is unambiguous. The selected destination path is
  whatever the cascade resolves to — the ancestor source only seeds
  the workflow name; archive semantics still apply at the destination.
- **Save modal — `Action` dropdown is now hidden (not just disabled)
  when no base candidate exists** anywhere in the destination's
  ancestry. Disabled `<option>` elements render too subtly across
  browsers; the only useful path in that case is "type a fresh name
  and save," which the Workflow Name field below already provides.
  The underlying value is pinned to `new` so submit takes the
  by-name path.
- The right-click canvas-node menu item is now **"Add to Kforge Labs"**
  (was "Add to Curated Sidebar").
- Directory header counts in the workflows tree now show the total
  workflows in the **whole subtree** (active + archived + descendants),
  not just direct children. A parent with empty direct workflows but
  populated subdirectories no longer shows "0".

### Internal
- Workflow operations are now path-addressed: every mutator and lookup
  takes a `string[]` path (e.g. `["UP-scale", "Type-A"]`). The store's
  internal API: `addDirectory(parentPath, name)`,
  `renameDirectory(parentPath, old, new)`, `deleteDirectory(parentPath, name)`,
  `saveWorkflowEntry(path, wfName, graph)`, `archive/unarchive/rename/deleteWorkflow(path, wfName)`,
  `moveWorkflow(srcPath, wfName, dstPath)`, plus new helpers
  `listAllDirectoryPaths()` and `pathsEqual(a, b)`.
- Reserved-name check: subdirectory names cannot be `Archive`
  (case-insensitive) at any non-root level — collides with the synthetic
  Archive folder rendered for archived workflows.
- **Workflow entries gain optional `tags: string[]`.** `normalizeDirNode`
  trims, drops empties, and dedupes case-sensitively, so old entries
  without a tags field load as `tags: []` and the rest of the code can
  assume the field always exists. New mutators in `workflows_store.js`:
  `getWorkflowTags(path, wfName)`, `addTag(path, wfName, tag)`,
  `removeTag(path, wfName, tag)`. `getWorkflowGraph(path, wfName)` is
  now also exported so the duplicate flow can deep-clone without
  reaching into the cache directly.
- **`showTagsModal` in `modals.js`** — chip-row UI with add/remove
  callbacks, mirrors the existing `showInputModal` / `showConfirmModal`
  surface so the modal shell, escape-key teardown, and overlay-click
  dismissal stay centralized.

### Documentation
- Closed out issue #28 (de-vendor upstream code under `upscaler_FIX/`
  and `nuke_CAM_exporter/`) by adding the audit-trail layer that the
  v0.1.4 / v0.1.5 registry-hygiene cleanup deferred:
  - `forks/THIRD_PARTY.md` gained a "De-vendored upstream code" section
    listing all six untracked trees with former path, upstream (where
    pinned) and per-tree provenance notes. Preamble now also documents
    the publish-history finding that none of these files were ever in
    a successful Comfy Registry publish.
  - `forks/forks_manifest.yaml` gained six entries with the `_devendored`
    suffix and `status: "removed"`, populating `source_repo` /
    `source_ref` / `local_paths` / `removed_in_release` / `license` per
    issue #28's acceptance criterion (2). Best-effort where upstream
    URLs were not pinned at vendor time. Closes #28.

## [0.2.0] - 2026-05-04

### Removed (BREAKING for any saved workflow using `Easy_Version`)
- Dropped `Easy_Version` (and the `k_easy_version.py` source) — the
  whole node was just a one-liner that turned an integer N into the
  string `vNNN` (zero-padded). Maintainer concluded it was the first
  trivial node they ever wrote and saw no real value. Anyone who needs
  this exact behavior can either:
  - inline the format string in their workflow downstream of an INT
    primitive (`f"v{N:03d}"`), or
  - use any of the more general string-format nodes in the ecosystem
    (KJ Nodes, ComfyUI-Custom-Scripts, etc.).
- Saved workflows that reference the `Easy_Version` ID will fail to
  load — same migration as v0.1.5's `__koolook_v1_0_1` cleanup.

### Fixed (carried over from PR #39, [Unreleased] in 0.1.8)
- `Easy_hdr_VAE_encode` (Koolook v2.3.3) now wraps the encoded tensor in
  the standard ComfyUI `LATENT` dict (`{"samples": t}`) instead of
  returning the raw tensor. Wiring this node into KSampler previously
  crashed with `IndexError: too many indices for tensor of dimension 5`
  on Wan 2.2 video workflows, because KSampler does
  `latent["samples"]` and the raw 5-D tensor doesn't support string
  indexing. The decoder side was already correct.
- `Easy_hdr_VAE_encode` / `Easy_hdr_VAE_decode` (Koolook v2.3.3) now
  produce proper N-frame sequences end-to-end on Wan 2.2 / Hunyuan /
  CogVideoX / LTX video workflows. Two combined fixes:
  1. **Rank-aware dispatch.** 5-D `(B, F, H, W, C)` video tensors paired
     with a 2-D image VAE are iterated frame-by-frame and stacked along
     a temporal axis on encode (and per-frame decoded then concatenated
     to a `(B*F, H, W, C)` ComfyUI IMAGE batch on decode). 5-D tensors
     with a 3-D-native VAE — identified via `vae.latent_dim == 3`,
     which is the actual attribute ComfyUI sets on its VAE class for
     video VAEs — pass through unchanged.
  2. **5-D output normalization on decode.** 3-D-aware VAEs return a
     5-D `(B, F, H, W, C)` tensor from `vae.decode()`. Without an
     explicit reshape, downstream IMAGE-typed nodes (Get Image Size,
     SaveImage) misread that as a single 4-D image with `count=1` and
     `height=F` — exactly the symptom we were debugging
     (`1056×41 count=1` after a 41-frame Wan 2.2 sequence). The fix
     mirrors the stock ComfyUI VAEDecode node
     ([nodes.py:303-304](https://github.com/comfyanonymous/ComfyUI/blob/master/nodes.py#L303-L304)):
     `if img.ndim == 5: img = img.reshape(-1, H, W, C)` — the leading
     `(B, F, …)` dims collapse into the standard ComfyUI batch axis,
     so SaveImage now writes `_00001.png … _0004N.png` instead of one
     squashed frame.

  The new `path=image | video-iter | video-3d` field in `debug_info`
  surfaces which dispatch branch fired per run, for live verification.

### Added (Easy_hdr_VAE_encode / decode feature parity with upstream)
- Restored the cinema-grade color-management surface from upstream
  Radiance v2.3.3's `RadianceVAE4KEncode/Decode`, while keeping the
  slim no-tile-engine code path that lets these nodes work with
  Wan 2.2 video VAEs (the original motivation for the Koolook fork —
  upstream's 4K cosine-blend tile engine errors with `"size of tensor a
  (192) must match the size of tensor b (132) at non-singleton dimension 4"`
  on video VAEs because the tiler's spatial alignment fights the video
  VAE's internal temporal-aware encoding).
- **12 source / target color spaces** (was 4): Linear, ACEScg,
  **ACES 2065-1**, **Rec.2020 Linear**, sRGB, Raw, plus six cinema log
  curves — **ARRI LogC3** (EI-aware, default 800), **ARRI LogC4**,
  **Sony S-Log3**, **Panasonic V-Log**, **DaVinci Intermediate**,
  **RED Log3G10**. Round-trip math (linearize → encode) and matrix
  constants ported verbatim from upstream's `color_utils.py` with the
  bug-fix history preserved in the docstrings. Conversions are
  rank-agnostic and 4-channel-aware (alpha passes through untouched).
- **`Compress (Log)` HDR mode** — the upstream HDR-clamp-free pipeline.
  On encode, the linear-space tensor is re-encoded through a cinema log
  curve (matches `source_space` if it's a log space, else ARRI LogC4)
  and goes into the VAE without a hard clamp. On decode, the VAE output
  is run through `_soft_log_shoulder` (tanh rolloff at the per-profile
  knee instead of a hard clamp at 1.0) and `_denoise_log_highlights`
  (3×3 box blur in log space, ramped quadratically toward the highlight
  region) before log→linear conversion. Per-profile parameters
  (`LOG_PROFILE_HDR_PARAMS` table) tune knee / ceiling / denoise
  threshold / strength to each curve's slope at code 1.0 — RED Log3G10
  gets the most conservative settings, Sony S-Log3 / ARRI LogC4 the
  loosest. Pair `hdr_mode="Compress (Log)"` on both encode and decode
  for HDR-clean roundtrips; mismatched modes produce garbage by design.
- **`latent_sampling`** parameter (`sample` / `mean` / `mode`).
  `sample` is ComfyUI's default random posterior sample. `mean` and
  `mode` use the posterior mean for deterministic, lowest-noise
  encoding — best for img2img where minimum reconstruction noise
  matters. The mean/mode path replicates ComfyUI's preprocessing
  (BHWC → BCHW → [-1, 1] scaling) before reaching `first_stage_model`,
  with a graceful fallback chain that never crashes.
- **Real alpha output.** `Easy_hdr_VAE_encode` now returns
  `(LATENT samples, STRING debug_info, IMAGE alpha)`. With
  `alpha_handling="Preserve"` and a 4-channel input, the alpha channel
  is surfaced as a separate IMAGE for downstream re-compositing
  post-decode (VAEs don't encode alpha, so it's routed around them).
  With `alpha_handling="Ignore"` or a 3-channel input, alpha is a
  zeros tensor of compatible shape — downstream wiring never fails.
  The previous `alpha_handling` flag was a documented no-op.
- **Exposure now applied in linear space** for all source spaces. The
  previous behavior multiplied raw input bytes by `2^exposure` even
  when the input was sRGB-gamma or log-coded, which produced visually
  wrong results for non-linear sources. Linear sources are unaffected
  (linear × 2^stops is the correct semantic). Raw source still does a
  raw-domain multiplication so users who know what they're feeding the
  VAE keep their bytes intact.
- New helper module
  [`forks/radiance_koolook/versions/v2_3_3/color_helpers.py`](forks/radiance_koolook/versions/v2_3_3/color_helpers.py)
  contains all log curves, soft-shoulder / log-denoise helpers,
  `encode_with_sampling_mode`, color-space matrices, and dispatch
  tables. `nodes_vae.py` keeps the node wiring + sequence dispatch.

### Intentionally NOT ported (upstream-only features)
- 4K cosine-blend tile engine and its `tile_size` / `overlap` /
  `processing_mode` knobs — this is the broken plumbing that motivated
  the slim fork in the first place.
- `inverse_tonemap` / `target_stops` (SDR→HDR expansion).
- `.rhdr` sidecar export and `rhdr_precision` (Radiance Viewer-specific).
- `crop_padding` (only relevant when tiling).

These can be added back in future patches if needed; the slim wrapper
stays at ~600 lines of node wiring + ~450 lines of color helpers,
~40% the size of upstream's `vae.py` (2,638 lines).

### Added (carried over from PR #40, also unreleased on main)
- `scripts/sync_to_dev.py` — pure-stdlib helper that copies the curated
  runtime files (`__init__.py`, `config.json`, top-level `k_*.py`,
  `forks/`, `web/`) into a live ComfyUI `custom_nodes/<pack>/` folder
  for fast local iteration without bumping a version. Reads the target
  path from the new `KOLOOK_COMFYUI_DEV_PATH` env var (auto-loads `.env`
  from repo root); errors out cleanly when unset.
- `.env.example` — `KOLOOK_COMFYUI_DEV_PATH=` entry with comment.
- `CLAUDE.md` — new "`dev-sync`" trigger-phrase section so the agent
  knows to run `scripts/sync_to_dev.py` when the maintainer says
  "dev-sync" / "sync dev" / "copy those files" mid-session.

### Net effect
- Node count drops from 8 to **7**.
- `Koolook/Pipeline` subfolder now has only 1 entry (`Easy AI Pipeline`)
  instead of 2.
- Faster local dev loop: a one-line tweak no longer requires cutting a
  full release just to test it inside ComfyUI.

## [0.1.8] - 2026-05-03

### Changed (categories — affects ComfyUI node-add menu)
- `Koolook/VFX` is gone. Three more granular subfolders replace it:
  - **`Koolook/Pipeline`** — `Easy_Version`, `EasyAIPipeline` (workflow
    setup nodes)
  - **`Koolook/Image`** — `EasyResize_Koolook`, `easy_ImageBatch` (joins
    `EasyResize_Koolook`; `easy_ImageBatch` was previously in VFX)
  - **`Koolook/VAE`** — `Easy_hdr_VAE_encode`, `Easy_hdr_VAE_decode`
    (now have their own subfolder; previously also in VFX)
- Workflows continue to load and run unchanged — `CATEGORY` is purely a
  UI-organization hint and only affects the node-add menu hierarchy.

### Fixed (search discoverability)
- `Easy_hdr_VAE_encode` / `Easy_hdr_VAE_decode` display names are now
  **"Easy HDR VAE Encode (Koolook)"** / **"Easy HDR VAE Decode (Koolook)"**.
  Previously the display name equalled the node ID (`Easy_hdr_VAE_encode`
  with no "Koolook" string), so typing `koolook` in ComfyUI's node-add
  search filter excluded them — they appeared registered but invisible.
  Now they group with the rest of the pack under that filter.
- Class IDs (`Easy_hdr_VAE_encode`, `Easy_hdr_VAE_decode`) are unchanged —
  saved workflows that reference these IDs continue to load.

## [0.1.7] - 2026-05-03

### Changed
- All node display names now suffix `(Koolook)` so they surface together
  when users search "koolook" in the ComfyUI node-add menu. Previously
  only `KoolookLoadCameraPosesAbsolute` (and the v2.3.3 fork nodes via
  their version suffix) matched.
  - `Easy_Version` → `Easy Version (Koolook)`
  - `EasyAIPipeline` → `Easy AI Pipeline (Koolook)`
  - `easy_ImageBatch` → `Easy Image Batch (Koolook)`
  - `EasyWan22Prompt` → `Wan 2.2 Easy Prompt (Koolook)`
  - `Easy_Version` category moved from `VFX/Utils` to `Koolook/VFX`
    to match the rest of the pack.
- Class keys (`NODE_CLASS_MAPPINGS`) are unchanged — saved workflows keep
  loading and running unchanged.
- Carried in via PR #36, then bundled into this release alongside the
  documentation reorg below.

### Docs reorg (root cleanup)
- Moved out of repo root (preserving git history via `git mv`):
  - `Glossary.md` → [`docs/reference/glossary.md`](docs/reference/glossary.md)
  - `RELEASING.md` → [`docs/maintainers/releasing.md`](docs/maintainers/releasing.md)
- Root now keeps only the four files that *must* live there:
  `README.md` (GitHub repo home + Comfy Registry description),
  `LICENSE` (`pyproject.toml` reference + GitHub license badge),
  `CHANGELOG.md` (tooling convention),
  `CLAUDE.md` (Claude Code agent instructions).
- New `docs/` structure with three audience buckets:
  - [`docs/user_guide/`](docs/user_guide/) — end-user, per-node guides + screenshots (`img/`)
  - [`docs/reference/`](docs/reference/) — lookup material (glossary, node inventory)
  - [`docs/maintainers/`](docs/maintainers/) — project-internal procedures
- Each bucket gets a short index `README.md` so the structure is
  navigable from `docs/README.md` downwards without grep.

### Added (new maintainer docs)
- [`docs/maintainers/registry-api.md`](docs/maintainers/registry-api.md) —
  documents the **undocumented** Comfy Registry version-management API
  that we reverse-engineered while cleaning up v0.1.0–0.1.5
  (deprecate / undeprecate / yank endpoints, status enum values,
  auto-deprecation behavior, ready-to-paste curl recipes, and an
  optional `registry-mgmt.yml` GitHub Actions workflow for codified
  version management without ever exposing the token to a shell).
- [`docs/maintainers/node-versioning.md`](docs/maintainers/node-versioning.md) —
  codifies the rules from the v0.1.5 PR-review discussion for safely
  changing a node's `INPUT_TYPES` / `RETURN_TYPES` / class names without
  breaking saved user workflows. Five rules + the suffix-version pattern
  + the alias-then-deprecate migration path + concrete worked examples.
- Cross-references in `README.md`, `CLAUDE.md`,
  `.github/ISSUE_TEMPLATE/release_checklist.md`, and the
  `add-external-fork` skill all updated to point at the new paths.
- Imported the existing untracked `docs/` images that the maintainer
  had already started adding locally (node inventory screenshot under
  `reference/`, Easy Image Batch helper image under `user_guide/img/`).

## [0.1.6] - 2026-05-03

### Renamed (with back-compat alias)
- `EasyResize` is now exposed canonically as **`EasyResize_Koolook`**
  (display: `Easy Resize (Koolook)`) to resolve a node-ID collision with
  `ComfyUI-EasyFilePaths`, which also registers the bare name `EasyResize`.
  The old `EasyResize` ID is kept as a **deprecated alias** pointing at
  the same class, so saved workflows still load and run unchanged. The
  alias will be removed in a future major release once the deprecation
  has had time to propagate.

### Attribution
- Added a proper SPDX header + GPL-3.0 attribution block to
  `k_easy_resize.py`, crediting `kijai/ComfyUI-KJNodes` (`Resize Image V2`)
  as the original interface inspiration. The Koolook implementation is a
  fresh write that materially extended the surface (aspect-ratio parser,
  keep_proportion modes, mask + composed outputs, device selection,
  target/original W/H reporting, color-panel passthrough). KJ Nodes is
  GPL-3.0, same as our pack — no relicense required.
- `forks/THIRD_PARTY.md` and `forks/forks_manifest.yaml` upgraded the
  KJ Nodes entry from `license: "unknown"` /
  `sync_state: "needs-upstream-reference"` to verified GPL-3.0 with full
  per-feature change notes.

### Notes for users
- If a saved workflow uses the bare `EasyResize` ID, it still works but
  the node's display name now reads
  `Easy Resize (deprecated, use 'Easy Resize (Koolook)')` as a hint to
  swap. New workflows should pick `EasyResize_Koolook` from the node-add
  menu.

## [0.1.5] - 2026-05-03

### Removed (BREAKING for anyone using the v1_0_1 namespaced IDs)
- Dropped the entire `forks/radiance_koolook/versions/v1_0_1/` folder
  (~5,200 lines, 26 namespaced nodes). The wrappers were vestigial —
  Koolook authors never used them, no internal workflow referenced any
  `__koolook_v1_0_1` suffixed ID, and the VAE pair was already
  superseded by `Easy_hdr_VAE_encode` / `Easy_hdr_VAE_decode` in v2_3_3.
- IDs that no longer exist after this release:
  `ImageToFloat32__koolook_v1_0_1`, `Float32ColorCorrect__koolook_v1_0_1`,
  `HDRExpandDynamicRange__koolook_v1_0_1`, `HDRToneMap__koolook_v1_0_1`,
  `ColorSpaceConvert__koolook_v1_0_1`, `SaveImageEXR__koolook_v1_0_1`,
  `LoadImageEXR__koolook_v1_0_1`, `LoadImageEXRSequence__koolook_v1_0_1`,
  `SaveImage16bit__koolook_v1_0_1`, `HDRHistogram__koolook_v1_0_1`,
  `LogCurveEncode__koolook_v1_0_1`, `LogCurveDecode__koolook_v1_0_1`,
  `HDRExposureBlend__koolook_v1_0_1`,
  `HDRShadowHighlightRecovery__koolook_v1_0_1`,
  `OCIOColorTransform__koolook_v1_0_1`,
  `OCIOListColorspaces__koolook_v1_0_1`, `GPUTensorOps__koolook_v1_0_1`,
  `HDR360Generate__koolook_v1_0_1`, `SaveHDRI__koolook_v1_0_1`,
  `ACES2OutputTransform__koolook_v1_0_1`,
  `DaVinciWideGamut__koolook_v1_0_1`, `ARRIWideGamut4__koolook_v1_0_1`,
  `RadianceVAEEncode__koolook_v1_0_1`,
  `RadianceVAEDecode__koolook_v1_0_1`, `k_easy_OCIO_v101`
  (the short ID for `RadianceOCIOColorTransformV2`),
  `RadianceLogCurveDecode__koolook_v1_0_1`.
- Migration path: install upstream Radiance directly
  (https://github.com/fxtdstudios/radiance) for the HDR/EXR/OCIO
  functionality. Use `Easy_hdr_VAE_encode/decode` (already in v0.1.3+)
  for video VAE workflows.
- Source recoverable via `git checkout HEAD~1 -- forks/radiance_koolook/versions/v1_0_1/`
  if you ever realize you need any of those wrappers back.

### Added (registry hygiene from the original v0.1.4 plan)
- `.gitignore` now excludes `upscaler_FIX/` and `nuke_CAM_exporter/` —
  the maintainer's local dev workspaces accidentally committed in
  Dec 2025. These were never imported by the package's root
  `__init__.py`, so they had no runtime effect, but the Comfy Registry's
  static scanner picked up `NODE_CLASS_MAPPINGS` from vendored 3rd-party
  clones inside them and counted ~12 spurious nodes against this pack
  in ComfyUI-Manager (yielding the misleading "44 nodes / 13 conflicts"
  badge).

### Removed (registry hygiene)
- `git rm -r --cached upscaler_FIX nuke_CAM_exporter` — 70 files
  untracked from git, files stay on the maintainer's local disk for
  reference. ~3.7 MB of unrelated content out of the published archive.

### Net effect on Manager / registry
- Node count drops from 44 to **8** (6 root Koolook + 2 v2_3_3 VAE).
- Spurious "Conflict with `ComfyUI-SuperUltimateVaceTools`" warnings
  disappear.
- Published archive shrinks by ~9 MB total (3.7 MB dev workspaces +
  ~5.2 MB v1_0_1 fork code).

### Notes for the maintainer
- After merging this PR and `git pull`-ing main, your local working tree
  will lose the `upscaler_FIX/` and `nuke_CAM_exporter/` folders (git
  applies the deletion). Back up first or restore via
  `git checkout HEAD~1 -- ...`. The `upscaler_FIX/` folder has already
  been physically moved to `../ComfyUI-Forks-BK/`; the
  `_Utils-CAM-track/` subfolder of `nuke_CAM_exporter/` has been moved
  to `../ComfyUI-Tools-BK/nuke_CAM_exporter/`. The remainder of
  `nuke_CAM_exporter/` (your actual Nuke pipeline work) is still on
  disk in MAIN but no longer tracked by git.

## [0.1.4] - 2026-05-03 (test-published only, superseded by 0.1.5)

### Removed (registry hygiene)
- Untracked the maintainer's local dev workspaces from git: `upscaler_FIX/`
  and `nuke_CAM_exporter/`. These were never imported by the package's root
  `__init__.py`, so they had no effect on what ComfyUI loaded at runtime,
  but the Comfy Registry's static scanner picked up the `NODE_CLASS_MAPPINGS`
  dicts inside vendored 3rd-party clones and counted them as part of this
  pack — yielding a misleading "44 nodes / 13 conflicts" badge in
  ComfyUI-Manager (against `ComfyUI-SuperUltimateVaceTools`,
  `ComfyUI-multigpu`, etc.). Files removed from index via
  `git rm -r --cached` (still on the maintainer's local disk) and the two
  paths are now `.gitignore`d so they cannot leak again.
- Net effect: published archive shrinks by ~3.7 MB (70 files), Manager's
  node count drops from 44 to ~32 (the actual runtime registrations:
  6 root Koolook + ~26 namespaced fork variants), and the spurious
  conflict warnings disappear. Same class of issue as the v0.1.2 GPL-3.0
  relicense — vendored 3rd-party code in MAIN, this time without the
  license-compatibility risk because none of it was ever imported.

### Notes for the maintainer
- After merging this PR and `git pull`-ing main, your local working tree
  will lose the `upscaler_FIX/` and `nuke_CAM_exporter/` folders (git
  applies the deletion). If you want to keep working on those locally,
  back them up first (`cp -r upscaler_FIX nuke_CAM_exporter ~/backup/`)
  or restore from history afterwards
  (`git checkout HEAD~1 -- upscaler_FIX nuke_CAM_exporter`); they will
  then live as untracked files in your working tree, ignored by the new
  `.gitignore` rules.

## [0.1.3] - 2026-05-03

### Renamed
- The new v2_3_3 VAE nodes are now exposed as `Easy_hdr_VAE_encode` /
  `Easy_hdr_VAE_decode` (clean IDs and display names — no
  `__koolook_v2_3_3` suffix), to avoid visual collision with upstream
  Radiance v2.3.3's `RadianceVAEEncode` / `RadianceVAEDecode` aliases
  in the ComfyUI node-add search. The version is still tracked
  structurally (file lives in `versions/v2_3_3/`) and textually
  (file header + UPSTREAM_PIN.yaml + forks/THIRD_PARTY.md). Other
  Koolook nodes in the v2_3_3 set (none today, but possible in future)
  would still use the `__koolook_v2_3_3` suffix by default — opt out
  via the new `SKIP_VERSION_SUFFIX` set in `versions/v2_3_3/__init__.py`.
- 0.1.2 was test-published to the registry with the previous
  `RadianceVAEEncode__koolook_v2_3_3` IDs but never tagged or formally
  released; bumping to 0.1.3 publishes the renamed version cleanly.

## [0.1.2] - 2026-05-03 (test-published only, superseded by 0.1.3)

### License (BREAKING)
- **Relicensed entire package to GPL-3.0.** v0.1.0 and v0.1.1 shipped under
  a claimed MIT license while already incorporating GPL-3.0-derived code from
  [fxtdstudios/radiance](https://github.com/fxtdstudios/radiance) under
  `forks/radiance_koolook/`. GPL-3.0 §5(c) requires the entire combined work
  to be GPL-3.0; relicensing aligns the package with what we actually ship
  and matches the dominant license posture of the ComfyUI custom-node
  ecosystem. Downstream users incorporating, linking to, or deriving from
  ComfyUI-Koolook must now distribute under GPL-3.0 (or compatible).
- Added `LICENSE` file at repo root with the full GPL-3.0 text
  (`pyproject.toml` previously referenced a `LICENSE` file that did not
  exist on disk).
- README "License" section rewritten to declare GPL-3.0 and explain the
  §5(c) implication.

### Added
- **`forks/radiance_koolook/versions/v2_3_3/`** — slim, video-friendly
  re-implementation of `RadianceVAEEncode` / `RadianceVAEDecode` exposed
  under the namespace suffix `__koolook_v2_3_3`. Mirrors the *interface
  surface* of upstream `RadianceVAE4KEncode` / `RadianceVAE4KDecode`
  but skips the 4K cosine-blend tile engine, which conflicts with
  modern video VAEs (Wan 2.2, Hunyuan, CogVideoX, LTX) that already
  handle their own temporal/spatial stitching internally. Fixes the
  `"size of tensor a (192) must match the size of tensor b (132) at
  non-singleton dimension 4"` runtime error users hit when chaining
  upstream's VAE encoder into Wan 2.2 video workflows.
- `forks/THIRD_PARTY.md` — full attribution entries for the v1.0.1
  baseline (already in the repo) and the new v2.3.3 VAE subset, with
  per-class change notes.
- `.claude/skills/license-pre-check/` — blocking license-compatibility
  audit skill for Claude Code. Run **before** copying or porting any
  third-party code; refuses to proceed on incompatible combinations
  (e.g. GPL upstream into MIT downstream).
- `.claude/skills/add-external-fork/` — Claude Code port of the
  existing Cursor skill, with a mandatory Phase 0 ("run
  license-pre-check first") and a Phase 3c requirement to add GPL §5(a)
  modification headers on derived files.
- `RELEASING.md` — canonical, step-by-step release procedure (was previously
  ad-hoc; the gaps it closes are exactly what caused the `v0.1.0` orphan-tag
  and `CristianP` `PublisherId` incidents).
- README "Release & Stability" section now links `RELEASING.md` and the
  per-release checklist template.
- `CLAUDE.md` now references `RELEASING.md` for agents.

### Changed
- `forks/forks_manifest.yaml` — explicit `license: "GPL-3.0"` and
  `license_verified_at` fields on the existing radiance entry (was
  `to_verify_at_source_ref`); pre-registered the new v2.3.3 VAE entry
  with verified license metadata.
- `.github/ISSUE_TEMPLATE/release_checklist.md` rewritten to mirror
  `RELEASING.md`, including the registry-publisher validation step.

### Notes for users
- If your workflow currently references the bare-name `RadianceVAEEncode`
  or `RadianceVAEDecode` (which routes to upstream Radiance v2.3.3's
  `RadianceVAE4KEncode/Decode` via that package's alias), and you hit
  the 4D vs 5D tensor mismatch on video workflows, switch to the
  namespaced `RadianceVAEEncode__koolook_v2_3_3` /
  `RadianceVAEDecode__koolook_v2_3_3` (display name suffix
  *(Koolook v2.3.3)*).
- The existing `__koolook_v1_0_1` namespaced nodes are unchanged and
  remain available for backward compatibility with saved workflows.

## [0.1.1] - 2026-05-03

### Fixed
- Re-tag release at current `main` HEAD so `git describe --tags` resolves locally
  (the original `v0.1.0` tag points at an orphaned merge commit that is no longer
  in `main`'s ancestry, which caused ComfyUI-Manager to display the installed
  version as "unknown").
- Correct `[tool.comfy] PublisherId` from the placeholder `CristianP` to the
  real publisher `kforgelabs`, which was the actual cause of every
  `Publish to Comfy registry` workflow failure (registry returned a misleading
  `400 "Failed to validate token"` because the declared publisher did not exist).

### Changed
- Bumped `pyproject.toml` `version` to `0.1.1` to match the new tag.

### Chore
- `.gitignore`: fix syntax and exclude `__pycache__/`, `*.pyo`, `.DS_Store`
  (carried forward from commit `8fc28d1`).

## [0.1.0] - 2026-04-24

### Added
- Fork tracking moved to `forks/` with centralized workflow documentation.
- Versioned Radiance fork package layout under `forks/radiance_koolook/versions/v1_0_1`.
- `CLAUDE.md` and `Glossary.md` to keep workflow and naming conventions explicit.

### Changed
- Root node loader now imports the versioned Radiance fork entrypoint.
- Release/checklist/template references updated to current fork-based paths.
- Introduced compact node ID override for OCIO transform: `k_easy_OCIO_v101`.

### Removed
- Legacy `ACES_workflow/radiance` tree and duplicated tracking paths under `third_party/`.
- Deprecated docs/assets/workflow artifacts removed during repository cleanup.
