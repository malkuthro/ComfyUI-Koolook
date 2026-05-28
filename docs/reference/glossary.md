# Glossary

Short, plain-English definitions for this repo workflow.

## Root `__init__.py`
- The ComfyUI entry file for this plugin.
- Its job is to import node mappings and expose `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`.
- Keep it clean and focused on loading nodes.

## Main Repo
- This repository: `ComfyUI-Koolook`.
- Stores Koolook-owned code, docs, and lightweight integration wrappers.
- Should not store full external third-party repositories.

## External Forks
- Raw upstream/fork reference repos kept outside MAIN.
- Default location is a sibling folder: `../ComfyUI-Forks`.
- Each external repo should be pinned to a version/tag + exact commit.

## Wrapper
- A small adapter module in `forks/` that loads locally tracked modified nodes and namespaces node IDs.
- Purpose: avoid collisions and keep external logic separate from root loader.
- For Radiance, the package entrypoint is `forks/radiance_koolook/__init__.py`.

## Version Folder
- A folder under `forks/radiance_koolook/versions/` that stores one maintained codebase snapshot (example: `v1_0_1`).
- Keep upstream parity metadata in `UPSTREAM_PIN.yaml` inside each version folder.

## Namespace Suffix
- A suffix added to internal node IDs (example: `__koolook_v1_0_1`).
- Prevents collisions between versions (v1, v2, upstream, etc.).
- Protects old workflows from breaking.

## `forks/forks_manifest.yaml`
- Machine-readable tracker for external dependencies.
- Contains source repo, source ref, pinned commit, and external relative path.
- This is the source of truth for reproducible setup on new machines.

## `forks/THIRD_PARTY.md`
- Human-readable attribution and notes.
- Explains what was modified and why.

## Portable Paths
- Prefer relative paths so setup works on Windows/macOS/Linux.
- Keep external references under `../ComfyUI-Forks` for portable machine setup.

## Pinned Commit
- The exact git commit hash you lock to for reproducibility.
- Use this when moving to another computer so behavior matches exactly.

## Sibling Project
- A related repository or folder that lives **outside MAIN** and is consulted (read-only) but never imported by MAIN at runtime.
- Examples: external forks root, ComfyUI knowledge database, personal knowledge database.
- Always referenced by env var (e.g. `KOLOOK_COMFYUI_KB_DIR`) — never by absolute path in committed files.
- Real paths live in `.env` (gitignored). The committed `.env.example` declares the variable names.
- Treated as runtime-optional: if the env var is unset and there is no portable default, the sibling is considered unavailable.

## `.env` / `.env.example`
- `.env.example` (committed) — public template listing the env var names with safe placeholder values.
- `.env` (gitignored) — each user's local file with real machine-specific paths.
- Pattern matches the existing `KOLOOK_FORKS_DIR` convention.

## EasyUse GET/SET Virtual Tunnel
- A pair of frontend-only EasyUse utility nodes: `easy setNode` stores a
  named value source and `easy getNode` exposes that value elsewhere in the
  graph by key.
- Render-time ComfyUI execution resolves the tunnel, but a browser-side
  preview button that simply reads the GET node widget will see only the key
  name (for example `OUT-folder`) rather than the actual upstream value.
- Preview code must follow `easy getNode` -> matching `easy setNode` -> set
  input link -> source widget value to match render-time behavior.
- Debug by reading the installed EasyUse frontend source
  (`comfyui-easy-use/web_version/v1/js/getset.js`) and simulating the relevant
  LiteGraph shape locally when the live `app.graph` object is not exposed.

## Automation module
- A self-contained AI-managed ComfyUI iteration task. Lives under
  `docs/automations/<grouping>/<task>/`, with the grouping typically being
  the diffusion model (`LTX-2.3/`) and the task being a specific generation
  goal (`base-1step/`, `audio-lipsync/`).
- Each module owns: a `README.md` (loop entry + workflow contract), a
  `handoff-checklist.md` (5-minute bootstrap), a `findings.md` (locked-in
  conclusions for that task), optionally a `backstory/` folder (the
  narrative that produced the module), optionally a `runs/` folder for
  in-repo run snapshots when the iteration touches code (not just widget
  values).
- Modules are independent: own workflow JSON, own working folder, own
  findings. The model grouping is only structural — it does not carry
  shared docs unless duplication forces a future `_shared/` extraction.
- When a module needs to modify an upstream node's code, the modification
  lives under `forks/<package>_koolook/` per the fork pattern (see
  *Wrapper* and *Version Folder* above), and the module's README
  cross-references the fork.

## Loop
- The save → render → feedback → snapshot cycle inside an automation
  module.
- Step 1: maintainer edits a knob in ComfyUI (or in a `forks/` source
  file) and **saves** the workflow.
- Step 2: maintainer queues the render.
- Step 3: maintainer reports the result in chat.
- Step 4: agent snapshots current state into the module's
  `runs/run-NNN_<label>/` (when the module ships in-repo runs) or just
  appends to the working-folder `_AI/iterations.md` (when card-and-log
  is all that's needed).
- Module-specific protocol lives at `<module>/runs/LOOP.md` for the
  fork-touching modules; lighter modules document the loop inline in
  their `README.md`.

## Run
- One iteration of the loop.
- For modules that ship in-repo run snapshots, a `run-NNN_<short-knob-summary>_<short-result-tag>`
  folder under `<module>/runs/` containing at minimum a `workflow.json`
  copy and a `notes.md` (maintainer feedback + agent interpretation).
- For all modules, also a row in the per-project working-folder
  `_AI/iterations.md` log (auto-rendered by `/make-card`).

## Backstory
- The running narrative that produced an automation module — the
  problem, the mechanism, the hypotheses considered, the rationale for
  why the module exists. Lives at `<module>/backstory/<topic>.md`.
- Distinct from `findings.md` (locked-in conclusions) and
  `runs/log.md` (rolling table of renders). The backstory is a
  reference for *why* and *how it got here*; findings are *what's
  true now*.

## Working folder
- The outside-the-repo per-project folder pointed at by
  `KOLOOK_AUTOMATIONS_WORK_DIR` in `.env`. Holds workflow JSON,
  rendered video, and the agent-managed `_AI/` subfolder with
  `card.png` + append-only `iterations.md`.
- One working folder per project. Multiple automation modules can
  point at the same working folder if they share workflow files;
  more commonly, each module's iteration uses its own working folder.

## Card
- A per-render tracking visual (PNG) rendered automatically at the end
  of a loop iteration. Travels alongside the rendered video into the
  NLE for side-by-side comparison.
- Each automation module ships its own card renderer in
  [scripts/](../../scripts/) to highlight what matters for *that*
  module's iteration:
  - `make_card.py` → base-1step card (Phase 1 / Phase 2 / Base · model /
    Base · locked / Base · scene / Outcome). Writes to
    `<working folder>/_AI/card.png`.
  - `make_card_audio.py` → audio-lipsync card (KNOB STATE / FORK STATE /
    SAMPLER / BASE notes / OUTCOME, plus an INERT warning when the
    Director is upstream). Writes inside the module's
    `runs/run-NNN_<label>/card.png`.
- Palette + font fallback chain is shared (see
  `scripts/make_card.py` and `scripts/make_card_audio.py`) so the two
  card families read as a set.

## Loop config (per-module)
- A JSON file co-located with the module's loop script that holds the
  customisable settings — workflow filename pattern, ComfyUI workflows
  subpath, tracked multiline titles, fork dir to pin, whether to
  render a card. Underscore-prefixed keys are ignored by the loader
  (carry inline documentation).
- Convention: `<script-name>.config.json` next to the script. The
  audio-lipsync loop config lives at
  [scripts/loop_audio.config.json](../../scripts/loop_audio.config.json).
- Required keys (validated at script start): `job_name`, `module_path`,
  `comfyui_workflows_subpath`, `workflow_pattern`,
  `skip_filename_substring`, `tracked_multilines`, `fork_to_track`,
  `render_card`.
- Per-module operating instructions live in the module's
  `CHEATSHEET.md` (e.g.
  [docs/automations/LTX-2.3/audio-lipsync/CHEATSHEET.md](../automations/LTX-2.3/audio-lipsync/CHEATSHEET.md)).
