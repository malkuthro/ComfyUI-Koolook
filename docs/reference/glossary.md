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

## Investigation folder
- A subfolder under `docs/investigations/` that holds everything related to
  one investigation — the narrative README, an `upstream.yaml` pin file,
  modified upstream code under `patches/`, and per-render snapshots under
  `runs/`.
- Numbered prefix for ordering and short topic identifier:
  `docs/investigations/<NN>_<topic-slug>/`.
- Examples: `00_LTX23-base-1-step/`, `01_LTX23-audio-file-lipsync/`.
- Created when a piece of investigative work needs its own code patches and
  iteration log; promoted to a proper `forks/<name>_koolook/` fork when the
  patches validate.

## JSON file folder
- ComfyUI's workflow library directory where the maintainer overwrites a
  single workflow file between iterations.
- Current example (investigation 01): `E:\_AI\portable\ComfyUI_windows_312\ComfyUI\user\default\workflows\LTX-23-audio_tests_v01.json`.
- Read-only from the agent's perspective — never edited by the agent
  (preserves the ComfyUI canvas layout). Agent copies it into the
  investigation's `runs/run-NNN_*/` folder on each render.

## Loop
- The save → render → feedback → snapshot cycle for an investigation.
- Step 1: maintainer edits and **saves** workflow in ComfyUI (overwrites the
  JSON file folder target).
- Step 2: maintainer queues render.
- Step 3: maintainer reports result in chat.
- Step 4: agent snapshots current state (workflow + relay_overrides + active
  patches) into a new run folder under
  `docs/investigations/<NN>_<topic>/runs/`.
- Per-investigation protocol lives at
  `docs/investigations/<NN>_<topic>/runs/LOOP.md`.

## Run
- One iteration of the loop.
- Folder named `run-NNN_<short-knob-summary>_<short-result-tag>` inside
  `docs/investigations/<NN>_<topic>/runs/`.
- Contains: `workflow.json` (snapshot), `relay_overrides.txt`,
  `patch_state.txt` (MAIN SHA + upstream SHA + synced files), and `notes.md`
  (maintainer feedback + agent interpretation).

## Investigation patches
- Modifications to a third-party node's source code, applied during the
  investigation phase before a fork is formally registered under `forks/`.
- Source of truth lives in
  `docs/investigations/<NN>_<topic>/patches/<file>` — versioned by git.
- Pushed to the live install via
  `python scripts/sync_investigation_patches.py <NN>_<topic>`, which reads
  `upstream.yaml` from the investigation folder for the target path and
  file list.
- Each investigation declares its own short chat-trigger phrase via
  `trigger:` in `upstream.yaml` (e.g. `sync-audio` for investigation 01).
- Promoted to a proper `forks/<name>_koolook/` fork once the approach
  validates (≥ 3 confirming runs per the LTX-2.3 findings rule).

## `sync_investigation_patches.py`
- Helper script that copies an investigation's `patches/*` files to the
  live install path declared in its `upstream.yaml`.
- Mirrors the pattern of `scripts/sync_to_dev.py` but for third-party node
  modifications outside `KOLOOK_COMFYUI_DEV_PATH`.
- Creates a per-day backup of the target's current state
  (`<filename>.bak.<YYYYMMDD>`) before overwriting.
- User-initiated only (never automatic) — same rule as `dev-sync`.
