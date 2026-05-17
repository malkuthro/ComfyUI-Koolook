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
