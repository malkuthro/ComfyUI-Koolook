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
