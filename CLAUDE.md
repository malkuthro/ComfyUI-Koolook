# ComfyUI-Koolook Agent Workflow

## Source-of-truth model

- MAIN repo: `ComfyUI-Koolook`
- External forks root (default): `../ComfyUI-Forks` (relative to MAIN)
- External third-party code must remain outside MAIN.

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

## Radiance v1 policy

- Package entrypoint: `forks/radiance_koolook/__init__.py`
- Modified nodes live in MAIN and are GitHub-tracked:
  - `forks/radiance_koolook/versions/v1_0_1/nodes_hdr.py`
  - `forks/radiance_koolook/versions/v1_0_1/nodes_color_management.py`
  - `forks/radiance_koolook/versions/v1_0_1/nodes_dna.py`
- External sibling checkout (`../ComfyUI-Forks/radiance-v1.0.1-koolook`) is raw reference only.
- Current baseline pin for raw comparison: `f1b8ae330848fa08aba24c9d3e355cb432d3515b` (`comfyui` tag).
- Exposed node IDs must stay namespaced with version suffixes (`__koolook_v1_0_1`) to avoid collisions.
- `forks/forks_manifest.yaml` must include `source_ref` + `pinned_commit`.

## Change management

- When upgrading external forks (v2, v3, etc.), add a new wrapper/version namespace.
- Never rename existing namespaced node IDs that already appear in saved workflows.
