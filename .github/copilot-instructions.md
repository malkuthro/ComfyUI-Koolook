## Purpose
Give an AI coding agent the minimal, high-value context to be productive in this repo (ComfyUI-Koolook).

## Quick orientation
- Repo provides custom ComfyUI nodes focused on VFX / video/image prompt generation and resizing.
- Key Python nodes live at the repo root: `k_easy_version.py`, `k_easy_wan22_prompt.py`, `k_easy_resize.py`, `k_ai_pipeline.py`.
- `config.json` drives dynamic UI/inputs for the Wan 2.2 prompt node.
- `web/ai_pipeline.js` contains ComfyUI front-end tweaks (preview buttons, read-only instruction widget).

## High-level architecture and runtime behavior
- Each node is implemented as a Python class exposing the ComfyUI node contract: `INPUT_TYPES`, `RETURN_TYPES`, `RETURN_NAMES`, `FUNCTION`, and `CATEGORY`.
- `__init__.py` imports these classes and exposes `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` so ComfyUI can register nodes.
- Nodes are imported at ComfyUI startup. `k_easy_wan22_prompt.py` reads `config.json` at import time to build inputs dynamically—editing `config.json` requires a ComfyUI restart to take effect.
- Front-end behaviour is extended via `web/ai_pipeline.js` which hooks `beforeRegisterNodeDef` and `nodeCreated` to add widgets and preview buttons for `EasyAIPipeline` and to dynamically update resize widgets for `EasyResize`.

## Important patterns & conventions (concrete)
- Dynamic inputs: `config.json` defines fields as `{ name, type, options, default }`. The node code iterates this to create combo/string widgets. See `config.json` and `k_easy_wan22_prompt.py`.
- Node signature pattern:
  - `INPUT_TYPES()` returns a dict with `required` inputs where each entry is a tuple like `("INT", {"default": 1, ...})`.
  - `RETURN_TYPES` and `RETURN_NAMES` describe outputs. `FUNCTION` is the callable to run.
  - Example: `k_ai_pipeline.py` exposes `generate_pipeline` via `FUNCTION = "generate_pipeline"` and returns `(file_path, name, version_str, output_directory, ...)`.
- File-path handling: nodes use `os.path.join(...)` then normalize separators with `.replace('\\','/')` and remove duplicate slashes. Versioning uses `f"v{version:03d}"` when not disabled.
- Overwrite protection: `k_ai_pipeline.generate_pipeline` raises a `ValueError` when the output directory exists and `enable_overwrite` is False. Agents should follow this behavior when changing save semantics.

## UI integration specifics
- `web/ai_pipeline.js` provides two preview buttons for `EasyAIPipeline`:
  - "Get output directory path" — builds the path from widget/widget-values and upstream simple widget values. If an input is connected to a complex node, it shows "Cannot preview..." because it can't resolve runtime values.
  - "Get output file path" — same but includes constructed filename using the `extension` widget and padded version string.
- The JS helper `getEffectiveValue(node, name)` tries to resolve widget value or simple upstream widget values. When adding or editing similar front-end helpers, mirror this approach.

## Developer workflows (how to run / debug)
- Install via ComfyUI Manager (recommended) or clone into `ComfyUI/custom_nodes/` and restart ComfyUI (see `README.md`).
- To see runtime exceptions / startup logs, run ComfyUI from a terminal and watch stdout/stderr (on Windows PowerShell):
  - `python main.py` (run from the ComfyUI root)
- To apply `config.json` changes: edit `config.json` and restart ComfyUI.

## Integration points & external references
- Prompt builders: designed to plug into nodes such as `PromptJSON` (see README). The Wan 2.2 prompt outputs comma-separated strings.
- Inspired by / interoperable with ComfyUI-KJNodes (Resize Image V2) — see README links.

## What to keep in mind when editing code
- Preserve the node contract fields (`INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, `CATEGORY`) to avoid breaking ComfyUI registration.
- When changing `config.json` fields, ensure defaults are valid and keep `output_type` consistent (current value: `"STRING"`).
- If adding front-end widgets, make changes in `web/ai_pipeline.js` to keep the UI in sync with node behavior (preview helpers, dynamic widget insertion/removal patterns).

## Useful examples to cite in edits
- Construct version string: `version_str = f"v{version:03d}"` in `k_ai_pipeline.py`.
- Normalize path: `os.path.join(*parts).replace('\\', '/')` followed by removing duplicate slashes.
- UI preview button logic: `web/ai_pipeline.js` uses `parts.filter(...).join('/')` and `.replace(/\\/g, "/").replace(/\/+/g, '/')` for browser-side cleaning.

## Where to look next (key files)
- `README.md` — repo purpose and installation instructions.
- `config.json` — dynamic prompt definitions.
- `k_ai_pipeline.py`, `k_easy_wan22_prompt.py`, `k_easy_resize.py`, `k_easy_version.py` — node implementations.
- `web/ai_pipeline.js` — front-end customizations.

## Quick ask for the maintainer
- Tell me if there are any hidden developer scripts or a preferred local dev command for running ComfyUI in dev mode (otherwise I will use `python main.py`).

---
If you'd like, I can iterate and trim or expand any section, or add short examples of small edits (e.g., add a new field to `config.json` and show the minimal Python changes needed).
