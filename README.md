# ComfyUI-Koolook-Nodes

This repository provides custom nodes for ComfyUI tailored for VFX and AI image/video generation workflows.

For change history see [`CHANGELOG.md`](CHANGELOG.md). For workflow conventions and naming see [`CLAUDE.md`](CLAUDE.md) and the [glossary](docs/reference/glossary.md). Full project documentation lives under [`docs/`](docs/).

### Koolook nodes (root package)

| Node | Display name | Category | Source |
|------|--------------|----------|--------|
| `Easy_Version` | Easy Version | `VFX/Utils` | `k_easy_version.py` |
| `EasyWan22Prompt` | Wan 2.2 Easy Prompt | `Koolook/Wan_Video` | `k_easy_wan22_prompt.py` |
| `EasyResize_Koolook` | Easy Resize (Koolook) | `Koolook/Image` | `k_easy_resize.py` |
| `EasyAIPipeline` | Easy AI Pipeline | `Koolook/VFX` | `k_ai_pipeline.py` |
| `easy_ImageBatch` | Easy Image Batch | `Koolook/VFX` | `k_easy_image_batch.py` |
| `KoolookLoadCameraPosesAbsolute` | Koolook Load Camera Poses (Absolute Path) | `Koolook/Camera` | `k_easy_track.py` |

Brief descriptions:

- **Easy Version**: Generates a padded version string like `v001` from an integer. Ideal for naming renders, sequences, or iterations in VFX pipelines.
- **Wan 2.2 Easy Prompt**: Dynamic prompt builder driven by `config.json` (light source, lighting type, time of day, etc.). Outputs a comma-separated prompt string and optional CLIP conditioning.
- **Easy Resize (Koolook)** — `EasyResize_Koolook`: Aspect-aware image resize that enforces divisibility (e.g., `32`) for model compatibility. Modes: stretch, letterbox, pillarbox; padding and cropping supported. Originally inspired by the `Resize Image V2` node from [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) (GPL-3.0 — see [`forks/THIRD_PARTY.md`](forks/THIRD_PARTY.md) for the full attribution + change log). The legacy bare-name `EasyResize` ID is kept as a deprecated alias for saved-workflow compatibility.
- **Easy AI Pipeline**: Aggregates VFX shot parameters (shot duration, seed, base path, shot name, AI method, version) into a fully formatted output file path with overwrite protection. For organized writes from Save EXR / Save Image.
- **Easy Image Batch**: Builds an `IMAGE` + `MASK` batch of length `total_frames` from 1–4 keyframe images placed at explicit frame indices (with `start_frame` offset). Empty frames are filled with black/gray and the mask marks empty=white / occupied=black. Useful for sparse keyframe control with Wan-style video models.
- **Koolook Load Camera Poses (Absolute Path)**: Loads RealEstate10k-style `CAMERACTRL_POSES` from any absolute path (no ComfyUI path stripping), with overrideable pose width/height. Pairs with AnimateDiff CameraCtrl downstream.

### Radiance Koolook nodes (v2_3_3 fork — video-friendly VAE)

These come from [`forks/radiance_koolook/versions/v2_3_3/`](forks/radiance_koolook/versions/v2_3_3) and load through the root `__init__.py`. They are slim, rank-agnostic re-implementations of the upstream Radiance v2.3.3 VAE encoder/decoder that work correctly with video VAEs (Wan 2.2, Hunyuan, CogVideoX, LTX) — they skip the upstream's 4K cosine-blend tile engine, which conflicts with video VAEs that already handle their own internal stitching.

| Node ID | Display name | Category |
|---|---|---|
| `Easy_hdr_VAE_encode` | `Easy_hdr_VAE_encode` | `Koolook/VFX` |
| `Easy_hdr_VAE_decode` | `Easy_hdr_VAE_decode` | `Koolook/VFX` |

> **History note:** the v0.1.0–v0.1.4 releases also shipped a much larger `versions/v1_0_1/` fork that wrapped 26 Radiance v1.0 classes (HDR, EXR, OCIO, log curves, etc.) under the `__koolook_v1_0_1` namespace. Those wrappers were vestigial — never used by Koolook authors — and were removed in v0.1.5. Users who want that functionality should install upstream Radiance directly. See [`forks/THIRD_PARTY.md`](forks/THIRD_PARTY.md) for full attribution and removal rationale.

See [`forks/README.md`](forks/README.md) for the full fork workflow.

## Features
- **Dynamic Inputs for Prompts**: Fields and dropdowns in Wan 2.2 Easy Prompt are defined in `config.json`, allowing easy customization without code changes—great for VFX artists iterating on lighting or camera parameters.
- **Image Resizing with Aspect Control**: Easy Resize ensures precise resolutions for AI models, with options for keep_proportion preservation, padding colors, and device selection (CPU/CUDA). Supports masks for compositing in VFX.
- **Pipeline Path Generation**: Easy AI Pipeline assembles VFX-style output paths and filenames from shot/version inputs, with optional version disabling and overwrite protection.
- **Sparse Keyframe Batches**: Easy Image Batch constructs aligned `IMAGE` + `MASK` batches from up to four keyframe images, ready for sparse video-model control.
- **Camera Pose Loading**: Koolook Load Camera Poses (Absolute Path) reads RealEstate10k pose TXT files from any disk location and emits `CAMERACTRL_POSES`.
- **Startup Loading**: Nodes load configurations at ComfyUI launch; edit `config.json` and restart to update.
- **Output Formatting**: Wan 2.2 Easy Prompt outputs comma-separated strings (skipping "none"), compatible with prompt builders. Easy Resize outputs resized images, masks, and width/height integers. Easy Version provides padded strings for filename appending.
- **Error Handling**: Fallbacks for missing/invalid JSON in Wan 2.2 Easy Prompt; validation for aspect ratios and divisibility in Easy Resize.
- **Compatibility**: Works with ComfyUI-Manager for easy installation via Git URL. The Koolook root nodes and the slim v2_3_3 VAE wrappers have no extra Python dependencies beyond `torch` (already required by ComfyUI itself).

## RunPod

RunPod management has been moved to a separate private repository for active operations and iteration.

- Runtime/deployment scaffolding is no longer maintained in this public repo.
- This public repo remains focused on the ComfyUI nodes/plugin code.

## Release & Stability

For production usage, install a pinned stable tag/commit instead of following moving branch heads.

- **Release procedure**: [`docs/maintainers/releasing.md`](docs/maintainers/releasing.md) (canonical, step-by-step)
- **Comfy Registry API** (deprecate / yank versions): [`docs/maintainers/registry-api.md`](docs/maintainers/registry-api.md)
- **Node versioning rules** (how to change a node without breaking saved workflows): [`docs/maintainers/node-versioning.md`](docs/maintainers/node-versioning.md)
- Per-release checklist template: [`.github/ISSUE_TEMPLATE/release_checklist.md`](.github/ISSUE_TEMPLATE/release_checklist.md)
- Change history: [`CHANGELOG.md`](CHANGELOG.md)
- Fork tracking: [`forks/THIRD_PARTY.md`](forks/THIRD_PARTY.md), [`forks/forks_manifest.yaml`](forks/forks_manifest.yaml)
- Fork workflow guide: [`forks/README.md`](forks/README.md)

### Tagging Policy (from now on)

- Repo releases: `vMAJOR.MINOR.PATCH` (example: `v0.2.0`).
- Fork code versions in paths use underscore format: `v1_0_1`, `v2_3_3`.
- Comfy node ID namespace suffixes match fork version paths: `__koolook_v1_0_1`.
- External raw checkout naming:
  - pinned: `<repo>-v<version>-koolook`
  - rolling upstream: `<repo>-main-upstream`

## Sibling Projects

Some related work lives in **sibling folders/repositories** outside MAIN. They are *consulted* (read-only knowledge sources, references, etc.) but never imported by MAIN at runtime. To keep the public repo free of machine-specific absolute paths, every sibling is referenced through an environment variable.

### Conventions

- Variable prefix: `KOLOOK_*` (matches the existing `KOLOOK_FORKS_DIR`).
- Real paths live in a local, gitignored `.env` file. The committed `.env.example` declares the variable names with placeholder values.
- Public docs and committed code reference siblings only via the env var name (and an optional portable relative default like `../<folder>`). Never hardcode an absolute path with a username.
- A sibling project is a *runtime-optional* reference. If the variable is unset and no default is documented, agents/code must treat the sibling as unavailable and continue gracefully.

### Registered siblings

| Role | Env var | Portable default | Status |
|------|---------|------------------|--------|
| External forks root (raw upstream clones) | `KOLOOK_FORKS_DIR` | `../ComfyUI-Forks` | documented in `forks/forks_manifest.yaml` |
| ComfyUI knowledge database | `KOLOOK_COMFYUI_KB_DIR` | `../ComfyUI_knowledge___DB` | read-only knowledge source |
| Personal knowledge database | `KOLOOK_PERSONAL_KB_DIR` | *(no default — user-specific)* | optional |

### Local setup

```bash
cp .env.example .env
# Edit .env to point at the real paths on your machine.
```

Do **not** commit `.env`. The default `.gitignore` already excludes `.env`, `.env.local`, and `.env.*.local`.

## External Fork Workflow (No Vendoring in MAIN)

This repository is the MAIN control repo. Large third-party repositories must stay outside MAIN.

- Default external forks root (relative): `../ComfyUI-Forks`
- MAIN stores only:
  - wrappers under `forks/`
  - provenance/tracking under `forks/`
- MAIN does not store full external trees (no vendored copies).

### Radiance Koolook Version Layout

- Package entry path: `forks/radiance_koolook/__init__.py`
- Active version path: `forks/radiance_koolook/versions/v2_3_3/`
- Modified node source is local and tracked in MAIN:
  - `forks/radiance_koolook/versions/v2_3_3/nodes_vae.py`
- External checkout is reference-only raw upstream:
  - default location: `../ComfyUI-Forks/radiance-v2.3.3-koolook`
  - baseline used for comparison: upstream commit `f262f47ddfda01ece154bf80c22769b1e4cef795` (the v2.3.3 release commit; the cleanest tag pointing at it is the typo'd `radinace2.3.3`, so we pin by SHA)
- Node IDs in the v2_3_3 fork are exposed verbatim as `Easy_hdr_VAE_encode` / `Easy_hdr_VAE_decode` (the `__koolook_v2_3_3` namespace suffix is skipped via `SKIP_VERSION_SUFFIX` in `versions/v2_3_3/__init__.py` because the Koolook-specific names already differentiate them from upstream).
- Each version folder should include `UPSTREAM_PIN.yaml` to keep parity with external pinned references.
- The historical `v1_0_1` folder was removed in v0.1.5; see [`forks/THIRD_PARTY.md`](forks/THIRD_PARTY.md).

### Version Pinning (Portable Across PCs/macOS)

- Track every external repo in `forks/forks_manifest.yaml` with:
  - `source_ref` (tag/version)
  - `pinned_commit` (exact hash used in production)
  - `external_checkout.relative_path_from_forks_root`
- Keep checkout folder names versioned, e.g.:
  - `radiance-v1.0.1-koolook`
  - `radiance-v2.3.3-koolook`

### Machine Setup Example

From the parent directory that contains `ComfyUI-Koolook`:

```bash
mkdir -p ../ComfyUI-Forks
cd ../ComfyUI-Forks
git clone https://github.com/fxtdstudios/radiance.git radiance-v1.0.1-koolook
cd radiance-v1.0.1-koolook
git checkout f1b8ae330848fa08aba24c9d3e355cb432d3515b
```

This external folder is for upstream comparison/sync work only.

## Installation
### Via ComfyUI-Manager (Recommended)
1. Open ComfyUI and access the Manager interface.
2. Go to "Install Custom Nodes" > "Install via Git URL".
3. Paste the repository URL: `https://github.com/malkuthro/ComfyUI-Koolook.git`.
4. Install and restart ComfyUI.
5. Nodes appear under these categories in the node search:
   - `VFX/Utils` — Easy Version
   - `Koolook/Wan_Video` — Wan 2.2 Easy Prompt
   - `Koolook/Image` — Easy Resize (Koolook)
   - `Koolook/VFX` — Easy AI Pipeline, Easy Image Batch
   - `Koolook/Camera` — Koolook Load Camera Poses (Absolute Path)
   - Radiance categories (set by upstream Radiance) for the namespaced `(Koolook v1.0.1)` nodes

### Manual Installation
1. Clone this repository into your `ComfyUI/custom_nodes/` directory:
   ```
   git clone https://github.com/malkuthro/ComfyUI-Koolook.git
   ```
2. Ensure `__init__.py` and `config.json` are in the cloned folder.
3. Restart ComfyUI.

The root Koolook nodes have no additional Python dependencies. To enable the Radiance Koolook v1.0.1 nodes that handle EXR / OCIO / HDR work, install the optional packages: `OpenEXR`, `Imath`, `PyOpenColorIO`, `colour-science`, `opencv-python`, and `imageio`.

## Usage

### Easy Version
1. Add the "Easy Version" node to your workflow under "VFX/Utils".
2. Connect an INT input (e.g., from a Constant node) to the "version" slot.
3. Output is a STRING like "v001" for input 1, "v011" for 11—use for appending to filenames in Save Image nodes or VFX render queues.

### Wan 2.2 Easy Prompt
1. Add the "Wan 2.2 Easy Prompt" node to your workflow under "Koolook/Wan_Video".
2. Select options from the dropdowns (e.g., Light Source: "Sunny lighting", Lighting Type: "Soft lighting").
3. Add custom text in the "body" field for main prompt content.
4. Optionally connect a CLIP for conditioning output.
5. Connect the STRING outputs (combined_prompt or fields_only) to a prompt input, such as the "prompt" slot in the PromptJSON node (from [NeuralSamurAI/ComfyUI-PromptJSON](https://github.com/NeuralSamurAI/ComfyUI-PromptJSON)).
6. To customize fields:
   - Edit `config.json` (see example below).
   - Restart ComfyUI to apply changes.
7. VFX Tip: Use for generating consistent lighting/shot descriptions in video generation workflows, like with AnimateDiff or SVD.

#### Example config.json
This is the default configuration based on Wan 2.2 parameters:
```json
{
  "category": "Koolook/Wan_Video",
  "node_name": "Wan_2.2",
  "fields": [
    {
      "name": "light_source",
      "type": "combo",
      "options": ["none", "Sunny lighting", "Artificial lighting", "Moonlighting", "Practical lighting", "Firelighting", "Fluorescent lighting", "Overcast lighting", "Mixed lighting"],
      "default": "none"
    },
    {
      "name": "lighting_type",
      "type": "combo",
      "options": ["none", "Soft lighting", "Hard lighting", "Top lighting", "Side lighting", "Underlighting", "Edge lighting", "Silhouette lighting", "Low contrast lighting", "High contrast lighting"],
      "default": "none"
    },
    // ... (additional fields like time_of_day, shot_size, etc.)
  ],
  "output_type": "STRING"
}
```

### Easy Resize (Koolook)
1. Add the "Easy Resize (Koolook)" node (ID `EasyResize_Koolook`) to your workflow under "Koolook/Image".
2. Connect an IMAGE (and optional MASK) input.
3. Set the base dimension (Width or Height), base size, aspect ratio (e.g., "16:9" for landscape, "9:16" for portrait), and other options like keep_proportion (stretch, letterbox, pillarbox), pad_color (e.g., "0, 0, 0" for black), crop_position, divisible_by, and device.
4. Connect the outputs: IMAGE and MASK for further processing, INT width/height for resolution-aware nodes like samplers or upscalers.
5. VFX Tip: Use for preprocessing footage in multi-resolution pipelines, ensuring divisibility for efficient GPU processing in tools like IPAdapter or ControlNet.

## Node Code Overview
The core logic is split across files for modularity:

Root Koolook nodes:
- **k_easy_version.py** — `Easy_Version` class for padded version-string formatting.
- **k_easy_wan22_prompt.py** — loads `config.json` at import time, builds inputs dynamically (combos for dropdowns, strings for text), and formats selections into prompt strings with optional CLIP conditioning.
- **k_easy_resize.py** — parses aspect ratios, computes divisible dimensions, resizes with `common_upscale`, and handles padding/cropping. Inspired by Resize Image V2 from ComfyUI-KJNodes.
- **k_ai_pipeline.py** — `EasyAIPipeline` builds VFX output paths/filenames from shot/version inputs and creates the output directory.
- **k_easy_image_batch.py** — `easy_ImageBatch` builds aligned `IMAGE`+`MASK` keyframe batches with placeholder fill.
- **k_easy_track.py** — `KoolookLoadCameraPosesAbsolute` loads RealEstate10k pose TXT files from any path.

Forked/wrapped nodes:
- **forks/radiance_koolook/** — version-namespaced wrapper for the Radiance fork. Currently exposes the `v2_3_3` set; modified node sources live in `forks/radiance_koolook/versions/v2_3_3/nodes_vae.py`. The wrapper supports the `__koolook_vX_Y_Z` ID suffix pattern but lets specific Koolook-named IDs (like `Easy_hdr_VAE_encode/decode`) ship without the suffix via the `SKIP_VERSION_SUFFIX` set.
- **forks/forks_manifest.yaml**, **forks/THIRD_PARTY.md**, **forks/README.md** — fork tracking metadata.

Glue and frontend:
- **__init__.py** — imports all node packages, merges `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`, and exposes `WEB_DIRECTORY = "./web"`.
- **web/** — JavaScript extensions loaded by ComfyUI for the Koolook nodes (e.g., `ai_pipeline.js`).

For advanced customizations (e.g., different output formats or additional resize modes), modify the `execute` or `adjust_to_aspect` methods in the respective files.

## Troubleshooting
- **ComfyUI Won't Start**: Check console logs for errors (run `python main.py`). Common: Missing `config.json` or invalid JSON—use a validator like jsonlint.com.
- **No Fields Visible in Wan 2.2 Easy Prompt**: Indicates JSON load failure; the node falls back to empty inputs.
- **Errors in Easy Resize**: Ensure aspect ratio is "w:h" format, divisible_by >=1, and pad_color is "r, g, b" floats (0.0-1.0).
- **Updates**: Pull latest from GitHub and restart.

## Credits
- Based on ComfyUI custom node examples.
- Parameters for Wan 2.2 Easy Prompt derived from Wan 2.2 docs (https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y).
- Easy Resize inspired by ComfyUI-KJNodes (https://github.com/kijai/ComfyUI-KJNodes), specifically the Resize Image V2 node in image_nodes.py.
- Compatible with nodes like PromptJSON for LLM workflows.

## License

**GPL-3.0** (see [`LICENSE`](LICENSE)).

This package incorporates and adapts code from
[fxtdstudios/radiance](https://github.com/fxtdstudios/radiance) (also
GPL-3.0) under `forks/radiance_koolook/`. Per GPL-3.0 §5(c), the entire
work is licensed as GPL-3.0. Attribution and per-fork modification notes
are tracked in [`forks/THIRD_PARTY.md`](forks/THIRD_PARTY.md) and
[`forks/forks_manifest.yaml`](forks/forks_manifest.yaml).

> **Note for downstream users:** any work that incorporates, links to, or
> derives from `ComfyUI-Koolook` must itself be distributed under GPL-3.0
> (or a compatible license). If you need a non-copyleft license for any
> piece of this code, contact the maintainers — some Koolook-original
> nodes (those with no upstream Radiance lineage) may be available under
> a more permissive license on request.