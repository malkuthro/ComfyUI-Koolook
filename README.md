# ComfyUI-Koolook-Nodes

This repository provides custom nodes for ComfyUI tailored for VFX and AI image/video generation workflows. It includes:

- **Easy Version**: A simple node that generates a version string like 'v001' based on an integer input, padded to three digits. Ideal for VFX versioning in ComfyUI workflows, such as naming renders, sequences, or iterations in pipelines involving tools like Save Image or batch processing.
- **Wan 2.2 Easy Prompt**: A dynamic prompt builder node that loads input fields and dropdown options from a `config.json` file. Inspired by Wan 2.2 parameters (e.g., light sources, lighting types, time of day), it generates comma-separated prompt strings for use in VFX workflows like structured prompts with nodes such as PromptJSON. Supports optional CLIP conditioning for direct integration into positive prompt slots.
- **Easy Resize**: An image resizing node that adjusts images based on a base dimension (width or height), aspect ratio (e.g., "16:9"), and ensures dimensions are divisible by a specified value (e.g., 32 for model compatibility). Supports stretch, letterbox, or pillarbox modes with padding and cropping options. Perfect for preparing images in VFX pipelines, ensuring compatibility with AI models like Stable Diffusion or video generation tools.

The code for the Easy Resize node is inspired by the Resize Image V2 node from ComfyUI-KJNodes (reference file: image_nodes.py, available at https://github.com/kijai/ComfyUI-KJNodes).

## Recent Changes
- **Node Splitting and Modularity**: All nodes have been split into separate Python files (`k_easy_version.py`, `k_easy_wan22_prompt.py`, `k_easy_resize.py`) for easier management, debugging, and extension in VFX workflows. The `__init__.py` now aggregates imports and mappings dynamically.
- **Naming Consistency**: Nodes renamed for clarity and prefixed internally (e.g., class names like `Easy_Version`, `EasyWan22Prompt`, `EasyResize`). Display names updated to "Easy Version", "Wan 2.2 Easy Prompt", and "Easy Resize".
- **Categories Updated**: Nodes appear under "VFX/Utils" (Easy Version), "Koolook/Wan_Video" (Wan 2.2 Easy Prompt), and "Koolook/Image" (Easy Resize) for better organization in ComfyUI's node search.
- **Improved Documentation**: Usage examples now include VFX-specific tips, such as integrating with versioning for render farms or resizing for multi-resolution pipelines.

## Features
- **Dynamic Inputs for Prompts**: Fields and dropdowns in Wan 2.2 Easy Prompt are defined in `config.json`, allowing easy customization without code changes—great for VFX artists iterating on lighting or camera parameters.
- **Image Resizing with Aspect Control**: Easy Resize ensures precise resolutions for AI models, with options for keep_proportion preservation, padding colors, and device selection (CPU/CUDA). Supports masks for compositing in VFX.
- **Startup Loading**: Nodes load configurations at ComfyUI launch; edit `config.json` and restart to update.
- **Output Formatting**: Wan 2.2 Easy Prompt outputs comma-separated strings (skipping "none"), compatible with prompt builders. Easy Resize outputs resized images, masks, and width/height integers. Easy Version provides padded strings for filename appending.
- **Error Handling**: Fallbacks for missing/invalid JSON in Wan 2.2 Easy Prompt; validation for aspect ratios and divisibility in Easy Resize.
- **Compatibility**: Works with ComfyUI-Manager for easy installation via Git URL. No additional dependencies beyond standard Python and ComfyUI libraries.

## RunPod

RunPod management has been moved to a separate private repository for active operations and iteration.

- Runtime/deployment scaffolding is no longer maintained in this public repo.
- This public repo remains focused on the ComfyUI nodes/plugin code.

## Release & Stability

For production usage, install a pinned stable tag/commit instead of following moving branch heads.

- Release workflow: `docs/RELEASE_WORKFLOW.md`
- Change history: `CHANGELOG.md`
- Third-party tracking: `third_party/THIRD_PARTY.md`, `third_party/forks_manifest.yaml`

## Installation
### Via ComfyUI-Manager (Recommended)
1. Open ComfyUI and access the Manager interface.
2. Go to "Install Custom Nodes" > "Install via Git URL".
3. Paste the repository URL: `https://github.com/malkuthro/ComfyUI-Koolook.git`.
4. Install and restart ComfyUI.
5. The nodes will appear:
   - "Easy Version" under "VFX/Utils".
   - "Wan 2.2 Easy Prompt" under "Koolook/Wan_Video".
   - "Easy Resize" under "Koolook/Image".

### Manual Installation
1. Clone this repository into your `ComfyUI/custom_nodes/` directory:
   ```
   git clone https://github.com/malkuthro/ComfyUI-Koolook.git
   ```
2. Ensure `__init__.py` and `config.json` are in the cloned folder.
3. Restart ComfyUI.

No additional dependencies required beyond standard Python libraries.

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

### Easy Resize
1. Add the "Easy Resize" node to your workflow under "Koolook/Image".
2. Connect an IMAGE (and optional MASK) input.
3. Set the base dimension (Width or Height), base size, aspect ratio (e.g., "16:9" for landscape, "9:16" for portrait), and other options like keep_proportion (stretch, letterbox, pillarbox), pad_color (e.g., "0, 0, 0" for black), crop_position, divisible_by, and device.
4. Connect the outputs: IMAGE and MASK for further processing, INT width/height for resolution-aware nodes like samplers or upscalers.
5. VFX Tip: Use for preprocessing footage in multi-resolution pipelines, ensuring divisibility for efficient GPU processing in tools like IPAdapter or ControlNet.

## Node Code Overview
The core logic is split across files for modularity:
- **k_easy_version.py**: Handles `Easy_Version` class for string formatting.
- **k_easy_wan22_prompt.py**: Loads `config.json` at import time, builds inputs dynamically (combos for dropdowns, strings for text), and formats selections into prompt strings with optional CLIP conditioning.
- **k_easy_resize.py**: Parses aspect ratios, computes divisible dimensions, resizes with common_upscale, and handles padding/cropping. Inspired by Resize Image V2 from ComfyUI-KJNodes.
- **__init__.py**: Imports all nodes and merges mappings for ComfyUI integration.

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
MIT License. Feel free to use, modify, and distribute.