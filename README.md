# ComfyUI-Koolook-Nodes

This repository provides custom nodes for ComfyUI tailored for VFX and AI image/video generation workflows. It includes:

- **Wan22EasyPrompt**: A dynamic prompt builder node that loads input fields and dropdown options from a `config.json` file. Inspired by Wan 2.2 parameters (e.g., light sources, lighting types, time of day), it generates comma-separated prompt strings for use in workflows like structured prompts with nodes such as PromptJSON.
- **ImageAspect**: An image resizing node that adjusts images based on a base dimension (width or height), aspect ratio (e.g., "16:9"), and ensures dimensions are divisible by a specified value (e.g., 32 for model compatibility). Supports stretch, letterbox, or pillarbox modes with padding and cropping options. Ideal for preparing images in VFX pipelines.

The code for the ImageAspect node is inspired by the Resize Image V2 node from ComfyUI-KJNodes (reference file: image_nodes.py, available at https://github.com/kijai/ComfyUI-KJNodes).

## Features
- **Dynamic Inputs for Prompts**: Fields and dropdowns in Wan22EasyPrompt are defined in `config.json`, allowing easy customization without code changes.
- **Image Resizing with Aspect Control**: ImageAspect ensures precise resolutions for AI models, with options for keep_proportion preservation, padding colors, and device selection (CPU/CUDA).
- **Startup Loading**: Nodes load configurations at ComfyUI launch; edit `config.json` and restart to update.
- **Output Formatting**: Wan22EasyPrompt outputs comma-separated strings (skipping "none"), compatible with prompt builders. ImageAspect outputs resized images, masks, and width/height integers.
- **Error Handling**: Fallbacks for missing/invalid JSON in Wan22EasyPrompt; validation for aspect ratios and divisibility in ImageAspect.
- **Compatibility**: Works with ComfyUI-Manager for easy installation via Git URL. No additional dependencies beyond standard Python and ComfyUI libraries.

## Installation
### Via ComfyUI-Manager (Recommended)
1. Open ComfyUI and access the Manager interface.
2. Go to "Install Custom Nodes" > "Install via Git URL".
3. Paste the repository URL: `https://github.com/malkuthro/ComfyUI-Koolook.git`.
4. Install and restart ComfyUI.
5. The nodes will appear:
   - "Wan 2.2 Easy Prompt" under "Koolook/Wan_Video".
   - "Image Aspect" under "Koolook/Image".

### Manual Installation
1. Clone this repository into your `ComfyUI/custom_nodes/` directory:
   ```
   git clone https://github.com/malkuthro/ComfyUI-Koolook.git
   ```
2. Ensure `__init__.py` and `config.json` are in the cloned folder.
3. Restart ComfyUI.

No additional dependencies required beyond standard Python libraries.

## Usage

### Wan22EasyPrompt
1. Add the "Wan 2.2 Easy Prompt" node to your workflow.
2. Select options from the dropdowns (e.g., Light Source: "Sunny lighting", Lighting Type: "Soft lighting").
3. Optionally connect a CLIP for conditioning output.
4. Connect the STRING outputs (combined_prompt or fields_only) to a prompt input, such as the "prompt" slot in the PromptJSON node (from [NeuralSamurAI/ComfyUI-PromptJSON](https://github.com/NeuralSamurAI/ComfyUI-PromptJSON)).
5. To customize fields:
   - Edit `config.json` (see example below).
   - Restart ComfyUI to apply changes.

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

### ImageAspect
1. Add the "Image Aspect" node to your workflow.
2. Connect an IMAGE (and optional MASK) input.
3. Set the base dimension (Width or Height), base size, aspect ratio (e.g., "16:9" for landscape, "9:16" for portrait), and other options like keep_proportion (stretch, letterbox, pillarbox), pad_color (e.g., "0, 0, 0" for black), crop_position, divisible_by, and device.
4. Connect the outputs: IMAGE and MASK for further processing, INT width/height for resolution-aware nodes.
5. Useful for VFX: Ensures model-compatible sizes (e.g., divisible by 32) while maintaining aspect ratios.

## Node Code Overview
The core logic is in `__init__.py`:
- **Wan22EasyPrompt**: Loads `config.json` at import time, builds inputs dynamically (combos for dropdowns, strings for text), and formats selections into prompt strings with optional CLIP conditioning.
- **ImageAspect**: Parses aspect ratios, computes divisible dimensions, resizes with common_upscale, and handles padding/cropping. Inspired by Resize Image V2 from ComfyUI-KJNodes.

For advanced customizations (e.g., different output formats or additional resize modes), modify the `execute` or `adjust_to_aspect` methods.

## Troubleshooting
- **ComfyUI Won't Start**: Check console logs for errors (run `python main.py`). Common: Missing `config.json` or invalid JSON—use a validator like jsonlint.com.
- **No Fields Visible in Wan22EasyPrompt**: Indicates JSON load failure; the node falls back to empty inputs.
- **Errors in ImageAspect**: Ensure aspect ratio is "w:h" format, divisible_by >=1, and pad_color is "r, g, b" floats (0.0-1.0).
- **Updates**: Pull latest from GitHub and restart.

## Credits
- Based on ComfyUI custom node examples.
- Parameters for Wan22EasyPrompt derived from Wan 2.2 docs (https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y).
- ImageAspect inspired by ComfyUI-KJNodes (https://github.com/kijai/ComfyUI-KJNodes), specifically the Resize Image V2 node in image_nodes.py.
- Compatible with nodes like PromptJSON for LLM workflows.

## License
MIT License. Feel free to use, modify, and distribute.