# ComfyUI-Wan-Parameters

This is a custom node for ComfyUI that allows users to dynamically define input fields and dropdown options via a JSON configuration file. It's designed for workflows involving AI image/video generation parameters, specifically inspired by the Wan 2.2 parameters from the provided documentation (e.g., light sources, lighting types, time of day, etc.). The node loads the config at startup and generates a STRING output that can be connected to other nodes like PromptJSON for structured prompts.

## Features
- **Dynamic Inputs**: Fields and dropdowns are defined in `config.json`, making it easy to customize without editing code.
- **Startup Loading**: The node is ready upon ComfyUI launch; edit the JSON and restart to update.
- **Output Formatting**: Concatenates selected values into a comma-separated string (e.g., "light_source: Sunny lighting, lighting_type: Soft lighting"), skipping "none". This can be piped into prompt builders.
- **Error Handling**: Includes fallback for missing/invalid JSON to prevent ComfyUI crashes.
- **Compatibility**: Works with ComfyUI-Manager for easy installation via Git URL.

## Installation
### Via ComfyUI-Manager (Recommended)
1. Open ComfyUI and access the Manager interface.
2. Go to "Install Custom Nodes" > "Install via Git URL".
3. Paste the repository URL: `https://github.com/yourusername/ComfyUI-Wan-Parameters.git` (replace with your actual repo URL).
4. Install and restart ComfyUI.
5. The node "Wan_2.2" will appear under the "Wan/Parameters" category.

### Manual Installation
1. Clone this repository into your `ComfyUI/custom_nodes/` directory:
   ```
   git clone https://github.com/yourusername/ComfyUI-Wan-Parameters.git
   ```
2. Ensure `json_config_node.py` and `config.json` are in the cloned folder.
3. Restart ComfyUI.

No additional dependencies required beyond standard Python libraries.

## Usage
1. Add the "Wan_2.2" node to your workflow.
2. Select options from the dropdowns (e.g., Light Source: "Sunny lighting", Lighting Type: "Soft lighting").
3. Connect the STRING output to a prompt input, such as the "prompt" slot in the PromptJSON node (from [NeuralSamurAI/ComfyUI-PromptJSON](https://github.com/NeuralSamurAI/ComfyUI-PromptJSON)).
4. To customize fields:
   - Edit `config.json` (see example below).
   - Restart ComfyUI to apply changes.

### Example config.json
This is the default configuration based on Wan 2.2 parameters:
```json
{
  "category": "Wan/Parameters",
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

## Node Code Overview
The core logic is in `json_config_node.py`:
- Loads `config.json` at import time.
- Builds input types dynamically (combos for dropdowns, strings for text).
- Executes by formatting selections into a prompt string.

For advanced customizations (e.g., different output formats like JSON), modify the `execute` method.

## Troubleshooting
- **ComfyUI Won't Start**: Check console logs for errors (run `python main.py`). Common: Missing `config.json` or invalid JSON—use a validator like jsonlint.com.
- **No Fields Visible**: Indicates JSON load failure; the node falls back to empty inputs.
- **Updates**: Pull latest from GitHub and restart.

## Credits
- Based on ComfyUI custom node examples.
- Parameters derived from Wan 2.2 docs (https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y).
- Compatible with nodes like PromptJSON for LLM workflows.

## License
MIT License. Feel free to use, modify, and distribute.