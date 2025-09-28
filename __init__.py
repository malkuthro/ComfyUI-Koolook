import json
import os

class Wan22EasyPrompt:
    """
    Wan 2.2 Easy Prompt node for ComfyUI.
    Loads dynamic inputs from config.json and outputs a formatted string for prompts.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # Get the directory of this script to locate config.json
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config.json')

        # Load the JSON config with error handling
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading config.json: {e}. Using dummy config.")
            config = {
                "category": "Wan/Parameters",
                "node_name": "Wan_2.2",
                "fields": [],
                "output_type": "STRING"
            }

        # Build the INPUT_TYPES dict dynamically
        input_types = {"required": {}}
        for field in config['fields']:
            name = field['name']
            if field['type'] == 'combo':
                options = field['options']
                defaults = {
                    "default": field.get('default', options[0]),
                    "lazy": True
                }
                input_types["required"][name] = (options, defaults)
            elif field['type'] == 'string':
                defaults = {
                    "default": field.get('default', ""),
                    "multiline": True,
                    "lazy": True
                }
                input_types["required"][name] = ("STRING", defaults)

        return input_types

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "execute"
    CATEGORY = "Wan/Parameters"
    OUTPUT_NODE = False

    def check_lazy_status(self, **kwargs):
        # All inputs are lazy; evaluate all if any are needed. For simplicity, require all.
        return list(kwargs.keys())  # Return all input names to evaluate everything

    def execute(self, **kwargs):
        # Concatenate selections into a prompt string, skipping "none"
        prompt_parts = []
        for key, value in kwargs.items():
            if value and value != "none":
                prompt_parts.append(f"{key}: {value}")
        return (", ".join(prompt_parts),)

    # Optional: IS_CHANGED for re-execution if config changes
    @classmethod
    def IS_CHANGED(cls):
        # Hash the config.json to re-execute if it changes
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config.json')
        try:
            with open(config_path, 'rb') as f:
                return hash(f.read())
        except FileNotFoundError:
            return "dummy"

# Export the node
NODE_CLASS_MAPPINGS = {
    "Wan22EasyPrompt": Wan22EasyPrompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan22EasyPrompt": "Wan 2.2 Easy Prompt"
}