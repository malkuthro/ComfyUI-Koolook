import json
import os

# Get the directory of this script to locate config.json
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.json')

# Load the JSON config
with open(config_path, 'r') as f:
    config = json.load(f)

# Build the INPUT_TYPES dict dynamically
input_types = {"required": {}}
for field in config['fields']:
    name = field['name']
    if field['type'] == 'combo':
        options = field['options']
        defaults = {"default": field.get('default', options[0])}
        input_types["required"][name] = (options, defaults)
    elif field['type'] == 'string':
        defaults = {"default": field.get('default', ""), "multiline": True}  # Optional: make it multiline
        input_types["required"][name] = ("STRING", defaults)

# Dynamically create the node class
class JsonConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return input_types

    RETURN_TYPES = (config.get('output_type', "STRING"),)
    FUNCTION = "execute"
    CATEGORY = config.get('category', "Custom")

    def execute(self, **kwargs):
        # Example: Concatenate selections into a prompt string (customize as needed)
        prompt_parts = []
        for key, value in kwargs.items():
            if value and value != "none":  # Skip 'none' if desired
                prompt_parts.append(f"{key}: {value}")
        return (", ".join(prompt_parts),)

# Register the node
NODE_CLASS_MAPPINGS = {
    config.get('node_name', "JsonConfigNode"): JsonConfigNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    config.get('node_name', "JsonConfigNode"): "JSON Config Node"
}