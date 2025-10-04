import json
import os
import torch
# Note: common_upscale is not used here, but if needed for future, import from comfy.utils

class EasyWan22Prompt:
    """
    Wan 2.2 Easy Prompt node for ComfyUI.
    Loads dynamic inputs from config.json, adds a body text input, and outputs two strings (combined prompt and fields only) plus optional CONDITIONING.
    Includes an optional CLIP input to encode the combined prompt into CONDITIONING, allowing use as a positive prompt node.
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
                "category": "Koolook/Wan_Video",
                "node_name": "Wan_2.2",
                "fields": [],
                "output_type": "STRING"
            }

        # Build the INPUT_TYPES dict dynamically from config fields
        input_types = {
            "required": {},
            "optional": {
                "clip": ("CLIP",)
            }
        }
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

        # Add the body text input at the end (appears at the bottom)
        input_types["required"]["body"] = ("STRING", {
            "multiline": True,
            "default": "",
            "lazy": True
        })

        return input_types

    RETURN_TYPES = ("STRING", "STRING", "CONDITIONING",)
    RETURN_NAMES = ("combined_prompt", "fields_only", "conditioning",)
    FUNCTION = "execute"
    CATEGORY = "Koolook/Wan_Video"
    OUTPUT_NODE = False

    def check_lazy_status(self, **kwargs):
        # Evaluate all inputs if needed
        return list(kwargs.keys())

    def execute(self, body, clip=None, **kwargs):
        # Collect selected options (values only), excluding body and skipping "none"
        prompt_parts = [value for key, value in kwargs.items() if value and value != "none"]
        fields_str = ", ".join(prompt_parts)

        # Combined: fields options + body (with comma if both present)
        combined = fields_str
        if body:
            if combined:
                combined += ", " + body
            else:
                combined = body

        # Conditioning: Encode the combined prompt if clip is provided, else empty list
        if clip is not None:
            tokens = clip.tokenize(combined)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            conditioning = [[cond, {"pooled_output": pooled}]]
        else:
            conditioning = []

        return (combined, fields_str, conditioning,)

    # Optional: IS_CHANGED for re-execution if config changes
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Hash the config.json to re-execute if it changes
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config.json')
        try:
            with open(config_path, 'rb') as f:
                return hash(f.read())
        except FileNotFoundError:
            return "dummy"

# Individual node mappings
NODE_CLASS_MAPPINGS = {
    "EasyWan22Prompt": EasyWan22Prompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyWan22Prompt": "Wan 2.2 Easy Prompt"
}