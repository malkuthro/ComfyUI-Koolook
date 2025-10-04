import os
import json

class Easy_Version:
    """
    A custom node that generates a version string like 'v001' based on an integer input.
    The integer is padded to three digits with leading zeros.
    Useful for VFX versioning in ComfyUI workflows, such as naming renders or sequences.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "version": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 999,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("version_string",)
    FUNCTION = "generate_version"
    CATEGORY = "VFX/Utils"
    OUTPUT_NODE = False

    def generate_version(self, version):
        # Format the version as 'v' followed by three-digit padded number
        version_str = f"v{version:03d}"
        return (version_str,)

# Individual node mappings
NODE_CLASS_MAPPINGS = {
    "Easy_Version": Easy_Version
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Easy_Version": "Easy Version"
}