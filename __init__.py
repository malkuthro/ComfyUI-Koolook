import json
import os
import torch
from comfy.utils import common_upscale
from nodes import MAX_RESOLUTION

class Wan22EasyPrompt:
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

class ImageAspect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "base_on": (["Width", "Height"], {"default": "Width"}),
                "base_size": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "aspect_ratio": ("STRING", {"default": "16:9"}),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "nearest-exact"}),
                "keep_proportion": (["stretch", "letterbox", "pillarbox"], {"default": "stretch"}),
                "pad_red": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "pad_green": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "pad_blue": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "crop_position": (["top", "bottom", "left", "right", "center"], {"default": "center"}),
                "divisible_by": ("INT", {"default": 32, "min": 1, "max": 128, "step": 1}),
                "device": (["cpu", "cuda"], {"default": "cpu"}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "width", "height")
    FUNCTION = "adjust_to_aspect"
    CATEGORY = "Koolook/Image"
    DESCRIPTION = """
Adjusts the input image to a target resolution based on a base dimension (width or height),
an aspect ratio (e.g., '16:9' for landscape, '9:16' for portrait), ensuring both dimensions
are divisible by the specified value. Supports stretch or pad/crop modes for proportion control.
Outputs the resized image, optional mask, and final width/height for VFX pipeline integration.
"""

    def adjust_to_aspect(self, image, base_on, base_size, aspect_ratio, upscale_method, keep_proportion, pad_red, pad_green, pad_blue, crop_position, divisible_by, device, mask=None):
        def round_to_nearest_multiple(number, multiple):
            return multiple * round(number / multiple)

        # Parse aspect ratio (order matters: '16:9' implies width > height if base allows)
        try:
            ar_parts = [int(part.strip()) for part in aspect_ratio.split(':')]
            if len(ar_parts) != 2 or ar_parts[0] <= 0 or ar_parts[1] <= 0:
                raise ValueError("Aspect ratio must be 'w:h' with positive integers, e.g., '16:9'.")
            ar_width, ar_height = ar_parts
        except ValueError as e:
            raise ValueError(str(e))

        ratio = ar_width / ar_height  # w/h ratio

        # Make base_size divisible by 'divisible_by'
        base_rounded = max(divisible_by, round_to_nearest_multiple(base_size, divisible_by))

        # Calculate target dimensions
        if base_on == "Width":
            target_width = base_rounded
            computed_height = target_width / ratio
            target_height = max(divisible_by, round_to_nearest_multiple(computed_height, divisible_by))
        else:  # Height
            target_height = base_rounded
            computed_width = target_height * ratio
            target_width = max(divisible_by, round_to_nearest_multiple(computed_width, divisible_by))

        # Move to specified device
        image = image.to(device)
        if mask is not None:
            mask = mask.to(device)

        # Image processing (inspired by ResizeImage logic in image_nodes.py)
        B, H, W, C = image.shape
        image = image.movedim(-1, 1)  # Channels-first

        if mask is not None:
            mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).movedim(-1, 1)  # Adjust for interpolation

        if keep_proportion == "stretch":
            # Simple resize
            out_image = common_upscale(image, target_width, target_height, upscale_method, "disabled")
            if mask is not None:
                out_mask = torch.nn.functional.interpolate(mask, size=(target_height, target_width), mode="nearest")
            else:
                out_mask = None
        else:
            # Preserve aspect with pad/crop (letterbox or pillarbox)
            scale = min(target_width / W, target_height / H)
            new_w = round_to_nearest_multiple(W * scale, divisible_by)
            new_h = round_to_nearest_multiple(H * scale, divisible_by)

            resized_image = common_upscale(image, new_w, new_h, upscale_method, "disabled")
            if mask is not None:
                resized_mask = torch.nn.functional.interpolate(mask, size=(new_h, new_w), mode="nearest")

            # Determine padding based on mode and position
            pad_l = pad_r = pad_t = pad_b = 0
            if keep_proportion == "letterbox":  # Add bars top/bottom (for wider target)
                total_pad_h = target_height - new_h
                if crop_position == "top":
                    pad_b = total_pad_h
                elif crop_position == "bottom":
                    pad_t = total_pad_h
                else:  # center, left, right (left/right not affecting vertical)
                    pad_t = total_pad_h // 2
                    pad_b = total_pad_h - pad_t
            elif keep_proportion == "pillarbox":  # Add bars left/right (for taller target)
                total_pad_w = target_width - new_w
                if crop_position == "left":
                    pad_r = total_pad_w
                elif crop_position == "right":
                    pad_l = total_pad_w
                else:  # center, top, bottom
                    pad_l = total_pad_w // 2
                    pad_r = total_pad_w - pad_l

            # Create background with pad color (assume C=3 for RGB)
            pad_color = [pad_red / 255.0, pad_green / 255.0, pad_blue / 255.0]
            pad_color_tensor = torch.tensor(pad_color, dtype=image.dtype, device=device).view(1, 3, 1, 1)
            out_image = pad_color_tensor.expand(B, 3, target_height, target_width)
            out_image[:, :, pad_t:pad_t + new_h, pad_l:pad_l + new_w] = resized_image

            if mask is not None:
                out_mask = torch.zeros((B, 1, target_height, target_width), dtype=mask.dtype, device=device)
                out_mask[:, :, pad_t:pad_t + new_h, pad_l:pad_l + new_w] = resized_mask
            else:
                out_mask = None

        out_image = out_image.movedim(1, -1)
        if out_mask is not None:
            out_mask = out_mask.movedim(1, -1).squeeze(1)

        return (out_image, out_mask, target_width, target_height)

# Export the nodes
NODE_CLASS_MAPPINGS = {
    "Wan22EasyPrompt": Wan22EasyPrompt,
    "ImageAspect": ImageAspect
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan22EasyPrompt": "Wan 2.2 Easy Prompt",
    "ImageAspect": "Image Aspect"
}