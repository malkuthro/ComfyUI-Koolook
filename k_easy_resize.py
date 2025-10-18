import torch
from comfy.utils import common_upscale
from nodes import MAX_RESOLUTION

class EasyResize:
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
                "pad_color": ("STRING", {"default": "0, 0, 0"}),
                "crop_position": (["top", "bottom", "left", "right", "center"], {"default": "center"}),
                "divisible_by": ("INT", {"default": 32, "min": 1, "max": 128, "step": 1}),
                "device": (["cpu", "cuda"], {"default": "cpu"}),
            },
            "optional": {
                "mask": ("MASK",),
                "panel_color": ("STRING", {"default": "0, 0, 0"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "width", "height", "COLOR_PANEL")
    FUNCTION = "adjust_to_aspect"
    CATEGORY = "Koolook/Image"
    DESCRIPTION = """
Adjusts the input image to a target resolution based on a base dimension (width or height),
an aspect ratio (e.g., '16:9' for landscape, '9:16' for portrait), ensuring both dimensions
are divisible by the specified value. Supports stretch or pad/crop modes for proportion control.
Outputs the resized image, optional mask, and final width/height for VFX pipeline integration.
Also outputs a solid color panel (canvas) of the target size using the specified panel_color.
"""

    def adjust_to_aspect(self, image, base_on, base_size, aspect_ratio, upscale_method, keep_proportion, pad_color, crop_position, divisible_by, device, mask=None, panel_color="0, 0, 0"):
        def round_to_nearest_multiple(number, multiple):
            return multiple * round(number / multiple)

        if divisible_by < 1:
            divisible_by = 1

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
            mask = mask.unsqueeze(1)  # Simplified: Add channel dim for interpolation [B, 1, H, W]

        # Image processing (inspired by ResizeImage logic in image_nodes.py)
        B, H, W, C = image.shape
        image = image.movedim(-1, 1)  # Channels-first

        has_alpha = (C == 4)

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

            # Parse pad_color
            pad_color = [float(x.strip()) for x in pad_color.split(',')]
            if len(pad_color) != 3:
                raise ValueError("Pad color must be three comma-separated floats, e.g., '0, 0, 0'.")

            # Create background with pad color (RGB or RGBA)
            pad_channels = 4 if has_alpha else 3
            pad_color_tensor = torch.zeros((1, pad_channels, 1, 1), dtype=image.dtype, device=device)
            pad_color_tensor[0, :3, 0, 0] = torch.tensor(pad_color, dtype=image.dtype, device=device)
            if has_alpha:
                # Default alpha to 1 (opaque) for padding
                pad_color_tensor[0, 3, 0, 0] = 1.0
            out_image = pad_color_tensor.expand(B, pad_channels, target_height, target_width).clone()
            out_image[:, :, pad_t:pad_t + new_h, pad_l:pad_l + new_w] = resized_image

            if mask is not None:
                out_mask = torch.zeros((B, 1, target_height, target_width), dtype=mask.dtype, device=device)
                out_mask[:, :, pad_t:pad_t + new_h, pad_l:pad_l + new_w] = resized_mask
            else:
                out_mask = None

        out_image = out_image.movedim(1, -1)
        if out_mask is not None:
            out_mask = out_mask.movedim(1, -1).squeeze(-1)  # Squeeze last dim (channel=1) -> [B, H', W']

        # Create color panel
        # Parse panel_color
        panel_color_list = [float(x.strip()) for x in panel_color.split(',')]
        if len(panel_color_list) != 3:
            raise ValueError("Panel color must be three comma-separated floats, e.g., '0, 0, 0'.")

        # Create panel tensor (B=1, C=3, H=target_height, W=target_width)
        panel_channels = 3  # RGB, no alpha
        color_panel = torch.zeros((1, panel_channels, target_height, target_width), dtype=image.dtype, device=device)
        color_panel[0, :, :, :] = torch.tensor(panel_color_list, dtype=image.dtype, device=device).view(panel_channels, 1, 1)
        color_panel = color_panel.movedim(1, -1)  # to [1, H, W, 3]

        return (out_image, out_mask, target_width, target_height, color_panel)

# Individual node mappings
NODE_CLASS_MAPPINGS = {
    "EasyResize": EasyResize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyResize": "Easy Resize"
}