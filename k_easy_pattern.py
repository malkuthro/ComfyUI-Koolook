# SPDX-License-Identifier: GPL-3.0-or-later
#
# ComfyUI-Koolook — Easy Pattern (Koolook variant)
# Copyright (C) 2026 ComfyUI-Koolook contributors (kforgelabs).
#
# This file is part of ComfyUI-Koolook, licensed under GPL-3.0-or-later.
# See the LICENSE file at the repo root for the full text.
#
# Originally drafted in an external AI-assisted chat session as a small
# standalone MIT-licensed test node, then ingested into ComfyUI-Koolook
# and relicensed under the project's GPL-3.0-or-later. No third-party
# upstream repository — this is in-house code.

"""
Easy Pattern node for ComfyUI-Koolook.

Generates a batch of solid-color images, optionally stamped with a per-frame
index number. Useful for testing sequence merging, frame insertion, batch
indexing, video pipeline order — anywhere a more flexible alternative to
KJNodes' `ImageBatchTestPattern` is needed.
"""

import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def _hex_to_rgb(hex_str):
    """Accepts '#RRGGBB', 'RRGGBB', '#RGB', or a 3-tuple-like string. Falls back to black."""
    if not isinstance(hex_str, str):
        return (0, 0, 0)
    s = hex_str.strip().lstrip("#")
    try:
        if len(s) == 3:
            return tuple(int(c * 2, 16) for c in s)
        if len(s) == 6:
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError:
        pass
    return (0, 0, 0)


# 50% Gray = sRGB (128, 128, 128); matches the `Gray` option in
# `EasyResize_Koolook`'s `pad_color_mode` and `easy_ImageBatch`'s
# `placeholder_color`, so the meaning of "Gray" is consistent across nodes.
PRESET_RGB = {
    "White": (255, 255, 255),
    "Black": (0, 0, 0),
    "Gray":  (128, 128, 128),
}


def _resolve_color(mode, custom_hex):
    """Mode wins; custom hex is only consulted when mode == 'Custom'."""
    if mode in PRESET_RGB:
        return PRESET_RGB[mode]
    return _hex_to_rgb(custom_hex)


def _load_font(size):
    """Try a few common font paths; fall back to PIL default if none work."""
    candidates = [
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        # Windows
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


class EasyPattern:
    """
    Configurable test-pattern batch.

    Outputs: IMAGE tensor of shape [B, H, W, 3] with values in [0, 1].
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Number of test-pattern images to generate.",
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 16,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Output image width.",
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 16,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Output image height.",
                }),
                "bg_color_mode": (["White", "Black", "Gray", "Custom"], {
                    "default": "Custom",
                    "tooltip": "Background color preset. Custom reads bg_color.",
                }),
                "show_text": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Draw the generated frame number text on each image.",
                }),
                "text_color_mode": (["White", "Black", "Gray", "Custom"], {
                    "default": "White",
                    "tooltip": "Text color preset. Custom reads text_color.",
                }),
                "start_from": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "tooltip": "Number shown on the first generated frame.",
                }),
                "step": ("INT", {
                    "default": 1,
                    "min": -1000,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Number added for each next frame. Negative values count backward.",
                }),
                "font_size": ("INT", {
                    "default": 256,
                    "min": 8,
                    "max": 1024,
                    "step": 1,
                    "tooltip": "Text size. Very large values may clip if the label is wider than the image.",
                }),
                "position": (
                    ["center", "top-left", "top-right", "bottom-left", "bottom-right"],
                    {
                        "default": "center",
                        "tooltip": "Where to place the text label.",
                    },
                ),
                "zero_pad": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Minimum digit count. 7 with padding 3 becomes 007.",
                }),
            },
            "optional": {
                "bg_color":   ("STRING", {
                    "default": "#C71585",
                    "tooltip": "Custom background color as #RRGGBB, RRGGBB, or #RGB.",
                }),
                "text_color": ("STRING", {
                    "default": "#FFFFFF",
                    "tooltip": "Custom text color as #RRGGBB, RRGGBB, or #RGB.",
                }),
                "prefix": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Text added before the generated number.",
                }),
                "suffix": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Text added after the generated number.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    CATEGORY = "Koolook/Testing"

    def generate(
        self,
        batch_size,
        width,
        height,
        bg_color_mode,
        show_text,
        text_color_mode,
        start_from,
        step,
        font_size,
        position,
        zero_pad=0,
        bg_color="#C71585",
        text_color="#FFFFFF",
        prefix="",
        suffix="",
    ):
        bg_rgb = _resolve_color(bg_color_mode, bg_color)
        fg_rgb = _resolve_color(text_color_mode, text_color)
        font = _load_font(font_size) if show_text else None

        frames = []
        for i in range(batch_size):
            img = Image.new("RGB", (width, height), bg_rgb)

            if show_text:
                n = start_from + i * step
                num_str = f"{n:0{zero_pad}d}" if zero_pad > 0 else str(n)
                text = f"{prefix}{num_str}{suffix}"

                draw = ImageDraw.Draw(img)
                # Measure with textbbox so it works on modern Pillow; fall back
                # to the deprecated textsize on very old installs.
                try:
                    bbox = draw.textbbox((0, 0), text, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    bx, by = bbox[0], bbox[1]
                except AttributeError:
                    tw, th = draw.textsize(text, font=font)
                    bx, by = 0, 0

                pad = max(8, font_size // 8)
                if position == "center":
                    x = (width - tw) // 2 - bx
                    y = (height - th) // 2 - by
                elif position == "top-left":
                    x, y = pad - bx, pad - by
                elif position == "top-right":
                    x, y = width - tw - pad - bx, pad - by
                elif position == "bottom-left":
                    x, y = pad - bx, height - th - pad - by
                else:  # bottom-right
                    x, y = width - tw - pad - bx, height - th - pad - by

                draw.text((x, y), text, font=font, fill=fg_rgb)

            arr = np.array(img, dtype=np.float32) / 255.0
            frames.append(arr)

        batch = np.stack(frames, axis=0)  # [B, H, W, 3]
        return (torch.from_numpy(batch),)


NODE_CLASS_MAPPINGS = {
    "Easy_Pattern": EasyPattern,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Easy_Pattern": "Easy Pattern (Koolook)",
}
