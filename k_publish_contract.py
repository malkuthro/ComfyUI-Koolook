# SPDX-License-Identifier: GPL-3.0-or-later
#
# ComfyUI-Koolook - publish contract nodes
# Copyright (C) 2026 ComfyUI-Koolook contributors (kforgelabs).
"""Controlled nodes for declaring externally published setup fields."""

from __future__ import annotations


INPUT_MODES: tuple[str, ...] = ("EXR", "QT", "Img", "Prompt")
INPUT_MODE_TO_INDEX = {name: index for index, name in enumerate(INPUT_MODES)}


def _resolve_mode_index(mode) -> int:
    if isinstance(mode, bool):
        return INPUT_MODE_TO_INDEX["Img"]
    if isinstance(mode, int) and 0 <= mode < len(INPUT_MODES):
        return mode
    text = str(mode).strip()
    if text.isdigit():
        index = int(text)
        if 0 <= index < len(INPUT_MODES):
            return index
    return INPUT_MODE_TO_INDEX.get(text, INPUT_MODE_TO_INDEX["Img"])


class Koolook_PublishInput:
    """App-facing setup inputs with stable output names for maintainers."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    list(INPUT_MODES),
                    {
                        "default": "Img",
                        "tooltip": "Source mode exposed to the external app. The switch output keeps this mode's stable numeric index.",
                    },
                ),
                "sequence_folder": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Directory path for sequence sources such as EXR or numbered image frames.",
                    },
                ),
                "qt_file": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Full path to one QuickTime/video file, including directory and filename.",
                    },
                ),
                "single_file": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Full path to one still image or single source file.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Prompt text field for setups that route a text prompt through the published app surface.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("sequence_folder", "qt_file", "single_file", "prompt", "switch")
    CATEGORY = "Koolook/Publish"
    FUNCTION = "run"
    DESCRIPTION = "Declare source inputs for a workflow published to an external app."

    def run(
        self,
        mode: str,
        sequence_folder: str,
        qt_file: str,
        single_file: str,
        prompt: str,
    ):
        return (
            sequence_folder,
            qt_file,
            single_file,
            prompt,
            _resolve_mode_index(mode),
        )


class Koolook_PublishOutput:
    """App-facing setup output controls."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Destination folder exposed to the external app and wired into downstream writer/path nodes.",
                    },
                ),
                "name": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Base output name exposed to the external app.",
                    },
                ),
                "version": (
                    "STRING",
                    {
                        "default": "1",
                        "multiline": False,
                        "tooltip": "Output version token or number exposed to the external app.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("folder", "name", "version")
    CATEGORY = "Koolook/Publish"
    FUNCTION = "run"
    DESCRIPTION = "Declare output folder, name, and version controls for a published setup."

    def run(self, folder: str, name: str, version: str):
        return (folder, name, version)


class Koolook_PublishResult:
    """Resolved setup result surfaced back to the external app."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "result": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Resolved result path, folder, or status string to show after the external app run finishes.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    CATEGORY = "Koolook/Publish"
    FUNCTION = "run"
    OUTPUT_NODE = True
    DESCRIPTION = "Report a resolved setup result string back to the published setup runner."

    def run(self, result: str):
        return {"ui": {"text": [result]}, "result": (result,)}


class Koolook_PublishRouter:
    """Switch-aligned payload router for externally published setup outputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "selector": (
                    "INT",
                    {
                        "default": INPUT_MODE_TO_INDEX["Img"],
                        "min": 0,
                        "max": len(INPUT_MODES) - 1,
                        "tooltip": "Switch index from Koolook Publish Input. Published setup runs keep only the matching writer branch.",
                    },
                ),
                "payload": (
                    "*",
                    {
                        "tooltip": "Main workflow payload to send to the switch-aligned writer branches.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("*", "*", "*", "*")
    RETURN_NAMES = INPUT_MODES
    CATEGORY = "Koolook/Publish"
    FUNCTION = "route"
    DESCRIPTION = "Route one payload into switch-aligned writer branches for published setup runs."

    def route(self, selector: int, payload):
        return (payload, payload, payload, payload)


NODE_CLASS_MAPPINGS = {
    "Koolook_PublishInput": Koolook_PublishInput,
    "Koolook_PublishOutput": Koolook_PublishOutput,
    "Koolook_PublishResult": Koolook_PublishResult,
    "Koolook_PublishRouter": Koolook_PublishRouter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koolook_PublishInput": "Koolook Publish Input",
    "Koolook_PublishOutput": "Koolook Publish Output",
    "Koolook_PublishResult": "Koolook Publish Result",
    "Koolook_PublishRouter": "Koolook Publish Router",
}
