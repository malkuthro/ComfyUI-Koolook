# SPDX-License-Identifier: GPL-3.0-or-later
#
# ComfyUI-Koolook - publish contract nodes
# Copyright (C) 2026 ComfyUI-Koolook contributors (kforgelabs).
"""Controlled nodes for declaring externally published setup fields."""

from __future__ import annotations


INPUT_MODES: tuple[str, ...] = ("EXR", "QT", "Img", "Prompt")
INPUT_MODE_TO_INDEX = {name: index for index, name in enumerate(INPUT_MODES)}


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
                        "tooltip": "External app source mode. The node outputs the stable numeric switch index.",
                    },
                ),
                "sequence_folder": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Folder path for EXR/image sequence sources.",
                    },
                ),
                "qt_file": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Full path to a QuickTime/video file.",
                    },
                ),
                "single_file": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Full path to a single source image/file.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Internal prompt placeholder for setups that need the shared input structure.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("sequence_folder", "qt_file", "single_file", "prompt", "switch")
    CATEGORY = "Koolook/Publish"
    FUNCTION = "run"

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
            INPUT_MODE_TO_INDEX.get(str(mode), INPUT_MODE_TO_INDEX["Img"]),
        )


class Koolook_PublishOutput:
    """App-facing setup output controls and result path placeholder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Folder where the setup should write outputs.",
                    },
                ),
                "name": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Base output name.",
                    },
                ),
                "version": (
                    "STRING",
                    {
                        "default": "1",
                        "multiline": False,
                        "tooltip": "Output version token or number.",
                    },
                ),
                "result": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Result path/status surfaced back to the external app.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("folder", "name", "version", "result")
    CATEGORY = "Koolook/Publish"
    FUNCTION = "run"

    def run(self, folder: str, name: str, version: str, result: str):
        return (folder, name, version, result)


NODE_CLASS_MAPPINGS = {
    "Koolook_PublishInput": Koolook_PublishInput,
    "Koolook_PublishOutput": Koolook_PublishOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koolook_PublishInput": "Koolook Publish Input",
    "Koolook_PublishOutput": "Koolook Publish Output",
}
