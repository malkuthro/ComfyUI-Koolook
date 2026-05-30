# SPDX-License-Identifier: GPL-3.0-or-later
#
# ComfyUI-Koolook - Easy Load Video
# Copyright (C) 2026 ComfyUI-Koolook contributors (kforgelabs).
#
# Runtime composition over Kosinkadink/ComfyUI-VideoHelperSuite (GPL-3.0).
# No VHS source is copied here. Easy_LoadVideo subclasses VHS_LoadVideoPath
# and composes a split input_path + video filename into the path string that
# upstream already knows how to load.
"""Path-aware Video Helper Suite loader for ComfyUI-Koolook."""
from __future__ import annotations

import os
from typing import Optional

try:
    import folder_paths  # ComfyUI core; present at runtime.
    import nodes as _comfy_nodes_module

    _VHS_LoadVideoPath = _comfy_nodes_module.NODE_CLASS_MAPPINGS.get("VHS_LoadVideoPath")
    if _VHS_LoadVideoPath is None:
        raise RuntimeError(
            "VHS_LoadVideoPath not present in ComfyUI's NODE_CLASS_MAPPINGS "
            "at Koolook load time. Either ComfyUI-VideoHelperSuite isn't "
            "installed, or it loaded after Koolook."
        )
    _VHS_AVAILABLE = True
    _VHS_IMPORT_ERROR: Optional[BaseException] = None
except (ImportError, RuntimeError) as _exc:
    _VHS_AVAILABLE = False
    _VHS_IMPORT_ERROR = _exc
    _VHS_LoadVideoPath = None  # type: ignore[assignment]


_SENTINEL_STRINGS = ("undefined", "null", "none")


def _normalize_text_input(value) -> str:
    """Coerce a ComfyUI STRING widget value to a clean Python string."""
    if value is None:
        return ""
    s = str(value).strip().strip('"').strip("'").strip()
    if "\n" in s or "\r" in s:
        for line in s.replace("\r", "\n").split("\n"):
            line = line.strip()
            if line and line.lower() not in _SENTINEL_STRINGS:
                s = line
                break
        else:
            s = ""
    if s.lower() in _SENTINEL_STRINGS:
        return ""
    return s


def _clean_text_lines(value) -> list[str]:
    """Return non-empty, non-sentinel lines from a connected text widget."""
    if value is None:
        return []
    s = str(value).strip().strip('"').strip("'").strip()
    lines = []
    for line in s.replace("\r", "\n").split("\n"):
        line = line.strip().strip('"').strip("'").strip()
        if line and line.lower() not in _SENTINEL_STRINGS:
            lines.append(line)
    return lines


def _input_root() -> str:
    try:
        return folder_paths.get_input_directory()
    except Exception:
        return os.getcwd()


def _basename_any_sep(path: str) -> str:
    return path.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]


def _compose_input_video_path(
    video: str,
    input_path: str,
    input_root: str | None = None,
) -> str:
    """Resolve split ``input_path`` + ``video`` into one loader path.

    Empty ``input_path`` is pass-through, so users can still paste a full path
    or URL into ``video`` exactly like upstream VHS_LoadVideoPath.

    When ``input_path`` is set, ``video`` is treated as a filename field:
    path components are stripped, then the name is joined under the directory.
    Absolute directories are used directly; relative directories are rooted in
    ComfyUI's input directory.

    When ``input_path`` is empty and ``video`` contains multiple non-empty
    lines, the first line is treated as the directory and the second as the
    filename; later lines are ignored. This matches connected text blocks that
    package a path/name pair for loader handoff.
    """
    raw_video_lines = _clean_text_lines(video)
    input_path = _normalize_text_input(input_path)
    if not input_path and len(raw_video_lines) >= 2:
        input_path = raw_video_lines[0]
        video = raw_video_lines[1]
    else:
        video = _normalize_text_input(video)
    if not input_path:
        return video

    name = _basename_any_sep(video) or video
    if not name:
        raise ValueError("video must include a filename when input_path is set.")

    clean_dir = input_path.rstrip("/\\")
    if os.path.isabs(clean_dir):
        return os.path.normpath(os.path.join(clean_dir, name))

    root = input_root if input_root is not None else _input_root()
    return os.path.normpath(os.path.join(root, clean_dir, name))


if _VHS_AVAILABLE:
    class Easy_LoadVideo(_VHS_LoadVideoPath):
        """VHS Load Video Path variant with split directory + filename fields."""

        CATEGORY = "Koolook/Video"
        DESCRIPTION = "Load Video variant with split input path and filename fields."

        @classmethod
        def INPUT_TYPES(cls):
            types = _VHS_LoadVideoPath.INPUT_TYPES()
            required = dict(types.get("required", {}))
            required.pop("video", None)
            types["required"] = {
                "input_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Folder containing the video. Absolute paths load "
                            "directly; relative paths start at ComfyUI's input folder."
                        ),
                    },
                ),
                "video": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Video filename, or a full path/URL when input_path is empty.",
                    },
                ),
                **required,
            }
            return types

        def load_video(self, *args, **kwargs):
            kwargs["video"] = _compose_input_video_path(
                kwargs.get("video", ""),
                kwargs.pop("input_path", ""),
            )
            return super().load_video(*args, **kwargs)

        @classmethod
        def IS_CHANGED(cls, video, input_path="", **kwargs):
            resolved = _compose_input_video_path(video, input_path)
            return _VHS_LoadVideoPath.IS_CHANGED(resolved, **kwargs)

        @classmethod
        def VALIDATE_INPUTS(cls, video, input_path="", **kwargs):
            resolved = _compose_input_video_path(video, input_path)
            return _VHS_LoadVideoPath.VALIDATE_INPUTS(resolved)

    NODE_CLASS_MAPPINGS = {"Easy_LoadVideo": Easy_LoadVideo}
    NODE_DISPLAY_NAME_MAPPINGS = {"Easy_LoadVideo": "Easy Load Video (Koolook)"}
else:
    print(
        f"[Koolook] Easy_LoadVideo skipped: "
        f"ComfyUI-VideoHelperSuite not importable ({_VHS_IMPORT_ERROR}). "
        f"Install VHS from ComfyUI-Manager to enable this node."
    )
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
