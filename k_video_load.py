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
    # VHS LoadVideoPath wraps the shared decoder with path validation that
    # rejects absolute/local handoff paths. Existing local videos can use the
    # same shared decoder directly; fail loudly if a future VHS refactor moves it.
    _VHS_LOAD_VIDEO_FN = _VHS_LoadVideoPath.load_video.__globals__.get("load_video")
    if _VHS_LOAD_VIDEO_FN is None:
        raise RuntimeError("VHS shared load_video function not found.")
    _VHS_AVAILABLE = True
    _VHS_IMPORT_ERROR: Optional[BaseException] = None
except (ImportError, RuntimeError) as _exc:
    _VHS_AVAILABLE = False
    _VHS_IMPORT_ERROR = _exc
    _VHS_LoadVideoPath = None  # type: ignore[assignment]
    _VHS_LOAD_VIDEO_FN = None


# Returned by _compose_input_video_path when input_path is an existing folder
# with no filename -- typically the unselected branch of an upstream mode switch
# (e.g. an EXR sequence dir handed to the video loader). load_video turns this
# into an empty, correctly-shaped result instead of crashing the whole prompt.
EMPTY_BRANCH_SENTINEL = "\x00__koolook_empty_video_branch__"

_SENTINEL_STRINGS = ("undefined", "null", "none")
_VIDEO_EXTENSIONS = {
    ".avi",
    ".gif",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".webm",
}


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


def _looks_like_video_file(path: str) -> bool:
    return os.path.splitext(_basename_any_sep(path))[1].lower() in _VIDEO_EXTENSIONS


def _strip_outer_quotes(path: str) -> str:
    return path.strip().strip('"').strip("'").strip()


def _is_existing_local_video_path(path: str) -> bool:
    clean = _strip_outer_quotes(path)
    return _looks_like_video_file(clean) and os.path.isfile(clean)


def _input_path_is_existing_dir(path: str, input_root: str | None = None) -> bool:
    """True when ``path`` resolves to a real directory on disk.

    Absolute dirs are checked directly; relative dirs are rooted under ComfyUI's
    input directory, matching how the loader resolves relative paths elsewhere.
    """
    candidate = _existing_local_path_candidate(path, input_root)
    return bool(candidate) and os.path.isdir(candidate)


def _existing_local_path_candidate(path: str, input_root: str | None = None) -> str:
    clean = _strip_outer_quotes(path)
    if not clean:
        return ""
    if os.path.isabs(clean):
        return os.path.normpath(clean)
    root = input_root if input_root is not None else _input_root()
    return os.path.normpath(os.path.join(root, clean))


def _normalize_path_input(value, input_root: str | None = None) -> str:
    """Normalize a path input while preserving wrapped full paths.

    Multiline text can mean either "directory + filename" or a single path
    wrapped by a text node. Prefer candidates that exist on disk, then candidates
    that at least look like video files, and otherwise keep the first line.
    """
    lines = _clean_text_lines(value)
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0]

    joined_text = "".join(lines)
    joined_path = os.path.join(lines[0].rstrip("/\\"), *lines[1:])
    candidates = [joined_text, joined_path]

    for candidate in candidates:
        resolved = _existing_local_path_candidate(candidate, input_root)
        if os.path.isfile(resolved):
            return candidate

    for candidate in candidates:
        if _looks_like_video_file(candidate):
            return candidate

    return lines[0]


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

    If ``input_path`` itself points at a video file and ``video`` is empty,
    that full file path is passed through. This lets
    ``Easy_VideoCombine.video_path`` wire directly into ``input_path``. If a
    multiline text node wraps that path, the lines are joined back together
    before validation/loading.

    When ``input_path`` is empty and ``video`` contains multiple non-empty
    lines, the first line is treated as the directory and the second as the
    filename; later lines are ignored. This matches connected text blocks that
    package a path/name pair for loader handoff.
    """
    raw_video_lines = _clean_text_lines(video)
    input_path = _normalize_path_input(input_path, input_root)
    if not input_path and len(raw_video_lines) >= 2:
        input_path = raw_video_lines[0]
        video = raw_video_lines[1]
    else:
        video = _normalize_text_input(video)
    if not input_path:
        return video

    clean_dir = input_path.rstrip("/\\")
    if not video and _looks_like_video_file(clean_dir):
        if os.path.isabs(clean_dir):
            return os.path.normpath(clean_dir)
        root = input_root if input_root is not None else _input_root()
        return os.path.normpath(os.path.join(root, clean_dir))

    name = _basename_any_sep(video) or video
    if not name:
        # input_path is a directory with no filename. If that folder actually
        # exists it is almost always the unselected branch of an upstream mode
        # switch -- e.g. an EXR sequence dir routed into the video loader. Signal
        # a graceful empty so the loader returns nothing instead of crashing the
        # whole prompt (a lenient sequence loader would just ignore it). A path
        # that does not exist is a genuine mistake and still raises.
        if _input_path_is_existing_dir(clean_dir, input_root):
            return EMPTY_BRANCH_SENTINEL
        raise ValueError("video must include a filename when input_path is set.")

    if os.path.isabs(clean_dir):
        return os.path.normpath(os.path.join(clean_dir, name))

    root = input_root if input_root is not None else _input_root()
    return os.path.normpath(os.path.join(root, clean_dir, name))


def _empty_value_for_type(type_name: str):
    """Build an empty, correctly-typed output value for a VHS return slot.

    ``torch`` is imported lazily so the path-composition logic stays importable
    (and unit-testable) without torch installed; it is always present at runtime
    inside ComfyUI.
    """
    name = str(type_name)
    if name in ("IMAGE", "MASK", "LATENT"):
        import torch

        if name == "IMAGE":
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        if name == "MASK":
            return torch.zeros((1, 64, 64), dtype=torch.float32)
        return {"samples": torch.zeros((1, 4, 8, 8), dtype=torch.float32)}
    if name == "AUDIO":
        import torch

        return {"waveform": torch.zeros((1, 2, 1), dtype=torch.float32), "sample_rate": 44100}
    if name == "INT":
        return 0
    if name == "FLOAT":
        return 0.0
    if name == "STRING":
        return ""
    if name == "VHS_VIDEOINFO":
        return {}
    return None


def _empty_load_video_result(return_types) -> tuple:
    return tuple(_empty_value_for_type(t) for t in return_types)


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
            composed = _compose_input_video_path(
                kwargs.get("video", ""),
                kwargs.pop("input_path", ""),
            )
            if composed == EMPTY_BRANCH_SENTINEL:
                # Unselected branch (existing folder, no filename): return an
                # empty, correctly-shaped result so the prompt does not crash.
                return _empty_load_video_result(self.RETURN_TYPES)
            kwargs["video"] = composed
            if _is_existing_local_video_path(kwargs["video"]):
                kwargs["video"] = _strip_outer_quotes(kwargs["video"])
                return _VHS_LOAD_VIDEO_FN(*args, **kwargs)
            return super().load_video(*args, **kwargs)

        @classmethod
        def IS_CHANGED(cls, video, input_path="", **kwargs):
            resolved = _compose_input_video_path(video, input_path)
            if resolved == EMPTY_BRANCH_SENTINEL:
                return EMPTY_BRANCH_SENTINEL
            return _VHS_LoadVideoPath.IS_CHANGED(resolved, **kwargs)

        @classmethod
        def VALIDATE_INPUTS(cls, video, input_path="", **kwargs):
            # Linked inputs are unresolved during ComfyUI validation; execution
            # receives the real value after the upstream text node runs.
            if video is None or input_path is None:
                return True
            resolved = _compose_input_video_path(video, input_path)
            # An existing-folder branch validates fine; it degrades to empty at
            # run time rather than blocking the whole prompt.
            if resolved == EMPTY_BRANCH_SENTINEL:
                return True
            if _is_existing_local_video_path(resolved):
                return True
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
