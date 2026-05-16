# SPDX-License-Identifier: GPL-3.0-or-later
#
# ComfyUI-Koolook — Easy Video Combine
# Copyright (C) 2026 ComfyUI-Koolook contributors (kforgelabs).
#
# This file is part of ComfyUI-Koolook, licensed under GPL-3.0-or-later.
# See the LICENSE file at the repo root for the full text.
#
# Runtime composition over Kosinkadink/ComfyUI-VideoHelperSuite (GPL-3.0).
# No VHS source is copied here — Easy_VideoCombine subclasses
# VHS_VideoCombine and scoped-patches folder_paths.get_save_image_path
# for the duration of one combine_video() call, letting upstream's
# encoder run untouched while the output is redirected to an absolute
# path on disk. If VHS isn't installed alongside Koolook, the node is
# simply not registered.
#
# The os.path.isabs(filename_prefix) discrimination pattern is borrowed
# from spacepxl/ComfyUI-HQ-Image-Save's SaveEXR (MIT) — same idea: one
# field carries both relative (sandboxed) and absolute (direct) paths
# so no new pin is needed on the canvas. See forks/THIRD_PARTY.md for
# the full attribution + change log.
"""Path-aware Video Combine for ComfyUI-Koolook.

Exposes ``Easy_VideoCombine`` (display name *Easy Video Combine
(Koolook)*, category ``Koolook/Video``). Identical to VHS's
``Video Combine`` except an absolute ``filename_prefix`` writes
directly to that location, and an optional ``create_path_if_missing``
toggle controls whether the parent directory is auto-created.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

try:
    import folder_paths  # ComfyUI core; always present at runtime.
    from videohelpersuite.nodes import VideoCombine as _VHS_VideoCombine
    _VHS_AVAILABLE = True
    _VHS_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:
    _VHS_AVAILABLE = False
    _VHS_IMPORT_ERROR = _exc


def _resolve_abs_target(
    filename_prefix: str,
    create_path_if_missing: bool,
) -> Optional[Tuple[str, str]]:
    """Discriminate relative vs absolute ``filename_prefix``.

    Returns ``None`` when the prefix is relative — caller should let
    upstream's sandboxed flow handle it unchanged. Returns
    ``(abs_dir, abs_basename)`` when the prefix is absolute; upstream's
    counter pattern will append ``"_<counter:05>.<ext>"`` to
    ``abs_basename`` inside ``abs_dir``.

    Raises ``FileNotFoundError`` if the parent directory does not exist
    and ``create_path_if_missing`` is ``False`` — a typo in a render
    path should surface as a clear error, not silently spawn a stray
    directory tree. Raises ``ValueError`` if the absolute prefix has no
    filename component (e.g. a bare drive root like ``"C:\\"``).
    """
    if not os.path.isabs(filename_prefix):
        return None

    normed = os.path.normpath(filename_prefix)
    abs_dir = os.path.dirname(normed)
    abs_base = os.path.basename(normed)

    if abs_base == "":
        raise ValueError(
            f"filename_prefix '{filename_prefix}' is an absolute path but "
            f"has no filename component. Append a filename root, e.g. "
            f"'{normed.rstrip(os.sep)}{os.sep}clipname'."
        )

    if not os.path.exists(abs_dir):
        if create_path_if_missing:
            os.makedirs(abs_dir, exist_ok=True)
        else:
            raise FileNotFoundError(
                f"Directory does not exist: {abs_dir}. Enable "
                f"'create_path_if_missing' on the Easy Video Combine node "
                f"to auto-create it."
            )

    return abs_dir, abs_base


if _VHS_AVAILABLE:
    class Easy_VideoCombine(_VHS_VideoCombine):
        """Path-aware variant of VHS's Video Combine.

        Behavior matrix:

        - ``filename_prefix`` is relative (e.g. ``"AnimateDiff"``,
          ``"shots/clip01"``) — pure passthrough to
          :meth:`videohelpersuite.nodes.VideoCombine.combine_video`. The
          ComfyUI output-directory sandbox stays in force.
        - ``filename_prefix`` is absolute (e.g.
          ``"E:/renders/shot01/v003"``) — the parent directory is
          resolved (and optionally created), then upstream's combine
          runs against that location. Counter (``_00001``), extension
          (``.mp4`` / ``.gif`` / …), metadata PNG, and audio mux all
          land alongside the video in ``abs_dir``.

        The UI preview tile in the workflow shows the standard "video
        written" pill, but Comfy's ``/view`` endpoint can only serve
        files under its own output directory — so absolute-path renders
        don't surface a clickable thumbnail. The file is still written,
        and the absolute path appears in the saved-files list.
        """

        CATEGORY = "Koolook/Video"

        @classmethod
        def INPUT_TYPES(cls):
            types = _VHS_VideoCombine.INPUT_TYPES()
            # Override upstream's bare filename_prefix entry with a
            # tooltip that documents the new dual-mode semantics.
            types["required"]["filename_prefix"] = (
                "STRING",
                {
                    "default": "AnimateDiff",
                    "tooltip": (
                        "Relative path -> ComfyUI's output directory "
                        "(sandboxed, identical to upstream). Absolute "
                        "path (e.g. 'E:/renders/shot01/v003') -> writes "
                        "directly there. Counter (_00001) and extension "
                        "are appended; treat this as the filename root, "
                        "not a directory."
                    ),
                },
            )
            types["optional"]["create_path_if_missing"] = (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": (
                        "When filename_prefix is absolute and the parent "
                        "directory does not exist, auto-create it. Off "
                        "by default so typos surface as errors instead "
                        "of creating stray directories."
                    ),
                },
            )
            return types

        def combine_video(self, *args, **kwargs):
            filename_prefix = kwargs.get("filename_prefix", "AnimateDiff")
            create_path_if_missing = kwargs.pop("create_path_if_missing", False)

            target = _resolve_abs_target(filename_prefix, create_path_if_missing)
            if target is None:
                return super().combine_video(*args, **kwargs)

            abs_dir, abs_base = target

            # VHS calls folder_paths.get_save_image_path() exactly once
            # in combine_video(). Patch it to return our absolute target
            # so the rest of upstream's pipeline (counter scan, metadata
            # PNG, ffmpeg encode, audio mux, batch manager) writes into
            # abs_dir/abs_base_*. The patch is scoped to one super() call
            # and restored in finally to guarantee no leak on exception.
            #
            # Concurrency note: ComfyUI executes nodes sequentially within
            # a workflow and serializes workflows at the prompt level, so
            # no other node observes the patched function during the call.
            original_get_save_path = folder_paths.get_save_image_path

            def patched_get_save_path(prefix, output_dir, *_a, **_kw):
                # VHS expects:
                # (full_output_folder, filename, counter_seed, subfolder, prefix)
                # Upstream re-scans full_output_folder for the real
                # counter, so counter_seed (0) is ignored. subfolder=""
                # makes the UI preview tile bypass the relative-to-
                # output-dir link Comfy builds for the /view endpoint.
                return (abs_dir, abs_base, 0, "", abs_base)

            try:
                folder_paths.get_save_image_path = patched_get_save_path
                return super().combine_video(*args, **kwargs)
            finally:
                folder_paths.get_save_image_path = original_get_save_path

    NODE_CLASS_MAPPINGS = {"Easy_VideoCombine": Easy_VideoCombine}
    NODE_DISPLAY_NAME_MAPPINGS = {"Easy_VideoCombine": "Easy Video Combine (Koolook)"}
else:
    print(
        f"[Koolook] Easy_VideoCombine skipped: "
        f"ComfyUI-VideoHelperSuite not importable ({_VHS_IMPORT_ERROR}). "
        f"Install VHS from ComfyUI-Manager to enable this node."
    )
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
