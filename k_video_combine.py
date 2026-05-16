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
    # Reach the VHS VideoCombine class through ComfyUI's global node
    # registry rather than `from videohelpersuite.nodes import ...`.
    # VHS uses relative imports internally (`from .logger import ...`,
    # `from .utils import ...`) which only resolve correctly when VHS
    # is loaded through ComfyUI's custom-node loader, not through a
    # plain absolute import. The global registry is the documented
    # cross-pack interface — ComfyUI populates it as each pack loads,
    # and Koolook loads after VHS in every install we've seen because
    # `comfyui-videohelpersuite` sorts before `koolook` (and even an
    # uppercase `ComfyUI-VideoHelperSuite` sorts before
    # `ComfyUI-Koolook` if both are present). If for some reason VHS
    # isn't loaded yet, the lookup returns None and we self-skip with
    # a clear message.
    import nodes as _comfy_nodes_module
    _VHS_VideoCombine = _comfy_nodes_module.NODE_CLASS_MAPPINGS.get("VHS_VideoCombine")
    if _VHS_VideoCombine is None:
        raise RuntimeError(
            "VHS_VideoCombine not present in ComfyUI's NODE_CLASS_MAPPINGS "
            "at Koolook load time. Either ComfyUI-VideoHelperSuite isn't "
            "installed, or it loaded after Koolook (unexpected — check "
            "custom_nodes folder names)."
        )
    _VHS_AVAILABLE = True
    _VHS_IMPORT_ERROR: Optional[BaseException] = None
except (ImportError, RuntimeError) as _exc:
    _VHS_AVAILABLE = False
    _VHS_IMPORT_ERROR = _exc
    _VHS_VideoCombine = None  # type: ignore[assignment]


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


_SENTINEL_STRINGS = ("undefined", "null", "none")


def _normalize_text_input(value) -> str:
    """Coerce a ComfyUI STRING widget value to a clean Python string.

    ComfyUI's frontend has a long-running quirk: when a STRING input
    with an empty default sits untouched, the value sometimes arrives
    at the backend as the literal string ``"undefined"`` (or ``"null"``
    / ``"None"``) instead of ``""``. Treating those as valid input
    causes nonsensical ``undefined/`` subdirectories on disk.

    Also defends against:
    - Surrounding quotes (users paste paths from Explorer with quotes).
    - Trailing whitespace including newlines (WAS Text Multiline can
      leave a trailing newline when the user pressed Enter at the end).
    - Embedded sentinel components separated by a newline (a multi-line
      Text Multiline node with a second sentinel-looking line).

    Any sentinel string (case-insensitive, post-strip) becomes ``""``;
    everything else is the cleaned value.
    """
    if value is None:
        return ""
    s = str(value).strip().strip('"').strip("'").strip()
    # If a newline split the value across lines, take the first
    # non-empty non-sentinel line.
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


def _strip_sentinel_components(path: str) -> str:
    """Remove any path component that's exactly a sentinel string.

    Belt-and-suspenders defense: if anything upstream of us managed
    to inject ``undefined`` / ``null`` / ``None`` as a path component
    (despite the per-field normalization above), drop it before the
    path reaches the filesystem. Preserves the rest of the path,
    including the drive root on Windows.

    Returns the path unchanged if no sentinels found, so this is a
    cheap pass for the common case.
    """
    if not path:
        return path
    # Split on whichever separator is present, defaulting to os.sep.
    sep = "\\" if "\\" in path else "/"
    parts = path.split(sep)
    cleaned = [
        p for p in parts
        if p == "" or p.lower() not in _SENTINEL_STRINGS
    ]
    if len(cleaned) == len(parts):
        return path
    return sep.join(cleaned)


def _compose_prefix(filename_prefix: str, output_directory: str) -> str:
    """Combine ``output_directory`` + ``filename_prefix`` into one path-prefix.

    When ``output_directory`` is empty, ``filename_prefix`` is returned
    unchanged — it retains the overloaded "relative-or-absolute prefix"
    semantics for back-compat with users who like the one-field flow.

    When ``output_directory`` is set, only the basename of
    ``filename_prefix`` becomes the filename root (any path components
    the user accidentally typed there are stripped, so they can change
    the name without re-typing the directory). The directory keeps its
    shape (absolute or relative), so the downstream
    :func:`_resolve_abs_target` discrimination still works on the
    composed prefix.
    """
    if not output_directory:
        return filename_prefix
    name_root = os.path.basename(filename_prefix.rstrip("/\\")) or "AnimateDiff"
    clean_dir = output_directory.rstrip("/\\")
    return clean_dir + os.sep + name_root


if _VHS_AVAILABLE:
    class Easy_VideoCombine(_VHS_VideoCombine):
        """Video Combine variant with absolute-path output."""

        CATEGORY = "Koolook/Video"
        DESCRIPTION = "Video Combine variant with absolute-path output."

        @classmethod
        def INPUT_TYPES(cls):
            types = _VHS_VideoCombine.INPUT_TYPES()
            types["required"]["filename_prefix"] = (
                "STRING",
                {
                    "default": "AnimateDiff",
                    "tooltip": "Filename root. Counter and extension are appended.",
                },
            )
            types["optional"]["output_directory"] = (
                "STRING",
                {
                    "default": "",
                    "tooltip": "Absolute or relative output directory.",
                },
            )
            types["optional"]["create_path_if_missing"] = (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "Auto-create the output directory if missing.",
                },
            )
            return types

        def combine_video(self, *args, **kwargs):
            # Defensive normalization — the frontend occasionally sends
            # the literal string "undefined" for STRING widgets with
            # empty defaults. Coerce those to "" before any path
            # composition so we never create an `undefined/` subdir.
            filename_prefix = _normalize_text_input(
                kwargs.get("filename_prefix", "AnimateDiff")
            ) or "AnimateDiff"
            kwargs["filename_prefix"] = filename_prefix
            output_directory = _normalize_text_input(
                kwargs.pop("output_directory", "")
            )
            create_path_if_missing = kwargs.pop("create_path_if_missing", False)

            # Split-mode: directory + name combined into one effective
            # prefix that the existing isabs discrimination handles
            # uniformly. When output_directory is empty this is a no-op.
            effective_prefix = _compose_prefix(filename_prefix, output_directory)
            # Belt-and-suspenders: drop any sentinel components that
            # slipped past per-field normalization (e.g. a path that
            # already carried `\undefined\` as a literal segment).
            effective_prefix = _strip_sentinel_components(effective_prefix)
            if effective_prefix != filename_prefix:
                kwargs["filename_prefix"] = effective_prefix
                filename_prefix = effective_prefix

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
