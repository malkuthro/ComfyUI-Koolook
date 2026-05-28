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

import datetime
import json
import os
import re
from pathlib import Path
from typing import Optional, Tuple

try:
    from .koolook_versioning import resolve_version_token
except ImportError:  # pragma: no cover - standalone (pytest / tooling)
    from koolook_versioning import resolve_version_token

try:
    import folder_paths  # ComfyUI core; always present at runtime.
    _KOLOOK_VIDEO_FORMATS_DIR = Path(__file__).resolve().parent / "video_formats"
    if _KOLOOK_VIDEO_FORMATS_DIR.exists():
        _paths, _exts = folder_paths.folder_names_and_paths.get(
            "VHS_video_formats",
            ((), {".json"}),
        )
        _format_path = str(_KOLOOK_VIDEO_FORMATS_DIR)
        if _format_path not in _paths:
            folder_paths.folder_names_and_paths["VHS_video_formats"] = (
                tuple(_paths) + (_format_path,),
                set(_exts) | {".json"},
            )

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
    if os.path.isdir(normed):
        abs_dir = normed
        abs_base = os.path.basename(normed.rstrip("/\\"))
        if abs_base == "":
            raise ValueError(
                f"filename_prefix '{filename_prefix}' is an absolute directory "
                f"but has no usable folder name for the filename root."
            )
        return abs_dir, abs_base

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


def _normalize_bool_input(value, default: bool = False) -> bool:
    """Coerce ComfyUI widget values to a real bool.

    This is intentionally stricter than Python truthiness. A corrupted
    saved workflow can shift a format-widget value such as ``"hq"`` into
    the ``pingpong`` slot; ``bool("hq")`` would enable ping-pong and
    double the video. Unknown strings fall back to ``default``.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "on"):
        return True
    if s in ("false", "0", "no", "off", ""):
        return False
    return default


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


def _display_format_name(format_name: str) -> str:
    """Hide the JSON suffix VHS exposes for external Koolook presets."""
    if format_name.startswith("video/koolook-") and format_name.endswith(".json"):
        return format_name[:-5]
    return format_name


def _runtime_format_name(format_name: str) -> str:
    """Map cleaned Koolook display names back to VHS's external file names."""
    if format_name.startswith("video/koolook-") and not format_name.endswith(".json"):
        return format_name + ".json"
    return format_name


def _metadata_sidecar_path(output_files: list[str]) -> Optional[str]:
    """Pick the JSON sidecar path from VHS's output file list."""
    if not output_files:
        return None
    return os.path.splitext(output_files[0])[0] + ".json"


def _add_metadata_json_sidecar(
    result,
    metadata_payload: dict,
    save_metadata_json: bool,
    save_metadata_png: bool,
):
    """Write a workflow metadata JSON next to VHS's output and return result."""
    if not save_metadata_json or not isinstance(result, dict):
        return result
    try:
        output_files = result.get("result", ((None, []),))[0][1]
        if not isinstance(output_files, list):
            return result
        json_path = _metadata_sidecar_path(output_files)
        if not json_path:
            return result
        Path(json_path).write_text(
            json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if save_metadata_png:
            output_files.append(json_path)
        elif output_files and output_files[0].lower().endswith(".png"):
            output_files[0] = json_path
        else:
            output_files.insert(0, json_path)
        ui = result.get("ui", {})
        gifs = ui.get("gifs", [])
        if gifs and isinstance(gifs[0], dict) and gifs[0].get("workflow", "").endswith(".png"):
            gifs[0]["workflow"] = os.path.basename(json_path)
    except Exception as exc:
        print(f"[Easy_VideoCombine] metadata JSON sidecar skipped: {exc!r}")
    return result


def _remove_audio_suffix_from_result(result, keep_silent_intermediate: bool):
    """Rename VHS's final ``-audio`` mux output to the clean base name."""
    if keep_silent_intermediate or not isinstance(result, dict):
        return result
    try:
        output_files = result.get("result", ((None, []),))[0][1]
        if not isinstance(output_files, list) or not output_files:
            return result
        final_path = Path(output_files[-1])
        if final_path.stem.endswith("-audio"):
            clean_path = final_path.with_name(final_path.stem[:-6] + final_path.suffix)
            if clean_path.exists() and clean_path != final_path:
                clean_path.unlink()
            final_path.replace(clean_path)
            output_files[-1] = str(clean_path)
            ui = result.get("ui", {})
            gifs = ui.get("gifs", [])
            if gifs and isinstance(gifs[0], dict):
                gifs[0]["filename"] = clean_path.name
                gifs[0]["fullpath"] = str(clean_path)
        output_files[:] = [
            path for path in output_files
            if not isinstance(path, str) or os.path.exists(path)
        ]
        deduped = []
        for path in output_files:
            if path not in deduped:
                deduped.append(path)
        output_files[:] = deduped
    except Exception as exc:
        print(f"[Easy_VideoCombine] audio suffix cleanup skipped: {exc!r}")
    return result


# VHS appends a zero-padded counter to every output (e.g. ``clip_00001.mp4``).
# Strict-version mode strips it so the filename is exactly ``<root>_<token>``.
_COUNTER_RE = re.compile(r"_\d{5}$")


def _build_metadata_payload(prompt, extra_pnginfo) -> dict:
    """Assemble the workflow metadata written to the JSON sidecar.

    Always records ``CreationTime``. Records the ``prompt`` graph when present.
    Crucially, when ``extra_pnginfo`` (which carries the ``workflow`` graph in a
    canvas run) is absent — the headless / API ``/prompt`` path, or any run
    where the frontend doesn't forward it — fall back to recording the prompt
    as ``workflow`` so the sidecar is never reduced to a bare timestamp (the
    "empty JSON" the maintainer reported).
    """
    payload = {"CreationTime": datetime.datetime.now().isoformat(" ")[:19]}
    if prompt is not None:
        payload["prompt"] = prompt
    if extra_pnginfo:
        payload.update(extra_pnginfo)
    elif prompt is not None:
        payload["workflow"] = prompt
    return payload


def _append_version_to_prefix(prefix: str, token: str) -> str:
    """Suffix the path-prefix's filename root with ``_<token>``.

    ``effective_prefix`` always ends in a filename component at the call site
    (``_compose_prefix`` guarantees a name root, never a bare directory), so the
    token attaches directly to the end — e.g. ``E:/renders/clip`` ->
    ``E:/renders/clip_v001``.
    """
    if not token:
        return prefix
    return f"{prefix}_{token}"


def _coerce_version_input(raw):
    """Defend the ``version`` field against a stale boolean.

    Removing the old ``keep_silent_intermediate`` BOOLEAN widget shifts a saved
    workflow's stored value into the new same-position ``version`` STRING slot
    (widgets serialize by index — see docs/maintainers/node-versioning.md). A
    leftover bool would otherwise become the literal token ``"False"`` / ``"True"``
    and silently force strict mode (``clip_False.mp4``). Treat any boolean — or
    its string form — as "no version" so old workflows fall back to the legacy
    counter, exactly as before.
    """
    if isinstance(raw, bool):
        return ""
    if str(raw).strip().lower() in ("true", "false"):
        return ""
    return raw


def _write_sidecar(json_path: str, metadata_payload: dict) -> bool:
    """Write the metadata JSON sidecar to ``json_path``. True on success."""
    try:
        Path(json_path).write_text(
            json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return True
    except Exception as exc:
        print(f"[Easy_VideoCombine] metadata JSON sidecar skipped: {exc!r}")
        return False


def _strip_counter_path(path_str: str, enable_overwrite: bool) -> str:
    """Rename a rendered video to drop VHS's trailing ``_NNNNN`` counter.

    Collision: if the counter-free name already exists and ``enable_overwrite``
    is False, keep the counter (return the original path) so nothing is
    clobbered; if True, replace it. Returns the resulting path string.
    """
    p = Path(path_str)
    stripped = _COUNTER_RE.sub("", p.stem)
    if stripped == p.stem:
        return path_str
    candidate = p.with_name(stripped + p.suffix)
    if candidate == p:
        return path_str
    if candidate.exists():
        if not enable_overwrite:
            print(
                f"[Easy_VideoCombine] {candidate.name} exists; kept counter "
                f"suffix to avoid overwrite (enable_overwrite to replace)."
            )
            return path_str
        candidate.unlink()
        print(f"[Easy_VideoCombine] replaced existing {candidate.name}.")
    p.replace(candidate)
    return str(candidate)


def _finalize_strict_version_output(
    result,
    enable_overwrite: bool,
    metadata_payload: dict,
    save_metadata_json: bool,
    save_metadata_png: bool = False,
):
    """Strict-version finalize: strip VHS's ``_NNNNN`` counter from the final
    video so the name is exactly ``<root>_<token>.<ext>``, align a kept
    metadata PNG to the same stem, and write the JSON sidecar beside them.

    Collision is handled in :func:`_strip_counter_path` (lossless unless
    ``enable_overwrite``); the sidecar always matches the video's final name.
    """
    if not isinstance(result, dict):
        return result
    try:
        output_files = result.get("result", ((None, []),))[0][1]
        if not isinstance(output_files, list) or not output_files:
            return result

        new_video = _strip_counter_path(output_files[-1], enable_overwrite)
        if new_video != output_files[-1]:
            output_files[-1] = new_video
            ui = result.get("ui", {})
            gifs = ui.get("gifs", [])
            if gifs and isinstance(gifs[0], dict):
                gifs[0]["filename"] = os.path.basename(new_video)
                gifs[0]["fullpath"] = new_video
        video_stem_path = os.path.splitext(new_video)[0]

        # Keep the metadata PNG (when requested) under the video's stem so the
        # PNG/JSON/video trio share one name; otherwise it's dropped below.
        if (
            save_metadata_png
            and isinstance(output_files[0], str)
            and output_files[0].lower().endswith(".png")
            and output_files[0] != new_video
        ):
            png = Path(output_files[0])
            aligned = png.with_name(os.path.basename(video_stem_path) + png.suffix)
            if aligned != png:
                try:
                    if aligned.exists():
                        aligned.unlink()
                    png.replace(aligned)
                    output_files[0] = str(aligned)
                except OSError as exc:
                    print(f"[Easy_VideoCombine] metadata PNG rename skipped: {exc!r}")

        if save_metadata_json:
            json_path = video_stem_path + ".json"
            if _write_sidecar(json_path, metadata_payload):
                if save_metadata_png:
                    if json_path not in output_files:
                        output_files.append(json_path)
                elif (
                    isinstance(output_files[0], str)
                    and output_files[0].lower().endswith(".png")
                ):
                    output_files[0] = json_path
                elif json_path not in output_files:
                    output_files.insert(0, json_path)
                ui = result.get("ui", {})
                gifs = ui.get("gifs", [])
                if (
                    gifs
                    and isinstance(gifs[0], dict)
                    and str(gifs[0].get("workflow", "")).endswith(".png")
                ):
                    gifs[0]["workflow"] = os.path.basename(json_path)
    except Exception as exc:
        print(f"[Easy_VideoCombine] strict-version finalize skipped: {exc!r}")
    return result


def _finalize_output(
    result,
    version_token: str,
    enable_overwrite: bool,
    metadata_payload: dict,
    save_metadata_json: bool,
    save_metadata_png: bool,
):
    """Route to strict-version finalize (token set -> counter stripped) or the
    legacy sidecar attach (token empty -> VHS counter kept)."""
    if version_token:
        return _finalize_strict_version_output(
            result, enable_overwrite, metadata_payload, save_metadata_json, save_metadata_png
        )
    return _add_metadata_json_sidecar(
        result, metadata_payload, save_metadata_json, save_metadata_png
    )


if _VHS_AVAILABLE:
    class Easy_VideoCombine(_VHS_VideoCombine):
        """Video Combine variant with absolute-path output."""

        CATEGORY = "Koolook/Video"
        DESCRIPTION = "Video Combine variant with absolute-path output."

        @classmethod
        def INPUT_TYPES(cls):
            types = _VHS_VideoCombine.INPUT_TYPES()
            format_options, format_config = types["required"]["format"]
            cleaned_format_options = []
            cleaned_format_widgets = {}
            format_widgets = dict(format_config.get("formats", {}))
            for option in format_options:
                cleaned = _display_format_name(option)
                cleaned_format_options.append(cleaned)
                if option in format_widgets:
                    cleaned_format_widgets[cleaned] = format_widgets[option]
            for option, widgets in format_widgets.items():
                cleaned_format_widgets.setdefault(_display_format_name(option), widgets)
            format_options = cleaned_format_options
            format_config = {**format_config, "formats": cleaned_format_widgets}
            if "video/ProRes" in format_options:
                types["required"]["format"] = (
                    format_options,
                    {**format_config, "default": "video/ProRes"},
                )
            types["required"]["filename_prefix"] = (
                "STRING",
                {
                    "default": "AnimateDiff",
                    "tooltip": "Filename root. Counter and extension are appended.",
                },
            )
            # Upstream VHS uses save_output to choose ComfyUI output/ vs temp/.
            # Easy_VideoCombine is an explicit file-writer, and absolute-path
            # mode ignores that distinction, so hiding it avoids a misleading
            # toggle while preserving the persistent-output behavior.
            types["required"].pop("save_output", None)
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
            types["optional"]["save_metadata_png"] = (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "Save the first-frame PNG with embedded workflow metadata.",
                },
            )
            types["optional"]["save_metadata_json"] = (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": "Save workflow metadata as a JSON sidecar.",
                },
            )
            types["optional"]["version"] = (
                "STRING",
                {
                    "default": "",
                    "tooltip": (
                        "Strict version token (e.g. 'v001'), typically wired "
                        "from a global version node. When set, the output is "
                        "named exactly <prefix>_<version>.<ext> with NO automatic "
                        "_NNNNN counter. Leave empty to keep VHS's automatic "
                        "counter (the casual default)."
                    ),
                },
            )
            types["optional"]["enable_overwrite"] = (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": (
                        "Strict-version mode only. If the versioned filename "
                        "already exists: on = replace it; off = keep VHS's "
                        "counter for this render so nothing is overwritten."
                    ),
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
            kwargs["pingpong"] = _normalize_bool_input(
                kwargs.get("pingpong", False),
                default=False,
            )
            if "format" in kwargs:
                kwargs["format"] = _runtime_format_name(kwargs["format"])
            kwargs["save_output"] = True
            create_path_if_missing = _normalize_bool_input(
                kwargs.pop("create_path_if_missing", False),
                default=False,
            )
            save_metadata_png = _normalize_bool_input(
                kwargs.pop("save_metadata_png", False),
                default=False,
            )
            save_metadata_json = _normalize_bool_input(
                kwargs.pop("save_metadata_json", True),
                default=True,
            )
            keep_silent_intermediate = _normalize_bool_input(
                kwargs.pop("keep_silent_intermediate", False),
                default=False,
            )
            enable_overwrite = _normalize_bool_input(
                kwargs.pop("enable_overwrite", False),
                default=False,
            )
            # Strict versioning: a wired/typed token replaces VHS's automatic
            # _NNNNN counter with a deterministic <prefix>_<token>.<ext> name.
            # Empty -> the counter stays (the casual, non-professional default).
            version_token = resolve_version_token(
                _coerce_version_input(kwargs.pop("version", ""))
            )
            metadata_payload = _build_metadata_payload(
                kwargs.get("prompt"), kwargs.get("extra_pnginfo")
            )

            # Inject VHS's hidden extra_options flags so upstream handles
            # metadata PNG and silent-intermediate cleanup according to
            # Koolook's explicit toggles. Copy extra_pnginfo before mutating
            # so other nodes referencing the same dict aren't affected.
            extra_pnginfo = dict(kwargs.get("extra_pnginfo") or {})
            workflow = dict(extra_pnginfo.get("workflow") or {})
            extra = dict(workflow.get("extra") or {})
            extra["VHS_MetadataImage"] = bool(save_metadata_png)
            extra["VHS_KeepIntermediate"] = bool(keep_silent_intermediate)
            workflow["extra"] = extra
            extra_pnginfo["workflow"] = workflow
            kwargs["extra_pnginfo"] = extra_pnginfo

            # Split-mode: directory + name combined into one effective
            # prefix that the existing isabs discrimination handles
            # uniformly. When output_directory is empty this is a no-op.
            effective_prefix = _compose_prefix(filename_prefix, output_directory)
            # Belt-and-suspenders: drop any sentinel components that
            # slipped past per-field normalization (e.g. a path that
            # already carried `\undefined\` as a literal segment).
            effective_prefix = _strip_sentinel_components(effective_prefix)
            # Strict versioning bakes the token into the filename root; VHS
            # then appends its counter, which _finalize_output strips back off.
            if version_token:
                effective_prefix = _append_version_to_prefix(effective_prefix, version_token)
            if effective_prefix != filename_prefix:
                kwargs["filename_prefix"] = effective_prefix
                filename_prefix = effective_prefix

            target = _resolve_abs_target(filename_prefix, create_path_if_missing)
            if target is None:
                result = super().combine_video(*args, **kwargs)
                result = _remove_audio_suffix_from_result(
                    result,
                    keep_silent_intermediate,
                )
                return _finalize_output(
                    result,
                    version_token,
                    enable_overwrite,
                    metadata_payload,
                    save_metadata_json,
                    save_metadata_png,
                )

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
                result = super().combine_video(*args, **kwargs)
                result = _remove_audio_suffix_from_result(
                    result,
                    keep_silent_intermediate,
                )
                return _finalize_output(
                    result,
                    version_token,
                    enable_overwrite,
                    metadata_payload,
                    save_metadata_json,
                    save_metadata_png,
                )
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
