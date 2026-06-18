# SPDX-License-Identifier: GPL-3.0-or-later
#
# ComfyUI-Koolook — shared version-token helpers.
# Copyright (C) 2026 ComfyUI-Koolook contributors (kforgelabs).
#
# This file is part of ComfyUI-Koolook, licensed under GPL-3.0-or-later.
# See the LICENSE file at the repo root for the full text.
"""Single source of truth for Koolook's ``vNNN`` output-versioning convention.

Both ``EasyAIPipeline`` and ``Easy_VideoCombine`` stamp a version token into
their output paths/filenames. Centralising the rule here keeps the token
identical across every node so a single "global version" source (a wired
STRING from one interface node) propagates consistently.

Two entry points:

- :func:`normalize_version_token` — clean a wired/typed STRING into a safe
  single filename component (or ``""``).
- :func:`resolve_version_token` — apply the full precedence rule used by
  nodes that also carry a legacy INT ``version`` widget.
"""
from __future__ import annotations

import os
import re

# Same frontend-quirk sentinels the node modules already defend against: an
# untouched STRING widget can arrive at the backend as the literal string
# "undefined" / "null" / "None" instead of "".
_SENTINEL_STRINGS = ("undefined", "null", "none")

# Typed into a version field to request filesystem auto-detection of the next
# free version instead of a literal token. Nodes detect this and call
# ``next_version_token`` with their own output directory + name.
_AUTO_VERSION_TOKENS = ("auto", "next")


def is_auto_version(value) -> bool:
    """True when a version field requests auto-detection (``auto`` / ``next``)."""
    return normalize_version_token(value).lower() in _AUTO_VERSION_TOKENS


def normalize_version_token(value) -> str:
    """Clean a wired/typed version string into a safe single token.

    Returns ``""`` for empty / whitespace-only / sentinel input. Otherwise
    the value is used **verbatim** — ``v001``, ``final``, ``take_3`` all pass
    through unchanged — after defensive cleanup:

    - control characters (newline/CR/tab) stripped (Text Multiline can leak a
      stray paragraph break),
    - surrounding whitespace and a matched pair of surrounding quotes removed
      (Explorer "Copy as path" style pastes),
    - frontend sentinels (``undefined``/``null``/``none``) collapsed to ``""``,
    - path separators flattened to ``_`` — a version token is a single
      filename component and must never introduce a subfolder.
    """
    if value is None:
        return ""
    s = str(value).replace("\r", "").replace("\n", "").replace("\t", "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    if s.lower() in _SENTINEL_STRINGS:
        return ""
    # A version token names one path component, never a path of its own.
    return s.replace("/", "_").replace("\\", "_")


def resolve_version_token(version, disable_versioning: bool = False) -> str:
    """Resolve the version token to stamp into a path/filename.

    One field, whether typed into the widget or wired in as a STRING:

    - ``disable_versioning`` true   -> ``""`` (the master off-switch).
    - empty / sentinel              -> ``""``.
    - a bare integer (``"2"``)      -> ``"v002"`` (back-compat with the old
      INT widget, and a convenience when typing a number).
    - anything else (``"v001"``,
      ``"final"``, a wired token)   -> used **verbatim**.
    """
    if disable_versioning:
        return ""
    token = normalize_version_token(version)
    if not token:
        return ""
    if token.isdigit():
        return f"v{int(token):03d}"
    return token


def next_version_token(
    directory,
    name,
    version_prefix: str = "v",
    padding: int = 3,
    start: int = 1,
) -> str:
    """Return the next free ``<prefix>NNN`` token for ``name`` in ``directory``.

    Scans ``directory`` for entries (files *and* subfolders) of the form
    ``<name>_<prefix><digits>`` -- e.g. ``bearMask_v002.png``,
    ``bearMask_v002.0001.exr``, or a ``bearMask_v003/`` sequence folder -- and
    returns the highest detected version plus one, zero-padded to ``padding``.

    A missing/empty/unreadable directory (or no matching entries) yields
    ``start`` (default ``v001``). Matching is on the **exact** base name, so a
    different shot's versions never bump this one. An empty ``name`` matches
    bare ``<prefix>NNN`` tokens. Whatever follows the digits (extension, frame
    number, ``_suffix``) is ignored.
    """
    prefix = normalize_version_token(version_prefix) or "v"
    try:
        pad = max(1, int(padding))
    except (TypeError, ValueError):
        pad = 3
    try:
        start_int = int(start)
    except (TypeError, ValueError):
        start_int = 1

    base = normalize_version_token(name)
    lead = f"{re.escape(base)}_" if base else ""
    pattern = re.compile(rf"^{lead}{re.escape(prefix)}(\d+)")

    highest: int | None = None
    try:
        entries = os.listdir(directory) if directory else []
    except OSError:
        entries = []
    for entry in entries:
        match = pattern.match(entry)
        if match:
            value = int(match.group(1))
            highest = value if highest is None else max(highest, value)

    nxt = start_int if highest is None else highest + 1
    return f"{prefix}{nxt:0{pad}d}"
