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

# Same frontend-quirk sentinels the node modules already defend against: an
# untouched STRING widget can arrive at the backend as the literal string
# "undefined" / "null" / "None" instead of "".
_SENTINEL_STRINGS = ("undefined", "null", "none")


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
