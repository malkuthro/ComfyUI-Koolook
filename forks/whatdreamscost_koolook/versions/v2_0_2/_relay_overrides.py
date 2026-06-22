# SPDX-License-Identifier: GPL-3.0-or-later
"""
Pure-stdlib parser for the LTXDirector node's ``relay_overrides`` widget.

Lives in its own module (not in ``ltx_director.py``) for two reasons:

1. ``ltx_director.py`` imports ``comfy_api`` at module load, so it can't be
   imported outside a running ComfyUI process. This module imports only
   stdlib + logging, so unit tests can exercise the parser directly.
2. Keeping this Koolook addition out of ``ltx_director.py`` shrinks the
   diff against vendored upstream — easier to spot the two upstream
   modifications in ``ltx_director.py`` when comparing against the pinned
   upstream commit.

Parser contract — see :func:`parse_relay_overrides`.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

log = logging.getLogger(__name__)


# Allowlist of recognised override keys and their coercion type. Anything
# outside this set (other than ``_``-prefixed comment keys) is dropped
# with a warning so typos like ``viedo_strength`` don't silently render
# at upstream defaults — they'd look identical to a clean default render
# in the iteration log, polluting any A/B comparison.
RELAY_OVERRIDE_KEYS: dict[str, type] = {
    "video_strength": float,
    "video_window_scale": float,
    "audio_strength": float,
    "audio_window_scale": float,
    "audio_epsilon": float,
}


_PAIR_LINE_RE = re.compile(
    r'^\s*(?P<key>"?[A-Za-z_][A-Za-z0-9_]*"?)\s*(?::|=)\s*(?P<value>[^,#/]+?)\s*,?\s*$'
)


def _parse_loose_relay_overrides(raw: str) -> Optional[dict]:
    """Parse Comfy Text Multiline-friendly override snippets."""
    text = str(raw or "").strip()
    if not text:
        return None

    decoder = json.JSONDecoder()
    idx = 0
    objects: list[dict] = []
    try:
        while idx < len(text):
            while idx < len(text) and text[idx].isspace():
                idx += 1
            if idx >= len(text):
                break
            obj, next_idx = decoder.raw_decode(text, idx)
            if not isinstance(obj, dict):
                raise ValueError
            objects.append(obj)
            idx = next_idx
        if objects:
            merged: dict = {}
            for obj in objects:
                merged.update(obj)
            return merged
    except (json.JSONDecodeError, ValueError):
        pass

    loose: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//")):
            continue
        if stripped in ("{", "}"):
            continue
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict):
                loose.update(obj)
                continue
        if "{" in stripped or "}" in stripped:
            return None
        match = _PAIR_LINE_RE.match(stripped.strip("{} \t"))
        if not match:
            return None
        key = match.group("key").strip().strip('"')
        value = match.group("value").strip().strip('"').strip("'")
        loose[key] = value

    return loose or None


def parse_relay_overrides(s: str) -> Optional[dict]:
    """Parse the LTXDirector ``relay_overrides`` multiline-string input.

    The widget value is JSON. Empty / whitespace input means "use upstream
    defaults" — the only silent fallback. Anything else that doesn't
    decode to a JSON object of numeric knob values is treated as
    operator error and raises ``ValueError`` so the render fails fast
    instead of silently producing a default render the maintainer would
    later mistake for a real "the override had no effect" result.

    Args:
        s: The widget's text body.

    Returns:
        A dict of validated ``{knob_name: float}`` overrides, or ``None``
        when the input is empty/whitespace (meaning "use upstream
        defaults"). The dict only contains keys from
        :data:`RELAY_OVERRIDE_KEYS`; unknown keys are dropped after
        logging a warning.

    Raises:
        ValueError: when the input is non-empty but doesn't parse as a
            JSON object, or when a recognised knob's value can't be
            coerced to ``float``. The error message names the offending
            knob and shows a minimal correct example so the operator can
            fix the widget without consulting the upstream paper.
    """
    if not s or not s.strip():
        return None

    try:
        opts = json.loads(s)
    except (json.JSONDecodeError, TypeError) as exc:
        opts = _parse_loose_relay_overrides(s)
        if opts is None:
            raise ValueError(
                "relay_overrides input is not valid JSON or a supported "
                f"key/value block ({type(exc).__name__}: {exc}). Empty "
                "the field to use upstream defaults, paste a JSON object "
                'like {"video_strength": 10.0}, or use '
                "video_strength: 10.0."
            ) from exc

    if not isinstance(opts, dict):
        raise ValueError(
            "relay_overrides must be a JSON object, got "
            f"{type(opts).__name__}. Empty the field to use upstream "
            'defaults, or paste a JSON object like '
            '{"video_strength": 10.0}.'
        )

    cleaned: dict[str, float] = {}
    unknown: list[str] = []

    for key, value in opts.items():
        if key.startswith("_"):
            # `_`-prefixed keys are reserved as inline JSON comments —
            # carried in the widget value, ignored at parse time.
            continue
        if key not in RELAY_OVERRIDE_KEYS:
            unknown.append(key)
            continue

        coerce = RELAY_OVERRIDE_KEYS[key]
        if isinstance(value, bool):
            raise ValueError(
                f"relay_overrides[{key!r}] must be a number, got "
                f"{value!r} (bool). Example: "
                f'{{"{key}": 10.0}}'
            )
        try:
            cleaned[key] = coerce(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"relay_overrides[{key!r}] must be a number, got "
                f"{value!r} ({type(value).__name__}). Example: "
                f'{{"{key}": 10.0}}'
            ) from exc

    if unknown:
        log.warning(
            "[PromptRelay] relay_overrides has unknown keys (typo?): %s. "
            "Recognised keys: %s. Use `_`-prefixed keys for inline "
            "comments.",
            sorted(unknown),
            sorted(RELAY_OVERRIDE_KEYS),
        )

    if not cleaned:
        return None

    log.info("[PromptRelay] Loaded relay_overrides from node input: %s", cleaned)
    return cleaned


__all__ = ["RELAY_OVERRIDE_KEYS", "parse_relay_overrides"]
