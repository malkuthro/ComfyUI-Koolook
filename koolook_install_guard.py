"""Duplicate-install detection (#162).

The Comfy Registry / ComfyUI-Manager install path creates
``custom_nodes/koolook/`` (derived from ``[project].name`` in our
``pyproject.toml``), while a ``git clone`` checkout typically lands as
``custom_nodes/ComfyUI-Koolook/``. A user who has both — common when a
Manager install gets shadowed by a dev clone — boots ComfyUI with two
parallel Koolook plugins. Both register the same ``/koolook/presets/*``
server routes, the same Kforge Labs sidebar tab, and write to the same
``/userdata/koolook_workflows.json`` file. The late-loaded plugin
silently overwrites the early loader's state and the user's workflow
store corrupts invisibly.

This module is the detection + resolution layer. It is intentionally
free of ComfyUI / aiohttp imports so it can be unit-tested with just
the standard library — see ``tests/test_install_guard.py``.

Resolution strategy: pick the alphabetically-first folder name as the
winner. Deterministic across all installs (so both copies agree on the
outcome without needing to coordinate at runtime), independent of
ComfyUI's load order. The non-winning install registers nothing — no
nodes, no routes, no sidebar — and prints a critical message naming
both paths so the user can resolve the duplicate manually.
"""
from __future__ import annotations

from pathlib import Path


def detect_duplicate_koolook_installs(here: Path) -> list[Path]:
    """Return every sibling directory under ``here.parent`` that also
    contains a ``koolook_routes.py`` marker file (and is not ``here``).

    The marker file is unique to Koolook installs; a sibling Comfy
    custom node that happens to share a folder name won't match. The
    list is sorted by folder name (case-insensitive) so the order is
    stable for log output and for ``pick_winning_install``.
    """
    parent = here.parent
    siblings: list[Path] = []
    try:
        entries = list(parent.iterdir())
    except OSError:
        # ``custom_nodes/`` unreadable — would also break the rest of
        # ComfyUI. Fall through to "no siblings detected"; the duplicate
        # symptom this guard catches only manifests when sibling
        # iteration works in the first place.
        return siblings
    for entry in entries:
        if entry == here or not entry.is_dir():
            continue
        if (entry / "koolook_routes.py").is_file():
            siblings.append(entry)
    siblings.sort(key=lambda p: p.name.lower())
    return siblings


def read_pyproject_version(install_dir: Path) -> str:
    """Best-effort extract of the ``version = "..."`` line from a
    sibling's ``pyproject.toml``. Returns ``"?"`` when the file is
    missing or unparseable — the critical log is still useful with
    just the paths.

    Deliberately a manual scan instead of ``tomllib`` to avoid pulling
    a Python 3.11+ requirement into the install-time guard. The format
    is regular enough that a one-line ``startswith("version")`` parser
    is robust to the few real-world variations (single vs double quotes,
    spaces around ``=``).
    """
    pyproj = install_dir / "pyproject.toml"
    if not pyproj.is_file():
        return "?"
    try:
        for line in pyproj.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            # Match ``version = "..."`` and ``version="..."`` but not
            # ``versioning = "..."`` or other near-collisions.
            if not stripped.startswith("version"):
                continue
            after_keyword = stripped[len("version"):].lstrip()
            if not after_keyword.startswith("="):
                continue
            value = after_keyword[1:].strip()
            # Strip a trailing inline comment (``# ...``) before the
            # quote-stripping in case a pyproject.toml gets clever.
            if " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            return value.strip('"').strip("'")
    except OSError:
        pass
    return "?"


def pick_winning_install(here: Path, siblings: list[Path]) -> Path:
    """Alphabetical-by-folder-name resolution. Stable across both
    installs so neither needs to coordinate at runtime; the loser
    figures out it's the loser by comparing this result against its
    own ``__file__``."""
    return sorted([here, *siblings], key=lambda p: p.name.lower())[0]


def build_duplicate_report(here: Path, siblings: list[Path]) -> tuple[bool, str]:
    """Produce the critical-log message for a duplicate-install
    situation. Returns ``(is_winning, message)`` where ``is_winning``
    is ``True`` when ``here`` was the alphabetically-first folder.

    Pure function — no I/O. Caller decides where to surface the message
    (``print()`` in ``__init__.py``, a logger in a test, etc.).
    """
    winner = pick_winning_install(here, siblings)
    is_winning = winner == here
    here_version = read_pyproject_version(here)
    sibling_lines = "\n".join(
        f"    - {s} (version: {read_pyproject_version(s)})"
        for s in siblings
    )
    header = (
        "[Koolook] CRITICAL: duplicate ComfyUI-Koolook installations detected.\n"
        f"  This install: {here} (version: {here_version})\n"
        f"  Other install(s):\n{sibling_lines}\n"
        f"  Active install (alphabetical winner): {winner}\n"
        "  Both copies register the same /koolook/presets/* routes, the same\n"
        "  Kforge Labs sidebar tab, and write to the same\n"
        "  /userdata/koolook_workflows.json file. Running both silently\n"
        "  corrupts your workflow store on every restart. Remove one of\n"
        "  the directories above and restart ComfyUI."
    )
    if is_winning:
        return True, header
    return False, (
        header + "\n"
        f"[Koolook] this install ({here.name}) is the non-winning duplicate; "
        "skipping node + route registration. Only the winning install above "
        "will serve Kforge Labs this session."
    )
