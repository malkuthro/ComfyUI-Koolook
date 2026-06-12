#!/usr/bin/env python3
"""
Scoped variant of ``dev-sync`` for the LTX 2.3 audio-lipsync automation.

Copies only what that automation's iteration loop touches:

    - forks/whatdreamscost_koolook/    the Koolook fork of LTXDirector +
                                       its modified prompt_relay + the
                                       verbatim-vendored patches.py
    - __init__.py                      root loader (registers the fork's
                                       node mappings; needs to be re-synced
                                       when the fork's NODE_CLASS_MAPPINGS
                                       set changes)
    - web/whatdreamscost_koolook/      Director timeline-editor extension
    - web/koolook_draft_guard.js       global Comfy draft-quota guard
                                       (formerly embedded in the Director
                                       extension; shipped so a scoped sync
                                       never strands a guard-less install)

It also ships ``koolook_install_guard.py`` + ``koolook_versioning.py`` — the
loader gates ``__init__.py`` imports at load — so a scoped sync never leaves
the loader pointing at a module the target lacks (#198 / #183).

Everything else in the live install - ``forks/radiance_koolook/``, the
root ``k_*.py`` nodes, unrelated ``web/`` assets, ``video_formats/`` -
is left untouched.

USER-INITIATED ONLY. Same rule as plain ``dev-sync`` (see project
``CLAUDE.md`` -> ``dev-sync`` section). Never run automatically:

  * after a commit
  * after a PR merge or ``/ship-pr``
  * at session end / wrap-up
  * on hook completion or "task complete" cleanup
  * from any agent skill that doesn't explicitly require it

Run only on the explicit user trigger phrase ``dev-sync-audio`` (or
``copy audio fork``, ``sync audio``, etc.). The maintainer typically has
multiple parallel sessions across worktrees; an unsolicited sync from one
silently destroys what another is reviewing.

The target path comes from ``KOLOOK_COMFYUI_DEV_PATH`` in ``.env`` -
identical to plain ``dev-sync``. No new env var.

Usage:
    python scripts/sync_to_dev_audio.py
    python scripts/sync_to_dev_audio.py --dry-run
    python scripts/sync_to_dev_audio.py --scope "vstr=10 trial"

Exit codes mirror ``sync_to_dev.py`` exactly:
    0  success
    2  KOLOOK_COMFYUI_DEV_PATH unset or target missing without --init
    3  --init refused

After copying Python files, restart ComfyUI manually so custom nodes are
re-imported. This scoped sync only copies files.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Reuse the dev-sync infrastructure for the .env loader, target validation,
# and chat-report shape. This scoped wrapper differs only in which paths get
# copied, and it never tries to manage the running ComfyUI process.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import sync_to_dev as _dev  # noqa: E402


AUDIO_PATHS: tuple[str, ...] = (
    "__init__.py",
    # __init__.py imports these at load, before any node group, and a missing
    # gate is catastrophic (not per-group-guarded): the install guard's
    # absolute-import fallback raises uncaught, and the koolook_versioning
    # context probe mislabels its ImportError as a non-Comfy context and
    # registers nothing. Ship them with __init__.py so a scoped sync never
    # leaves the loader pointing at a module the target lacks (#198 / #183).
    "koolook_install_guard.py",
    "koolook_versioning.py",
    "forks/whatdreamscost_koolook",
    "web/whatdreamscost_koolook",
    # The Comfy draft-quota guard used to be embedded in the Director web
    # extension above; it is global now. Ship it with every scoped audio
    # sync so replacing web/whatdreamscost_koolook/ can never leave a dev
    # install without the guard.
    "web/koolook_draft_guard.js",
)

STALE_AUDIO_PATHS: tuple[str, ...] = (
    "web/whatdreamscost_koolook_v1_3_2",
)


def target_is_repo_root(target: Path) -> bool:
    """Return True when the configured dev target is this source repo."""
    try:
        return target.resolve() == _dev.REPO_ROOT.resolve()
    except OSError:
        return False


def remove_stale_paths(target: Path, *, dry_run: bool, verbose: bool) -> int:
    """Remove old audio-sync paths that were renamed.

    The v1.3.9 upgrade moved the Director web extension from a versioned
    folder to a stable one. Existing dev installs can still have the old
    folder on disk, causing ComfyUI to load two timeline extensions for
    legacy workflows. Keep this scoped to explicit known paths.
    """
    removed = 0
    for rel in STALE_AUDIO_PATHS:
        stale = target / rel
        if not stale.exists() and not stale.is_symlink():
            continue
        if dry_run:
            if verbose:
                print(f"would remove stale: {rel}")
            removed += 1
            continue
        if stale.is_symlink() or stale.is_file():
            stale.unlink()
        elif stale.is_dir():
            shutil.rmtree(stale)
        else:
            stale.unlink()
        if verbose:
            print(f"removed stale: {rel}")
        removed += 1
    return removed


def _find_dotenv() -> Path | None:
    direct = _dev.REPO_ROOT / ".env"
    if direct.exists():
        return direct
    git_marker = _dev.REPO_ROOT / ".git"
    if not git_marker.is_file():
        return None
    try:
        content = git_marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not content.startswith("gitdir:"):
        return None
    gitdir = Path(content.split(":", 1)[1].strip())
    if "worktrees" not in gitdir.parts:
        return None
    idx = gitdir.parts.index("worktrees")
    main_repo_root = Path(*gitdir.parts[:idx]).parent
    candidate = main_repo_root / ".env"
    return candidate if candidate.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without touching the target.",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help=(
            "Create the target directory if it doesn't exist. The parent "
            "(typically a ComfyUI custom_nodes/ folder) must already be "
            "in place. Use on first-run only."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print each file/dir as it's copied.",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="audio-lipsync fork edit",
        help=(
            "Short (<=10 word) description for the chat report's second line "
            "and the in-browser footer in `web/_dev_build.json`. Defaults to "
            "'audio-lipsync fork edit' so the build identifier always names "
            "this module even when the maintainer doesn't pass a scope."
        ),
    )
    args = parser.parse_args()

    env_path = _find_dotenv()
    if env_path is not None:
        _dev.load_dotenv(env_path)

    target_str = os.environ.get("KOLOOK_COMFYUI_DEV_PATH")
    if not target_str:
        print(
            "KOLOOK_COMFYUI_DEV_PATH not set. Add it to .env (see .env.example).",
            file=sys.stderr,
        )
        return 2

    target = Path(target_str).expanduser()
    if target_is_repo_root(target):
        print(
            "KOLOOK_COMFYUI_DEV_PATH points at this source repo; refusing dev-sync.",
            file=sys.stderr,
        )
        return 2

    err = _dev.ensure_target(target, init=args.init)
    if err is not None:
        return err

    n = _dev.sync(
        target,
        dry_run=args.dry_run,
        verbose=args.verbose,
        paths=AUDIO_PATHS,
    )
    removed = remove_stale_paths(target, dry_run=args.dry_run, verbose=args.verbose)
    verb = "would sync" if args.dry_run else "synced"
    print(_dev.build_line())
    stale_note = f"; removed {removed} stale" if removed else ""
    print(f"{verb} {n} entries{stale_note} -> {target}  (dev-sync-audio)")
    if not args.dry_run:
        _dev.write_build_info(target, args.scope)
    return 0


if __name__ == "__main__":
    sys.exit(main())
