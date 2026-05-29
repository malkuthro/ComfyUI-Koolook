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

Everything else in the live install — ``forks/radiance_koolook/``, the
root ``k_*.py`` nodes, unrelated ``web/`` assets, ``video_formats/`` —
is left untouched.

USER-INITIATED ONLY. Same rule as plain ``dev-sync`` (see project
``CLAUDE.md`` → ``dev-sync`` section). Never run automatically:

  * after a commit
  * after a PR merge or ``/ship-pr``
  * at session end / wrap-up
  * on hook completion or "task complete" cleanup
  * from any agent skill that doesn't explicitly require it

Run only on the explicit user trigger phrase ``dev-sync-audio`` (or
``copy audio fork``, ``sync audio``, etc.). The maintainer typically has
multiple parallel sessions across worktrees; an unsolicited sync from one
silently destroys what another is reviewing.

The target path comes from ``KOLOOK_COMFYUI_DEV_PATH`` in ``.env`` —
identical to plain ``dev-sync``. No new env var.

Usage:
    python scripts/sync_to_dev_audio.py
    python scripts/sync_to_dev_audio.py --dry-run
    python scripts/sync_to_dev_audio.py --scope "vstr=10 trial"
    python scripts/sync_to_dev_audio.py --no-restart

Exit codes mirror ``sync_to_dev.py`` exactly:
    0  success
    2  KOLOOK_COMFYUI_DEV_PATH unset or target missing without --init
    3  --init refused

After copying, the script triggers a ComfyUI-Manager reboot so the new
Python files are actually loaded (custom-node modules load once at server
start; a file-only sync leaves ``.py`` changes invisible). Use
``--no-restart`` to opt out.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Reuse the dev-sync infrastructure verbatim — same ``.env`` loader,
# same target validation, same restart, same chat-report shape — so the
# only thing that differs between full and scoped syncs is *which paths
# get copied*. Per-module wrappers stay small and the safety guarantees
# (target validation, parent-directory sanity check, restart fallbacks)
# only need to be audited in one place.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import sync_to_dev as _dev  # noqa: E402


AUDIO_PATHS: tuple[str, ...] = (
    "__init__.py",
    "forks/whatdreamscost_koolook",
    "web/whatdreamscost_koolook",
)


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
            "Short (≤10 word) description for the chat report's second line "
            "and the in-browser footer in `web/_dev_build.json`. Defaults to "
            "'audio-lipsync fork edit' so the build identifier always names "
            "this module even when the maintainer doesn't pass a scope."
        ),
    )
    parser.add_argument(
        "--no-restart",
        action="store_true",
        help=(
            "Skip the auto-restart of the live ComfyUI server. Without a "
            "restart, the new fork code stays loaded as the previous version "
            "(custom-node .py files load once at server start)."
        ),
    )
    parser.add_argument(
        "--restart-url",
        type=str,
        default=_dev.DEFAULT_RESTART_URL,
        help=(
            f"Override the restart endpoint. Default: {_dev.DEFAULT_RESTART_URL}. "
            f"Ignored when --no-restart is set or --dry-run is used."
        ),
    )
    args = parser.parse_args()

    _dev.load_dotenv(_dev.REPO_ROOT / ".env")

    target_str = os.environ.get("KOLOOK_COMFYUI_DEV_PATH")
    if not target_str:
        print(
            "KOLOOK_COMFYUI_DEV_PATH not set. Add it to .env (see .env.example).",
            file=sys.stderr,
        )
        return 2

    target = Path(target_str).expanduser()
    err = _dev.ensure_target(target, init=args.init)
    if err is not None:
        return err

    n = _dev.sync(
        target,
        dry_run=args.dry_run,
        verbose=args.verbose,
        paths=AUDIO_PATHS,
    )
    verb = "would sync" if args.dry_run else "synced"
    print(_dev.build_line())
    print(f"{verb} {n} entries -> {target}  (dev-sync-audio)")
    if not args.dry_run:
        _dev.write_build_info(target, args.scope)
        if not args.no_restart:
            _ok, msg = _dev.trigger_restart(args.restart_url)
            print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
