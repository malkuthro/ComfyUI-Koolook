#!/usr/bin/env python3
"""
Copy runtime-relevant Koolook files into a live ComfyUI custom_nodes
folder so a fix can be tested without a tag-and-publish round-trip.

USER-INITIATED ONLY. This script overwrites a live ComfyUI install.
Agents must NEVER run it automatically — not after a commit, not after
a PR merge or ship-pr, not at session end, not from any "task complete"
cleanup. The maintainer typically has multiple parallel sessions across
worktrees, and an unsolicited sync from one silently destroys what
another is reviewing. See project CLAUDE.md `dev-sync` section for the
full policy. Run only on the explicit user trigger phrase.

The target path is read from the KOLOOK_COMFYUI_DEV_PATH environment
variable (loaded from `.env` at the repo root if present). The variable
is intentionally kept out of the committed tree — see `.env.example`.

`KOLOOK_COMFYUI_DEV_PATH` should point at the eventual Koolook
subdirectory inside `custom_nodes/`, NOT at the `custom_nodes/` parent.
Example layouts:
    macOS:   /Volumes/Data/ComfyUI/custom_nodes/ComfyUI-Koolook
    Windows: C:/ComfyUI_portable/ComfyUI/custom_nodes/ComfyUI-Koolook

Usage:
    python scripts/sync_to_dev.py            # copy + auto-restart ComfyUI
    python scripts/sync_to_dev.py --dry-run  # show what would copy
    python scripts/sync_to_dev.py --no-restart # copy only; don't restart
    python scripts/sync_to_dev.py --init     # first-run: create the
                                             # target folder if missing
                                             # (parent custom_nodes/ must
                                             # already exist), then sync

After copying, the script POSTs to ComfyUI-Manager's reboot endpoint
(`http://127.0.0.1:8188/manager/reboot` by default) so the live server
re-execs and picks up the new Python files. This is on by default
because the only reason to invoke this script is to see new code —
file-only sync without a restart leaves Python `.py` changes invisible
(custom-node modules load once at server start). Use `--no-restart` to
opt out, e.g. when staging files for a session you don't want to disturb.
Restart failures (Manager not installed, ComfyUI not running, etc.)
print a diagnostic but do NOT fail the sync.

Exit codes:
    0  success (sync completed; restart is best-effort and won't change this)
    2  KOLOOK_COMFYUI_DEV_PATH unset, parent missing, or target missing
       (without --init)
    3  --init refused: parent is not an existing directory or doesn't
       resemble a ComfyUI custom_nodes/ folder

This script never reaches outside the repo, never deletes anything in
the source, and only touches paths under the configured target. It does
overwrite files in the target — that's the point.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

# Default ComfyUI-Manager reboot endpoint. ComfyUI-Manager registers this
# as a GET route; hitting it makes the server `os.execv` itself, so the
# same terminal/process picks the new node code up without a manual kill
# + relaunch cycle. Override via `--restart-url` or skip via
# `--no-restart` if your install runs on a different port / has no
# Manager / you want to stage files without disturbing the live session.
DEFAULT_RESTART_URL = "http://127.0.0.1:8188/manager/reboot"

REPO_ROOT = Path(__file__).resolve().parent.parent


def _get_short_sha() -> str | None:
    """Best-effort short commit SHA of the source tree being synced.
    Returns ``None`` if git isn't reachable or the call fails — the
    summary line just omits the SHA in that case.
    """
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=2,
        )
        if r.returncode == 0:
            sha = r.stdout.strip()
            if sha:
                return sha
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _get_worktree_name() -> str:
    """Returns the basename of the source tree being synced — useful when
    the maintainer is running multiple parallel ComfyUI installs and
    needs to know which checkout fed the most recent sync.

    For a worktree at ``.../ComfyUI-Koolook/.claude/worktrees/foo`` this
    returns ``foo``; for the main repo at ``.../ComfyUI-Koolook`` it
    returns ``ComfyUI-Koolook``. Either is informative enough to
    disambiguate."""
    return REPO_ROOT.name


def _build_line() -> str:
    """Composes the two-piece header line consumed by the chat-report
    convention defined in project CLAUDE.md:

        <short-sha> - <worktree-name>

    SHA falls back to ``unknown`` if git is unreachable (we always need
    SOMETHING in slot 1 — the line shape is part of the convention).
    Worktree name comes from ``REPO_ROOT.name`` and is always present."""
    sha = _get_short_sha() or "unknown"
    return f"{sha} - {_get_worktree_name()}"


def trigger_restart(url: str, timeout: float = 2.0) -> tuple[bool, str]:
    """Best-effort POST/GET to ComfyUI-Manager's reboot endpoint.

    Returns ``(success, human_message)``. Never raises — the sync
    itself succeeded by the time we call this, and a failed restart
    should never mask that with a non-zero exit. The maintainer can
    always restart manually if the auto-restart didn't take.

    Failure-mode mapping:
        - Connection refused          → ComfyUI not running; nothing to do.
        - HTTP 404                    → Manager not installed at this URL.
        - HTTP 2xx                    → restart triggered cleanly.
        - Connection reset / timeout  → the server died mid-response (the
          restart's `os.execv` raced our read); the request still landed,
          so we treat this as success.
    """
    # Restrict to http/https. `urlopen` would otherwise accept `file://`,
    # `ftp://`, etc. — a typo in `--restart-url` could surprise the
    # maintainer by opening a local file via the same code path that
    # normally talks to ComfyUI-Manager. Static analyzers (Bandit B310)
    # also flag the broader urlopen surface; this guard makes the
    # `# nosec` annotation below truthful rather than dismissive.
    scheme = urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        return False, (
            f"restart skipped: --restart-url must be http(s); got "
            f"scheme '{scheme}' in {url}"
        )

    try:
        # ComfyUI-Manager registers /manager/reboot as GET, but POST
        # also lands harmlessly — using GET keeps it simple.
        urllib.request.urlopen(url, timeout=timeout)  # nosec B310 - scheme validated above
        return True, "restart triggered"
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False, (
                f"restart skipped: ComfyUI-Manager not detected at {url} "
                f"(404). Install Manager or pass --no-restart."
            )
        return False, f"restart skipped: endpoint returned HTTP {e.code}"
    except urllib.error.URLError as e:
        reason = e.reason
        if isinstance(reason, ConnectionRefusedError):
            return False, f"restart skipped: ComfyUI not running at {url}"
        # Read timeout / connection reset usually means the server
        # received the request and rebooted before completing the
        # response — desired outcome.
        if isinstance(reason, (TimeoutError, ConnectionResetError)) or "timed out" in str(reason).lower():
            return True, "restart triggered (server stopped responding during reboot — normal)"
        return False, f"restart skipped: unreachable ({reason})"
    except (ConnectionResetError, OSError):
        # Same race as above: server began rebooting before we finished
        # reading. The request landed; that's what matters.
        return True, "restart triggered (connection reset during reboot)"


def write_build_info(target: Path, scope: str | None) -> None:
    """Drop a tiny JSON next to the sidebar JS so the in-browser footer
    can render `dev <sha> · <time>` (and an italic <scope> on a second
    line) — same identifier the chat report quotes, but visible in the
    running ComfyUI itself. Absent on registry installs (the file is
    only written by this dev script), so the footer stays empty there.

    Best-effort: missing git → omit `commit` field; no `--scope` → omit
    `scope`; the timestamp alone is still useful for the maintainer to
    eyeball "did my last sync land in this browser tab?\""""
    info: dict[str, str] = {
        "synced_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "worktree": _get_worktree_name(),
    }
    sha = _get_short_sha()
    if sha:
        info["commit"] = sha
    if scope:
        info["scope"] = scope
    out = target / "web" / "_dev_build.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

# Files / dirs ComfyUI actually loads at runtime. Anything outside this
# list (CI, docs, .claude/, .github/, .cursor/, CHANGELOG, LICENSE,
# pyproject.toml, README, fork manifest YAML, etc.) does not affect what
# ComfyUI executes and is intentionally skipped.
RUNTIME_PATHS: tuple[str, ...] = (
    "__init__.py",
    "config.json",
    "k_ai_pipeline.py",
    "k_easy_image_batch.py",
    "k_easy_pattern.py",
    "k_easy_resize.py",
    "k_easy_track.py",
    "k_easy_wan22_prompt.py",
    "k_video_combine.py",
    "koolook_routes.py",
    "forks",
    "video_formats",
    "web",
)

# When copying directories, exclude these subpaths — they are dev-only
# metadata that ComfyUI doesn't need and which can churn unnecessarily.
DIR_EXCLUDES: tuple[str, ...] = (
    "__pycache__",
    "UPSTREAM_PIN.yaml",
    "THIRD_PARTY.md",
    "forks_manifest.yaml",
    "README.md",
)


def load_dotenv(env_path: Path) -> None:
    """Minimal `.env` loader. No dependency on python-dotenv."""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        # Don't overwrite anything already in the environment.
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _ignore(_dir: str, names: list[str]) -> list[str]:
    return [n for n in names if n in DIR_EXCLUDES]


def sync(target: Path, dry_run: bool, verbose: bool) -> int:
    copied = 0
    for rel in RUNTIME_PATHS:
        src = REPO_ROOT / rel
        if not src.exists():
            continue
        dst = target / rel
        if dry_run:
            if verbose:
                print(f"would copy: {rel}")
            copied += 1
            continue
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=_ignore)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        if verbose:
            print(f"copied: {rel}")
        copied += 1
    return copied


def _looks_like_custom_nodes(parent: Path) -> bool:
    """Heuristic: parent is named `custom_nodes` OR sits inside a
    directory named `ComfyUI`. A bit conservative — saves the user from
    a `KOLOOK_COMFYUI_DEV_PATH` typo that would otherwise create a fresh
    `ComfyUI-Koolook/` somewhere unexpected on disk."""
    if parent.name == "custom_nodes":
        return True
    grandparent = parent.parent
    return grandparent.name.lower() == "comfyui" and parent.is_dir()


def ensure_target(target: Path, init: bool) -> int | None:
    """Validate the target. Returns an exit code (2 or 3) on error,
    or None on success (with target now guaranteed to exist as a dir)."""
    if target.exists():
        if not target.is_dir():
            print(f"target is not a directory: {target}", file=sys.stderr)
            return 2
        return None
    # Target is missing.
    if not init:
        print(
            f"target does not exist: {target}\n"
            f"first time on this machine? re-run with --init to create it.",
            file=sys.stderr,
        )
        return 2
    parent = target.parent
    if not parent.exists() or not parent.is_dir():
        print(
            f"--init refused: parent does not exist: {parent}\n"
            f"check KOLOOK_COMFYUI_DEV_PATH — the *parent* (typically a "
            f"ComfyUI custom_nodes/ folder) must already be in place.",
            file=sys.stderr,
        )
        return 3
    if not _looks_like_custom_nodes(parent):
        print(
            f"--init refused: parent doesn't look like a ComfyUI "
            f"custom_nodes/ folder: {parent}\n"
            f"expected the parent to be named 'custom_nodes' or to sit "
            f"inside a 'ComfyUI' directory. If this really is your "
            f"ComfyUI install, create the target manually with "
            f"`mkdir -p \"{target}\"` and re-run without --init.",
            file=sys.stderr,
        )
        return 3
    target.mkdir(parents=False, exist_ok=False)
    print(f"created target: {target}")
    return None


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
        help=(
            "Print each file/dir as it's copied. Default output is just "
            "the one-line build summary."
        ),
    )
    parser.add_argument(
        "--scope",
        type=str,
        default=None,
        help=(
            "Short (≤10 word) description of what this build is about — "
            "the same scope summary that goes in the chat report's second "
            "line. Persisted in `web/_dev_build.json` and rendered in the "
            "Kforge Labs sidebar footer (italic, second line, below the "
            "`dev <sha> · <time>` identifier) so the maintainer can "
            "correlate live ComfyUI state with chat history when juggling "
            "multiple parallel worktree sessions. Optional — when absent, "
            "the footer renders just identifier + timestamp."
        ),
    )
    parser.add_argument(
        "--no-restart",
        action="store_true",
        help=(
            "Skip the auto-restart of the live ComfyUI server. By default "
            "the script POSTs to ComfyUI-Manager's reboot endpoint after a "
            "successful sync so the new Python files are actually loaded "
            "(custom-node `.py` files load once at server start; without a "
            "restart the maintainer's changes won't be visible). Use this "
            "flag when you want to stage files without disturbing a live "
            "session (rare — `dev-sync` is normally invoked precisely "
            "because you want to see new code)."
        ),
    )
    parser.add_argument(
        "--restart-url",
        type=str,
        default=DEFAULT_RESTART_URL,
        help=(
            f"Override the restart endpoint. Default: {DEFAULT_RESTART_URL}. "
            f"Adjust if your ComfyUI listens on a non-standard host/port. "
            f"Ignored when --no-restart is set or --dry-run is used."
        ),
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    target_str = os.environ.get("KOLOOK_COMFYUI_DEV_PATH")
    if not target_str:
        print(
            "KOLOOK_COMFYUI_DEV_PATH not set. Add it to .env "
            "(see .env.example).",
            file=sys.stderr,
        )
        return 2

    target = Path(target_str).expanduser()
    err = ensure_target(target, init=args.init)
    if err is not None:
        return err

    n = sync(target, dry_run=args.dry_run, verbose=args.verbose)
    verb = "would sync" if args.dry_run else "synced"
    # Two-line summary — see project CLAUDE.md `dev-sync` section for the
    # chat-report convention that consumes this output. Header first so
    # the maintainer's eye lands on the build identifier before the
    # mechanical sync details.
    print(_build_line())
    print(f"{verb} {n} entries -> {target}")
    if not args.dry_run:
        write_build_info(target, args.scope)
        if not args.no_restart:
            ok, msg = trigger_restart(args.restart_url)
            print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
