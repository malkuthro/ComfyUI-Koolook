#!/usr/bin/env python3
"""
Copy runtime-relevant Koolook files into a live ComfyUI custom_nodes
folder so a fix can be tested without a tag-and-publish round-trip.

USER-INITIATED ONLY. This script overwrites a live ComfyUI install.
Agents must NEVER run it automatically - not after a commit, not after
a PR merge or ship-pr, not at session end, not from any "task complete"
cleanup. The maintainer typically has multiple parallel sessions across
worktrees, and an unsolicited sync from one silently destroys what
another is reviewing. See project CLAUDE.md `dev-sync` section for the
full policy. Run only on the explicit user trigger phrase.

The target path is read from the KOLOOK_COMFYUI_DEV_PATH environment
variable (loaded from `.env` at the repo root if present). The variable
is intentionally kept out of the committed tree - see `.env.example`.

`KOLOOK_COMFYUI_DEV_PATH` should point at the eventual Koolook
subdirectory inside `custom_nodes/`, NOT at the `custom_nodes/` parent.
Target ``custom_nodes/koolook/`` — that's where ComfyUI-Manager and the
Comfy Registry install (derived from ``[project].name`` in
``pyproject.toml``), so dev-sync overwrites the Manager install in place.
Targeting ``custom_nodes/ComfyUI-Koolook/`` instead spawns a parallel
install; ``__init__.py``'s duplicate-install guard logs a critical
message and disables the non-winning copy (issue #162).

Example layouts:
    macOS:   /Volumes/Data/ComfyUI/custom_nodes/koolook
    Windows: C:/ComfyUI_portable/ComfyUI/custom_nodes/koolook

Usage:
    python scripts/sync_to_dev.py            # copy files
    python scripts/sync_to_dev.py --dry-run  # show what would copy
    python scripts/sync_to_dev.py --init     # first-run: create the
                                             # target folder if missing
                                             # (parent custom_nodes/ must
                                             # already exist), then sync

After copying Python files, restart ComfyUI manually so custom-node modules
are re-imported. This script only copies files.

Exit codes:
    0  success
    2  KOLOOK_COMFYUI_DEV_PATH unset, parent missing, or target missing
       (without --init)
    3  --init refused: parent is not an existing directory or doesn't
       resemble a ComfyUI custom_nodes/ folder

This script never reaches outside the repo, never deletes anything in
the source, and only touches paths under the configured target. It does
overwrite files in the target - that's the point.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _get_short_sha() -> str | None:
    """Best-effort short commit SHA of the source tree being synced.
    Returns ``None`` if git isn't reachable or the call fails - the
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
    """Returns the basename of the source tree being synced - useful when
    the maintainer is running multiple parallel ComfyUI installs and
    needs to know which checkout fed the most recent sync.

    For a worktree at ``.../ComfyUI-Koolook/.claude/worktrees/foo`` this
    returns ``foo``; for the main repo at ``.../ComfyUI-Koolook`` it
    returns ``ComfyUI-Koolook``. Either is informative enough to
    disambiguate."""
    return REPO_ROOT.name


def build_line() -> str:
    """Composes the two-piece header line consumed by the chat-report
    convention defined in project CLAUDE.md:

        <short-sha> - <worktree-name>

    SHA falls back to ``unknown`` if git is unreachable (we always need
    SOMETHING in slot 1 - the line shape is part of the convention).
    Worktree name comes from ``REPO_ROOT.name`` and is always present.

    Public - consumed by scoped per-module wrappers like
    ``sync_to_dev_audio.py`` so every dev-sync variant emits the same
    chat-report header.
    """
    sha = _get_short_sha() or "unknown"
    return f"{sha} - {_get_worktree_name()}"


# Backwards-compatible alias (the function was private until the audio
# wrapper landed). Drop after the next release cycle once we're sure no
# downstream caller imports the underscore name.
_build_line = build_line


def write_build_info(target: Path, scope: str | None) -> None:
    """Drop a tiny JSON next to the sidebar JS so the in-browser footer
    can render `dev <sha> * <time>` (and an italic <scope> on a second
    line) - same identifier the chat report quotes, but visible in the
    running ComfyUI itself. Absent on registry installs (the file is
    only written by this dev script), so the footer stays empty there.

    Best-effort: missing git -> omit `commit` field; no `--scope` -> omit
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

# Files / dirs ComfyUI loads at runtime, plus the package metadata ComfyUI
# Manager reads when showing the installed custom-node version. Anything
# outside this list (CI, docs, .claude/, .github/, .cursor/, CHANGELOG, LICENSE,
# README, fork manifest YAML, etc.) does not affect what ComfyUI executes or
# how the dev install is identified, and is intentionally skipped.
RUNTIME_PATHS: tuple[str, ...] = (
    "pyproject.toml",
    "__init__.py",
    "config.json",
    "k_ai_pipeline.py",
    "k_clean_latent_slice.py",
    "k_easy_image_batch.py",
    "k_easy_pattern.py",
    "k_easy_resize.py",
    "k_easy_track.py",
    "k_easy_utility.py",
    "k_easy_wan22_prompt.py",
    "k_loop_status.py",
    "k_ltx_av_bind_schedule.py",
    "k_ltx_guide_reference_strength.py",
    "k_publish_contract.py",
    "k_video_combine.py",
    "k_video_load.py",
    "koolook_install_guard.py",
    "koolook_routes.py",
    "koolook_setup_runner.py",
    "koolook_setups.py",
    "koolook_versioning.py",
    "forks",
    "video_formats",
    "web",
)

# When copying directories, exclude these subpaths - they are dev-only
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


def find_dotenv() -> Path | None:
    """Locate the `.env` to load.

    Worktree root first, then the **main repo root** when running from a git
    worktree (where `.git` is a file pointing at
    `<main>/.git/worktrees/<name>`). The committed `.env` lives only in the
    main checkout, so without this fallback `dev-sync` is unusable from a
    worktree. Mirrors `scripts/make_card.py` and the documented behavior in
    the dev-sync docs.
    """
    direct = REPO_ROOT / ".env"
    if direct.exists():
        return direct
    git_marker = REPO_ROOT / ".git"
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


def _ignore(_dir: str, names: list[str]) -> list[str]:
    return [n for n in names if n in DIR_EXCLUDES]


def sync(
    target: Path,
    dry_run: bool,
    verbose: bool,
    paths: tuple[str, ...] = RUNTIME_PATHS,
) -> int:
    """Copy each entry in ``paths`` (relative to the repo root) to the
    matching subpath under ``target``. Directories are recursively
    copied with ``DIR_EXCLUDES`` filtered out and the previous dest
    subtree removed first; files are overwritten in place.

    The optional ``paths`` argument lets scoped wrappers (e.g.
    ``sync_to_dev_audio.py``) reuse this function with a smaller set -
    just the subtree their automation module touches. The default is
    the full ``RUNTIME_PATHS`` (every file ComfyUI loads at runtime).
    """
    copied = 0
    for rel in paths:
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
    directory named `ComfyUI`. A bit conservative - saves the user from
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
            f"check KOLOOK_COMFYUI_DEV_PATH - the *parent* (typically a "
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
            "Short (<=10 word) description of what this build is about - "
            "the same scope summary that goes in the chat report's second "
            "line. Persisted in `web/_dev_build.json` and rendered in the "
            "Kforge Labs sidebar footer (italic, second line, below the "
            "`dev <sha> * <time>` identifier) so the maintainer can "
            "correlate live ComfyUI state with chat history when juggling "
            "multiple parallel worktree sessions. Optional - when absent, "
            "the footer renders just identifier + timestamp."
        ),
    )
    args = parser.parse_args()

    env_path = find_dotenv()
    if env_path is not None:
        load_dotenv(env_path)

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
    # Two-line summary - see project CLAUDE.md `dev-sync` section for the
    # chat-report convention that consumes this output. Header first so
    # the maintainer's eye lands on the build identifier before the
    # mechanical sync details.
    print(build_line())
    print(f"{verb} {n} entries -> {target}")
    if not args.dry_run:
        write_build_info(target, args.scope)
    return 0


if __name__ == "__main__":
    sys.exit(main())
