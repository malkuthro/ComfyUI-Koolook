#!/usr/bin/env python3
"""
Copy runtime-relevant Koolook files into a live ComfyUI custom_nodes
folder so a fix can be tested without a tag-and-publish round-trip.

The target path is read from the KOLOOK_COMFYUI_DEV_PATH environment
variable (loaded from `.env` at the repo root if present). The variable
is intentionally kept out of the committed tree — see `.env.example`.

`KOLOOK_COMFYUI_DEV_PATH` should point at the eventual Koolook
subdirectory inside `custom_nodes/`, NOT at the `custom_nodes/` parent.
Example layouts:
    macOS:   /Volumes/Data/ComfyUI/custom_nodes/ComfyUI-Koolook
    Windows: C:/ComfyUI_portable/ComfyUI/custom_nodes/ComfyUI-Koolook

Usage:
    python scripts/sync_to_dev.py            # copy all runtime files
    python scripts/sync_to_dev.py --dry-run  # show what would copy
    python scripts/sync_to_dev.py --init     # first-run: create the
                                             # target folder if missing
                                             # (parent custom_nodes/ must
                                             # already exist), then sync

Exit codes:
    0  success
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
import os
import shutil
import subprocess
import sys
from pathlib import Path

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


def _get_version() -> str | None:
    """Read the project version from ``pyproject.toml`` without taking
    a dependency on ``tomllib`` / ``tomli`` — a flat string match for
    ``version = "x.y.z"`` is good enough for our flat header section
    and keeps the script importable on older Pythons."""
    pyproject = REPO_ROOT / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        for raw in pyproject.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("version"):
                _, _, val = line.partition("=")
                v = val.strip().strip('"').strip("'")
                if v:
                    return v
    except OSError:
        pass
    return None


def _build_id() -> str:
    """Composes ``<short-sha> v<version>`` for the post-sync summary line.
    Either component may be missing; the joined result skips None
    cleanly. Used by the chat-report convention defined in
    project CLAUDE.md so every dev-sync message carries a stable
    identifier the maintainer can match against running installs."""
    parts: list[str] = []
    sha = _get_short_sha()
    if sha:
        parts.append(sha)
    version = _get_version()
    if version:
        parts.append(f"v{version}")
    return " ".join(parts)

# Files / dirs ComfyUI actually loads at runtime. Anything outside this
# list (CI, docs, .claude/, .github/, .cursor/, CHANGELOG, LICENSE,
# pyproject.toml, README, fork manifest YAML, etc.) does not affect what
# ComfyUI executes and is intentionally skipped.
RUNTIME_PATHS: tuple[str, ...] = (
    "__init__.py",
    "config.json",
    "k_ai_pipeline.py",
    "k_easy_image_batch.py",
    "k_easy_resize.py",
    "k_easy_track.py",
    "k_easy_wan22_prompt.py",
    "koolook_routes.py",
    "forks",
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
    build = _build_id()
    build_part = f" @ {build}" if build else ""
    # Single-line build summary — see project CLAUDE.md `dev-sync` section
    # for the chat-report convention that consumes this output.
    print(f"{verb} {n} entries{build_part} -> {target}")
    if not args.dry_run:
        print("restart ComfyUI to pick up changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
