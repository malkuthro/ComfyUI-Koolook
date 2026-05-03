#!/usr/bin/env python3
"""
Copy runtime-relevant Koolook files into a live ComfyUI custom_nodes
folder so a fix can be tested without a tag-and-publish round-trip.

The target path is read from the KOLOOK_COMFYUI_DEV_PATH environment
variable (loaded from `.env` at the repo root if present). The variable
is intentionally kept out of the committed tree — see `.env.example`.

Usage:
    python scripts/sync_to_dev.py            # copy all runtime files
    python scripts/sync_to_dev.py --dry-run  # show what would copy

Exit codes:
    0  success
    2  KOLOOK_COMFYUI_DEV_PATH unset or target missing

This script never reaches outside the repo, never deletes anything in
the source, and only touches paths under the configured target. It does
overwrite files in the target — that's the point.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

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
    "k_easy_version.py",
    "k_easy_wan22_prompt.py",
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


def sync(target: Path, dry_run: bool) -> int:
    copied = 0
    for rel in RUNTIME_PATHS:
        src = REPO_ROOT / rel
        if not src.exists():
            continue
        dst = target / rel
        if dry_run:
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
        print(f"copied: {rel}")
        copied += 1
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without touching the target.",
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
    if not target.exists():
        print(f"target does not exist: {target}", file=sys.stderr)
        return 2
    if not target.is_dir():
        print(f"target is not a directory: {target}", file=sys.stderr)
        return 2

    n = sync(target, dry_run=args.dry_run)
    verb = "would sync" if args.dry_run else "synced"
    print(f"\n{verb} {n} entries -> {target}")
    if not args.dry_run:
        print("restart ComfyUI to pick up changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
