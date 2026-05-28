#!/usr/bin/env python3
"""sync_investigation_patches.py — push modified upstream files from MAIN to a live install.

Mirrors the `scripts/sync_to_dev.py` pattern but for third-party node modifications
under `docs/investigations/<NN>_<topic>/patches/`. MAIN is the source of truth;
the live install is a destination only.

Usage:
    python scripts/sync_investigation_patches.py <investigation-folder> [--dry-run]

Example:
    python scripts/sync_investigation_patches.py 01_LTX23-audio-file-lipsync

The investigation folder must contain `upstream.yaml` declaring:
  - sync_target_env: name of the env var holding the absolute target path
  - files: list of filenames to copy from patches/ into the target

Errors cleanly when the env var is unset, the target path is missing,
or upstream.yaml is malformed.

Backup convention: <filename>.bak.<YYYYMMDD> created in the target dir
before the first overwrite of the day. Subsequent same-day syncs do not
re-create the backup (preserves the day's earliest pre-sync state).
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
from pathlib import Path

import json

REPO_ROOT = Path(__file__).resolve().parent.parent
INVESTIGATIONS_DIR = REPO_ROOT / "docs" / "investigations"


def _load_dotenv() -> None:
    """Load .env from REPO_ROOT, falling back to the main worktree's .env when
    REPO_ROOT is itself a git worktree (e.g. running from .claude/worktrees/<name>/).
    First-set-wins per env-var.
    """
    candidates = [REPO_ROOT / ".env"]
    try:
        common_dir = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if common_dir:
            main_root = Path(common_dir).resolve().parent
            if main_root.resolve() != REPO_ROOT.resolve():
                candidates.append(main_root / ".env")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    for env_path in candidates:
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _short_sha(path: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(path),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"


def _fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    _load_dotenv()

    parser = argparse.ArgumentParser(
        description="Sync modified upstream files from a Koolook investigation folder to a live ComfyUI install."
    )
    parser.add_argument(
        "folder",
        help="Investigation folder name under docs/investigations/ (e.g. 01_LTX23-audio-file-lipsync)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without copying or backing up")
    args = parser.parse_args()

    inv_dir = INVESTIGATIONS_DIR / args.folder
    if not inv_dir.is_dir():
        return _fail(f"Investigation folder not found: {inv_dir}")

    upstream_json = inv_dir / "upstream.json"
    if not upstream_json.is_file():
        return _fail(f"Missing upstream.json in {inv_dir}")

    try:
        spec = json.loads(upstream_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return _fail(f"upstream.json is not valid JSON: {e}")

    if not isinstance(spec, dict):
        return _fail("upstream.json must be a top-level object")

    env_var = spec.get("sync_target_env")
    if not env_var:
        return _fail("upstream.json is missing `sync_target_env`")

    target_root = os.environ.get(env_var)
    if not target_root:
        return _fail(f"${env_var} is unset. Add it to .env (see .env.example).")

    target_root_p = Path(target_root)
    if not target_root_p.is_dir():
        return _fail(f"Target directory does not exist: {target_root_p}")

    files = spec.get("files") or []
    if not files:
        return _fail("upstream.json has no `files` array")

    patches_dir = inv_dir / "patches"
    if not patches_dir.is_dir():
        return _fail(f"Missing patches/ directory in {inv_dir}")

    # Two-line header — first line is parsed by chat reports verbatim
    main_sha = _short_sha(REPO_ROOT)
    trigger = spec.get("trigger") or args.folder
    print(f"{main_sha} - {args.folder}")
    print(f"sync-investigation: {trigger}")

    today = dt.date.today().strftime("%Y%m%d")
    copied = 0
    skipped = 0

    for fname in files:
        src = patches_dir / fname
        dst = target_root_p / fname
        if not src.is_file():
            print(f"  SKIP {fname}: not in patches/")
            skipped += 1
            continue
        if not dst.is_file():
            print(f"  SKIP {fname}: target does not exist at destination")
            skipped += 1
            continue

        bak = dst.with_suffix(dst.suffix + f".bak.{today}")
        if args.dry_run:
            print(f"  DRY  {fname} -> {dst}")
            continue

        if not bak.exists():
            shutil.copy2(dst, bak)

        shutil.copy2(src, dst)
        print(f"  OK   {fname} -> {dst}")
        copied += 1

    if args.dry_run:
        print(f"(dry-run) would have synced {len(files)} files; skipped {skipped}")
    else:
        print(f"synced {copied} files; skipped {skipped}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
