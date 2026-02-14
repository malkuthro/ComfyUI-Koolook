#!/usr/bin/env python3
"""Simple report for third-party modification tracking.

Reads third_party/forks_manifest.yaml and prints a short summary for maintainers.
"""
from __future__ import annotations

from pathlib import Path
import sys

try:
    import yaml
except Exception:
    print("Missing dependency: pyyaml\nInstall with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "third_party" / "forks_manifest.yaml"

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return 1

    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    entries = data.get("entries", [])

    print("Third-party modification report")
    print("=" * 34)
    print(f"Manifest: {manifest_path}")
    print(f"Entries: {len(entries)}\n")

    for i, entry in enumerate(entries, 1):
        print(f"{i}. {entry.get('name', entry.get('id', 'unnamed'))}")
        print(f"   id: {entry.get('id', '-')}")
        print(f"   status: {entry.get('status', '-')}")
        print(f"   source: {entry.get('source_repo', '-')}")
        paths = entry.get("local_paths", []) or []
        if paths:
            print("   local paths:")
            for p in paths:
                print(f"     - {p}")
        else:
            print("   local paths: (none)")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
