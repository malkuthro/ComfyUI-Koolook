"""Watch the make-card working folder; auto-render a card whenever a JSON
is added or saved.

Usage:
    python scripts/watch_cards.py              # uses .claude/skills/make-card/work-folder.txt
    python scripts/watch_cards.py <folder>     # override
    python scripts/watch_cards.py --interval 1 # poll every 1 s (default 2)

Run it once in a terminal at the start of a session and forget about it.
Each time ComfyUI (or you) drops a new workflow JSON into the folder, a
matching `<stem>_card.png` is rendered next to it within a couple of
seconds. Skips JSONs whose card is already up to date.

Stop with Ctrl+C.
"""
from __future__ import annotations
import os
import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAKE_CARD = ROOT / "scripts" / "make_card.py"
DEFAULT_CFG = ROOT / ".claude" / "skills" / "make-card" / "work-folder.txt"
PYTHON_EXE = sys.executable or "python"


def resolve_folder(argv: list[str]) -> Path:
    args = [a for a in argv if not a.startswith("-")]
    if args:
        return Path(args[0]).expanduser().resolve()
    if DEFAULT_CFG.exists():
        return Path(DEFAULT_CFG.read_text(encoding="utf-8").strip()).resolve()
    raise SystemExit(f"No folder argument and no config at {DEFAULT_CFG}")


def needs_render(json_path: Path) -> Path | None:
    """Return the target PNG path if it's missing or older than the JSON.
    Target lives in <working folder>/_AI/card.png — matches make_card.py."""
    ai_dir = json_path.parent / "_AI"
    png = ai_dir / "card.png"
    if not png.exists() or png.stat().st_mtime < json_path.stat().st_mtime:
        return png
    return None


def render(json_path: Path, png_path: Path) -> bool:
    print(f"[{time.strftime('%H:%M:%S')}] rendering {json_path.name} ...", flush=True)
    # Let make_card.py compute its own output path (writes to _AI/card.png).
    # Passing only the JSON keeps the watcher and the script in lockstep.
    r = subprocess.run(
        [PYTHON_EXE, str(MAKE_CARD), str(json_path)],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        print(f"  -> {png_path.relative_to(png_path.parent.parent)}", flush=True)
        return True
    print(f"  !! make_card.py failed (rc={r.returncode})\n{r.stderr}", flush=True)
    return False


def main():
    interval = 2.0
    if "--interval" in sys.argv:
        i = sys.argv.index("--interval")
        interval = float(sys.argv[i + 1])
    folder = resolve_folder(sys.argv[1:])
    if not folder.is_dir():
        raise SystemExit(f"Folder does not exist: {folder}")

    print(f"watching {folder}  (every {interval}s, Ctrl+C to stop)", flush=True)
    seen_mtimes: dict[Path, float] = {}
    try:
        while True:
            for json_path in folder.glob("*.json"):
                mtime = json_path.stat().st_mtime
                if seen_mtimes.get(json_path) == mtime:
                    continue
                target = needs_render(json_path)
                if target is not None:
                    render(json_path, target)
                seen_mtimes[json_path] = mtime
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nstopped.", flush=True)


if __name__ == "__main__":
    main()
