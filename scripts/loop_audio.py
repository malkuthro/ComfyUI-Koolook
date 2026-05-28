#!/usr/bin/env python3
"""
``loop-audio`` — record one iteration of the LTX 2.3 audio-lipsync loop.

End-to-end automation for one render cycle: locate the workflow the
maintainer just saved inside ComfyUI, snapshot the workflow JSON +
relay-overrides + fork state + notes into a numbered ``runs/run-NNN_``
folder under this module, render the audio-lipsync card alongside, and
append a row to ``runs/log.md``. No manual steps.

Settings — *which* module, *which* workflow filename to look for,
*which* multiline titles to extract, *which* fork dir to pin — all live
in :file:`loop_audio.config.json` next to this script. Edit the config,
not the Python, when the convention shifts.

USER-INITIATED ONLY. Same rule as ``dev-sync`` and ``dev-sync-audio``
(see project ``CLAUDE.md``). Trigger phrase: ``loop-audio``.

Discovery chain:

  KOLOOK_COMFYUI_DEV_PATH ── walk up two levels ──► <ComfyUI>/
      └── <comfyui_workflows_subpath from config> ── glob workflow_pattern ──►
                  newest mtime (excluding skip_filename_substring) ──► workflow.json

Snapshot folder layout (always created):

  <module_path>/runs/run-NNN_<auto-label>/
    ├── workflow.json        copy of the workflow at submission
    ├── relay_overrides.txt  RELAY_OVERRIDES multiline body
    ├── patch_state.txt      MAIN sha + last dev-sync-audio + fork-dir status
    ├── notes.md             feedback + scores + mechanical interp
    └── card.png             when render_card=true in config

Plus one row appended to <module_path>/runs/log.md.

Usage:

  python scripts/loop_audio.py
  python scripts/loop_audio.py --dry-run
  python scripts/loop_audio.py --label <override>
  python scripts/loop_audio.py --no-log
  python scripts/loop_audio.py --no-card
  python scripts/loop_audio.py --workflow <explicit-path>
  python scripts/loop_audio.py --config <explicit-config-path>

Exit codes:

  0  snapshot landed (or would have, in --dry-run)
  2  KOLOOK_COMFYUI_DEV_PATH unset / ComfyUI workflows dir missing / config missing
  3  no workflow matching the configured pattern
  4  config JSON malformed
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_suffix(".config.json")


# --------------------------------------------------------------------------
# Config — read once into a small dataclass-ish dict. Underscore-prefixed
# keys are ignored (same convention as the LTXDirector relay_overrides
# widget — keeps inline documentation co-located with the values).
# --------------------------------------------------------------------------


REQUIRED_CONFIG_KEYS = (
    "job_name",
    "module_path",
    "comfyui_workflows_subpath",
    "workflow_pattern",
    "skip_filename_substring",
    "tracked_multilines",
    "fork_to_track",
    "render_card",
)


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        print(
            f"Config not found: {path}\n"
            f"Each loop script defaults to <script>.config.json beside it; "
            f"pass --config to point elsewhere.",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Malformed config {path}: {exc}", file=sys.stderr)
        sys.exit(4)
    if not isinstance(raw, dict):
        print(f"Config root must be an object: {path}", file=sys.stderr)
        sys.exit(4)
    cfg = {k: v for k, v in raw.items() if not k.startswith("_")}
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        print(
            f"Config {path} is missing required keys: {missing}",
            file=sys.stderr,
        )
        sys.exit(4)
    return cfg


# --------------------------------------------------------------------------
# .env loader (minimal — same shape as scripts/sync_to_dev.py's helper).
# --------------------------------------------------------------------------


def load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# --------------------------------------------------------------------------
# Workflow discovery — turn config + env into a concrete JSON path.
# --------------------------------------------------------------------------


def resolve_workflows_dir(cfg: dict[str, Any]) -> Path:
    """``KOLOOK_COMFYUI_WORKFLOWS_DIR`` wins when set (lets the maintainer
    point the loop at any workflow library). Otherwise derive
    ``<ComfyUI>/<comfyui_workflows_subpath from config>`` from
    ``KOLOOK_COMFYUI_DEV_PATH`` by walking up two levels from the
    ``custom_nodes/<koolook>`` leaf."""
    override = os.environ.get("KOLOOK_COMFYUI_WORKFLOWS_DIR")
    if override:
        p = Path(override).expanduser()
        if not p.is_dir():
            print(
                f"KOLOOK_COMFYUI_WORKFLOWS_DIR points at a non-dir: {p}",
                file=sys.stderr,
            )
            sys.exit(2)
        return p

    dev = os.environ.get("KOLOOK_COMFYUI_DEV_PATH")
    if not dev:
        print(
            "Neither KOLOOK_COMFYUI_WORKFLOWS_DIR nor "
            "KOLOOK_COMFYUI_DEV_PATH is set in .env. Set one (see .env.example).",
            file=sys.stderr,
        )
        sys.exit(2)
    leaf = Path(dev).expanduser()
    workflows = leaf.parent.parent / cfg["comfyui_workflows_subpath"]
    if not workflows.is_dir():
        print(
            f"Workflows dir not found: {workflows}\n"
            f"(derived from KOLOOK_COMFYUI_DEV_PATH={dev} + "
            f"config.comfyui_workflows_subpath={cfg['comfyui_workflows_subpath']})",
            file=sys.stderr,
        )
        sys.exit(2)
    return workflows


def find_workflow(workflows_dir: Path, cfg: dict[str, Any]) -> Path:
    skip = cfg["skip_filename_substring"].lower()
    candidates = [
        p for p in workflows_dir.glob(cfg["workflow_pattern"])
        if skip not in p.stem.lower()
    ]
    if not candidates:
        print(
            f"No workflow JSON matching {cfg['workflow_pattern']!r} in "
            f"{workflows_dir}",
            file=sys.stderr,
        )
        sys.exit(3)
    return max(candidates, key=lambda p: p.stat().st_mtime)


# --------------------------------------------------------------------------
# Workflow content extraction.
# --------------------------------------------------------------------------


def extract_multilines(
    nodes: list[dict], tracked_titles: list[str]
) -> dict[str, str]:
    out: dict[str, str] = {}
    needles = [t.lower() for t in tracked_titles]
    for n in nodes:
        if n.get("type") != "Text Multiline":
            continue
        title = (n.get("title") or "").strip().lower()
        body = (n.get("widgets_values") or [""])[0] or ""
        for needle in needles:
            if needle in title and needle not in out:
                out[needle] = body
                break
    return out


def extract_director_state(nodes: list[dict]) -> dict[str, Any]:
    for n in nodes:
        t = n.get("type", "")
        if "LTXDirector" not in t or "Guide" in t:
            continue
        wv = n.get("widgets_values") or []
        return {
            "variant": t,
            "is_koolook": "koolook" in t.lower(),
            "use_custom_audio": wv[8] if len(wv) > 8 else None,
            "frame_rate": wv[9] if len(wv) > 9 else None,
            "duration_seconds": wv[5] if len(wv) > 5 else None,
            "duration_frames": wv[1] if len(wv) > 1 else None,
            "custom_width": wv[11] if len(wv) > 11 else None,
            "custom_height": wv[12] if len(wv) > 12 else None,
            "epsilon": wv[6] if len(wv) > 6 else None,
        }
    return {"variant": "(no LTXDirector node found)"}


def extract_scheduler_chain(nodes: list[dict]) -> dict[str, Any]:
    schedulers, samplers, noises, cfgs = [], [], [], []
    for n in nodes:
        t = n.get("type", "")
        wv = n.get("widgets_values") or []
        if t == "BasicScheduler":
            schedulers.append(wv)
        elif t == "KSamplerSelect":
            samplers.append(wv[0] if wv else None)
        elif t == "RandomNoise":
            noises.append(wv)
        elif t == "CFGGuider":
            cfgs.append(wv[0] if wv else None)
    return {
        "schedulers": schedulers,
        "samplers": samplers,
        "noises": noises,
        "cfgs": cfgs,
    }


SCORE_PAT = re.compile(
    r"^\s*(?P<axis>motion|sync|sharp(?:ness)?)\s*[:=]?\s*"
    r"(?P<n>\d+)(?:\s*/\s*5)?\s*$",
    re.IGNORECASE,
)


def parse_feedback(body: str) -> tuple[dict[str, Optional[int]], list[str]]:
    scores: dict[str, Optional[int]] = {
        "motion": None, "sync": None, "sharp": None,
    }
    text_lines: list[str] = []
    for line in body.splitlines():
        m = SCORE_PAT.match(line)
        if m:
            axis = m.group("axis").lower()
            axis = "sharp" if axis.startswith("sharp") else axis
            scores[axis] = int(m.group("n"))
        elif line.strip():
            text_lines.append(line.strip())
    return scores, text_lines


# --------------------------------------------------------------------------
# Repo / fork state.
# --------------------------------------------------------------------------


def short_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=2,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def fork_dir_status(fork_rel: str) -> str:
    try:
        r = subprocess.run(
            ["git", "status", "--short", "--", fork_rel],
            cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode != 0:
            return f"(git status failed: {r.stderr.strip()[:80]})"
        return r.stdout.strip() or "clean (matches HEAD)"
    except (OSError, subprocess.TimeoutExpired):
        return "(git unreachable)"


def read_dev_build_json() -> dict[str, str]:
    dev = os.environ.get("KOLOOK_COMFYUI_DEV_PATH", "")
    if not dev:
        return {}
    p = Path(dev) / "web" / "_dev_build.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


# --------------------------------------------------------------------------
# Run folder naming.
# --------------------------------------------------------------------------


def next_run_number(runs_dir: Path) -> int:
    if not runs_dir.is_dir():
        return 1
    nums: list[int] = []
    for child in runs_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("run-"):
            continue
        m = re.match(r"run-(\d+)", child.name)
        if m:
            nums.append(int(m.group(1)))
    return max(nums) + 1 if nums else 1


_LABEL_BAD = re.compile(r"[^a-z0-9._-]+")


def autogen_label(
    multilines: dict[str, str],
    director: dict[str, Any],
    relay_overrides_raw: str,
) -> str:
    name = multilines.get("name", "").strip()
    slug = _LABEL_BAD.sub("-", name.lower()).strip("-") or "unnamed"
    director_tag = "koolook" if director.get("is_koolook") else "upstream"
    audio_tag = (
        "audio-on" if director.get("use_custom_audio") is True
        else "audio-off"
    )
    knob_tag = ""
    if relay_overrides_raw.strip():
        try:
            opts = json.loads(relay_overrides_raw)
            if isinstance(opts, dict) and "video_strength" in opts:
                knob_tag = f"_vstr{opts['video_strength']}"
        except (json.JSONDecodeError, TypeError):
            pass
    label = f"{slug}_{director_tag}_{audio_tag}{knob_tag}"
    if len(label) > 60:
        label = label[:60].rstrip("-_")
    return label


# --------------------------------------------------------------------------
# Snapshot artifact writers.
# --------------------------------------------------------------------------


def render_relay_overrides_txt(raw: str, director: dict[str, Any]) -> str:
    body = raw.strip() if raw and raw.strip() else "(empty — upstream defaults)"
    inert_note = ""
    if not director.get("is_koolook") and raw.strip():
        inert_note = (
            "\n\n# NOTE: this value is INERT for this render — the Director "
            "node in the workflow is upstream `LTXDirector`, which has no "
            "`relay_overrides` input. Swap to "
            "`LTXDirector__koolook_v1_3_2` to make this active.\n"
        )
    return f"{body}\n{inert_note}"


def render_patch_state(build: dict[str, str], fork_status: str) -> str:
    return (
        f"# Fork state at submission\n\n"
        f"MAIN SHA              : {short_sha()}\n"
        f"Last dev-sync-audio   : {build.get('commit', '(no _dev_build.json)')}"
        f"  ({build.get('synced_at', '?')})\n"
        f"Sync scope tag        : {build.get('scope', '(none)')}\n"
        f"Sync worktree         : {build.get('worktree', '?')}\n"
        f"Fork dir status       : {fork_status}\n"
    )


def render_notes_md(
    director: dict[str, Any],
    multilines: dict[str, str],
    scheduler: dict[str, Any],
    scores: dict[str, Optional[int]],
    feedback_lines: list[str],
) -> str:
    info_body = multilines.get("overlay - info", "(none)")
    info_indented = "\n".join("    " + line for line in info_body.splitlines())
    feedback_body = "\n".join(feedback_lines) if feedback_lines else "(none)"
    scores_line = " · ".join(
        f"{k}: {v if v is not None else '?'}/5"
        for k, v in scores.items()
    )
    director_kind = (
        "Koolook variant (`LTXDirector__koolook_v1_3_2`)"
        if director.get("is_koolook")
        else "**upstream** `LTXDirector` (Koolook variant NOT wired — "
        "relay_overrides + per-segment σ are INERT this render)"
    )
    audio_state = (
        "**ON** — external audio file driving lip-sync"
        if director.get("use_custom_audio") is True
        else "off — model-generated audio from `[Audio]:` prompt tag"
        if director.get("use_custom_audio") is False
        else "(could not read)"
    )
    return (
        f"# Run notes\n\n"
        f"## Maintainer feedback (OVERLAY - FEEDBACK, verbatim)\n\n"
        f"{feedback_body}\n\n"
        f"**Scores:** {scores_line}\n\n"
        f"## OVERLAY - INFO (verbatim)\n\n"
        f"{info_indented}\n\n"
        f"## Mechanical interp\n\n"
        f"- Director node: {director_kind}\n"
        f"- `use_custom_audio`: {audio_state}\n"
        f"- ε (epsilon): {director.get('epsilon')!r}\n"
        f"- Duration: {director.get('duration_frames')} frames "
        f"@ {director.get('frame_rate')} fps "
        f"({director.get('duration_seconds')} sec)\n"
        f"- Director custom WxH: "
        f"{director.get('custom_width')}x{director.get('custom_height')} "
        f"(may be overridden by wired inputs upstream)\n"
        f"- Scheduler chain (BasicScheduler widgets): "
        f"{scheduler['schedulers']}\n"
        f"- Sampler picks: {scheduler['samplers']}\n"
        f"- RandomNoise widgets: {scheduler['noises']}\n"
        f"- CFGGuider widgets: {scheduler['cfgs']}\n"
    )


def render_log_row(
    nnn: int,
    director: dict[str, Any],
    relay_overrides_raw: str,
    scheduler: dict[str, Any],
    scores: dict[str, Optional[int]],
    feedback_lines: list[str],
) -> str:
    director_cell = director.get("variant", "?")
    relay_cell = (
        f"`{relay_overrides_raw.strip()}`"
        if relay_overrides_raw.strip()
        else "(empty → upstream defaults)"
    )
    audio_cell = (
        "ON" if director.get("use_custom_audio") is True
        else "off" if director.get("use_custom_audio") is False
        else "?"
    )
    phase1 = phase2 = "?"
    schedulers = scheduler["schedulers"]
    if len(schedulers) >= 2:
        s1, s2 = schedulers[0], schedulers[1]
        if len(s2) > 2 and s2[2] == 1.0 and len(s1) > 2 and s1[2] < 1.0:
            s1, s2 = s2, s1
        phase1 = f"{s1[0]} {s1[1]}stp d={s1[2]}" if len(s1) >= 3 else str(s1)
        phase2 = f"{s2[0]} {s2[1]}stp d={s2[2]}" if len(s2) >= 3 else str(s2)
    elif schedulers:
        s1 = schedulers[0]
        phase1 = f"{s1[0]} {s1[1]}stp d={s1[2]}" if len(s1) >= 3 else str(s1)
    score_cell = (
        f"M{scores['motion'] or '?'}·S{scores['sync'] or '?'}·"
        f"Sh{scores['sharp'] or '?'}"
    )
    notes_cell = (
        " ".join(feedback_lines)[:120] if feedback_lines else "(none)"
    ).replace("|", "\\|")
    return (
        f"| {nnn:03d} | {date.today().isoformat()} | `{director_cell}` "
        f"| {relay_cell} | {audio_cell} | {phase1} | {phase2} "
        f"| {score_cell} | {notes_cell} |\n"
    )


# --------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------


def _build_state_for_card(
    nnn: int, label: str, wf_path: Path,
    multilines: dict[str, str],
    director: dict[str, Any], scheduler: dict[str, Any],
    scores: dict[str, Optional[int]], feedback_lines: list[str],
    fork_rel: str,
) -> dict[str, Any]:
    return {
        "run_number": nnn,
        "run_label": label,
        "date": date.today().isoformat(),
        "name": multilines.get("name", "(unnamed)"),
        "workflow_name": wf_path.name,
        "relay_overrides_raw": multilines.get("relay_overrides", ""),
        "director": director,
        "scheduler": scheduler,
        "scores": scores,
        "feedback_lines": feedback_lines,
        "info_body": multilines.get("overlay - info", ""),
        "build": read_dev_build_json(),
        "main_sha": short_sha(),
        "fork_dir_status": fork_dir_status(fork_rel),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                   help="Loop config JSON (default: alongside this script).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be captured. No files written.")
    p.add_argument("--label", default=None,
                   help="Override the auto-generated run label suffix.")
    p.add_argument("--no-log", action="store_true",
                   help="Skip the log.md row append.")
    p.add_argument("--no-card", action="store_true",
                   help="Skip the card render even if config.render_card is true.")
    p.add_argument("--workflow", type=Path, default=None,
                   help="Explicit workflow JSON path (skip auto-discover).")
    args = p.parse_args()

    cfg = load_config(args.config)
    load_dotenv(REPO_ROOT / ".env")

    module_dir = REPO_ROOT / cfg["module_path"]
    runs_dir = module_dir / "runs"
    log_path = runs_dir / "log.md"

    if args.workflow:
        wf_path = args.workflow.expanduser().resolve()
        if not wf_path.is_file():
            print(f"workflow not found: {wf_path}", file=sys.stderr)
            return 3
    else:
        wf_path = find_workflow(resolve_workflows_dir(cfg), cfg)

    with wf_path.open(encoding="utf-8") as f:
        wf = json.load(f)
    nodes = wf.get("nodes") or []

    multilines = extract_multilines(nodes, cfg["tracked_multilines"])
    director = extract_director_state(nodes)
    scheduler = extract_scheduler_chain(nodes)
    scores, feedback_lines = parse_feedback(
        multilines.get("overlay - feedback", "")
    )

    relay_overrides_raw = multilines.get("relay_overrides", "")
    label = args.label or autogen_label(
        multilines, director, relay_overrides_raw
    )
    nnn = next_run_number(runs_dir)
    run_dir = runs_dir / f"run-{nnn:03d}_{label}"

    # Two-line chat-report header (matches dev-sync convention).
    print(f"{short_sha()} - {REPO_ROOT.name}")
    print(
        f"loop-{cfg['job_name']} run-{nnn:03d}  workflow={wf_path.name}  "
        f"director={director.get('variant')}  "
        f"audio={'ON' if director.get('use_custom_audio') is True else 'off'}"
    )

    if args.dry_run:
        print(f"  (dry-run) would write: {run_dir.relative_to(REPO_ROOT)}")
        print(f"  (dry-run) would append row to: {log_path.relative_to(REPO_ROOT)}")
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(wf_path, run_dir / "workflow.json")
    (run_dir / "relay_overrides.txt").write_text(
        render_relay_overrides_txt(relay_overrides_raw, director),
        encoding="utf-8",
    )
    (run_dir / "patch_state.txt").write_text(
        render_patch_state(read_dev_build_json(), fork_dir_status(cfg["fork_to_track"])),
        encoding="utf-8",
    )
    (run_dir / "notes.md").write_text(
        render_notes_md(
            director, multilines, scheduler, scores, feedback_lines,
        ),
        encoding="utf-8",
    )

    card_status = "skipped"
    if cfg["render_card"] and not args.no_card:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from make_card_audio import render_audio_card  # type: ignore[import-not-found]

            state = _build_state_for_card(
                nnn, label, wf_path,
                multilines, director, scheduler,
                scores, feedback_lines,
                cfg["fork_to_track"],
            )
            render_audio_card(state, run_dir / "card.png")
            card_status = "rendered"
        except ImportError as exc:
            card_status = f"skipped ({exc})"

    if not args.no_log:
        row = render_log_row(
            nnn, director, relay_overrides_raw, scheduler, scores, feedback_lines,
        )
        with log_path.open("a", encoding="utf-8") as f:
            f.write(row)

    print(f"  wrote:  {run_dir.relative_to(REPO_ROOT)}")
    print(f"  card:   {card_status}")
    if not args.no_log:
        print(f"  logged: {log_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
