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
    ├── runNNN_workflow.json copy of the workflow at submission
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
import hashlib
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
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
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


def find_dotenv() -> Optional[Path]:
    """Find .env starting at this worktree; fall back to the main repo
    root via git's common-dir when running from a worktree. Mirrors
    scripts/make_card.py so both scripts behave the same when invoked
    from a fresh worktree where the maintainer hasn't copied the .env
    over yet."""
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


DIRECTOR_TYPE = "LTXDirector__koolook"
LEGACY_DIRECTOR_TYPES = ("LTXDirector__koolook_v1_3_2",)
UPSTREAM_DIRECTOR_TYPE = "LTXDirector"
DIRECTOR_TYPES = (DIRECTOR_TYPE, *LEGACY_DIRECTOR_TYPES, UPSTREAM_DIRECTOR_TYPE)

# Koolook Director widget order — verified empirically against saved
# workflow JSON. The Comfy frontend serialises widgets in their original
# (upstream) order even after the Koolook fork reordered them in the
# schema; the only new widget — `relay_overrides` — is appended at the
# end. So the saved widgets_values matches the legacy LTXDirector order
# plus one extra slot.
#
# MIRROR: this dict is also documented in
# docs/automations/LTX-2.3/audio-lipsync/reading-graph.schema.yaml under
# `source_families.director_node.widget_order` (and indirectly under
# `state_dict.director_derived`). Update both when this dict changes,
# or the YAML drifts silently.
DIRECTOR_WIDX = {
    "global_prompt":    0,
    "duration_frames":  1,
    "duration_seconds": 2,
    "timeline_data":    3,
    "local_prompts":    4,
    "segment_lengths":  5,
    "epsilon":          6,
    "guide_strength":   7,
    "use_custom_audio": 8,
    "frame_rate":       9,
    "display_mode":     10,
    "custom_width":     11,
    "custom_height":    12,
    "resize_method":    13,
    "divisible_by":     14,
    "img_compression":  15,
    "relay_overrides":  16,
}


def _normalize_title(s: str) -> str:
    return " ".join(s.replace("_", " ").lower().split())


def _normalize_capture_key(s: str) -> str:
    return s.strip().lower()


def _normalize_alias_map(
    tracked_titles: list[str] | dict[str, Any],
) -> dict[str, list[str]]:
    if isinstance(tracked_titles, dict):
        out: dict[str, list[str]] = {}
        for key, aliases in tracked_titles.items():
            if isinstance(aliases, str):
                alias_list = [aliases]
            else:
                alias_list = list(aliases or [])
            out[_normalize_capture_key(str(key))] = [
                _normalize_title(str(alias))
                for alias in alias_list
                if str(alias).strip()
            ]
        return out
    return {
        _normalize_capture_key(str(needle)): [_normalize_title(str(needle))]
        for needle in tracked_titles
    }


def normalize_tracked_multilines(
    tracked_titles: list[str] | dict[str, Any],
) -> dict[str, list[str]]:
    """Return semantic capture keys mapped to title aliases.

    Older configs used a flat list where key == title substring. Newer
    configs use {"semantic_key": ["preferred title", "fallback title"]}.
    """
    return _normalize_alias_map(tracked_titles)


def extract_multilines(
    nodes: list[dict], tracked_titles: list[str] | dict[str, Any]
) -> dict[str, list[str]]:
    """Collect bodies by semantic key.

    The canvas legitimately ships duplicates for some keys — working-folder
    paths can have a project-mount and a local-mirror copy, for example — so
    we keep all hits and let the caller pick which to use. Alias lists are
    priority ordered; if two aliases for the same semantic key match
    different nodes, preferred aliases appear first in the captured list.
    """
    tracked = normalize_tracked_multilines(tracked_titles)
    captured: dict[str, list[tuple[int, int, str]]] = {
        key: [] for key in tracked
    }
    candidates: list[tuple[int, int, str, str]] = []
    for key, aliases in tracked.items():
        for priority, alias in enumerate(aliases):
            # Sort longest-first within the same priority so a future short
            # alias cannot shadow a more specific one.
            candidates.append((priority, -len(alias), key, alias))
    candidates.sort()

    for order, n in enumerate(nodes):
        if n.get("type") != "Text Multiline":
            continue
        title = _normalize_title(n.get("title") or "")
        body = (n.get("widgets_values") or [""])[0] or ""
        for priority, _, key, alias in candidates:
            if alias in title:
                captured[key].append((priority, order, body))
                break
    return {
        key: [body for _, _, body in sorted(values)]
        for key, values in captured.items()
    }


SETUP_VALUE_NODE_TYPES = {
    "Text Multiline",
    "PrimitiveBoolean",
    "PrimitiveFloat",
    "PrimitiveInt",
    "PrimitiveString",
}


def node_widget_value(n: dict) -> str:
    values = n.get("widgets_values")
    if values is None:
        return ""
    if isinstance(values, list):
        if not values:
            return ""
        value = values[0]
    else:
        value = values
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def extract_setup_variables(
    nodes: list[dict], tracked_titles: list[str] | dict[str, Any]
) -> dict[str, list[str]]:
    """Collect setup-card variable values from source nodes.

    This deliberately ignores GetNode/SetNode relay plumbing. For v02,
    version/run are PrimitiveInt nodes while the other setup values are
    Text Multiline nodes.
    """
    tracked = _normalize_alias_map(tracked_titles)
    captured: dict[str, list[tuple[int, int, str]]] = {
        key: [] for key in tracked
    }
    candidates: list[tuple[int, int, str, str]] = []
    for key, aliases in tracked.items():
        for priority, alias in enumerate(aliases):
            candidates.append((priority, -len(alias), key, alias))
    candidates.sort()

    for order, n in enumerate(nodes):
        if n.get("type") not in SETUP_VALUE_NODE_TYPES:
            continue
        title = _normalize_title(n.get("title") or "")
        value = node_widget_value(n)
        for priority, _, key, alias in candidates:
            if alias == title:
                captured[key].append((priority, order, value))
                break
    return {
        key: [value for _, _, value in sorted(values)]
        for key, values in captured.items()
    }


def first_multiline(ml: dict[str, list[str]], needle: str) -> str:
    return (ml.get(needle) or [""])[0]


def _parse_int(value: str) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def widget_value_by_name(node: dict, name: str, default: Any = "") -> Any:
    values = node.get("widgets_values")
    if isinstance(values, dict):
        value = values.get(name)
        return default if value is None else value
    if not isinstance(values, list):
        return default

    index = 0
    for inp in node.get("inputs") or []:
        if not isinstance(inp, dict) or "widget" not in inp:
            continue
        widget = inp.get("widget") or {}
        widget_name = widget.get("name") if isinstance(widget, dict) else None
        if widget_name == name and index < len(values):
            value = values[index]
            return default if value is None else value
        index += 1
    return default


def output_suffix_from_workflow(nodes: list[dict]) -> str:
    """Best-effort QuickTime suffix from the active VideoCombine format.

    The OUT settings group builds the name stem from the base name plus
    format suffix; the combine node then appends the version. If the
    format is unclear, h264 is the current setup default.
    """
    for n in nodes:
        if n.get("type") != "Easy_VideoCombine":
            continue
        fmt = str(widget_value_by_name(n, "format") or "")
        values = n.get("widgets_values")
        if not fmt and isinstance(values, list) and len(values) > 3:
            fmt = str(values[3] or "")
        fmt_l = fmt.lower()
        if "prores" in fmt_l:
            return "ProRes"
        if "h264" in fmt_l or "mp4" in fmt_l:
            return "h264"
    return "h264"


def expected_output_tracking(
    nodes: list[dict],
    multilines: dict[str, list[str]],
    setup_variables: dict[str, list[str]],
) -> dict[str, str]:
    """Expected external render target for the current setup.

    This deliberately uses the current setup variables rather than
    ``easy showAnything`` readouts, which may still contain the previous
    ComfyUI render's filenames.
    """
    folder = pick_existing_path(multilines.get("working_folder") or [])
    base_name = first_multiline(multilines, "name").strip()
    version = _parse_int(first_multiline(setup_variables, "version"))
    suffix = output_suffix_from_workflow(nodes)
    stem_parts = [part for part in (base_name, suffix) if part]
    stem = "_".join(stem_parts)
    version_tag = f"v{version:03d}" if version is not None else ""
    if stem and version_tag:
        stem = f"{stem}_{version_tag}"
    return {
        "folder": folder,
        "name": stem,
        "version_tag": version_tag,
        "format_suffix": suffix,
    }


def delivery_card_path(output_tracking: dict[str, str]) -> Optional[Path]:
    folder = (output_tracking.get("folder") or "").strip()
    name = (output_tracking.get("name") or "").strip()
    if not folder or not name:
        return None
    return Path(folder) / "cards" / f"{name}_card.png"


def copy_delivery_card(card_path: Path, output_tracking: dict[str, str]) -> str:
    delivery_path = delivery_card_path(output_tracking)
    if delivery_path is None:
        return "(not configured)"
    try:
        delivery_path.parent.mkdir(parents=True, exist_ok=True)
        if delivery_path.exists():
            return f"exists (left in place: {delivery_path})"
        shutil.copy2(card_path, delivery_path)
    except OSError as exc:
        return f"failed ({exc})"
    return str(delivery_path)


def scrub_path_for_metadata(value: str) -> str:
    """Store a portable fingerprint instead of full workstation paths."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    if re.match(r"^[A-Za-z]:[/\\]", raw) or raw.startswith(("/", "\\\\")):
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        tail = Path(raw.replace("\\", "/")).name or "(folder)"
        return f"{tail} [path-sha256:{digest}]"
    return raw


def pick_existing_path(candidates: list[str]) -> str:
    """Pick the first candidate that resolves to a real directory on
    this machine. Falls back to the first non-empty candidate so the
    path still appears on the card even if its drive isn't mounted
    here (worktree-on-N, render-target-on-W, etc.)."""
    for c in candidates:
        cleaned = c.strip().strip('"\'').rstrip("/\\")
        if cleaned and Path(cleaned).is_dir():
            return cleaned
    for c in candidates:
        cleaned = c.strip().strip('"\'').rstrip("/\\")
        if cleaned:
            return cleaned
    return ""


def wrap_path(path: str, max_chars: int = 40) -> list[str]:
    """Wrap a filesystem path onto multiple lines, breaking only on
    `/` or `\\` separators so directory names stay whole. Used by the
    card renderer so long mount paths don't bleed past the card edge."""
    if not path:
        return [""]
    raw = path.replace("/", "\\")
    parts = [p for p in raw.split("\\") if p]
    if not parts:
        return [path]
    lines: list[str] = []
    cur = ""
    for i, p in enumerate(parts):
        token = p + ("\\" if i < len(parts) - 1 else "")
        if cur and len(cur) + len(token) > max_chars:
            lines.append(cur)
            cur = token
        else:
            cur = cur + token
    if cur:
        lines.append(cur)
    return lines


def extract_director(nodes: list[dict]) -> Optional[dict]:
    """Return the active Director node dict (or None).

    Prefer the Koolook Director when both variants are on the canvas, but
    accept upstream ``LTXDirector`` for A/B comparison runs. The full node
    is returned so callers can read both widgets_values AND inputs[] socket
    wiring; the audio_vae link is what decides whether audio gets generated
    at all (see derive_audio_state).
    """
    for dtype in DIRECTOR_TYPES:
        for n in nodes:
            if n.get("type") == dtype:
                return n
    return None


def director_type(node: Optional[dict]) -> str:
    return node.get("type") if node else "(missing)"


def director_flavor(node: Optional[dict]) -> str:
    dtype = director_type(node)
    if dtype in (DIRECTOR_TYPE, *LEGACY_DIRECTOR_TYPES):
        return "Koolook v1.3.9"
    if dtype == UPSTREAM_DIRECTOR_TYPE:
        return "Original upstream"
    return "(missing)"


def is_koolook_director(node: Optional[dict]) -> bool:
    return director_type(node) in (DIRECTOR_TYPE, *LEGACY_DIRECTOR_TYPES)


def director_widget(node: Optional[dict], key: str) -> Any:
    if not node:
        return None
    named = widget_value_by_name(node, key, default=None)
    if named is not None:
        return named
    wv = node.get("widgets_values") or []
    idx = DIRECTOR_WIDX.get(key)
    if idx is None or idx >= len(wv):
        return None
    return wv[idx]


def is_input_wired(node: Optional[dict], input_name: str) -> Optional[bool]:
    """True iff a named input socket on the Director has a non-null
    link. None when the node or input isn't found."""
    if not node:
        return None
    for inp in node.get("inputs") or []:
        if inp.get("name") == input_name:
            return inp.get("link") is not None
    return None


def _coerce_segment_numeric(seg: dict) -> dict:
    """Coerce a timeline segment's numeric fields (start, length,
    trimStart) to int. The upstream Director itself does this
    defensively at ltx_director.py:232-234 — the saved JSON sometimes
    contains string values for these fields, and downstream
    arithmetic (`start / fps`, `<` comparisons) breaks silently
    without coercion. Failures collapse to 0 so a malformed segment
    can't crash the loop."""
    out = dict(seg)
    for k in ("start", "length", "trimStart"):
        if k in out:
            try:
                out[k] = int(float(out[k]))
            except (TypeError, ValueError):
                print(
                    f"WARNING: timeline segment {k}={out[k]!r} is not numeric; "
                    "using 0 for loop/card extraction.",
                    file=sys.stderr,
                )
                out[k] = 0
    return out


def parse_timeline(director_node: Optional[dict]) -> dict[str, list]:
    """Parse the timeline_data JSON widget into a dict of segments +
    audioSegments. Both default to empty lists on any error so the
    rest of the pipeline can iterate without further guards.

    Per-segment numeric fields are coerced via _coerce_segment_numeric
    so downstream arithmetic stays type-safe regardless of what shape
    the Comfy frontend saved into the JSON."""
    raw = director_widget(director_node, "timeline_data") or ""
    if not raw:
        return {"segments": [], "audioSegments": []}
    try:
        tl = json.loads(raw)
    except json.JSONDecodeError:
        return {"segments": [], "audioSegments": []}
    if not isinstance(tl, dict):
        return {"segments": [], "audioSegments": []}
    return {
        "segments": [
            _coerce_segment_numeric(s)
            for s in (tl.get("segments") or [])
            if isinstance(s, dict)
        ],
        "audioSegments": [
            _coerce_segment_numeric(s)
            for s in (tl.get("audioSegments") or [])
            if isinstance(s, dict)
        ],
    }


def derive_audio_state(
    director_node: Optional[dict], timeline: dict[str, list]
) -> str:
    """Reduce director presence plus three structural audio signals —
    audio_vae wiring, use_custom_audio toggle, audioSegments count — to one label that
    mirrors how the Director would actually behave at runtime
    (see forks/.../ltx_director.py: audio_vae None gates everything;
    then use_custom_audio chooses between the encoded path and the
    empty/model-gen path).

    Five distinct outcomes — "(no director)" is split off so a
    missing-director workflow doesn't get the same label as a
    director-present-but-VAE-unwired one, which silently collapsed
    pre-fix because is_input_wired(None, …) also returns None."""
    if director_node is None:
        return "(no director)"
    vae_wired = is_input_wired(director_node, "audio_vae")
    use_custom = director_widget(director_node, "use_custom_audio")
    audio_segs = timeline.get("audioSegments") or []
    # vae_wired can be None (input socket missing from this node's
    # schema, e.g. an older workflow before audio_vae existed) OR
    # False (socket present but unwired). Both functionally mean
    # "no audio latent produced", so collapse them here — but only
    # AFTER the director-missing case has been split off above.
    if not vae_wired:
        return "off (no VAE)"
    if use_custom is True:
        return "custom" if audio_segs else "custom (empty)"
    return "model-gen"


def video_segment_has_audio(video_seg: dict, audio_segs: list[dict]) -> bool:
    """Does an audioSegments[] entry overlap this video segment's time
    range? Matches how _build_combined_audio aligns the audio waveform
    onto the global timeline."""
    v_start = video_seg.get("start", 0)
    v_end = v_start + video_seg.get("length", 0)
    for a in audio_segs:
        a_start = a.get("start", 0)
        a_end = a_start + a.get("length", 0)
        if a_start < v_end and a_end > v_start:
            return True
    return False


def segment_prompt_mode(segments: list[dict]) -> str:
    """Classify whether timeline segments share one prompt or vary.

    This is a structural prompt check only. It normalizes whitespace but
    does not inspect prompt meaning, so it stays inside the card's source
    rule: Director timeline JSON in, small capture label out.
    """
    if not segments:
        return "none"
    prompts = [
        " ".join(str(seg.get("prompt") or "").split())
        for seg in segments
    ]
    if any(not prompt for prompt in prompts):
        return "missing"
    if len(prompts) == 1:
        return "single"
    return "same" if len(set(prompts)) == 1 else "per-segment"


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
        return json.loads(p.read_text(encoding="utf-8-sig"))
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
    log_path = runs_dir / "log.md"
    if log_path.is_file():
        try:
            for line in log_path.read_text(encoding="utf-8").splitlines():
                m = re.match(r"\|\s*(\d{3,})\s*\|", line)
                if m:
                    nums.append(int(m.group(1)))
        except OSError:
            pass
    return max(nums) + 1 if nums else 1


_LABEL_BAD = re.compile(r"[^a-z0-9._-]+")


def autogen_label(
    name: str,
    director_node: Optional[dict],
    relay_overrides_raw: str,
) -> str:
    slug = _LABEL_BAD.sub("-", name.lower()).strip("-") or "unnamed"
    dtype = director_type(director_node)
    if dtype == UPSTREAM_DIRECTOR_TYPE:
        director_tag = "upstream"
    elif is_koolook_director(director_node):
        director_tag = "koolook"
    else:
        director_tag = "missing"
    use_custom = director_widget(director_node, "use_custom_audio")
    audio_tag = "audio-on" if use_custom is True else "audio-off"
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


def render_relay_overrides_txt(
    raw: str, director_node: Optional[dict]
) -> str:
    body = raw.strip() if raw and raw.strip() else "(empty — upstream defaults)"
    missing_note = ""
    if director_node is None:
        missing_note = (
            "\n\n# WARNING: no supported LTXDirector node found in "
            "the workflow — the relay_overrides value above isn't being "
            "consumed by anything. Add the Koolook Director to make it active.\n"
        )
    elif not is_koolook_director(director_node):
        missing_note = (
            "\n\n# NOTE: this value is INERT for this render — the Director "
            f"node in the workflow is upstream `{director_type(director_node)}`, "
            "which has no `relay_overrides` input. Swap to "
            "`LTXDirector__koolook` to make this active.\n"
        )
    return f"{body}\n{missing_note}"


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


def card_metadata(
    nnn: int,
    label: str,
    wf_path: Path,
    multilines: dict[str, list[str]],
    setup_variables: dict[str, list[str]],
    output_tracking: dict[str, str],
    director_node: Optional[dict],
    timeline: dict[str, list],
    audio_src: str,
    build: dict[str, str],
    fork_status: str,
) -> dict[str, Any]:
    segments = timeline.get("segments") or []
    audio_segments = timeline.get("audioSegments") or []
    setup_input_path = first_multiline(setup_variables, "input_path_exr")
    output_folder = output_tracking.get("folder", "")
    return {
        "schema": "koolook.audio_loop.card_metadata.v1",
        "run": {
            "capture_number": f"{nnn:03d}",
            "label": label,
            "date": date.today().isoformat(),
            "workflow": wf_path.name,
            "setup_name": wf_path.stem,
        },
        "setup": {
            "base_name": first_multiline(multilines, "name").strip(),
            "working_folder": scrub_path_for_metadata(output_folder),
            "input_path_exr": scrub_path_for_metadata(setup_input_path),
            "global_version": first_multiline(setup_variables, "version"),
            "global_run_offset": first_multiline(setup_variables, "run_offset"),
            "relay_overrides": first_multiline(multilines, "relay_overrides").strip(),
        },
        "output": {
            "folder": scrub_path_for_metadata(output_folder),
            "name": output_tracking.get("name", ""),
            "version_tag": output_tracking.get("version_tag", ""),
            "format_suffix": output_tracking.get("format_suffix", ""),
        },
        "director": {
            "type": director_type(director_node),
            "flavor": director_flavor(director_node),
            "audio_src": audio_src,
            "epsilon": director_widget(director_node, "epsilon"),
            "duration_frames": director_widget(director_node, "duration_frames"),
            "duration_seconds": director_widget(director_node, "duration_seconds"),
            "frame_rate": director_widget(director_node, "frame_rate"),
            "segment_prompt_mode": segment_prompt_mode(segments),
            "video_segments": len(segments),
            "audio_segments": len(audio_segments),
        },
        "repo": {
            "main_sha": short_sha(),
            "last_dev_sync_audio": build.get("commit", ""),
            "last_dev_sync_at": build.get("synced_at", ""),
            "sync_scope_tag": build.get("scope", ""),
            "sync_worktree": build.get("worktree", ""),
            "fork_dir_status": fork_status,
        },
    }


def _notes_value(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return "`(missing)`"
    if "\n" not in cleaned:
        return f"`{cleaned}`"
    indented = "\n".join(f"  {line}" for line in cleaned.splitlines())
    return f"\n{indented}"


def render_setup_variables_md(
    nnn: int,
    wf_path: Path,
    multilines: dict[str, list[str]],
    setup_variables: dict[str, list[str]],
    director_node: Optional[dict],
    timeline: dict[str, list],
    output_tracking: dict[str, str],
) -> str:
    segments = timeline.get("segments") or []
    audio_segs = timeline.get("audioSegments") or []
    dur_f = director_widget(director_node, "duration_frames")
    dur_s = director_widget(director_node, "duration_seconds")
    fps = director_widget(director_node, "frame_rate")
    epsilon = director_widget(director_node, "epsilon")
    rows = [
        ("Capture run number", f"{nnn:03d} (from runs folder/log)"),
        ("Setup name", wf_path.stem),
        ("Image segments", f"{len(segments)} video / {len(audio_segs)} audio"),
        ("Prompts / similarity", segment_prompt_mode(segments)),
        ("Commit No.", short_sha()),
        ("LTX-Director (flavour)", director_flavor(director_node)),
        ("Audio src", derive_audio_state(director_node, timeline)),
        ("epsilon", "" if epsilon is None else str(epsilon)),
        ("Duration", f"{dur_f} frames @ {fps} fps ({dur_s} sec)"),
        ("RELAY_OVERRIDES", first_multiline(multilines, "relay_overrides")),
        (
            "GLOBAL [ path ] - working folder",
            first_multiline(multilines, "working_folder"),
        ),
        ("GLOBAL [ base name ]", first_multiline(multilines, "name")),
        ("INPUT Path [ EXR ]", first_multiline(setup_variables, "input_path_exr")),
        ("GLOBAL [ version ]", first_multiline(setup_variables, "version")),
        ("GLOBAL [ run offset ]", first_multiline(setup_variables, "run_offset")),
        ("Output folder", output_tracking.get("folder", "")),
        ("Output name", output_tracking.get("name", "")),
        ("OVERLAY - INFO", "(captured verbatim above)"),
        ("OVERLAY - FEEDBACK", "(captured verbatim above)"),
    ]
    body = "\n".join(
        f"- {label}: {_notes_value(value)}" for label, value in rows
    )
    return f"## SETUP variables (captured)\n\n{body}\n\n"


def render_notes_md(
    nnn: int,
    wf_path: Path,
    multilines: dict[str, list[str]],
    setup_variables: dict[str, list[str]],
    director_node: Optional[dict],
    timeline: dict[str, list],
    audio_src: str,
    info_body: str,
    feedback_lines: list[str],
    scores: dict[str, Optional[int]],
    output_tracking: dict[str, str],
) -> str:
    """Notes pulled exclusively from the two source families the card
    is allowed to read: the OVERLAY-* multilines and the active
    Director node. No widget-scraping of BasicScheduler / KSamplerSelect /
    RandomNoise / CFGGuider — those don't define what this loop sweeps."""
    info_indented = (
        "\n".join("    " + line for line in info_body.splitlines())
        if info_body.strip()
        else "    (none)"
    )
    feedback_body = "\n".join(feedback_lines) if feedback_lines else "(none)"
    scores_line = " · ".join(
        f"{k}: {v if v is not None else '?'}/5"
        for k, v in scores.items()
    )

    if director_node is None:
        director_kind = (
            "**missing** — no `LTXDirector__koolook` node on the "
            "canvas; this render didn't run through the Koolook fork."
        )
    elif not is_koolook_director(director_node):
        director_kind = (
            f"**upstream** `{director_type(director_node)}` "
            "(Koolook variant NOT wired — relay_overrides + per-segment "
            "sigma are INERT this render)"
        )
    else:
        director_kind = (
            f"{director_flavor(director_node)} (`{director_type(director_node)}`)"
        )

    epsilon = director_widget(director_node, "epsilon")
    dur_f = director_widget(director_node, "duration_frames")
    dur_s = director_widget(director_node, "duration_seconds")
    fps = director_widget(director_node, "frame_rate")
    segments = timeline.get("segments") or []
    audio_segs = timeline.get("audioSegments") or []
    prompt_mode = segment_prompt_mode(segments)

    return (
        f"# Run notes\n\n"
        f"## Maintainer feedback (OVERLAY - FEEDBACK, verbatim)\n\n"
        f"{feedback_body}\n\n"
        f"**Scores:** {scores_line}\n\n"
        f"## OVERLAY - INFO (verbatim)\n\n"
        f"{info_indented}\n\n"
        f"{render_setup_variables_md(nnn, wf_path, multilines, setup_variables, director_node, timeline, output_tracking)}"
        f"## Director node — structural state\n\n"
        f"- Variant: {director_kind}\n"
        f"- Audio src: {audio_src}\n"
        f"- ε (epsilon): {epsilon!r}\n"
        f"- Duration: {dur_f} frames @ {fps} fps ({dur_s} sec)\n"
        f"- Segments: {len(segments)} video / "
        f"{len(audio_segs)} audio\n"
        f"- Segment prompt mode: {prompt_mode}\n"
    )


def render_log_row(
    nnn: int,
    director_node: Optional[dict],
    relay_overrides_raw: str,
    audio_src: str,
    timeline: dict[str, list],
    scores: dict[str, Optional[int]],
    feedback_lines: list[str],
) -> str:
    """Rolling-table row aligned with the card's data-source rule —
    multilines + active Director only. Scheduler/sampler columns are
    deliberately absent (they aren't what this loop sweeps)."""
    director_cell = director_type(director_node)
    relay_cell = (
        f"`{relay_overrides_raw.strip()}`"
        if relay_overrides_raw.strip()
        else "(empty → defaults)"
    )
    segments = timeline.get("segments") or []
    audio_segments = timeline.get("audioSegments") or []
    seg_cell = f"{len(segments)}v/{len(audio_segments)}a"
    # `or '?'` would map a legitimate 0/5 to '?' because 0 is falsy;
    # explicit None-check preserves the score the maintainer typed.
    # Parallels render_audio_card._s(): the log uses bare digits, while
    # the card uses N/5 labels.
    def _score(v: Optional[int]) -> str:
        return str(v) if v is not None else "?"
    score_cell = (
        f"M{_score(scores.get('motion'))}·S{_score(scores.get('sync'))}·"
        f"Sh{_score(scores.get('sharp'))}"
    )
    notes_cell = (
        " ".join(feedback_lines)[:120] if feedback_lines else "(none)"
    ).replace("|", "\\|")
    return (
        f"| {nnn:03d} | {date.today().isoformat()} | `{director_cell}` "
        f"| {relay_cell} | {audio_src} | {seg_cell} "
        f"| {score_cell} | {notes_cell} |\n"
    )


# --------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------


def _build_state_for_card(
    nnn: int, label: str, wf_path: Path,
    multilines: dict[str, list[str]],
    output_tracking: dict[str, str],
    metadata: dict[str, Any],
    director_node: Optional[dict],
    timeline: dict[str, list],
    audio_src: str,
    scores: dict[str, Optional[int]], feedback_lines: list[str],
) -> dict[str, Any]:
    """Card state — strict subset matching the agreed source families:
    five tracked multilines + the active Director node's own values.
    No git, no _dev_build.json, no scheduler scrape.

    Director's duration_frames / duration_seconds widgets are NOT
    included — the card dropped the Duration row in favour of
    per-segment time ranges, and notes.md reads them straight from
    `director_node` via `director_widget`. Keeping them out of state
    prevents the consumed-but-unused asymmetry the review caught."""
    return {
        "run_number": nnn,
        "run_label": label,
        "date": date.today().isoformat(),
        "workflow_name": f"run{nnn:03d}_workflow.json",
        "name": first_multiline(multilines, "name").strip() or "(unnamed)",
        "relay_overrides_raw": first_multiline(multilines, "relay_overrides"),
        "info_body": first_multiline(multilines, "overlay - info").rstrip(),
        "feedback_lines": feedback_lines,
        "scores": scores,
        "work_folder": pick_existing_path(
            multilines.get("working_folder") or []
        ),
        "output_folder": output_tracking.get("folder", ""),
        "output_name": output_tracking.get("name", ""),
        "metadata": metadata,
        "director_node": director_node,
        "director_variant": director_type(director_node),
        "director_flavor": director_flavor(director_node),
        "audio_src": audio_src,
        "epsilon": director_widget(director_node, "epsilon"),
        "frame_rate": director_widget(director_node, "frame_rate"),
        "segments": timeline.get("segments") or [],
        "audio_segments": timeline.get("audioSegments") or [],
        "segment_prompt_mode": segment_prompt_mode(
            timeline.get("segments") or []
        ),
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
    env_file = find_dotenv()
    if env_file is not None:
        load_dotenv(env_file)

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

    with wf_path.open(encoding="utf-8-sig") as f:
        wf = json.load(f)
    nodes = wf.get("nodes") or []

    multilines = extract_multilines(nodes, cfg["tracked_multilines"])
    setup_variables = extract_setup_variables(
        nodes, cfg.get("tracked_setup_variables", {})
    )
    output_tracking = expected_output_tracking(nodes, multilines, setup_variables)
    director_node = extract_director(nodes)
    timeline = parse_timeline(director_node)
    audio_src = derive_audio_state(director_node, timeline)
    scores, feedback_lines = parse_feedback(
        first_multiline(multilines, "overlay - feedback")
    )

    name = first_multiline(multilines, "name").strip()
    relay_overrides_raw = first_multiline(multilines, "relay_overrides")
    info_body = first_multiline(multilines, "overlay - info").rstrip()
    label = args.label or autogen_label(
        name, director_node, relay_overrides_raw
    )
    nnn = next_run_number(runs_dir)
    run_dir = runs_dir / f"run-{nnn:03d}_{label}"
    build = read_dev_build_json()
    fork_status = fork_dir_status(cfg["fork_to_track"])
    metadata = card_metadata(
        nnn, label, wf_path, multilines, setup_variables, output_tracking,
        director_node, timeline, audio_src, build, fork_status,
    )

    # Two-line chat-report header (matches dev-sync convention).
    print(f"{short_sha()} - {REPO_ROOT.name}")
    print(
        f"loop-{cfg['job_name']} run-{nnn:03d}  workflow={wf_path.name}  "
        f"director={director_type(director_node)}  "
        f"audio={audio_src}"
    )

    if args.dry_run:
        print(f"  (dry-run) would write: {run_dir.relative_to(REPO_ROOT)}")
        print(f"  (dry-run) would append row to: {log_path.relative_to(REPO_ROOT)}")
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(wf_path, run_dir / f"run{nnn:03d}_workflow.json")
    (run_dir / "relay_overrides.txt").write_text(
        render_relay_overrides_txt(relay_overrides_raw, director_node),
        encoding="utf-8",
    )
    (run_dir / "patch_state.txt").write_text(
        render_patch_state(build, fork_status),
        encoding="utf-8",
    )
    (run_dir / "notes.md").write_text(
        render_notes_md(
            nnn, wf_path, multilines, setup_variables,
            director_node, timeline, audio_src,
            info_body, feedback_lines, scores, output_tracking,
        ),
        encoding="utf-8",
    )

    card_status = "skipped"
    delivery_status = "skipped"
    if cfg["render_card"] and not args.no_card:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from make_card_audio import render_audio_card  # type: ignore[import-not-found]

            state = _build_state_for_card(
                nnn, label, wf_path,
                multilines, output_tracking, metadata, director_node, timeline,
                audio_src, scores, feedback_lines,
            )
            card_path = render_audio_card(state, run_dir / "card.png")
            card_status = "rendered"
            delivery_status = copy_delivery_card(card_path, output_tracking)
        except ImportError as exc:
            card_status = f"skipped ({exc})"
        except OSError as exc:
            card_status = f"failed ({exc})"

    if not args.no_log:
        row = render_log_row(
            nnn, director_node, relay_overrides_raw, audio_src,
            timeline, scores, feedback_lines,
        )
        with log_path.open("a", encoding="utf-8") as f:
            f.write(row)

    print(f"  wrote:  {run_dir.relative_to(REPO_ROOT)}")
    print(f"  card:   {card_status}")
    print(f"  card delivery: {delivery_status}")
    if not args.no_log:
        print(f"  logged: {log_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
