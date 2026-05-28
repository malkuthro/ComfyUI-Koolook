#!/usr/bin/env python3
"""
Audio-lipsync card renderer — vertical PIL card scoped to the
``docs/automations/LTX-2.3/audio-lipsync/`` iteration loop. Renders
into the run-NNN snapshot folder produced by ``scripts/loop_audio.py``,
so each card travels with its workflow JSON + relay_overrides + patch
state + notes.

Sister to ``scripts/make_card.py`` (base-1step automation), not a
replacement. The two layouts highlight different things:

  base-1step       Phase 1 / Phase 2 / Base · model / Base · locked /
                   Base · scene / Outcome — the upstream-default
                   parameter sweep view.
  audio-lipsync    KNOB STATE / FORK STATE / SAMPLER / BASE notes /
                   OUTCOME — the fork-iteration view, dominated by
                   the relay_overrides JSON + which fork SHA produced
                   the render.

Same palette + font fallback chain as make_card.py so the two card
families read as a set. ~540 px wide, vertical, dark, beside-a-video.

The renderer is callable two ways:

1. From loop_audio.py at the end of a snapshot, with the in-memory
   state dict (no disk read):

       from scripts.make_card_audio import render_audio_card
       render_audio_card(state, out_path=run_dir / "card.png")

2. Standalone against an existing run folder, e.g. for re-renders or
   for refreshing the layout after a script tweak:

       python scripts/make_card_audio.py <run_dir>

Exit codes:
  0  card rendered (or would render, in --dry-run)
  2  run folder missing or unreadable
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


# --------------------------------------------------------------------------
# Palette & fonts — copied from make_card.py so the audio-lipsync card
# reads as the same family. Identical numeric values keep both renderers
# in sync visually until/unless we factor them into a shared module.
# --------------------------------------------------------------------------

W            = 540
PAD_X        = 28
PAD_TOP      = 28
PAD_BOTTOM   = 28

BG_OUTER     = (14, 14, 14)
BG_CARD      = (21, 21, 21)
BG_SECTION   = (26, 26, 31)
BORDER       = (48, 47, 47)
TEXT         = (249, 250, 251)
MUTED        = (143, 149, 156)
DIM          = (200, 204, 209)
HEADER_GREY  = (201, 204, 209)
ACCENT_KNOB  = (255, 184, 77)    # amber — knob state (the thing that changes)
ACCENT_BASE  = (109, 180, 255)   # sky — frozen / locked
ACCENT_FORK  = (175, 138, 230)   # violet — fork state, distinct from knob
ACCENT_OUT   = (123, 207, 128)   # green — outcome
NOTE_BG      = (12, 12, 12)
RADIUS       = 14
SECTION_RADIUS = 8


def _load_font(filenames: list[str], size: int) -> ImageFont.FreeTypeFont:
    """Try OS-specific font paths; fall back to PIL's default."""
    win = Path(r"C:/Windows/Fonts")
    mac = Path("/Library/Fonts")
    linux = Path("/usr/share/fonts")
    for fn in filenames:
        for p in (win / fn, mac / fn, linux / fn):
            if p.is_file():
                try:
                    return ImageFont.truetype(str(p), size)
                except OSError:
                    continue
    return ImageFont.load_default()


F_TITLE   = _load_font(["segoeuib.ttf", "Arial Bold.ttf"], 26)
F_SUB     = _load_font(["segoeui.ttf",  "Arial.ttf"],      16)
F_SECTION = _load_font(["segoeuib.ttf", "Arial Bold.ttf"], 14)
F_KEY     = _load_font(["segoeui.ttf",  "Arial.ttf"],      14)
F_MONO    = _load_font(["consola.ttf",  "Menlo.ttc", "DejaVuSansMono.ttf"], 14)
F_NOTE    = _load_font(["segoeuii.ttf", "Arial Italic.ttf"], 14)


# --------------------------------------------------------------------------
# Drawing helpers — minimal layout primitives, not a layout engine. If
# this card grows past 5–6 sections we should refactor into a generic
# vertical-section layout shared with make_card.py.
# --------------------------------------------------------------------------


def _draw_section_box(
    draw: ImageDraw.ImageDraw,
    y: int,
    title: str,
    rows: list[tuple[str, str]],
    accent: tuple[int, int, int],
    body_lines: list[str] | None = None,
) -> int:
    """One titled section with a colored header band + a vertical
    key/value table + an optional free-text block at the bottom.
    Returns the new ``y`` cursor after the section (caller stacks)."""
    inner_pad = 12
    title_h = 24
    row_h = 20
    note_pad = 8

    body_h = (
        len(body_lines) * row_h if body_lines else 0
    )
    section_h = (
        title_h + inner_pad
        + len(rows) * row_h
        + (note_pad + body_h if body_lines else 0)
        + inner_pad
    )

    box_l, box_r = PAD_X, W - PAD_X
    draw.rounded_rectangle(
        (box_l, y, box_r, y + section_h),
        radius=SECTION_RADIUS,
        fill=BG_SECTION,
        outline=BORDER,
        width=1,
    )

    # Header band
    draw.rectangle(
        (box_l, y, box_r, y + title_h),
        fill=accent,
    )
    draw.text(
        (box_l + 10, y + 3),
        title,
        font=F_SECTION,
        fill=(20, 20, 20),
    )

    # Rows
    row_y = y + title_h + inner_pad
    key_x = box_l + 14
    val_x = box_l + 130
    for k, v in rows:
        draw.text((key_x, row_y), k, font=F_KEY, fill=MUTED)
        draw.text((val_x, row_y), v, font=F_MONO, fill=TEXT)
        row_y += row_h

    # Optional body block (used for OVERLAY - INFO notes, feedback text)
    if body_lines:
        body_y = row_y + note_pad
        for line in body_lines:
            draw.text(
                (key_x, body_y),
                line,
                font=F_NOTE,
                fill=DIM,
            )
            body_y += row_h

    return y + section_h + 8


def _wrap_text(s: str, max_chars: int) -> list[str]:
    """Hard wrap by character count. Good enough for short body blocks."""
    out: list[str] = []
    for raw in s.splitlines():
        if not raw.strip():
            continue
        if len(raw) <= max_chars:
            out.append(raw)
            continue
        words = raw.split()
        line = ""
        for w in words:
            if len(line) + len(w) + 1 > max_chars:
                if line:
                    out.append(line)
                line = w
            else:
                line = (line + " " + w) if line else w
        if line:
            out.append(line)
    return out


# --------------------------------------------------------------------------
# State -> card. ``state`` is the dict loop_audio.py builds; keys
# intentionally mirror the script's extraction.
# --------------------------------------------------------------------------


def render_audio_card(state: dict[str, Any], out_path: Path) -> Path:
    """Render the card PNG. ``state`` keys consumed:

      run_number       int    1-based NNN
      run_label        str    sanitized slug (used in the run folder)
      date             str    ISO date (today)
      name             str    NAME multiline body
      workflow_name    str    source workflow filename (no path)
      relay_overrides_raw    str    RELAY_OVERRIDES body verbatim
      director         dict   from extract_director_state()
      scheduler        dict   from extract_scheduler_chain()
      scores           dict   {motion,sync,sharp -> int|None}
      feedback_lines   list[str]
      info_body        str    OVERLAY - INFO multiline body
      build            dict   from web/_dev_build.json (may be empty)
      main_sha         str    short repo HEAD SHA
      fork_dir_status  str    'clean (matches HEAD)' or git-status output
    """
    director = state["director"]
    scheduler = state["scheduler"]
    scores = state["scores"]
    build = state.get("build") or {}

    # ---- Build section rows ----------------------------------------

    relay_raw = state["relay_overrides_raw"].strip()
    relay_summary = relay_raw if relay_raw else "(empty → upstream defaults)"
    custom_audio = director.get("use_custom_audio")
    knob_rows = [
        ("relay_overrides", relay_summary[:48]),
        ("custom audio", (
            "ON" if custom_audio is True
            else "off" if custom_audio is False
            else "?"
        )),
    ]

    director_variant = director.get("variant", "?")
    is_koolook = director.get("is_koolook")
    fork_rows = [
        ("MAIN sha", state["main_sha"]),
        ("synced (script)", f"{build.get('commit', '—')} "
                            f"{build.get('synced_at', '')}".strip()),
        ("sync scope", build.get("scope", "—")),
        ("director node", director_variant),
        ("fork files", state["fork_dir_status"][:60]),
    ]

    schedulers = scheduler["schedulers"]
    samplers = scheduler["samplers"]
    noises = scheduler["noises"]
    cfgs = scheduler["cfgs"]
    if len(schedulers) >= 2:
        s1, s2 = schedulers[0], schedulers[1]
        if len(s2) > 2 and s2[2] == 1.0 and len(s1) > 2 and s1[2] < 1.0:
            s1, s2 = s2, s1
        p1 = f"{s1[0]} · {s1[1]} steps · d={s1[2]}"
        p2 = f"{s2[0]} · {s2[1]} steps · d={s2[2]}"
    elif schedulers:
        p1 = f"{schedulers[0][0]} · {schedulers[0][1]} steps · d={schedulers[0][2]}"
        p2 = "—"
    else:
        p1 = p2 = "—"
    sampler_rows = [
        ("Phase 1", p1),
        ("Phase 2", p2),
        ("samplers", " / ".join(str(s) for s in samplers) or "—"),
        ("seed", str(noises[0][0]) if noises and noises[0] else "—"),
        ("CFG", " / ".join(str(c) for c in cfgs) or "—"),
    ]

    info_body = (state.get("info_body") or "").strip()
    info_lines = _wrap_text(info_body, 52) if info_body else None

    score_cell = (
        f"Motion {scores['motion'] or '?'}/5  "
        f"·  Sync {scores['sync'] or '?'}/5  "
        f"·  Sharp {scores['sharp'] or '?'}/5"
    )
    outcome_rows = [("scores", score_cell)]
    feedback_lines = state.get("feedback_lines") or []
    feedback_wrapped = []
    for line in feedback_lines:
        feedback_wrapped.extend(_wrap_text(line, 52))

    # ---- Compute card height by walking sections in advance --------
    title_block_h = 64
    section_pad = 8

    def _section_h(rows: list, body_lines: list[str] | None) -> int:
        title_h = 24
        inner_pad = 12
        row_h = 20
        note_pad = 8
        body_h = len(body_lines) * row_h if body_lines else 0
        return (
            title_h + inner_pad + len(rows) * row_h
            + (note_pad + body_h if body_lines else 0)
            + inner_pad
        )

    sections = [
        (knob_rows, None),
        (fork_rows, None),
        (sampler_rows, None),
        ([], info_lines) if info_lines else None,
        (outcome_rows, feedback_wrapped),
    ]
    sections = [s for s in sections if s is not None]
    total_h = (
        PAD_TOP + title_block_h
        + sum(_section_h(rows, body) + section_pad for rows, body in sections)
        + PAD_BOTTOM
    )

    # ---- Render ---------------------------------------------------
    img = Image.new("RGB", (W, total_h), BG_OUTER)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        (8, 8, W - 8, total_h - 8),
        radius=RADIUS, fill=BG_CARD, outline=BORDER, width=1,
    )

    y = PAD_TOP
    title = f"Run {state['run_number']:03d} — {state['name']}"
    draw.text((PAD_X, y), title, font=F_TITLE, fill=TEXT)
    y += 32
    sub = (
        f"{state['date']} · audio-lipsync · {state['workflow_name']}"
    )
    draw.text((PAD_X, y), sub, font=F_SUB, fill=MUTED)
    y += 28

    section_specs = [
        ("KNOB STATE", knob_rows, None, ACCENT_KNOB),
        ("FORK STATE", fork_rows, None, ACCENT_FORK),
        ("SAMPLER", sampler_rows, None, ACCENT_BASE),
    ]
    if info_lines:
        section_specs.append(("BASE notes", [], info_lines, ACCENT_BASE))
    section_specs.append(
        ("OUTCOME", outcome_rows, feedback_wrapped, ACCENT_OUT)
    )

    for title, rows, body_lines, accent in section_specs:
        y = _draw_section_box(draw, y, title, rows, accent, body_lines)

    if not is_koolook and relay_raw:
        warn_h = 30
        draw.rounded_rectangle(
            (PAD_X, y, W - PAD_X, y + warn_h),
            radius=SECTION_RADIUS,
            fill=NOTE_BG, outline=ACCENT_KNOB, width=1,
        )
        draw.text(
            (PAD_X + 10, y + 7),
            "⚠ Director is upstream — relay_overrides INERT",
            font=F_NOTE, fill=ACCENT_KNOB,
        )
        y += warn_h + 8

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path))
    return out_path


# --------------------------------------------------------------------------
# Standalone entry point — rebuild a card from an existing run folder.
# Used for re-renders after a script tweak; loop_audio.py calls the
# function directly without going through this CLI.
# --------------------------------------------------------------------------


def _rebuild_state_from_run_dir(run_dir: Path) -> dict[str, Any]:
    """Reconstruct enough of the loop_audio state dict to re-render
    the card. We re-read workflow.json + the small txt artifacts that
    the snapshot already preserves."""
    # Local import to keep loop_audio import cheap when this script is
    # imported as a render helper.
    from loop_audio import (  # type: ignore[import-not-found]
        extract_director_state, extract_multilines, extract_scheduler_chain,
        parse_feedback, read_dev_build_json, short_sha, fork_dir_status,
    )

    wf_path = run_dir / "workflow.json"
    if not wf_path.is_file():
        print(f"missing workflow.json in {run_dir}", file=sys.stderr)
        sys.exit(2)
    with wf_path.open(encoding="utf-8") as f:
        wf = json.load(f)
    nodes = wf.get("nodes") or []

    multilines = extract_multilines(nodes)
    director = extract_director_state(nodes)
    scheduler = extract_scheduler_chain(nodes)
    scores, feedback_lines = parse_feedback(
        multilines.get("overlay - feedback", "")
    )

    # NNN is in the folder name: run-NNN_<label>
    m = run_dir.name.split("_", 1)
    nnn = int(m[0].removeprefix("run-")) if m[0].startswith("run-") else 0
    label = m[1] if len(m) > 1 else ""

    return {
        "run_number": nnn,
        "run_label": label,
        "date": run_dir.stat().st_mtime,  # caller can override
        "name": multilines.get("name", "(unnamed)"),
        "workflow_name": "workflow.json",
        "relay_overrides_raw": multilines.get("relay_overrides", ""),
        "director": director,
        "scheduler": scheduler,
        "scores": scores,
        "feedback_lines": feedback_lines,
        "info_body": multilines.get("overlay - info", ""),
        "build": read_dev_build_json(),
        "main_sha": short_sha(),
        "fork_dir_status": fork_dir_status(),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"run folder not found: {run_dir}", file=sys.stderr)
        return 2

    # Make loop_audio importable when running this script standalone.
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    state = _rebuild_state_from_run_dir(run_dir)
    from datetime import date
    state["date"] = date.today().isoformat()

    out = run_dir / "card.png"
    if args.dry_run:
        print(f"(dry-run) would write: {out}")
        return 0
    render_audio_card(state, out)
    print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
