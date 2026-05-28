#!/usr/bin/env python3
"""
Audio-lipsync card renderer — vertical PIL card scoped to the
``docs/automations/LTX-2.3/audio-lipsync/`` iteration loop.

Sibling to ``scripts/make_card.py`` (base-1step). The two families
share palette, font fallback chain, and layout primitives — so the
cards read as a set — but each highlights what matters for its module:

  base-1step       Phase 1 / Phase 2 / Base · model / Base · locked /
                   Base · scene / Outcome — the upstream-default
                   parameter sweep view.
  audio-lipsync    Knob state / Fork state / Sampler / Base · notes /
                   Feedback / Outcome — the fork-iteration view,
                   dominated by the relay_overrides JSON + which fork
                   SHA produced the render.

Visual conventions copied 1:1 from make_card.py so the rendering is
consistent: small uppercase accent-coloured section labels (no solid
header bars), subtle panel outlines, key/value mono-rows, and
left-accented note boxes for the prose blocks. When make_card.py
evolves, update both here and there until we factor the primitives
into a shared module.

Two entry points:

1. From loop_audio.py at end-of-snapshot:

       from scripts.make_card_audio import render_audio_card
       render_audio_card(state, run_dir / "card.png")

2. Standalone, against an existing run folder (re-render after a
   layout tweak):

       python scripts/make_card_audio.py <run_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


# --------------------------------------------------------------------------
# Palette + geometry — kept in lockstep with scripts/make_card.py.
# --------------------------------------------------------------------------

W            = 540
PAD_X        = 28
PAD_TOP      = 28
PAD_BOTTOM   = 28
BG_OUTER     = (14, 14, 14)
BG_CARD      = (21, 21, 21)
BG_SECTION   = (26, 26, 31)
BORDER       = (48, 47, 47)
BORDER_STR   = (79, 79, 84)
TEXT         = (249, 250, 251)
MUTED        = (143, 149, 156)
DIM          = (200, 204, 209)
HEADER_GREY  = (201, 204, 209)
ACCENT_RUN   = (255, 184, 77)    # amber  — knob state (what changes per render)
ACCENT_BASE  = (109, 180, 255)   # sky    — frozen base / fork pin / sampler
ACCENT_FORK  = (175, 138, 230)   # violet — fork state, visually distinct from knob
ACCENT_OUT   = (123, 207, 128)   # green  — post-render outcome
NOTE_BG      = (12, 12, 12)
RADIUS       = 14
SECTION_RADIUS = 8


# --------------------------------------------------------------------------
# Fonts — same fallback chain + same sizes as scripts/make_card.py.
# --------------------------------------------------------------------------

WIN_FONTS = Path(r"C:/Windows/Fonts")


def _load_font(filenames: list[str], size: int) -> ImageFont.FreeTypeFont:
    for fn in filenames:
        for p in (WIN_FONTS / fn, Path("/Library/Fonts") / fn,
                  Path("/usr/share/fonts") / fn):
            if p.exists():
                try:
                    return ImageFont.truetype(str(p), size)
                except OSError:
                    continue
    return ImageFont.load_default()


F_TITLE   = _load_font(["segoeuib.ttf", "Arial Bold.ttf"], 28)
F_SUB     = _load_font(["segoeui.ttf",  "Arial.ttf"],      17)
F_H2      = _load_font(["segoeuib.ttf", "Arial Bold.ttf"], 16)
F_TAG     = _load_font(["segoeuib.ttf", "Arial Bold.ttf"], 12)
F_SECTION = _load_font(["segoeuib.ttf", "Arial Bold.ttf"], 17)
F_MONO    = _load_font(["consola.ttf",  "Menlo.ttc", "DejaVuSansMono.ttf"], 19)
F_NOTE    = _load_font(["segoeuii.ttf", "Arial Italic.ttf"], 18)


# --------------------------------------------------------------------------
# Drawing primitives — copied 1:1 from scripts/make_card.py. The two
# renderers stay in visual sync by sharing these signatures.
# --------------------------------------------------------------------------


def _wrap_text(text: str, max_chars: int, keep_blank_lines: bool = False) -> list[str]:
    out: list[str] = []
    for raw_line in text.split("\n"):
        if not raw_line.strip():
            if keep_blank_lines:
                out.append("")
            continue
        words = raw_line.split()
        cur = ""
        for w in words:
            if len(cur) + len(w) + 1 <= max_chars:
                cur = (cur + " " + w).strip()
            else:
                if cur:
                    out.append(cur)
                cur = w
        if cur:
            out.append(cur)
    return out


def _draw_kv_row(
    draw: ImageDraw.ImageDraw, x: int, y: int,
    key: str, val: str, key_w: int,
) -> int:
    draw.text((x, y), key, font=F_MONO, fill=MUTED)
    draw.text((x + key_w, y), str(val), font=F_MONO, fill=TEXT)
    return y + 26


def _draw_text_box(
    draw: ImageDraw.ImageDraw, x: int, y: int, width: int,
    label: str, content: str, accent: tuple[int, int, int],
    max_lines: int = 4, char_per_line: int = 42,
) -> int:
    """Left-accented note block — colored 4-px bar on the left, label
    in the accent colour, italic body text. Same shape as make_card.py."""
    lines = _wrap_text(content, char_per_line)[:max_lines]
    box_h = 28 + 22 * max(1, len(lines)) + 14
    draw.rounded_rectangle(
        [x - 12, y, x + width, y + box_h],
        radius=4, fill=NOTE_BG,
    )
    draw.rectangle(
        [x - 12, y, x - 8, y + box_h],
        fill=accent,
    )
    draw.text((x + 4, y + 8), label.upper(), font=F_H2, fill=accent)
    line_y = y + 36
    for line in lines:
        draw.text((x + 4, line_y), line, font=F_NOTE, fill=DIM)
        line_y += 22
    return y + box_h + 4


def _draw_section(
    draw: ImageDraw.ImageDraw, x: int, y: int, width: int,
    accent: tuple[int, int, int], label: str, body_h: int,
    bg: tuple[int, int, int] = BG_SECTION,
    border: tuple[int, int, int] = BORDER,
) -> tuple[int, int, int, int]:
    """Subtle panel with an accent-coloured uppercase label at the top.
    Returns (content_x, content_y, content_w, end_y) so the caller
    stacks kv-rows directly underneath."""
    HEADER_H = 36
    section_h = HEADER_H + body_h + 14
    draw.rounded_rectangle(
        [x, y, x + width, y + section_h],
        radius=SECTION_RADIUS, fill=bg, outline=border, width=1,
    )
    draw.text((x + 14, y + 10), label.upper(), font=F_SECTION, fill=accent)
    return x + 14, y + HEADER_H, width - 28, y + section_h + 10


def _section_body_rows(num_rows: int, extra: int = 0) -> int:
    return num_rows * 26 + extra


# --------------------------------------------------------------------------
# Renderer — driven entirely by the ``state`` dict that loop_audio.py
# builds. Keep the rendering pure-state-in/PNG-out so the snapshot
# folder is always re-renderable.
# --------------------------------------------------------------------------


def render_audio_card(state: dict[str, Any], out_path: Path) -> Path:
    """Render the audio-lipsync card PNG. ``state`` keys consumed:

      run_number, run_label, date, name, workflow_name
      relay_overrides_raw, director, scheduler, scores
      feedback_lines, info_body, build, main_sha, fork_dir_status
    """
    director = state["director"]
    scheduler = state["scheduler"]
    scores = state["scores"]
    build = state.get("build") or {}

    # ----- canvas (oversized, cropped after layout) -----
    canvas_h = 2000
    img = Image.new("RGB", (W, canvas_h), BG_OUTER)
    draw = ImageDraw.Draw(img)

    inset = 18
    draw.rounded_rectangle(
        [inset, inset, W - inset, canvas_h - inset],
        radius=RADIUS, fill=BG_CARD, outline=BORDER, width=1,
    )

    x = inset + PAD_X
    y = inset + PAD_TOP
    inner_w = W - 2 * inset - 2 * PAD_X

    # ----- HEADER -----
    title_line = f"Run {state['run_number']:03d} — {state['name']}"
    sub_line = (
        f"{state['date']} · audio-lipsync · {state['workflow_name']}"
    )
    draw.text((x, y), title_line, font=F_TITLE, fill=TEXT)
    y += 38
    draw.text((x, y), sub_line, font=F_SUB, fill=MUTED)
    y += 30
    draw.line([(x, y), (x + inner_w, y)], fill=BORDER, width=1)
    y += 18

    key_w = 152

    # ----- KNOB STATE (amber) -----
    relay_raw = state["relay_overrides_raw"].strip()
    custom_audio = director.get("use_custom_audio")
    audio_cell = (
        "ON · custom file" if custom_audio is True
        else "off · model-generated" if custom_audio is False
        else "?"
    )
    relay_summary = relay_raw if relay_raw else "(empty → defaults)"
    body_h = _section_body_rows(2)
    cx, cy, cw, end_y = _draw_section(
        draw, x, y, inner_w, ACCENT_RUN, "Knob state", body_h,
    )
    cy = _draw_kv_row(draw, cx, cy, "relay_overrides", relay_summary[:32], key_w)
    cy = _draw_kv_row(draw, cx, cy, "Custom audio", audio_cell, key_w)
    y = end_y

    # ----- FORK STATE (violet) -----
    director_variant = director.get("variant", "?")
    sync_line = (
        f"{build.get('commit', '—')} · {build.get('synced_at', '?')}"
    )
    body_h = _section_body_rows(5)
    cx, cy, cw, end_y = _draw_section(
        draw, x, y, inner_w, ACCENT_FORK, "Fork state", body_h,
    )
    cy = _draw_kv_row(draw, cx, cy, "MAIN sha", state["main_sha"], key_w)
    cy = _draw_kv_row(draw, cx, cy, "Synced (script)", sync_line[:28], key_w)
    cy = _draw_kv_row(draw, cx, cy, "Sync scope", build.get("scope", "—")[:28], key_w)
    cy = _draw_kv_row(draw, cx, cy, "Director node", director_variant[:28], key_w)
    cy = _draw_kv_row(
        draw, cx, cy, "Fork files",
        state["fork_dir_status"][:28], key_w,
    )
    y = end_y

    # ----- SAMPLER (sky) -----
    schedulers = scheduler["schedulers"]
    samplers = scheduler["samplers"]
    noises = scheduler["noises"]
    cfgs = scheduler["cfgs"]
    if len(schedulers) >= 2:
        s1, s2 = schedulers[0], schedulers[1]
        if len(s2) > 2 and s2[2] == 1.0 and len(s1) > 2 and s1[2] < 1.0:
            s1, s2 = s2, s1
        p1 = f"{s1[0]} · {s1[1]} stp · d={s1[2]}"
        p2 = f"{s2[0]} · {s2[1]} stp · d={s2[2]}"
    elif schedulers:
        p1 = f"{schedulers[0][0]} · {schedulers[0][1]} stp · d={schedulers[0][2]}"
        p2 = "—"
    else:
        p1 = p2 = "—"
    sampler_join = " / ".join(str(s) for s in samplers) if samplers else "—"
    seed_val = str(noises[0][0]) if noises and noises[0] else "—"
    cfg_join = " / ".join(str(c) for c in cfgs) if cfgs else "—"

    body_h = _section_body_rows(5)
    cx, cy, cw, end_y = _draw_section(
        draw, x, y, inner_w, ACCENT_BASE, "Sampler", body_h,
    )
    cy = _draw_kv_row(draw, cx, cy, "Phase 1", p1, key_w)
    cy = _draw_kv_row(draw, cx, cy, "Phase 2", p2, key_w)
    cy = _draw_kv_row(draw, cx, cy, "Samplers", sampler_join, key_w)
    cy = _draw_kv_row(draw, cx, cy, "Seed", f"{seed_val} (fixed)", key_w)
    cy = _draw_kv_row(draw, cx, cy, "CFG", cfg_join, key_w)
    y = end_y

    # ----- BASE · NOTES (left-accented amber block) -----
    info_body = (state.get("info_body") or "").strip()
    if info_body:
        y = _draw_text_box(
            draw, x, y, inner_w, "Base · notes (Δ this run)",
            info_body, ACCENT_RUN, max_lines=18, char_per_line=42,
        )

    # ----- FEEDBACK (left-accented green block) -----
    feedback_lines = state.get("feedback_lines") or []
    feedback_text = "\n".join(feedback_lines) if feedback_lines else "(none)"
    y = _draw_text_box(
        draw, x, y, inner_w, "Feedback (video)",
        feedback_text, ACCENT_OUT, max_lines=6, char_per_line=42,
    )

    # ----- OUTCOME (sky → green scores row) -----
    body_h = _section_body_rows(1)
    cx, cy, cw, end_y = _draw_section(
        draw, x, y, inner_w, ACCENT_OUT, "Outcome", body_h,
    )
    score_cell = (
        f"Motion {scores['motion'] or '?'}/5"
        f"   Sync {scores['sync'] or '?'}/5"
        f"   Sharp {scores['sharp'] or '?'}/5"
    )
    draw.text((cx, cy), score_cell, font=F_MONO, fill=TEXT)
    y = end_y

    # ----- INERT WARNING (conditional) -----
    is_koolook = director.get("is_koolook")
    if not is_koolook and relay_raw:
        warn_text = (
            "Director is upstream LTXDirector — relay_overrides and per-segment "
            "σ are INERT for this render. Swap to LTX Director (Koolook v1.3.2) "
            "to make the knobs active."
        )
        y = _draw_text_box(
            draw, x, y, inner_w, "⚠ Director note",
            warn_text, ACCENT_RUN, max_lines=8, char_per_line=42,
        )

    # ----- crop to actual content + save -----
    final_h = y + PAD_BOTTOM + inset
    img = img.crop((0, 0, W, final_h))
    # redraw outer rounded-rect outline at the new height so the
    # bottom curve matches the top
    out_draw = ImageDraw.Draw(img)
    out_draw.rounded_rectangle(
        [inset, inset, W - inset, final_h - inset],
        radius=RADIUS, outline=BORDER, width=1,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path))
    return out_path


# --------------------------------------------------------------------------
# Standalone CLI — re-render a card from an existing run folder.
# --------------------------------------------------------------------------


def _rebuild_state_from_run_dir(run_dir: Path) -> dict[str, Any]:
    """Reconstruct enough of the loop_audio state dict to re-render
    the card. Re-reads workflow.json + small txt artifacts already in
    the snapshot."""
    from loop_audio import (  # type: ignore[import-not-found]
        extract_director_state, extract_multilines, extract_scheduler_chain,
        parse_feedback, read_dev_build_json, short_sha, fork_dir_status,
        load_config, DEFAULT_CONFIG_PATH,
    )

    wf_path = run_dir / "workflow.json"
    if not wf_path.is_file():
        print(f"missing workflow.json in {run_dir}", file=sys.stderr)
        sys.exit(2)
    with wf_path.open(encoding="utf-8") as f:
        wf = json.load(f)
    nodes = wf.get("nodes") or []

    cfg = load_config(DEFAULT_CONFIG_PATH)
    multilines = extract_multilines(nodes, cfg["tracked_multilines"])
    director = extract_director_state(nodes)
    scheduler = extract_scheduler_chain(nodes)
    scores, feedback_lines = parse_feedback(
        multilines.get("overlay - feedback", "")
    )

    name_parts = run_dir.name.split("_", 1)
    nnn = int(name_parts[0].removeprefix("run-")) if name_parts[0].startswith("run-") else 0
    label = name_parts[1] if len(name_parts) > 1 else ""

    from datetime import date
    return {
        "run_number": nnn,
        "run_label": label,
        "date": date.today().isoformat(),
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
        "fork_dir_status": fork_dir_status(cfg["fork_to_track"]),
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

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    # Load the repo's .env so the standalone re-render gets the same
    # KOLOOK_COMFYUI_DEV_PATH the live loop_audio run had — otherwise
    # the rebuilt state has an empty `build` dict and the FORK STATE
    # rows show `—` placeholders.
    from loop_audio import load_dotenv, REPO_ROOT  # type: ignore[import-not-found]
    load_dotenv(REPO_ROOT / ".env")

    state = _rebuild_state_from_run_dir(run_dir)

    out = run_dir / "card.png"
    if args.dry_run:
        print(f"(dry-run) would write: {out}")
        return 0
    render_audio_card(state, out)
    print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
