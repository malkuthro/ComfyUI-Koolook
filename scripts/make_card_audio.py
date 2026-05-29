#!/usr/bin/env python3
"""
Audio-lipsync card renderer — vertical PIL card scoped to the
``docs/automations/LTX-2.3/audio-lipsync/`` iteration loop.

Sibling to ``scripts/make_card.py`` (base-1step). Shares palette,
font fallback chain, and section primitives so the two families read
as a set, but the data sources are deliberately narrower:

Two — and only two — source families feed this card:

  1. The five ``Text Multiline`` nodes tracked by the loop config
     (name / relay_overrides / overlay - info / overlay - feedback /
     working_folder).
  2. The ``LTXDirector__koolook_v1_3_2`` node's own widget values and
     input-socket wiring (epsilon, frame_rate, duration_frames/seconds,
     timeline_data segments + audioSegments, use_custom_audio toggle,
     audio_vae link state).

Notably absent: BasicScheduler / KSamplerSelect / RandomNoise /
CFGGuider widget scrapes, ``_dev_build.json`` fork-state, ``git status``
output. Those don't define what this loop sweeps. Adding them would
push the card into showing values the maintainer's curated multiline
notes don't claim to summarise.

Card sections (top to bottom):

  HEADER            run-NNN — {name} · date · job · workflow filename
  KNOB STATE        relay_overrides (the per-render knob)
  BASE · NOTES      overlay - info (verbatim, Δ this run)
  BASE · LOCKED     epsilon · Audio src · Working folder (path-wrapped)
  BASE · SCENE      Segments (N) — indented segment list with time
                    ranges + flat Prompt/Audio/Keyframe coverage rows
  POST-RENDER       feedback body + outcome scores

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
TEXT         = (249, 250, 251)
MUTED        = (143, 149, 156)
DIM          = (200, 204, 209)
ACCENT_RUN   = (255, 184, 77)    # amber  — per-render knob state
ACCENT_BASE  = (109, 180, 255)   # sky    — locked / scene
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
# Drawing primitives — copied 1:1 from scripts/make_card.py.
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
    keep_blank_lines: bool = False,
) -> int:
    """Left-accented note block — colored 4-px bar on the left, label
    in the accent colour, italic body text."""
    lines = _wrap_text(content, char_per_line, keep_blank_lines=keep_blank_lines)[:max_lines]
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
    stacks rows directly underneath."""
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
# Renderer — pure state-in / PNG-out. The ``state`` dict is built by
# loop_audio._build_state_for_card. All values come from the two source
# families documented at the top of this module.
# --------------------------------------------------------------------------


def _wrap_path(path: str, max_chars: int = 40) -> list[str]:
    """Wrap a filesystem path onto multiple lines, breaking only on
    ``/`` or ``\\`` separators so directory names stay whole."""
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


def _video_segment_has_audio(video_seg: dict, audio_segs: list[dict]) -> bool:
    v_start = video_seg.get("start", 0)
    v_end = v_start + video_seg.get("length", 0)
    for a in audio_segs:
        a_start = a.get("start", 0)
        a_end = a_start + a.get("length", 0)
        if a_start < v_end and a_end > v_start:
            return True
    return False


def render_audio_card(state: dict[str, Any], out_path: Path) -> Path:
    """Render the audio-lipsync card PNG. ``state`` keys consumed:

      run_number, run_label, date, workflow_name
      name, relay_overrides_raw, info_body, feedback_lines, scores,
      work_folder
      director_node, director_variant, audio_src, epsilon,
      duration_frames, duration_seconds, frame_rate,
      segments, audio_segments
    """
    name              = state.get("name") or "(unnamed)"
    relay             = (state.get("relay_overrides_raw") or "").strip()
    info_body         = (state.get("info_body") or "").rstrip()
    feedback_lines    = state.get("feedback_lines") or []
    scores            = state.get("scores") or {}
    work_folder       = state.get("work_folder") or ""
    audio_src         = state.get("audio_src") or "?"
    epsilon           = state.get("epsilon")
    fps               = state.get("frame_rate")
    segments          = state.get("segments") or []
    audio_segs        = state.get("audio_segments") or []
    # NOTE: duration_frames / duration_seconds intentionally not read here
    # — the dropped "Duration" row used them; segment time ranges convey
    # the same info now. They stay in the state dict for notes.md.

    canvas_h = 2400
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
    # Wider key column so multi-word labels (relay_overrides, Working
    # folder) don't collide with their values.
    key_w = 178

    # ----- HEADER -----
    title_line = f"Run {state['run_number']:03d} — {name}"
    sub_line = (
        f"{state['date']} · audio-lipsync · {state['workflow_name']}"
    )
    draw.text((x, y), title_line, font=F_TITLE, fill=TEXT)
    y += 38
    draw.text((x, y), sub_line, font=F_SUB, fill=MUTED)
    y += 30
    draw.line([(x, y), (x + inner_w, y)], fill=BORDER, width=1)
    y += 18

    # ----- KNOB STATE (amber) — relay_overrides from the multiline -----
    relay_disp = relay if relay else "(empty → defaults)"
    body_h = _section_body_rows(1)
    cx, cy, cw, end_y = _draw_section(
        draw, x, y, inner_w, ACCENT_RUN, "Knob state", body_h,
    )
    _draw_kv_row(draw, cx, cy, "relay_overrides", relay_disp[:30], key_w)
    y = end_y

    # ----- BASE · NOTES (Δ this run, amber) — overlay - info verbatim -----
    note_text = info_body if info_body.strip() else "(no Δ this render)"
    y = _draw_text_box(
        draw, x, y, inner_w, "Base · notes (Δ this run)",
        note_text, ACCENT_RUN, max_lines=18, char_per_line=42,
        keep_blank_lines=True,
    )

    # ----- BASE · LOCKED (sky) — Koolook Director widgets + working folder -----
    locked_rows: list[tuple[str, str]] = []
    if epsilon is not None:
        locked_rows.append(("epsilon", str(epsilon)))
    locked_rows.append(("Audio src", audio_src))

    path_lines = _wrap_path(work_folder, max_chars=40) if work_folder else []
    body_h = _section_body_rows(len(locked_rows))
    if path_lines:
        body_h += 26 + 26 * len(path_lines) + 4
    cx, cy, cw, end_y = _draw_section(
        draw, x, y, inner_w, ACCENT_BASE, "Base · locked", body_h,
    )
    for k, v in locked_rows:
        cy = _draw_kv_row(draw, cx, cy, k, v, key_w)
    if path_lines:
        draw.text((cx, cy), "Working folder", font=F_MONO, fill=MUTED)
        cy += 26
        for line in path_lines:
            draw.text((cx + 12, cy), line, font=F_MONO, fill=TEXT)
            cy += 26
    y = end_y

    # ----- BASE · SCENE (sky) — flat layout with one indented child -----
    # Segments parent row + indented segment list + flat coverage rows
    # for Prompt / Audio / Keyframe. Coverage is aggregated across all
    # visible segments — "[x]" only when every segment has that field.
    seg_rows = min(len(segments), 6)
    indent_seg = 18
    body_h = 26 + 26 * seg_rows + 8 + 26 * 3 + 4
    cx, cy, cw, end_y = _draw_section(
        draw, x, y, inner_w, ACCENT_BASE, "Base · scene", body_h,
    )
    cy = _draw_kv_row(draw, cx, cy, "Segments", f"({len(segments)})", key_w)

    visible = segments[:seg_rows]
    all_have_prompt   = bool(visible) and all((s.get("prompt") or "") for s in visible)
    all_have_audio    = bool(visible) and all(
        _video_segment_has_audio(s, audio_segs) for s in visible
    )
    all_have_keyframe = bool(visible) and all(s.get("imageFile") for s in visible)

    for i, seg in enumerate(visible):
        start = seg.get("start", 0)
        length = seg.get("length", 0)
        start_s = start / fps if fps else 0
        end_s = (start + length) / fps if fps else 0
        prompt = seg.get("prompt") or ""
        has_p = bool(prompt)
        has_a = _video_segment_has_audio(seg, audio_segs)
        has_k = bool(seg.get("imageFile"))
        header_col = ACCENT_OUT if (has_p and has_a and has_k) else ACCENT_BASE
        header = f"{i+1}) {start_s:.0f} to {end_s:.0f} seconds"
        draw.text((cx + indent_seg, cy), header, font=F_MONO, fill=header_col)
        cy += 26
    cy += 8

    for label, present in (
        ("Prompt",   all_have_prompt),
        ("Audio",    all_have_audio),
        ("Keyframe", all_have_keyframe),
    ):
        mark = "[x]" if present else "[ ]"
        mark_col = ACCENT_OUT if present else MUTED
        draw.text((cx, cy), label, font=F_MONO, fill=MUTED)
        draw.text((cx + key_w, cy), mark, font=F_MONO, fill=mark_col)
        cy += 26
    y = end_y

    # ----- POST-RENDER (green) — FEEDBACK + OUTCOME -----
    feedback_text = "\n".join(feedback_lines) if feedback_lines else "(no feedback)"
    fb_lines = _wrap_text(feedback_text, 42)[:6]
    body_h = (20 + max(24, 22 * len(fb_lines)) + 14 + 20 + 26)
    cx, cy, cw, end_y = _draw_section(
        draw, x, y, inner_w, ACCENT_OUT, "Post-render", body_h,
    )
    draw.text((cx, cy), "FEEDBACK", font=F_TAG, fill=ACCENT_OUT)
    cy += 20
    for line in fb_lines:
        draw.text((cx, cy), line, font=F_NOTE, fill=DIM)
        cy += 22
    cy += 6
    draw.text((cx, cy), "OUTCOME", font=F_TAG, fill=ACCENT_OUT)
    cy += 20

    def _s(v: Any) -> str:
        return f"{v}/5" if v is not None else "?/5"
    draw.text(
        (cx, cy),
        f"Motion {_s(scores.get('motion'))}     "
        f"Sync {_s(scores.get('sync'))}     "
        f"Sharp {_s(scores.get('sharp'))}",
        font=F_MONO, fill=TEXT,
    )
    y = end_y

    # ----- crop to actual content + save -----
    final_h = y + PAD_BOTTOM + inset
    img = img.crop((0, 0, W, final_h))
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
    """Reconstruct the loop_audio state dict from an on-disk run
    folder. Reads the run's frozen workflow.json and walks it through
    the same extraction helpers loop_audio.py uses live."""
    from loop_audio import (  # type: ignore[import-not-found]
        DIRECTOR_TYPE, derive_audio_state, director_widget,
        extract_director, extract_multilines, find_dotenv,
        first_multiline, load_config, load_dotenv,
        parse_feedback, parse_timeline, pick_existing_path,
        DEFAULT_CONFIG_PATH,
    )

    wf_path = run_dir / "workflow.json"
    if not wf_path.is_file():
        print(f"missing workflow.json in {run_dir}", file=sys.stderr)
        sys.exit(2)
    with wf_path.open(encoding="utf-8") as f:
        wf = json.load(f)
    nodes = wf.get("nodes") or []

    # Load env so any env-var lookups inside the helpers (none right
    # now, but a defensive match for make_card.py's pattern) work.
    env_file = find_dotenv()
    if env_file is not None:
        load_dotenv(env_file)

    cfg = load_config(DEFAULT_CONFIG_PATH)
    multilines = extract_multilines(nodes, cfg["tracked_multilines"])
    director_node = extract_director(nodes)
    timeline = parse_timeline(director_node)
    audio_src = derive_audio_state(director_node, timeline)
    scores, feedback_lines = parse_feedback(
        first_multiline(multilines, "overlay - feedback")
    )

    name_parts = run_dir.name.split("_", 1)
    nnn = int(name_parts[0].removeprefix("run-")) if name_parts[0].startswith("run-") else 0
    label = name_parts[1] if len(name_parts) > 1 else ""

    from datetime import date
    return {
        "run_number": nnn,
        "run_label": label,
        "date": date.today().isoformat(),
        "workflow_name": "workflow.json",
        "name": first_multiline(multilines, "name").strip() or "(unnamed)",
        "relay_overrides_raw": first_multiline(multilines, "relay_overrides"),
        "info_body": first_multiline(multilines, "overlay - info").rstrip(),
        "feedback_lines": feedback_lines,
        "scores": scores,
        "work_folder": pick_existing_path(
            multilines.get("working_folder") or []
        ),
        "director_node": director_node,
        "director_variant": DIRECTOR_TYPE if director_node else "(missing)",
        "audio_src": audio_src,
        "epsilon": director_widget(director_node, "epsilon"),
        "duration_frames": director_widget(director_node, "duration_frames"),
        "duration_seconds": director_widget(director_node, "duration_seconds"),
        "frame_rate": director_widget(director_node, "frame_rate"),
        "segments": timeline.get("segments") or [],
        "audio_segments": timeline.get("audioSegments") or [],
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
