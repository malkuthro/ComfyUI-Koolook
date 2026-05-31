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
  2. The active ``LTXDirector`` node's own widget values and
     input-socket wiring (Koolook for modified runs, upstream original
     for A/B comparison; epsilon, frame_rate, timeline_data segments +
     audioSegments, use_custom_audio toggle, audio_vae link state).

Notably absent: BasicScheduler / KSamplerSelect / RandomNoise /
CFGGuider widget scrapes, ``_dev_build.json`` fork-state, ``git status``
output. Those don't define what this loop sweeps. Adding them would
push the card into showing values the maintainer's curated multiline
notes don't claim to summarise.

Card sections (top to bottom):

  HEADER            run-NNN — {name} · date · job · workflow filename
  KNOB STATE        relay_overrides (the per-render knob)
  BASE · NOTES      overlay - info (verbatim, Δ this run)
  BASE · LOCKED     Director · epsilon · Audio src · Working folder
                    (path-wrapped)
  BASE · SCENE      Segments (N) — indented segment list with time
                    ranges + Prompt mode + flat Prompt/Audio/Keyframe
                    coverage rows
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
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo


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
    key: str, val: str, key_w: int, max_w: int | None = None,
) -> int:
    draw.text((x, y), key, font=F_MONO, fill=MUTED)
    value = str(val)
    if max_w is not None:
        value = _trim_middle_to_width(draw, value, F_MONO, max_w)
    draw.text((x + key_w, y), value, font=F_MONO, fill=TEXT)
    return y + 26


def _trim_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_w: int,
) -> str:
    if draw.textbbox((0, 0), text, font=font)[2] <= max_w:
        return text
    ellipsis = "..."
    lo = 0
    hi = len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = text[:mid].rstrip() + ellipsis
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_w:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + ellipsis


def _trim_middle_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_w: int,
) -> str:
    if draw.textbbox((0, 0), text, font=font)[2] <= max_w:
        return text
    ellipsis = "..."
    keep_tail = min(18, max(8, len(text) // 3))
    tail = text[-keep_tail:]
    lo = 0
    hi = max(0, len(text) - keep_tail)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = text[:mid].rstrip("_- ") + ellipsis + tail
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_w:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip("_- ") + ellipsis + tail


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
        [x, y, x + width, y + box_h],
        radius=SECTION_RADIUS, fill=BG_SECTION, outline=BORDER, width=1,
    )
    draw.text((x + 12, y + 8), label.upper(), font=F_H2, fill=accent)
    line_y = y + 36
    for line in lines:
        draw.text((x + 12, line_y), line, font=F_NOTE, fill=DIM)
        line_y += 22
    return y + box_h + 10


def _draw_header_box(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    run_line: str,
    module_name: str,
    name_line: str,
) -> int:
    box_h = 118
    draw.rounded_rectangle(
        [x, y, x + width, y + box_h],
        radius=SECTION_RADIUS, fill=BG_SECTION, outline=BORDER, width=1,
    )
    tx = x + 14
    ty = y + 14
    run_id, _, run_date = run_line.partition(" - ")
    draw.text((tx, ty), run_id, font=F_H2, fill=ACCENT_RUN)
    run_w = draw.textbbox((0, 0), run_id + "  ", font=F_H2)[2]
    if run_date:
        draw.text((tx + run_w, ty), run_date, font=F_SUB, fill=MUTED)
    ty += 27
    module_label = "Module:"
    draw.text((tx, ty), module_label, font=F_SUB, fill=ACCENT_BASE)
    label_w = draw.textbbox((0, 0), module_label + " ", font=F_SUB)[2]
    draw.text((tx + label_w, ty), module_name, font=F_SUB, fill=TEXT)
    ty += 31
    draw.text((tx, ty), name_line, font=F_TITLE, fill=TEXT)
    return y + box_h + 10


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


def _draw_score_chip(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    label: str,
    value: Any,
) -> None:
    draw.rounded_rectangle(
        [x, y, x + width, y + 50],
        radius=6, fill=BG_CARD, outline=BORDER, width=1,
    )
    label_w = draw.textbbox((0, 0), label, font=F_TAG)[2]
    draw.text((x + (width - label_w) // 2, y + 7), label, font=F_TAG, fill=ACCENT_OUT)
    score = f"{value}/5" if value is not None else "?/5"
    score_w = draw.textbbox((0, 0), score, font=F_MONO)[2]
    draw.text((x + (width - score_w) // 2, y + 24), score, font=F_MONO, fill=TEXT)


# --------------------------------------------------------------------------
# Renderer — pure state-in / PNG-out. The ``state`` dict is built by
# loop_audio._build_state_for_card. All values come from the two source
# families documented at the top of this module.
# --------------------------------------------------------------------------


# Shared helpers live in loop_audio so the renderer and extractor agree
# byte-for-byte on path-wrapping rules and audio-overlap geometry.
# Importing here keeps the standalone CLI working (it adds scripts/ to
# sys.path before invoking render_audio_card).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from loop_audio import (  # type: ignore[import-not-found]  # noqa: E402
    copy_delivery_card,
    video_segment_has_audio,
    wrap_path,
)


def _audio_source_label(audio_src: str) -> str:
    labels = {
        "custom": "custom audio ON",
        "custom (empty)": "custom audio ON (empty)",
        "model-gen": "model audio",
        "off (no VAE)": "audio OFF",
        "(no director)": "no director",
    }
    return labels.get(audio_src, audio_src)


def render_audio_card(state: dict[str, Any], out_path: Path) -> Path:
    """Render the audio-lipsync card PNG. ``state`` keys consumed:

      run_number, run_label, date, workflow_name
      name, relay_overrides_raw, info_body, feedback_lines, scores,
      work_folder, output_folder, output_name
      director_variant, director_flavor, audio_src, epsilon, frame_rate,
      segments, audio_segments, segment_prompt_mode
    """
    name              = state.get("name") or "(unnamed)"
    relay             = (state.get("relay_overrides_raw") or "").strip()
    info_body         = (state.get("info_body") or "").rstrip()
    feedback_lines    = state.get("feedback_lines") or []
    scores            = state.get("scores") or {}
    work_folder       = state.get("work_folder") or ""
    output_folder     = state.get("output_folder") or ""
    output_name       = state.get("output_name") or ""
    metadata          = state.get("metadata") or {}
    director_variant  = state.get("director_variant") or "(missing)"
    director_flavor   = state.get("director_flavor") or director_variant
    director_pin_tag  = state.get("director_pin_tag") or (
        (metadata.get("director") or {}).get("pin_tag", "")
    )
    audio_src         = state.get("audio_src") or "?"
    epsilon           = state.get("epsilon")
    fps               = state.get("frame_rate")
    segments          = state.get("segments") or []
    audio_segs        = state.get("audio_segments") or []
    prompt_mode       = state.get("segment_prompt_mode") or "none"
    run_meta          = metadata.get("run") or {}
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
    module_name = state.get("module_name") or "audio-lipsync"
    run_line = f"Run {state['run_number']:03d} - {state['date']}"
    name_line = _trim_to_width(draw, str(name), F_TITLE, inner_w)
    y = _draw_header_box(draw, x, y, inner_w, run_line, module_name, name_line)

    # ----- BASE / RUN -----
    path_lines = wrap_path(work_folder, max_chars=40) if work_folder else []
    output_lines = wrap_path(output_folder, max_chars=40) if output_folder else []
    body_h = 0
    if output_name:
        body_h += 52
    source_workflow = state.get("source_workflow_name") or run_meta.get("workflow")
    if source_workflow:
        source_workflow = str(Path(str(source_workflow)).with_suffix(""))
    source_workflow_lines = (
        wrap_path(str(source_workflow), max_chars=40) if source_workflow else []
    )
    if source_workflow_lines:
        body_h += 26 + 26 * len(source_workflow_lines) + 4
    if path_lines:
        body_h += 26 + 26 * len(path_lines) + 4
    if output_lines:
        body_h += 26 + 26 * len(output_lines) + 4
    cx, cy, cw, end_y = _draw_section(
        draw, x, y, inner_w, ACCENT_BASE, "Base / run", body_h,
    )
    if source_workflow_lines:
        draw.text((cx, cy), "Copied from", font=F_MONO, fill=MUTED)
        cy += 26
        for line in source_workflow_lines:
            draw.text((cx + 12, cy), line, font=F_MONO, fill=TEXT)
            cy += 26
        cy += 4
    if output_name:
        draw.text((cx, cy), "Output name", font=F_MONO, fill=MUTED)
        cy += 26
        draw.text(
            (cx + 12, cy),
            _trim_middle_to_width(draw, output_name, F_MONO, cw - 12),
            font=F_MONO,
            fill=TEXT,
        )
        cy += 26
    if path_lines:
        draw.text((cx, cy), "Working folder", font=F_MONO, fill=MUTED)
        cy += 26
        for line in path_lines:
            draw.text((cx + 12, cy), line, font=F_MONO, fill=TEXT)
            cy += 26
    if output_lines:
        draw.text((cx, cy), "Output folder", font=F_MONO, fill=MUTED)
        cy += 26
        for line in output_lines:
            draw.text((cx + 12, cy), line, font=F_MONO, fill=TEXT)
            cy += 26
    y = end_y

    # ----- DIRECTOR READINGS -----
    seg_rows = min(len(segments), 6)
    indent_seg = 18
    director_rows = 5 if epsilon is not None else 4
    body_h = 26 * director_rows + 26 * 3 + 26 * seg_rows + 8 + 26 * 3 + 4
    cx, cy, cw, end_y = _draw_section(
        draw, x, y, inner_w, ACCENT_BASE, "Director readings", body_h,
    )
    cy = _draw_kv_row(
        draw, cx, cy, "Director", str(director_flavor),
        key_w, cw - key_w,
    )
    cy = _draw_kv_row(
        draw, cx, cy, "Pin tag",
        _trim_middle_to_width(draw, str(director_pin_tag), F_MONO, cw - key_w),
        key_w, cw - key_w,
    )
    if epsilon is not None:
        cy = _draw_kv_row(draw, cx, cy, "epsilon", str(epsilon), key_w)
    cy = _draw_kv_row(draw, cx, cy, "Audio", _audio_source_label(audio_src), key_w)
    cy = _draw_kv_row(draw, cx, cy, "Frame rate", str(fps or "?"), key_w)
    cy = _draw_kv_row(draw, cx, cy, "Image segments", f"({len(segments)})", key_w)
    cy = _draw_kv_row(draw, cx, cy, "Audio segments", f"({len(audio_segs)})", key_w)
    cy = _draw_kv_row(draw, cx, cy, "Prompt mode", prompt_mode, key_w)

    visible = segments[:seg_rows]
    all_have_prompt = bool(visible) and all((s.get("prompt") or "") for s in visible)
    all_have_audio = bool(visible) and all(
        video_segment_has_audio(s, audio_segs) for s in visible
    )
    all_have_keyframe = bool(visible) and all(s.get("imageFile") for s in visible)

    for i, seg in enumerate(visible):
        start = seg.get("start", 0)
        length = seg.get("length", 0)
        prompt = seg.get("prompt") or ""
        has_p = bool(prompt)
        has_a = video_segment_has_audio(seg, audio_segs)
        has_k = bool(seg.get("imageFile"))
        header_col = ACCENT_OUT if (has_p and has_a and has_k) else ACCENT_BASE
        if fps:
            start_s = start / fps
            end_s = (start + length) / fps
            header = f"{i+1}) {start_s:.0f} to {end_s:.0f} seconds"
        else:
            header = f"{i+1}) frames {start}-{start + length} (no fps)"
        draw.text((cx + indent_seg, cy), header, font=F_MONO, fill=header_col)
        cy += 26
    cy += 8

    for label, present in (
        ("Prompt", all_have_prompt),
        ("Audio", all_have_audio),
        ("Keyframe", all_have_keyframe),
    ):
        mark = "[x]" if present else "[ ]"
        mark_col = ACCENT_OUT if present else MUTED
        draw.text((cx, cy), label, font=F_MONO, fill=MUTED)
        draw.text((cx + key_w, cy), mark, font=F_MONO, fill=mark_col)
        cy += 26
    y = end_y

    # ----- AMBER NOTES -----
    relay_disp = relay if relay else "(empty -> defaults)"
    if relay and director_variant == "LTXDirector":
        relay_disp = f"{relay}\n\nINERT: active Director is upstream LTXDirector."
    y = _draw_text_box(
        draw, x, y, inner_w, "Knob state",
        relay_disp, ACCENT_RUN, max_lines=6, char_per_line=42,
        keep_blank_lines=True,
    )
    note_text = info_body if info_body.strip() else "(no note this render)"
    y = _draw_text_box(
        draw, x, y, inner_w, "Base notes",
        note_text, ACCENT_RUN, max_lines=18, char_per_line=42,
        keep_blank_lines=True,
    )

    # ----- POST-RENDER (green) — FEEDBACK + OUTCOME -----
    feedback_text = "\n".join(feedback_lines) if feedback_lines else "(no feedback)"
    fb_lines = _wrap_text(feedback_text, 42)[:6]
    body_h = (20 + max(24, 22 * len(fb_lines)) + 14 + 20 + 50)
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

    gap = 8
    chip_w = (cw - 2 * gap) // 3
    for idx, (label, key) in enumerate((
        ("Motion", "motion"),
        ("Sync", "sync"),
        ("Sharp", "sharp"),
    )):
        _draw_score_chip(
            draw, cx + idx * (chip_w + gap), cy,
            chip_w, label, scores.get(key),
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
    pnginfo = None
    metadata = state.get("metadata")
    if metadata:
        pnginfo = PngInfo()
        pnginfo.add_text(
            "koolook_audio_loop",
            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        )
    img.save(str(out_path), pnginfo=pnginfo)
    return out_path


# --------------------------------------------------------------------------
# Standalone CLI — re-render a card from an existing run folder.
# --------------------------------------------------------------------------


def _rebuild_state_from_run_dir(run_dir: Path) -> dict[str, Any]:
    """Reconstruct the loop_audio state dict from an on-disk run
    folder. Reads the run's frozen runNNN_workflow.json and walks it through
    the same extraction helpers loop_audio.py uses live."""
    from loop_audio import (  # type: ignore[import-not-found]
        derive_audio_state, detect_upstream_whatdreamscost_version,
        director_flavor, director_pin_tag, director_type, director_widget,
        expected_output_tracking, extract_director, extract_multilines,
        extract_setup_variables, find_dotenv,
        first_multiline, load_config, load_dotenv,
        parse_feedback, parse_timeline, pick_existing_path,
        scrub_path_for_metadata, segment_prompt_mode,
        DEFAULT_CONFIG_PATH,
    )

    name_parts = run_dir.name.split("_", 1)
    m = re.match(r"run-(\d+)$", name_parts[0])
    nnn = int(m.group(1)) if m else 0
    label = name_parts[1] if len(name_parts) > 1 else ""
    tagged_workflow = run_dir / f"run{nnn:03d}_workflow.json"
    wf_path = tagged_workflow if tagged_workflow.is_file() else run_dir / "workflow.json"
    if not wf_path.is_file():
        print(f"missing run workflow JSON in {run_dir}", file=sys.stderr)
        sys.exit(2)
    with wf_path.open(encoding="utf-8-sig") as f:
        wf = json.load(f)
    nodes = wf.get("nodes") or []

    # Load env so any env-var lookups inside the helpers (none right
    # now, but a defensive match for make_card.py's pattern) work.
    env_file = find_dotenv()
    if env_file is not None:
        load_dotenv(env_file)

    cfg = load_config(DEFAULT_CONFIG_PATH)
    multilines = extract_multilines(nodes, cfg["tracked_multilines"])
    setup_variables = extract_setup_variables(
        nodes, cfg.get("tracked_setup_variables", {})
    )
    output_tracking = expected_output_tracking(nodes, multilines, setup_variables)
    director_node = extract_director(nodes, wf.get("links") or [])
    timeline = parse_timeline(director_node)
    audio_src = derive_audio_state(director_node, timeline)
    upstream_whatdreamscost_version = detect_upstream_whatdreamscost_version()
    pin_tag = director_pin_tag(director_node, upstream_whatdreamscost_version)
    scores, feedback_lines = parse_feedback(
        first_multiline(multilines, "overlay - feedback")
    )
    patch_meta: dict[str, str] = {}
    patch_path = run_dir / "patch_state.txt"
    if patch_path.is_file():
        for line in patch_path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            patch_meta[key.strip()] = value.strip()
    source_workflow_name = ""
    notes_path = run_dir / "notes.md"
    if notes_path.is_file():
        for line in notes_path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"- Setup name:\s+`([^`]+)`", line)
            if match:
                source_workflow_name = f"{match.group(1)}.json"
                break

    from datetime import date
    workflow_name = wf_path.name
    return {
        "run_number": nnn,
        "run_label": label,
        "date": date.today().isoformat(),
        "workflow_name": workflow_name,
        "source_workflow_name": source_workflow_name,
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
        "output_tracking": output_tracking,
        "metadata": {
            "schema": "koolook.audio_loop.card_metadata.v1",
            "source": "rebuilt from frozen run folder",
            "run": {
                "capture_number": f"{nnn:03d}",
                "label": label,
                "date": date.today().isoformat(),
                "workflow": workflow_name,
                "source_workflow": source_workflow_name,
                "archived_workflow": workflow_name,
                "setup_name": (
                    Path(source_workflow_name).stem
                    if source_workflow_name else wf_path.stem
                ),
            },
            "setup": {
                "base_name": first_multiline(multilines, "name").strip(),
                "working_folder": scrub_path_for_metadata(output_tracking.get("folder", "")),
                "input_path_exr": scrub_path_for_metadata(
                    first_multiline(setup_variables, "input_path_exr")
                ),
                "global_version": first_multiline(setup_variables, "version"),
                "global_run_offset": first_multiline(setup_variables, "run_offset"),
                "relay_overrides": first_multiline(multilines, "relay_overrides").strip(),
            },
            "output": {
                **output_tracking,
                "folder": scrub_path_for_metadata(output_tracking.get("folder", "")),
            },
            "director": {
                "type": director_type(director_node),
                "flavor": director_flavor(director_node),
                "pin_tag": pin_tag,
                "upstream_whatdreamscost_version": upstream_whatdreamscost_version,
                "audio_src": audio_src,
                "epsilon": director_widget(director_node, "epsilon"),
                "duration_frames": director_widget(director_node, "duration_frames"),
                "duration_seconds": director_widget(director_node, "duration_seconds"),
                "frame_rate": director_widget(director_node, "frame_rate"),
                "segment_prompt_mode": segment_prompt_mode(
                    timeline.get("segments") or []
                ),
                "video_segments": len(timeline.get("segments") or []),
                "audio_segments": len(timeline.get("audioSegments") or []),
            },
            "repo": {
                "main_sha": patch_meta.get("MAIN SHA", ""),
                "sync_script": "sync_to_dev_audio.py",
                "last_dev_sync_audio": patch_meta.get("Last dev-sync-audio", ""),
                "sync_scope_tag": patch_meta.get("Sync scope tag", ""),
                "sync_worktree": patch_meta.get("Sync worktree", ""),
                "fork_dir_status": patch_meta.get("Fork dir status", ""),
            },
        },
        "director_node": director_node,
        "director_variant": director_type(director_node),
        "director_flavor": director_flavor(director_node),
        "director_pin_tag": pin_tag,
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
    p.add_argument("run_dir", type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"run folder not found: {run_dir}", file=sys.stderr)
        return 2

    # sys.path already pointed at scripts/ by the module-level import of
    # wrap_path/video_segment_has_audio above; no need to repeat it here.
    state = _rebuild_state_from_run_dir(run_dir)

    out = run_dir / "card.png"
    if args.dry_run:
        print(f"(dry-run) would write: {out}")
        return 0
    render_audio_card(state, out)
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            state.get("metadata") or {},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    delivery_status = copy_delivery_card(
        out,
        state.get("output_tracking") or {},
        state.get("run_number"),
        overwrite=True,
    )
    print(f"wrote: {out}")
    print(f"wrote: {metadata_path}")
    print(f"delivered: {delivery_status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
