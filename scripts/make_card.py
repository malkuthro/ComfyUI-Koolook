"""Render an experiment-tracking card PNG from a ComfyUI workflow JSON.

Usage:
    python scripts/make_card.py <workflow.json> [output.png]

Reads the LTX Director run/base context from the workflow JSON, paints a
self-contained card matching the look of
docs/investigations/experiments/combined-card.html, and writes it to PNG.

PNG can then be loaded into a ComfyUI workflow via a LoadImage node and
composited alongside the rendered video, or dropped straight into an NLE.

Tested against LTX_Director_4k_v0[1-3].json.
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def _load_dotenv(env_path: Path) -> None:
    """Minimal `.env` loader. Mirrors scripts/sync_to_dev.py — no python-dotenv dep."""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _find_dotenv() -> Path | None:
    """Find .env in the current worktree first, then fall back to the main repo
    root (resolved via git's common-dir) when running from a worktree.
    Returns the first existing .env found, or None."""
    repo_root = Path(__file__).resolve().parent.parent
    direct = repo_root / ".env"
    if direct.exists():
        return direct
    # Try the main repo root via git's common-dir.
    git_marker = repo_root / ".git"
    common_dir = None
    if git_marker.is_file():
        # Worktree marker file contains: "gitdir: <path-to>/<main>/.git/worktrees/<name>"
        try:
            content = git_marker.read_text(encoding="utf-8").strip()
            if content.startswith("gitdir:"):
                gitdir = Path(content.split(":", 1)[1].strip())
                # walk up from .git/worktrees/<name>/ to find <main repo>/
                if "worktrees" in gitdir.parts:
                    idx = gitdir.parts.index("worktrees")
                    common_dir = Path(*gitdir.parts[:idx])  # ends at <main>/.git
                    main_root = common_dir.parent
                    candidate = main_root / ".env"
                    if candidate.exists():
                        return candidate
        except Exception:
            pass
    return None


# Resolve repo root and load .env once on import — same pattern as sync_to_dev.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_env_file = _find_dotenv()
if _env_file is not None:
    _load_dotenv(_env_file)

# ---------- palette + geometry (kept in sync with combined-card.html) ----------
W            = 540           # slim vertical card — sits beside a video
PAD_X        = 28
PAD_TOP      = 28
PAD_BOTTOM   = 28
# Palette mirrored from docs/designs/snapshot-dialogs.html
BG_OUTER     = (14, 14, 14)      # --bg
BG_CARD      = (21, 21, 21)      # --panel
BG_SECTION   = (26, 26, 31)      # --panel-soft  (each section gets this)
BORDER       = (48, 47, 47)      # --border
BORDER_STR   = (79, 79, 84)      # --border-strong
TEXT         = (249, 250, 251)   # --text
MUTED        = (143, 149, 156)   # --muted
DIM          = (200, 204, 209)   # close to --text but slightly down
HEADER_GREY  = (201, 204, 209)   # snapshot dialog H2 colour
ACCENT_RUN   = (255, 184, 77)    # --amber  (PHASE 1 / changeable)
ACCENT_BASE  = (109, 180, 255)   # --sky    (frozen base)
ACCENT_OUT   = (123, 207, 128)   # --green  (post-render)
NOTE_BG      = (12, 12, 12)
RADIUS       = 14
SECTION_RADIUS = 8

# ---------- font loading ----------
WIN_FONTS = Path(r"C:/Windows/Fonts")
def load_font(filenames: list[str], size: int) -> ImageFont.FreeTypeFont:
    for fn in filenames:
        for p in (WIN_FONTS / fn, Path("/Library/Fonts") / fn, Path("/usr/share/fonts") / fn):
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()

F_TITLE = load_font(["segoeuib.ttf", "Arial Bold.ttf"], 28)
F_SUB   = load_font(["segoeui.ttf",  "Arial.ttf"],      17)
F_H2    = load_font(["segoeuib.ttf", "Arial Bold.ttf"], 16)
F_TAG     = load_font(["segoeuib.ttf", "Arial Bold.ttf"], 12)   # FEEDBACK / OUTCOME sub-labels
F_SECTION = load_font(["segoeuib.ttf", "Arial Bold.ttf"], 17)   # main section titles (PHASE 1 etc.)
F_MONO  = load_font(["consola.ttf",  "Menlo.ttc", "DejaVuSansMono.ttf"], 19)
F_NOTE  = load_font(["segoeuii.ttf", "Arial Italic.ttf"], 18)

# ---------- workflow extraction ----------
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi")
LOG_NAME   = "iterations.md"
AI_SUBDIR  = "_AI"    # AI-managed artifacts live here, separate from user content
CARD_NAME  = "card.png"
# Files containing this marker (case-insensitive) in their basename are
# post-loop outputs (the card already composited with the video) — skip
# them in auto-discovery so the script never picks its own output as input.
SKIP_MARKER = "loop"

def _is_loop_output(p: Path) -> bool:
    return SKIP_MARKER in p.stem.lower()
LOG_HEADER = (
    "# LTX Director — iterations log\n\n"
    "Append-only. Newest at bottom. Generated by `scripts/make_card.py`.\n\n"
    "| # | When | Run | Format | Denoise | JSON | Video | Δ from base | Feedback |\n"
    "| - | ---- | --- | ------ | ------- | ---- | ----- | ----------- | -------- |\n"
)


def log_iteration(folder: Path, json_path: Path, data: dict) -> None:
    """Append one row to <folder>/_AI/iterations.md. Dedupe by (json filename + json mtime).
    Markdown table — newest at bottom. Creates the file + the _AI subdir on first call."""
    ai_dir = folder / AI_SUBDIR
    ai_dir.mkdir(exist_ok=True)
    log_path = ai_dir / LOG_NAME
    json_mtime = json_path.stat().st_mtime
    json_mtime_iso = time.strftime("%Y-%m-%d %H:%M", time.localtime(json_mtime))
    fingerprint = f"{json_path.name}@{int(json_mtime)}"

    existing = ""
    last_fp = ""
    next_idx = 1
    if log_path.exists():
        existing = log_path.read_text(encoding="utf-8")
        rows = [ln for ln in existing.splitlines()
                if ln.startswith("| ") and not ln.startswith("| -")
                and not ln.startswith("| #")]
        next_idx = len(rows) + 1
        if rows:
            # last row's hidden HTML-comment fingerprint, if present
            tail_comment_re = re.compile(r"<!--fp:([^>]+)-->")
            for ln in reversed(rows):
                m = tail_comment_re.search(ln)
                if m:
                    last_fp = m.group(1)
                    break

    if last_fp == fingerprint:
        # Same JSON, same mtime — already logged. Skip.
        return

    def trunc(s, n):
        s = (s or "").replace("\n", " ").strip()
        return (s[: n - 1] + "…") if len(s) > n else s

    row = (
        f"| {next_idx} "
        f"| {json_mtime_iso} "
        f"| {data.get('run_label', '?')} "
        f"| {data.get('format', '?')} "
        f"| {data.get('denoise', '?')} "
        f"| `{json_path.name}` "
        f"| `{data.get('render_output', '—')}` "
        f"| {trunc(data.get('note', ''), 40)} "
        f"| {trunc(data.get('feedback', ''), 40)} "
        f"|<!--fp:{fingerprint}-->\n"
    )

    if not existing:
        log_path.write_text(LOG_HEADER + row, encoding="utf-8")
    else:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(row)


def load_workflow(path: Path) -> dict:
    """Load a ComfyUI workflow JSON. Accepts two formats:
       1. raw editor JSON (top-level 'nodes' + 'links' keys), or
       2. metadata bundle ({"CreationTime": ..., "prompt": ..., "workflow": {...}})
          — the format ComfyUI writes alongside saved outputs.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "nodes" in raw:
        return raw
    if "workflow" in raw and isinstance(raw["workflow"], dict) and "nodes" in raw["workflow"]:
        return raw["workflow"]
    raise SystemExit(f"Unrecognised JSON format in {path} — no 'nodes' key found.")

def find_newest_video(folder: Path):
    """Return (Path, mtime) for the newest video file in folder, or (None, None).
    Skips files marked as loop outputs (see SKIP_MARKER)."""
    candidates = [p for p in folder.iterdir()
                  if p.is_file()
                  and p.suffix.lower() in VIDEO_EXTS
                  and not _is_loop_output(p)]
    if not candidates:
        return None, None
    best = max(candidates, key=lambda p: p.stat().st_mtime)
    return best, best.stat().st_mtime

def fmt_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"

def find(wf, t):
    return [n for n in wf["nodes"] if n.get("type") == t]

def widgets(n): return n.get("widgets_values", []) or []

def resolve_input(wf, node, name):
    """Return the upstream node feeding a named input, or None."""
    links = {link[0]: link for link in wf["links"]}
    by_id = {n["id"]: n for n in wf["nodes"]}
    for inp in node.get("inputs", []) or []:
        if inp.get("name") == name and inp.get("link") is not None:
            link = links.get(inp["link"])
            if link:
                return by_id.get(link[1])
    return None

def follow_setnode(wf, getnode):
    """GetNode -> the value-producing source via SetNode of the same variable."""
    var = widgets(getnode)[0] if widgets(getnode) else None
    if not var:
        return None
    for n in wf["nodes"]:
        if n.get("type") == "SetNode" and widgets(n)[:1] == [var]:
            # SetNode's incoming link is the actual value source
            links = {link[0]: link for link in wf["links"]}
            by_id = {n["id"]: n for n in wf["nodes"]}
            for inp in n.get("inputs", []) or []:
                if inp.get("link") is not None:
                    link = links.get(inp["link"])
                    if link:
                        src = by_id.get(link[1])
                        # If it's a switch, resolve the selected value
                        if src and src.get("type") == "easy anythingIndexSwitch":
                            idx_node = resolve_input(wf, src, "index")
                            idx = widgets(idx_node)[0] if idx_node and widgets(idx_node) else 0
                            val_node = resolve_input(wf, src, f"value{idx}")
                            return val_node
                        return src
    return None

def res_label(w, h):
    return {(3744,2112):"4K", (3840,2160):"4K", (2560,1440):"2K",
            (1920,1080):"HD", (1280,720):"HD"}.get((w,h), f"{w}x{h}")

def extract(wf):
    """Pull every field we need to draw the card."""
    d = find(wf, "LTXDirector")[0]
    dw = widgets(d)
    # LTXDirector widget order (verified in v01-v03):
    #  0 global_prompt, 1 dur_frames, 2 dur_sec, 3 timeline_data, 4 local_prompts,
    #  5 segment_lengths, 6 epsilon, 7 guide_strength, 8 use_custom_audio,
    #  9 frame_rate, 10 display_mode, 11 custom_width, 12 custom_height,
    # 13 resize_method, 14 divisible_by, 15 img_compression
    epsilon         = dw[6]
    seg_lengths_str = dw[5]
    fps             = dw[9]
    use_custom_aud  = dw[8]
    dur_frames      = dw[1]
    dur_seconds     = dw[2]

    # custom_width / height may be widget value OR fed by GetNode chain
    cw = resolve_input(wf, d, "custom_width")
    ch = resolve_input(wf, d, "custom_height")
    if cw and cw.get("type") == "GetNode":
        cw = follow_setnode(wf, cw)
    if ch and ch.get("type") == "GetNode":
        ch = follow_setnode(wf, ch)
    width  = widgets(cw)[0] if cw and widgets(cw) else dw[11]
    height = widgets(ch)[0] if ch and widgets(ch) else dw[12]

    # parse timeline_data segments
    try:
        tl = json.loads(dw[3]) if dw[3] else {}
        segments = tl.get("segments", [])
    except Exception:
        segments = []
    seg_lengths = [int(x.strip()) for x in seg_lengths_str.split(",") if x.strip()]

    # phase 1 chain — first BasicScheduler with denoise == 1.0, else first one
    schedulers = find(wf, "BasicScheduler")
    p1 = next((s for s in schedulers if widgets(s)[2] == 1), schedulers[0] if schedulers else None)
    sched_name, steps, denoise = (widgets(p1) if p1 else ("?", "?", "?"))

    # Director Guide scale_by per phase — pick the one fed by LTXDirector (phase 1)
    guides = find(wf, "LTXDirectorGuide")
    p1_guide = None
    for g in guides:
        latent_src = resolve_input(wf, g, "latent")
        if latent_src and latent_src.get("type") == "LTXDirector":
            p1_guide = g
            break
    if p1_guide is None and guides:
        p1_guide = guides[0]
    mult = widgets(p1_guide)[0] if p1_guide else "?"

    # Pull three named Text Multiline nodes from the workflow:
    #   - "OVERLAY - INFO" (or any title containing INFO)       -> base notes (Δ from baseline)
    #   - "OVERLAY - FEEDBACK" / "FEEDBACK" / "OBSERVATIONS"     -> video feedback + scores
    #   - "Working_Folder_PATH" / "OUT_working_folder"           -> authoritative output folder
    note, feedback, work_folder = "", "", ""
    for n in wf["nodes"]:
        if n.get("type") != "Text Multiline":
            continue
        title = (n.get("title") or "").upper()
        ws = widgets(n)
        if not (ws and isinstance(ws[0], str)):
            continue
        text = ws[0]
        if not note and ("INFO" in title or ("OVERLAY" in title and "FEEDBACK" not in title)):
            for marker in ("BASE (notes):", "BASE:"):
                if marker in text:
                    note = text.split(marker, 1)[1].strip()
                    break
        if not feedback and ("FEEDBACK" in title or "OBSERVATION" in title or "VIDEO" in title):
            feedback = text.strip()
        if not work_folder and ("WORKING_FOLDER" in title or "WORK_FOLDER" in title
                                 or "OUT_WORKING" in title or "WORKING-FOLDER" in title):
            work_folder = text.strip().strip("\"' ").rstrip("\\/")

    # Parse scores out of the feedback text. Pattern: "motion: 4/5" / "sync 3" / "sharpness: 5/5".
    scores = {"motion": None, "sync": None, "sharp": None}
    feedback_lines = []
    score_re = re.compile(
        r"^\s*(motion|sync|sharp(?:ness)?|sharpness)\s*[:=]?\s*(\d+)\s*(?:/\s*\d+)?\s*$",
        re.IGNORECASE,
    )
    for line in feedback.splitlines():
        m = score_re.match(line)
        if m:
            key = m.group(1).lower()
            if key.startswith("sharp"):
                key = "sharp"
            scores[key] = int(m.group(2))
        elif line.strip():
            feedback_lines.append(line.rstrip())
    feedback = "\n".join(feedback_lines).strip()

    # model stack
    ckpt = (widgets(find(wf, "CheckpointLoaderSimple")[0]) if find(wf, "CheckpointLoaderSimple") else [""])[0]
    lora_nodes = find(wf, "LoraLoaderModelOnly")
    lora = ""
    if lora_nodes:
        lw = widgets(lora_nodes[0])
        lora_name = lw[0].replace("\\", "/").split("/")[-1].replace(".safetensors", "")
        lora_strength = lw[1] if len(lw) > 1 else ""
        lora = f"{lora_name} @ {lora_strength}"
    clip_nodes = find(wf, "DualCLIPLoader")
    clip = ""
    if clip_nodes:
        cw_ = widgets(clip_nodes[0])
        clip = cw_[0].replace("\\", "/").split("/")[-1].replace(".safetensors", "")
    vaes = find(wf, "VAELoaderKJ")
    video_vae = audio_vae = ""
    for v in vaes:
        n = widgets(v)[0].replace(".safetensors", "")
        if "video" in n:
            video_vae = n
        elif "audio" in n:
            audio_vae = n

    # seed
    rn = find(wf, "RandomNoise")
    seed = widgets(rn[0])[0] if rn else "?"

    return {
        "format": res_label(width, height),
        "resolution": f"{width} × {height}",
        "mult": mult,
        "scheduler": sched_name,
        "steps": steps,
        "denoise": denoise,
        "note": note,
        "feedback": feedback,
        "scores": scores,
        "work_folder_from_wf": work_folder,
        "epsilon": epsilon,
        "seed": seed,
        "fps": fps,
        "dur_frames": dur_frames,
        "dur_seconds": dur_seconds,
        "segments": segments,
        "seg_lengths": seg_lengths,
        "use_custom_audio": use_custom_aud,
        "ckpt": Path(ckpt).name.replace(".safetensors", ""),
        "lora": lora,
        "clip": Path(clip).name,
        "video_vae": video_vae,
        "audio_vae": audio_vae,
    }

# ---------- drawing helpers ----------
def draw_kv_row(draw, x, y, key, val, key_w):
    draw.text((x, y), key, font=F_MONO, fill=MUTED)
    draw.text((x + key_w, y), str(val), font=F_MONO, fill=TEXT)
    return y + 26

def draw_h2(draw, x, y, label, color):
    draw.text((x, y), label.upper(), font=F_H2, fill=color)
    return y + 30

def wrap_text(text, max_chars):
    """Naive word-wrap to fit max_chars per line."""
    out = []
    for raw_line in text.split("\n"):
        if not raw_line.strip():
            out.append("")
            continue
        words, cur = raw_line.split(), ""
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

def draw_text_box(draw, x, y, width, label, content, accent, max_lines=3):
    """Draw a left-accented note box with a header label + wrapped content lines."""
    char_per_line = 42
    lines = wrap_text(content, char_per_line)[:max_lines]
    box_h = 26 + 22 * max(1, len(lines)) + 14
    draw.rounded_rectangle(
        [x - 12, y, x + width, y + box_h],
        radius=4, fill=NOTE_BG,
    )
    draw.rectangle([x - 12, y, x - 8, y + box_h], fill=accent)
    draw.text((x + 4, y + 8), label, font=F_H2, fill=accent)
    line_y = y + 34
    for line in lines:
        draw.text((x + 4, line_y), line, font=F_NOTE, fill=DIM)
        line_y += 22
    return y + box_h + 4

def draw_section(draw, x, y, width, accent, label, body_lines_height,
                  bg=BG_SECTION, border=BORDER):
    """Draw a section panel header + return (content_x, content_y, content_w, end_y).
    Body content is drawn by the caller starting at (content_x, content_y)."""
    HEADER_H = 36
    section_h = HEADER_H + body_lines_height + 14
    draw.rounded_rectangle(
        [x, y, x + width, y + section_h],
        radius=SECTION_RADIUS, fill=bg, outline=border, width=1,
    )
    # header label — bigger, no side bar, accent colour
    draw.text((x + 14, y + 10), label.upper(), font=F_SECTION, fill=accent)
    return x + 14, y + HEADER_H, width - 28, y + section_h + 10

def section_body_rows(num_rows, extra=0):
    """Pixel height needed for N kv rows plus optional extra padding."""
    return num_rows * 26 + extra

def render(data: dict, out_path: Path) -> None:
    # Oversized canvas; crop after laying out.
    canvas_h = 2000
    img = Image.new("RGB", (W, canvas_h), BG_OUTER)
    draw = ImageDraw.Draw(img)

    inset = 18
    # outer card
    draw.rounded_rectangle(
        [inset, inset, W - inset, canvas_h - inset],
        radius=RADIUS, fill=BG_CARD, outline=BORDER, width=1,
    )

    x = inset + PAD_X
    y = inset + PAD_TOP
    inner_w = W - 2 * inset - 2 * PAD_X

    # ---------- HEADER ----------
    import datetime
    today = datetime.date.today().isoformat()
    ver = data.get("run_label", "?")
    stage_label = "single-stage" if data.get("denoise", 1) == 1 else "two-stage"
    res_label = data.get("format", "")
    title_line = f"Run {ver or '?'} — {stage_label} {res_label}".strip()
    sub_line = f"{today} · {data.get('json_name', '')}"
    draw.text((x, y), title_line, font=F_TITLE, fill=TEXT)
    y += 38
    draw.text((x, y), sub_line, font=F_SUB, fill=MUTED)
    y += 30
    # hairline divider under header
    draw.line([(x, y), (x + inner_w, y)], fill=BORDER, width=1)
    y += 18

    key_w = 122

    # ---------- PHASE 1 ----------
    body_h = section_body_rows(5)
    cx, cy, cw, end_y = draw_section(draw, x, y, inner_w, ACCENT_RUN, "Phase 1", body_h)
    cy = draw_kv_row(draw, cx, cy, "Format",    f"{data['format']} · {data['resolution']}", key_w)
    cy = draw_kv_row(draw, cx, cy, "Mult",      data["mult"],       key_w)
    cy = draw_kv_row(draw, cx, cy, "Scheduler", data["scheduler"],  key_w)
    cy = draw_kv_row(draw, cx, cy, "Steps",     data["steps"],      key_w)
    cy = draw_kv_row(draw, cx, cy, "Denoise",   data["denoise"],    key_w)
    y = end_y

    # ---------- BASE NOTES ----------
    note_text = data["note"] or "(no changes)"
    note_lines = wrap_text(note_text, 42)[:4]
    body_h = max(28, 24 * len(note_lines)) + 4
    cx, cy, cw, end_y = draw_section(draw, x, y, inner_w, ACCENT_RUN,
                                       "Base · notes (Δ this run)", body_h)
    for line in note_lines:
        draw.text((cx, cy), line, font=F_NOTE, fill=DIM)
        cy += 24
    y = end_y

    # ---------- BASE · LOCKED ----------
    body_h = section_body_rows(5)
    cx, cy, cw, end_y = draw_section(draw, x, y, inner_w, ACCENT_BASE, "Base · locked", body_h)
    cy = draw_kv_row(draw, cx, cy, "Sampler",   "euler",                        key_w)
    cy = draw_kv_row(draw, cx, cy, "CFG",       "1.0",                          key_w)
    cy = draw_kv_row(draw, cx, cy, "Seed",      f"{data['seed']} (fixed)",       key_w)
    cy = draw_kv_row(draw, cx, cy, "epsilon",   data["epsilon"],                 key_w)
    cy = draw_kv_row(draw, cx, cy, "Audio src", "model-gen"
                    if not data["use_custom_audio"] else "custom",               key_w)
    y = end_y

    # ---------- BASE · SCENE ----------
    seg_rows = min(len(data["segments"]), 6)
    body_h = section_body_rows(2 + seg_rows, extra=4)
    cx, cy, cw, end_y = draw_section(draw, x, y, inner_w, ACCENT_BASE, "Base · scene", body_h)
    cy = draw_kv_row(draw, cx, cy, "Duration",
                    f"{data['dur_frames']} f · {data['dur_seconds']} s @ {data['fps']} fps", key_w)
    cy = draw_kv_row(draw, cx, cy, "Segments",
                    f"{len(data['segments'])} × {data['seg_lengths'][0] if data['seg_lengths'] else '?'} f", key_w)
    for i, seg in enumerate(data["segments"][:seg_rows]):
        start_s = seg.get("start", 0) / data["fps"]
        end_s = (seg.get("start", 0) + seg.get("length", 0)) / data["fps"]
        has_prompt = "[x]" if seg.get("prompt") else "[ ]"
        has_audio  = "[x]" if "Audio:" in (seg.get("prompt") or "") else "[ ]"
        has_kf     = "[x]" if seg.get("imageFile") else "[ ]"
        all_set = (has_prompt == has_audio == has_kf == "[x]")
        fill_col = ACCENT_OUT if all_set else TEXT
        draw.text((cx, cy), f"{i+1})", font=F_MONO, fill=ACCENT_BASE)
        draw.text((cx + 32, cy), f"{start_s:.0f}-{end_s:.0f}s", font=F_MONO, fill=DIM)
        draw.text((cx + 110, cy), f"P{has_prompt}  A{has_audio}  K{has_kf}",
                  font=F_MONO, fill=fill_col)
        cy += 26
    y = end_y

    # ---------- POST-RENDER ----------
    feedback_text = data["feedback"] or "(add observations)"
    feedback_lines = wrap_text(feedback_text, 42)[:5]
    body_h = (
        section_body_rows(2)              # render time + output
        + 8                                # spacer
        + 20                               # FEEDBACK sublabel
        + max(24, 22 * len(feedback_lines))
        + 14                               # spacer before outcome
        + 20                               # OUTCOME sublabel
        + 26                               # outcome scores row
    )
    cx, cy, cw, end_y = draw_section(draw, x, y, inner_w, ACCENT_OUT, "Post-render", body_h)
    cy = draw_kv_row(draw, cx, cy, "Render time", data.get("render_duration", "?"), key_w)
    cy = draw_kv_row(draw, cx, cy, "Output",      data.get("render_output", "?")[:36], key_w)
    cy += 8

    # FEEDBACK sublabel + body
    draw.text((cx, cy), "FEEDBACK", font=F_TAG, fill=ACCENT_OUT)
    cy += 20
    for line in feedback_lines:
        draw.text((cx, cy), line, font=F_NOTE, fill=DIM)
        cy += 22
    cy += 6

    # OUTCOME sublabel + scores (auto-filled from FEEDBACK parsing)
    draw.text((cx, cy), "OUTCOME", font=F_TAG, fill=ACCENT_OUT)
    cy += 20
    s = data.get("scores", {}) or {}
    def _s(v): return f"{v}/5" if v is not None else "?/5"
    draw.text((cx, cy),
              f"Motion {_s(s.get('motion'))}     Sync {_s(s.get('sync'))}     Sharp {_s(s.get('sharp'))}",
              font=F_MONO, fill=TEXT)
    y = end_y

    # ---------- final crop ----------
    final_h = y + PAD_BOTTOM + inset
    final = Image.new("RGB", (W, final_h), BG_OUTER)
    draw2 = ImageDraw.Draw(final)
    draw2.rounded_rectangle(
        [inset, inset, W - inset, final_h - inset],
        radius=RADIUS, fill=BG_CARD, outline=BORDER, width=1,
    )
    content = img.crop((inset + 1, inset + 1, W - inset - 1, final_h - inset - 1))
    final.paste(content, (inset + 1, inset + 1))
    final.save(out_path)
    print(f"wrote {out_path} ({W}×{final_h})")

# ---------- CLI ----------
def _autodiscover_json() -> Path:
    """Find the newest *.json in $KOLOOK_AUTOMATIONS_WORK_DIR. Fails clearly."""
    folder = os.environ.get("KOLOOK_AUTOMATIONS_WORK_DIR", "").strip()
    if not folder:
        raise SystemExit(
            "KOLOOK_AUTOMATIONS_WORK_DIR not set. Add it to `.env` (see `.env.example`) "
            "or pass the JSON path explicitly: `make_card.py <workflow.json>`."
        )
    p = Path(folder)
    if not p.is_dir():
        raise SystemExit(f"KOLOOK_AUTOMATIONS_WORK_DIR does not exist on disk: {p}")
    candidates = sorted(
        (j for j in p.glob("*.json") if not _is_loop_output(j)),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(f"No *.json files found in {p}. Export a workflow from ComfyUI first.")
    return candidates[0]


def main():
    if len(sys.argv) >= 2:
        wf_path = Path(sys.argv[1])
    else:
        wf_path = _autodiscover_json()
        print(f"auto-discovered: {wf_path}")
    wf = load_workflow(wf_path)
    data = extract(wf)
    data["json_name"] = wf_path.name
    data["json_stem"] = wf_path.stem

    # Output folder priority (first existing wins):
    #   1. Working_Folder_PATH Text Multiline inside the workflow JSON.
    #   2. KOLOOK_AUTOMATIONS_WORK_DIR env var (from .env).
    #   3. The JSON's own parent directory.
    candidates = []
    wf_folder = data.get("work_folder_from_wf") or ""
    if wf_folder:
        candidates.append(("Working_Folder_PATH node", Path(wf_folder)))
    env_folder = os.environ.get("KOLOOK_AUTOMATIONS_WORK_DIR", "").strip()
    if env_folder:
        candidates.append(("KOLOOK_AUTOMATIONS_WORK_DIR", Path(env_folder)))
    candidates.append(("JSON parent directory", wf_path.parent))

    folder = None
    for label, p in candidates:
        try:
            p = p.resolve()
        except Exception:
            continue
        if p.is_dir():
            folder = p
            break
    if folder is None:
        # Last-resort fallback — use the unresolved JSON parent.
        folder = wf_path.parent

    # Output path: stable filename in the _AI subfolder of the working folder.
    # Keeps managed artifacts (card.png + iterations.md) separated from user
    # content (JSONs, MP4s) so the maintainer can wipe one without the other.
    if len(sys.argv) > 2:
        out_path = Path(sys.argv[2])
    else:
        ai_dir = folder / AI_SUBDIR
        ai_dir.mkdir(exist_ok=True)
        out_path = ai_dir / CARD_NAME
    video, vmtime = find_newest_video(folder)
    if video is not None:
        # Try same-basename match first; otherwise fall back to "newest video newer than JSON".
        same_stem = video.stem.startswith(wf_path.stem) or wf_path.stem.startswith(video.stem)
        delta = vmtime - wf_path.stat().st_mtime
        if same_stem:
            # ComfyUI saves both files at end-of-render, so timestamps don't bracket the duration.
            # Show "—" — render time is unknowable from filesystem alone.
            data["render_duration"] = "—"
        elif delta >= 0:
            data["render_duration"] = fmt_duration(delta)
        else:
            data["render_duration"] = "pending"
        data["render_output"] = video.name
    else:
        data["render_duration"] = "pending"
        data["render_output"] = "(no video yet)"

    # run_label: prefer v01/v04 style, else fall back to ComfyUI sequence number
    stem = wf_path.stem
    m = re.search(r"(?:^|[_\-])v(\d{1,4})(?:[_\-]|$)", stem)
    if m:
        data["run_label"] = f"v{m.group(1)}"
    else:
        m = re.search(r"_(\d{4,6})(?:_|$)", stem)
        data["run_label"] = f"#{int(m.group(1))}" if m else "?"

    render(data, out_path)
    log_iteration(folder, wf_path, data)

if __name__ == "__main__":
    main()
