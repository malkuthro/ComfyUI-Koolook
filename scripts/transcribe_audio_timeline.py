#!/usr/bin/env python3
"""Build timed LTXDirector prompt segments from a speech audio file.

The audio-lipsync loop uses this as a bridge between raw custom audio and
Prompt Relay: transcribe the line, group timestamped words into short phrases,
and emit Director-shaped ``timeline_data`` / ``local_prompts`` /
``segment_lengths`` fields that can be pasted into, or later applied to, a
workflow JSON.

Usage:
  python scripts/transcribe_audio_timeline.py <audio.mp3>
  python scripts/transcribe_audio_timeline.py <audio.mp3> --workflow workflow.json
  python scripts/transcribe_audio_timeline.py <audio.mp3> --out timed.json

Requires the optional audio extra:
  python -m pip install -e ".[audio]"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from k_audio_timeline import (
    DEFAULT_PAUSE_TEMPLATE,
    DEFAULT_PROMPT_TEMPLATE,
    Phrase,
    Word,
    group_words,
    transcribe_words,
)


def _frame(value: float, fps: float) -> int:
    return max(0, int(round(value * fps)))


def load_director_timeline(workflow_path: Path) -> tuple[dict[str, Any], int, float]:
    data = json.loads(workflow_path.read_text(encoding="utf-8"))
    for node in data.get("nodes", []):
        if str(node.get("type", "")).startswith("LTXDirector__koolook"):
            values = node.get("widgets_values") or []
            timeline = json.loads(values[3]) if len(values) > 3 and values[3] else {}
            duration_frames = int(values[5]) if len(values) > 5 else 0
            fps = float(values[9]) if len(values) > 9 else 24.0
            return timeline, duration_frames, fps
    raise SystemExit(f"No Koolook LTXDirector node found in {workflow_path}")


def patch_workflow(
    workflow_path: Path,
    out_path: Path,
    *,
    timeline_data: dict[str, Any],
    local_prompts: str,
    segment_lengths: str,
) -> None:
    data = json.loads(workflow_path.read_text(encoding="utf-8"))
    for node in data.get("nodes", []):
        if str(node.get("type", "")).startswith("LTXDirector__koolook"):
            values = node.setdefault("widgets_values", [])
            if len(values) < 6:
                values.extend([""] * (6 - len(values)))
            values[3] = json.dumps(timeline_data, ensure_ascii=False, separators=(",", ":"))
            values[4] = local_prompts
            values[5] = segment_lengths
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            return
    raise SystemExit(f"No Koolook LTXDirector node found in {workflow_path}")


def build_prompt_spans(
    phrases: list[Phrase],
    *,
    fps: float,
    duration_frames: int,
    prompt_template: str,
    pause_template: str,
    pause_threshold_seconds: float,
) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for phrase in phrases:
        start = _frame(phrase.start, fps)
        end = max(start + 1, _frame(phrase.end, fps))
        if start - cursor >= _frame(pause_threshold_seconds, fps):
            spans.append((cursor, start, pause_template))
        elif start > cursor and spans:
            prev_start, _prev_end, prev_prompt = spans[-1]
            spans[-1] = (prev_start, start, prev_prompt)
        spans.append((start, end, prompt_template.format(text=phrase.text)))
        cursor = end

    if duration_frames > cursor:
        spans.append((cursor, duration_frames, pause_template))

    return [(start, max(start + 1, end), prompt) for start, end, prompt in spans]


def build_timeline(
    phrases: list[Phrase],
    *,
    fps: float,
    duration_frames: int,
    base_timeline: dict[str, Any] | None,
    prompt_template: str,
    pause_template: str,
    pause_threshold_seconds: float,
) -> dict[str, Any]:
    base_timeline = base_timeline or {}
    source_segments = base_timeline.get("segments") or []
    source = source_segments[0] if source_segments else {}
    audio_segments = base_timeline.get("audioSegments") or []

    spans = build_prompt_spans(
        phrases,
        fps=fps,
        duration_frames=duration_frames,
        prompt_template=prompt_template,
        pause_template=pause_template,
        pause_threshold_seconds=pause_threshold_seconds,
    )

    segments: list[dict[str, Any]] = []
    for idx, (start, end, prompt) in enumerate(spans):
        seg = {
            "id": f"speech_{idx + 1:03d}",
            "start": start,
            "length": end - start,
            "prompt": prompt,
            "type": source.get("type", "image"),
        }
        for key in ("imageFile", "imageB64"):
            if key in source:
                seg[key] = source[key]
        segments.append(seg)

    return {
        **base_timeline,
        "segments": segments,
        "audioSegments": audio_segments,
    }


def output_payload(
    phrases: list[Phrase],
    *,
    fps: float,
    timeline_data: dict[str, Any],
) -> dict[str, Any]:
    local_prompts = [seg["prompt"] for seg in timeline_data["segments"]]
    segment_lengths = [str(seg["length"]) for seg in timeline_data["segments"]]
    return {
        "fps": fps,
        "phrases": [
            {"start": p.start, "end": p.end, "text": p.text}
            for p in phrases
        ],
        "local_prompts": " | ".join(local_prompts),
        "segment_lengths": ",".join(segment_lengths),
        "timeline_data": timeline_data,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--workflow", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--patched-workflow", type=Path)
    parser.add_argument("--model", default="base.en")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--max-phrase-seconds", type=float, default=1.2)
    parser.add_argument("--max-words", type=int, default=6)
    parser.add_argument("--gap-seconds", type=float, default=0.45)
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--pause-threshold-seconds", type=float, default=0.2)
    parser.add_argument("--pause-template", default=DEFAULT_PAUSE_TEMPLATE)
    args = parser.parse_args(argv)

    if not args.audio.is_file():
        print(f"Audio file not found: {args.audio}", file=sys.stderr)
        return 2

    base_timeline = None
    duration_frames = 0
    fps = args.fps
    if args.workflow:
        base_timeline, duration_frames, fps = load_director_timeline(args.workflow)

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    words = transcribe_words(args.audio, args.model, device)
    phrases = group_words(
        words,
        max_phrase_seconds=args.max_phrase_seconds,
        max_words=args.max_words,
        gap_seconds=args.gap_seconds,
    )
    timeline_data = build_timeline(
        phrases,
        fps=fps,
        duration_frames=duration_frames or _frame(phrases[-1].end, fps),
        base_timeline=base_timeline,
        prompt_template=args.prompt_template,
        pause_template=args.pause_template,
        pause_threshold_seconds=args.pause_threshold_seconds,
    )
    payload = output_payload(phrases, fps=fps, timeline_data=timeline_data)
    text = json.dumps(payload, indent=2, ensure_ascii=False)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote: {args.out}")
    else:
        print(text)
    if args.patched_workflow:
        if not args.workflow:
            print("--patched-workflow requires --workflow", file=sys.stderr)
            return 2
        patch_workflow(
            args.workflow,
            args.patched_workflow,
            timeline_data=timeline_data,
            local_prompts=payload["local_prompts"],
            segment_lengths=payload["segment_lengths"],
        )
        print(f"patched workflow: {args.patched_workflow}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
