# SPDX-License-Identifier: GPL-3.0-or-later
#
# ComfyUI-Koolook - Audio transcript timeline
# Copyright (C) 2026 ComfyUI-Koolook contributors (kforgelabs).
"""Comfy node that turns speech audio into timed LTXDirector prompts."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

try:
    import folder_paths  # ComfyUI core; present at runtime.
except ImportError:  # pragma: no cover - tests run outside ComfyUI.
    folder_paths = None  # type: ignore[assignment]


DEFAULT_PROMPT_TEMPLATE = (
    '[Audio timing, frames {start_frame:03d}-{end_frame:03d}, '
    '{start_seconds:.2f}-{end_seconds:.2f}s]: '
    'The bear says exactly "{text}". '
    "The mouth, jaw, and lips form the spoken words within this frame range."
)
DEFAULT_PAUSE_TEMPLATE = (
    "[Audio timing, frames {start_frame:03d}-{end_frame:03d}, "
    "{start_seconds:.2f}-{end_seconds:.2f}s]: "
    "Silence/reaction pause. The bear does not speak; mouth closes naturally "
    "while the face stays alive and attentive."
)


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Phrase:
    start: float
    end: float
    text: str


def _input_root() -> Path:
    if folder_paths is not None:
        return Path(folder_paths.get_input_directory())
    return Path.cwd()


def _resolve_input_file(name_or_path: str) -> Path:
    text = str(name_or_path or "").strip().strip('"').strip("'")
    path = Path(text)
    if path.is_absolute():
        return path
    return _input_root() / text


def _clean_word(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _phrase_text(words: list[Word]) -> str:
    text = " ".join(w.text for w in words)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _frame(value: float, fps: float) -> int:
    return max(0, int(round(value * fps)))


def _timeline_defaults(timeline_data: str) -> dict:
    if not str(timeline_data or "").strip():
        return {}
    data = json.loads(timeline_data)
    if not isinstance(data, dict):
        return {}
    first_image = next(
        (
            seg
            for seg in data.get("segments", [])
            if isinstance(seg, dict) and seg.get("imageFile")
        ),
        {},
    )
    first_audio = next(
        (
            seg
            for seg in data.get("audioSegments", [])
            if isinstance(seg, dict) and seg.get("audioFile")
        ),
        {},
    )
    return {
        "image_file": first_image.get("imageFile", ""),
        "audio_file": first_audio.get("audioFile", ""),
        "audio_start_frames": int(float(first_audio.get("start", 0) or 0)),
        "audio_trim_start_frames": int(float(first_audio.get("trimStart", 0) or 0)),
        "audio_length_frames": int(float(first_audio.get("length", 0) or 0)),
    }


def shift_phrases_to_timeline(
    phrases: list[Phrase],
    *,
    frame_rate: float,
    audio_start_frames: int,
    audio_trim_start_frames: int,
    duration_frames: int,
) -> list[Phrase]:
    offset_seconds = (audio_start_frames - audio_trim_start_frames) / frame_rate
    shifted: list[Phrase] = []
    for phrase in phrases:
        start = phrase.start + offset_seconds
        end = phrase.end + offset_seconds
        if end <= 0 or start >= duration_frames / frame_rate:
            continue
        shifted.append(Phrase(max(0.0, start), max(0.0, end), phrase.text))
    return shifted


def _format_prompt(
    template: str,
    *,
    text: str = "",
    start: int,
    end: int,
    fps: float,
) -> str:
    return template.format(
        text=text,
        start_frame=start,
        end_frame=max(start, end - 1),
        end_exclusive_frame=end,
        length_frames=max(1, end - start),
        start_seconds=start / fps,
        end_seconds=end / fps,
    )


def transcribe_words(audio_path: Path, model_size_or_path: str, device: str) -> list[Word]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Install Koolook's optional audio "
            'tools with: python -m pip install -e ".[audio]"'
        ) from exc

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size_or_path, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        vad_filter=False,
    )

    words: list[Word] = []
    for seg in segments:
        for raw in seg.words or []:
            text = _clean_word(raw.word)
            if text:
                words.append(Word(float(raw.start), float(raw.end), text))
    return words


def group_words(
    words: list[Word],
    *,
    max_phrase_seconds: float,
    max_words: int,
    gap_seconds: float,
) -> list[Phrase]:
    phrases: list[Phrase] = []
    current: list[Word] = []

    def flush() -> None:
        nonlocal current
        if current:
            phrases.append(Phrase(current[0].start, current[-1].end, _phrase_text(current)))
            current = []

    for word in words:
        if not current:
            current = [word]
            continue
        gap = word.start - current[-1].end
        span = word.end - current[0].start
        punctuation_break = current[-1].text.endswith((".", "!", "?"))
        if (
            gap >= gap_seconds
            or span > max_phrase_seconds
            or len(current) >= max_words
            or punctuation_break
        ):
            flush()
        current.append(word)

    flush()
    return phrases


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
            spans.append((
                cursor,
                start,
                _format_prompt(pause_template, start=cursor, end=start, fps=fps),
            ))
        elif start > cursor and spans:
            prev_start, _prev_end, prev_prompt = spans[-1]
            spans[-1] = (prev_start, start, prev_prompt)
        spans.append((
            start,
            end,
            _format_prompt(
                prompt_template,
                text=phrase.text,
                start=start,
                end=end,
                fps=fps,
            ),
        ))
        cursor = end

    if duration_frames > cursor:
        spans.append((
            cursor,
            duration_frames,
            _format_prompt(
                pause_template,
                start=cursor,
                end=duration_frames,
                fps=fps,
            ),
        ))

    return [(start, max(start + 1, end), prompt) for start, end, prompt in spans]


def build_timeline_data(
    phrases: list[Phrase],
    *,
    image_file: str,
    audio_file: str,
    fps: float,
    duration_frames: int,
    audio_length_frames: int,
    prompt_template: str,
    pause_template: str,
    pause_threshold_seconds: float,
) -> dict:
    spans = build_prompt_spans(
        phrases,
        fps=fps,
        duration_frames=duration_frames,
        prompt_template=prompt_template,
        pause_template=pause_template,
        pause_threshold_seconds=pause_threshold_seconds,
    )
    segments = [
        {
            "id": f"speech_{idx + 1:03d}",
            "start": start,
            "length": end - start,
            "prompt": prompt,
            "type": "image",
            "imageFile": image_file,
        }
        for idx, (start, end, prompt) in enumerate(spans)
    ]
    audio_len = audio_length_frames if audio_length_frames > 0 else duration_frames
    audio_segments = [
        {
            "id": "audio_001",
            "type": "audio",
            "start": 0,
            "length": audio_len,
            "trimStart": 0,
            "audioDurationFrames": audio_len,
            "audioFile": audio_file,
            "fileName": os.path.basename(audio_file),
        }
    ]
    return {"segments": segments, "audioSegments": audio_segments}


class KoolookAudioTranscriptTimeline:
    """Generate LTXDirector timeline strings from a speech audio file."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_file": ("STRING", {"default": "", "multiline": True, "tooltip": "Audio filename in ComfyUI input/, or an absolute path."}),
                "image_file": ("STRING", {"default": "", "tooltip": "Source image filename in ComfyUI input/."}),
                "duration_frames": ("INT", {"default": 120, "min": 1, "max": 10000, "step": 1}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 1.0}),
                "audio_length_frames": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "model_size_or_path": ("STRING", {"default": "base.en", "multiline": True, "tooltip": "faster-whisper model name, or a local model folder."}),
                "device": (["cpu", "cuda"], {"default": "cpu"}),
                "prompt_template": ("STRING", {"default": DEFAULT_PROMPT_TEMPLATE, "multiline": True}),
                "pause_template": ("STRING", {"default": DEFAULT_PAUSE_TEMPLATE, "multiline": True}),
                "max_phrase_seconds": ("FLOAT", {"default": 1.2, "min": 0.2, "max": 10.0, "step": 0.1}),
                "max_words": ("INT", {"default": 6, "min": 1, "max": 30, "step": 1}),
                "gap_seconds": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 5.0, "step": 0.05}),
                "pause_threshold_seconds": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 5.0, "step": 0.05}),
            },
            "optional": {
                "timeline_data": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional timeline JSON from Koolook Timeline Editor. When set, the first audio/image segment fills audio_file, image_file, trim, and timeline offset."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("timeline_data", "local_prompts", "segment_lengths", "transcript_json")
    CATEGORY = "Koolook/Audio"
    FUNCTION = "run"

    def run(
        self,
        audio_file: str,
        image_file: str,
        duration_frames: int,
        frame_rate: float,
        audio_length_frames: int,
        model_size_or_path: str,
        device: str,
        prompt_template: str,
        pause_template: str,
        max_phrase_seconds: float,
        max_words: int,
        gap_seconds: float,
        pause_threshold_seconds: float,
        timeline_data: str = "",
    ):
        timeline = _timeline_defaults(timeline_data)
        if timeline.get("audio_file"):
            audio_file = timeline["audio_file"]
        if timeline.get("image_file"):
            image_file = timeline["image_file"]
        if timeline.get("audio_length_frames"):
            audio_length_frames = timeline["audio_length_frames"]

        audio_path = _resolve_input_file(audio_file)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        words = transcribe_words(audio_path, model_size_or_path, device)
        phrases = group_words(
            words,
            max_phrase_seconds=float(max_phrase_seconds),
            max_words=int(max_words),
            gap_seconds=float(gap_seconds),
        )
        phrases = shift_phrases_to_timeline(
            phrases,
            frame_rate=float(frame_rate),
            audio_start_frames=int(timeline.get("audio_start_frames", 0)),
            audio_trim_start_frames=int(timeline.get("audio_trim_start_frames", 0)),
            duration_frames=int(duration_frames),
        )
        timeline = build_timeline_data(
            phrases,
            image_file=image_file,
            audio_file=audio_file,
            fps=float(frame_rate),
            duration_frames=int(duration_frames),
            audio_length_frames=int(audio_length_frames),
            prompt_template=prompt_template,
            pause_template=pause_template,
            pause_threshold_seconds=float(pause_threshold_seconds),
        )
        local_prompts = " | ".join(seg["prompt"] for seg in timeline["segments"])
        segment_lengths = ",".join(str(seg["length"]) for seg in timeline["segments"])
        transcript = {
            "phrases": [
                {"start": p.start, "end": p.end, "text": p.text}
                for p in phrases
            ],
        }
        return (
            json.dumps(timeline, ensure_ascii=False, separators=(",", ":")),
            local_prompts,
            segment_lengths,
            json.dumps(transcript, ensure_ascii=False, indent=2),
        )


NODE_CLASS_MAPPINGS = {"KoolookAudioTranscriptTimeline": KoolookAudioTranscriptTimeline}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KoolookAudioTranscriptTimeline": "Koolook Audio Transcript Timeline"
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "Word",
    "Phrase",
    "build_prompt_spans",
    "build_timeline_data",
    "group_words",
]
