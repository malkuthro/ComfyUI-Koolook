"""Pure helper coverage for audio transcript timeline tooling."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import k_audio_timeline


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "transcribe_audio_timeline.py"
_spec = importlib.util.spec_from_file_location("transcribe_audio_timeline", _SCRIPT)
transcribe_audio_timeline = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules[_spec.name] = transcribe_audio_timeline
_spec.loader.exec_module(transcribe_audio_timeline)


Word = k_audio_timeline.Word
Phrase = k_audio_timeline.Phrase


def test_group_words_splits_on_gap_and_punctuation():
    words = [
        Word(0.0, 0.2, "This"),
        Word(0.2, 0.4, "works."),
        Word(0.5, 0.7, "Next"),
        Word(1.4, 1.6, "line"),
    ]

    phrases = k_audio_timeline.group_words(
        words,
        max_phrase_seconds=2.0,
        max_words=6,
        gap_seconds=0.45,
    )

    assert [p.text for p in phrases] == ["This works.", "Next", "line"]


def test_build_prompt_spans_includes_silence_gaps_and_tail():
    phrases = [
        Phrase(0.0, 0.9, "This can't be real."),
        Phrase(1.44, 2.06, "It's made by AI."),
    ]

    spans = k_audio_timeline.build_prompt_spans(
        phrases,
        fps=24.0,
        duration_frames=120,
        prompt_template='says "{text}"',
        pause_template="mouth closed",
        pause_threshold_seconds=0.2,
    )

    assert spans == [
        (0, 22, 'says "This can\'t be real."'),
        (22, 35, "mouth closed"),
        (35, 49, 'says "It\'s made by AI."'),
        (49, 120, "mouth closed"),
    ]


def test_build_prompt_spans_supports_precise_timing_template_fields():
    phrases = [Phrase(1.0, 1.5, "What?")]

    spans = k_audio_timeline.build_prompt_spans(
        phrases,
        fps=24.0,
        duration_frames=48,
        prompt_template='frames {start_frame:03d}-{end_frame:03d}: "{text}"',
        pause_template="pause {start_seconds:.2f}-{end_seconds:.2f}s",
        pause_threshold_seconds=0.2,
    )

    assert spans == [
        (0, 24, "pause 0.00-1.00s"),
        (24, 36, 'frames 024-035: "What?"'),
        (36, 48, "pause 1.50-2.00s"),
    ]


def test_timeline_data_offsets_transcript_timing(monkeypatch, tmp_path: Path):
    audio = tmp_path / "line.mp3"
    audio.write_bytes(b"fake")
    monkeypatch.setattr(k_audio_timeline, "_resolve_input_file", lambda _name: audio)
    monkeypatch.setattr(
        k_audio_timeline,
        "transcribe_words",
        lambda *_args, **_kwargs: [Word(0.0, 0.5, "What?")],
    )
    timeline_data = json.dumps(
        {
            "segments": [{"imageFile": "bear.png", "start": 0, "length": 120}],
            "audioSegments": [
                {
                    "audioFile": "line.mp3",
                    "start": 24,
                    "trimStart": 0,
                    "length": 65,
                }
            ],
        }
    )

    node = k_audio_timeline.KoolookAudioTranscriptTimeline()
    _timeline_data, local_prompts, segment_lengths, transcript_json = node.run(
        audio_file="ignored.mp3",
        image_file="ignored.png",
        duration_frames=120,
        frame_rate=24.0,
        audio_length_frames=0,
        model_size_or_path="base.en",
        device="cpu",
        prompt_template='frames {start_frame:03d}-{end_frame:03d}: "{text}"',
        pause_template="pause {start_frame:03d}-{end_frame:03d}",
        max_phrase_seconds=1.2,
        max_words=6,
        gap_seconds=0.45,
        pause_threshold_seconds=0.2,
        timeline_data=timeline_data,
    )

    assert 'frames 024-035: "What?"' in local_prompts
    assert segment_lengths == "24,12,84"
    assert json.loads(transcript_json)["phrases"][0]["start"] == 1.0


def test_timeline_data_transcribes_all_separated_audio_segments(monkeypatch, tmp_path: Path):
    for name in ("line_a.mp3", "line_b.mp3"):
        (tmp_path / name).write_bytes(b"fake")

    monkeypatch.setattr(k_audio_timeline, "_resolve_input_file", lambda name: tmp_path / name)

    def fake_transcribe(audio_path: Path, *_args, **_kwargs):
        if audio_path.name == "line_a.mp3":
            return [Word(0.0, 0.5, "First.")]
        return [Word(0.0, 0.5, "Second.")]

    monkeypatch.setattr(k_audio_timeline, "transcribe_words", fake_transcribe)
    timeline_data = json.dumps(
        {
            "segments": [{"imageFile": "bear.png", "start": 0, "length": 120}],
            "audioSegments": [
                {"id": "a", "audioFile": "line_a.mp3", "start": 0, "trimStart": 0, "length": 12},
                {"id": "b", "audioFile": "line_b.mp3", "start": 48, "trimStart": 0, "length": 12},
            ],
        }
    )

    node = k_audio_timeline.KoolookAudioTranscriptTimeline()
    output_timeline, local_prompts, segment_lengths, transcript_json = node.run(
        audio_file="ignored.mp3",
        image_file="ignored.png",
        duration_frames=120,
        frame_rate=24.0,
        audio_length_frames=0,
        model_size_or_path="base.en",
        device="cpu",
        prompt_template='frames {start_frame:03d}-{end_frame:03d}: "{text}"',
        pause_template="pause {start_frame:03d}-{end_frame:03d}",
        max_phrase_seconds=1.2,
        max_words=6,
        gap_seconds=0.45,
        pause_threshold_seconds=0.2,
        timeline_data=timeline_data,
    )

    assert 'frames 000-011: "First."' in local_prompts
    assert 'frames 048-059: "Second."' in local_prompts
    assert segment_lengths == "12,36,12,60"
    assert [item["audioFile"] for item in json.loads(output_timeline)["audioSegments"]] == [
        "line_a.mp3",
        "line_b.mp3",
    ]
    transcript = json.loads(transcript_json)
    assert [phrase["text"] for phrase in transcript["phrases"]] == ["First.", "Second."]
    assert transcript["phrases"][1]["start"] == 2.0


def test_timeline_data_respects_audio_segment_trim_and_length(monkeypatch, tmp_path: Path):
    audio = tmp_path / "line.mp3"
    audio.write_bytes(b"fake")
    monkeypatch.setattr(k_audio_timeline, "_resolve_input_file", lambda _name: audio)
    monkeypatch.setattr(
        k_audio_timeline,
        "transcribe_words",
        lambda *_args, **_kwargs: [
            Word(0.0, 0.4, "Skipped."),
            Word(1.0, 1.5, "Kept."),
            Word(2.1, 2.5, "Past clip."),
        ],
    )
    timeline_data = json.dumps(
        {
            "segments": [{"imageFile": "bear.png", "start": 0, "length": 120}],
            "audioSegments": [
                {
                    "audioFile": "line.mp3",
                    "start": 24,
                    "trimStart": 24,
                    "length": 24,
                }
            ],
        }
    )

    node = k_audio_timeline.KoolookAudioTranscriptTimeline()
    _timeline_data, local_prompts, segment_lengths, transcript_json = node.run(
        audio_file="ignored.mp3",
        image_file="ignored.png",
        duration_frames=120,
        frame_rate=24.0,
        audio_length_frames=0,
        model_size_or_path="base.en",
        device="cpu",
        prompt_template='frames {start_frame:03d}-{end_frame:03d}: "{text}"',
        pause_template="pause {start_frame:03d}-{end_frame:03d}",
        max_phrase_seconds=1.2,
        max_words=6,
        gap_seconds=0.45,
        pause_threshold_seconds=0.2,
        timeline_data=timeline_data,
    )

    assert 'frames 024-035: "Kept."' in local_prompts
    assert "Skipped." not in local_prompts
    assert "Past clip." not in local_prompts
    assert segment_lengths == "24,12,84"
    assert json.loads(transcript_json)["phrases"] == [
        {"start": 1.0, "end": 1.5, "text": "Kept."}
    ]


def test_timeline_data_combines_active_image_prompt_with_audio_timing(monkeypatch, tmp_path: Path):
    audio = tmp_path / "line.mp3"
    audio.write_bytes(b"fake")
    monkeypatch.setattr(k_audio_timeline, "_resolve_input_file", lambda _name: audio)
    monkeypatch.setattr(
        k_audio_timeline,
        "transcribe_words",
        lambda *_args, **_kwargs: [Word(2.0, 2.5, "Hello.")],
    )
    timeline_data = json.dumps(
        {
            "segments": [
                {
                    "imageFile": "bear_a.png",
                    "start": 0,
                    "length": 48,
                    "prompt": "The bear waits in a forest.",
                },
                {
                    "imageFile": "bear_b.png",
                    "start": 48,
                    "length": 72,
                    "prompt": "The bear leans closer to camera.",
                },
            ],
            "audioSegments": [
                {"audioFile": "line.mp3", "start": 0, "trimStart": 0, "length": 120},
            ],
        }
    )

    node = k_audio_timeline.KoolookAudioTranscriptTimeline()
    output_timeline, local_prompts, _segment_lengths, _transcript_json = node.run(
        audio_file="ignored.mp3",
        image_file="ignored.png",
        duration_frames=120,
        frame_rate=24.0,
        audio_length_frames=0,
        model_size_or_path="base.en",
        device="cpu",
        prompt_template='frames {start_frame:03d}-{end_frame:03d}: "{text}"',
        pause_template="pause {start_frame:03d}-{end_frame:03d}",
        max_phrase_seconds=1.2,
        max_words=6,
        gap_seconds=0.45,
        pause_threshold_seconds=0.2,
        timeline_data=timeline_data,
    )

    assert 'The bear waits in a forest. pause 000-047' in local_prompts
    assert 'The bear leans closer to camera. frames 048-059: "Hello."' in local_prompts
    timeline = json.loads(output_timeline)
    assert timeline["segments"][0]["imageFile"] == "bear_a.png"
    assert timeline["segments"][1]["imageFile"] == "bear_b.png"


def test_timeline_data_uses_first_non_empty_image_prompt_as_scene_fallback(monkeypatch, tmp_path: Path):
    audio = tmp_path / "line.mp3"
    audio.write_bytes(b"fake")
    monkeypatch.setattr(k_audio_timeline, "_resolve_input_file", lambda _name: audio)
    monkeypatch.setattr(
        k_audio_timeline,
        "transcribe_words",
        lambda *_args, **_kwargs: [Word(0.0, 0.5, "Hello.")],
    )
    timeline_data = json.dumps(
        {
            "segments": [
                {"imageFile": "bear_a.png", "start": 0, "length": 100, "prompt": ""},
                {"imageFile": "bear_b.png", "start": 100, "length": 20, "prompt": "The bear faces camera."},
            ],
            "audioSegments": [
                {"audioFile": "line.mp3", "start": 0, "trimStart": 0, "length": 120},
            ],
        }
    )

    node = k_audio_timeline.KoolookAudioTranscriptTimeline()
    _output_timeline, local_prompts, _segment_lengths, _transcript_json = node.run(
        audio_file="ignored.mp3",
        image_file="ignored.png",
        duration_frames=120,
        frame_rate=24.0,
        audio_length_frames=0,
        model_size_or_path="base.en",
        device="cpu",
        prompt_template='frames {start_frame:03d}-{end_frame:03d}: "{text}"',
        pause_template="pause {start_frame:03d}-{end_frame:03d}",
        max_phrase_seconds=1.2,
        max_words=6,
        gap_seconds=0.45,
        pause_threshold_seconds=0.2,
        timeline_data=timeline_data,
    )

    assert 'The bear faces camera. frames 000-011: "Hello."' in local_prompts


def test_build_timeline_data_builds_comfy_director_fields():
    phrases = [Phrase(0.0, 0.5, "Hello.")]

    timeline = k_audio_timeline.build_timeline_data(
        phrases,
        image_file="bear.png",
        audio_file="line.mp3",
        fps=24.0,
        duration_frames=24,
        audio_length_frames=13,
        prompt_template='says "{text}"',
        pause_template="mouth closed",
        pause_threshold_seconds=0.2,
    )

    assert timeline["audioSegments"][0]["audioFile"] == "line.mp3"
    assert timeline["audioSegments"][0]["length"] == 13
    assert timeline["segments"][0]["imageFile"] == "bear.png"
    assert timeline["segments"][0]["prompt"] == 'says "Hello."'
    assert timeline["segments"][1]["prompt"] == "mouth closed"


def test_comfy_node_formats_outputs(monkeypatch, tmp_path: Path):
    audio = tmp_path / "line.mp3"
    audio.write_bytes(b"fake")
    monkeypatch.setattr(k_audio_timeline, "_resolve_input_file", lambda _name: audio)
    monkeypatch.setattr(
        k_audio_timeline,
        "transcribe_words",
        lambda *_args, **_kwargs: [
            Word(0.0, 0.9, "This"),
            Word(1.44, 2.06, "works."),
        ],
    )

    node = k_audio_timeline.KoolookAudioTranscriptTimeline()
    timeline_data, local_prompts, segment_lengths, transcript_json = node.run(
        audio_file="line.mp3",
        image_file="bear.png",
        duration_frames=120,
        frame_rate=24.0,
        audio_length_frames=65,
        model_size_or_path="base.en",
        device="cpu",
        prompt_template='bear says "{text}"',
        pause_template="mouth closed",
        max_phrase_seconds=1.2,
        max_words=6,
        gap_seconds=0.45,
        pause_threshold_seconds=0.2,
    )

    timeline = json.loads(timeline_data)
    assert len(timeline["segments"]) == 4
    assert timeline["audioSegments"][0]["length"] == 65
    assert timeline["segments"][0]["prompt"] == 'bear says "This"'
    assert "mouth closed" in local_prompts
    assert segment_lengths == "22,13,14,71"
    assert json.loads(transcript_json)["phrases"][1]["text"] == "works."


def test_patch_workflow_updates_director_fields(tmp_path: Path):
    workflow = tmp_path / "workflow.json"
    patched = tmp_path / "patched.json"
    workflow.write_text(
        '{"nodes":[{"type":"LTXDirector__koolook","widgets_values":["g",1,1,"{}","old","1"]}]}',
        encoding="utf-8",
    )

    transcribe_audio_timeline.patch_workflow(
        workflow,
        patched,
        timeline_data={"segments": [{"prompt": "new"}], "audioSegments": []},
        local_prompts="new",
        segment_lengths="24",
    )

    data = __import__("json").loads(patched.read_text(encoding="utf-8"))
    values = data["nodes"][0]["widgets_values"]
    assert __import__("json").loads(values[3]) == {
        "segments": [{"prompt": "new"}],
        "audioSegments": [],
    }
    assert values[4] == "new"
    assert values[5] == "24"
