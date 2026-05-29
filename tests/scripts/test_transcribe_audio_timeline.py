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
