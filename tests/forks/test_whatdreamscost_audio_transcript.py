"""Tests for Koolook's Director-level audio transcript hook."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


def _install_import_stubs(monkeypatch):
    class _ComfyNode:
        pass

    class _IoStub(types.SimpleNamespace):
        def Custom(self, name):
            return name

        def __getattr__(self, _name):
            return types.SimpleNamespace(
                Input=lambda *args, **kwargs: None,
                Output=lambda *args, **kwargs: None,
            )

    comfy = types.ModuleType("comfy")
    comfy.model_management = types.ModuleType("comfy.model_management")
    comfy.ldm = types.ModuleType("comfy.ldm")
    comfy.ldm.modules = types.ModuleType("comfy.ldm.modules")
    comfy.ldm.modules.attention = types.ModuleType("comfy.ldm.modules.attention")

    comfy_api = types.ModuleType("comfy_api")
    comfy_api.latest = types.ModuleType("comfy_api.latest")
    comfy_api.latest.io = _IoStub(
        ComfyNode=_ComfyNode,
        NodeOutput=lambda *args: args,
        Schema=lambda *args, **kwargs: None,
    )

    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.float32 = object()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "av", types.ModuleType("av"))
    monkeypatch.setitem(sys.modules, "folder_paths", types.ModuleType("folder_paths"))
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", comfy.model_management)
    monkeypatch.setitem(sys.modules, "comfy.ldm", comfy.ldm)
    monkeypatch.setitem(sys.modules, "comfy.ldm.modules", comfy.ldm.modules)
    monkeypatch.setitem(sys.modules, "comfy.ldm.modules.attention", comfy.ldm.modules.attention)
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", comfy_api.latest)


def _load_director(monkeypatch):
    _install_import_stubs(monkeypatch)
    repo_root = Path(__file__).resolve().parents[2]
    version_dir = repo_root / "forks" / "whatdreamscost_koolook" / "versions" / "v1_3_9"
    package_names = [
        ("forks", repo_root / "forks"),
        ("forks.whatdreamscost_koolook", repo_root / "forks" / "whatdreamscost_koolook"),
        ("forks.whatdreamscost_koolook.versions", repo_root / "forks" / "whatdreamscost_koolook" / "versions"),
        ("forks.whatdreamscost_koolook.versions.v1_3_9", version_dir),
    ]
    for name, path in package_names:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)

    module_name = "forks.whatdreamscost_koolook.versions.v1_3_9.ltx_director"
    spec = importlib.util.spec_from_file_location(module_name, version_dir / "ltx_director.py")
    assert spec and spec.loader
    director = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, director)
    spec.loader.exec_module(director)
    return director


def test_audio_transcript_json_preserves_existing_timeline_segments(monkeypatch):
    director = _load_director(monkeypatch)
    transcript = {
        "phrases": [
            {"start": 0.0, "end": 0.5, "text": "What?"},
            {"start": 1.06, "end": 1.74, "text": "What's going on?"},
            {"start": 1.96, "end": 2.82, "text": "This is made by AI."},
        ]
    }
    timeline_data, local_prompts, segment_lengths = director._apply_audio_transcript_json(
        json.dumps(
            {
                "segments": [
                    {
                        "id": "old",
                        "type": "image",
                        "start": 0,
                        "length": 120,
                        "imageFile": "bear.png",
                        "prompt": "old prompt",
                    }
                ],
                "audioSegments": [{"id": "audio_001", "audioFile": "line.mp3"}],
            }
        ),
        "old local",
        "120",
        json.dumps(transcript),
        duration_frames=120,
        frame_rate=24.0,
    )

    timeline = json.loads(timeline_data)
    assert segment_lengths == "12,13,17,5,21,52"
    assert 'says "What?"' in local_prompts
    assert 'says "What\'s going on?"' in local_prompts
    assert 'says "This is made by AI."' in local_prompts
    assert timeline["audioSegments"] == [{"id": "audio_001", "audioFile": "line.mp3"}]
    assert timeline["segments"] == [
        {
            "id": "old",
            "type": "image",
            "start": 0,
            "length": 120,
            "imageFile": "bear.png",
            "prompt": "old prompt",
        }
    ]


def test_audio_transcript_json_builds_segments_when_timeline_has_none(monkeypatch):
    director = _load_director(monkeypatch)
    transcript = {"phrases": [{"start": 0.0, "end": 0.5, "text": "What?"}]}

    timeline_data, local_prompts, segment_lengths = director._apply_audio_transcript_json(
        json.dumps({"audioSegments": [{"id": "audio_001", "audioFile": "line.mp3"}]}),
        "old local",
        "120",
        json.dumps(transcript),
        duration_frames=24,
        frame_rate=24.0,
    )

    timeline = json.loads(timeline_data)
    assert segment_lengths == "12,12"
    assert 'says "What?"' in local_prompts
    assert [segment["start"] for segment in timeline["segments"]] == [0, 12]


def test_audio_transcript_json_rejects_empty_phrases(monkeypatch):
    director = _load_director(monkeypatch)

    with pytest.raises(ValueError, match="phrases must not be empty"):
        director._apply_audio_transcript_json(
            "{}",
            "",
            "",
            json.dumps({"phrases": []}),
            duration_frames=24,
            frame_rate=24.0,
        )
