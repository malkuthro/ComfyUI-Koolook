"""Tests for Easy_LoadVideo's split input path composition."""
from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Mapping
from pathlib import Path

import pytest

import k_video_load
from k_video_load import (
    EMPTY_BRANCH_SENTINEL,
    _SilentFallbackAudioMap,
    _compose_input_video_path,
    _guard_lazy_audio,
    _is_existing_local_video_path,
    _is_lazy_mapping,
    _normalize_path_input,
    _normalize_text_input,
    _reports_source_has_no_audio,
)


def test_empty_input_path_passes_video_through() -> None:
    assert _compose_input_video_path("E:/plates/shot01.mp4", "") == "E:/plates/shot01.mp4"
    assert _compose_input_video_path("https://example.test/clip.mp4", "") == "https://example.test/clip.mp4"


def test_empty_input_path_accepts_directory_and_filename_lines() -> None:
    root = Path.cwd().anchor
    absolute_dir = str(Path(root) / "projects" / "shot01")
    composed = _compose_input_video_path(
        f"{absolute_dir}\nplate_main.mp4",
        "",
    )
    assert Path(composed) == Path(absolute_dir) / "plate_main.mp4"


def test_absolute_input_path_joins_filename() -> None:
    # Use a real tmp_path so the absolute-path branch is platform-neutral.
    root = Path.cwd().anchor
    absolute_dir = str(Path(root) / "projects" / "shot01")
    composed = _compose_input_video_path(
        "plate_main.mp4",
        absolute_dir,
    )
    assert Path(composed) == Path(absolute_dir) / "plate_main.mp4"


def test_input_path_accepts_full_video_path_when_video_empty() -> None:
    root = Path.cwd().anchor
    video_path = str(Path(root) / "projects" / "shot01" / "plate_main.mp4")

    assert Path(_compose_input_video_path("", video_path)) == Path(video_path)


def test_existing_local_video_path_detection(tmp_path: Path) -> None:
    video = tmp_path / "plate_main.mp4"
    video.write_bytes(b"fake")

    assert _is_existing_local_video_path(str(video)) is True
    assert _is_existing_local_video_path(str(tmp_path / "plate_main.json")) is False
    assert _is_existing_local_video_path(str(tmp_path / "missing.mp4")) is False


def test_relative_input_path_accepts_full_video_path_when_video_empty(tmp_path: Path) -> None:
    composed = _compose_input_video_path(
        "",
        "shots/shot01/plate_main.mp4",
        input_root=str(tmp_path),
    )
    assert Path(composed) == tmp_path / "shots" / "shot01" / "plate_main.mp4"


def test_wrapped_full_video_path_in_input_path_rejoins_before_loading(tmp_path: Path) -> None:
    folder = tmp_path / "Runs-Audio-File"
    folder.mkdir()
    video = folder / "Bear_2x-FR_AudioFile_K-Dir_h264_v002.mp4"
    video.write_bytes(b"fake")
    wrapped = str(video).replace("Runs-Audio-File", "Runs-Audio-\nFile")

    composed = _compose_input_video_path("", wrapped)

    assert Path(composed) == video


def test_input_path_accepts_directory_and_filename_lines_when_video_empty(tmp_path: Path) -> None:
    video = tmp_path / "plate_main.mp4"
    video.write_bytes(b"fake")

    composed = _compose_input_video_path("", f"{tmp_path}\nplate_main.mp4")

    assert Path(composed) == video


def test_normalize_path_input_prefers_existing_wrapped_path(tmp_path: Path) -> None:
    folder = tmp_path / "Runs-Audio-File"
    folder.mkdir()
    video = folder / "clip.mov"
    video.write_bytes(b"fake")

    assert _normalize_path_input(str(video).replace("Runs-Audio-File", "Runs-Audio-\nFile")) == str(video)


def test_relative_input_path_roots_under_comfy_input_dir(tmp_path: Path) -> None:
    composed = _compose_input_video_path(
        "plate_main.mp4",
        "shots/shot01",
        input_root=str(tmp_path),
    )
    assert Path(composed) == tmp_path / "shots" / "shot01" / "plate_main.mp4"


def test_input_path_strips_video_path_components(tmp_path: Path) -> None:
    composed = _compose_input_video_path(
        "accidental/subdir/plate_main.mp4",
        "shots/shot01",
        input_root=str(tmp_path),
    )
    assert Path(composed) == tmp_path / "shots" / "shot01" / "plate_main.mp4"


def test_input_path_strips_windows_video_path_components(tmp_path: Path) -> None:
    composed = _compose_input_video_path(
        "accidental\\subdir\\plate_main.mp4",
        "shots/shot01",
        input_root=str(tmp_path),
    )
    assert Path(composed) == tmp_path / "shots" / "shot01" / "plate_main.mp4"


def test_input_path_requires_filename() -> None:
    # A non-existent path with no filename is a genuine mistake and still errors.
    with pytest.raises(ValueError, match="video must include a filename"):
        _compose_input_video_path("", "/no/such/projects/shot01")


def test_existing_directory_input_path_degrades_to_empty_sentinel(tmp_path: Path) -> None:
    # A real folder (e.g. an EXR sequence dir) handed to the video loader is the
    # unselected branch of an upstream mode switch. Instead of crashing the whole
    # prompt, _compose_input_video_path signals a graceful empty so the loader can
    # return nothing rather than raise -- mirroring how lenient sequence loaders
    # ignore a wrong-type path.
    seq_dir = tmp_path / "exr_seq"
    seq_dir.mkdir()
    assert _compose_input_video_path("", str(seq_dir)) == EMPTY_BRANCH_SENTINEL


def test_relative_existing_directory_input_path_degrades(tmp_path: Path) -> None:
    (tmp_path / "shots" / "shot01").mkdir(parents=True)
    assert (
        _compose_input_video_path("", "shots/shot01", input_root=str(tmp_path))
        == EMPTY_BRANCH_SENTINEL
    )


def test_normalize_text_input_handles_frontend_sentinels() -> None:
    assert _normalize_text_input("undefined") == ""
    assert _normalize_text_input("None") == ""
    assert _normalize_text_input("  plate_main.mp4\nundefined") == "plate_main.mp4"


def test_easy_load_video_calls_vhs_loader_directly_for_existing_full_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []

    def fake_shared_loader(**kwargs):
        calls.append(kwargs)
        return ("loaded", kwargs["video"])

    class FakeVHSLoadVideoPath:
        # Mirrors the real VHS node: (IMAGE, INT, AUDIO, VHS_VIDEOINFO).
        RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO")

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "video": ("STRING", {}),
                    "force_rate": ("FLOAT", {}),
                },
                "optional": {},
            }

        def load_video(self, **kwargs):
            raise AssertionError("strict VHS LoadVideoPath.load_video should be bypassed")

        @classmethod
        def IS_CHANGED(cls, video, **kwargs):
            return video

        @classmethod
        def VALIDATE_INPUTS(cls, video):
            return f"strict validator rejected {video}"

    FakeVHSLoadVideoPath.load_video.__globals__["load_video"] = fake_shared_loader
    fake_nodes = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={"VHS_LoadVideoPath": FakeVHSLoadVideoPath}
    )
    fake_folder_paths = types.SimpleNamespace(get_input_directory=lambda: str(tmp_path))
    monkeypatch.setitem(sys.modules, "nodes", fake_nodes)
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths)
    loaded = importlib.reload(k_video_load)

    video = tmp_path / "plate_main.mp4"
    video.write_bytes(b"fake")
    result = loaded.Easy_LoadVideo().load_video(
        input_path=str(video),
        video="",
        force_rate=0,
    )

    assert result == ("loaded", str(video))
    assert calls[-1]["video"] == str(video)
    assert loaded.Easy_LoadVideo.VALIDATE_INPUTS("", str(video)) is True

    monkeypatch.delitem(sys.modules, "nodes")
    monkeypatch.delitem(sys.modules, "folder_paths")
    importlib.reload(k_video_load)


def test_easy_load_video_validation_defers_when_input_path_is_linked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []

    class FakeVHSLoadVideoPath:
        # Mirrors the real VHS node: (IMAGE, INT, AUDIO, VHS_VIDEOINFO).
        RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO")

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"video": ("STRING", {})}, "optional": {}}

        def load_video(self, **kwargs):
            return ("loaded", kwargs["video"])

        @classmethod
        def IS_CHANGED(cls, video, **kwargs):
            return video

        @classmethod
        def VALIDATE_INPUTS(cls, video):
            calls.append(video)
            return f"strict validator rejected {video}"

    FakeVHSLoadVideoPath.load_video.__globals__["load_video"] = lambda **kwargs: (
        "loaded",
        kwargs["video"],
    )
    fake_nodes = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={"VHS_LoadVideoPath": FakeVHSLoadVideoPath}
    )
    fake_folder_paths = types.SimpleNamespace(get_input_directory=lambda: str(tmp_path))
    monkeypatch.setitem(sys.modules, "nodes", fake_nodes)
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths)
    loaded = importlib.reload(k_video_load)

    assert loaded.Easy_LoadVideo.VALIDATE_INPUTS(
        video="",
        input_path=None,
        force_rate=0,
        custom_width=0,
        custom_height=0,
        frame_load_cap=0,
        skip_first_frames=0,
        select_every_nth=1,
        format="AnimateDiff",
        unique_id="627",
    ) is True
    assert calls == []

    monkeypatch.delitem(sys.modules, "nodes")
    monkeypatch.delitem(sys.modules, "folder_paths")
    importlib.reload(k_video_load)


def test_easy_load_video_rejoins_wrapped_input_path_before_direct_vhs_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []

    def fake_shared_loader(**kwargs):
        calls.append(kwargs)
        return ("loaded", kwargs["video"])

    class FakeVHSLoadVideoPath:
        # Mirrors the real VHS node: (IMAGE, INT, AUDIO, VHS_VIDEOINFO).
        RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO")

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"video": ("STRING", {})}, "optional": {}}

        def load_video(self, **kwargs):
            raise AssertionError("strict VHS LoadVideoPath.load_video should be bypassed")

        @classmethod
        def IS_CHANGED(cls, video, **kwargs):
            return video

        @classmethod
        def VALIDATE_INPUTS(cls, video):
            return f"strict validator rejected {video}"

    FakeVHSLoadVideoPath.load_video.__globals__["load_video"] = fake_shared_loader
    fake_nodes = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={"VHS_LoadVideoPath": FakeVHSLoadVideoPath}
    )
    fake_folder_paths = types.SimpleNamespace(get_input_directory=lambda: str(tmp_path))
    monkeypatch.setitem(sys.modules, "nodes", fake_nodes)
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths)
    loaded = importlib.reload(k_video_load)

    folder = tmp_path / "Runs-Audio-File"
    folder.mkdir()
    video = folder / "Bear_2x-FR_AudioFile_K-Dir_h264_v002.mp4"
    video.write_bytes(b"fake")
    wrapped = str(video).replace("Runs-Audio-File", "Runs-Audio-\nFile")

    result = loaded.Easy_LoadVideo().load_video(input_path=wrapped, video="")

    assert result == ("loaded", str(video))
    assert calls[-1]["video"] == str(video)

    monkeypatch.delitem(sys.modules, "nodes")
    monkeypatch.delitem(sys.modules, "folder_paths")
    importlib.reload(k_video_load)


def test_easy_load_video_returns_empty_result_for_existing_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # When the loader is handed an existing folder (the unselected EXR branch of
    # an upstream switch), it must return an empty, correctly-shaped result
    # instead of raising -- so the dead branch never crashes the prompt.
    class FakeVHSLoadVideoPath:
        RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO")

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"video": ("STRING", {})}, "optional": {}}

        def load_video(self, **kwargs):
            raise AssertionError("loader must not run for an empty/unselected branch")

    FakeVHSLoadVideoPath.load_video.__globals__["load_video"] = lambda **kwargs: (
        _ for _ in ()
    ).throw(AssertionError("shared loader must not run for an empty branch"))
    fake_nodes = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={"VHS_LoadVideoPath": FakeVHSLoadVideoPath}
    )
    fake_folder_paths = types.SimpleNamespace(get_input_directory=lambda: str(tmp_path))
    fake_torch = types.SimpleNamespace(
        zeros=lambda *shape, **kw: ("zeros", shape),
        float32="float32",
    )
    monkeypatch.setitem(sys.modules, "nodes", fake_nodes)
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    loaded = importlib.reload(k_video_load)

    seq_dir = tmp_path / "exr_seq"
    seq_dir.mkdir()
    result = loaded.Easy_LoadVideo().load_video(input_path=str(seq_dir), video="")
    captured = capsys.readouterr()

    assert isinstance(result, tuple) and len(result) == 4
    assert result[0] == ("zeros", ((1, 64, 64, 3),))  # empty IMAGE (shape passed as one arg)
    assert result[1] == 0  # frame_count
    assert result[2]["sample_rate"] == 44100  # empty AUDIO
    assert result[3] == {}  # empty VHS_VIDEOINFO
    assert "existing folder with no video filename" in captured.out

    monkeypatch.delitem(sys.modules, "nodes")
    monkeypatch.delitem(sys.modules, "folder_paths")
    monkeypatch.delitem(sys.modules, "torch")
    importlib.reload(k_video_load)


# --- audio-less source guard -----------------------------------------------
#
# ComfyUI core recurses into Mapping outputs hunting for model patchers, which
# forces VHS's lazy audio map to shell out to ffmpeg even when nothing consumes
# the AUDIO socket. A Nuke mov64 write has video + timecode and no audio, so
# that call fails and used to take down the whole prompt.
#
# Contract: silence is substituted only on EVIDENCE OF ABSENCE -- ffmpeg opened
# the input, listed its streams, none was audio, and the muxer had nothing to
# write. Absence of evidence (missing binary, permission error, timeout,
# truncated log) must stay loud.

# Trimmed from a real failure: Nuke ProRes 4444, video + timecode, no audio.
_NO_AUDIO_REPORT = """VHS failed to extract audio from cow_cropped.mov:
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'cow_cropped.mov':
  Stream #0:0[0x1]: Video: prores (4444) (ap4h / 0x68347061), 1920x1080, 25 fps
  Stream #0:1[0x2](eng): Data: none (tmcd / 0x64636D74), 0 kb/s
Output #0, f32le, to 'pipe:':
Output file does not contain any stream
Error opening output files: Invalid argument
"""

_HAS_AUDIO_REPORT = """VHS failed to extract audio from take_02.mov:
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'take_02.mov':
  Stream #0:0[0x1]: Video: prores (4444) (ap4h / 0x68347061), 1920x1080, 25 fps
  Stream #0:1[0x2](eng): Audio: pcm_s16le, 48000 Hz, stereo, s16, 1536 kb/s
Some other ffmpeg failure
"""

# Failures that say NOTHING about whether the source has audio. Every one of
# these was silently converted to silence before the contract was tightened.
_AMBIGUOUS_REPORTS = {
    "missing ffmpeg binary":
        "[WinError 2] The system cannot find the file specified: 'ffmpeg.exe'",
    "permission denied":
        "VHS failed to extract audio from locked.mov:\nlocked.mov: Permission denied\n",
    "timeout":
        "Command '['ffmpeg', '-i', 'x.mov']' timed out after 30 seconds",
    "empty message": "",
    "truncated before stream listing":
        "VHS failed to extract audio from x.mov:\n"
        "Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'x.mov':\n",
    "input never opened":
        "VHS failed to extract audio from gone.mov:\ngone.mov: No such file or directory\n",
    "non-ffmpeg exception": "RuntimeError: something unrelated blew up",
}


class _ExplodingAudioMap(Mapping):
    """Stands in for VHS's LazyAudioMap: raises on materialization."""

    def __init__(self, report, file="cow_cropped.mov"):
        self.report = report
        self.file = file
        self.calls = 0

    def _boom(self):
        self.calls += 1
        raise Exception(self.report)

    def __getitem__(self, key):
        self._boom()

    def __iter__(self):
        self._boom()

    def __len__(self):
        self._boom()


@pytest.fixture
def stub_torch(monkeypatch: pytest.MonkeyPatch):
    """Minimal torch so _empty_value_for_type('AUDIO') works without real torch."""
    fake = types.ModuleType("torch")
    fake.float32 = "float32"
    fake.zeros = lambda shape, dtype=None: ("zeros", shape, dtype)
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


# --- the predicate ---------------------------------------------------------

def test_no_audio_recognised_only_from_a_complete_ffmpeg_report() -> None:
    assert _reports_source_has_no_audio(_NO_AUDIO_REPORT) is True


def test_a_source_with_audio_is_never_treated_as_missing_audio() -> None:
    assert _reports_source_has_no_audio(_HAS_AUDIO_REPORT) is False


@pytest.mark.parametrize("label", sorted(_AMBIGUOUS_REPORTS))
def test_ambiguous_failures_are_not_treated_as_missing_audio(label: str) -> None:
    # Absence of evidence is not evidence of absence: none of these prove the
    # source lacks audio, so none may be downgraded to silence.
    assert _reports_source_has_no_audio(_AMBIGUOUS_REPORTS[label]) is False


def test_predicate_does_not_mistake_a_data_or_video_stream_for_audio() -> None:
    assert _reports_source_has_no_audio("Stream #0:1[0x2](eng): Data: none (tmcd)") is False
    assert _reports_source_has_no_audio("Stream #0:0[0x1]: Video: prores (4444)") is False
    assert _reports_source_has_no_audio(None) is False


# --- the lazy wrapper ------------------------------------------------------

def test_silent_fallback_is_lazy_until_accessed() -> None:
    inner = _ExplodingAudioMap(_NO_AUDIO_REPORT)
    _SilentFallbackAudioMap(inner)

    # Merely wrapping must not shell out to ffmpeg.
    assert inner.calls == 0


def test_silent_fallback_substitutes_silence_when_source_has_no_audio(stub_torch) -> None:
    guarded = _SilentFallbackAudioMap(_ExplodingAudioMap(_NO_AUDIO_REPORT))

    assert dict(guarded) == {
        "waveform": ("zeros", (1, 2, 1), "float32"),
        "sample_rate": 44100,
    }
    assert guarded["sample_rate"] == 44100
    assert len(guarded) == 2


def test_silent_fallback_reraises_when_the_source_does_have_audio() -> None:
    guarded = _SilentFallbackAudioMap(_ExplodingAudioMap(_HAS_AUDIO_REPORT))

    with pytest.raises(Exception) as excinfo:
        dict(guarded)
    assert "Some other ffmpeg failure" in str(excinfo.value)


@pytest.mark.parametrize("label", sorted(_AMBIGUOUS_REPORTS))
def test_silent_fallback_reraises_ambiguous_failures(label: str, stub_torch) -> None:
    # stub_torch is requested deliberately: if the guard swallowed these,
    # silence would be constructible and the test could pass for the wrong
    # reason. It must raise even when silence is available.
    guarded = _SilentFallbackAudioMap(_ExplodingAudioMap(_AMBIGUOUS_REPORTS[label]))

    with pytest.raises(Exception):
        dict(guarded)


def test_silent_fallback_resolves_once(stub_torch) -> None:
    inner = _ExplodingAudioMap(_NO_AUDIO_REPORT)
    guarded = _SilentFallbackAudioMap(inner)

    dict(guarded)
    dict(guarded)
    assert inner.calls == 1


def test_silent_fallback_passes_real_audio_through() -> None:
    class _Lazy(Mapping):
        def __init__(self):
            self._d = {"waveform": "wave", "sample_rate": 48000}

        def __getitem__(self, k):
            return self._d[k]

        def __iter__(self):
            return iter(self._d)

        def __len__(self):
            return len(self._d)

    guarded = _SilentFallbackAudioMap(_Lazy())
    assert dict(guarded) == {"waveform": "wave", "sample_rate": 48000}


# --- slot scoping ----------------------------------------------------------

def test_is_lazy_mapping_targets_only_deferred_maps() -> None:
    # VHS_VIDEOINFO and already-materialized audio are plain dicts; leave them be.
    assert _is_lazy_mapping({"waveform": 1, "sample_rate": 44100}) is False
    assert _is_lazy_mapping({}) is False
    assert _is_lazy_mapping("string") is False
    assert _is_lazy_mapping(None) is False
    assert _is_lazy_mapping(_ExplodingAudioMap(_NO_AUDIO_REPORT)) is True


def test_guard_wraps_the_declared_audio_slot() -> None:
    lazy = _ExplodingAudioMap(_NO_AUDIO_REPORT)
    info = {"source_fps": 25}
    # Reach through the module: other tests reload k_video_load, which refreshes
    # the module dict in place, so a top-level-imported class can go stale while
    # the helper resolves the rebound one.
    result = k_video_load._guard_lazy_audio(
        ("images", 8, lazy, info), ("IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO")
    )

    assert result[0] == "images"
    assert result[1] == 8
    assert isinstance(result[2], k_video_load._SilentFallbackAudioMap)
    assert result[3] is info


def test_guard_leaves_non_audio_lazy_mappings_alone() -> None:
    # A deferred Mapping in a non-AUDIO slot keeps its own failure mode --
    # rewriting it as missing audio would mask an unrelated error.
    lazy = _ExplodingAudioMap(_NO_AUDIO_REPORT)
    result = k_video_load._guard_lazy_audio(("images", lazy), ("IMAGE", "VHS_VIDEOINFO"))

    assert result[1] is lazy


def test_guard_handles_missing_or_short_return_types() -> None:
    lazy = _ExplodingAudioMap(_NO_AUDIO_REPORT)
    assert k_video_load._guard_lazy_audio(("a", lazy), None) == ("a", lazy)
    assert k_video_load._guard_lazy_audio(("a", lazy), ()) == ("a", lazy)
    # More return types than values must not IndexError.
    assert k_video_load._guard_lazy_audio(("a",), ("IMAGE", "AUDIO")) == ("a",)


def test_guard_passes_non_tuple_results_through() -> None:
    assert k_video_load._guard_lazy_audio(None, ("AUDIO",)) is None
    sentinel = {"not": "a tuple"}
    assert k_video_load._guard_lazy_audio(sentinel, ("AUDIO",)) is sentinel
