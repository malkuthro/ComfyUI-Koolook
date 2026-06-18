"""Tests for Easy_LoadVideo's split input path composition."""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

import k_video_load
from k_video_load import (
    EMPTY_BRANCH_SENTINEL,
    _compose_input_video_path,
    _is_existing_local_video_path,
    _normalize_path_input,
    _normalize_text_input,
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

    assert isinstance(result, tuple) and len(result) == 4
    assert result[0] == ("zeros", ((1, 64, 64, 3),))  # empty IMAGE (shape passed as one arg)
    assert result[1] == 0  # frame_count
    assert result[2]["sample_rate"] == 44100  # empty AUDIO
    assert result[3] == {}  # empty VHS_VIDEOINFO

    monkeypatch.delitem(sys.modules, "nodes")
    monkeypatch.delitem(sys.modules, "folder_paths")
    monkeypatch.delitem(sys.modules, "torch")
    importlib.reload(k_video_load)
