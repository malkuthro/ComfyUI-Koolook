"""Pure-function coverage for scripts/loop_audio.py.

Anchored by the PR #185 multi-agent review which surfaced three
silent-failure HIGHs on the audio-source-derivation path:

  * derive_audio_state collapsed missing-Director and unwired-VAE
    into the same label;
  * extract_multilines used dict-insertion order, so a shorter needle
    declared first could silently shadow a longer match;
  * parse_timeline returned segments verbatim, so a string-valued
    `start` field would crash the renderer at `start / fps` or quietly
    break `<` comparisons in video_segment_has_audio.

These tests pin each fix plus the surrounding helpers so future edits
can't silently regress.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Load the script directly — it lives under scripts/, not under a package.
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "loop_audio.py"
sys.path.insert(0, str(_SCRIPT.parent))
_spec = importlib.util.spec_from_file_location("loop_audio", _SCRIPT)
assert _spec is not None and _spec.loader is not None
loop_audio = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(loop_audio)


# --- helpers used by multiple tests -----------------------------------------


def _director(
    *,
    node_type: str = "LTXDirector__koolook",
    use_custom_audio: bool = False,
    audio_vae_link: int | None = None,
    timeline: dict | None = None,
    epsilon: float = 0.001,
    fps: int = 24,
) -> dict:
    """Build a minimal Koolook Director node dict matching the saved
    widget order (DIRECTOR_WIDX). Only fields the helpers under test
    actually read are populated; the rest are placeholders."""
    import json as _json
    timeline_str = _json.dumps(timeline) if timeline is not None else ""
    wv = [
        "",                # 0  global_prompt
        120,               # 1  duration_frames
        5,                 # 2  duration_seconds
        timeline_str,      # 3  timeline_data
        "",                # 4  local_prompts
        "120",             # 5  segment_lengths
        epsilon,           # 6  epsilon
        "1.00",            # 7  guide_strength
        use_custom_audio,  # 8  use_custom_audio
        fps,               # 9  frame_rate
        "seconds",         # 10 display_mode
        0,                 # 11 custom_width
        0,                 # 12 custom_height
        "maintain aspect ratio",  # 13 resize_method
        32,                # 14 divisible_by
        18,                # 15 img_compression
        "",                # 16 relay_overrides
    ]
    return {
        "type": node_type,
        "widgets_values": wv,
        "inputs": [
            {"name": "audio_vae", "link": audio_vae_link},
            {"name": "use_custom_audio", "link": None},
        ],
    }


# --- extract_director — stable, legacy, and upstream IDs --------------------


@pytest.mark.parametrize(
    "node_type",
    ["LTXDirector__koolook", "LTXDirector__koolook_v1_3_2", "LTXDirector"],
)
def test_extract_director_accepts_supported_director_ids(node_type):
    node = _director(node_type=node_type)
    assert loop_audio.extract_director([node]) is node


def test_extract_director_prefers_koolook_over_upstream():
    upstream = _director(node_type="LTXDirector")
    koolook = _director(node_type="LTXDirector__koolook")
    assert loop_audio.extract_director([upstream, koolook]) is koolook


def test_extract_director_prefers_guide_wired_upstream_over_idle_koolook():
    upstream = {"id": 10, **_director(node_type="LTXDirector")}
    upstream["outputs"] = [{"name": "guide_data", "links": [100]}]
    koolook = {"id": 20, **_director(node_type="LTXDirector__koolook")}
    koolook["outputs"] = [{"name": "guide_data", "links": []}]
    reroute = {
        "id": 30,
        "type": "Reroute",
        "inputs": [{"name": "", "link": 100}],
        "outputs": [{"name": "", "links": [101]}],
    }
    guide = {
        "id": 40,
        "type": "LTXDirectorGuide",
        "inputs": [{"name": "guide_data", "link": 101}],
    }

    assert (
        loop_audio.extract_director(
            [koolook, upstream, reroute, guide],
            [
                [100, 10, 4, 30, 0, "GUIDE_DATA"],
                [101, 30, 0, 40, 4, "GUIDE_DATA"],
            ],
        )
        is upstream
    )


def test_director_widget_uses_named_widget_before_positional_fallback():
    node = {
        "type": "LTXDirector__koolook",
        "inputs": [
            {"name": "duration_frames", "widget": {"name": "duration_frames"}},
            {"name": "epsilon", "widget": {"name": "epsilon"}},
            {"name": "use_custom_audio", "widget": {"name": "use_custom_audio"}},
        ],
        "widgets_values": [144, 0.004, True],
    }

    assert loop_audio.director_widget(node, "epsilon") == 0.004
    assert loop_audio.director_widget(node, "use_custom_audio") is True


def test_director_widget_keeps_legacy_positional_fallback():
    node = _director(epsilon=0.002)
    assert loop_audio.director_widget(node, "epsilon") == 0.002


@pytest.mark.parametrize(
    "node_type, expected",
    [
        ("LTXDirector__koolook", "Koolook"),
        ("LTXDirector__koolook_v1_3_2", "Koolook"),
        ("LTXDirector", "Original upstream"),
    ],
)
def test_director_flavor_labels_supported_directors(node_type, expected):
    assert loop_audio.director_flavor(_director(node_type=node_type)) == expected


@pytest.mark.parametrize(
    "node_type, upstream_version, expected",
    [
        ("LTXDirector__koolook", "", "v1.3.9"),
        ("LTXDirector__koolook_v1_3_2", "", "v1.3.9"),
        ("LTXDirector", "1.3.2", "v1.3.2"),
        ("LTXDirector", "", "(unknown upstream pin)"),
    ],
)
def test_director_pin_tag_labels_lock_version(node_type, upstream_version, expected):
    assert (
        loop_audio.director_pin_tag(
            _director(node_type=node_type), upstream_version
        )
        == expected
    )


# --- derive_audio_state — 5 distinct states --------------------------------


@pytest.mark.parametrize(
    "director_node, timeline, expected",
    [
        # (no director) — split off so a missing-Director workflow can't
        # silently look identical to a director-present-but-VAE-unwired
        # one (PR #185 review HIGH-1).
        (None, {"segments": [], "audioSegments": []}, "(no director)"),
        # off (no VAE) — Director present, audio_vae socket unwired.
        (
            _director(audio_vae_link=None),
            {"segments": [], "audioSegments": []},
            "off (no VAE)",
        ),
        # model-gen — VAE wired, use_custom_audio=False. audioSegments
        # are ignored on this path.
        (
            _director(audio_vae_link=42, use_custom_audio=False),
            {"segments": [], "audioSegments": []},
            "model-gen",
        ),
        # custom — VAE wired, use_custom_audio=True, audioSegments
        # non-empty.
        (
            _director(audio_vae_link=42, use_custom_audio=True),
            {"segments": [], "audioSegments": [{"start": 0, "length": 100}]},
            "custom",
        ),
        # custom (empty) — VAE wired, use_custom_audio=True, but no
        # audioSegments uploaded.
        (
            _director(audio_vae_link=42, use_custom_audio=True),
            {"segments": [], "audioSegments": []},
            "custom (empty)",
        ),
    ],
)
def test_derive_audio_state_all_five_states(director_node, timeline, expected):
    assert loop_audio.derive_audio_state(director_node, timeline) == expected


# --- extract_multilines — longest-needle-wins guards future drift ----------


def _multiline(title: str, body: str) -> dict:
    return {
        "type": "Text Multiline",
        "title": title,
        "widgets_values": [body],
    }


def test_extract_multilines_keeps_all_hits_for_one_needle():
    """Working_Folder_PATH legitimately appears twice on the canvas
    (project mount + local mirror). Both bodies must come back so the
    caller can pick by reachability."""
    nodes = [
        _multiline("Working_Folder_PATH", "W:/projects/foo"),
        _multiline("Working_Folder_PATH", "e:/local/foo"),
    ]
    out = loop_audio.extract_multilines(nodes, ["working_folder"])
    assert out["working_folder"] == ["W:/projects/foo", "e:/local/foo"]


def test_extract_multilines_longest_needle_wins():
    """If a future config adds a short substring that's contained in
    an existing longer one, the longer match should win — regardless
    of declaration order. The pre-fix loop iterated in dict-insertion
    order and would have routed both nodes to the shorter needle."""
    nodes = [
        _multiline("RELAY_OVERRIDES", "{video_strength: 10}"),
    ]
    # Shorter "relay" declared FIRST in the config — pre-fix this would
    # have shadowed "relay_overrides". Longest-first ordering routes
    # the node to the more specific needle.
    out = loop_audio.extract_multilines(
        nodes, ["relay", "relay_overrides"]
    )
    assert out["relay_overrides"] == ["{video_strength: 10}"]
    assert out["relay"] == []


def test_extract_multilines_first_match_wins_per_node():
    """A node's title can match multiple needles. The loop should
    record at most one match per node (the longest) to avoid
    double-counting."""
    nodes = [_multiline("NAME overlay - info combined", "one node")]
    out = loop_audio.extract_multilines(
        nodes, ["name", "overlay - info"]
    )
    # Both needles match the title, but the longer one wins and the
    # per-node break keeps the same node from being counted twice.
    assert out["name"] == []
    assert out["overlay - info"] == ["one node"]


def test_extract_multilines_accepts_prioritized_alias_map():
    nodes = [
        _multiline("NAME", "old setup name"),
        _multiline("GLOBAL [ base name ]", "new base name"),
        _multiline("GLOBAL [ path ] - working folder", "E:/runs"),
    ]
    out = loop_audio.extract_multilines(
        nodes,
        {
            "name": ["global [ base name ]", "name"],
            "working_folder": [
                "global [ path ] - working folder",
                "working_folder",
            ],
        },
    )
    assert out["name"] == ["new base name", "old setup name"]
    assert out["working_folder"] == ["E:/runs"]


def test_extract_setup_variables_reads_text_and_primitive_source_nodes():
    nodes = [
        _multiline("INPUT Path [ EXR ]", "W:/plates"),
        {
            "type": "PrimitiveInt",
            "title": "GLOBAL [ version ]",
            "widgets_values": [1, "fixed"],
        },
        {
            "type": "PrimitiveInt",
            "title": "GLOBAL [ run offset ]",
            "widgets_values": [7, "fixed"],
        },
        {
            "type": "GetNode",
            "title": "Get_GLOBAL [ version ]",
            "widgets_values": ["GLOBAL [ version ]"],
        },
        {
            "type": "SetNode",
            "title": "Set_GLOBAL [ run ]",
            "widgets_values": ["GLOBAL [ run ]"],
        },
    ]
    out = loop_audio.extract_setup_variables(
        nodes,
        {
            "input_path_exr": ["input path [ exr ]"],
            "version": ["global [ version ]"],
            "run_offset": ["global [ run offset ]"],
        },
    )
    assert out["input_path_exr"] == ["W:/plates"]
    assert out["version"] == ["1"]
    assert out["run_offset"] == ["7"]


def test_expected_output_tracking_uses_current_setup_values():
    nodes = [
        {
            "type": "Easy_VideoCombine",
            "widgets_values": {
                "format": "video/koolook-ASTRA-h264",
            },
        },
        {
            "type": "easy showAnything",
            "widgets_values": [
                '[true, ["E:/old/Previous_h264_v001.mp4"]]',
            ],
        },
    ]
    out = loop_audio.expected_output_tracking(
        nodes,
        {
            "working_folder": ["E:/current/renders"],
            "name": ["Bear_2x-FR_AudioFile-K_Dir"],
        },
        {"version": ["2"]},
    )
    assert out["folder"] == "E:/current/renders"
    assert out["name"] == "Bear_2x-FR_AudioFile-K_Dir_h264_v002"


def test_output_suffix_uses_named_widget_before_positional_fallback():
    nodes = [
        {
            "type": "Easy_VideoCombine",
            "inputs": [
                {"name": "frame_rate", "widget": {"name": "frame_rate"}},
                {"name": "format", "widget": {"name": "format"}},
                {"name": "version", "widget": {"name": "version"}},
            ],
            "widgets_values": [24, "video/ProRes", "999"],
        },
    ]
    assert loop_audio.output_suffix_from_workflow(nodes) == "ProRes"


def test_output_suffix_keeps_legacy_positional_fallback():
    nodes = [
        {
            "type": "Easy_VideoCombine",
            "widgets_values": [24, 0, "upscaled", "video/koolook-ASTRA-h264"],
        },
    ]
    assert loop_audio.output_suffix_from_workflow(nodes) == "h264"


def test_delivery_card_path_uses_output_folder_and_name():
    out = loop_audio.delivery_card_path({
        "folder": "E:/current/renders",
        "name": "Bear_h264_v002",
    })
    assert out == Path("E:/current/renders") / "cards" / "Bear_h264_v002_card.png"


def test_delivery_card_path_can_include_run_number():
    out = loop_audio.delivery_card_path(
        {
            "folder": "E:/current/renders",
            "name": "Bear_h264_v002",
        },
        7,
    )
    assert out == (
        Path("E:/current/renders")
        / "cards"
        / "Bear_h264_v002_run007_card.png"
    )


def test_delivery_card_path_skips_when_output_is_unknown():
    assert loop_audio.delivery_card_path({"folder": "E:/renders"}) is None


def test_copy_delivery_card_reports_failure_without_raising(monkeypatch, tmp_path: Path):
    def fail_copy(_src, _dst):
        raise OSError("drive unavailable")

    monkeypatch.setattr(loop_audio.shutil, "copy2", fail_copy)

    status = loop_audio.copy_delivery_card(
        tmp_path / "card.png",
        {"folder": str(tmp_path), "name": "Bear_h264_v002"},
    )

    assert status == "failed (drive unavailable)"


def test_copy_delivery_card_creates_missing_cards_folder(tmp_path: Path):
    card = tmp_path / "source.png"
    folder = tmp_path / "renders"
    card.write_text("new", encoding="utf-8")

    status = loop_audio.copy_delivery_card(
        card,
        {"folder": str(folder), "name": "Bear_h264_v002"},
        7,
    )

    expected = folder / "cards" / "Bear_h264_v002_run007_card.png"
    assert status == str(expected)
    assert expected.read_text(encoding="utf-8") == "new"


def test_copy_delivery_card_leaves_existing_file_in_place(tmp_path: Path):
    card = tmp_path / "source.png"
    folder = tmp_path / "renders"
    existing = folder / "cards" / "Bear_h264_v002_card.png"
    card.write_text("new", encoding="utf-8")
    existing.parent.mkdir(parents=True)
    existing.write_text("old", encoding="utf-8")

    status = loop_audio.copy_delivery_card(
        card,
        {"folder": str(folder), "name": "Bear_h264_v002"},
    )

    assert status.startswith("exists (left in place:")
    assert existing.read_text(encoding="utf-8") == "old"


def test_copy_delivery_card_can_overwrite_existing_file(tmp_path: Path):
    card = tmp_path / "source.png"
    folder = tmp_path / "renders"
    existing = folder / "cards" / "Bear_h264_v002_run007_card.png"
    card.write_text("new", encoding="utf-8")
    existing.parent.mkdir(parents=True)
    existing.write_text("old", encoding="utf-8")

    status = loop_audio.copy_delivery_card(
        card,
        {"folder": str(folder), "name": "Bear_h264_v002"},
        7,
        overwrite=True,
    )

    assert status == str(existing)
    assert existing.read_text(encoding="utf-8") == "new"


def test_card_metadata_scrubs_path_bearing_fields():
    metadata = loop_audio.card_metadata(
        4,
        "label",
        Path("run004_workflow.json"),
        {"name": ["Bear"], "relay_overrides": ["{}"]},
        {"input_path_exr": ["W:/projects/client_codename/shot/v003"]},
        {
            "folder": "E:/Jobs/Client/Comfy/Runs",
            "name": "Bear_h264_v002",
            "version_tag": "v002",
            "format_suffix": "h264",
        },
        _director(),
        {"segments": [], "audioSegments": []},
        "custom",
        {},
        "clean",
        "1.3.2",
    )

    encoded = json.dumps(metadata)
    assert "W:/projects" not in encoded
    assert "client_codename" not in encoded
    assert "E:/Jobs" not in encoded
    assert "path-sha256:" in encoded


def test_card_metadata_does_not_store_boolean_frame_rate():
    metadata = loop_audio.card_metadata(
        4,
        "label",
        Path("run004_workflow.json"),
        {"name": ["Bear"], "relay_overrides": ["{}"]},
        {},
        {"folder": "", "name": ""},
        _director(fps=True),
        {"segments": [], "audioSegments": []},
        "model-gen",
        {},
        "clean",
        "1.3.2",
    )

    assert metadata["director"]["frame_rate"] is None


def test_sanitize_workflow_for_archive_redacts_absolute_paths():
    workflow = {
        "nodes": [
            {
                "widgets_values": [
                    "E:/Jobs/Client/Comfy/Runs",
                    "notes\nW:/projects/client_codename/shot/v003\nok",
                    '["e:\\\\G-Drive-BaconX\\\\Jobs\\\\Jeep_Animals\\\\render.json"]',
                    "<PROJECTS>/samsung_goat/vfx/assets",
                    "W:/projects/client_codename/shot/v003",
                    "relative/path/is-kept",
                    "https://example.com/kept",
                ]
            }
        ]
    }

    sanitized = loop_audio.sanitize_workflow_for_archive(workflow)
    encoded = json.dumps(sanitized)

    assert "E:/Jobs" not in encoded
    assert "W:/projects" not in encoded
    assert "client_codename" not in encoded
    assert "samsung_goat" not in encoded
    assert "relative/path/is-kept" in encoded
    assert "https://example.com/kept" in encoded
    assert "path-sha256:" in encoded


def test_render_notes_scrubs_paths_and_rejects_boolean_fps():
    notes = loop_audio.render_notes_md(
        5,
        Path("LTX-23-audio_tests_03.json"),
        {
            "name": ["Bear"],
            "relay_overrides": [""],
            "overlay - info": [""],
            "overlay - feedback": [""],
            "working_folder": ["E:/Jobs/Client/Comfy/Runs"],
        },
        {
            "input_path_exr": ["W:/projects/client_codename/shot/v003"],
            "version": ["1"],
            "run_offset": ["0"],
        },
        _director(fps=True),
        {"segments": [], "audioSegments": []},
        "custom",
        "",
        [],
        {"motion": None, "sync": None, "sharp": None},
        {"folder": "E:/Jobs/Client/Comfy/Runs", "name": "Bear_h264_v001"},
    )

    assert "E:/Jobs" not in notes
    assert "W:/projects" not in notes
    assert "client_codename" not in notes
    assert "path-sha256:" in notes
    assert "True fps" not in notes
    assert "(unknown fps)" in notes


def test_audio_card_embeds_metadata_payload(tmp_path: Path):
    from PIL import Image

    from make_card_audio import render_audio_card

    metadata = {
        "schema": "koolook.audio_loop.card_metadata.v1",
        "repo": {"main_sha": "abc1234"},
        "output": {"name": "Bear_h264_v002"},
    }
    out = tmp_path / "card.png"
    render_audio_card(
        {
            "run_number": 3,
            "date": "2026-05-30",
            "workflow_name": "workflow.json",
            "name": "Bear",
            "relay_overrides_raw": "{}",
            "info_body": "",
            "feedback_lines": [],
            "scores": {},
            "work_folder": "E:/renders",
            "output_folder": "E:/renders",
            "output_name": "Bear_h264_v002",
            "director_flavor": "Koolook v1.3.9",
            "audio_src": "custom",
            "epsilon": 0.001,
            "frame_rate": 24,
            "segments": [],
            "audio_segments": [],
            "segment_prompt_mode": "none",
            "metadata": metadata,
        },
        out,
    )
    embedded = json.loads(Image.open(out).info["koolook_audio_loop"])
    assert embedded == metadata


def test_rebuild_state_handles_non_numeric_run_dir_and_bom_workflow(tmp_path: Path):
    from make_card_audio import _rebuild_state_from_run_dir

    run_dir = tmp_path / "run-foo_label"
    run_dir.mkdir()
    (run_dir / "workflow.json").write_text(
        json.dumps({"nodes": []}),
        encoding="utf-8-sig",
    )

    state = _rebuild_state_from_run_dir(run_dir)

    assert state["run_number"] == 0
    assert state["run_label"] == "label"


def test_rebuild_state_preserves_date_and_splits_repo_sync_metadata(tmp_path: Path):
    from make_card_audio import _rebuild_state_from_run_dir

    run_dir = tmp_path / "run-005_label"
    run_dir.mkdir()
    (run_dir / "run005_workflow.json").write_text(
        json.dumps({"nodes": [], "links": []}),
        encoding="utf-8",
    )
    (run_dir / "metadata.json").write_text(
        json.dumps({"run": {"date": "2026-05-01"}}),
        encoding="utf-8",
    )
    (run_dir / "patch_state.txt").write_text(
        "\n".join(
            [
                "MAIN SHA              : abc1234",
                "Last dev-sync-audio   : def5678  (2026-05-02 11:22)",
                "Sync scope tag        : relay parser",
                "Sync worktree         : ComfyUI-Koolook",
                "Fork dir status       : clean",
            ]
        ),
        encoding="utf-8",
    )

    state = _rebuild_state_from_run_dir(run_dir)
    metadata = state["metadata"]

    assert state["date"] == "2026-05-01"
    assert metadata["run"]["date"] == "2026-05-01"
    assert metadata["repo"]["last_dev_sync_audio"] == "def5678"
    assert metadata["repo"]["last_dev_sync_at"] == "2026-05-02 11:22"


def test_extract_multilines_ignores_non_text_multiline_nodes():
    nodes = [
        {"type": "LTXDirector__koolook_v1_3_2", "title": "NAME"},
        _multiline("NAME", "real"),
    ]
    out = loop_audio.extract_multilines(nodes, ["name"])
    assert out["name"] == ["real"]


# --- parse_timeline — coerce numeric segment fields ------------------------


def test_parse_timeline_coerces_string_numerics():
    """The Comfy frontend sometimes saves numeric segment fields as
    strings. parse_timeline must coerce them so downstream arithmetic
    in the renderer (`start / fps`, `<` comparisons) stays type-safe
    instead of crashing or doing lexical comparisons."""
    node = _director(timeline={
        "segments": [
            {"id": "a", "start": "0", "length": "120", "prompt": "x"},
        ],
        "audioSegments": [],
    })
    tl = loop_audio.parse_timeline(node)
    seg = tl["segments"][0]
    assert seg["start"] == 0
    assert seg["length"] == 120
    assert isinstance(seg["start"], int)
    assert isinstance(seg["length"], int)
    # Non-numeric fields pass through.
    assert seg["prompt"] == "x"


def test_parse_timeline_coerces_float_to_int():
    node = _director(timeline={
        "segments": [{"start": 1.7, "length": 12.4}],
        "audioSegments": [],
    })
    tl = loop_audio.parse_timeline(node)
    # int(float(...)) truncates toward 0, matching the upstream
    # Director's _build_combined_audio coercion behavior.
    assert tl["segments"][0]["start"] == 1
    assert tl["segments"][0]["length"] == 12


def test_parse_timeline_collapses_bad_values_to_zero():
    """A malformed segment shouldn't crash the whole loop — the
    offending field collapses to 0 and we keep going."""
    node = _director(timeline={
        "segments": [{"start": "not a number", "length": 100}],
        "audioSegments": [],
    })
    tl = loop_audio.parse_timeline(node)
    assert tl["segments"][0]["start"] == 0
    assert tl["segments"][0]["length"] == 100


def test_parse_timeline_handles_malformed_json():
    """Invalid JSON in timeline_data must not raise — the helper
    returns empty lists so the renderer falls back to "(N=0)" and
    proceeds."""
    bad = _director()
    bad["widgets_values"][3] = "{not json"
    assert loop_audio.parse_timeline(bad) == {
        "segments": [], "audioSegments": [],
    }


def test_parse_timeline_handles_missing_director():
    assert loop_audio.parse_timeline(None) == {
        "segments": [], "audioSegments": [],
    }


def test_parse_timeline_drops_non_dict_segments():
    node = _director(timeline={
        "segments": [
            {"start": 0, "length": 60},  # kept
            "not a dict",                 # dropped
            None,                         # dropped
        ],
        "audioSegments": [],
    })
    tl = loop_audio.parse_timeline(node)
    assert len(tl["segments"]) == 1


# --- video_segment_has_audio — overlap boundaries --------------------------


@pytest.mark.parametrize(
    "video, audio_segs, expected",
    [
        # Same range — overlaps.
        ({"start": 0, "length": 100}, [{"start": 0, "length": 100}], True),
        # Audio starts inside video — overlaps.
        ({"start": 0, "length": 100}, [{"start": 50, "length": 50}], True),
        # Audio ends inside video — overlaps.
        ({"start": 50, "length": 100}, [{"start": 0, "length": 75}], True),
        # Audio strictly before video — no overlap.
        ({"start": 100, "length": 50}, [{"start": 0, "length": 100}], False),
        # Audio strictly after video — no overlap.
        ({"start": 0, "length": 50}, [{"start": 100, "length": 50}], False),
        # Audio exactly touches video end — half-open intervals
        # (a_start < v_end) treats this as NOT overlapping.
        ({"start": 0, "length": 50}, [{"start": 50, "length": 50}], False),
        # Multiple audio segs — any one overlap is enough.
        (
            {"start": 0, "length": 50},
            [{"start": 100, "length": 10}, {"start": 25, "length": 10}],
            True,
        ),
        # No audio segs — false.
        ({"start": 0, "length": 50}, [], False),
    ],
)
def test_video_segment_has_audio_boundaries(video, audio_segs, expected):
    assert loop_audio.video_segment_has_audio(video, audio_segs) is expected


# --- segment_prompt_mode — same vs per-segment prompt check -----------------


@pytest.mark.parametrize(
    "segments, expected",
    [
        ([], "none"),
        ([{"prompt": "one prompt"}], "single"),
        (
            [{"prompt": "same prompt"}, {"prompt": "same   prompt"}],
            "same",
        ),
        (
            [{"prompt": "wide shot"}, {"prompt": "close up"}],
            "per-segment",
        ),
        (
            [{"prompt": "wide shot"}, {"prompt": ""}],
            "missing",
        ),
    ],
)
def test_segment_prompt_mode_classifies_prompt_sequence(segments, expected):
    assert loop_audio.segment_prompt_mode(segments) == expected


# --- parse_feedback — score 0 must round-trip (PR #185 review MEDIUM-9) ----


def test_parse_feedback_extracts_scores_with_lines():
    body = (
        "Looking solid overall\n"
        "Sync drifts in the last second\n"
        "motion: 4/5\n"
        "sync: 3/5\n"
        "sharpness: 5/5\n"
    )
    scores, lines = loop_audio.parse_feedback(body)
    assert scores == {"motion": 4, "sync": 3, "sharp": 5}
    assert lines == ["Looking solid overall", "Sync drifts in the last second"]


def test_parse_feedback_preserves_zero_as_score():
    """0 is a legitimate score. Pre-fix the log row coerced it to '?'
    via `or '?'`; parse_feedback itself stores 0 correctly — this test
    pins that contract so a future refactor can't subtly inject `or 0`
    semantics."""
    scores, _ = loop_audio.parse_feedback("motion: 0/5\nsync: 0\nsharp: 0\n")
    assert scores == {"motion": 0, "sync": 0, "sharp": 0}


def test_render_log_row_preserves_zero_scores():
    row = loop_audio.render_log_row(
        3,
        _director(),
        "",
        "model-gen",
        {"segments": [], "audioSegments": []},
        {"motion": 0, "sync": 0, "sharp": 0},
        [],
    )
    assert "M0·S0·Sh0" in row


def test_render_log_row_records_video_and_audio_segment_counts():
    row = loop_audio.render_log_row(
        3,
        _director(),
        "",
        "custom",
        {"segments": [{}, {}], "audioSegments": [{}, {}]},
        {"motion": None, "sync": None, "sharp": None},
        [],
    )
    assert "| custom | 2v/2a |" in row


def test_parse_feedback_accepts_sharpness_alias():
    """Both 'sharp' and 'sharpness' are accepted axis names. Both map
    to the 'sharp' key."""
    scores, _ = loop_audio.parse_feedback("sharpness: 4\n")
    assert scores["sharp"] == 4


def test_parse_feedback_case_insensitive():
    scores, _ = loop_audio.parse_feedback("MOTION: 3\nSync 4\n")
    assert scores["motion"] == 3
    assert scores["sync"] == 4


def test_parse_feedback_empty_body_returns_blank_scores():
    scores, lines = loop_audio.parse_feedback("")
    assert scores == {"motion": None, "sync": None, "sharp": None}
    assert lines == []


# --- is_input_wired — None vs False semantics matter for derive_audio_state ---


def test_is_input_wired_none_when_director_missing():
    assert loop_audio.is_input_wired(None, "audio_vae") is None


def test_is_input_wired_true_when_link_set():
    node = {"inputs": [{"name": "audio_vae", "link": 42}]}
    assert loop_audio.is_input_wired(node, "audio_vae") is True


def test_is_input_wired_false_when_link_null():
    node = {"inputs": [{"name": "audio_vae", "link": None}]}
    assert loop_audio.is_input_wired(node, "audio_vae") is False


def test_is_input_wired_none_when_input_socket_absent():
    """An older Director schema might not have the named socket at
    all — same outcome as 'unwired' at runtime (no audio latent
    produced), but a distinct value here so callers can tell."""
    node = {"inputs": [{"name": "model", "link": 1}]}
    assert loop_audio.is_input_wired(node, "audio_vae") is None


# --- wrap_path — long mount paths shouldn't bleed past the card edge -------


def test_wrap_path_breaks_on_separator():
    out = loop_audio.wrap_path(
        "e:/G-Drive-BaconX/Jobs/Jeep_Animals/ComfyUI_LTX23/Phase2",
        max_chars=30,
    )
    # Joined with backslashes, never split mid-segment.
    for line in out:
        assert "/" not in line  # normalised to backslash separators
    joined = "".join(out)
    assert joined == "e:\\G-Drive-BaconX\\Jobs\\Jeep_Animals\\ComfyUI_LTX23\\Phase2"


def test_wrap_path_returns_single_empty_string_for_empty_input():
    assert loop_audio.wrap_path("") == [""]


def test_wrap_path_handles_single_segment_longer_than_max():
    """A directory name longer than max_chars still gets its own
    line — we never split mid-name."""
    out = loop_audio.wrap_path("verylongsingledirectoryname", max_chars=10)
    assert len(out) == 1
    assert out[0] == "verylongsingledirectoryname"


# --- pick_existing_path — picks reachable path, falls back to first ------


def test_pick_existing_path_prefers_real_directory(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    out = loop_audio.pick_existing_path([
        "Z:/never-exists",
        str(real),
        "Y:/also-never",
    ])
    assert out == str(real)


def test_pick_existing_path_falls_back_to_first_nonempty():
    out = loop_audio.pick_existing_path(["Z:/never-exists", "Y:/also-never"])
    assert out == "Z:/never-exists"


def test_pick_existing_path_returns_empty_when_all_empty():
    assert loop_audio.pick_existing_path(["", "   ", "\""]) == ""


# --- first_multiline + autogen_label sanity checks ------------------------


def test_first_multiline_returns_first_or_empty():
    assert loop_audio.first_multiline({"name": ["foo", "bar"]}, "name") == "foo"
    assert loop_audio.first_multiline({}, "name") == ""
    assert loop_audio.first_multiline({"name": []}, "name") == ""


def test_autogen_label_when_director_missing():
    label = loop_audio.autogen_label("Bear_3x", None, "")
    assert "missing" in label
    assert "audio-off" in label


def test_autogen_label_when_director_present():
    node = _director(use_custom_audio=True, audio_vae_link=42)
    label = loop_audio.autogen_label(
        "Bear_3x", node, '{"video_strength": 10.0}'
    )
    assert "koolook" in label
    assert "audio-on" in label
    assert "vstr10.0" in label


def test_autogen_label_when_director_is_upstream():
    label = loop_audio.autogen_label(
        "Bear_3x", _director(node_type="LTXDirector"), ""
    )
    assert "upstream" in label
    assert "audio-off" in label


def test_relay_overrides_txt_marks_upstream_director_inert():
    txt = loop_audio.render_relay_overrides_txt(
        '{"video_strength": 10.0}',
        _director(node_type="LTXDirector"),
    )
    assert "INERT" in txt
    assert "LTXDirector" in txt


def test_next_run_number_reads_folders_and_log(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "run-001_alpha").mkdir()
    (runs / "log.md").write_text(
        "| Run | Date |\n"
        "|---|---|\n"
        "| 002 | 2026-05-29 |\n",
        encoding="utf-8",
    )
    assert loop_audio.next_run_number(runs) == 3
