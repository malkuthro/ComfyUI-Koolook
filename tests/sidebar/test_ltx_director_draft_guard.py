"""Frontend contract checks for the LTX Director timeline extension.

As of the v2.0.2 fork, web/whatdreamscost_koolook/ltx_director.js is the
upstream 2.0.2 timeline editor vendored from a pinned commit and retargeted to
the Koolook node id. One Koolook media-scrub delta is deliberately reinstated
on top: upstream 2.0.2 persists media via imageFile/audioFile + /api/view URLs
*but also* stamps a base64 poster JPEG onto every video segment
(canvas.toDataURL) and keeps it across serialize round-trips, so large
timelines blow the browser draft-storage quota (endless "Failed to save
workflow draft" toasts on frontend 1.44+). The scrub drops those rebuildable
blobs from persisted timeline_data; the localStorage quota guard itself still
lives exclusively in the global web/koolook_draft_guard.js.
"""
from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LTX_DIRECTOR_JS = REPO_ROOT / "web" / "whatdreamscost_koolook" / "ltx_director.js"

requires_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not on PATH"
)


def run_node_scenario(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", "--input-type=module"],
        input=source,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_ltx_director_does_not_embed_the_draft_guard() -> None:
    """The quota guard lives in web/koolook_draft_guard.js (all keys, every
    page). The node editor must not re-install a localStorage wrapper or it
    would double-install and drift behind the global guard's key coverage.
    """
    source = LTX_DIRECTOR_JS.read_text(encoding="utf-8")

    assert "installComfyDraftQuotaGuard" not in source
    assert "pruneComfyDraftCache" not in source
    assert "Comfy.Workflow.Drafts" not in source
    assert "localStorage.setItem =" not in source
    assert "web/koolook_draft_guard.js" in source  # pointer for archaeology


def test_ltx_director_js_is_retargeted_to_the_koolook_node() -> None:
    """The vendored upstream editor must bind to the Koolook node id under a
    unique extension name, never the bare upstream `LTXDirector` (that would
    collide with the installed upstream WhatDreamsCost extension).
    """
    source = LTX_DIRECTOR_JS.read_text(encoding="utf-8")

    assert 'name: "LTXDirector_Koolook"' in source
    assert 'nodeData.name === "LTXDirector__koolook"' in source
    # must not bind the bare upstream node type
    assert 'nodeData.name === "LTXDirector"' not in source
    assert 'name: "LTXDirector",' not in source


def test_ltx_director_js_is_vendored_from_pinned_upstream() -> None:
    """Provenance header pins the exact upstream commit it was vendored from,
    so the editor version is auditable against forks_manifest.yaml.
    """
    source = LTX_DIRECTOR_JS.read_text(encoding="utf-8")

    assert "Vendored VERBATIM from WhatDreamsCost-ComfyUI" in source
    assert "fe09f73756df202d08341c66b4dc5fc8d2acca22" in source


def test_every_timeline_serialize_and_restore_path_applies_the_scrub() -> None:
    """commitChanges persists three segment arrays and parseInitial restores
    three; each map must route through the scrub or one path re-accumulates
    base64 posters and the draft-quota failure returns. (The disk Save /
    Save As payload in _getTimelineSavePayload is deliberately unscrubbed —
    exported .json files keep upstream's format and meet the restore scrub
    on load.)
    """
    source = LTX_DIRECTOR_JS.read_text(encoding="utf-8")

    assert source.count("return scrubTimelinePreviewMedia(rest);") == 6


def _scrub_harness_prelude() -> str:
    """Node prelude that extracts the KOOLOOK DELTA block from the vendored
    editor (which is not an ES module and needs `api` from the ComfyUI page)
    and evaluates it against a stubbed `api`.
    """
    return textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { readFileSync } from "node:fs";

        const src = readFileSync("./web/whatdreamscost_koolook/ltx_director.js", "utf8");
        const startMark = src.indexOf("--- KOOLOOK DELTA: draft-quota scrub");
        const endMark = src.indexOf("--- end KOOLOOK DELTA");
        assert.ok(startMark > 0 && endMark > startMark, "delta block markers missing");
        const block = src.slice(src.lastIndexOf("\\n", startMark) + 1, src.lastIndexOf("\\n", endMark));

        const api = { apiURL: (p) => "/api" + p };
        const { scrub, persist } = new Function(
          "api",
          block + "\\nreturn { scrub: scrubTimelinePreviewMedia, persist: persistScrubbedTimelineWidget };",
        )(api);
        """
    )


@requires_node
def test_scrub_drops_rebuildable_poster_blobs_and_keeps_everything_else() -> None:
    script = _scrub_harness_prelude() + textwrap.dedent(
        """
        // Main-track video poster (path lives in imageFile): rebuildable
        // from the video element, dropped.
        const video = scrub({
          type: "video", imageFile: "whatdreamscost/clip.mp4",
          imageB64: "data:image/jpeg;base64,AAAA", prompt: "keep me",
        });
        assert.equal(video.imageB64, undefined);
        assert.equal(video.prompt, "keep me");

        // Motion-track video poster (path lives in videoFile, imageFile is
        // never set on motion segments): equally rebuildable, dropped.
        const motion = scrub({
          type: "motion_video", videoFile: "whatdreamscost/mo.mp4",
          imageB64: "data:image/jpeg;base64,BBBB",
        });
        assert.equal(motion.imageB64, undefined);

        // 2.0.2 image segments store a /view URL in imageB64: untouched.
        const image = scrub({
          type: "image", imageFile: "whatdreamscost/pic.png",
          imageB64: "/api/view?filename=pic.png",
        });
        assert.equal(image.imageB64, "/api/view?filename=pic.png");

        // Data-URL without a backing file: nothing could rebuild it, keep it.
        const orphanImage = scrub({
          type: "image", imageB64: "data:image/png;base64,CCCC",
        });
        assert.equal(orphanImage.imageB64, "data:image/png;base64,CCCC");
        const orphanVideo = scrub({
          type: "video", imageB64: "data:image/jpeg;base64,DDDD",
        });
        assert.equal(orphanVideo.imageB64, "data:image/jpeg;base64,DDDD");

        // Legacy inline audio is raw base64 (no data: prefix) and only a
        // fallback when no file resolves: dropped when audioFile exists,
        // kept when it is the only source.
        const audio = scrub({
          type: "audio", audioFile: "whatdreamscost/voice.wav",
          audioB64: "UklGRg==",
        });
        assert.equal(audio.audioB64, undefined);
        const orphanAudio = scrub({ type: "audio", audioB64: "UklGRg==" });
        assert.equal(orphanAudio.audioB64, "UklGRg==");
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


@requires_node
def test_scrub_rewrites_legacy_image_blobs_to_view_urls() -> None:
    script = _scrub_harness_prelude() + textwrap.dedent(
        """
        // Legacy pre-2.0.2 image segment: full data-URL replaced by the
        // /view URL loadMedia() can render from (deleting would blank it).
        const image = scrub({
          type: "image", imageFile: "whatdreamscost/pic.png",
          imageB64: "data:image/png;base64,EEEE",
        });
        assert.equal(
          image.imageB64,
          "/api/view?filename=pic.png&type=input&subfolder=whatdreamscost",
        );

        // Legacy Windows-style backslash paths normalize to the same URL.
        const backslash = scrub({
          type: "image", imageFile: "whatdreamscost\\\\pic.png",
          imageB64: "data:image/png;base64,FFFF",
        });
        assert.equal(
          backslash.imageB64,
          "/api/view?filename=pic.png&type=input&subfolder=whatdreamscost",
        );

        // A malformed path yields no /view URL: keep the data-URL rather than
        // blanking the preview with an empty string.
        const malformed = scrub({
          type: "image", imageFile: "whatdreamscost/",
          imageB64: "data:image/png;base64,GGGG",
        });
        assert.equal(malformed.imageB64, "data:image/png;base64,GGGG");
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


@requires_node
def test_restore_rewrites_the_widget_so_fat_payloads_never_reach_the_draft() -> None:
    """parseInitial scrubs the in-memory timeline, but onSerialize (and
    LiteGraph's widgets_values, which the draft autosave serializes) reads the
    widget. A legacy fat payload must be rewritten on restore, before any edit
    triggers commitChanges.
    """
    script = _scrub_harness_prelude() + textwrap.dedent(
        """
        // Fat legacy payload in the widget -> rewritten from the scrubbed copy.
        const fat = { segments: [{ type: "video", imageFile: "a/clip.mp4" }] };
        const widget = {
          value: JSON.stringify({
            segments: [{
              type: "video", imageFile: "a/clip.mp4",
              imageB64: "data:image/jpeg;base64," + "A".repeat(50000),
            }],
          }),
        };
        const before = widget.value.length;
        persist(widget, fat);
        assert.ok(widget.value.length < before / 10, "widget was not slimmed");
        assert.ok(!widget.value.includes("data:"), "data-URL survived");
        assert.deepEqual(JSON.parse(widget.value), fat);

        // Already-clean widget: untouched (no needless churn on normal loads).
        const clean = { value: '{"segments":[]}' };
        persist(clean, { segments: [{ injected: true }] });
        assert.equal(clean.value, '{"segments":[]}');

        // Defensive: missing widget / non-string value must not throw.
        persist(null, fat);
        persist({ value: undefined }, fat);

        // Unserializable timeline leaves the original value intact.
        const cyclic = {};
        cyclic.self = cyclic;
        const survivor = { value: '{"x":"data:image/png;base64,AAAA"}' };
        persist(survivor, cyclic);
        assert.equal(survivor.value, '{"x":"data:image/png;base64,AAAA"}');
        """
    )

    result = run_node_scenario(script)
    assert result.returncode == 0, result.stderr


def test_both_restore_paths_sync_the_widget() -> None:
    """The editor constructor and the onConfigure sync block both restore from
    the widget; each must write the scrubbed copy back or the fat payload
    survives on that path.
    """
    source = LTX_DIRECTOR_JS.read_text(encoding="utf-8")

    assert source.count("persistScrubbedTimelineWidget(") == 3  # 1 def + 2 calls
