"""Frontend contract checks for the LTX Director timeline extension."""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LTX_DIRECTOR_JS = REPO_ROOT / "web" / "whatdreamscost_koolook" / "ltx_director.js"


def test_ltx_director_no_longer_embeds_the_draft_guard() -> None:
    """The quota guard moved to web/koolook_draft_guard.js (all key
    generations, every page). A stale copy here would double-install the
    localStorage wrapper and drift behind the global guard's key coverage,
    so its absence is a contract, not an accident.
    """
    source = LTX_DIRECTOR_JS.read_text(encoding="utf-8")

    assert "installComfyDraftQuotaGuard" not in source
    assert "pruneComfyDraftCache" not in source
    assert "Comfy.Workflow.Drafts" not in source
    assert "localStorage.setItem =" not in source
    assert "web/koolook_draft_guard.js" in source  # pointer for archaeology


def test_timeline_serialization_drops_preview_only_media() -> None:
    source = LTX_DIRECTOR_JS.read_text(encoding="utf-8")

    assert "function serializeTimelineSegment(seg)" in source
    assert "imageB64" in source
    assert "audioB64" in source
    assert "segments: sortedSegments.map(s => serializeTimelineSegment(s))" in source
    assert "audioSegments: (this.timeline.audioSegments || []).map(s => serializeTimelineSegment(s))" in source


def test_timeline_rehydrates_image_previews_from_saved_image_file() -> None:
    source = LTX_DIRECTOR_JS.read_text(encoding="utf-8")

    assert "function inputImageUrl(imageFile)" in source
    assert "function imagePreviewSrc(seg)" in source
    assert "function scrubPersistedPreviewMedia(seg)" in source
    assert 'replace(/\\\\/g, "/")' in source
    assert "if (seg.imageFile && seg.imageB64) delete seg.imageB64;" in source
    assert "if (seg.audioFile && seg.audioB64) delete seg.audioB64;" in source
    assert "p.segments.map(scrubPersistedPreviewMedia)" in source
    assert "p.audioSegments.map(scrubPersistedPreviewMedia)" in source
    assert "const src = imagePreviewSrc(seg);" in source
    assert "seg._imagePreviewSrc = src;" in source
    assert "seg.imgObj.src = src;" in source
