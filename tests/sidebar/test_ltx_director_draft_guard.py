"""Frontend contract checks for the LTX Director timeline extension."""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LTX_DIRECTOR_JS = REPO_ROOT / "web" / "whatdreamscost_koolook" / "ltx_director.js"


def test_ltx_director_prunes_comfy_draft_cache() -> None:
    source = LTX_DIRECTOR_JS.read_text(encoding="utf-8")

    assert 'COMFY_DRAFTS_KEY = "Comfy.Workflow.Drafts"' in source
    assert "function pruneComfyDraftCache()" in source
    assert "function installComfyDraftQuotaGuard()" in source
    assert "function evictOldestComfyDraft(originalSetItem)" in source
    assert "function showComfyDraftQuotaWarning(message)" in source
    assert "localStorage.setItem = (key, value) =>" in source
    assert 'err.name === "QuotaExceededError"' in source
    assert "evictOldestComfyDraft(originalSetItem)" in source
    assert "Koolook removed the oldest draft only" in source
    assert 'originalSetItem(COMFY_DRAFTS_KEY, "{}")' not in source
    assert "installComfyDraftQuotaGuard();" in source
    assert "pruneComfyDraftCache();" in source
    assert "MAX_COMFY_DRAFT_CACHE_CHARS" in source


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
