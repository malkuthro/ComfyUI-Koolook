"""Frontend contract checks for the LTX Director timeline extension.

As of the v2.0.2 fork, web/whatdreamscost_koolook/ltx_director.js is the
upstream 2.0.2 timeline editor vendored VERBATIM and retargeted to the Koolook
node id, rather than the old (1.3.9-era) editor carrying Koolook media-scrub
deltas. Upstream 2.0.2 persists media via imageFile/audioFile + /api/view URLs,
so those scrub deltas are obsolete; the localStorage quota guard still lives
exclusively in the global web/koolook_draft_guard.js.
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LTX_DIRECTOR_JS = REPO_ROOT / "web" / "whatdreamscost_koolook" / "ltx_director.js"


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
