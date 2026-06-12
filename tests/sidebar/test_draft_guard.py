"""Static contracts for the global Comfy draft-storage guard.

Behavior is covered by tests/js/test_draft_guard.mjs (run by CI's node
step); the checks here pin the integration surfaces the node suite cannot
see: which file ships the guard, the key-prefix contract that makes it
generation-agnostic, that the scoped audio dev-sync still carries it, and
that CI actually runs the node suite.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD_JS = REPO_ROOT / "web" / "koolook_draft_guard.js"


def test_guard_is_generation_agnostic() -> None:
    source = GUARD_JS.read_text(encoding="utf-8")

    # Prefix matching is the future-proofing contract: every known draft
    # key generation starts with this prefix, and unknown future keys under
    # it degrade to last-resort eviction instead of being invisible (which
    # is how the previous exact-name guard went stale on frontend 1.44).
    assert 'DRAFT_KEY_PREFIX = "Comfy.Workflow.Draft"' in source
    assert 'V2_INDEX_PREFIX = "Comfy.Workflow.DraftIndex.v2:"' in source
    assert 'V2_PAYLOAD_PREFIX = "Comfy.Workflow.Draft.v2:"' in source
    # Multi-install collision sentinel: older guard copies (pre-move
    # releases still embed one in ltx_director.js) check the same flag, so
    # exactly one wrapper installs no matter which copy loads first.
    assert "__koolookDraftQuotaGuardInstalled" in source
    # Both mechanisms must run at page load.
    assert "installComfyDraftQuotaGuard();" in source
    assert "pruneComfyDraftCache();" in source
    # Multi-browser quota error detection (Chrome name, Firefox name/codes).
    assert "QuotaExceededError" in source
    assert "NS_ERROR_DOM_QUOTA_REACHED" in source


def test_scoped_audio_sync_ships_the_guard() -> None:
    """dev-sync-audio replaces web/whatdreamscost_koolook/, which used to
    embed the guard. Shipping the global guard file alongside keeps a
    scoped sync from stranding a dev install with no guard at all.
    """
    script = REPO_ROOT / "scripts" / "sync_to_dev_audio.py"
    spec = importlib.util.spec_from_file_location(
        "sync_to_dev_audio_for_draft_guard_test", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert "web/koolook_draft_guard.js" in module.AUDIO_PATHS


def test_ci_runs_the_node_behavior_suite() -> None:
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "node tests/js/test_draft_guard.mjs" in ci
