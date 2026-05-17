from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_JS = REPO_ROOT / "web" / "sidebar" / "snapshot.js"
WORKFLOWS_STORE_JS = REPO_ROOT / "web" / "sidebar" / "workflows_store.js"
CONSTANTS_JS = REPO_ROOT / "web" / "sidebar" / "constants.js"


def test_cache_bust_helper_is_shared() -> None:
    constants = CONSTANTS_JS.read_text(encoding="utf-8")
    snapshot = SNAPSHOT_JS.read_text(encoding="utf-8")
    workflows = WORKFLOWS_STORE_JS.read_text(encoding="utf-8")

    assert "export function noStoreUrl(" in constants
    assert "function noStoreUrl(" not in snapshot
    assert "function noStoreUrl(" not in workflows
    assert "noStoreUrl," in snapshot
    assert "noStoreUrl," in workflows


def test_mutable_snapshot_reads_bypass_browser_cache() -> None:
    text = SNAPSHOT_JS.read_text(encoding="utf-8")

    mutable_read_patterns = [
        "fetch(\n            noStoreUrl(`${ROUTE_FILE}?name=${encodeURIComponent(fullName)}${dirParam}`),\n            { cache: \"no-store\" }\n        )",
        "fetch(noStoreUrl(ROUTE_INFO), { cache: \"no-store\" })",
        "fetch(noStoreUrl(ROUTE_SETTINGS), { cache: \"no-store\" })",
        "fetch(noStoreUrl(`${ROUTE_BROWSE}${q ? `?${q}` : \"\"}`), {\n        cache: \"no-store\",\n    })",
        "fetch(noStoreUrl(url), { cache: \"no-store\" })",
        "fetch(noStoreUrl(`${ROUTE_FILE}${fileQuery(fileName, dir)}`), {\n        cache: \"no-store\",\n    })",
        "fetch(noStoreUrl(`${ROUTE_FILE}${fileQuery(fileName, dir)}`), {\n            method: \"HEAD\",\n            cache: \"no-store\",\n        })",
        "fetch(noStoreUrl(ROUTE_AUTOSAVES_LIST), { cache: \"no-store\" })",
        "fetch(\n                noStoreUrl(`${ROUTE_FILE}?name=${encodeURIComponent(row.name)}${dirParam}`),\n                { cache: \"no-store\" }\n            )",
    ]

    for pattern in mutable_read_patterns:
        assert pattern in text


def test_workflow_startup_read_bypasses_browser_cache_in_source() -> None:
    text = WORKFLOWS_STORE_JS.read_text(encoding="utf-8")

    assert (
        "fetch(noStoreUrl(`/userdata/${WORKFLOWS_USERDATA_PATH}`), {\n"
        "            cache: \"no-store\",\n"
        "        })"
    ) in text
