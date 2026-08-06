"""Contract tests for the public Registry metrics summary."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "collect_registry_metrics.py"
SPEC = importlib.util.spec_from_file_location("collect_registry_metrics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
collector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = collector
SPEC.loader.exec_module(collector)


def _row(date: str, downloads: int, delta: int, version: str = "0.4.4") -> dict[str, str]:
    return {
        "date": date,
        "timestamp_utc": f"{date}T08:00:00Z",
        "downloads_total": str(downloads),
        "daily_delta": str(delta),
        "rolling_7d_avg": "0.0",
        "latest_version": version,
        "latest_status": "Active",
        "pending_versions": "",
        "active_versions": version,
    }


def test_summary_matches_the_v030_landing_page_contract():
    rows = [
        _row("2026-08-04", 25315, 15),
        _row("2026-08-05", 26242, 927),
        _row("2026-08-06", 26870, 628),
    ]

    assert collector._make_summary(rows) == {
        "asOf": "2026-08-06",
        "sinceVersion": "0.3.0",
        "downloadsSince": 26775,
        "days": 90,
        "avgPerDay": 297.5,
        "peakDate": "2026-08-05",
        "peakDelta": 927,
        "latestVersion": "0.4.4",
    }


def test_summary_is_pretty_printed_json(tmp_path: Path):
    summary = collector._make_summary([_row("2026-08-06", 26870, 628)])
    output = tmp_path / "summary.json"

    collector._write_summary(output, summary)

    assert json.loads(output.read_text(encoding="utf-8")) == summary
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_registry_metrics_workflow_commits_the_summary():
    workflow = (REPO_ROOT / ".github" / "workflows" / "registry-metrics.yml").read_text(
        encoding="utf-8"
    )

    assert "metrics/summary.json" in workflow
