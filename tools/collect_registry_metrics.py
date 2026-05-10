#!/usr/bin/env python3
"""Collect Comfy Registry download trend snapshots.

The Comfy Registry currently exposes downloads at the node/package level,
not per version. This script records that package-level total over time and
adds version/status context so we can see when a new release becomes active.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_ROOT = "https://api.comfy.org"
NODE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
CSV_FIELDS = [
    "date",
    "timestamp_utc",
    "downloads_total",
    "daily_delta",
    "rolling_7d_avg",
    "latest_version",
    "latest_status",
    "pending_versions",
    "active_versions",
]


@dataclass(frozen=True)
class Snapshot:
    date: str
    timestamp_utc: str
    downloads_total: int
    daily_delta: int
    rolling_7d_avg: str
    latest_version: str
    latest_status: str
    pending_versions: str
    active_versions: str

    def as_row(self) -> dict[str, str]:
        return {
            "date": self.date,
            "timestamp_utc": self.timestamp_utc,
            "downloads_total": str(self.downloads_total),
            "daily_delta": str(self.daily_delta),
            "rolling_7d_avg": self.rolling_7d_avg,
            "latest_version": self.latest_version,
            "latest_status": self.latest_status,
            "pending_versions": self.pending_versions,
            "active_versions": self.active_versions,
        }


def _fetch_json(url: str) -> Any:
    if not url.startswith(f"{API_ROOT}/"):
        raise ValueError(f"refusing non-registry URL: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "koolook-metrics/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:  # nosec B310
        return json.loads(response.read().decode("utf-8"))


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_rows(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _version_status(version: dict[str, Any]) -> str:
    return str(version.get("status", "")).replace("NodeVersionStatus", "")


def _rolling_average(rows: list[dict[str, str]]) -> str:
    deltas = [int(row.get("daily_delta") or 0) for row in rows[-7:]]
    if not deltas:
        return "0.0"
    return f"{sum(deltas) / len(deltas):.1f}"


def _make_snapshot(node: dict[str, Any], versions: list[dict[str, Any]], rows: list[dict[str, str]]) -> Snapshot:
    now = datetime.now(timezone.utc)
    downloads = int(node.get("downloads") or 0)
    previous_downloads = int(rows[-1]["downloads_total"]) if rows else downloads
    daily_delta = downloads - previous_downloads

    latest = node.get("latest_version") or {}
    pending = [
        str(item.get("version", ""))
        for item in versions
        if item.get("status") == "NodeVersionStatusPending"
    ]
    active = [
        str(item.get("version", ""))
        for item in versions
        if item.get("status") == "NodeVersionStatusActive"
    ]

    candidate = Snapshot(
        date=now.date().isoformat(),
        timestamp_utc=now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        downloads_total=downloads,
        daily_delta=daily_delta,
        rolling_7d_avg="0.0",
        latest_version=str(latest.get("version", "")),
        latest_status=_version_status(latest),
        pending_versions=", ".join(pending),
        active_versions=", ".join(active),
    )
    average_rows = [*rows, candidate.as_row()]
    return Snapshot(
        **{
            **candidate.as_row(),
            "downloads_total": candidate.downloads_total,
            "daily_delta": candidate.daily_delta,
            "rolling_7d_avg": _rolling_average(average_rows),
        }
    )


def _upsert_today(rows: list[dict[str, str]], snapshot: Snapshot) -> list[dict[str, str]]:
    today = snapshot.date
    kept = [row for row in rows if row.get("date") != today]
    return [*kept, snapshot.as_row()]


def _render_overview(node_id: str, rows: list[dict[str, str]]) -> str:
    if not rows:
        return f"# Comfy Registry Metrics: {node_id}\n\nNo snapshots recorded yet.\n"

    latest = rows[-1]
    recent = rows[-14:]
    table = "\n".join(
        "| {date} | {downloads_total} | {daily_delta} | {rolling_7d_avg} | {latest_version} | {latest_status} | {pending_versions} |".format(**row)
        for row in reversed(recent)
    )
    return f"""# Comfy Registry Metrics: {node_id}

Last updated: `{latest["timestamp_utc"]}`

## Current

- Total downloads: **{latest["downloads_total"]}**
- Latest active version: **{latest["latest_version"]}** ({latest["latest_status"]})
- Pending versions: **{latest["pending_versions"] or "none"}**
- Last recorded daily change: **{latest["daily_delta"]}**
- Rolling 7-day average: **{latest["rolling_7d_avg"]}/day**

## Recent Trend

| Date | Total | Daily change | 7-day avg | Latest | Status | Pending |
|---|---:|---:|---:|---|---|---|
{table}

Note: Comfy Registry exposes downloads at package level, not per version.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", default="koolook")
    parser.add_argument("--output-dir", default="metrics")
    args = parser.parse_args()
    if not NODE_ID_RE.fullmatch(args.node):
        parser.error("--node must be a Comfy Registry node id")

    output_dir = Path(args.output_dir)
    csv_path = output_dir / "comfy-registry-downloads.csv"
    overview_path = output_dir / "README.md"

    node = _fetch_json(f"{API_ROOT}/nodes/{args.node}")
    versions = _fetch_json(f"{API_ROOT}/nodes/{args.node}/versions")
    rows = _read_rows(csv_path)
    snapshot = _make_snapshot(node, versions, rows)
    updated_rows = _upsert_today(rows, snapshot)

    _write_rows(csv_path, updated_rows)
    overview_path.write_text(_render_overview(args.node, updated_rows), encoding="utf-8")

    print(
        f"{snapshot.date}: downloads={snapshot.downloads_total} "
        f"delta={snapshot.daily_delta} pending={snapshot.pending_versions or 'none'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
