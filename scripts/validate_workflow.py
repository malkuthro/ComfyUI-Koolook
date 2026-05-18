"""Validate a ComfyUI workflow JSON for internal consistency.

Usage:
    python scripts/validate_workflow.py <workflow.json> [--quiet]

Checks performed:
- The file parses as JSON.
- Every link referenced by node inputs/outputs is declared in the top-level
  `links` table.
- Every entry in the `links` table is referenced by both endpoints (no
  orphans).
- For each link, the source/destination nodes and slots exist and the slot
  type matches the declared link type.
- `last_node_id >= max(node ids)` and `last_link_id >= max(link ids)`.
- No duplicate node IDs or link IDs.

Exit codes:
    0  All checks passed.
    1  Problems found (printed to stderr).
    2  File not found or not valid JSON.

This catches the class of bugs that come from hand-editing workflow JSONs
(e.g. when an agent re-wires a graph to build a slim variant). Run it after
any non-trivial edit before re-loading the workflow into ComfyUI — Comfy's
loader will sometimes silently drop or mis-wire the bad parts, producing a
graph that looks fine but behaves wrong.

stdlib only, no third-party dependencies.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def validate(data: dict) -> list[str]:
    """Return a list of problem strings; empty list means OK.

    Takes a parsed workflow dict (so callers can reuse already-loaded data).
    """
    problems: list[str] = []

    nodes = data.get("nodes") or []
    links = data.get("links") or []

    # Duplicate IDs.
    node_ids = [n.get("id") for n in nodes]
    link_ids = [link[0] for link in links if isinstance(link, list) and link]
    dup_nodes = sorted({i for i in node_ids if node_ids.count(i) > 1})
    dup_links = sorted({i for i in link_ids if link_ids.count(i) > 1})
    if dup_nodes:
        problems.append(f"Duplicate node IDs: {dup_nodes}")
    if dup_links:
        problems.append(f"Duplicate link IDs: {dup_links}")

    # Header counters should bound observed IDs (Comfy uses these to assign
    # the next id when adding nodes; if they lag, the next add can collide).
    if node_ids and data.get("last_node_id", -1) < max(node_ids):
        problems.append(
            f"last_node_id={data.get('last_node_id')} < max(node id)={max(node_ids)}"
        )
    if link_ids and data.get("last_link_id", -1) < max(link_ids):
        problems.append(
            f"last_link_id={data.get('last_link_id')} < max(link id)={max(link_ids)}"
        )

    # Cross-reference link declarations vs node IO references.
    declared = set(link_ids)
    referenced: set[int] = set()
    for n in nodes:
        for inp in n.get("inputs") or []:
            if isinstance(inp.get("link"), int):
                referenced.add(inp["link"])
        for out in n.get("outputs") or []:
            for lk in out.get("links") or []:
                if isinstance(lk, int):
                    referenced.add(lk)

    for missing in sorted(referenced - declared):
        problems.append(
            f"link {missing}: referenced by a node but not in the links table"
        )
    for orphan in sorted(declared - referenced):
        problems.append(
            f"link {orphan}: declared in the links table but unused by any node"
        )

    # Per-link endpoint validation.
    node_by_id = {n.get("id"): n for n in nodes}
    for link in links:
        if not isinstance(link, list) or len(link) != 6:
            problems.append(
                f"link {link!r}: malformed — expected [id, src, src_slot, dst, dst_slot, type]"
            )
            continue
        lid, src, src_slot, dst, dst_slot, ltype = link
        src_node = node_by_id.get(src)
        dst_node = node_by_id.get(dst)
        if src_node is None:
            problems.append(f"link {lid}: src node {src} missing")
            continue
        if dst_node is None:
            problems.append(f"link {lid}: dst node {dst} missing")
            continue
        src_outs = src_node.get("outputs") or []
        dst_ins = dst_node.get("inputs") or []

        # Slot index bounds + type checks.
        if not isinstance(src_slot, int) or src_slot < 0 or src_slot >= len(src_outs):
            problems.append(
                f"link {lid}: src slot {src_slot} out of bounds on node {src} "
                f"(has {len(src_outs)} outputs)"
            )
        elif src_outs[src_slot].get("type") != ltype:
            problems.append(
                f"link {lid}: src type mismatch — link declares {ltype!r}, "
                f"slot is {src_outs[src_slot].get('type')!r}"
            )
        if not isinstance(dst_slot, int) or dst_slot < 0 or dst_slot >= len(dst_ins):
            problems.append(
                f"link {lid}: dst slot {dst_slot} out of bounds on node {dst} "
                f"(has {len(dst_ins)} inputs)"
            )
        elif dst_ins[dst_slot].get("type") != ltype:
            problems.append(
                f"link {lid}: dst type mismatch — link declares {ltype!r}, "
                f"slot is {dst_ins[dst_slot].get('type')!r}"
            )

        # Endpoint cross-check: node IO must reference this link id symmetrically.
        if isinstance(src_slot, int) and 0 <= src_slot < len(src_outs):
            out_links = src_outs[src_slot].get("links") or []
            if lid not in out_links:
                problems.append(
                    f"link {lid}: missing from node {src} output[{src_slot}].links={out_links}"
                )
        if isinstance(dst_slot, int) and 0 <= dst_slot < len(dst_ins):
            in_link = dst_ins[dst_slot].get("link")
            if in_link != lid:
                problems.append(
                    f"link {lid}: node {dst} input[{dst_slot}].link={in_link} "
                    f"does not match"
                )

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to the workflow JSON file.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress the OK summary; only print problems.",
    )
    args = parser.parse_args(argv)

    if not args.path.exists():
        print(f"error: file not found: {args.path}", file=sys.stderr)
        return 2

    try:
        data = json.loads(args.path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"error: {args.path.name}: JSON parse error: {exc}", file=sys.stderr)
        return 2

    problems = validate(data)
    if problems:
        print(
            f"FAIL: {args.path.name} ({len(problems)} problem(s))",
            file=sys.stderr,
        )
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(
            f"OK: {args.path.name} — "
            f"{len(data.get('nodes') or [])} nodes, "
            f"{len(data.get('links') or [])} links, "
            f"{len(data.get('groups') or [])} groups"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
