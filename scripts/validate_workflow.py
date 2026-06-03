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


def _sanitize_slots(
    node: dict, key: str, problems: list[str]
) -> list[dict]:
    """Return a list-of-dicts view of node[key], appending a problem for any
    malformed entry. Non-list `node[key]` → reported and treated as empty.
    Non-dict entries → reported and dropped.
    """
    slots = node.get(key)
    if slots is None:
        return []
    if not isinstance(slots, list):
        problems.append(
            f"node {node.get('id')!r}: {key!r} must be a list, got {type(slots).__name__}"
        )
        return []
    cleaned: list[dict] = []
    for j, slot in enumerate(slots):
        if not isinstance(slot, dict):
            problems.append(
                f"node {node.get('id')!r}: {key}[{j}] must be an object, got {type(slot).__name__}"
            )
            continue
        cleaned.append(slot)
    return cleaned


def _slot_type_matches(slot_type, link_type) -> bool:
    """ComfyUI wildcard sockets (`*`) accept any concrete link type."""
    return slot_type == link_type or slot_type == "*"


def validate(data) -> list[str]:
    """Return a list of problem strings; empty list means OK.

    Defensive: every shape assumption is guarded so that malformed-but-valid
    JSON produces problem lines, never an uncaught exception. Top-level not
    being an object is the only short-circuit return (nothing else is safe
    to do without a dict).
    """
    problems: list[str] = []

    # Top-level must be a JSON object.
    if not isinstance(data, dict):
        return [
            f"top-level JSON must be an object, got {type(data).__name__}"
        ]

    # `nodes` and `links` arrays — must be lists; non-list reported and skipped.
    raw_nodes = data.get("nodes") or []
    if not isinstance(raw_nodes, list):
        problems.append(
            f"'nodes' must be a list, got {type(raw_nodes).__name__}"
        )
        raw_nodes = []

    raw_links = data.get("links") or []
    if not isinstance(raw_links, list):
        problems.append(
            f"'links' must be a list, got {type(raw_links).__name__}"
        )
        raw_links = []

    # Filter to well-shaped node entries, recording each bad one.
    nodes: list[dict] = []
    for i, n in enumerate(raw_nodes):
        if not isinstance(n, dict):
            problems.append(
                f"nodes[{i}]: must be an object, got {type(n).__name__}"
            )
            continue
        nodes.append(n)

    # Filter to well-shaped link entries. Each must be a 6-element list.
    links: list[list] = []
    for i, link in enumerate(raw_links):
        if not isinstance(link, list):
            problems.append(
                f"links[{i}]: must be a list of 6 elements, got {type(link).__name__}"
            )
            continue
        if len(link) != 6:
            problems.append(
                f"links[{i}]: malformed — expected 6 elements "
                f"[id, src, src_slot, dst, dst_slot, type], got {len(link)}"
            )
            continue
        links.append(link)

    # Pre-sanitize each node's inputs/outputs into list-of-dict views so
    # downstream code can index safely. Caches per-node so problems aren't
    # double-reported when a node is touched by multiple links.
    sanitized_io: dict[int, dict[str, list[dict]]] = {}
    for n in nodes:
        sanitized_io[id(n)] = {
            "inputs": _sanitize_slots(n, "inputs", problems),
            "outputs": _sanitize_slots(n, "outputs", problems),
        }

    # Duplicate IDs (over good entries).
    node_ids = [n.get("id") for n in nodes]
    link_ids = [link[0] for link in links]
    dup_nodes = sorted({i for i in node_ids if node_ids.count(i) > 1})
    dup_links = sorted({i for i in link_ids if link_ids.count(i) > 1})
    if dup_nodes:
        problems.append(f"Duplicate node IDs: {dup_nodes}")
    if dup_links:
        problems.append(f"Duplicate link IDs: {dup_links}")

    # Header counters should bound observed IDs (Comfy uses these to assign
    # the next id when adding nodes; if they lag, the next add can collide).
    int_node_ids = [i for i in node_ids if isinstance(i, int)]
    int_link_ids = [i for i in link_ids if isinstance(i, int)]
    if int_node_ids:
        last_node_id = data.get("last_node_id", -1)
        if not isinstance(last_node_id, int) or last_node_id < max(int_node_ids):
            problems.append(
                f"last_node_id={last_node_id!r} < max(node id)={max(int_node_ids)}"
            )
    if int_link_ids:
        last_link_id = data.get("last_link_id", -1)
        if not isinstance(last_link_id, int) or last_link_id < max(int_link_ids):
            problems.append(
                f"last_link_id={last_link_id!r} < max(link id)={max(int_link_ids)}"
            )

    # Cross-reference link declarations vs node IO references.
    declared = set(int_link_ids)
    referenced: set[int] = set()
    for n in nodes:
        io = sanitized_io[id(n)]
        for inp in io["inputs"]:
            link_val = inp.get("link")
            if isinstance(link_val, int):
                referenced.add(link_val)
        for out in io["outputs"]:
            out_links = out.get("links") or []
            if not isinstance(out_links, list):
                problems.append(
                    f"node {n.get('id')!r}: output.links must be a list, "
                    f"got {type(out_links).__name__}"
                )
                continue
            for lk in out_links:
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
    # Build node_by_id over int ids only — non-int ids can't match a link's
    # src/dst (which are ints in the link tuple), and treating them as keys
    # would mask bad data.
    node_by_id = {n.get("id"): n for n in nodes if isinstance(n.get("id"), int)}
    for link in links:
        lid, src, src_slot, dst, dst_slot, ltype = link
        src_node = node_by_id.get(src)
        dst_node = node_by_id.get(dst)
        if src_node is None:
            problems.append(f"link {lid}: src node {src!r} missing")
            continue
        if dst_node is None:
            problems.append(f"link {lid}: dst node {dst!r} missing")
            continue
        src_outs = sanitized_io[id(src_node)]["outputs"]
        dst_ins = sanitized_io[id(dst_node)]["inputs"]

        # Slot index bounds + type checks.
        if not isinstance(src_slot, int) or src_slot < 0 or src_slot >= len(src_outs):
            problems.append(
                f"link {lid}: src slot {src_slot!r} out of bounds on node {src} "
                f"(has {len(src_outs)} outputs)"
            )
        elif not _slot_type_matches(src_outs[src_slot].get("type"), ltype):
            problems.append(
                f"link {lid}: src type mismatch — link declares {ltype!r}, "
                f"slot is {src_outs[src_slot].get('type')!r}"
            )
        if not isinstance(dst_slot, int) or dst_slot < 0 or dst_slot >= len(dst_ins):
            problems.append(
                f"link {lid}: dst slot {dst_slot!r} out of bounds on node {dst} "
                f"(has {len(dst_ins)} inputs)"
            )
        elif not _slot_type_matches(dst_ins[dst_slot].get("type"), ltype):
            problems.append(
                f"link {lid}: dst type mismatch — link declares {ltype!r}, "
                f"slot is {dst_ins[dst_slot].get('type')!r}"
            )

        # Endpoint cross-check: node IO must reference this link id symmetrically.
        if isinstance(src_slot, int) and 0 <= src_slot < len(src_outs):
            out_links = src_outs[src_slot].get("links") or []
            if isinstance(out_links, list) and lid not in out_links:
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
