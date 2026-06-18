# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate the Easy Image Batch "all versions" demo workflow.

Emits ``easy-image-batch-demo-setup.json`` — a ComfyUI (litegraph v0.4)
workflow that instantiates every row of the behavior matrix
(`easy-image-batch-behavior.md`) as real nodes:

    Easy Pattern (source / inserts / single) ─▶ Easy Image Batch ─▶ Preview Image

Each scenario is its own labelled group so the maintainer can load the file
in ComfyUI and run/inspect all modes side by side. Generating from a script
(rather than hand-authoring JSON) keeps node-ids, slot indices, and links
internally consistent — the script self-validates before writing.

Run:  python docs/designs/build_easy_image_batch_demo.py
"""
from __future__ import annotations

import json
import os

# ---- id allocators ----------------------------------------------------------
_node_id = 0
_link_id = 0
nodes: list[dict] = []
links: list[list] = []
groups: list[dict] = []


def _nid() -> int:
    global _node_id
    _node_id += 1
    return _node_id


def _lid() -> int:
    global _link_id
    _link_id += 1
    return _link_id


def koolook_props(ntype: str) -> dict:
    return {
        "cnr_id": "koolook",
        "ver": "0.2.0",
        "Node name for S&R": ntype,
        "ue_properties": {
            "widget_ue_connectable": {},
            "input_ue_unconnectable": {},
            "version": "7.8",
        },
    }


def easy_pattern(pos, batch_size, hex_bg, start=1, title=None, color="#232", bg="#353"):
    """A numbered test-pattern generator. Returns (node_id, out_slot=0)."""
    nid = _nid()
    nodes.append({
        "id": nid, "type": "Easy_Pattern", "pos": list(pos),
        "size": [266, 398], "flags": {}, "order": nid, "mode": 0,
        "inputs": [],
        "outputs": [{"name": "images", "type": "IMAGE", "links": [], "slot_index": 0}],
        "title": title or f"pattern x{batch_size}",
        "properties": koolook_props("Easy_Pattern"),
        # batch_size, width, height, bg_mode, show_text, text_mode, start, step,
        # font_size, position, zero_pad, bg_color, text_color, prefix, suffix
        "widgets_values": [
            batch_size, 256, 256, "Custom", True, "White", start, 1, 110,
            "center", 0, hex_bg, "#FFFFFF", "", "",
        ],
        "color": color, "bgcolor": bg,
    })
    return nid, 0


def easy_batch(pos, *, title, total=24, cut=1, width=512, height=512,
               ph="Gray", invert=False, frames="",
               im1f=3, im2f=9, im3f=13, im4f=17):
    """An easy_ImageBatch node. Inputs are the 6 IMAGE slots in definition order:
    0 keyframes_insert, 1 source_batch, 2 image1, 3 image2, 4 image3, 5 image4."""
    nid = _nid()
    in_names = ["keyframes_insert", "source_batch", "image1", "image2", "image3", "image4"]
    nodes.append({
        "id": nid, "type": "easy_ImageBatch", "pos": list(pos),
        "size": [310, 320], "flags": {}, "order": nid, "mode": 0,
        "inputs": [{"name": n, "type": "IMAGE", "link": None} for n in in_names],
        "outputs": [
            {"name": "image_batch", "type": "IMAGE", "links": [], "slot_index": 0},
            {"name": "alpha_batch", "type": "MASK", "links": [], "slot_index": 1},
            {"name": "selected_image_batch", "type": "IMAGE", "links": [], "slot_index": 2},
            {"name": "selected_frames", "type": "STRING", "links": [], "slot_index": 3},
        ],
        "title": title,
        "properties": koolook_props("easy_ImageBatch"),
        # Widget order = INPUT_TYPES required-then-optional, IMAGE slots skipped:
        # total, cut, placeholder, invert, source_frames, image1_frame,
        # image2_frame, image3_frame, image4_frame, width, height.
        "widgets_values": [total, cut, ph, invert, frames, im1f,
                           im2f, im3f, im4f, width, height],
    })
    return nid


def preview(pos, src_node, src_slot=0, title="Preview image_batch"):
    nid = _nid()
    lid = _lid()
    nodes.append({
        "id": nid, "type": "PreviewImage", "pos": list(pos),
        "size": [300, 300], "flags": {}, "order": nid, "mode": 0,
        "inputs": [{"name": "images", "type": "IMAGE", "link": lid}],
        "outputs": [],
        "title": title,
        "properties": {"cnr_id": "comfy-core", "ver": "0.3.0", "Node name for S&R": "PreviewImage"},
    })
    _connect_existing(lid, src_node, src_slot, nid, 0, "IMAGE")
    return nid


def connect(src_node, src_slot, dst_node, dst_slot, typ="IMAGE"):
    lid = _lid()
    _connect_existing(lid, src_node, src_slot, dst_node, dst_slot, typ)


def _connect_existing(lid, src_node, src_slot, dst_node, dst_slot, typ):
    links.append([lid, src_node, src_slot, dst_node, dst_slot, typ])
    for n in nodes:
        if n["id"] == src_node:
            n["outputs"][src_slot]["links"].append(lid)
        if n["id"] == dst_node:
            n["inputs"][dst_slot]["link"] = lid


def group(title, x, y, w, h, color="#3f789e"):
    groups.append({
        "id": len(groups) + 1, "title": title,
        "bounding": [x, y, w, h], "color": color, "font_size": 22, "flags": {},
    })


# ---- shared generators (left column) ----------------------------------------
GX = 40
src_node, src_out = easy_pattern((GX, 40), 24, "#1f4f74", title="source_batch (24f, blue)",
                                 color="#322", bg="#533")
ins_node, ins_out = easy_pattern((GX, 480), 3, "#1d6a3a", title="keyframes_insert (3f, green)",
                                 color="#232", bg="#353")
one_node, one_out = easy_pattern((GX, 920), 1, "#8a5a16", title="single frame (1f, orange)",
                                 color="#432", bg="#653")

# ---- scenarios (matrix rows) ------------------------------------------------
# Each: easy_ImageBatch + a PreviewImage on image_batch, in its own group.
BX, PX = 360, 700           # easy_batch x, preview x
ROW_H = 360
SCN_W = PX + 320 - BX + 40

def scenario(i, title, *, batch_kwargs, wire):
    y = 40 + i * ROW_H
    b = easy_batch((BX, y), title=title, **batch_kwargs)
    wire(b)                                   # connect inputs to generators
    preview((PX, y), b, 0)
    group(title, BX - 20, y - 56, SCN_W, ROW_H - 30)
    return b

scenario(0, "1 · empty → clean (width/height)",
         batch_kwargs=dict(total=12, frames=""),
         wire=lambda b: None)

scenario(1, "2 · select: source + list",
         batch_kwargs=dict(frames="1, 5, 9, 13, 17, 21"),
         wire=lambda b: connect(src_node, src_out, b, 1))

scenario(2, "3 · select: 4 slots (image1–4)",
         batch_kwargs=dict(frames="", im1f=3, im2f=8, im3f=14, im4f=20),
         wire=lambda b: [connect(one_node, one_out, b, s) for s in (2, 3, 4, 5)])

scenario(3, "4 · select: list + slot override (image1@3)",
         batch_kwargs=dict(frames="1, 5, 9, 13, 17", im1f=3),
         wire=lambda b: (connect(src_node, src_out, b, 1),
                         connect(one_node, one_out, b, 2)))

scenario(4, "6 · offset over placeholder",
         batch_kwargs=dict(frames="3, 9, 15"),
         wire=lambda b: connect(ins_node, ins_out, b, 0))

scenario(5, "7 · offset empty list → clean",
         batch_kwargs=dict(frames=""),
         wire=lambda b: connect(ins_node, ins_out, b, 0))

scenario(6, "8 · insert-over-source",
         batch_kwargs=dict(frames="5, 12, 19"),
         wire=lambda b: (connect(ins_node, ins_out, b, 0),
                         connect(src_node, src_out, b, 1)))

scenario(7, "9 · insert + empty list → source passthrough",
         batch_kwargs=dict(frames=""),
         wire=lambda b: (connect(ins_node, ins_out, b, 0),
                         connect(src_node, src_out, b, 1)))

scenario(8, "10 · insert-over-source + slot override (image1@3)",
         batch_kwargs=dict(frames="5, 12, 19", im1f=3),
         wire=lambda b: (connect(ins_node, ins_out, b, 0),
                         connect(src_node, src_out, b, 1),
                         connect(one_node, one_out, b, 2)))

scenario(9, "cut window · cut_start=10, total=16",
         batch_kwargs=dict(total=16, cut=10, frames="10, 14, 18, 22"),
         wire=lambda b: connect(src_node, src_out, b, 1))

# ---- assemble + validate ----------------------------------------------------
doc = {
    "id": "easy-image-batch-demo",
    "revision": 0,
    "last_node_id": _node_id,
    "last_link_id": _link_id,
    "nodes": nodes,
    "links": links,
    "groups": groups,
    "config": {},
    "extra": {},
    "version": 0.4,
}

by_id = {n["id"]: n for n in nodes}
for lid, sn, ss, dn, ds, typ in links:
    assert sn in by_id and dn in by_id, f"link {lid}: missing node"
    assert ss < len(by_id[sn]["outputs"]), f"link {lid}: bad out slot {ss}"
    assert ds < len(by_id[dn]["inputs"]), f"link {lid}: bad in slot {ds}"
assert len({n["id"] for n in nodes}) == len(nodes), "duplicate node id"

out_path = os.path.join(os.path.dirname(__file__), "easy-image-batch-demo-setup.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(doc, f, indent=2)
print(f"wrote {out_path}: {len(nodes)} nodes, {len(links)} links, {len(groups)} groups")
