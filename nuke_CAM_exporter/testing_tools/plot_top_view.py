#!/usr/bin/env python3
"""
Quick top-view visualizer for camera paths.

Example:
    py -3 nuke_CAM_exporter/testing_tools/plot_top_view.py ^
        --config nuke_CAM_exporter/configs/camera_config_v01.json ^
        --output nuke_CAM_exporter/testing_tools/top_view.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import patches

NUKE_SWAP_AXES = True
NUKE_INVERT_HORIZONTAL = True
NUKE_INVERT_VERTICAL = True
NUKE_HORIZONTAL_LABEL = "+Z (frame left)  <-  ->  -Z (frame right)"
NUKE_VERTICAL_LABEL = "+X (frame down)  v  ^  -X (frame up)"


def euler_zxy_to_matrix(rx_rad: float, ry_rad: float, rz_rad: float) -> List[List[float]]:
    sx, cx = math.sin(rx_rad), math.cos(rx_rad)
    sy, cy = math.sin(ry_rad), math.cos(ry_rad)
    sz, cz = math.sin(rz_rad), math.cos(rz_rad)

    Rx = (
        (1.0, 0.0, 0.0),
        (0.0, cx, -sx),
        (0.0, sx, cx),
    )
    Ry = (
        (cy, 0.0, sy),
        (0.0, 1.0, 0.0),
        (-sy, 0.0, cy),
    )
    Rz = (
        (cz, -sz, 0.0),
        (sz, cz, 0.0),
        (0.0, 0.0, 1.0),
    )

    def mat_mul(a, b):
        return [
            [
                a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c]
                for c in range(3)
            ]
            for r in range(3)
        ]

    return mat_mul(Ry, mat_mul(Rx, Rz))


def load_track_samples(
    translation_path: Path,
    rotation_path: Path | None,
) -> List[dict]:
    samples: List[dict] = []

    def append_sample(tx: float, ty: float, tz: float, rx_deg: float | None, ry_deg: float | None, rz_deg: float | None):
        forward = None
        if rx_deg is not None and ry_deg is not None and rz_deg is not None:
            R = euler_zxy_to_matrix(
                math.radians(rx_deg),
                math.radians(ry_deg),
                math.radians(rz_deg),
            )
            # Camera in Nuke looks toward -Z (viewing down the negative Z axis)
            forward = (-R[0][2], -R[1][2], -R[2][2])
        samples.append({"translation": (tx, ty, tz), "forward": forward})

    lines = [l.strip() for l in translation_path.read_text().splitlines() if l.strip()]
    if rotation_path:
        rot_lines = [l.strip() for l in rotation_path.read_text().splitlines() if l.strip()]
        if len(lines) != len(rot_lines):
            raise ValueError("Translation and rotation ASCI files must have the same number of lines.")
        for t_line, r_line in zip(lines, rot_lines):
            tx, ty, tz = map(float, t_line.split()[:3])
            rx_deg, ry_deg, rz_deg = map(float, r_line.split()[:3])
            append_sample(tx, ty, tz, rx_deg, ry_deg, rz_deg)
    else:
        for line in lines:
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(
                    "Combined ASCI expected 6 columns per line (Tx Rx Ty Ry Tz Rz). "
                    f"Got {len(parts)} columns: {line}"
                )
            tx = float(parts[0])
            rx_deg = float(parts[1])
            ty = float(parts[2])
            ry_deg = float(parts[3])
            tz = float(parts[4])
            rz_deg = float(parts[5])
            append_sample(tx, ty, tz, rx_deg, ry_deg, rz_deg)

    return samples


def parse_pose_file(path: Path, limit: int | None) -> List[dict]:
    rows: List[dict] = []
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    for line in lines[1:]:  # skip header
        values = list(map(float, line.split()))
        if len(values) != 19:
            raise ValueError(f"Expected 19 columns per pose row, got {len(values)}")
        rotation = [
            [values[7], values[8], values[9]],
            [values[11], values[12], values[13]],
            [values[15], values[16], values[17]],
        ]
        translation = (values[10], values[14], values[18])
        rows.append({"R": rotation, "t": translation})
        if limit is not None and len(rows) >= limit:
            break
    return rows


def scale_rows(rows: Iterable[Tuple[float, float, float]], scale: float) -> List[Tuple[float, float, float]]:
    if scale == 1.0:
        return list(rows)
    return [(x * scale, y * scale, z * scale) for x, y, z in rows]


def split_axes(rows: Sequence[Tuple[float, float, float]]) -> Tuple[List[float], List[float], List[float]]:
    xs, ys, zs = [], [], []
    for x, y, z in rows:
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return xs, ys, zs


def load_config(path: Path | None) -> dict:
    if not path:
        return {}
    data = json.loads(path.read_text())
    data["_config_dir"] = path.parent
    return data


def resolve_path(config: dict, key: str) -> Path | None:
    value = config.get(key)
    if value is None:
        return None
    candidate = Path(value)
    if not candidate.is_absolute() and "_config_dir" in config:
        candidate = (config["_config_dir"] / candidate).resolve()
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Top-view plot of camera path (Nuke-style top view).")
    parser.add_argument("--config", type=Path, help="Optional JSON config (same format as converter).")
    parser.add_argument("--translation", type=Path, help="Translation ASCI file (Tx Ty Tz).")
    parser.add_argument("--rotation", type=Path, help="Rotation ASCI file (Rx Ry Rz) when exports are split.")
    parser.add_argument(
        "--pose",
        type=Path,
        help="Converted pose TXT (to overlay world-to-camera translations). Optional.",
    )
    parser.add_argument("--unit-scale", type=float, default=None, help="Scale to apply to translation vectors.")
    parser.add_argument("--limit", type=int, help="Maximum number of frames to plot.")
    parser.add_argument("--output", type=Path, help="Optional PNG to save instead of showing interactively.")
    parser.add_argument("--show", action="store_true", help="Force interactive window even when saving.")
    square = parser.add_argument_group("workspace overlays")
    square.add_argument(
        "--square-size",
        type=float,
        help="Explicit side length for centered square (meters). Overrides auto sizing.",
    )
    square.add_argument(
        "--square-padding",
        type=float,
        default=0.15,
        help="Extra fractional padding when auto-sizing the square (default 0.15).",
    )
    square.add_argument(
        "--square-color",
        default="#00b894",
        help="Line color for the square/highlight overlays.",
    )
    square.add_argument(
        "--highlight-quadrant",
        choices=["++", "+-", "-+", "--"],
        default="+-",
        help="Quadrant to shade (horizontal sign first, default '+-').",
    )
    square.add_argument(
        "--highlight-alpha",
        type=float,
        default=0.15,
        help="Alpha used for quadrant shading.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)

    def resolved_path(cli_value: Path | None, config_key: str) -> Path | None:
        if cli_value:
            return cli_value.resolve()
        return resolve_path(config, config_key)

    translation_path = resolved_path(args.translation, "input_translation")
    rotation_path = resolved_path(args.rotation, "input_rotation")
    if translation_path is None:
        translation_path = resolved_path(None, "input")
        rotation_path = None
    if translation_path is None:
        parser.error("Translation source missing (pass --translation or provide input/_translation in config).")

    pose_path = resolved_path(args.pose, "output")
    unit_scale = args.unit_scale if args.unit_scale is not None else config.get("unit_scale", 1.0)

    frame_cfg = config.get("frame_range", {})
    start_idx = frame_cfg.get("start", 0)
    limit = args.limit
    if limit is None:
        limit = frame_cfg.get("max_frames")
        if limit is None and frame_cfg.get("end") is not None:
            limit = max(0, frame_cfg["end"] - start_idx)

    track_samples = load_track_samples(translation_path, rotation_path)
    track_samples = track_samples[start_idx:]
    if limit is not None:
        track_samples = track_samples[:limit]

    for sample in track_samples:
        tx, ty, tz = sample["translation"]
        sample["translation"] = (tx * unit_scale, ty * unit_scale, tz * unit_scale)

    translations_scaled = [sample["translation"] for sample in track_samples]
    tx, _, tz = split_axes(translations_scaled)

    def orient_axes(xs: Sequence[float], zs: Sequence[float]) -> Tuple[List[float], List[float]]:
        horiz, vert = (list(zs), list(xs)) if NUKE_SWAP_AXES else (list(xs), list(zs))
        if NUKE_INVERT_HORIZONTAL:
            horiz = [-val for val in horiz]
        if NUKE_INVERT_VERTICAL:
            vert = [-val for val in vert]
        return horiz, vert

    def orient_point(x: float, z: float) -> Tuple[float, float]:
        hx, hy = orient_axes([x], [z])
        return hx[0], hy[0]

    hx, hy = orient_axes(tx, tz)
    track_screen_points = list(zip(hx, hy))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.6)
    ax.axvline(0.0, color="#666666", linewidth=0.8, alpha=0.6)
    ax.scatter([0], [0], c="#222222", s=20, marker="x", label="Origin", zorder=5)

    ax.scatter(hx, hy, s=12, c="tab:green", label="ASCI center (scaled)")
    ax.plot(hx, hy, color="tab:green", alpha=0.3)

    pose_rows: List[dict] | None = None
    pose_screen_points: List[Tuple[float, float]] | None = None
    if pose_path:
        pose_rows = parse_pose_file(pose_path, None)
        pose_rows = pose_rows[start_idx:]
        if limit is not None:
            pose_rows = pose_rows[:limit]
        pose_translations = [row["t"] for row in pose_rows]
        px, _, pz = split_axes(pose_translations)
        phx, phy = orient_axes(px, pz)
        pose_screen_points = list(zip(phx, phy))
        ax.scatter(phx, phy, s=12, c="tab:purple", label="Pose t (world-to-camera)")
        ax.plot(phx, phy, color="tab:purple", alpha=0.3)

    ax.set_xlabel(NUKE_HORIZONTAL_LABEL)
    ax.set_ylabel(NUKE_VERTICAL_LABEL)

    if hx and hy:
        if args.square_size is not None:
            half = args.square_size / 2.0
        else:
            max_extent = max(max(abs(val) for val in hx), max(abs(val) for val in hy), 1e-6)
            half = max_extent * (1.0 + args.square_padding)
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        square = patches.Rectangle(
            (-half, -half),
            half * 2,
            half * 2,
            linewidth=1.2,
            edgecolor=args.square_color,
            facecolor="none",
            linestyle="-",
            zorder=2,
        )
        ax.add_patch(square)

        if args.highlight_quadrant:
            x0 = 0.0 if args.highlight_quadrant[0] == "+" else -half
            y0 = 0.0 if args.highlight_quadrant[1] == "+" else -half
            width = half
            height = half
            highlight = patches.Rectangle(
                (x0, y0),
                width,
                height,
                facecolor=args.square_color,
                alpha=args.highlight_alpha,
                edgecolor="none",
                zorder=1,
            )
            ax.add_patch(highlight)
        pad = half * 0.04
        label_kwargs = dict(color="#414141", fontsize=9, fontweight="semibold")
        ax.text(-half + pad, pad, "+Z", ha="left", va="bottom", **label_kwargs)
        ax.text(half - pad, pad, "-Z", ha="right", va="bottom", **label_kwargs)
        ax.text(-pad, -half + pad, "+X", ha="right", va="top", **label_kwargs)
        ax.text(-pad, half - pad, "-X", ha="right", va="bottom", **label_kwargs)
        nuke_forward_samples = [s for s in track_samples if s["forward"] is not None]
        if nuke_forward_samples:
            arrow_scale = max(half * 0.2, 0.05)
            head_width = half * 0.05
            head_length = half * 0.08
            for idx, label in ((0, "Start"), (-1, "End")):
                base_x, base_y = track_screen_points[idx]
                forward = track_samples[idx]["forward"]
                dir_x, dir_y = orient_point(forward[0], forward[2])
                ax.arrow(
                    base_x,
                    base_y,
                    dir_x * arrow_scale,
                    dir_y * arrow_scale,
                    head_width=head_width,
                    head_length=head_length,
                    length_includes_head=True,
                    color="tab:blue",
                    linewidth=1.5,
                    alpha=0.9,
                    zorder=4,
                )
                ax.text(
                    base_x,
                    base_y,
                    label,
                    color="tab:blue",
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=5,
                )
            ax.scatter([], [], marker="^", color="tab:blue", label="Nuke cam forward")
    ax.set_title("Camera path (top view)")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {args.output}")
        if not args.show:
            plt.close(fig)
            return

    plt.show()


if __name__ == "__main__":
    main()

