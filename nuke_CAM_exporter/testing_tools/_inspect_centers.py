from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pose_path = ROOT / "outputs/sh0032_unit-1-split_converted.txt"
pose_lines = [l.strip() for l in pose_path.read_text().splitlines() if l.strip()][1:]
tr_path = ROOT / "inputs/sh0032_unit-1-split_tr.asci"
tr_lines = [l.strip() for l in tr_path.read_text().splitlines()][:len(pose_lines)]
unit_scale = 0.2

def parse_pose_line(line):
    vals = list(map(float, line.split()))
    return [vals[7:11], vals[11:15], vals[15:19]]

print(f"frames {len(pose_lines)}")
for idx in (0, 10, 40, 79):
    tx, ty, tz = map(float, tr_lines[idx].split())
    tx *= unit_scale; ty *= unit_scale; tz *= unit_scale
    rows = parse_pose_line(pose_lines[idx])
    R = [row[:3] for row in rows]
    t = [row[3] for row in rows]
    Rt = [[R[j][i] for j in range(3)] for i in range(3)]
    C = [-sum(Rt[i][k] * t[k] for k in range(3)) for i in range(3)]
    print(idx, "orig", (round(tx,6), round(ty,6), round(tz,6)), "from_pose", tuple(round(c,6) for c in C))