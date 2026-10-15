"""Resting-hand wobble split by direction, from a replay timeline (tracker triangulation, head frame) or a rig
trace (rig output). For the stillest 2 s windows per side: RMS deviation along the viewing ray (depth, from the
head origin) versus perpendicular to it (lateral), and the 2D-equivalent noise. Depth noise much larger than
lateral noise means the per-landmark 2D noise is small and the stereo geometry (short baseline, wide lens)
amplifies it into depth: an estimation-geometry limit, not a filtering mistake.
usage: python still_axes.py <timeline.txt|rig.txt> [--window 2] [--top 10]
"""
import argparse, math
from collections import defaultdict

ap = argparse.ArgumentParser(); ap.add_argument('path'); ap.add_argument('--window', type=float, default=2); ap.add_argument('--top', type=int, default=10)
a = ap.parse_args()
data = defaultdict(list)
with open(a.path, errors='replace') as f:
    lines = [l.split() for l in f]
if any(p and p[0] == 'rig' for p in lines[:50]):
    stamp = None; t0 = None
    for i, p in enumerate(lines):
        if not p: continue
        if p[0] == 'frame': stamp = int(p[1]); t0 = t0 if t0 is not None else stamp
        elif p[0] == 'rig' and len(p) >= 13 and p[3] == 'fitted' and stamp is not None:
            nxt = lines[i + 1] if i + 1 < len(lines) else []
            if len(nxt) >= 2 and nxt[0] == 'mono' and nxt[1] == p[1]: continue
            data[int(p[2])].append(((stamp - t0) / 1e9, (float(p[10]) / 1000, float(p[11]) / 1000, float(p[12]) / 1000)))
    unit = 'rig output'
else:
    for p in lines:
        if len(p) >= 7 and p[2] in ('L', 'R') and p[1].isdigit() and int(p[3]) >= 2:
            data[0 if p[2] == 'L' else 1].append((float(p[0]), (float(p[4]), float(p[5]), float(p[6]))))
    unit = 'tracker triangulation'
print(f'{unit}: {a.path}')

def rms(v): return math.sqrt(sum(x * x for x in v) / len(v)) if v else float('nan')
for side in (0, 1):
    d = data[side]
    if not d: continue
    w = a.window; start = d[0][0]; n = int((d[-1][0] - start) // w) + 1
    rows = []
    for i in range(n):
        win = [x for x in d if start + i * w <= x[0] < start + (i + 1) * w]
        if len(win) < 20: continue
        m = [sum(x[1][k] for x in win) / len(win) for k in range(3)]
        dist = math.hypot(*m); ray = [c / dist for c in m]
        along, lat = [], []
        for _, p in win:
            dv = [p[k] - m[k] for k in range(3)]
            s = sum(dv[k] * ray[k] for k in range(3))
            along.append(s); lat.append(math.sqrt(max(0, sum(c * c for c in dv) - s * s)))
        rows.append((start + i * w, dist, 1000 * rms(along), 1000 * rms(lat), 1000 * rms([math.dist(p, m) for _, p in win])))
    rows.sort(key=lambda r: r[4])
    top = rows[:a.top]
    print(f'  side {side} ({"left" if side == 0 else "right"}), stillest {len(top)} windows: '
          f'total {sorted(r[4] for r in top)[len(top)//2]:.1f} mm = depth {sorted(r[2] for r in top)[len(top)//2]:.1f} mm + lateral {sorted(r[3] for r in top)[len(top)//2]:.1f} mm (medians), '
          f'hand distance {sorted(r[1] for r in top)[len(top)//2]:.2f} m')
    for t, dist, al, la, tot in top: print(f'    {t:6.1f}s  d={dist:.2f} m  depth {al:5.1f}  lateral {la:5.1f}  total {tot:5.1f} mm')
