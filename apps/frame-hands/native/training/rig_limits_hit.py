"""Which rig joints are pinned at their limits, per time window and side. Reads a rig trace and the side profiles
(limits). Prints, per window, the pixel RMSE median and the joints whose fitted angle sits within 1 degree of a
limit in at least a third of the window's fits, with the fraction. Joint names: t0..t3 thumb, i0..i3 index,
m0..m3 middle, r0..r3 ring, p0..p3 pinky (k0/k1 = base two axes, k2 = PIP, k3 = DIP; thumb: CMC a/b, MCP, IP).
usage: python rig_limits_hit.py <rig.txt> <profiles-dir> [--window 8]
"""
import argparse, math
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('rig'); ap.add_argument('profiles'); ap.add_argument('--window', type=float, default=8)
a = ap.parse_args()
NAMES = [f'{f}{k}' for f in 'timrp' for k in range(4)]
limits = {}
for side in (0, 1):
    rows = Path(a.profiles, f'side{side}.fhrig').read_text().splitlines()
    j = np.array([list(map(float, r.split())) for r in rows[1:21]]); limits[side] = (j[:, 6], j[:, 7])
with open(a.rig, errors='replace') as f: lines = [l.split() for l in f]
rec = []; stamp = None; t0 = None
for p in lines:
    if not p: continue
    if p[0] == 'frame': stamp = int(p[1]); t0 = t0 if t0 is not None else stamp
    elif p[0] == 'rig' and p[3] == 'fitted' and len(p) >= 33 and stamp is not None:
        rec.append(((stamp - t0) / 1e9, int(p[2]), float(p[5]), np.array([float(x) for x in p[13:33]])))
w = a.window; end = rec[-1][0] if rec else 0
tol = math.radians(1.0)
print('window | side | fits | rmse | joints at a limit (fraction of fits, L=lower U=upper)')
for i in range(int(end // w) + 1):
    for side in (0, 1):
        win = [r for r in rec if i * w <= r[0] < (i + 1) * w and r[1] == side]
        if len(win) < 10: continue
        A = np.array([r[3] for r in win]); lo, hi = limits[side]
        atlo = (A <= lo + tol).mean(0); athi = (A >= hi - tol).mean(0)
        hits = [f'{NAMES[k]}{"L" if atlo[k] >= athi[k] else "U"}:{max(atlo[k], athi[k]):.2f}' for k in range(20) if max(atlo[k], athi[k]) >= 0.33]
        rmse = float(np.median([r[2] for r in win]))
        print(f'{i * w:5.0f}s |  {side}   | {len(win):4d} | {rmse:5.2f} | ' + ' '.join(hits))
