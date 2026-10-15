"""Rig fit quality over time: per window, fitted/insufficient updates, stereo vs mono fits, pixel RMSE median,
and the median absolute thumb joint angles (rig joints 0-3) so a pose like a vertical thumbs-up can be checked
for a collapsed or saturated thumb. Reads a rig trace (FRAME_HANDS_RIG_TRACE).
usage: python rig_windows.py <rig.txt> [--window 8]
"""
import argparse, math
from collections import defaultdict

ap = argparse.ArgumentParser(); ap.add_argument('path'); ap.add_argument('--window', type=float, default=8)
a = ap.parse_args()
with open(a.path, errors='replace') as f: lines = [l.split() for l in f]
rows = []   # (t, side, status, mono, rmse, thumb angles(4))
stamp = None; t0 = None
for i, p in enumerate(lines):
    if not p: continue
    if p[0] == 'frame': stamp = int(p[1]); t0 = t0 if t0 is not None else stamp
    elif p[0] == 'rig' and stamp is not None:
        nxt = lines[i + 1] if i + 1 < len(lines) else []
        mono = len(nxt) >= 2 and nxt[0] == 'mono' and nxt[1] == p[1]
        if p[3] == 'fitted' and len(p) >= 33:
            rows.append(((stamp - t0) / 1e9, int(p[2]), 'fitted', mono, float(p[5]), [math.degrees(float(x)) for x in p[13:17]]))
        else:
            rows.append(((stamp - t0) / 1e9, int(p[2]), p[3], mono, float('nan'), None))
w = a.window; end = rows[-1][0] if rows else 0
med = lambda v: sorted(v)[len(v) // 2] if v else float('nan')
print('window | side | fitted insuff | mono%  | rmse px | thumb joint angles deg (median |a|)')
for i in range(int(end // w) + 1):
    for side in (0, 1):
        win = [r for r in rows if i * w <= r[0] < (i + 1) * w and r[1] == side]
        if not win: continue
        fitted = [r for r in win if r[2] == 'fitted']; insuff = len(win) - len(fitted)
        mono = sum(1 for r in fitted if r[3])
        rmse = med([r[4] for r in fitted])
        th = [med([abs(r[5][k]) for r in fitted]) for k in range(4)] if fitted else [float('nan')] * 4
        print(f'{i * w:5.0f}s |  {side}   | {len(fitted):5d}  {insuff:4d}  | {100 * mono / max(1, len(fitted)):4.0f}% | {rmse:6.2f}  | ' + ' '.join(f'{x:5.1f}' for x in th))
