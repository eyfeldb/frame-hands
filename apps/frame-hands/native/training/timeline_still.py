"""Where were the hands still, and did the tracker's output stay still there?

From a replay timeline: per hand side (by wrist x) and 2 s window, the stereo-tracked hand's wrist spread
(RMS about the window mean, mm) and the drift of the 0.5 s moving mean across the window (mm). Prints the windows
sorted by spread so resting-hand windows (small spread) and their drift are visible, plus a histogram.
usage: python timeline_still.py <timeline.txt> [--window 2]
"""
import argparse, math
from collections import defaultdict

ap = argparse.ArgumentParser(); ap.add_argument('path'); ap.add_argument('--window', type=float, default=2)
a = ap.parse_args()
pts = defaultdict(list)   # side -> [(t, (x,y,z), nviews)]
with open(a.path, errors='replace') as f:
    for line in f:
        p = line.split()
        if len(p) >= 7 and p[2] in ('L', 'R') and p[1].isdigit():
            try: pts[p[2]].append((float(p[0]), (float(p[4]), float(p[5]), float(p[6])), int(p[3])))
            except ValueError: pass

def rms(v): return math.sqrt(sum(x * x for x in v) / len(v))
for side in ('L', 'R'):
    rows = []
    data = pts[side]
    if not data: continue
    t0 = data[0][0]; tend = data[-1][0]
    w = a.window; n = int((tend - t0) // w) + 1
    for i in range(n):
        win = [d for d in data if t0 + i * w <= d[0] < t0 + (i + 1) * w and d[2] >= 2]
        if len(win) < 20: continue
        m = [sum(d[1][k] for d in win) / len(win) for k in range(3)]
        spread = 1000 * rms([math.dist(d[1], m) for d in win])
        means = []
        for d in win[::5]:
            nb = [e for e in win if abs(e[0] - d[0]) <= 0.25]
            means.append([sum(e[1][k] for e in nb) / len(nb) for k in range(3)])
        drift = 1000 * max(math.dist(p, q) for p in means for q in means)
        rows.append((t0 + i * w, len(win), spread, drift))
    rows.sort(key=lambda r: r[2])
    print(f'\n{side} hand (by position), stereo frames only: {len(rows)} windows of {w:.0f} s')
    bins = [(0, 3), (3, 6), (6, 12), (12, 25), (25, 1e9)]
    hist = {b: sum(1 for r in rows if b[0] <= r[2] < b[1]) for b in bins}
    print('  spread histogram (mm): ' + ', '.join(f'{lo}-{"" if hi > 1e8 else hi}: {hist[(lo, hi)]}' for lo, hi in bins))
    print('  stillest 12 windows:  start  frames  spread mm  drift of 0.5 s mean mm')
    for s, c, sp, dr in rows[:12]: print(f'    {s:6.1f}  {c:4d}   {sp:6.1f}   {dr:6.1f}')
