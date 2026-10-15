"""Resting-hand wobble of the rig fit output (FRAME_HANDS_RIG_TRACE rig.txt).

Per side and 2 s window (stereo fits only unless --mono): wrist spread (RMS about the window mean, mm) and the
drift of the 0.5 s moving mean across the window (mm). Prints a spread histogram and the stillest windows, so
runs with different smoothing or prior settings can be compared on the same recording.
usage: python rig_still.py <rig.txt> [<rig.txt> ...] [--window 2] [--mono]
"""
import argparse, math
from collections import defaultdict

ap = argparse.ArgumentParser(); ap.add_argument('paths', nargs='+'); ap.add_argument('--window', type=float, default=2)
ap.add_argument('--mono', action='store_true', help='include single-camera fits')
a = ap.parse_args()

def rms(v): return math.sqrt(sum(x * x for x in v) / len(v))

for path in a.paths:
    data = defaultdict(list)   # side -> [(t_s, (x,y,z) mm)]
    with open(path, errors='replace') as f:
        lines = [line.split() for line in f]
    stamp = None; t0 = None
    for i, p in enumerate(lines):
        if not p: continue
        if p[0] == 'frame':
            stamp = int(p[1]); t0 = t0 if t0 is not None else stamp
        elif p[0] == 'rig' and len(p) >= 13 and p[3] == 'fitted' and stamp is not None:
            nxt = lines[i + 1] if i + 1 < len(lines) else []
            mono = len(nxt) >= 2 and nxt[0] == 'mono' and nxt[1] == p[1]
            if mono and not a.mono: continue
            data[int(p[2])].append(((stamp - t0) / 1e9, (float(p[10]), float(p[11]), float(p[12]))))
    print(f'\n== {path}')
    for side in (0, 1):
        d = data[side]
        if not d: continue
        w = a.window; start = d[0][0]; n = int((d[-1][0] - start) // w) + 1
        rows = []
        for i in range(n):
            win = [x for x in d if start + i * w <= x[0] < start + (i + 1) * w]
            if len(win) < 20: continue
            m = [sum(x[1][k] for x in win) / len(win) for k in range(3)]
            spread = rms([math.dist(x[1], m) for x in win])
            means = []
            for x in win[::5]:
                nb = [e for e in win if abs(e[0] - x[0]) <= 0.25]
                means.append([sum(e[1][k] for e in nb) / len(nb) for k in range(3)])
            drift = max(math.dist(p, q) for p in means for q in means)
            rows.append((start + i * w, len(win), spread, drift))
        rows.sort(key=lambda r: r[2])
        bins = [(0, 3), (3, 6), (6, 12), (12, 25), (25, 1e9)]
        hist = {b: sum(1 for r in rows if b[0] <= r[2] < b[1]) for b in bins}
        still = rows[:10]
        print(f'  side {side} ({"left" if side == 0 else "right"}): {len(rows)} windows; spread mm ' +
              ', '.join(f'{lo}-{"" if hi > 1e8 else hi}: {hist[(lo, hi)]}' for lo, hi in bins) +
              f';  stillest 10: spread median {sorted(r[2] for r in still)[len(still)//2]:.1f} mm, drift median {sorted(r[3] for r in still)[len(still)//2]:.1f} mm')
        print('    ' + '  '.join(f'{r[0]:.0f}s:{r[2]:.1f}/{r[3]:.1f}' for r in still))
