"""One row per replay directory (summary.txt, rig.txt, views.txt): hand presence, stereo pixel RMSE, per-frame wrist
step of the rig output (p50/p99 mm), resting-hand spread (stillest 10 windows, median L/R mm), 2D wrist step of the
tracker views (median px, both cameras), sets with three or more hands. For comparing models, rig profiles and
tracker flags on the same recording.
usage: python compare_replays.py <replay-dir> [<replay-dir> ...] [--window 2]
"""
import argparse, math, re
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument('dirs', nargs='+'); ap.add_argument('--window', type=float, default=2)
a = ap.parse_args()

def rms(v): return math.sqrt(sum(x * x for x in v) / len(v)) if v else float('nan')

def rig_stats(path):
    with open(path, errors='replace') as f: lines = [l.split() for l in f]
    stamp = None; t0 = None; last = {}; steps = []; rmse = []; by_side = defaultdict(list)
    for i, p in enumerate(lines):
        if not p: continue
        if p[0] == 'frame': stamp = int(p[1]); t0 = t0 if t0 is not None else stamp
        elif p[0] == 'rig' and len(p) >= 13 and p[3] == 'fitted' and stamp is not None:
            nxt = lines[i + 1] if i + 1 < len(lines) else []
            if len(nxt) >= 2 and nxt[0] == 'mono' and nxt[1] == p[1]: continue
            w = (float(p[10]), float(p[11]), float(p[12])); src = p[1]
            rmse.append(float(p[5]))
            if src in last and stamp - last[src][0] <= 70_000_000: steps.append(math.dist(w, last[src][1]))
            last[src] = (stamp, w)
            by_side[int(p[2])].append(((stamp - t0) / 1e9, w))
    steps.sort(); rmse.sort()
    still = {}
    for side, d in by_side.items():
        if not d: continue
        start = d[0][0]; n = int((d[-1][0] - start) // a.window) + 1; rows = []
        for i in range(n):
            win = [x for x in d if start + i * a.window <= x[0] < start + (i + 1) * a.window]
            if len(win) < 20: continue
            m = [sum(x[1][k] for x in win) / len(win) for k in range(3)]
            rows.append(rms([math.dist(x[1], m) for x in win]))
        rows.sort(); top = rows[:10]
        still[side] = top[len(top) // 2] if top else float('nan')
    q = lambda arr, f: arr[min(len(arr) - 1, int(len(arr) * f))] if arr else float('nan')
    return dict(rmse=q(rmse, .5), step50=q(steps, .5), step99=q(steps, .99), stillL=still.get(0, float('nan')), stillR=still.get(1, float('nan')))

def views_stats(path):
    # 2D wrist step per (hand id, camera) between consecutive sets (tracker audit format)
    prev = {}; steps = []; cur_t = None; hands = {}
    with open(path, errors='replace') as f:
        for line in f:
            p = line.split()
            if not p: continue
            if p[0] == 'set': cur_t = int(p[1])
            elif p[0] == 'view' and len(p) >= 47 and cur_t is not None:
                key = (p[2], p[1])
                try: x, y = float(p[5]), float(p[6])
                except ValueError: continue
                if key in prev and 0 < cur_t - prev[key][0] <= 60_000_000: steps.append(math.hypot(x - prev[key][1], y - prev[key][2]))
                prev[key] = (cur_t, x, y)
    steps.sort()
    return steps[len(steps) // 2] if steps else float('nan')

def summary_stats(path):
    text = Path(path).read_text(errors='replace')
    m = re.search(r'a hand on the left (\d+)%, right (\d+)%, both (\d+)%', text)
    both = int(m.group(3)) if m else -1
    return dict(both=both)

def timeline_three(path):
    frames = defaultdict(int); n = 0
    with open(path, errors='replace') as f:
        for line in f:
            p = line.split()
            if len(p) >= 7 and p[2] in ('L', 'R') and p[1].isdigit(): frames[p[0]] += 1
    three = sum(1 for v in frames.values() if v >= 3)
    return three / 30.0

print(f'{"replay":34s} both%  rmse px  wrist step p50/p99 mm  still L/R mm  2D step px  3+hands s')
for d in a.dirs:
    d = Path(d)
    r = rig_stats(d / 'rig.txt'); s = summary_stats(d / 'summary.txt')
    v = views_stats(d / 'views.txt') if (d / 'views.txt').exists() else float('nan')
    t3 = timeline_three(d / 'timeline.txt') if (d / 'timeline.txt').exists() else float('nan')
    print(f'{d.name:34s} {s["both"]:4d}   {r["rmse"]:5.2f}    {r["step50"]:5.1f} / {r["step99"]:5.1f}          {r["stillL"]:4.1f} / {r["stillR"]:4.1f}     {v:5.2f}      {t3:5.1f}')
