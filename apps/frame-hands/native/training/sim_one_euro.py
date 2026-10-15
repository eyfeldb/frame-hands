"""Simulate the presentation One Euro filter (processor/baseline_motion.gd: speed-adaptive cutoff on position)
on a rig trace's wrist positions, for several (min cutoff Hz, beta Hz per m/s) settings, and report resting
wobble (stillest 2 s windows: spread RMS and drift of the 0.5 s mean) next to motion fidelity (per-frame step
p99/max, path length, lag = mean distance from the unfiltered position while moving faster than 0.3 m/s).
usage: python sim_one_euro.py <rig.txt> [--settings 4,15 2,15 1.5,15 1.5,30]
"""
import argparse, math
from collections import defaultdict

ap = argparse.ArgumentParser(); ap.add_argument('path'); ap.add_argument('--settings', nargs='*', default=['4,15', '2,15', '1.5,15', '1.5,30', '1,30'])
a = ap.parse_args()
with open(a.path, errors='replace') as f: lines = [l.split() for l in f]
tracks = defaultdict(list)   # source -> [(t_s, side, (x,y,z) m)]
stamp = None; t0 = None
for i, p in enumerate(lines):
    if not p: continue
    if p[0] == 'frame': stamp = int(p[1]); t0 = t0 if t0 is not None else stamp
    elif p[0] == 'rig' and len(p) >= 13 and p[3] == 'fitted' and stamp is not None:
        nxt = lines[i + 1] if i + 1 < len(lines) else []
        if len(nxt) >= 2 and nxt[0] == 'mono' and nxt[1] == p[1]: continue
        tracks[p[1]].append(((stamp - t0) / 1e9, int(p[2]), (float(p[10]) / 1000, float(p[11]) / 1000, float(p[12]) / 1000)))

def alpha(cutoff, dt): return 1 / (1 + 1 / (2 * math.pi * cutoff * dt))
def sub(a, b): return [a[k] - b[k] for k in range(3)]
def norm(v): return math.sqrt(sum(x * x for x in v))
def rms(v): return math.sqrt(sum(x * x for x in v) / len(v)) if v else float('nan')

def run(minc, beta):
    out = defaultdict(list)   # side -> [(t, filtered, raw)]
    for src, obs in tracks.items():
        pos = None; vel = [0, 0, 0]; raw_prev = None; t_prev = None
        for t, side, p in obs:
            if pos is None or t_prev is None or t - t_prev > 0.15:
                pos = list(p); vel = [0, 0, 0]
            else:
                dt = t - t_prev
                v = [c / dt for c in sub(p, raw_prev)]
                da = alpha(6.0, dt)
                if sum(v[k] * vel[k] for k in range(3)) < 0 or norm(v) < 0.03: vel = v
                else: vel = [vel[k] + (v[k] - vel[k]) * da for k in range(3)]
                sp = min(norm(vel), 2.0)
                pa = alpha(minc + beta * sp, dt)
                pos = [pos[k] + (p[k] - pos[k]) * pa for k in range(3)]
            raw_prev = p; t_prev = t
            out[side].append((t, tuple(pos), p))
    return out

print(a.path)
print('setting      | still spread L/R mm | drift L/R mm | step p99 / max mm | path m | lag while moving mm')
for s in a.settings:
    minc, beta = (float(x) for x in s.split(','))
    out = run(minc, beta)
    still = {}; drift = {}
    for side in (0, 1):
        d = sorted(out[side])
        if not d: continue
        start = d[0][0]; n = int((d[-1][0] - start) // 2) + 1; rows = []
        for i in range(n):
            win = [x for x in d if start + i * 2 <= x[0] < start + (i + 1) * 2]
            if len(win) < 20: continue
            m = [sum(x[1][k] for x in win) / len(win) for k in range(3)]
            spread = 1000 * rms([norm(sub(x[1], m)) for x in win])
            means = []
            for x in win[::5]:
                nb = [e for e in win if abs(e[0] - x[0]) <= 0.25]
                means.append([sum(e[1][k] for e in nb) / len(nb) for k in range(3)])
            dr = 1000 * max(norm(sub(p, q)) for p in means for q in means)
            rows.append((spread, dr))
        rows.sort(); top = rows[:10]
        still[side] = sorted(r[0] for r in top)[len(top) // 2]; drift[side] = sorted(r[1] for r in top)[len(top) // 2]
    steps = []; path = 0; lags = []
    for side in (0, 1):
        d = sorted(out[side]); prev = None
        for t, f, r in d:
            if prev is not None and 0.001 < t - prev[0] <= 0.07:
                st = 1000 * norm(sub(f, prev[1])); steps.append(st); path += st / 1000
                speed = norm(sub(r, prev[2])) / (t - prev[0])
                if speed > 0.3: lags.append(1000 * norm(sub(f, r)))
            prev = (t, f, r)
    steps.sort()
    print(f'{s:12s} | {still.get(0, float("nan")):5.1f} / {still.get(1, float("nan")):5.1f}        | {drift.get(0, float("nan")):5.1f} / {drift.get(1, float("nan")):5.1f}  | {steps[int(len(steps) * .99)]:5.1f} / {steps[-1]:6.1f}   | {path:5.1f}  | {sum(lags) / max(1, len(lags)):5.1f} (n={len(lags)})')
