"""Simulate a hold-radius stabiliser on a rig trace's wrist positions and compare with the plain output.

Stabiliser: an anchor follows the observations. While the observation stays within `radius` of the anchor and
the recent speed is below `speed`, the output is the anchor (held still). When the observation leaves the
radius, the anchor catches up exponentially with time constant `catch` seconds, so the output accelerates
toward the hand instead of jumping, and once it is moving (speed above `speed`) it tracks the observation
directly. Reports resting spread/drift (stillest 2 s windows), per-frame step p99/max, path length, and the
lag (mean distance from the raw observation) while moving faster than 0.3 m/s and during the first 150 ms of
each movement onset.
usage: python sim_deadband.py <rig.txt> [--settings "r,speed,catch" ...]   (mm, m/s, s; e.g. 8,0.08,0.06)
"""
import argparse, math
from collections import defaultdict

ap = argparse.ArgumentParser(); ap.add_argument('path')
ap.add_argument('--settings', nargs='*', default=['0,0,0', '5,0.06,0.05', '8,0.08,0.06', '12,0.10,0.08', '8,0.08,0.12'])
a = ap.parse_args()
with open(a.path, errors='replace') as f: lines = [l.split() for l in f]
tracks = defaultdict(list)
stamp = None; t0 = None
for i, p in enumerate(lines):
    if not p: continue
    if p[0] == 'frame': stamp = int(p[1]); t0 = t0 if t0 is not None else stamp
    elif p[0] == 'rig' and len(p) >= 13 and p[3] == 'fitted' and stamp is not None:
        nxt = lines[i + 1] if i + 1 < len(lines) else []
        if len(nxt) >= 2 and nxt[0] == 'mono' and nxt[1] == p[1]: continue
        tracks[p[1]].append(((stamp - t0) / 1e9, int(p[2]), (float(p[10]) / 1000, float(p[11]) / 1000, float(p[12]) / 1000)))

def sub(a, b): return [a[k] - b[k] for k in range(3)]
def norm(v): return math.sqrt(sum(x * x for x in v))
def rms(v): return math.sqrt(sum(x * x for x in v) / len(v)) if v else float('nan')

def run(radius_mm, speed_lim, catch):
    radius = radius_mm / 1000
    out = defaultdict(list)
    for src, obs in tracks.items():
        anchor = None; t_prev = None; p_prev = None; speed = 0.0; moving = False
        for t, side, p in obs:
            if anchor is None or t_prev is None or t - t_prev > 0.15:
                anchor = list(p); speed = 0.0; moving = False
            else:
                dt = t - t_prev
                v = norm(sub(p, p_prev)) / dt
                speed += (v - speed) * min(1.0, dt / 0.08)          # ~80 ms speed estimate
                d = norm(sub(p, anchor))
                if radius <= 0:
                    anchor = list(p)
                elif moving:
                    anchor = list(p)
                    if speed < speed_lim and d < radius: moving = False
                elif d > radius or speed > speed_lim:
                    # catch up exponentially toward the observation; declare moving once close or fast
                    k = 1 - math.exp(-dt / max(catch, 1e-3))
                    anchor = [anchor[i] + (p[i] - anchor[i]) * k for i in range(3)]
                    if norm(sub(p, anchor)) < radius * 0.3 or speed > 2 * speed_lim: moving = True; anchor = list(p)
                # else: hold the anchor
            out[side].append((t, tuple(anchor), p))
            t_prev = t; p_prev = p
    return out

print(a.path)
print('radius,speed,catch | still spread L/R mm | drift L/R mm | step p99 / max mm | path m | lag moving mm | lag at onset mm')
for s in a.settings:
    r, sp, c = (float(x) for x in s.split(','))
    out = run(r, sp, c)
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
            rows.append((spread, 1000 * max(norm(sub(p, q)) for p in means for q in means)))
        rows.sort(); top = rows[:10]
        still[side] = sorted(x[0] for x in top)[len(top) // 2]; drift[side] = sorted(x[1] for x in top)[len(top) // 2]
    steps = []; path = 0; lags = []; onset = []
    for side in (0, 1):
        d = sorted(out[side]); prev = None; slow_run = 0; onset_t = None
        for t, f, raw in d:
            if prev is not None and 0.001 < t - prev[0] <= 0.07:
                st = 1000 * norm(sub(f, prev[1])); steps.append(st); path += st / 1000
                speed = norm(sub(raw, prev[2])) / (t - prev[0])
                if speed > 0.3:
                    lags.append(1000 * norm(sub(f, raw)))
                    if slow_run >= 10 and onset_t is None: onset_t = t
                    slow_run = 0
                else:
                    slow_run += 1
                if onset_t is not None:
                    if t - onset_t <= 0.15: onset.append(1000 * norm(sub(f, raw)))
                    else: onset_t = None
            prev = (t, f, raw)
    steps.sort()
    print(f'{s:16s} | {still.get(0, float("nan")):5.1f} / {still.get(1, float("nan")):5.1f}        | {drift.get(0, float("nan")):5.1f} / {drift.get(1, float("nan")):5.1f}  | {steps[int(len(steps) * .99)]:5.1f} / {steps[-1]:6.1f}   | {path:5.1f}  | {sum(lags) / max(1, len(lags)):5.1f}        | {sum(onset) / max(1, len(onset)):5.1f} (n={len(onset)})')
