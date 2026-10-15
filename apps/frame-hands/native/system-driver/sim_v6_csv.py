"""Replay the raw wrist rows of a dual_probe.py CSV through the v6 speed-adaptive smoother (position only) and
compare raw vs smoothed per-frame steps per 0.5 s window, plus the direction consistency of raw steps
(fraction of consecutive step pairs pointing the same way: ~0.5 = noise, ~1 = real motion).
usage: python sim_v6_csv.py dual.csv [cutoff_min=1.0] [slope=20] [noise_speed=0.08] [speed_cutoff=2.0]"""
import math, sys
from collections import defaultdict
path = sys.argv[1]; cmin = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0; slope = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
noise = float(sys.argv[4]) if len(sys.argv) > 4 else 0.08; scut = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0

def alpha(c, dt): return 1 / (1 + 1 / (2 * math.pi * c * dt))

raw = defaultdict(list)
for line in open(path):
    f = line.strip().split(',')
    if f[0] != 'raw': continue
    raw[int(f[1])].append((float(f[2]), int(f[3]), (float(f[4]), float(f[5]), float(f[6]))))
for side in sorted(raw):
    rows = raw[side]; pos = None; sp = 0.0; prev = None; prev_step = None
    out = defaultdict(lambda: [[], [], [], []])  # raw steps, shown steps, same-direction flags, speed_filt
    for t, sid, p in rows:
        if prev is None or t - prev[0] > 0.15 or sid != prev[1]:
            pos = p; sp = 0.0; prev = (t, sid, p); prev_step = None; continue
        dt = t - prev[0]
        step = tuple(p[i] - prev[2][i] for i in range(3)); n = math.sqrt(sum(x * x for x in step))
        if prev_step is not None:
            dot = sum(step[i] * prev_step[i] for i in range(3))
            out[int(t * 2) / 2][2].append(dot > 0)
        v = n / dt; sp += (v - sp) * alpha(scut, dt)
        a = alpha(cmin + slope * max(0.0, sp - noise), dt)
        new = tuple(pos[i] + (p[i] - pos[i]) * a for i in range(3))
        out[int(t * 2) / 2][0].append(n * 1000); out[int(t * 2) / 2][1].append(math.sqrt(sum((new[i] - pos[i]) ** 2 for i in range(3))) * 1000)
        out[int(t * 2) / 2][3].append(sp)
        pos = new; prev = (t, sid, p); prev_step = step
    print(f'{"LR"[side]}: win  raw p50/max mm  same-dir  speed_filt | v6 shown p50/max mm')
    for w in sorted(out):
        r, s, d, spd = out[w]
        if not r: continue
        r2 = sorted(r); s2 = sorted(s)
        print(f'  {w:4.1f}  {r2[len(r2) // 2]:5.1f}/{r2[-1]:6.1f}   {sum(d) / max(1, len(d)):4.2f}   {sum(spd) / len(spd):5.2f} | {s2[len(s2) // 2]:5.2f}/{s2[-1]:6.1f}')
