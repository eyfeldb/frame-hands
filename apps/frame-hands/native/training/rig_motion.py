"""Does a tighter wrist prior lag fast motion? Per rig trace: distribution of the per-frame wrist step of stereo
fits (p50/p90/p99/max, mm) and the total path length per second of tracking; a prior that lags shrinks the large
steps and the path length, a prior that only removes jitter shrinks the small steps.
usage: python rig_motion.py <rig.txt> [<rig.txt> ...]"""
import math, sys

for path in sys.argv[1:]:
    with open(path, errors='replace') as f:
        lines = [l.split() for l in f]
    stamp = None; last = {}; steps = []; path_len = 0.0; secs = 0.0; t_first = None
    for i, p in enumerate(lines):
        if not p: continue
        if p[0] == 'frame':
            stamp = int(p[1]); t_first = t_first if t_first is not None else stamp
        elif p[0] == 'rig' and len(p) >= 13 and p[3] == 'fitted':
            nxt = lines[i + 1] if i + 1 < len(lines) else []
            if len(nxt) >= 2 and nxt[0] == 'mono' and nxt[1] == p[1]: continue
            src = p[1]; w = (float(p[10]), float(p[11]), float(p[12]))
            if src in last and stamp - last[src][0] <= 70_000_000:
                d = math.dist(w, last[src][1]); steps.append(d); path_len += d
            last[src] = (stamp, w)
    steps.sort(); n = len(steps)
    q = lambda f: steps[min(n - 1, int(n * f))]
    secs = (stamp - t_first) / 1e9
    large = sum(1 for s in steps if s > 20)
    print(f'{path.split(chr(92))[-2]}: steps {n}, p50 {q(.5):.1f} p90 {q(.9):.1f} p99 {q(.99):.1f} max {steps[-1]:.1f} mm; '
          f'steps > 20 mm: {large}; path {path_len/1000:.1f} m over {secs:.0f} s')
