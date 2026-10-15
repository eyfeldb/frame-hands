"""2D stability of the live tracker from a session's live-views.txt (FRAME_HANDS_AUDIT).

For each hand side and camera: wrist pixel step between consecutive sets (median, p90) and, per 10 s window,
the wrist position's standard deviation after removing a 1 s moving average (jitter, not motion), plus how many
sets saw the hand in that camera. Tells whether a "hands move while still" complaint originates in the tracker's
image-space output or further downstream (rig fit, PC transport, game).

usage: python live_views_jitter.py <live-views.txt> [--window 10]
"""
import argparse, math, statistics, sys
from collections import defaultdict

def parse(path):
    # audit format (track/main.cpp): "set <t> <capture> <n>", "hand <id> <right_score> <nviews>",
    # "view <cam> <hand id> <presence> <right> x0 y0 ... x20 y20", "end"
    sets = []  # (t_ns, {(side, cam): (x, y)})
    cur = None
    sides = {}
    with open(path, errors='replace') as f:
        for line in f:
            p = line.split()
            if not p: continue
            if p[0] == 'set':
                cur = (int(p[1]), {}); sets.append(cur); sides = {}
            elif p[0] == 'hand' and cur is not None:
                sides[p[1]] = 1 if float(p[2]) > 0.5 else 0
            elif p[0] == 'view' and cur is not None and p[2] in sides and len(p) >= 47:
                cam = p[1]
                try: x, y = float(p[5]), float(p[6])
                except ValueError: continue
                cur[1][(sides[p[2]], cam)] = (x, y)
    return sets

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('path'); ap.add_argument('--window', type=float, default=10)
    a = ap.parse_args()
    sets = parse(a.path)
    if len(sets) < 2: print('no sets'); return
    t0 = sets[0][0]
    dur = (sets[-1][0] - t0) / 1e9
    print(f'{len(sets)} sets over {dur:.1f} s ({len(sets)/max(dur,1e-9):.1f}/s)')
    keys = sorted({k for _, d in sets for k in d})
    for key in keys:
        side, cam = key
        pts = [(t, d[key]) for t, d in sets if key in d]
        steps = []
        for (ta, pa), (tb, pb) in zip(pts, pts[1:]):
            if tb - ta < 60_000_000:  # consecutive frames only
                steps.append(math.hypot(pb[0]-pa[0], pb[1]-pa[1]))
        if not steps: continue
        steps.sort()
        print(f'\n{"left" if side == 0 else "right"} hand in {cam}: seen in {len(pts)}/{len(sets)} sets; wrist step median {steps[len(steps)//2]:.2f} px  p90 {steps[int(len(steps)*.9)]:.2f} px')
        # per-window jitter: deviation from a centred 1 s moving average
        w = a.window
        n = int(dur // w) + 1
        rows = []
        for i in range(n):
            lo, hi = t0 + i*w*1e9, t0 + (i+1)*w*1e9
            win = [(t, p) for t, p in pts if lo <= t < hi]
            if len(win) < 10: rows.append((i, len(win), None, None)); continue
            dev = []
            for j, (t, p) in enumerate(win):
                nb = [q for s, q in win if abs(s - t) <= 500_000_000]
                mx = sum(q[0] for q in nb)/len(nb); my = sum(q[1] for q in nb)/len(nb)
                dev.append(math.hypot(p[0]-mx, p[1]-my))
            motion = statistics.pstdev([p[0] for _, p in win]) + statistics.pstdev([p[1] for _, p in win])
            rows.append((i, len(win), statistics.median(dev), motion))
        print('  window  sets  jitter(px, median dev from 1 s mean)  spread(px, std x+y)')
        for i, c, j, m in rows:
            print(f'  {i*w:5.0f}s  {c:4d}  ' + ('   -' if j is None else f'  {j:5.2f}') + ('' if m is None else f'                               {m:7.1f}'))

if __name__ == '__main__': main()
