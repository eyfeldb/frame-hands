"""Episodes in a replay timeline (ft-handreplay-rig --timeline): per frame the hands present by side.

Lists stretches where a side is absent for >= min_gap seconds, stretches with three or more hands, and stretches
with two hands of the same side (duplicates), with start time and length. Timeline lines look like
"<t> <id> <L|R> <nviews> x y z" for each hand of a frame; other lines (curl, pinchd, view, '-') are ignored.
usage: python timeline_episodes.py <timeline.txt> [--min-gap 0.5]
"""
import argparse
from collections import defaultdict

ap = argparse.ArgumentParser(); ap.add_argument('path'); ap.add_argument('--min-gap', type=float, default=0.5)
a = ap.parse_args()
frames = defaultdict(list)   # t -> [(id, side)]
times = []
with open(a.path, errors='replace') as f:
    for line in f:
        p = line.split()
        if len(p) < 2: continue
        try: t = float(p[0])
        except ValueError: continue
        if t not in frames: times.append(t); frames[t]
        if len(p) >= 7 and p[2] in ('L', 'R') and p[1].isdigit():
            frames[t].append((int(p[1]), p[2]))
times.sort()
dur = times[-1] - times[0] if times else 0
print(f'{len(times)} frames over {dur:.1f} s')

def episodes(pred, label, min_len):
    out = []; start = None
    for i, t in enumerate(times):
        hit = pred(frames[t])
        if hit and start is None: start = t
        if not hit and start is not None:
            if t - start >= min_len: out.append((start, t - start))
            start = None
    if start is not None and times[-1] - start >= min_len: out.append((start, times[-1] - start))
    total = sum(d for _, d in out)
    print(f'\n{label}: {len(out)} episodes, {total:.1f} s total')
    for s, d in out: print(f'  {s:7.2f} s  for {d:5.2f} s')

episodes(lambda h: not any(s == 'L' for _, s in h), 'left hand absent', a.min_gap)
episodes(lambda h: not any(s == 'R' for _, s in h), 'right hand absent', a.min_gap)
episodes(lambda h: len(h) >= 3, 'three or more hands', 0.1)
episodes(lambda h: sum(s == 'L' for _, s in h) >= 2, 'two left hands', 0.1)
episodes(lambda h: sum(s == 'R' for _, s in h) >= 2, 'two right hands', 0.1)
episodes(lambda h: len(h) == 0, 'no hands', a.min_gap)
