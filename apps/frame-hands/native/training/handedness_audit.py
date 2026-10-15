"""Handedness over time from a live audit (live-views.txt): per 5 s window, sets with two hands, sets where both
hands carry the same label, and each hand's right-score median by wrist x (position side), to see where the
model's left/right output disagrees with where the hand is.
usage: python handedness_audit.py <live-views.txt> [--window 5]
"""
import argparse
from collections import defaultdict

ap = argparse.ArgumentParser(); ap.add_argument('path'); ap.add_argument('--window', type=float, default=5)
a = ap.parse_args()
sets = []   # (t, [(right_score, nviews, x_mean)])
cur = None; hands = {}; views = defaultdict(list)
with open(a.path, errors='replace') as f:
    for line in f:
        p = line.split()
        if not p: continue
        if p[0] == 'set':
            if cur is not None: sets.append((cur, [(r, nv, sum(views[h]) / len(views[h]) if views[h] else None) for h, (r, nv) in hands.items()]))
            cur = int(p[1]); hands = {}; views = defaultdict(list)
        elif p[0] == 'hand' and cur is not None: hands[p[1]] = (float(p[2]), int(p[3]))
        elif p[0] == 'view' and cur is not None and len(p) >= 7:
            try: views[p[2]].append(float(p[5]))
            except ValueError: pass
if cur is not None: sets.append((cur, [(r, nv, sum(views[h]) / len(views[h]) if views[h] else None) for h, (r, nv) in hands.items()]))
t0 = sets[0][0]
print(f'{len(sets)} sets; window  two-hand sets  same-label  stereo hands  right-score of hands (sorted by image x) ')
w = a.window; n = int((sets[-1][0] - t0) / 1e9 // w) + 1
tot2 = tot_same = 0
for i in range(n):
    win = [s for s in sets if t0 + i * w * 1e9 <= s[0] < t0 + (i + 1) * w * 1e9]
    two = [s for s in win if len(s[1]) == 2]
    same = [s for s in two if (s[1][0][0] > 0.5) == (s[1][1][0] > 0.5)]
    stereo = sum(1 for s in win for h in s[1] if h[1] >= 2)
    scores = []
    for s in two:
        hs = sorted(s[1], key=lambda h: (h[2] if h[2] is not None else 0))
        scores.append((hs[0][0], hs[1][0]))
    med = lambda k: sorted(x[k] for x in scores)[len(scores) // 2] if scores else float('nan')
    tot2 += len(two); tot_same += len(same)
    print(f'  {i * w:5.0f}s   {len(two):4d}          {len(same):4d}        {stereo:5d}        {med(0):.2f} / {med(1):.2f}')
print(f'total two-hand sets {tot2}, same label {tot_same} ({100 * tot_same / max(1, tot2):.0f}%)')
