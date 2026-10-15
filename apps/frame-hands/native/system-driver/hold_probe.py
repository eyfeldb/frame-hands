"""Frame-rate probe of the FHMODEL2 mailbox for the hold stabiliser: samples every new packet for N seconds and
reports, per hand, how often the shown wrist moved (> 1 mm) and how often the index PIP bend changed (> 1 deg)
between consecutive packets, plus the largest single-step changes. A stationary hand should show near-zero
change fractions; a moving one shows large fractions with small steps (no jumps).
usage: python3 hold_probe.py <input.bin> [seconds]"""
import math, struct, sys, time
from pathlib import Path
p = Path(sys.argv[1]); seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 10
H = struct.Struct('<8sIIQQQ'); HAND = struct.Struct('<IIQQ63f')

def ang(a, b, c):
    v1 = [a[i] - b[i] for i in range(3)]; v2 = [c[i] - b[i] for i in range(3)]
    n1 = math.sqrt(sum(x * x for x in v1)); n2 = math.sqrt(sum(x * x for x in v2))
    if n1 < 1e-9 or n2 < 1e-9: return float('nan')
    return 180.0 - math.degrees(math.acos(max(-1.0, min(1.0, sum(v1[i] * v2[i] for i in range(3)) / (n1 * n2)))))

end = time.monotonic() + seconds; last = None
rows = {0: [], 1: []}
while time.monotonic() < end:
    try: d = p.read_bytes()
    except OSError: time.sleep(.005); continue
    if len(d) != 592: time.sleep(.005); continue
    seq = H.unpack_from(d)[3]
    if seq == last: time.sleep(.003); continue
    last = seq
    for s in range(2):
        ident, flags, cap, until, *pts = HAND.unpack_from(d, H.size + s * HAND.size)
        if not flags: rows[s].append(None); continue
        P = [pts[3 * k:3 * k + 3] for k in range(21)]
        rows[s].append((time.monotonic(), P[0], ang(P[5], P[6], P[7]), ang(P[9], P[10], P[11]), ang(P[2], P[3], P[4])))
for s in range(2):
    r = rows[s]; n = sum(1 for x in r if x); print(f'{"LR"[s]}: {len(r)} packets, {n} with hand')
    moved = changed = pairs = 0; steps = []; asteps = []; per_sec = {}
    for a, b in zip(r, r[1:]):
        if not a or not b: continue
        pairs += 1
        dm = math.dist(a[1], b[1]) * 1000; da = max(abs(a[2] - b[2]), abs(a[3] - b[3]), abs(a[4] - b[4]))
        moved += dm > 1; changed += da > 1; steps.append(dm); asteps.append(da)
        sec = int(b[0] - r[0][0]) if r[0] else int(b[0]); per_sec.setdefault(sec, [0, 0, 0]); per_sec[sec][0] += 1; per_sec[sec][1] += dm > 1; per_sec[sec][2] += da > 1
    if pairs:
        steps.sort(); asteps.sort()
        print(f'  pairs {pairs}: wrist moved >1mm in {moved / pairs:.0%}, max step {steps[-1]:.1f} mm, p90 {steps[int(.9 * (pairs - 1))]:.1f} mm')
        print(f'  index/middle PIP or thumb IP changed >1deg in {changed / pairs:.0%}, max step {asteps[-1]:.0f} deg, p90 {asteps[int(.9 * (pairs - 1))]:.0f} deg')
        print('  per second (pairs, wrist-moved, angle-changed): ' + ' '.join(f'{k}:{v[0]}/{v[1]}/{v[2]}' for k, v in sorted(per_sec.items())))
