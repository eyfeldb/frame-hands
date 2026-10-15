"""Sample an FHMODEL2 hand mailbox for N seconds: update cadence, hand presence, capture ages and wrist motion
between fresh observations (jitter proxy). Works on the headset (/run/user/<uid>/frame-hands-native/input.bin)
and on the PC (the registered bundle's bin/win64/input.bin).
usage: python sample_mailbox.py <input.bin> [seconds]"""
import math, struct, sys, time
from pathlib import Path
p = Path(sys.argv[1]); seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 10
H = struct.Struct('<8sIIQQQ'); HAND = struct.Struct('<IIQQ63f')
now_ms = lambda: time.monotonic_ns() // 1000000 if sys.platform != 'win32' else time.perf_counter_ns() // 1000000
last = None; seqs = []; ages = []; hands = []; last_obs = {}; steps = {0: [], 1: []}; obs_gaps = {0: [], 1: []}
end = time.monotonic() + seconds
while time.monotonic() < end:
    try: d = p.read_bytes()
    except OSError: time.sleep(.002); continue
    if len(d) != 592: time.sleep(.002); continue
    magic, ver, size, seq, published, source = H.unpack_from(d)
    if seq != last:
        last = seq; n = now_ms(); seqs.append(n); c = 0
        for s in range(2):
            ident, flags, cap, until, *pts = HAND.unpack_from(d, H.size + s * HAND.size)
            if not flags: continue
            c += 1; ages.append(n - cap); wrist = pts[0:3]
            prev = last_obs.get(s)
            if prev is None or prev[0] != cap:          # a fresh observation (capture stamp changed)
                if prev is not None and 0 < cap - prev[0] <= 100:
                    steps[s].append(1000 * math.dist(wrist, prev[1])); obs_gaps[s].append(cap - prev[0])
                last_obs[s] = (cap, wrist)
        hands.append(c)
    time.sleep(.002)
pct = lambda a, q: (round(sorted(a)[int(len(a) * q)], 1) if a else None)
print('updates', len(seqs), 'in', seconds, 's; hands per update', {k: hands.count(k) for k in sorted(set(hands))})
if len(seqs) > 2:
    g = [b - a for a, b in zip(seqs, seqs[1:])]
    print('update gap ms p50/p90/max', pct(g, .5), pct(g, .9), max(g))
    print('hand capture age ms p50/p90/max', pct(ages, .5), pct(ages, .9), max(ages) if ages else None)
    for s in (0, 1):
        print(f'side {s}: fresh observations {len(steps[s])}, observation gap ms p50/p90 {pct(obs_gaps[s], .5)}/{pct(obs_gaps[s], .9)}, wrist step mm p50/p90 {pct(steps[s], .5)}/{pct(steps[s], .9)}')
