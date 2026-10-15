"""Record raw rig output (session pose.json, 30 Hz) and shown output (FHMODEL2 mailbox) side by side, then
report per hand, per 0.5 s window: raw wrist step p50/max (mm), shown wrist step p50/max, raw index-PIP change
max (deg), shown index-PIP change max, hand presence; plus the largest shown steps with the raw step at that time.
usage: python3 dual_probe.py <session-dir> <input.bin> [seconds] [--csv out.csv]"""
import json, math, struct, sys, time
from pathlib import Path
sess = Path(sys.argv[1]); mb = Path(sys.argv[2]); seconds = float(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else 20
csv = sys.argv[sys.argv.index('--csv') + 1] if '--csv' in sys.argv else None
H = struct.Struct('<8sIIQQQ'); HAND = struct.Struct('<IIQQ63f')

def ang(a, b, c):
    v1 = [a[i] - b[i] for i in range(3)]; v2 = [c[i] - b[i] for i in range(3)]
    n1 = math.sqrt(sum(x * x for x in v1)); n2 = math.sqrt(sum(x * x for x in v2))
    if n1 < 1e-9 or n2 < 1e-9: return float('nan')
    return 180.0 - math.degrees(math.acos(max(-1.0, min(1.0, sum(v1[i] * v2[i] for i in range(3)) / (n1 * n2)))))

raw = {0: [], 1: []}; shown = {0: [], 1: []}; last_cap = None; last_seq = None
t0 = time.monotonic(); end = t0 + seconds
while time.monotonic() < end:
    now = time.monotonic() - t0
    try: d = json.loads((sess / 'pose.json').read_text())
    except Exception: d = None
    if d and d.get('capture_ns') != last_cap:
        last_cap = d.get('capture_ns')
        for h in d.get('hands', []):
            w = h['wrist_world_m']; a = h['joint_angles']
            raw[h['side']].append((now, h['source_id'], (w[0][3], w[1][3], w[2][3]), math.degrees(a[6])))
    try: b = mb.read_bytes()
    except OSError: b = b''
    if len(b) == 592:
        seq = H.unpack_from(b)[3]
        if seq != last_seq:
            last_seq = seq
            for s in range(2):
                ident, flags, cap, until, *pts = HAND.unpack_from(b, H.size + s * HAND.size)
                if flags:
                    P = [pts[3 * k:3 * k + 3] for k in range(21)]
                    shown[s].append((now, ident, P[0], ang(P[5], P[6], P[7])))
    time.sleep(0.004)

if csv:
    with open(csv, 'w') as f:
        for kind, store in (('raw', raw), ('shown', shown)):
            for s in store:
                for t, i, w, a in store[s]: f.write(f'{kind},{s},{t:.3f},{i},{w[0]:.4f},{w[1]:.4f},{w[2]:.4f},{a:.1f}\n')

def steps(rows):
    out = []
    for a, b in zip(rows, rows[1:]):
        if b[0] - a[0] > 0.2: continue
        out.append((b[0], math.dist(a[2], b[2]) * 1000, abs(b[3] - a[3]), a[1] != b[1]))
    return out

for s in (0, 1):
    rs, ss = steps(raw[s]), steps(shown[s])
    print(f'{"LR"[s]}: raw {len(raw[s])} frames ({len(set(i for _, i, _, _ in raw[s]))} ids), shown {len(shown[s])} packets')
    print('  win  raw: n  wrist p50/max mm  idx max deg | shown: n  wrist p50/max mm  idx max deg')
    w = 0.0
    while w < seconds:
        r = [x for x in rs if w <= x[0] < w + 0.5]; q = [x for x in ss if w <= x[0] < w + 0.5]
        def fmt(v):
            if not v: return '   -'
            m = sorted(x[1] for x in v); return f'{len(v):3d}  {m[len(m) // 2]:5.1f}/{m[-1]:6.1f}   {max(x[2] for x in v):5.1f}'
        ids = ' id!' if any(x[3] for x in r) else ''
        print(f'  {w:4.1f}  {fmt(r)}   | {fmt(q)}{ids}')
        w += 0.5
    big = sorted(ss, key=lambda x: -x[1])[:6]
    for t, mm, dg, idc in big:
        near = [x for x in rs if abs(x[0] - t) < 0.08]
        print(f'  shown step {mm:6.1f} mm at {t:5.2f}s (idx {dg:4.0f} deg){" new id" if idc else ""}; raw steps within 80 ms: ' + ' '.join(f'{x[1]:.0f}' for x in near))
