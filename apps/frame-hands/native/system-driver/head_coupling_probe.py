"""Head-coupling test for the native hand pipeline. Protocol: wearer rests both hands on the desk (truly still in
the world) and slowly moves/turns the head for the whole capture. The probe logs every pose.json packet (world
wrist + head pose at capture time, 30 Hz) and the shown mailbox wrist (90 Hz), then reports, per hand:
  world wrist RMS (should be ~2-4 mm if the hands are still and the head pose is right),
  head-frame wrist RMS (the tracker's own noise),
  the time offset tau of the head pose that minimises world wrist motion (0 = capture-time head is right; a
  nonzero tau means the hand is placed with a head pose from the wrong moment, so it swims with head motion),
  and the RMS at that best tau.
usage: python3 head_coupling_probe.py <session-dir> <input.bin> [seconds=30] [--csv out.csv]
analysis only on a saved csv: python3 head_coupling_probe.py --analyse out.csv
"""
import json, math, struct, sys, time
from pathlib import Path

H = struct.Struct('<8sIIQQQ'); HAND = struct.Struct('<IIQQ63f')


def mat(rows): return [[float(x) for x in r] for r in rows]


def inv(m):
    R = [[m[j][i] for j in range(3)] for i in range(3)]; t = [m[i][3] for i in range(3)]
    ti = [-sum(R[i][k] * t[k] for k in range(3)) for i in range(3)]
    return [R[0] + [ti[0]], R[1] + [ti[1]], R[2] + [ti[2]]]


def apply(m, p): return [sum(m[i][k] * p[k] for k in range(3)) + m[i][3] for i in range(3)]


def capture(sess, mb, seconds, csv):
    last_cap = None; last_seq = None; t0 = time.monotonic(); end = t0 + seconds
    out = open(csv, 'w') if csv else None; rows = []
    while time.monotonic() < end:
        now = time.monotonic() - t0
        try: d = json.loads((sess / 'pose.json').read_text())
        except Exception: d = None
        if d and d.get('capture_ns') != last_cap and d.get('context_valid'):
            last_cap = d['capture_ns']; head = d.get('head_world_m')
            for h in d.get('hands', []):
                w = h['wrist_world_m']; rec = ('raw', h['side'], now, d['capture_ns'], w, head)
                rows.append(rec)
                if out: out.write(json.dumps(rec) + '\n')
        try: b = mb.read_bytes()
        except OSError: b = b''
        if len(b) == 592:
            seq = H.unpack_from(b)[3]
            if seq != last_seq:
                last_seq = seq
                for s in range(2):
                    ident, flags, cap, until, *pts = HAND.unpack_from(b, H.size + s * HAND.size)
                    if flags:
                        rec = ('shown', s, now, cap, pts[0:3], None); rows.append(rec)
                        if out: out.write(json.dumps(rec) + '\n')
        time.sleep(0.004)
    if out: out.close()
    return rows


def analyse(rows):
    import numpy as np
    for side in (0, 1):
        raw = [r for r in rows if r[0] == 'raw' and r[1] == side and r[5]]
        if len(raw) < 60: print(f'{"LR"[side]}: only {len(raw)} raw frames'); continue
        t = np.array([r[3] for r in raw]) / 1e9; t -= t[0]
        world = np.array([[r[4][i][3] for i in range(3)] for r in raw])
        heads = [mat(r[5]) for r in raw]
        local = np.array([apply(inv(h), w) for h, w in zip(heads, world)])
        def rms(p): d = p - p.mean(0); return math.sqrt((d ** 2).sum(1).mean()) * 1000
        print(f'{"LR"[side]}: {len(raw)} frames over {t[-1]:.1f} s;  world wrist RMS {rms(world):.1f} mm;  head-frame wrist RMS {rms(local):.1f} mm')
        # head motion magnitude
        hp = np.array([[h[i][3] for i in range(3)] for h in heads]); print(f'   head position range {np.ptp(hp, 0).max() * 1000:.0f} mm')
        # best time offset: re-place the head-frame wrist with the head pose at t + tau (nearest sample)
        best = None
        for tau in np.arange(-0.15, 0.151, 0.01):
            idx = np.searchsorted(t, t + tau); idx = np.clip(idx, 0, len(t) - 1)
            w2 = np.array([apply(heads[j], local[i]) for i, j in enumerate(idx)])
            r = rms(w2)
            if best is None or r < best[1]: best = (tau, r)
        print(f'   best head-pose time offset {best[0] * 1000:+.0f} ms -> world RMS {best[1]:.1f} mm (0 ms = capture-time head pose is right)')
        shown = np.array([r[4] for r in rows if r[0] == 'shown' and r[1] == side])
        if len(shown) > 60: print(f'   shown (mailbox) wrist RMS {rms(shown):.1f} mm over {len(shown)} packets')


if __name__ == '__main__':
    if sys.argv[1] == '--analyse':
        analyse([tuple(json.loads(l)) for l in open(sys.argv[2])])
    else:
        sess = Path(sys.argv[1]); mb = Path(sys.argv[2])
        seconds = float(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else 30
        csv = sys.argv[sys.argv.index('--csv') + 1] if '--csv' in sys.argv else None
        rows = capture(sess, mb, seconds, csv)
        try: analyse(rows)
        except ImportError: print('numpy missing here; analyse the csv on the PC with --analyse')
