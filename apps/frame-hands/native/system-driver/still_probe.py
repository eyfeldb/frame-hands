"""Still-hands jitter probe on the FHMODEL2 mailbox (headset or PC).

Waits until the mailbox exists (and, with --wait-tracking, until the hands service status is "tracking"), then
samples for N seconds and reports per hand: fresh-observation cadence, wrist and index-tip step between
consecutive observations (p50/p90, mm), detrended jitter (RMS about a centred 1 s moving mean, mm) in 5 s
windows, and the largest 5 s drift of the window mean. Also writes a CSV of every observation when --csv given.
usage: python3 still_probe.py <input.bin> [seconds] [--csv out.csv] [--wait-tracking status.json]"""
import csv, json, math, struct, sys, time
from pathlib import Path

args = sys.argv[1:]
csv_path = None; status_path = None
if '--csv' in args: i = args.index('--csv'); csv_path = Path(args[i + 1]); del args[i:i + 2]
if '--wait-tracking' in args: i = args.index('--wait-tracking'); status_path = Path(args[i + 1]); del args[i:i + 2]
p = Path(args[0]); seconds = float(args[1]) if len(args) > 1 else 30
H = struct.Struct('<8sIIQQQ'); HAND = struct.Struct('<IIQQ63f')

deadline = time.monotonic() + 1800
while time.monotonic() < deadline:
    ok = p.exists()
    if ok and status_path is not None:
        try: ok = json.loads(status_path.read_text()).get('state') == 'tracking'
        except Exception: ok = False
    if ok: break
    time.sleep(0.5)
else:
    print('gave up waiting for the mailbox'); sys.exit(1)
print('sampling', seconds, 's from', time.strftime('%H:%M:%S'), flush=True)

obs = {0: [], 1: []}   # (capture_ms, wrist xyz, index tip xyz)
last_seq = None; last_cap = {}
end = time.monotonic() + seconds
while time.monotonic() < end:
    try: d = p.read_bytes()
    except OSError: time.sleep(.002); continue
    if len(d) != 592: time.sleep(.002); continue
    magic, ver, size, seq, published, source = H.unpack_from(d)
    if seq != last_seq:
        last_seq = seq
        for s in range(2):
            ident, flags, cap, until, *pts = HAND.unpack_from(d, H.size + s * HAND.size)
            if not flags or last_cap.get(s) == cap: continue
            last_cap[s] = cap
            obs[s].append((cap, tuple(pts[0:3]), tuple(pts[24:27])))
    time.sleep(.002)

if csv_path:
    with csv_path.open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['side', 'capture_ms', 'wx', 'wy', 'wz', 'ix', 'iy', 'iz'])
        for s in (0, 1):
            for cap, wr, ix in obs[s]: w.writerow([s, cap, *wr, *ix])

def pct(a, q): return round(sorted(a)[min(len(a) - 1, int(len(a) * q))], 2) if a else None
def rms(v): return math.sqrt(sum(x * x for x in v) / len(v)) if v else float('nan')

for s in (0, 1):
    o = obs[s]
    name = 'left ' if s == 0 else 'right'
    if len(o) < 5: print(f'{name}: {len(o)} observations'); continue
    gaps = [b[0] - a[0] for a, b in zip(o, o[1:])]
    wsteps = [1000 * math.dist(a[1], b[1]) for a, b in zip(o, o[1:]) if b[0] - a[0] <= 100]
    isteps = [1000 * math.dist(a[2], b[2]) for a, b in zip(o, o[1:]) if b[0] - a[0] <= 100]
    dur = (o[-1][0] - o[0][0]) / 1000
    print(f'{name}: {len(o)} observations over {dur:.1f} s ({len(o) / max(dur, 1e-9):.1f}/s), gap ms p50/p90/max {pct(gaps, .5)}/{pct(gaps, .9)}/{max(gaps)}')
    print(f'   wrist step mm p50/p90 {pct(wsteps, .5)}/{pct(wsteps, .9)}   index-tip step mm p50/p90 {pct(isteps, .5)}/{pct(isteps, .9)}')
    t0 = o[0][0]
    print('   window   obs   wrist jitter RMS mm   index jitter RMS mm   wrist drift of 1 s mean mm (max in window)   rel. index-wrist jitter mm')
    for w in range(int(dur // 5) + 1):
        win = [x for x in o if w * 5000 <= x[0] - t0 < (w + 1) * 5000]
        if len(win) < 10: continue
        dev_w, dev_i, dev_r, means = [], [], [], []
        for cap, wr, ix in win:
            nb = [x for x in win if abs(x[0] - cap) <= 500]
            mw = [sum(x[1][k] for x in nb) / len(nb) for k in range(3)]
            mi = [sum(x[2][k] for x in nb) / len(nb) for k in range(3)]
            mr = [mi[k] - mw[k] for k in range(3)]
            dev_w.append(1000 * math.dist(wr, mw)); dev_i.append(1000 * math.dist(ix, mi))
            dev_r.append(1000 * math.dist([ix[k] - wr[k] for k in range(3)], mr)); means.append(mw)
        drift = 1000 * max(math.dist(a, b) for a in means[::5] for b in means[::5])
        print(f'   {w * 5:3d}-{w * 5 + 5:<3d}  {len(win):4d}   {rms(dev_w):8.2f}            {rms(dev_i):8.2f}              {drift:8.1f}                                   {rms(dev_r):8.2f}')
