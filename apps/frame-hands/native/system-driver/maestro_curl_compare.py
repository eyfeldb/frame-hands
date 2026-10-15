"""Compare finger articulation at the two ends of the Maestro chain for one capture session:
  headset side: finger bends computed from the 21 published points (scn-*.jsonl 'shown' rows, 90 Hz)
  game side:    SteamVR skeletal summary finger curls (0..1) from capture.csv, which SteamVR derives from the 31
                bone transforms our PC driver publishes (what Maestro's hand mesh is driven by)
If the headset shows fists (PIP bends 90+ deg) but SteamVR curls stay low, the retarget/bone convention is what
breaks the hand shape in the game.
usage: python maestro_curl_compare.py <session-dir>
"""
import csv, json, math, re, sys
from pathlib import Path
import numpy as np

d = Path(sys.argv[1])
FINGERS = {'index': (5, 6, 7, 8), 'middle': (9, 10, 11, 12), 'ring': (13, 14, 15, 16), 'pinky': (17, 18, 19, 20)}


def ang(a, b, c):
    v1 = np.subtract(a, b); v2 = np.subtract(c, b); n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9: return float('nan')
    return 180 - math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2) / (n1 * n2))))))


# headset: absolute time from chunk name + probe_t
bends = {0: [], 1: []}
for f in sorted(d.glob('scn-*.jsonl')):
    m = re.search(r'scn-(\d{2})(\d{2})(\d{2})', f.name); base = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
    for line in open(f, errors='replace'):
        try: r = json.loads(line)
        except ValueError: continue
        if r.get('kind') != 'shown': continue
        t = base + r['probe_t']
        for h in r.get('hands', []):
            P = h['points']
            bends[h['side']].append((t, [ang(P[i[0]], P[i[1]], P[i[2]]) for i in FINGERS.values()], ang(P[2], P[3], P[4])))
for s in (0, 1):
    if not bends[s]: continue
    t = np.array([b[0] for b in bends[s]]); pip = np.array([b[1] for b in bends[s]])
    print(f"headset {'LR'[s]}: {len(t)} packets {t[0] % 86400 / 3600:.3f}h..{t[-1] % 86400 / 3600:.3f}h; PIP bend deg p50/p90/max per finger: " +
          ' '.join(f"{n} {np.nanpercentile(pip[:, i], 50):.0f}/{np.nanpercentile(pip[:, i], 90):.0f}/{np.nanmax(pip[:, i]):.0f}" for i, n in enumerate(FINGERS)))
    fist = (pip[:, :3] > 80).all(1)
    print(f"   frames with index+middle+ring PIP > 80 deg (fist): {fist.mean():.0%}")

# PC: skeletal summary curls
rows = [r for r in csv.DictReader(open(d / 'capture.csv', newline='')) if None not in r.values() and all(v != '' for v in r.values())]
for s in 'lr':
    ok = np.array([float(r[f'{s}_valid']) > 0 and float(r[f'{s}_skel_err']) == 0 for r in rows])
    curls = np.array([[float(r[f'{s}_curl_{n}']) for n in ('thumb', 'index', 'middle', 'ring', 'pinky')] for r in rows])[ok]
    if not len(curls): print(f'pc {s.upper()}: no valid skeleton rows'); continue
    print(f"pc {s.upper()}: {len(curls)} valid rows; SteamVR summary curl p50/p90/max per finger: " +
          ' '.join(f"{n} {np.percentile(curls[:, i], 50):.2f}/{np.percentile(curls[:, i], 90):.2f}/{curls[:, i].max():.2f}" for i, n in enumerate(('thumb', 'index', 'middle', 'ring', 'pinky'))))
    fist = (curls[:, 1:4] > 0.6).all(1)
    print(f"   rows with index+middle+ring curl > 0.6 (fist): {fist.mean():.0%}")
# SteamVR log errors
vs = d / 'vrserver-slice.txt'
if vs.exists():
    for l in vs.read_text(errors='replace').splitlines():
        if '[Error]' in l or 'standby' in l.lower() or 'fh_skeletal' in l: print('   log: ' + l[:170])
