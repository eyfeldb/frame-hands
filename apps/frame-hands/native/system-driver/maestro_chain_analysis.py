"""Cross-stage analysis of one maestro_capture session folder:
  headset recorder chunks (scn-*.jsonl: pose.json packets + FHMODEL2 mailbox at 90 Hz),
  capture.csv from capture-session.exe (PC SteamVR device poses + PC mailbox, ~250 Hz),
  relay-status.json, vrserver-slice.txt, headset-journal.txt.
Reports, per hand: availability at each stage (headset tracker -> headset mailbox -> PC mailbox -> SteamVR device
valid) with gap counts/durations so a disappearance can be attributed; PC mailbox vs SteamVR device position
agreement and lag; the angle between the SteamVR device forward axis and the mailbox wrist->index direction (its
spread is the orientation noise the game sees, its mean the convention offset); head motion; relay delay stats;
SteamVR log events (device connect/disconnect, role changes, vrlink hand devices).
usage: python maestro_chain_analysis.py <session-dir>
"""
import csv, json, math, re, sys
from pathlib import Path
import numpy as np

d = Path(sys.argv[1])


def gaps(t, ok, min_gap=0.1):
    """-> list of (start, duration) where ok is False for at least min_gap seconds."""
    out = []; start = None
    for i in range(len(t)):
        if not ok[i] and start is None: start = t[i]
        if ok[i] and start is not None:
            if t[i] - start >= min_gap: out.append((start, t[i] - start))
            start = None
    if start is not None and t[-1] - start >= min_gap: out.append((start, t[-1] - start))
    return out


print(f'== {d.name}')
# ---- relay status
rs = d / 'relay-status.json'
if rs.exists():
    s = json.loads(rs.read_text())
    print(f"relay: {s.get('status')}; packets {s.get('packets')} hand_outputs {s.get('hand_outputs')}; one-way delay p50/p90/max "
          f"{s.get('delay_p50_ms')}/{s.get('delay_p90_ms')}/{s.get('delay_max_ms')} ms; max rtt {s.get('max_rtt_ms')} ms")

# ---- headset chunks
raw_hands = {0: [], 1: []}; shown = {0: [], 1: []}; shown_t = []
for f in sorted(d.glob('scn-*.jsonl')):
    m = re.search(r'scn-(\d{2})(\d{2})(\d{2})', f.name); base = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) if m else 0
    for line in open(f, errors='replace'):
        try: r = json.loads(line)
        except ValueError: continue
        if r.get('kind') == 'raw':
            if r.get('context_valid'):
                for h in r.get('hands', []): raw_hands[h['side']].append((r['capture_ns'] / 1e9, h['source_id']))
        elif r.get('kind') == 'shown':
            tt = base + r['probe_t']; shown_t.append(tt)
            present = {h['side']: h for h in r.get('hands', [])}
            for s in (0, 1): shown[s].append((tt, s in present, present[s]['capture_ns'] / 1e6 if s in present else None))
if shown_t:
    for s in (0, 1):
        t = np.array([x[0] for x in shown[s]]); ok = np.array([x[1] for x in shown[s]])
        g = gaps(t, ok)
        print(f"headset mailbox {'LR'[s]}: present {ok.mean():.0%} of {len(ok)} packets over {t[-1] - t[0]:.0f} s; gaps >=0.1 s: {len(g)}"
              + (f", longest {max(x[1] for x in g):.2f} s" if g else ''))
else:
    print('headset chunks: none (recorder had no session during the capture)')

# ---- PC capture
cp = d / 'capture.csv'
if cp.exists():
    rows = [r for r in csv.DictReader(open(cp, newline='')) if None not in r.values() and all(v != '' for v in r.values())]
    if rows:
        col = lambda k: np.array([float(r[k]) for r in rows])
        t = col('t_ms') / 1000; t -= t[0]
        hmd = np.stack([col('hmd_x'), col('hmd_y'), col('hmd_z')], 1)
        print(f"pc capture: {len(rows)} rows over {t[-1]:.0f} s; HMD position range {np.ptp(hmd, 0).max() * 100:.0f} cm; mailbox readable {col('mb_ok').mean():.0%}")
        for s in 'lr':
            mb_ok = (col(f'mb_{s}_flags') > 0) & (col('mb_ok') > 0)
            dev_ok = (col(f'{s}_conn') > 0) & (col(f'{s}_valid') > 0)
            idx = col(f'{s}_idx')
            gm = gaps(t, mb_ok); gd = gaps(t, dev_ok)
            print(f"  {s.upper()}: PC mailbox hand present {mb_ok.mean():.0%} (gaps {len(gm)}" + (f", longest {max(x[1] for x in gm):.2f} s" if gm else '') +
                  f");  SteamVR device valid {dev_ok.mean():.0%} (gaps {len(gd)}" + (f", longest {max(x[1] for x in gd):.2f} s" if gd else '') +
                  f"); device index {int(np.median(idx[idx >= 0])) if (idx >= 0).any() else -1}")
            both = mb_ok & dev_ok
            if both.sum() > 50:
                mbw = np.stack([col(f'mb_{s}_wx'), col(f'mb_{s}_wy'), col(f'mb_{s}_wz')], 1)[both]
                mbi = np.stack([col(f'mb_{s}_ix'), col(f'mb_{s}_iy'), col(f'mb_{s}_iz')], 1)[both]
                dev = np.stack([col(f'{s}_x'), col(f'{s}_y'), col(f'{s}_z')], 1)[both]
                fwd = np.stack([col(f'{s}_fx'), col(f'{s}_fy'), col(f'{s}_fz')], 1)[both]
                dist = np.linalg.norm(dev - mbw, axis=1) * 1000
                wi = mbi - mbw; wi /= np.maximum(1e-6, np.linalg.norm(wi, axis=1))[:, None]
                ang = np.degrees(np.arccos(np.clip((wi * fwd).sum(1), -1, 1)))
                # orientation noise: angle change of the device forward axis per 11 ms step where the mailbox did not change
                same = np.linalg.norm(np.diff(mbw, axis=0), axis=1) < 1e-6
                dfwd = np.degrees(np.arccos(np.clip((fwd[1:] * fwd[:-1]).sum(1), -1, 1)))
                print(f"     device vs mailbox wrist: median offset {np.median(dist):.0f} mm, p90 {np.percentile(dist, 90):.0f} mm")
                print(f"     angle(device forward, wrist->index tip): mean {ang.mean():.0f} deg, std {ang.std():.1f} deg (spread = orientation noise reaching the game)")
                if same.any(): print(f"     device forward change between rows with an unchanged mailbox: median {np.median(dfwd[same]):.2f} deg (should be ~0: anything else is driver/SteamVR-side motion)")
                # per-second attribution table of dropouts
                mbs = col(f'mb_{s}_capture')[both]
                print(f"     mailbox capture stamps advance in {np.mean(np.diff(mbs) > 0):.0%} of rows (30 Hz feed into a 250 Hz capture => ~12% expected)")

# ---- SteamVR log slice
vs = d / 'vrserver-slice.txt'
if vs.exists():
    txt = vs.read_text(errors='replace').splitlines()
    pats = {'fh_skeletal': r'fh_skeletal', 'optical device': r'frame-hands-optical', 'vrlink controller': r'SVLController', 'role': r'[Rr]ole', 'standby': r'[Ss]tandby', 'error': r'\[Error\]'}
    counts = {k: sum(1 for l in txt if re.search(p, l)) for k, p in pats.items()}
    print('steamvr log slice: ' + ', '.join(f'{k} {v}' for k, v in counts.items()) + f' ({len(txt)} lines)')
    for l in txt:
        if re.search(r'frame-hands-optical|SVLController|Frame Hands|role', l) and 'Checking input type' not in l: print('   ' + l[:170])
# ---- headset journal
hj = d / 'headset-journal.txt'
if hj.exists():
    lines = [l for l in hj.read_text(errors='replace').splitlines() if re.search(r'Camera frames stopped|restart|Native hands ready|standby|Put Frame', l)]
    print('headset journal events: ' + str(len(lines)))
    for l in lines[-8:]: print('   ' + l[:170])
