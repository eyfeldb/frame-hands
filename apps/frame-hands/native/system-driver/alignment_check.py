"""Headset-world -> PC-world alignment check for a maestro_capture session (crooked-wrist question).

Both sides track the same head. R_T = R_pc_head * R_native_head^T should be one constant rotation; the relay's
WorldAlignment estimates it and uses it to place the hands in the PC world. This script:
  1. estimates R_T from head poses (PC capture.csv HMD rows vs headset pose.json head_world_m rows), with the two
     clocks aligned by cross-correlating head angular speed, and reports how much R_T wanders over time;
  2. compares the hand direction (wrist -> index tip) published by the headset, rotated by R_T, with the direction
     the PC mailbox holds at the aligned times: the angle between them is the alignment error the game sees.
usage: python alignment_check.py <session-dir>
"""
import csv, json, math, re, sys
from pathlib import Path
import numpy as np

d = Path(sys.argv[1])


def rot_from_rows(rows): return np.array([[rows[i][j] for j in range(3)] for i in range(3)], float)
def pos_from_rows(rows): return np.array([rows[i][3] for i in range(3)], float)
def ang_between(a, b):
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return math.degrees(math.acos(max(-1, min(1, float(np.dot(a, b)) / n)))) if n > 1e-9 else float('nan')
def rot_angle(R): return math.degrees(math.acos(max(-1, min(1, (np.trace(R) - 1) / 2))))


# headset: raw rows -> (t, R_head, hands{side: (R? no: wrist, index tip)}) using mailbox points nearest in time
heads = []; hands = {0: [], 1: []}
for f in sorted(d.glob('scn-*.jsonl')):
    m = re.search(r'scn-(\d{2})(\d{2})(\d{2})', f.name); base = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) if m else 0
    for line in open(f, errors='replace'):
        try: r = json.loads(line)
        except ValueError: continue
        if r.get('kind') == 'raw' and r.get('context_valid') and r.get('head_world_m'):
            heads.append((base + r['probe_t'], rot_from_rows(r['head_world_m']), pos_from_rows(r['head_world_m'])))
        elif r.get('kind') == 'shown':
            for h in r.get('hands', []):
                P = h['points']; hands[h['side']].append((base + r['probe_t'], np.array(P[0]), np.array(P[8])))
if len(heads) < 100: raise SystemExit('too few headset head poses')
ht = np.array([h[0] for h in heads]); hR = np.stack([h[1] for h in heads])

# PC: capture.csv rows -> (t, R_hmd via forward axis only? we only have position + forward axis) -> use forward axis
rows = [r for r in csv.DictReader(open(d / 'capture.csv', newline='')) if None not in r.values() and all(v != '' for v in r.values())]
import os, datetime as dt
pt = np.array([float(r['t_ms']) for r in rows]) / 1000
end = dt.datetime.fromtimestamp(os.path.getmtime(d / 'capture.csv')); end_sod = end.hour * 3600 + end.minute * 60 + end.second + end.microsecond / 1e6
pt = end_sod - (pt[-1] - pt)   # QPC counter -> wall seconds-of-day, anchored at the file's end
pfwd = np.array([[float(r['hmd_fx']), float(r['hmd_fy']), float(r['hmd_fz'])] for r in rows])
ppos = np.array([[float(r['hmd_x']), float(r['hmd_y']), float(r['hmd_z'])] for r in rows])
hfwd = -hR[:, :, 2]   # headset forward = -z column

# clock alignment by cross-correlating head forward-axis angular speed
def speed_series(t, f):
    dt = np.diff(t); a = np.array([ang_between(f[i], f[i + 1]) for i in range(len(f) - 1)]); return t[1:], a / np.maximum(dt, 1e-3)
ts_h, sp_h = speed_series(ht, hfwd); ts_p, sp_p = speed_series(pt, pfwd)
grid = np.arange(0, min(ts_h[-1] - ts_h[0], ts_p[-1] - ts_p[0]), 0.02)
a = np.interp(grid + ts_h[0], ts_h, sp_h); b = np.interp(grid + ts_p[0], ts_p, sp_p)
a -= a.mean(); b -= b.mean(); best = (0, -1)
for lag in np.arange(-10, 10.01, 0.02):   # seconds: pc_time = headset_time + offset (both wall seconds-of-day now)
    bi = np.interp(grid + ts_p[0] + lag, ts_p, sp_p); bi -= bi.mean()
    c = float(np.dot(a, bi) / (np.linalg.norm(a) * np.linalg.norm(bi) + 1e-9))
    if c > best[1]: best = (lag, c)
lag, corr = best
print(f'clock offset (pc = headset + {lag:+.2f} s), head-motion correlation {corr:.2f} (head position range: headset {np.ptp([h[2] for h in heads], axis=0).max() * 100:.0f} cm, pc {np.ptp(ppos, axis=0).max() * 100:.0f} cm)')
if corr < 0.3: print('  weak alignment of clocks: head barely moved; the offset estimate is unreliable')

# per matched sample: angle between pc forward and headset forward (= rotation part of R_T projected on the view axis)
def nearest(t_arr, t): i = np.searchsorted(t_arr, t); i = min(max(i, 0), len(t_arr) - 1); return i
angles = []
for i in range(0, len(pt), 25):
    j = nearest(ht, pt[i] - lag); angles.append(ang_between(pfwd[i], hfwd[j]))
angles = np.array(angles)
print(f'angle(pc head forward, headset head forward) over time: median {np.median(angles):.1f} deg, p10 {np.percentile(angles, 10):.1f}, p90 {np.percentile(angles, 90):.1f} (a constant value = fixed yaw between the two worlds; spread = drift)')

# hands: headset wrist->index direction vs pc mailbox wrist->index direction, after removing the constant yaw
# estimated from the heads (rotation about y that maps headset forward onto pc forward, median over time)
def yaw(v): return math.atan2(-v[0], -v[2])
yaws = []
for i in range(0, len(pt), 25):
    j = nearest(ht, pt[i] - lag); yaws.append(yaw(pfwd[i]) - yaw(hfwd[j]))
dy = float(np.median(np.unwrap(np.array(yaws))))
Ry = np.array([[math.cos(dy), 0, math.sin(dy)], [0, 1, 0], [-math.sin(dy), 0, math.cos(dy)]])
print(f'constant yaw between worlds (from heads): {math.degrees(dy):+.1f} deg')
for s, name in ((0, 'l'), (1, 'r')):
    hh = hands[s]
    if len(hh) < 50: continue
    tt = np.array([x[0] for x in hh]); dirs = np.stack([x[2] - x[1] for x in hh])
    errs = []; errs_raw = []
    for r in rows[::10]:
        if r[f'mb_{name}_flags'] != '1': continue
        t = float(r['t_ms']) / 1000; j = nearest(tt, t - lag)
        if abs(tt[j] - (t - lag)) > 0.1: continue
        pd = np.array([float(r[f'mb_{name}_ix']) - float(r[f'mb_{name}_wx']), float(r[f'mb_{name}_iy']) - float(r[f'mb_{name}_wy']), float(r[f'mb_{name}_iz']) - float(r[f'mb_{name}_wz'])])
        errs.append(ang_between(Ry @ dirs[j], pd)); errs_raw.append(ang_between(dirs[j], pd))
    errs = np.array(errs)
    if not len(errs): print(f'hand {name.upper()}: no time-matched samples'); continue
    print(f'hand {name.upper()}: angle(headset wrist->index rotated by the head yaw, pc mailbox wrist->index): median {np.median(errs):.1f} deg, p90 {np.percentile(errs, 90):.1f} (n={len(errs)}); without any yaw correction median {np.median(errs_raw):.1f}')
print('reading: hand error ~ head-forward spread => the alignment is as good as the head poses allow; hand error >> head spread => the relay rotates the hands wrongly')
