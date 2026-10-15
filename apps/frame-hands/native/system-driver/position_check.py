"""Hand POSITION check across the headset -> PC chain for a maestro_capture session.
At time-matched instants (clocks aligned by head motion), compares the wrist position relative to the head on the
headset (pose.json head pose + mailbox points) with the wrist position relative to the HMD on the PC (capture.csv:
HMD position/forward + PC mailbox wrist): distance head->wrist, height below the head, forward and lateral offsets,
and the left-right wrist spacing. A rigid world alignment preserves all of these; differences are placement error.
usage: python position_check.py <session-dir> [clock_offset_s]
"""
import csv, json, math, re, sys
from pathlib import Path
import numpy as np

d = Path(sys.argv[1])
heads = []; hands = {0: [], 1: []}
for f in sorted(d.glob('scn-*.jsonl')):
    m = re.search(r'scn-(\d{2})(\d{2})(\d{2})', f.name); base = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) if m else 0
    for line in open(f, errors='replace'):
        try: r = json.loads(line)
        except ValueError: continue
        if r.get('kind') == 'raw' and r.get('context_valid') and r.get('head_world_m'):
            H = np.array(r['head_world_m'], float)[:3]; heads.append((base + r['probe_t'], H[:, 3], -H[:, 2], H[:, 0]))   # pos, forward, right
        elif r.get('kind') == 'shown':
            for h in r.get('hands', []): hands[h['side']].append((base + r['probe_t'], np.array(h['points'][0])))
ht = np.array([h[0] for h in heads])
rows = [r for r in csv.DictReader(open(d / 'capture.csv', newline='')) if None not in r.values() and all(v != '' for v in r.values())]
import os, datetime as dt
pt = np.array([float(r['t_ms']) for r in rows]) / 1000
# capture.csv t_ms is a QPC counter; anchor it to wall seconds-of-day with the file's end time (last row written at mtime)
end = dt.datetime.fromtimestamp(os.path.getmtime(d / 'capture.csv')); end_sod = end.hour * 3600 + end.minute * 60 + end.second + end.microsecond / 1e6
pt = end_sod - (pt[-1] - pt)
ppos = np.array([[float(r['hmd_x']), float(r['hmd_y']), float(r['hmd_z'])] for r in rows]); pfwd = np.array([[float(r['hmd_fx']), float(r['hmd_fy']), float(r['hmd_fz'])] for r in rows])

# clock offset: reuse the head-motion cross-correlation of alignment_check (vertical head position works too)
def series(t, p): return t, p[:, 1]
if len(sys.argv) > 2: lag = float(sys.argv[2])
else:
    hp = np.array([h[1] for h in heads]); grid = np.arange(0, min(ht[-1] - ht[0], pt[-1] - pt[0]), 0.02)
    a = np.interp(grid + ht[0], ht, hp[:, 1]); a -= a.mean(); best = (0, -1)
    for lag in np.arange(-60, 60.01, 0.02):
        b = np.interp(grid + pt[0] + lag, pt, ppos[:, 1]); b -= b.mean(); c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        if c > best[1]: best = (lag, c)
    lag, corr = best; print(f'clock offset pc = headset + {lag:+.2f} s (head height correlation {corr:.2f})')

def nearest(t_arr, t): i = np.searchsorted(t_arr, t); return min(max(i, 0), len(t_arr) - 1)
def horiz(v): return np.array([v[0], 0, v[2]]) / (np.linalg.norm([v[0], v[2]]) + 1e-9)
def decompose(wrist, head_pos, head_fwd):
    rel = wrist - head_pos; f = horiz(head_fwd); right = np.cross(f, [0, 1, 0])
    return dict(dist=np.linalg.norm(rel), height=rel[1], forward=float(np.dot(rel, f)), lateral=float(np.dot(rel, right)))

for s, name in ((0, 'l'), (1, 'r')):
    tt = np.array([x[0] for x in hands[s]]); ww = np.array([x[1] for x in hands[s]]) if hands[s] else np.zeros((0, 3))
    diffs = {k: [] for k in ('dist', 'height', 'forward', 'lateral')}; n = 0
    for i in range(0, len(rows), 5):
        r = rows[i]
        if r[f'mb_{name}_flags'] != '1': continue
        t = pt[i] - lag; j = nearest(tt, t)
        if not len(tt) or abs(tt[j] - t) > 0.05: continue
        k = nearest(ht, t)
        hs = decompose(ww[j], heads[k][1], heads[k][2])
        ps = decompose(np.array([float(r[f'mb_{name}_wx']), float(r[f'mb_{name}_wy']), float(r[f'mb_{name}_wz'])]), ppos[i], pfwd[i])
        for key in diffs: diffs[key].append(ps[key] - hs[key])
        n += 1
    if n < 20: print(f'{name.upper()}: too few matched samples ({n})'); continue
    print(f'{name.upper()} wrist, PC minus headset, relative to the head ({n} samples): ' + ', '.join(f"{k} {np.median(v) * 100:+.1f} cm (p10 {np.percentile(v, 10) * 100:+.1f}, p90 {np.percentile(v, 90) * 100:+.1f})" for k, v in diffs.items()))

# left-right spacing
if hands[0] and hands[1]:
    tl = np.array([x[0] for x in hands[0]]); wl = np.array([x[1] for x in hands[0]]); tr = np.array([x[0] for x in hands[1]]); wr = np.array([x[1] for x in hands[1]])
    sp_h = []; sp_p = []
    for i in range(0, len(rows), 5):
        r = rows[i]
        if r['mb_l_flags'] != '1' or r['mb_r_flags'] != '1': continue
        t = pt[i] - lag; j = nearest(tl, t); k = nearest(tr, t)
        if abs(tl[j] - t) > 0.05 or abs(tr[k] - t) > 0.05: continue
        sp_h.append(np.linalg.norm(wl[j] - wr[k]))
        sp_p.append(math.dist([float(r['mb_l_wx']), float(r['mb_l_wy']), float(r['mb_l_wz'])], [float(r['mb_r_wx']), float(r['mb_r_wy']), float(r['mb_r_wz'])]))
    if sp_h: print(f'left-right wrist spacing: headset median {np.median(sp_h) * 100:.1f} cm, PC median {np.median(sp_p) * 100:.1f} cm, PC minus headset median {np.median(np.subtract(sp_p, sp_h)) * 100:+.1f} cm ({len(sp_h)} samples)')
