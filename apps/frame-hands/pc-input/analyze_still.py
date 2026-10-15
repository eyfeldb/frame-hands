"""Still-hand drift check: per 5 s window, wrist position spread (RMS about the window mean, mm) for the headset
mailbox, the PC mailbox and the SteamVR output, next to head translation/rotation in that window.
usage: python analyze_still.py <capture.csv> <headset-mailbox.csv>"""
import csv, sys
import numpy as np


def rows(path):
    with open(path, newline='') as f:
        r = list(csv.DictReader(f))
    return {k: np.array([float(x[k]) for x in r]) for k in r[0]}


def spread(p):
    return float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean()) * 1000) if len(p) > 5 else float('nan')


c = rows(sys.argv[1]); h = rows(sys.argv[2])
t0 = c['t_ms'][0]; th0 = h['t_ms'][0]
hmd = np.stack([c['hmd_x'], c['hmd_y'], c['hmd_z']], 1); fwd = np.stack([c['hmd_fx'], c['hmd_fy'], c['hmd_fz']], 1)
print('window  | head move mm / head turn deg | headset-mailbox wrist spread L/R | PC-mailbox wrist spread L/R | SteamVR wrist spread L/R')
for w in range(8):
    a, b = w * 5000, (w + 1) * 5000
    m = (c['t_ms'] - t0 >= a) & (c['t_ms'] - t0 < b); mh = (h['t_ms'] - th0 >= a) & (h['t_ms'] - th0 < b)
    if m.sum() < 10:
        continue
    head_move = spread(hmd[m]); f = fwd[m]; turn = float(np.degrees(np.arccos(np.clip((f * f.mean(0)).sum(1) / (np.linalg.norm(f.mean(0)) + 1e-9), -1, 1))).max())
    out = [f'{a // 1000:2d}-{b // 1000:2d} s | {head_move:5.1f} / {turn:4.1f}']
    for src, prefix, flag in (('headset', 'h', None), ('pc', 'c', None), ('steamvr', 's', None)):
        vals = []
        for s, mb in (('l', 'l'), ('r', 'r')):
            if src == 'headset':
                on = mh & (h[f'{mb}_flags'] > 0); p = np.stack([h[f'{mb}_x'], h[f'{mb}_y'], h[f'{mb}_z']], 1)[on]
            elif src == 'pc':
                on = m & (c[f'mb_{mb}_flags'] > 0); p = np.stack([c[f'mb_{mb}_wx'], c[f'mb_{mb}_wy'], c[f'mb_{mb}_wz']], 1)[on]
            else:
                on = m & (c[f'{s}_valid'] > 0) & (c[f'mb_{mb}_flags'] > 0); p = np.stack([c[f'{s}_x'], c[f'{s}_y'], c[f'{s}_z']], 1)[on]
            vals.append(spread(p))
        out.append(f'{vals[0]:6.1f} / {vals[1]:6.1f}')
    print(' | '.join(out))
