"""Two-camera disagreement of the live landmark model per landmark group and time window.

For every set where a hand is seen by both forward cameras, triangulate each of the 21 landmarks and report the
ray-distance residual (mm) per group: palm (0,5,9,13,17), thumb (1-4), index (5-8), middle (9-12), ring (13-16),
pinky (17-20). A group whose residual rises in some windows is where the 2D model is inconsistent between views
(e.g. a vertical thumbs-up); if the residuals stay low while the rig RMSE rises, the rig fit is the problem.
usage: python stereo_consistency.py <session-dir> [--window 8] [--min-presence 0.8]
"""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import fh_common as fh  # noqa: E402

GROUPS = {'palm': [0, 5, 9, 13, 17], 'thumb': [1, 2, 3, 4], 'index': [5, 6, 7, 8], 'middle': [9, 10, 11, 12], 'ring': [13, 14, 15, 16], 'pinky': [17, 18, 19, 20]}
ap = argparse.ArgumentParser(); ap.add_argument('session'); ap.add_argument('--window', type=float, default=8); ap.add_argument('--min-presence', type=float, default=0.8)
a = ap.parse_args()
session = Path(a.session)
cams = fh.load_calibration(session / 'frame' / 'snapshot' / 'device' / 'persist')
audit = fh.read_audit(session / 'frame' / 'live-views.txt')
stamps = sorted(audit); t0 = stamps[0]
rows = []   # (t, side, residuals(21))
for s in stamps:
    a_ = audit[s]; by_id = {}
    for v in a_['views']:
        if v['presence'] >= a.min_presence and v['camera'] in cams: by_id.setdefault(v['id'], []).append(v)
    for hid, vs in by_id.items():
        if len(vs) < 2 or hid not in a_['hands']: continue
        rays = [cams[v['camera']].ray(v['pts']) for v in vs[:2]]
        origins = [cams[v['camera']].origin for v in vs[:2]]
        res = np.zeros(21)
        for k in range(21):
            _, res[k] = fh.triangulate(origins, [r[k] for r in rays], [1.0, 1.0])
        rows.append(((s - t0) / 1e9, 1 if a_['hands'][hid][0] > 0.5 else 0, res * 1000))
w = a.window; end = rows[-1][0] if rows else 0
print('window | side | hands |  ' + '  '.join(f'{g:>6s}' for g in GROUPS) + '   (median ray residual mm)')
for i in range(int(end // w) + 1):
    for side in (0, 1):
        win = [r for r in rows if i * w <= r[0] < (i + 1) * w and r[1] == side]
        if len(win) < 10: continue
        R = np.array([r[2] for r in win])
        print(f'{i * w:5.0f}s |  {side}   | {len(win):5d} |  ' + '  '.join(f'{np.median(R[:, idx]):6.1f}' for idx in GROUPS.values()))
