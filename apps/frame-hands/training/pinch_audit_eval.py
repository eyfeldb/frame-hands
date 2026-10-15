"""Pinch ground truth from a LIVE audit: triangulate thumb/index tips (both cameras) per hand and side,
print a compact gap timeline (min gap per half second) and simulate the dashboard pinch rules.

usage: python pinch_audit_eval.py <session-dir> [--side 0|1|both] [--from s] [--to s]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fh_common as fh  # noqa: E402
from pinch_trace_eval import simulate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session'); ap.add_argument('--side', default='both'); ap.add_argument('--from', dest='t0', type=float, default=0); ap.add_argument('--to', dest='t1', type=float, default=1e9)
    a = ap.parse_args()
    session = Path(a.session)
    cams = fh.load_calibration(session / 'frame' / 'snapshot' / 'device' / 'persist')
    audit = fh.read_audit(session / 'frame' / 'live-views.txt')
    stamps = sorted(audit); start = stamps[0]
    rows = {0: [], 1: []}
    for s in stamps:
        t = (s - start) / 1e9
        if t < a.t0 or t > a.t1:
            continue
        rec = audit[s]
        for hid, (right, nv) in rec['hands'].items():
            views = [(cams[v['camera']], v['pts'], v['presence']) for v in rec['views'] if v['id'] == hid and v['presence'] >= 0.5 and v['camera'] in cams]
            if len({c.name if hasattr(c, 'name') else id(c) for c, _, _ in views}) < 2:
                continue
            pts, res = fh.triangulate_hand(cams, views)
            if res > 0.02:
                continue
            gap = float(np.linalg.norm(pts[4] - pts[8])); width = float(np.linalg.norm(pts[5] - pts[17]))
            rows[1 if right > 0.5 else 0].append((s, hid, gap, width, 0.0))
    sides = [0, 1] if a.side == 'both' else [int(a.side)]
    for side in sides:
        r = sorted(rows[side])
        if not r:
            print(f'side {side}: no stereo hands'); continue
        gaps = np.array([x[2] for x in r]) * 1000; ts = np.array([(x[0] - start) / 1e9 for x in r]); widths = np.array([x[3] for x in r]) * 1000
        print(f'== side {side} ({"left" if side == 0 else "right"}): {len(r)} stereo updates, width median {np.median(widths):.0f} mm, gap median {np.median(gaps):.0f} mm')
        # timeline: min gap per 0.5 s bin, as a row of numbers (mm); '.' = no stereo hand
        edges = np.arange(np.floor(ts.min()), ts.max() + 0.5, 0.5)
        line = []
        for b0 in edges:
            m = (ts >= b0) & (ts < b0 + 0.5)
            line.append(f'{gaps[m].min():3.0f}' if m.any() else '  .')
        for i in range(0, len(line), 40):
            print(f'  t={edges[i]:6.1f}s ' + ' '.join(line[i:i + 40]))
        rules = {
            'old (0.30w 16-30, 50 ms)': (0.30, .016, .030, 0.52, .030, .050, 50),
            'installed E (0.20w 11-16, 120 ms keep)': (0.20, .011, .016, 0.50, .030, .045, 120, True),
            'F (0.18w 10-14, 120 ms keep)': (0.18, .010, .014, 0.50, .030, .045, 120, True),
            'G (0.16w 9-13, 150 ms keep)': (0.16, .009, .013, 0.50, .030, .045, 150, True),
        }
        for name, p in rules.items():
            ep = simulate(r, *p)
            print(f'  {name}: {len(ep)} pinches: ' + ' '.join(f'{(e[3] - start) / 1e9:.1f}s/{e[1]:.2f}s/{e[2] * 1000:.0f}mm' for e in ep))


if __name__ == '__main__':
    main()
