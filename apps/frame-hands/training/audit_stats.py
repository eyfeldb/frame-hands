"""Quick statistics of a LIVE audit (live-views.txt): coverage, stereo share, where hands sat in each camera.

usage: python audit_stats.py <session-dir or live-views.txt>
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fh_common as fh  # noqa: E402


def main():
    p = Path(sys.argv[1]); p = p / 'frame' / 'live-views.txt' if p.is_dir() else p
    audit = fh.read_audit(p)
    stamps = sorted(audit)
    if not stamps:
        print('empty audit'); return
    dur = (stamps[-1] - stamps[0]) / 1e9
    hands_per_set = Counter(len(audit[s]['hands']) for s in stamps)
    nviews = Counter(); cams = Counter(); xs = {'slam_left': [], 'slam_right': []}; ys = {'slam_left': [], 'slam_right': []}
    for s in stamps:
        for hid, (right, nv) in audit[s]['hands'].items():
            nviews[nv] += 1
        for v in audit[s]['views']:
            if v['presence'] < 0.5 or v['id'] not in audit[s]['hands']:
                continue
            cams[v['camera']] += 1
            pts = np.asarray(v['pts']); xs[v['camera']].append(pts[0, 0]); ys[v['camera']].append(pts[0, 1])
    total = sum(nviews.values())
    print(f'{p.parent.parent.name}: {len(stamps)} sets, {dur:.0f} s, {len(stamps) / max(dur, 1e-6):.1f} sets/s')
    print('hands per set: ' + ', '.join(f'{k}: {100 * v / len(stamps):.0f}%' for k, v in sorted(hands_per_set.items())))
    print('hand updates by views: ' + ', '.join(f'{k} view(s): {100 * v / max(total, 1):.0f}%' for k, v in sorted(nviews.items())))
    for cam in ('slam_left', 'slam_right'):
        if xs[cam]:
            x = np.array(xs[cam]); y = np.array(ys[cam])
            print(f'{cam}: {cams[cam]} tracked views; wrist x 10/50/90% {np.percentile(x, 10):.0f}/{np.percentile(x, 50):.0f}/{np.percentile(x, 90):.0f} px, y {np.percentile(y, 10):.0f}/{np.percentile(y, 50):.0f}/{np.percentile(y, 90):.0f} px (image 1056x1024)')


if __name__ == '__main__':
    main()
