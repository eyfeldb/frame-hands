"""Montage of recorded sets with all LIVE views drawn (id, side, presence, stereo/mono), both cameras.

usage: python review_sets.py <session-dir> <out.png> [--min-hands 3] [--n 8] [--seed 0]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fh_common as fh  # noqa: E402
from review_interaction import BONES, draw  # noqa: E402

COLORS = [(0, 255, 0), (0, 0, 255), (255, 200, 0), (255, 0, 255), (0, 255, 255)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session'); ap.add_argument('out'); ap.add_argument('--min-hands', type=int, default=0); ap.add_argument('--n', type=int, default=8); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    session = Path(a.session)
    rec = fh.Recording(session / 'frame' / 'sets.bin')
    audit = fh.read_audit(session / 'frame' / 'live-views.txt')
    stamps = rec.stamps()
    cand = [i for i, s in enumerate(stamps) if s in audit and len(audit[s]['hands']) >= a.min_hands and audit[s]['views']]
    rng = np.random.default_rng(a.seed)
    pick = sorted(rng.choice(cand, size=min(a.n, len(cand)), replace=False))
    rows = []
    for i in pick:
        images = rec.read(i); a_ = audit[stamps[i]]
        ids = sorted(a_['hands'])
        tiles = []
        for cam in ('slam_left', 'slam_right'):
            img = cv2.cvtColor(images[cam], cv2.COLOR_GRAY2BGR)
            for v in a_['views']:
                if v['camera'] != cam or v['presence'] < 0.5:
                    continue
                k = ids.index(v['id']) if v['id'] in ids else 4
                r, nv = a_['hands'].get(v['id'], (v['right'], 0))
                draw(img, v['pts'], COLORS[k % 5], 2 if nv >= 2 else 1)
                cv2.putText(img, f"id{v['id']} {'R' if r > 0.5 else 'L'} p{v['presence']:.2f} v{nv}", tuple(np.round(v['pts'][0]).astype(int) + np.array([5, 18])), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS[k % 5], 2)
            cv2.putText(img, f"set {i} t={(stamps[i] - stamps[0]) / 1e9:.1f}s {cam} hands={len(ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            tiles.append(cv2.resize(img, (660, 640)))
        rows.append(np.hstack(tiles))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, np.vstack(rows)); print('wrote', a.out, 'candidates', len(cand))


if __name__ == '__main__':
    main()
