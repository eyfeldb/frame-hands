"""Montage of specific recorded sets (by camera stamp) with LIVE audit landmarks and a zoomed crop of one hand side.

usage: python montage_at_stamps.py <session-dir> <out.png> --stamps <ns> [<ns> ...] [--side 0|1] [--offset N]
  --offset shifts each pick by N sets (e.g. -3 shows the frames just before a pinch began)
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fh_common as fh  # noqa: E402
from review_interaction import draw  # noqa: E402

COLORS = [(0, 255, 0), (0, 0, 255), (255, 200, 0), (255, 0, 255), (0, 255, 255)]


def zoom(img, pts, size=320):
    p = np.asarray(pts); lo = p.min(0); hi = p.max(0); c = (lo + hi) / 2; half = max(float((hi - lo).max()) * 0.75, 60)
    x0, y0 = int(c[0] - half), int(c[1] - half); x1, y1 = int(c[0] + half), int(c[1] + half)
    x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, img.shape[1]), min(y1, img.shape[0])
    crop = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    crop[y0c - y0:y1c - y0, x0c - x0:x1c - x0] = img[y0c:y1c, x0c:x1c]
    return cv2.resize(crop, (size, size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session'); ap.add_argument('out'); ap.add_argument('--stamps', type=int, nargs='*', default=[])
    ap.add_argument('--seconds', type=float, nargs='*', default=[], help='picks as seconds from the first recorded set (alternative to --stamps)')
    ap.add_argument('--side', type=int, default=0); ap.add_argument('--offset', type=int, default=0); ap.add_argument('--labels', nargs='*', default=[])
    a = ap.parse_args()
    session = Path(a.session)
    rec = fh.Recording(session / 'frame' / 'sets.bin')
    audit = fh.read_audit(session / 'frame' / 'live-views.txt')
    stamps = rec.stamps(); index = {s: i for i, s in enumerate(stamps)}
    picks = list(a.stamps) + [int(stamps[0] + t * 1e9) for t in a.seconds]
    if not a.labels:
        a.labels = [f'{t:.1f}s' for t in a.seconds]
    rows = []
    for k, st in enumerate(picks):
        if st not in index:
            # nearest
            i = int(np.argmin(np.abs(np.asarray(stamps) - st)))
        else:
            i = index[st]
        i = min(max(i + a.offset, 0), len(stamps) - 1)
        images = rec.read(i); a_ = audit.get(stamps[i], {'hands': {}, 'views': []})
        ids = sorted(a_['hands'])
        tiles = []
        for cam in ('slam_left', 'slam_right'):
            img = cv2.cvtColor(images[cam], cv2.COLOR_GRAY2BGR); crops = []
            for v in a_['views']:
                if v['camera'] != cam or v['presence'] < 0.5:
                    continue
                kk = ids.index(v['id']) if v['id'] in ids else 4
                r, nv = a_['hands'].get(v['id'], (v['right'], 0))
                draw(img, v['pts'], COLORS[kk % 5], 2)
                if (r > 0.5) == (a.side == 1):
                    crops.append(zoom(img, v['pts']))
            label = a.labels[k] if k < len(a.labels) else ''
            cv2.putText(img, f"set {i} t={(stamps[i] - stamps[0]) / 1e9:.1f}s {cam} {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            tile = cv2.resize(img, (560, 544))
            crop = crops[0] if crops else np.zeros((320, 320, 3), np.uint8)
            crop = cv2.resize(crop, (544, 544))
            tiles.append(np.hstack([tile, crop]))
        rows.append(np.hstack(tiles))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, np.vstack(rows)); print('wrote', a.out, len(rows), 'rows')


if __name__ == '__main__':
    main()
