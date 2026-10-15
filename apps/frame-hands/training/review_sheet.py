"""Compact contact sheet of a recording for review: one tile per `--every` seconds, right camera (or both),
downscaled, with the time and the number of live-audit hands stamped on it. For deciding which windows of a
recording are hands-only before building a dataset (`build_hist_dataset.py --keep`).
usage: python review_sheet.py <session-dir> <out.png> [--every 4] [--cols 6] [--width 300] [--both]
"""
import argparse, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import fh_common as fh  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('session'); ap.add_argument('out')
ap.add_argument('--every', type=float, default=4); ap.add_argument('--cols', type=int, default=6); ap.add_argument('--width', type=int, default=300); ap.add_argument('--both', action='store_true')
a = ap.parse_args()
session = Path(a.session)
rec = fh.Recording(session / 'frame' / 'sets.bin')
audit = fh.read_audit(session / 'frame' / 'live-views.txt') if (session / 'frame' / 'live-views.txt').exists() else {}
stamps = rec.stamps(); t0 = stamps[0]
picks = []; next_t = 0.0
for i, s in enumerate(stamps):
    t = (s - t0) / 1e9
    if t >= next_t: picks.append((i, t)); next_t += a.every
tiles = []
for i, t in picks:
    imgs = rec.read(i)
    cams = ['slam_right', 'slam_left'] if a.both else ['slam_right']
    parts = []
    for cam in cams:
        img = imgs.get(cam)
        if img is None: continue
        h, w = img.shape[:2]; scale = a.width / w
        small = cv2.resize(img, (a.width, int(h * scale)))
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR) if small.ndim == 2 else small
        au = audit.get(stamps[i]); n = len(au['hands']) if au else 0
        if au:
            for v in au['views']:
                if v['camera'] != cam: continue
                pts = (np.asarray(v['pts']) * scale).astype(int)
                for p in pts: cv2.circle(small, (int(p[0]), int(p[1])), 1, (0, 255, 0), -1)
        cv2.putText(small, f'{t:5.1f}s h{n}', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        parts.append(small)
    if parts: tiles.append(np.hstack(parts))
if not tiles: raise SystemExit('no frames')
th, tw = tiles[0].shape[:2]
rows = []
for r in range(0, len(tiles), a.cols):
    row = tiles[r:r + a.cols]
    while len(row) < a.cols: row.append(np.zeros((th, tw, 3), np.uint8))
    rows.append(np.hstack(row))
sheet = np.vstack(rows)
cv2.imwrite(a.out, sheet)
print(f'wrote {a.out}: {len(tiles)} tiles, every {a.every} s, {sheet.shape[1]}x{sheet.shape[0]}')
