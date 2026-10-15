"""Montage of history-dataset crops: label landmarks (green), history landmarks (cyan), base model prediction (red).

usage: python qa_hist_montage.py <shard-dir> <out.png> [--kind tracked|hard|neg|any] [--n 24] [--seed 0]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fh_common as fh  # noqa: E402

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]


def draw(img, pts, color, thick=1):
    for a, b in BONES:
        cv2.line(img, tuple(np.round(pts[a]).astype(int)), tuple(np.round(pts[b]).astype(int)), color, thick)
    for p in pts:
        cv2.circle(img, tuple(np.round(p).astype(int)), 2, color, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('shard_dir'); ap.add_argument('out'); ap.add_argument('--kind', default='any'); ap.add_argument('--n', type=int, default=24); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    files = sorted(Path(a.shard_dir).glob('shard-*.npz'))
    z = np.load(files[int(np.random.default_rng(a.seed).integers(len(files)))])
    kinds = z['kind']
    idx = [i for i in range(len(kinds)) if a.kind == 'any' or kinds[i] == a.kind]
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(idx, size=min(a.n, len(idx)), replace=False)
    cols = 6; rows = int(np.ceil(len(pick) / cols)); n = fh.HAND_SIZE
    canvas = np.zeros((rows * n, cols * n, 3), np.uint8)
    for k, i in enumerate(pick):
        tile = cv2.cvtColor(z['images'][i], cv2.COLOR_GRAY2BGR)
        if z['hkind'][i] > 0:
            draw(tile, z['hist'][i], (255, 255, 0), 1)
        if z['lw'][i] > 0:
            draw(tile, z['label'][i], (0, 255, 0), 1)
        draw(tile, z['ptsbase'][i], (0, 0, 255), 1)
        cv2.putText(tile, f"{kinds[i]} lw{z['lw'][i]:.1f} h{int(z['hkind'][i])} pb{z['pbase'][i]:.2f}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        r, c = divmod(k, cols)
        canvas[r * n:(r + 1) * n, c * n:(c + 1) * n] = tile
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, canvas)
    print('wrote', a.out, 'kinds in shard', {k: int((kinds == k).sum()) for k in np.unique(kinds)})


if __name__ == '__main__':
    main()
