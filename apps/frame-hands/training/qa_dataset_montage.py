"""Render dataset tiles with their palm box/keypoint labels to a PNG montage for visual QA.

usage: python qa_dataset_montage.py <shard-dir> <out.png> [--n 24] [--kind pos|hard|neg|any] [--seed 0]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fh_common as fh  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('shard_dir'); ap.add_argument('out')
    ap.add_argument('--n', type=int, default=24); ap.add_argument('--kind', default='any'); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    files = sorted(Path(a.shard_dir).glob('shard-*.npz'))
    z = np.load(files[0])
    images, boxes, meta = z['images'], z['boxes'], json.loads(str(z['meta']))
    idx = [i for i, m in enumerate(meta) if a.kind == 'any' or m['kind'] == a.kind]
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(idx, size=min(a.n, len(idx)), replace=False)
    cols = 6
    rows = int(np.ceil(len(pick) / cols))
    canvas = np.zeros((rows * fh.PALM_SIZE, cols * fh.PALM_SIZE, 3), np.uint8)
    for k, i in enumerate(pick):
        tile = cv2.cvtColor(fh.clahe_4x4(np.ascontiguousarray(images[i])), cv2.COLOR_GRAY2BGR)
        for b in boxes[i]:
            if b[18] <= 0:
                continue
            cx, cy, s = b[0], b[1], b[2]
            color = (0, 255, 0) if b[17] >= 1 else (0, 200, 255)
            cv2.rectangle(tile, (int(cx - s / 2), int(cy - s / 2)), (int(cx + s / 2), int(cy + s / 2)), color, 1)
            kp = b[3:17].reshape(7, 2)
            for j, (x, y) in enumerate(kp):
                cv2.circle(tile, (int(x), int(y)), 2, (255, 0, 0) if j == 0 else (255, 255, 0) if j == 2 else (0, 0, 255), -1)
            cv2.line(tile, tuple(kp[0].astype(int)), tuple(kp[2].astype(int)), (255, 255, 255), 1)
        cv2.putText(tile, meta[i]['kind'], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        r, c = divmod(k, cols)
        canvas[r * fh.PALM_SIZE:(r + 1) * fh.PALM_SIZE, c * fh.PALM_SIZE:(c + 1) * fh.PALM_SIZE] = tile
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, canvas)
    kinds = {}
    for m in meta:
        kinds[m['kind']] = kinds.get(m['kind'], 0) + 1
    print('wrote', a.out, 'shard samples', len(meta), 'kinds', kinds)


if __name__ == '__main__':
    main()
