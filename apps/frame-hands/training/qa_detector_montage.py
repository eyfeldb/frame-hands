"""Side-by-side palm detections, baseline vs candidate ncnn models, on recorded tiles.

Each row: the same tile (CLAHE, as the detector sees it) with baseline detections (left) and
candidate detections (right); boxes labelled with scores, tracked-hand palms marked with a
green cross. Mix of tiles containing tracked hands and hand-free tiles. Read-only.

usage: python qa_detector_montage.py <session-dir> <baseline-models> <candidate-models> <out.png> [--n 10] [--threshold 0.2] [--seed 0]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fh_common as fh  # noqa: E402
from eval_palm import tiles_for  # noqa: E402


def draw(patch, palms, pts_list, tm, label):
    img = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    for pts in pts_list:
        c = fh.to_crop(tm, pts[9])[0]
        cv2.drawMarker(img, (int(c[0]), int(c[1])), (0, 255, 0), cv2.MARKER_CROSS, 14, 1)
    for p in palms:
        c, s = p['crop_center'], p['crop_size']
        col = (0, 165, 255) if p['score'] < 0.5 else (0, 0, 255)
        cv2.rectangle(img, (int(c[0] - s[0] / 2), int(c[1] - s[1] / 2)), (int(c[0] + s[0] / 2), int(c[1] + s[1] / 2)), col, 1)
        cv2.putText(img, f"{p['score']:.2f}", (int(c[0] - s[0] / 2), max(10, int(c[1] - s[1] / 2) - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    cv2.putText(img, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session'); ap.add_argument('baseline'); ap.add_argument('candidate'); ap.add_argument('out')
    ap.add_argument('--n', type=int, default=10); ap.add_argument('--threshold', type=float, default=0.2); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    session = Path(a.session)
    rec = fh.Recording(session / 'frame' / 'sets.bin')
    audit = fh.read_audit(session / 'frame' / 'live-views.txt')
    cams = fh.load_calibration(session / 'frame' / 'snapshot' / 'device' / 'persist')
    tiles = {n: tiles_for(c) for n, c in cams.items()}
    base, cand = fh.NcnnNets(a.baseline), fh.NcnnNets(a.candidate)
    stamps = rec.stamps()
    rng = np.random.default_rng(a.seed)
    rows = []
    sets = [i for i, s in enumerate(stamps) if s in audit]
    while len(rows) < a.n:
        i = int(rng.choice(sets))
        images = rec.read(i)
        cam_name = str(rng.choice(list(cams)))
        img = images[cam_name]
        views = [v for v in audit[stamps[i]]['views'] if v['camera'] == cam_name and v['presence'] >= 0.5]
        want_hand = len(rows) % 3 != 2
        cands = [t for t in tiles[cam_name] if (any(np.all(np.abs(v['pts'][9] - t[0]) < t[1] / 2 * 0.9) for v in views)) == want_hand]
        if not cands:
            continue
        center, size, rot = cands[rng.integers(len(cands))]
        pb, tm, patch = base.palms(img, center, size, rot, threshold=a.threshold)
        pc, _, _ = cand.palms(img, center, size, rot, threshold=a.threshold)
        pts = [v['pts'] for v in views]
        rows.append(np.hstack((draw(patch, pb, pts, tm, f'base t={(stamps[i]-stamps[0])/1e9:.1f}s'), draw(patch, pc, pts, tm, 'candidate'))))
    canvas = np.vstack(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, canvas)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
