"""Which landmarks do the palm detector's 7 keypoints and box correspond to?

Collects detector outputs that land on confidently tracked hands and fits, in the hand's
own crop frame, (a) the nearest of the 21 landmarks for each palm keypoint, (b) the box
centre as palm-landmark centroid plus an offset along the wrist->middle axis, (c) the box
size as a multiple of the palm landmark extent. These define the training targets for a
fine-tuned detector so its outputs keep the convention the tracker decodes. Read-only.

usage: python fit_palm_convention.py <session-dir> <models-dir> [--sets N] [--out json]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fh_common as fh  # noqa: E402
from eval_palm import tiles_for  # noqa: E402

PALM_LM = (0, 1, 2, 5, 9, 13, 17)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session'); ap.add_argument('models')
    ap.add_argument('--sets', type=int, default=150)
    ap.add_argument('--out')
    a = ap.parse_args()
    session = Path(a.session)
    rec = fh.Recording(session / 'frame' / 'sets.bin')
    audit = fh.read_audit(session / 'frame' / 'live-views.txt')
    cams = fh.load_calibration(session / 'frame' / 'snapshot' / 'device' / 'persist')
    nets = fh.NcnnNets(a.models)
    tiles = {n: tiles_for(c, rotation_search=True) for n, c in cams.items()}
    stamps = rec.stamps()
    cand = [i for i, s in enumerate(stamps) if s in audit and any(v['presence'] >= 0.9 for v in audit[s]['views'])]
    rng = np.random.default_rng(2)
    pick = sorted(rng.choice(cand, size=min(a.sets, len(cand)), replace=False))
    nearest = []          # (7,) landmark index per detection
    rows = []             # per matched detection: features in the hand's upright frame
    for i in pick:
        images = rec.read(i)
        for v in audit[stamps[i]]['views']:
            if v['presence'] < 0.9 or v['camera'] not in images:
                continue
            img = images[v['camera']]
            pts = v['pts']
            hs = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
            # a few tiles whose square contains the palm, with the tracker's rotations
            use = [t for t in tiles[v['camera']] if np.all(np.abs(pts[9] - t[0]) < t[1] / 2)]
            rng.shuffle(use)
            for center, size, rot in use[:3]:
                palms, _, _ = nets.palms(img, center, size, rot, threshold=0.5)
                for p in palms:
                    if np.linalg.norm(p['center'] - pts[9]) > 0.5 * hs:
                        continue
                    d = np.linalg.norm(p['kp'][:, None, :] - pts[None, :, :], axis=2)   # (7,21)
                    nearest.append(d.argmin(1))
                    # hand frame: origin at palm-landmark centroid, up = wrist->middle, unit = palm extent
                    palm = pts[list(PALM_LM)]
                    cen = palm.mean(0)
                    up = pts[9] - pts[0]; up /= np.linalg.norm(up) + 1e-9
                    right = np.array([up[1], -up[0]])
                    ext = max(np.ptp(palm @ right), np.ptp(palm @ up))
                    off = p['center'] - cen
                    rows.append(dict(off_right=float(off @ right) / ext, off_up=float(off @ up) / ext,
                                     w=float(p['size'][0]) / ext, h=float(p['size'][1]) / ext,
                                     score=p['score'], tile_rot_rel=float(fh.wrap_angle(rot - math.atan2(up[0], -up[1]))),
                                     kp_off=((p['kp'] - cen) @ np.stack((right, up), 1) / ext).tolist()))
    nearest = np.array(nearest)
    modes = [int(np.bincount(nearest[:, k], minlength=21).argmax()) for k in range(7)]
    agree = [float(np.mean(nearest[:, k] == modes[k])) for k in range(7)]
    R = {k: np.array([r[k] for r in rows]) for k in ('off_right', 'off_up', 'w', 'h', 'score')}
    kp = np.array([r['kp_off'] for r in rows])   # (N,7,2) in hand frame / extent
    report = dict(
        session=session.name, matched_detections=len(rows),
        keypoint_nearest_landmark=modes, keypoint_agreement=agree,
        box_center_offset_over_extent=dict(right=dict(median=float(np.median(R['off_right'])), iqr=float(np.subtract(*np.percentile(R['off_right'], [75, 25])))),
                                           up=dict(median=float(np.median(R['off_up'])), iqr=float(np.subtract(*np.percentile(R['off_up'], [75, 25]))))),
        box_size_over_extent=dict(w=dict(median=float(np.median(R['w'])), iqr=float(np.subtract(*np.percentile(R['w'], [75, 25])))),
                                  h=dict(median=float(np.median(R['h'])), iqr=float(np.subtract(*np.percentile(R['h'], [75, 25]))))),
        keypoint_offsets_over_extent_median=np.median(kp, 0).tolist(),
        keypoint_offsets_iqr=np.subtract(*np.percentile(kp, [75, 25], axis=0)).tolist(),
        score=dict(p50=float(np.median(R['score'])), p10=float(np.percentile(R['score'], 10))),
        note='Hand frame: origin palm-landmark centroid, up = wrist->middle MCP, unit = palm landmark extent (landmarks 0,1,2,5,9,13,17).',
    )
    print(json.dumps(report, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
