"""How often does the palm detector see a hand the tracker was already following?

Uses the tracker's own search-tile layout (tracker.cpp add_tiles: 3x3 at 0.45, 2x2 at 0.65,
1x1 at 1.0 of the image, rotated so shoulder->hand points up). For sampled processed
sets, every tile in each camera is run; a tracked view counts as recalled if any tile's
palm lands within half a hand size of its palm centre (landmark 9). Tiles far from all
tracked hands give a false-positive rate. Optional --rotation-search adds the +-45/90/180
variants the live config uses. Read-only.

usage: python eval_palm.py <session-dir> <models-dir> [--sets N] [--threshold 0.2] [--rotation-search] [--out report.json]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fh_common as fh  # noqa: E402

SHOULDERS = np.array([[0.17, -0.25, 0.08], [-0.17, -0.25, 0.08]])


def tiles_for(cam, rotation_search=False):
    out = []
    for frac, gx, gy in ((0.45, 3, 3), (0.65, 2, 2), (1.0, 1, 1)):
        s = frac * max(cam.width, cam.height)
        for i in range(gx):
            for j in range(gy):
                x = s / 2 + (cam.width - s) * i / (gx - 1) if gx > 1 else cam.width / 2.0
                y = s / 2 + (cam.height - s) * j / (gy - 1) if gy > 1 else cam.height / 2.0
                ray = cam.ray([[x, y]])[0]
                p = cam.origin + ray * 0.45
                d = p - SHOULDERS[0 if p[0] > 0 else 1]
                d /= np.linalg.norm(d)
                a, _ = cam.project([p]); b, _ = cam.project([p + d * 0.05])
                a, b = a[0], b[0]
                rot = math.atan2(b[0] - a[0], -(b[1] - a[1]))
                out.append((np.array([x, y]), s, rot))
    if rotation_search:
        base = list(out)
        for c, s, r in base:
            for ang in (-math.pi / 2, -math.pi / 4, math.pi / 4, math.pi / 2, math.pi):
                out.append((c, s, r + ang))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session'); ap.add_argument('models')
    ap.add_argument('--sets', type=int, default=120)
    ap.add_argument('--threshold', type=float, default=0.2)
    ap.add_argument('--rotation-search', action='store_true')
    ap.add_argument('--out')
    a = ap.parse_args()
    session = Path(a.session)
    rec = fh.Recording(session / 'frame' / 'sets.bin')
    audit = fh.read_audit(session / 'frame' / 'live-views.txt')
    cams = fh.load_calibration(session / 'frame' / 'snapshot' / 'device' / 'persist')
    nets = fh.NcnnNets(a.models)
    tiles = {n: tiles_for(c, a.rotation_search) for n, c in cams.items()}
    stamps = rec.stamps()
    candidates = [i for i, s in enumerate(stamps) if s in audit and any(v['presence'] >= 0.5 for v in audit[s]['views'])]
    rng = np.random.default_rng(1)
    pick = sorted(rng.choice(candidates, size=min(a.sets, len(candidates)), replace=False))
    recalled = 0; total = 0; best_scores = []; fp_tiles = 0; clean_tiles = 0; palm_ms = []
    per_cam = {n: [0, 0] for n in cams}
    hand_tile_hits = []   # how many tiles found each hand
    for i in pick:
        images = rec.read(i)
        views = [v for v in audit[stamps[i]]['views'] if v['presence'] >= 0.5]
        for cam_name, cam in cams.items():
            if cam_name not in images:
                continue
            img = images[cam_name]
            cam_views = [v for v in views if v['camera'] == cam_name]
            found_per_view = [0] * len(cam_views)
            best = [0.0] * len(cam_views)
            for center, size, rot in tiles[cam_name]:
                t0 = time.perf_counter()
                palms, _, _ = nets.palms(img, center, size, rot, threshold=a.threshold)
                palm_ms.append((time.perf_counter() - t0) * 1000)
                tile_has_hand = False
                for p in palms:
                    matched = False
                    for k, v in enumerate(cam_views):
                        hs = max(np.ptp(v['pts'][:, 0]), np.ptp(v['pts'][:, 1]))
                        if np.linalg.norm(p['center'] - v['pts'][9]) < 0.5 * max(hs, 40):
                            found_per_view[k] += 1; best[k] = max(best[k], p['score']); matched = True
                    tile_has_hand = tile_has_hand or matched
                    if not matched:
                        # a detection not on any tracked hand: count once per tile as a false positive
                        pass
                # tile covers a tracked hand if that hand's palm is inside the tile square (rotation ignored)
                covers = any(np.all(np.abs(v['pts'][9] - center) < size / 2) for v in cam_views)
                if not covers:
                    clean_tiles += 1
                    if palms:
                        fp_tiles += 1
            for k, v in enumerate(cam_views):
                total += 1; per_cam[cam_name][1] += 1
                hand_tile_hits.append(found_per_view[k])
                if found_per_view[k]:
                    recalled += 1; per_cam[cam_name][0] += 1
                best_scores.append(best[k])
    best_scores = np.array(best_scores)
    report = dict(
        session=session.name, sets=len(pick), threshold=a.threshold, rotation_search=a.rotation_search,
        tiles_per_camera=len(next(iter(tiles.values()))),
        tracked_views=total, any_tile_recall=recalled / max(1, total),
        per_camera_recall={n: (h / max(1, t)) for n, (h, t) in per_cam.items()},
        tiles_finding_each_hand=dict(zero=int(np.sum(np.array(hand_tile_hits) == 0)), one=int(np.sum(np.array(hand_tile_hits) == 1)),
                                     two_plus=int(np.sum(np.array(hand_tile_hits) >= 2))),
        best_score_on_recalled=dict(p50=float(np.median(best_scores[best_scores > 0])) if (best_scores > 0).any() else None,
                                    p10=float(np.percentile(best_scores[best_scores > 0], 10)) if (best_scores > 0).any() else None),
        false_positive_rate_on_hand_free_tiles=fp_tiles / max(1, clean_tiles), hand_free_tiles=clean_tiles,
        palm_ms=dict(p50=float(np.median(palm_ms)), p95=float(np.percentile(palm_ms, 95))),
        note='Recall is against hands the compact tracker already held (LIVE audit), i.e. the easy cases. Hands it never acquired are not counted.',
    )
    print(json.dumps(report, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
