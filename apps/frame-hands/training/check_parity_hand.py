"""Can the PC reproduce the device's LIVE landmark outputs from the raw recording?

For tracked views, the device cropped frame t with roi_from_points(landmarks at t-1).
We redo exactly that with the deployed ncnn models on the PC and compare to the
recorded landmarks at t. Also checks the palm detector finds a tracked hand on a
tile around it. Read-only.

usage: python check_parity_hand.py <session-dir> <models-dir> [--samples N] [--out report.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fh_common as fh  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('models')
    ap.add_argument('--samples', type=int, default=60)
    ap.add_argument('--out')
    ap.add_argument('--fp32', action='store_true')
    a = ap.parse_args()
    session = Path(a.session)
    rec = fh.Recording(session / 'frame' / 'sets.bin')
    audit = fh.read_audit(session / 'frame' / 'live-views.txt')
    nets = fh.NcnnNets(a.models, fp16=not a.fp32)
    stamps = rec.stamps()
    # consecutive processed sets where the same (camera,id) view exists in both
    pairs = []
    for i in range(1, len(stamps)):
        prev, cur = audit.get(stamps[i - 1]), audit.get(stamps[i])
        if not prev or not cur:
            continue
        for v in cur['views']:
            for pv in prev['views']:
                if pv['camera'] == v['camera'] and pv['id'] == v['id'] and pv['presence'] >= 0.5:
                    pairs.append((i, pv, v))
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(pairs), size=min(a.samples, len(pairs)), replace=False))
    errs, pres_d, right_d, times = [], [], [], []
    palm_hits = 0; palm_tries = 0
    for k in pick:
        i, pv, v = pairs[k]
        img = rec.read(i)[v['camera']]
        center, size, rot = fh.roi_from_points(pv['pts'])
        t0 = time.perf_counter()
        r = nets.landmarks(img, center, size, rot)
        times.append((time.perf_counter() - t0) * 1000)
        e = np.linalg.norm(r['pts'] - v['pts'], axis=1)
        errs.append(e)
        pres_d.append(r['presence'] - v['presence'])
        right_d.append(r['right'] - v['right'])
        # palm detector on a tile centred on this hand (size 0.45 of the image like the first tile ring)
        palm_tries += 1
        palms, _, _ = nets.palms(img, center, 0.45 * max(img.shape), rot, threshold=0.2)
        if any(np.linalg.norm(p['center'] - v['pts'][9]) < 0.5 * size for p in palms):
            palm_hits += 1
    errs = np.array(errs)
    report = dict(
        session=session.name, models=str(a.models), fp16=not a.fp32, pairs_available=len(pairs), sampled=len(pick),
        landmark_px_error=dict(mean=float(errs.mean()), p50=float(np.median(errs)), p95=float(np.percentile(errs, 95)),
                               max=float(errs.max()), per_sample_mean_p95=float(np.percentile(errs.mean(1), 95))),
        presence_delta=dict(mean_abs=float(np.mean(np.abs(pres_d))), max_abs=float(np.max(np.abs(pres_d)))),
        right_delta=dict(mean_abs=float(np.mean(np.abs(right_d))), max_abs=float(np.max(np.abs(right_d)))),
        hand_model_ms=dict(p50=float(np.median(times)), p95=float(np.percentile(times, 95))),
        palm_tile_recall_on_tracked_hand=palm_hits / max(1, palm_tries),
        note='Differences come from x64 vs ARM FP16 arithmetic and the warp implementation; LIVE labels are the device outputs, not ground truth.',
    )
    print(json.dumps(report, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
