"""Build a Frame-domain palm-detector dataset from a recorded session.

Labels come from the device's own stereo-confirmed hands (LIVE audit): a view with presence
>= 0.8 whose hand was triangulated from both cameras. Hard positives are views missing for
1..4 frames between two confirmed sightings of the same hand in the same camera, with
landmarks linearly interpolated (weight 0.5). Clean negatives are tiles containing no
labelled hand, taken only from sets where exactly two opposite-side stereo hands were
tracked, so an unlabelled hand cannot be hiding in them.

Tiles follow the tracker's 84-tile live layout (with rotation search) plus random jitter.
Raw (pre-CLAHE) 192x192 uint8 tiles are stored; CLAHE and photometric augmentation happen at
training time. Output: npz shards + index.json under --out (private, ignored). Read-only on inputs.

usage: python build_palm_dataset.py <session-dir> --out <dir> [--tiles-per-hand 3] [--negatives-per-set 1] [--every 1]
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
from eval_palm import tiles_for  # noqa: E402

SHARD = 2000
MAX_BOXES = 4


def hand_labels(audit, stamps, cams):
    """Per set index -> list of (camera, pts21, weight). Includes interpolated hard positives."""
    per_set = {i: [] for i in range(len(stamps))}
    # confirmed views
    confirmed = {}   # (cam, id) -> {set_index: pts}
    for i, s in enumerate(stamps):
        a = audit.get(s)
        if not a:
            continue
        for v in a['views']:
            h = a['hands'].get(v['id'])
            # audit 'hand' row: (right score, number of camera views); >= 2 means triangulated from both cameras
            if v['presence'] >= 0.8 and h is not None and h[1] >= 2 and v['camera'] in cams:
                per_set[i].append((v['camera'], v['pts'], 1.0))
                confirmed.setdefault((v['camera'], v['id']), {})[i] = v['pts']
    # interpolate short gaps
    hard = 0
    for key, d in confirmed.items():
        idx = sorted(d)
        for a_i, b_i in zip(idx, idx[1:]):
            gap = b_i - a_i - 1
            if 1 <= gap <= 4:
                for j in range(a_i + 1, b_i):
                    t = (j - a_i) / (b_i - a_i)
                    pts = d[a_i] * (1 - t) + d[b_i] * t
                    per_set[j].append((key[0], pts, 0.5)); hard += 1
    # sets where exactly two opposite-side stereo hands were tracked -> safe for negatives
    clean = set()
    for i, s in enumerate(stamps):
        a = audit.get(s)
        if not a:
            continue
        st = [(r, nv) for r, nv in a['hands'].values() if nv >= 2]
        if len(st) == 2 and len(a['hands']) == 2 and (st[0][0] > 0.5) != (st[1][0] > 0.5):
            clean.add(i)
    return per_set, clean, hard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session'); ap.add_argument('--out', required=True)
    ap.add_argument('--tiles-per-hand', type=int, default=3)
    ap.add_argument('--negatives-per-set', type=int, default=1)
    ap.add_argument('--every', type=int, default=1, help='use every Nth set')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    session = Path(a.session); out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = fh.Recording(session / 'frame' / 'sets.bin')
    audit = fh.read_audit(session / 'frame' / 'live-views.txt')
    cams = fh.load_calibration(session / 'frame' / 'snapshot' / 'device' / 'persist')
    tiles = {n: tiles_for(c, rotation_search=True) for n, c in cams.items()}
    stamps = rec.stamps()
    per_set, clean, hard = hand_labels(audit, stamps, cams)
    rng = np.random.default_rng(a.seed)
    imgs, boxes, nbox, meta = [], [], [], []
    shard_id = 0; counts = dict(positive_tiles=0, hard_positive_tiles=0, negative_tiles=0, boxes=0, sets=0)
    t0 = time.perf_counter()

    def flush():
        nonlocal imgs, boxes, nbox, meta, shard_id
        if not imgs:
            return
        np.savez(out / f'shard-{shard_id:03d}.npz', images=np.stack(imgs), boxes=np.stack(boxes).astype(np.float32),
                 nbox=np.array(nbox, np.int32), meta=np.array(json.dumps(meta)))
        shard_id += 1; imgs, boxes, nbox, meta = [], [], [], []

    def add(img, center, size, rot, labels, cam_name, i, kind):
        tm = fh.crop_matrix(center, size, rot, fh.PALM_SIZE)
        patch = fh.crop_patch(img, tm, fh.PALM_SIZE)
        arr = np.zeros((MAX_BOXES, 19), np.float32)   # cx cy side kp(14) weight valid
        n = 0
        for pts, w in labels:
            c, side, kp = fh.palm_box_from_landmarks(pts)
            cc = fh.to_crop(tm, c)[0]
            s = side * fh.PALM_SIZE / size
            kpc = fh.to_crop(tm, kp)
            # keep boxes whose centre is inside the tile with some margin
            if not (-0.1 * fh.PALM_SIZE <= cc[0] <= 1.1 * fh.PALM_SIZE and -0.1 * fh.PALM_SIZE <= cc[1] <= 1.1 * fh.PALM_SIZE):
                continue
            if n >= MAX_BOXES:
                break
            arr[n, :3] = (cc[0], cc[1], s); arr[n, 3:17] = kpc.reshape(-1); arr[n, 17] = w; arr[n, 18] = 1
            n += 1
        imgs.append(patch); boxes.append(arr); nbox.append(n)
        meta.append(dict(session=session.name, set=i, stamp=stamps[i], camera=cam_name, center=[float(center[0]), float(center[1])],
                         size=float(size), rotation=float(rot), kind=kind))
        counts['boxes'] += n
        if len(imgs) >= SHARD:
            flush()

    for i in range(0, len(stamps), a.every):
        labels_i = per_set[i]
        if not labels_i and i not in clean:
            continue
        images = rec.read(i)
        counts['sets'] += 1
        for cam_name, cam in cams.items():
            if cam_name not in images:
                continue
            img = images[cam_name]
            cam_labels = [(pts, w) for c, pts, w in labels_i if c == cam_name]
            # positives: tiles whose square contains a labelled palm, plus jitter
            for pts, w in cam_labels:
                palm = pts[9]
                containing = [t for t in tiles[cam_name] if np.all(np.abs(palm - t[0]) < t[1] / 2 * 0.9)]
                if not containing:
                    continue
                sel = rng.choice(len(containing), size=min(a.tiles_per_hand, len(containing)), replace=False)
                for k in sel:
                    center, size, rot = containing[k]
                    jc = center + rng.uniform(-0.1, 0.1, 2) * size
                    js = size * rng.uniform(0.8, 1.25)
                    jr = rot + math.radians(rng.uniform(-15, 15))
                    others = [(p, ww) for p, ww in cam_labels]
                    add(img, jc, js, jr, others, cam_name, i, 'hard' if w < 1 else 'pos')
                    counts['hard_positive_tiles' if w < 1 else 'positive_tiles'] += 1
            # negatives only in clean sets
            if i in clean:
                for _ in range(a.negatives_per_set):
                    for _try in range(10):
                        center, size, rot = tiles[cam_name][rng.integers(len(tiles[cam_name]))]
                        jc = center + rng.uniform(-0.1, 0.1, 2) * size
                        js = size * rng.uniform(0.8, 1.25)
                        jr = rot + math.radians(rng.uniform(-15, 15))
                        far = all(np.max(np.abs(pts[9] - jc)) > js / 2 * 1.15 for pts, _ in cam_labels)
                        if far:
                            add(img, jc, js, jr, [], cam_name, i, 'neg'); counts['negative_tiles'] += 1
                            break
        if counts['sets'] % 200 == 0:
            print(f"{session.name}: {counts['sets']} sets, {len(imgs) + shard_id * SHARD} samples, {time.perf_counter() - t0:.0f}s", flush=True)
    flush()
    summary = dict(session=session.name, shards=shard_id, hard_positive_views=hard, clean_sets=len(clean), **counts,
                   seconds=time.perf_counter() - t0, every=a.every, tiles_per_hand=a.tiles_per_hand)
    (out / 'index.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
