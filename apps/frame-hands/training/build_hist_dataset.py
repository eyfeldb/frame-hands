"""Dataset for the history-conditioned hand landmark model (option A, MEgATrack style).

Per sample: a 224x224 tracker-style crop (uint8), its crop->image affine, the 21 label landmarks in crop px
with a reliability weight, the history landmarks in crop px (extrapolated from the hand's triangulated
3D at t-1,t-2, projected into this camera) with a kind flag, an optional stale history (from ~20 sets
earlier), presence/handedness targets, and the base model's outputs on the same crop (for distillation
of the world landmarks and as a reference).

Sample kinds:
  tracked: a LIVE view at set t; crop from roi_from_points(view at t-1) when it exists (jittered), else from
           the current view (fresh acquisition). Label = stereo-triangulated 3D reprojected (w=1.0) when the
           pair residual is small, else the view's own landmarks (w=0.7).
  hard:    sets inside a 1-4 frame gap of a (camera,id) track; 3D interpolated from the neighbours,
           reprojected as label (w=0.5); crop from the interpolated 2D (jittered). Presence target 1.
  neg:     background crops in clean sets (exactly two stereo hands, opposite sides), presence 0; 30% carry
           a fake history map (a hand's history drawn where no hand is) so history cannot create a hand.
Read-only on inputs; writes npz shards + index.json under --out.

usage: python build_hist_dataset.py <session-dir> <base-models-dir> --out <dir> [--every 1] [--seed 0]
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

SHARD = 1500
N = fh.HAND_SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session'); ap.add_argument('models'); ap.add_argument('--out', required=True)
    ap.add_argument('--every', type=int, default=1); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--neg-per-set', type=int, default=1)
    ap.add_argument('--ghost-neg', action='store_true', help='ghost tracks (single-camera third hands / same-side duplicates) become presence-0 negatives')
    ap.add_argument('--keep', default='', help='only sets inside these second ranges from the first set, e.g. "5-40,62-118" (reviewed hands-only windows)')
    a = ap.parse_args()
    keep = [tuple(float(x) for x in part.split('-')) for part in a.keep.split(',') if part.strip()]
    session = Path(a.session); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rec = fh.Recording(session / 'frame' / 'sets.bin')
    audit = fh.read_audit(session / 'frame' / 'live-views.txt')
    cams = fh.load_calibration(session / 'frame' / 'snapshot' / 'device' / 'persist')
    nets = fh.NcnnNets(a.models)
    stamps = rec.stamps()
    rng = np.random.default_rng(a.seed)
    # per set: views by (cam,id); per set per id: triangulated 3D
    views_at = {}      # (i, cam, id) -> view dict
    tri_at = {}        # (i, id) -> (pts3 (21,3), residual m)
    right_at = {}      # (i, id) -> right score (audit hand row)
    for i, s in enumerate(stamps):
        a_ = audit.get(s)
        if not a_:
            continue
        by_id = {}
        for v in a_['views']:
            if v['presence'] >= 0.5 and v['camera'] in cams:
                views_at[(i, v['camera'], v['id'])] = v; by_id.setdefault(v['id'], []).append(v)
        for hid, (r, nv) in a_['hands'].items():
            right_at[(i, hid)] = r
        for hid, vs in by_id.items():
            if len(vs) >= 2:
                pts, res = fh.triangulate_hand(cams, [(cams[v['camera']], v['pts'], v['presence']) for v in vs[:2]])
                tri_at[(i, hid)] = (pts, res)
    clean = set()
    for i, s in enumerate(stamps):
        a_ = audit.get(s)
        if not a_:
            continue
        st = [(r, nv) for r, nv in a_['hands'].values() if nv >= 2]
        if len(st) == 2 and len(a_['hands']) == 2 and (st[0][0] > 0.5) != (st[1][0] > 0.5):
            clean.add(i)
    # --ghost-neg: tracks that are ghosts (forearms, monitor edges): mostly single-camera for their whole life and
    # usually a third hand beside two others, or a single-camera duplicate of a same-labelled two-camera hand.
    # Their views become presence-0 negatives (with their real history map) and never positives; other tracks
    # that were never seen by two cameras are left out of the positives as unverifiable.
    life = {}; mono_only = set(); ghosts = set()
    if a.ghost_neg:
        for i, s in enumerate(stamps):
            a_ = audit.get(s)
            if not a_:
                continue
            for hid, (r, nv) in a_['hands'].items():
                d = life.setdefault(hid, dict(sets=0, stereo=0, third=0, dup=0))
                d['sets'] += 1; d['stereo'] += nv >= 2
                others = [(h2, r2, nv2) for h2, (r2, nv2) in a_['hands'].items() if h2 != hid]
                if len(others) >= 2: d['third'] += 1
                if nv < 2 and any(nv2 >= 2 and (r2 > 0.5) == (r > 0.5) for _, r2, nv2 in others): d['dup'] += 1
        # v2 rule: a ghost is a mostly single-camera track that is a THIRD hand (two other hands present) for at least
        # half of its life. The earlier "single-camera duplicate of a same-labelled stereo hand" criterion also caught
        # real hands with a wrong handedness label and the retrained model then dropped real hands (in-air both-hands
        # 88 -> 78%). Other single-camera tracks stay positives (label = their own landmarks, weight 0.7) as before.
        for hid, d in life.items():
            if d['sets'] < 15: continue
            if d['stereo'] / d['sets'] < 0.2 and d['third'] / d['sets'] >= 0.5: ghosts.add(hid)
        print(f'ghost tracks: {sorted(ghosts)} ({sum(life[g]["sets"] for g in ghosts)} sets)', flush=True)

    def hist3d(i, hid):
        """Extrapolated 3D at set i from t-1,t-2 triangulations -> (pts, kind) kind: 0 none, 1 prev, 2 extrapolated."""
        p1 = tri_at.get((i - 1, hid)); p2 = tri_at.get((i - 2, hid))
        if p1 and p2 and p1[1] < 0.02 and p2[1] < 0.02:
            ex = 2 * p1[0] - p2[0]
            # bound the extrapolation step to 8 cm
            d = ex - p1[0]; n = np.linalg.norm(d, axis=1, keepdims=True)
            ex = p1[0] + d * np.minimum(1.0, 0.08 / np.maximum(n, 1e-9))
            return ex, 2
        if p1 and p1[1] < 0.02:
            return p1[0], 1
        return None, 0

    def project(cam, pts3):
        uv, z = cam.project(pts3)
        return uv if np.all(z > 0.03) else None

    shards = dict(img=[], tm=[], label=[], lw=[], hist=[], hkind=[], stale=[], sok=[], pres=[], right=[], wbase=[], pbase=[], rbase=[], ptsbase=[], kind=[], meta=[])
    shard_id = 0; counts = dict(tracked=0, tracked_stereo_label=0, hard=0, neg=0, neg_fake_hist=0, sets=0)
    t0 = time.perf_counter()

    def flush():
        nonlocal shard_id
        if not shards['img']:
            return
        np.savez(out / f'shard-{shard_id:03d}.npz',
                 images=np.stack(shards['img']), tm=np.stack(shards['tm']).astype(np.float32), label=np.stack(shards['label']).astype(np.float32),
                 lw=np.array(shards['lw'], np.float32), hist=np.stack(shards['hist']).astype(np.float32), hkind=np.array(shards['hkind'], np.int8),
                 stale=np.stack(shards['stale']).astype(np.float32), sok=np.array(shards['sok'], np.int8), pres=np.array(shards['pres'], np.float32),
                 right=np.array(shards['right'], np.float32), wbase=np.stack(shards['wbase']).astype(np.float32), pbase=np.array(shards['pbase'], np.float32),
                 rbase=np.array(shards['rbase'], np.float32), ptsbase=np.stack(shards['ptsbase']).astype(np.float32), kind=np.array(shards['kind']),
                 meta=np.array(json.dumps(shards['meta'])))
        shard_id += 1
        for k in shards:
            shards[k] = []

    def add(img, cam_name, center, size, rot, label_img, lw, hist_img, hkind, stale_img, sok, pres, right, kind, i, hid):
        tm = fh.crop_matrix(center, size, rot, N)
        patch = fh.crop_patch(img, tm, N)
        base = nets._run(nets.hand, fh.model_input(patch), ('out0', 'out1', 'out2', 'out3'))
        screen = base[0].reshape(21, 3)
        zero = np.zeros((21, 2), np.float32)
        shards['img'].append(patch); shards['tm'].append(tm)
        shards['label'].append(fh.to_crop(tm, label_img) if label_img is not None else zero); shards['lw'].append(lw)
        shards['hist'].append(fh.to_crop(tm, hist_img) if hist_img is not None else zero); shards['hkind'].append(hkind)
        shards['stale'].append(fh.to_crop(tm, stale_img) if stale_img is not None else zero); shards['sok'].append(sok)
        shards['pres'].append(pres); shards['right'].append(right)
        shards['wbase'].append(base[3].reshape(21, 3)); shards['pbase'].append(float(base[1][0])); shards['rbase'].append(float(base[2][0]))
        shards['ptsbase'].append(screen[:, :2]); shards['kind'].append(kind)
        shards['meta'].append(dict(session=session.name, set=i, stamp=stamps[i], camera=cam_name, id=hid, center=[float(center[0]), float(center[1])], size=float(size), rotation=float(rot)))
        if len(shards['img']) >= SHARD:
            flush()

    for i in range(0, len(stamps), a.every):
        s = stamps[i]; a_ = audit.get(s)
        if not a_:
            continue
        if keep and not any(lo <= (s - stamps[0]) / 1e9 <= hi for lo, hi in keep):
            continue
        images = None
        # tracked views
        for cam_name, cam in cams.items():
            for v in a_['views']:
                if v['camera'] != cam_name or v['presence'] < 0.5:
                    continue
                if images is None:
                    images = rec.read(i); counts['sets'] += 1
                hid = v['id']
                if hid in ghosts:
                    # a ghost view: the tracked crop as the tracker would make it, presence 0, its real history
                    prev = views_at.get((i - 1, cam_name, hid))
                    center, size, rot = fh.roi_from_points(prev['pts'] if prev is not None else v['pts'])
                    center = center + rng.normal(0, 0.04, 2) * size; size *= rng.uniform(0.9, 1.15); rot += math.radians(rng.uniform(-10, 10))
                    h3, hk = hist3d(i, hid)
                    hist = project(cam, h3) if h3 is not None else None
                    if hist is None:
                        hist = v['pts'] + rng.normal(0, 4, 2); hk = 1
                    add(images[cam_name], cam_name, center, size, rot, None, 0.0, hist, hk, None, 0, 0.0, 0.0, 'neg', i, hid)
                    counts['neg'] += 1; counts['ghost_neg'] = counts.get('ghost_neg', 0) + 1
                    continue
                if hid in mono_only:
                    continue
                prev = views_at.get((i - 1, cam_name, hid))
                if prev is not None:
                    center, size, rot = fh.roi_from_points(prev['pts'])
                    center = center + rng.normal(0, 0.04, 2) * size; size *= rng.uniform(0.9, 1.15); rot += math.radians(rng.uniform(-10, 10))
                else:
                    center, size, rot = fh.roi_from_points(v['pts'])
                    center = center + rng.normal(0, 0.08, 2) * size; size *= rng.uniform(0.85, 1.3); rot += math.radians(rng.uniform(-20, 20))
                tri = tri_at.get((i, hid))
                if tri and tri[1] < 0.012:
                    label = project(cam, tri[0]); lw = 1.0 if label is not None else 0.0
                    if label is None:
                        label = v['pts']; lw = 0.7
                    else:
                        counts['tracked_stereo_label'] += 1
                else:
                    label, lw = v['pts'], 0.7
                h3, hk = hist3d(i, hid)
                hist = project(cam, h3) if h3 is not None else None
                if hist is None:
                    hk = 0
                st = tri_at.get((i - 20, hid)); stale = project(cam, st[0]) if st else None
                right = right_at.get((i, hid), v['right'])
                add(images[cam_name], cam_name, center, size, rot, label, lw, hist, hk, stale, int(stale is not None), 1.0, float(right > 0.5), 'tracked', i, hid)
                counts['tracked'] += 1
        # hard: gaps in (cam,id) tracks ending at this set
        for cam_name, cam in cams.items():
            for v in a_['views']:
                if v['camera'] != cam_name or v['presence'] < 0.5:
                    continue
                hid = v['id']
                if views_at.get((i - 1, cam_name, hid)) is not None:
                    continue
                # find previous view of this (cam,id) within 5 sets
                for g in range(2, 6):
                    pv = views_at.get((i - g, cam_name, hid))
                    if pv is None:
                        continue
                    ta, tb = tri_at.get((i - g, hid)), tri_at.get((i, hid))
                    if not ta or not tb or ta[1] > 0.012 or tb[1] > 0.012:
                        break
                    if images is None:
                        images = rec.read(i); counts['sets'] += 1
                    for j in range(i - g + 1, i):
                        a_j = (j - (i - g)) / g
                        p3 = ta[0] * (1 - a_j) + tb[0] * a_j
                        label = project(cam, p3)
                        if label is None:
                            continue
                        imgs_j = rec.read(j)
                        if cam_name not in imgs_j:
                            continue
                        center, size, rot = fh.roi_from_points(label)
                        center = center + rng.normal(0, 0.05, 2) * size; size *= rng.uniform(0.9, 1.2); rot += math.radians(rng.uniform(-12, 12))
                        h3, hk = hist3d(j, hid)
                        if h3 is None:   # history from the last triangulation before the gap
                            h3, hk = ta[0], 1
                        hist = project(cam, h3)
                        right = right_at.get((i, hid), v['right'])
                        add(imgs_j[cam_name], cam_name, center, size, rot, label, 0.5, hist, hk if hist is not None else 0, None, 0, 1.0, float(right > 0.5), 'hard', j, hid)
                        counts['hard'] += 1
                    break
        # negatives in clean sets
        if i in clean:
            if images is None:
                images = rec.read(i); counts['sets'] += 1
            for cam_name, cam in cams.items():
                hand_pts = [v['pts'] for v in a_['views'] if v['camera'] == cam_name and v['presence'] >= 0.5]
                for _ in range(a.neg_per_set):
                    for _try in range(12):
                        size = rng.uniform(120, 420); center = np.array([rng.uniform(size / 2, cam.width - size / 2), rng.uniform(size / 2, cam.height - size / 2)]); rot = rng.uniform(-math.pi, math.pi)
                        if all(np.max(np.abs(p[9] - center)) > size * 0.75 for p in hand_pts) and cam.off_axis_deg(center[None])[0] < 80:
                            fake = None; hk = 0
                            if hand_pts and rng.random() < 0.3:
                                src = hand_pts[rng.integers(len(hand_pts))]
                                fake = src - src[9] + center + rng.normal(0, 10, 2); hk = 2; counts['neg_fake_hist'] += 1
                            add(images[cam_name], cam_name, center, size, rot, None, 0.0, fake, hk, None, 0, 0.0, 0.0, 'neg', i, -1)
                            counts['neg'] += 1
                            break
        if counts['sets'] % 300 == 0 and counts['sets']:
            print(f"{session.name}: {counts['sets']} sets, {len(shards['img']) + shard_id * SHARD} samples, {time.perf_counter() - t0:.0f}s", flush=True)
    flush()
    summary = dict(session=session.name, shards=shard_id, clean_sets=len(clean), **counts, seconds=time.perf_counter() - t0)
    (out / 'index.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
