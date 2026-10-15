"""Fine-tune the MediaPipe palm detector on Frame-domain tiles (build_palm_dataset.py shards).

Model: the OpenCV Zoo ONNX converted with onnx2torch (NHWC input, outputs (1,2016,18) raw box/keypoint
offsets in 192-px units relative to the anchor centre, and (1,2016,1) logits), exactly what the
tracker decodes. Loss: focal BCE on the logits with anchor assignment by distance to the palm
centre, smooth-L1 on offsets for positive anchors (centre, side x2, 7 keypoints). Photometric
augmentation then CLAHE(4x4, clip 2) like the tracker, in the data loader.

Writes checkpoints and a metrics log under --out. Export with export_ncnn.py / finalize_palm.py.

usage: python train_palm.py --source <palm.onnx> --train <shard-dir> [--train <dir> ...] --dev <shard-dir> --out <dir>
                            [--epochs 20] [--batch 64] [--lr 1e-4] [--workers 6]
"""
import argparse
import glob
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fh_common as fh  # noqa: E402

STRIDE_OF_ANCHOR = np.concatenate((np.full(24 * 24 * 2, 8.0), np.full(12 * 12 * 6, 16.0)))


class Shards:
    """Tiles from one or more shard directories, concatenated once into .npy caches and memory-mapped
    lazily in each worker (no per-worker copies). `limit` takes a fixed random subset."""

    def __init__(self, dirs, augment, cache, limit=0, seed=0):
        self.files = sorted(f for d in dirs for f in glob.glob(str(Path(d) / 'shard-*.npz')))
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1('|'.join(f'{f}:{Path(f).stat().st_size}' for f in self.files).encode()).hexdigest()[:12]
        self.img_path, self.box_path = cache / f'images-{key}.npy', cache / f'boxes-{key}.npy'
        if not self.img_path.exists() or not self.box_path.exists():
            imgs, boxes = [], []
            for f in self.files:
                z = np.load(f)
                imgs.append(z['images']); boxes.append(z['boxes'])
            np.save(self.img_path, np.concatenate(imgs)); np.save(self.box_path, np.concatenate(boxes))
        n = np.load(self.box_path, mmap_mode='r').shape[0]
        self.order = np.arange(n)
        if limit and limit < n:
            self.order = np.sort(np.random.default_rng(seed).choice(n, size=limit, replace=False))
        self.augment = augment
        self._images = self._boxes = None   # opened lazily per worker (the object is pickled to workers on Windows)

    def __len__(self):
        return len(self.order)

    def __getitem__(self, j):
        if self._images is None:
            self._images = np.load(self.img_path, mmap_mode='r'); self._boxes = np.load(self.box_path, mmap_mode='r')
        i = int(self.order[j])
        img = np.array(self._images[i])
        boxes = np.array(self._boxes[i])
        if self.augment:
            rng = np.random.default_rng((i * 7919 + int(time.time() * 1000)) % (2 ** 32))
            f = img.astype(np.float32)
            # exposure / backlight: gamma, gain, offset, occasional strong darkening of the hand region
            gamma = math.exp(rng.normal(0, 0.35))
            f = 255.0 * (f / 255.0) ** gamma
            f = f * rng.uniform(0.6, 1.4) + rng.uniform(-30, 30)
            if rng.random() < 0.3:
                f = f + rng.normal(0, rng.uniform(2, 10), f.shape)
            if rng.random() < 0.2:
                import cv2
                k = int(rng.choice([3, 5]))
                f = cv2.GaussianBlur(f, (k, k), 0)
            if rng.random() < 0.5:   # horizontal flip: swaps nothing semantic for palms (handedness is not predicted)
                f = f[:, ::-1]
                boxes[:, 0] = np.where(boxes[:, 18] > 0, fh.PALM_SIZE - boxes[:, 0], boxes[:, 0])
                kp = boxes[:, 3:17].reshape(-1, 7, 2)
                kp[:, :, 0] = np.where(boxes[:, 18:19] > 0, fh.PALM_SIZE - kp[:, :, 0], kp[:, :, 0])
                boxes[:, 3:17] = kp.reshape(-1, 14)
            img = np.clip(f, 0, 255).astype(np.uint8)
        import torch
        img = fh.clahe_4x4(np.ascontiguousarray(img))
        x = img.astype(np.float32) / 255.0
        x = np.stack((x, x, x), -1)   # NHWC
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(boxes)


def assign(anchors, boxes, nbox):
    """boxes (B,M,19) -> cls target (B,N) in {0,1,-1 ignore}, reg target (B,N,18), reg weight (B,N)."""
    import torch
    B, N = boxes.shape[0], anchors.shape[0]
    cls = torch.zeros(B, N, device=boxes.device)
    reg = torch.zeros(B, N, 18, device=boxes.device)
    rw = torch.zeros(B, N, device=boxes.device)
    cw = torch.ones(B, N, device=boxes.device)
    a = anchors[None]            # (1,N,2)
    stride = torch.as_tensor(STRIDE_OF_ANCHOR, device=boxes.device, dtype=torch.float32)[None]
    for b in range(B):
        valid = boxes[b, :, 18] > 0
        if not valid.any():
            continue
        bx = boxes[b][valid]       # (m,19)
        d = torch.linalg.norm(a[0][:, None, :] - bx[None, :, :2], dim=2)   # (N,m)
        side = bx[:, 2][None]                                               # (1,m)
        pos_r = torch.maximum(0.3 * side, 0.75 * stride.T)                  # (N,m)
        ign_r = torch.maximum(0.6 * side, 1.5 * stride.T)
        best_m = d.argmin(1)
        dmin = d.gather(1, best_m[:, None])[:, 0]
        is_pos = dmin < pos_r.gather(1, best_m[:, None])[:, 0]
        is_ign = (~is_pos) & (dmin < ign_r.gather(1, best_m[:, None])[:, 0])
        # also guarantee the nearest anchor per level is positive
        for m in range(bx.shape[0]):
            for s in (8.0, 16.0):
                lvl = (stride[0] == s)
                dd = torch.where(lvl, d[:, m], torch.full_like(d[:, m], 1e9))
                is_pos[dd.argmin()] = True
        tgt = bx[best_m]            # (N,19)
        w = tgt[:, 17]
        cls[b] = torch.where(is_pos, torch.ones_like(cls[b]), torch.where(is_ign, torch.full_like(cls[b], -1.0), torch.zeros_like(cls[b])))
        cw[b] = torch.where(is_pos, w, torch.ones_like(w))
        reg[b, :, 0:2] = tgt[:, 0:2] - a[0]
        reg[b, :, 2:4] = tgt[:, 2:3].expand(-1, 2)
        reg[b, :, 4:18] = (tgt[:, 3:17].reshape(N, 7, 2) - a[0][:, None, :]).reshape(N, 14)
        rw[b] = torch.where(is_pos, w, torch.zeros_like(w))
    return cls, reg, rw, cw


def focal(logits, cls, cw, alpha=0.25, gamma=2.0):
    import torch
    import torch.nn.functional as F
    valid = cls >= 0
    t = cls.clamp(min=0)
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, t, reduction='none')
    pt = torch.where(t > 0, p, 1 - p)
    w = torch.where(t > 0, alpha * torch.ones_like(p), (1 - alpha) * torch.ones_like(p)) * (1 - pt) ** gamma
    loss = (w * ce * cw * valid).sum()
    return loss / max(1.0, float((cls > 0).sum()))


def evaluate(model, loader, anchors, device, threshold=0.3):
    """Dev metrics with the tracker's decode: recall of labelled palms (full-weight only), false
    positives per negative tile, mean centre error of matched detections (tile px)."""
    import torch
    model.eval()
    anc = anchors.cpu().numpy()
    tp = fn = 0; fp_tiles = 0; neg_tiles = 0; errs = []; score_pos = []
    with torch.no_grad():
        for x, boxes in loader:
            out = model(x.to(device))
            raw, logits = out[0].float().cpu().numpy(), out[1].float().cpu().numpy().reshape(x.shape[0], -1)
            for b in range(x.shape[0]):
                dets = fh.decode_palms(raw[b], logits[b], anc, threshold)
                gts = boxes[b][boxes[b][:, 18] > 0].numpy()
                full = gts[gts[:, 17] >= 1.0]
                if len(gts) == 0:
                    neg_tiles += 1
                    fp_tiles += int(len(dets) > 0)
                    continue
                for g in full:
                    best = None
                    for d in dets:
                        e = np.linalg.norm(d['c'] - g[:2])
                        if e < 0.5 * max(g[2], 12) and (best is None or e < best[0]):
                            best = (e, d['score'])
                    if best:
                        tp += 1; errs.append(best[0]); score_pos.append(best[1])
                    else:
                        fn += 1
    return dict(recall=tp / max(1, tp + fn), tp=tp, fn=fn, fp_tile_rate=fp_tiles / max(1, neg_tiles), neg_tiles=neg_tiles,
                centre_err_px=float(np.median(errs)) if errs else None, score_p10=float(np.percentile(score_pos, 10)) if score_pos else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True); ap.add_argument('--train', action='append', required=True)
    ap.add_argument('--dev', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--eval-threshold', type=float, default=0.3)
    ap.add_argument('--limit-train', type=int, default=0)
    ap.add_argument('--dev-limit', type=int, default=4000)
    a = ap.parse_args()
    import torch
    from palm_model import load_model
    torch.manual_seed(0)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(a.source).to(device)
    anchors = torch.as_tensor(fh.palm_anchors(), dtype=torch.float32, device=device)
    train = Shards(a.train, augment=True, cache=out / 'cache', limit=a.limit_train)
    dev = Shards([a.dev], augment=False, cache=out / 'cache', limit=a.dev_limit)
    print(f'train {len(train)} tiles, dev {len(dev)} tiles, device {device}', flush=True)
    tl = torch.utils.data.DataLoader(train, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    dl = torch.utils.data.DataLoader(dev, batch_size=a.batch, shuffle=False, num_workers=max(1, a.workers // 2))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    steps = a.epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1, anneal_strategy='cos', div_factor=10, final_div_factor=20)
    base = evaluate(model, dl, anchors, device, a.eval_threshold)
    log = [dict(epoch=0, dev=base)]
    print('baseline dev:', json.dumps(base), flush=True)
    best = base['recall'] - base['fp_tile_rate']
    torch.save(model.state_dict(), out / 'baseline.pt')
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.perf_counter(); tot = dict(cls=0.0, reg=0.0, n=0)
        for x, boxes in tl:
            x, boxes = x.to(device, non_blocking=True), boxes.to(device, non_blocking=True)
            raw, logits = model(x)
            logits = logits.reshape(x.shape[0], -1)
            cls, reg, rw, cw = assign(anchors, boxes, None)
            lc = focal(logits, cls, cw)
            diff = torch.nn.functional.smooth_l1_loss(raw / 16.0, reg / 16.0, reduction='none', beta=0.5).sum(2)
            lr_ = (diff * rw).sum() / max(1.0, float(rw.sum()))
            loss = lc + lr_
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            tot['cls'] += float(lc); tot['reg'] += float(lr_); tot['n'] += 1
        m = evaluate(model, dl, anchors, device, a.eval_threshold)
        rec = dict(epoch=ep, train_cls=tot['cls'] / tot['n'], train_reg=tot['reg'] / tot['n'], seconds=time.perf_counter() - t0, dev=m, lr=sched.get_last_lr()[0])
        log.append(rec)
        print(json.dumps(rec), flush=True)
        torch.save(model.state_dict(), out / f'epoch-{ep:03d}.pt')
        score = m['recall'] - m['fp_tile_rate']
        if score > best:
            best = score
            torch.save(model.state_dict(), out / 'best.pt')
            (out / 'best.json').write_text(json.dumps(rec, indent=1))
        (out / 'log.json').write_text(json.dumps(log, indent=1))


if __name__ == '__main__':
    main()
