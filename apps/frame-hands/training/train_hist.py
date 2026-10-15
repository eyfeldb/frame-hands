"""Fine-tune the MediaPipe hand landmark model with a history channel (option A, MEgATrack style).

Input: the 224x224 grey crop in channels 0 and 1, and a rendered history map in channel 2 (Gaussian blobs at
the extrapolated previous-pose landmarks, intensity coded per finger). Model surgery folds the pretrained
channel-2 stem weights into channels 0/1 and zeroes channel 2, so the untouched model is reproduced exactly at
start. History augmentation per sample: none (p_none), stale (p_stale, from ~20 sets earlier when available),
else the extrapolated history plus Gaussian pixel noise. Losses: L1 on 2D landmarks (crop px) weighted by label
reliability; BCE presence; BCE handedness (hand crops); L1 distillation of the world landmarks to the base model.

usage: python train_hist.py --source <handpose.onnx> --train <shard-dir> [--train ...] --dev <shard-dir> --out <dir>
                            [--epochs 12] [--batch 64] [--lr 1e-4] [--workers 6]
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

N = fh.HAND_SIZE
FINGER_OF = np.array([0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5])      # 0 wrist, 1 thumb .. 5 pinky
INTENSITY = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5], np.float32)


class Shards:
    KEYS = ('images', 'label', 'lw', 'hist', 'hkind', 'stale', 'sok', 'pres', 'right', 'wbase', 'kind', 'tm', 'key')

    def __init__(self, dirs, augment, cache, limit=0, seed=0, cutout=0.0, eval_occlude=False):
        self.cutout = cutout; self.eval_occlude = eval_occlude
        self.files = sorted(f for d in dirs for f in glob.glob(str(Path(d) / 'shard-*.npz')))
        cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1('|'.join(f'{f}:{Path(f).stat().st_size}' for f in self.files).encode()).hexdigest()[:12]
        self.paths = {k: cache / f'{k}-{key}.npy' for k in self.KEYS}
        if not all(p.exists() for p in self.paths.values()):
            acc = {k: [] for k in self.KEYS}
            for f in self.files:
                z = np.load(f)
                meta = json.loads(str(z['meta']))
                for k in self.KEYS:
                    if k == 'kind':
                        acc[k].append(np.array([{'tracked': 0, 'hard': 1, 'neg': 2}[str(x)] for x in z[k]], np.int8))
                    elif k == 'key':
                        # (session hash, set, camera, hand id): consecutive sets of one (session, camera, id) form temporal triplets
                        acc[k].append(np.array([[int(hashlib.sha1(m['session'].encode()).hexdigest()[:8], 16), m['set'],
                                                 0 if m['camera'] == 'slam_left' else 1, m['id']] for m in meta], np.int64))
                    else:
                        acc[k].append(z[k])
            for k in self.KEYS:
                np.save(self.paths[k], np.concatenate(acc[k]))
        n = np.load(self.paths['pres'], mmap_mode='r').shape[0]
        self.order = np.arange(n)
        if limit and limit < n:
            self.order = np.sort(np.random.default_rng(seed).choice(n, size=limit, replace=False))
        self.augment = augment; self._a = None

    def __len__(self):
        return len(self.order)

    def __getitem__(self, j):
        return self.item(int(self.order[j]))

    def item(self, i):
        import torch
        if self._a is None:
            self._a = {k: np.load(p, mmap_mode='r') for k, p in self.paths.items()}
        A = self._a
        img = np.array(A['images'][i]); label = np.array(A['label'][i]); hist = np.array(A['hist'][i]); hkind = int(A['hkind'][i])
        stale = np.array(A['stale'][i]); sok = int(A['sok'][i])
        if self.augment:
            rng = np.random.default_rng((i * 7919 + int(time.time() * 1000)) % (2 ** 32))
            f = img.astype(np.float32)
            f = 255.0 * (f / 255.0) ** math.exp(rng.normal(0, 0.3)); f = f * rng.uniform(0.6, 1.4) + rng.uniform(-25, 25)
            if rng.random() < 0.3:
                f = f + rng.normal(0, rng.uniform(2, 8), f.shape)
            # synthetic occlusion: erase regions over 1-2 fingers (label stays: stereo-consistent) or a random block
            if self.cutout and float(A['lw'][i]) > 0 and rng.random() < self.cutout:
                for _ in range(rng.integers(1, 3)):
                    if rng.random() < 0.7:
                        fi = int(rng.integers(1, 6)); pts = label[FINGER_OF == fi]
                        c = pts.mean(0) + rng.normal(0, 6, 2); half = max(14.0, np.ptp(pts, axis=0).max() * 0.6)
                    else:
                        c = rng.uniform(40, N - 40, 2); half = rng.uniform(20, 50)
                    x0, y0 = int(max(0, c[0] - half)), int(max(0, c[1] - half)); x1, y1 = int(min(N, c[0] + half)), int(min(N, c[1] + half))
                    if x1 > x0 and y1 > y0:
                        fill = rng.choice([0.0, 1.0, 2.0])
                        f[y0:y1, x0:x1] = (rng.uniform(0, 255) if fill == 0 else (f[y0:y1, x0:x1].mean() if fill == 1 else rng.normal(128, 40, (y1 - y0, x1 - x0))))
            img = np.clip(f, 0, 255).astype(np.uint8)
            # history policy
            r = rng.random()
            if hkind == 0 or r < 0.15:
                hkind = 0
            elif r < 0.25 and sok:
                hist = stale + rng.normal(0, 3, hist.shape); hkind = 3
            else:
                hist = hist + rng.normal(0, rng.uniform(1.5, 8.0), hist.shape)
                if rng.random() < 0.2:                      # whole-hand shift (fast motion / wrong extrapolation)
                    hist = hist + rng.normal(0, 12, 2)
        if self.eval_occlude and float(A['lw'][i]) > 0:
            # deterministic finger occlusion for evaluation: erase one finger's region with its local mean
            rng = np.random.default_rng(i); fi = int(rng.integers(1, 6)); pts = label[FINGER_OF == fi]
            c = pts.mean(0); half = max(14.0, np.ptp(pts, axis=0).max() * 0.6)
            x0, y0 = int(max(0, c[0] - half)), int(max(0, c[1] - half)); x1, y1 = int(min(N, c[0] + half)), int(min(N, c[1] + half))
            if x1 > x0 and y1 > y0:
                img = img.copy(); img[y0:y1, x0:x1] = int(img[y0:y1, x0:x1].mean())
        x = img.astype(np.float32) / 255.0
        return (torch.from_numpy(x), torch.from_numpy(hist.astype(np.float32)), torch.tensor(hkind), torch.from_numpy(label.astype(np.float32)),
                torch.tensor(float(A['lw'][i])), torch.tensor(float(A['pres'][i])), torch.tensor(float(A['right'][i])),
                torch.from_numpy(np.array(A['wbase'][i], np.float32)), torch.tensor(int(A['kind'][i])))


class Triplets(Shards):
    """Temporal triplets (set-1, set, set+1) of one (session, camera, hand id) among the tracked crops with a label
    (UmeTrack-style acceleration loss). Each member is loaded and augmented like a single sample; the three crop
    affines come along so predictions can be mapped into the centre crop's pixel frame."""

    def __init__(self, dirs, augment, cache, limit=0, seed=0, cutout=0.0):
        super().__init__(dirs, augment, cache, cutout=cutout)
        A = {k: np.load(self.paths[k], mmap_mode='r') for k in ('key', 'kind', 'lw')}
        key = np.array(A['key']); ok = (np.array(A['kind']) == 0) & (np.array(A['lw']) > 0)
        rows = {tuple(key[i]): i for i in np.flatnonzero(ok)}
        trip = []
        for (s, t, c, h), i in rows.items():
            p = rows.get((s, t - 1, c, h)); n = rows.get((s, t + 1, c, h))
            if p is not None and n is not None:
                trip.append((p, i, n))
        self.triplets = np.array(trip, np.int64).reshape(-1, 3)
        if limit and limit < len(self.triplets):
            self.triplets = self.triplets[np.sort(np.random.default_rng(seed).choice(len(self.triplets), size=limit, replace=False))]

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, j):
        import torch
        p, i, n = (int(v) for v in self.triplets[j])
        members = [self.item(k) for k in (p, i, n)]
        tms = torch.from_numpy(np.stack([np.array(self._a['tm'][k], np.float32) for k in (p, i, n)]))
        return tuple(torch.stack([m[f] for m in members]) for f in range(9)) + (tms,)


def centre_frame(screens, tms):
    """screens (B,3,21,2) crop px of the three members, tms (B,3,2,3) crop->image affines -> all three mapped into
    the centre member's crop pixel frame (B,3,21,2)."""
    import torch
    A = tms[..., :2]; b = tms[..., 2]                       # (B,3,2,2), (B,3,2)
    img = torch.einsum('bmij,bmkj->bmki', A, screens) + b[:, :, None, :]
    Ac = A[:, 1]; bc = b[:, 1]
    inv = torch.linalg.inv(Ac)                               # (B,2,2)
    return torch.einsum('bij,bmkj->bmki', inv, img - bc[:, None, None, :])


def acceleration(screens, tms):
    """Per-sample mean L1 acceleration (crop px) of the three members' landmarks in the centre crop frame."""
    c = centre_frame(screens, tms)
    return (c[:, 2] + c[:, 0] - 2 * c[:, 1]).abs().mean(-1).mean(-1)


def evaluate_jitter(model, loader, device):
    import torch
    model.eval(); acc = []
    with torch.no_grad():
        for x, hist, hkind, label, lw, pres, right, wbase, kind, tms in loader:
            B = x.shape[0]
            nhwc = assemble(x.flatten(0, 1), hist.flatten(0, 1), hkind.flatten(0, 1), device)
            screen, _, _, _ = run_model(model, nhwc)
            acc.extend(acceleration(screen[..., :2].reshape(B, 3, 21, 2), tms.to(device)).cpu().tolist())
    return float(np.mean(acc)) if acc else None


def render_history(hist, hkind, device, sigma=5.0):
    """hist (B,21,2) crop px, hkind (B,) -> (B,N,N) float map; zero where hkind == 0."""
    import torch
    B = hist.shape[0]
    ys = torch.arange(N, device=device, dtype=torch.float32)
    gy = ys.view(1, 1, N, 1); gx = ys.view(1, 1, 1, N)
    px = hist[..., 0].view(B, 21, 1, 1); py = hist[..., 1].view(B, 21, 1, 1)
    d2 = (gx - px) ** 2 + (gy - py) ** 2
    inten = torch.as_tensor(INTENSITY[FINGER_OF], device=device).view(1, 21, 1, 1)
    m = (inten * torch.exp(-d2 / (2 * sigma * sigma))).amax(1)
    return m * (hkind > 0).float().view(B, 1, 1)


def assemble(x, hist, hkind, device):
    """-> NHWC (B,N,N,3): grey, grey, history."""
    import torch
    h = render_history(hist.to(device), hkind.to(device), device)
    g = x.to(device)
    return torch.stack((g, g, h), -1)


def stem_surgery(model):
    """Fold pretrained channel-2 stem weights into channels 0/1 and zero channel 2 (identical output for grey input)."""
    import torch
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Conv2d) and mod.in_channels == 3:
            with torch.no_grad():
                w = mod.weight
                w[:, 0] += w[:, 2] / 2; w[:, 1] += w[:, 2] / 2; w[:, 2] = 0
            return name
    raise RuntimeError('stem conv not found')


def run_model(model, nhwc):
    out = model(nhwc)
    out = out if isinstance(out, (tuple, list)) else [out]
    screen, pres, right, world = out
    return screen.reshape(-1, 21, 3), pres.reshape(-1), right.reshape(-1), world.reshape(-1, 21, 3)


def evaluate(model, loader, device, history_mode='given'):
    """history_mode: 'given' (dataset history as stored, no noise), 'none' (zero map), 'stale' (stale when available)."""
    import torch
    model.eval()
    agg = dict(tracked_err=[], tracked_err_stereo=[], hard_err=[], hard_pres=[], tracked_pres=[], neg_pres=[], neg_fakehist_pres=[], right_acc=[])
    with torch.no_grad():
        for x, hist, hkind, label, lw, pres, right, wbase, kind in loader:
            if history_mode == 'none':
                hkind = torch.zeros_like(hkind)
            nhwc = assemble(x, hist, hkind, device)
            screen, p, r, w = run_model(model, nhwc)
            pts = screen[..., :2].cpu(); p = torch.sigmoid(p).cpu() if (p.min() < 0 or p.max() > 1) else p.cpu(); r = r.cpu()
            err = torch.linalg.norm(pts - label, dim=-1).mean(-1)
            tr = kind == 0; hd = kind == 1; ng = kind == 2
            agg['tracked_err'].extend(err[tr & (lw > 0)].tolist()); agg['tracked_err_stereo'].extend(err[tr & (lw >= 1.0)].tolist())
            agg['hard_err'].extend(err[hd].tolist()); agg['hard_pres'].extend((p[hd] >= 0.5).float().tolist()); agg['tracked_pres'].extend((p[tr] >= 0.5).float().tolist())
            agg['neg_pres'].extend((p[ng & (hkind == 0)] >= 0.5).float().tolist()); agg['neg_fakehist_pres'].extend((p[ng & (hkind > 0)] >= 0.5).float().tolist())
            agg['right_acc'].extend(((r[tr] > 0.5).float() == right[tr]).float().tolist())
    return {k: (float(np.mean(v)) if v else None) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True); ap.add_argument('--train', action='append', required=True); ap.add_argument('--dev', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=12); ap.add_argument('--batch', type=int, default=64); ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--dev-limit', type=int, default=6000); ap.add_argument('--world-weight', type=float, default=1.0)
    ap.add_argument('--cutout', type=float, default=0.0, help='probability of synthetic finger occlusion on hand crops')
    ap.add_argument('--init', default=None, help='start from this history-model checkpoint (after stem surgery); with --epochs 0 only evaluates it')
    ap.add_argument('--temporal', type=float, default=0.0, help='weight of the acceleration loss on temporal triplets (0: off); each step then adds a triplet batch')
    ap.add_argument('--triplet-batch', type=int, default=24, help='triplets per step when --temporal is on (3 crops each)')
    a = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from palm_model import load_model
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = load_model(a.source).to(device)
    stem = stem_surgery(model); print('stem conv:', stem, flush=True)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=device)); print('init from', a.init, flush=True)
    train = Shards(a.train, augment=True, cache=out / 'cache', cutout=a.cutout); dev = Shards([a.dev], augment=False, cache=out / 'cache', limit=a.dev_limit)
    dev_occ = Shards([a.dev], augment=False, cache=out / 'cache', limit=a.dev_limit, eval_occlude=True)
    print(f'train {len(train)} dev {len(dev)} device {device}', flush=True)
    tl = torch.utils.data.DataLoader(train, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    dl = torch.utils.data.DataLoader(dev, batch_size=a.batch, shuffle=False, num_workers=max(1, a.workers // 2))
    dlo = torch.utils.data.DataLoader(dev_occ, batch_size=a.batch, shuffle=False, num_workers=max(1, a.workers // 2))
    # temporal triplets: dev jitter is always measured; training triplets only with --temporal
    dev_trip = Triplets([a.dev], augment=False, cache=out / 'cache', limit=a.dev_limit)
    dlt = torch.utils.data.DataLoader(dev_trip, batch_size=a.batch // 2, shuffle=False, num_workers=max(1, a.workers // 2))
    tlt = None
    if a.temporal > 0:
        train_trip = Triplets(a.train, augment=True, cache=out / 'cache', cutout=a.cutout)
        tlt = torch.utils.data.DataLoader(train_trip, batch_size=a.triplet_batch, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
        print(f'triplets train {len(train_trip)} dev {len(dev_trip)}', flush=True)
    else:
        print(f'triplets dev {len(dev_trip)}', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    steps = a.epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1, div_factor=10, final_div_factor=20) if steps else None
    log = []
    def full_eval():
        return dict(given=evaluate(model, dl, device, 'given'), none=evaluate(model, dl, device, 'none'),
                    occluded_given=evaluate(model, dlo, device, 'given'), occluded_none=evaluate(model, dlo, device, 'none'),
                    jitter_px=evaluate_jitter(model, dlt, device))
    base = full_eval()
    print('epoch 0 (identical to base):' if not a.init else 'epoch 0 (init checkpoint):', json.dumps(base), flush=True); log.append(dict(epoch=0, dev=base))
    (out / 'log.json').write_text(json.dumps(log, indent=1))
    best = None
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.perf_counter(); tot = dict(lm=0.0, pres=0.0, right=0.0, world=0.0, acc=0.0, n=0)
        trip_iter = iter(tlt) if tlt is not None else None
        for x, hist, hkind, label, lw, pres, right, wbase, kind in tl:
            T = 0; tms = None
            if trip_iter is not None:
                try:
                    tx, thist, thkind, tlabel, tlw, tpres, tright, twbase, tkind, tms = next(trip_iter)
                except StopIteration:
                    trip_iter = iter(tlt); tx, thist, thkind, tlabel, tlw, tpres, tright, twbase, tkind, tms = next(trip_iter)
                T = tx.shape[0]
                x = torch.cat([x, tx.flatten(0, 1)]); hist = torch.cat([hist, thist.flatten(0, 1)]); hkind = torch.cat([hkind, thkind.flatten(0, 1)])
                label = torch.cat([label, tlabel.flatten(0, 1)]); lw = torch.cat([lw, tlw.flatten(0, 1)]); pres = torch.cat([pres, tpres.flatten(0, 1)])
                right = torch.cat([right, tright.flatten(0, 1)]); wbase = torch.cat([wbase, twbase.flatten(0, 1)]); kind = torch.cat([kind, tkind.flatten(0, 1)])
            nhwc = assemble(x, hist, hkind, device)
            screen, p, r, w = run_model(model, nhwc)
            label = label.to(device); lw = lw.to(device); pres = pres.to(device); right = right.to(device); wbase = wbase.to(device); kind = kind.to(device)
            hand = (kind != 2).float()
            l_lm = ((screen[..., :2] - label).abs().mean(-1).mean(-1) * lw).sum() / lw.sum().clamp(min=1) / N * 100.0
            l_pres = F.binary_cross_entropy(p.clamp(1e-5, 1 - 1e-5), pres)
            l_right = (F.binary_cross_entropy(r.clamp(1e-5, 1 - 1e-5), right, reduction='none') * hand).sum() / hand.sum().clamp(min=1)
            l_world = ((w - wbase).abs().mean(-1).mean(-1) * hand).sum() / hand.sum().clamp(min=1) * 100.0
            loss = l_lm + l_pres + 0.5 * l_right + a.world_weight * l_world
            l_acc = torch.zeros((), device=device)
            if T:
                # acceleration of the triplet members' landmarks in the centre crop frame, same units as l_lm
                l_acc = acceleration(screen[-3 * T:, :, :2].reshape(T, 3, 21, 2), tms.to(device)).mean() / N * 100.0
                loss = loss + a.temporal * l_acc
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
            tot['lm'] += l_lm.item(); tot['pres'] += l_pres.item(); tot['right'] += l_right.item(); tot['world'] += l_world.item(); tot['acc'] += l_acc.item(); tot['n'] += 1
        m = full_eval()
        rec = dict(epoch=ep, train={k: v / tot['n'] for k, v in tot.items() if k != 'n'}, seconds=time.perf_counter() - t0, dev=m)
        log.append(rec); print(json.dumps(rec), flush=True)
        torch.save(model.state_dict(), out / f'epoch-{ep:03d}.pt')
        g = m['given']
        score = g['tracked_err_stereo'] + (g['hard_err'] or 0) * 0.5 - 10 * (g['hard_pres'] or 0) + 10 * ((g['neg_pres'] or 0) + (g['neg_fakehist_pres'] or 0))   # a dev split may have no hard crops
        if a.temporal > 0 and m['jitter_px'] is not None:
            score += m['jitter_px']   # px of acceleration count like px of error when training for smoothness
        if best is None or score < best:
            best = score; torch.save(model.state_dict(), out / 'best.pt'); (out / 'best.json').write_text(json.dumps(rec, indent=1))
        (out / 'log.json').write_text(json.dumps(log, indent=1))


if __name__ == '__main__':
    main()
