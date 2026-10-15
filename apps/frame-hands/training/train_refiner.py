"""Train the causal GRU pose refiner on native rig traces; evaluate against raw fit and the baseline filter.

usage: python train_refiner.py --data <refiner-data dir> --profiles <tracker-rig-v9/profiles> --out <dir>
         [--train 030445Z 050930Z 032721Z] [--dev 040502Z] [--epochs 40] [--hidden 128]

Loss (FK joint-position space, mm): weighted L1 between FK(refined) and FK(target) per finger joint, plus
acceleration L1 of FK(refined) (jitter), plus small angle/translation regularisers. Synthetic corruption on
inputs: finger dropouts (frozen/drifting angles with bad evidence), Gaussian noise, missed frames (held
input, observed=0). Metrics: jitter (mean joint acceleration mm/frame^2), error to target on reliable
frames (mm), error on corrupted-finger frames (infill, mm), lag estimate (ms) from cross-correlation of
palm speed against the raw fit.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import refiner_data as rd  # noqa: E402
import refiner_model as rm  # noqa: E402
import refiner_targets as rt  # noqa: E402
from refiner_baseline import run_baseline_on_track  # noqa: E402

# Rig landmark ids per finger (hand_rig.h mediapipe_to_rig): tips 0-4, wrist 5, thumb MCP/IP 6-7,
# index 8-10, middle 11-13, ring 14-16, pinky 17-19, palm centre 20.
LM_FINGER = [[6, 7, 0], [8, 9, 10, 1], [11, 12, 13, 2], [14, 15, 16, 3], [17, 18, 19, 4]]
PALM = [20, 5, 8, 11, 14, 17]


def attach_teacher(tr, teacher):
    """Match UmeTrack landmarks (head mm) to track frames by capture stamp and side. Adds
    tr['teacher'] (N,21,3), tr['teacher_w'] (N,21) weights (0 where missing)."""
    n = len(tr['t'])
    tr['teacher'] = np.zeros((n, 21, 3)); tr['teacher_w'] = np.zeros((n, 21))
    if teacher is None:
        return
    stamps, sides = teacher['stamps'], teacher['sides']
    order = np.argsort(stamps); s_sorted = stamps[order]
    for i in range(n):
        st = int(round(tr['t'][i] * 1e9))
        j = np.searchsorted(s_sorted, st)
        for k in (j - 1, j, j + 1):
            if 0 <= k < len(s_sorted) and abs(int(s_sorted[k]) - st) <= 200_000 and sides[order[k]] == tr['side']:
                idx = order[k]
                tr['teacher'][i] = teacher['landmarks'][idx]
                sig = teacher['sigmas'][idx]
                tr['teacher_w'][i] = 1.0 / (1.0 + np.clip(sig, 0, 1) / max(1e-6, float(np.median(sig)) if np.median(sig) > 0 else 1.0))
                if not tr['observed'][i]:
                    tr['teacher_w'][i] *= 0.5
                break


def load_split(data_dir, names, rigs, teacher_dir=None):
    tracks = []
    for name in names:
        teacher = None
        if teacher_dir and (Path(teacher_dir) / f'{name}.npz').exists():
            z = np.load(Path(teacher_dir) / f'{name}.npz')
            teacher = dict(stamps=z['stamps'], sides=z['sides'], landmarks=z['landmarks'], sigmas=z['sigmas'])
        for tr in rd.load_sequences(Path(data_dir) / f'{name}.npz'):
            tr['feat'] = rm.features(tr, tr['side'])
            tr['target'] = rt.build_targets(tr, rigs[tr['side']])
            attach_teacher(tr, teacher)
            tr['recording'] = name
            tracks.append(tr)
    return tracks


def corrupt(feat, lower, upper, rng):
    """Synthetic input corruption on a window (T,FEAT) copy. Returns feat, mask (T,5) of corrupted fingers."""
    f = feat.copy(); T = f.shape[0]; mask = np.zeros((T, 5), bool)
    # finger dropout spans
    for _ in range(rng.integers(0, 3)):
        fi = int(rng.integers(5)); s = int(rng.integers(0, max(1, T - 3))); L = int(rng.integers(3, 12)); e = min(T, s + L)
        joints = list(range(4 * fi, 4 * fi + 4))
        mode = rng.integers(3)
        if mode == 0:      # freeze
            f[s:e, joints] = f[s, joints]
        elif mode == 1:    # drift toward a limit
            tgt = np.where(rng.random(4) < 0.5, lower[joints], upper[joints])
            for k in range(s, e):
                a = min(1.0, (k - s + 1) / 6.0); f[k, joints] = f[k, joints] * (1 - a) + tgt * a
        else:              # large noise
            f[s:e, joints] += rng.normal(0, 0.25, (e - s, 4))
        f[s:e, 51 + 2 * fi] = rng.uniform(0.2, 0.6); f[s:e, 52 + 2 * fi] = rng.uniform(0.4, 1.0)
        mask[s:e, fi] = True
    # global noise
    f[:, 0:20] += rng.normal(0, 0.02, (T, 20)); f[:, 24:27] += rng.normal(0, 0.002, (T, 3))
    # missed frames: hold previous input
    for _ in range(rng.integers(0, 3)):
        s = int(rng.integers(1, T)); L = int(rng.integers(1, 4)); e = min(T, s + L)
        f[s:e, :] = f[s - 1, :]; f[s:e, 63] = 0.0; f[s:e, 51:61:2] = 0.0; f[s:e, 52:61:2] = 1.0
    # recompute deltas from (possibly corrupted) angles/translation
    f[1:, 27:47] = np.diff(f[:, 0:20], axis=0); f[0, 27:47] = 0
    f[1:, 47:50] = np.diff(f[:, 24:27], axis=0); f[0, 47:50] = 0
    f[:, 0:20] = np.clip(f[:, 0:20], lower, upper)
    return f, mask


def batches(tracks, window, batch, rng, lower, upper, augment=True):
    while True:
        xs, tgt_a, tgt_q, tgt_t, wf, ww, masks, sides, teach, teach_w = [], [], [], [], [], [], [], [], [], []
        for _ in range(batch):
            tr = tracks[rng.integers(len(tracks))]
            n = len(tr['t'])
            if n <= window:
                s = 0; e = n
            else:
                s = int(rng.integers(0, n - window)); e = s + window
            f = tr['feat'][s:e]
            if augment:
                f, m = corrupt(f, lower[tr['side']], upper[tr['side']], rng)
            else:
                m = np.zeros((e - s, 5), bool)
            pad = window - (e - s)
            def P(a):
                return np.pad(a, ((0, pad),) + ((0, 0),) * (a.ndim - 1), mode='edge') if pad else a
            xs.append(P(f)); masks.append(P(m)); sides.append(tr['side'])
            t = tr['target']
            tgt_a.append(P(t['angles'][s:e])); tgt_q.append(P(t['q'][s:e])); tgt_t.append(P(t['tr'][s:e]))
            wf.append(P(t['w_finger'][s:e]) * (1 - pad / window if pad else 1)); ww.append(P(t['w_wrist'][s:e]))
            teach.append(P(tr['teacher'][s:e])); tw = P(tr['teacher_w'][s:e]).copy()
            if pad:
                tw[-pad:] = 0
            teach_w.append(tw)
        yield (np.stack(xs), np.stack(tgt_a), np.stack(tgt_q), np.stack(tgt_t), np.stack(wf), np.stack(ww), np.stack(masks), np.array(sides), np.stack(teach), np.stack(teach_w))


def fk_batch(fks, angles, q, tr_m, sides):
    """Per-side FK on a batch (B,T,...) -> (B,T,21,3) mm."""
    import torch
    out = torch.zeros(*angles.shape[:2], 21, 3, dtype=angles.dtype, device=angles.device)
    for side in (0, 1):
        idx = (sides == side).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        B, T = len(idx), angles.shape[1]
        pts = fks[side].landmarks(angles[idx].reshape(B * T, 20), q[idx].reshape(B * T, 4), tr_m[idx].reshape(B * T, 3) * 1000.0, side)
        out[idx] = pts.reshape(B, T, 21, 3)
    return out


def finger_weights_to_landmarks(wf):
    """(B,T,5) -> (B,T,21) weights on rig landmarks."""
    import torch
    w = torch.zeros(*wf.shape[:2], 21, dtype=wf.dtype, device=wf.device)
    for f, lms in enumerate(LM_FINGER):
        for l in lms:
            w[..., l] = wf[..., f]
    w[..., 20] = wf.mean(-1); w[..., 5] = wf.mean(-1)
    return w


def evaluate(model, tracks, fks, lower, upper, device, rng):
    """Returns metrics for raw, baseline filter and refiner on the given tracks (no augmentation, plus a
    corrupted copy for infill)."""
    import torch
    model.eval()
    res = {k: dict(jitter=[], err=[], infill=[], terr=[], tinfill=[]) for k in ('raw', 'baseline', 'refiner', 'target', 'chain')}
    lags = []
    with torch.no_grad():
        for tr in tracks:
            side = tr['side']; n = len(tr['t'])
            if n < 12:
                continue
            x = torch.tensor(tr['feat'][None], device=device)
            out, _ = model(x)
            ang_in = x[..., 0:20]; q_in = x[..., 20:24]; tr_in = x[..., 24:27]
            ang_r, q_r, tr_r = rm.apply_correction(ang_in, q_in, tr_in, out, lower[side], upper[side])
            sides = torch.tensor([side], device=device)
            t = tr['target']
            tgt_pts = fk_batch(fks, torch.tensor(t['angles'][None], device=device, dtype=torch.float64), torch.tensor(t['q'][None], device=device, dtype=torch.float64), torch.tensor(t['tr'][None], device=device, dtype=torch.float64), sides)
            wl = finger_weights_to_landmarks(torch.tensor(t['w_finger'][None], device=device, dtype=torch.float64))
            reliable = (wl >= 1.0)
            # baseline filter
            bq, bt, ba = run_baseline_on_track(tr['t'], tr['q'], tr['tr'], tr['angles'])
            okb = ~np.isnan(ba[:, 0])
            variants = dict(raw=(ang_in.double(), q_in.double(), tr_in.double()), refiner=(ang_r.double(), q_r.double(), tr_r.double()),
                            target=(torch.tensor(t['angles'][None], device=device, dtype=torch.float64), torch.tensor(t['q'][None], device=device, dtype=torch.float64), torch.tensor(t['tr'][None], device=device, dtype=torch.float64)))
            if okb.sum() > 10:
                variants['baseline'] = (torch.tensor(np.where(okb[:, None], ba, tr['angles'])[None], device=device, dtype=torch.float64),
                                        torch.tensor(np.where(okb[:, None], bq, tr['q'])[None], device=device, dtype=torch.float64),
                                        torch.tensor(np.where(okb[:, None], bt, tr['tr'])[None] / 1000.0, device=device, dtype=torch.float64))
            # chain: refiner output through the same processor filter (where it would sit live)
            ar, qr, trr = ang_r[0].cpu().numpy().astype(np.float64), q_r[0].cpu().numpy().astype(np.float64), tr_r[0].cpu().numpy().astype(np.float64) * 1000.0
            cq, ct, ca = run_baseline_on_track(tr['t'], qr, trr, ar)
            okc = ~np.isnan(ca[:, 0])
            if okc.sum() > 10:
                variants['chain'] = (torch.tensor(np.where(okc[:, None], ca, ar)[None], device=device, dtype=torch.float64),
                                     torch.tensor(np.where(okc[:, None], cq, qr)[None], device=device, dtype=torch.float64),
                                     torch.tensor(np.where(okc[:, None], ct, trr)[None] / 1000.0, device=device, dtype=torch.float64))
            teacher = torch.tensor(tr['teacher'][None], device=device, dtype=torch.float64)
            tmask = torch.tensor(tr['teacher_w'][None], device=device) > 0
            for name, (a_, q_, t_) in variants.items():
                pts = fk_batch(fks, a_, q_, t_, sides)
                acc = pts[:, 2:] - 2 * pts[:, 1:-1] + pts[:, :-2]
                obs = torch.tensor(tr['observed'][None], device=device)
                m = obs[:, 2:] & obs[:, 1:-1] & obs[:, :-2]
                res[name]['jitter'].append(torch.linalg.norm(acc, dim=-1)[m].mean().item() if m.any() else np.nan)
                err = torch.linalg.norm(pts - tgt_pts, dim=-1)
                res[name]['err'].append(err[reliable].mean().item() if reliable.any() else np.nan)
                if tmask.any():
                    res[name]['terr'].append(torch.linalg.norm(pts - teacher, dim=-1)[tmask].mean().item())
            # infill: corrupt one finger span, measure refiner error vs target on corrupted frames
            f, mask = corrupt(tr['feat'], lower[side].cpu().numpy(), upper[side].cpu().numpy(), rng)
            xc = torch.tensor(f[None], device=device)
            outc, _ = model(xc)
            ang_c, q_c, tr_c = rm.apply_correction(xc[..., 0:20], xc[..., 20:24], xc[..., 24:27], outc, lower[side], upper[side])
            pts_c = fk_batch(fks, ang_c.double(), q_c.double(), tr_c.double(), sides)
            pts_raw_c = fk_batch(fks, xc[..., 0:20].double(), xc[..., 20:24].double(), xc[..., 24:27].double(), sides)
            ml = finger_weights_to_landmarks(torch.tensor(mask[None].astype(np.float64), device=device)) > 0
            ml = ml & reliable
            if ml.any():
                res['refiner']['infill'].append(torch.linalg.norm(pts_c - tgt_pts, dim=-1)[ml].mean().item())
                res['raw']['infill'].append(torch.linalg.norm(pts_raw_c - tgt_pts, dim=-1)[ml].mean().item())
            mlt = (finger_weights_to_landmarks(torch.tensor(mask[None].astype(np.float64), device=device)) > 0) & tmask
            if mlt.any():
                res['refiner']['tinfill'].append(torch.linalg.norm(pts_c - teacher, dim=-1)[mlt].mean().item())
                res['raw']['tinfill'].append(torch.linalg.norm(pts_raw_c - teacher, dim=-1)[mlt].mean().item())
            # lag: palm centroid speed cross-correlation, refiner vs raw
            p_raw = fk_batch(fks, ang_in.double(), q_in.double(), tr_in.double(), sides)[0, :, PALM].mean(1).cpu().numpy()
            p_ref = fk_batch(fks, ang_r.double(), q_r.double(), tr_r.double(), sides)[0, :, PALM].mean(1).cpu().numpy()
            s_raw = np.linalg.norm(np.diff(p_raw, axis=0), axis=1); s_ref = np.linalg.norm(np.diff(p_ref, axis=0), axis=1)
            if len(s_raw) > 30 and s_raw.std() > 0 and s_ref.std() > 0:
                best = None
                for lag in range(0, 4):
                    c = np.corrcoef(s_raw[:len(s_raw) - lag] if lag else s_raw, s_ref[lag:])[0, 1]
                    if best is None or c > best[1]:
                        best = (lag, c)
                lags.append(best[0] * 1000 / 30)
    summary = {k: {m: float(np.nanmean(v[m])) if v[m] else None for m in v} for k, v in res.items()}
    summary['refiner']['lag_ms_vs_raw_p50'] = float(np.median(lags)) if lags else None
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--profiles', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--train', nargs='*', default=['20261005T030445Z', '20261005T050930Z', '20261009T032721Z'])
    ap.add_argument('--dev', nargs='*', default=['20261009T040502Z'])
    ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--steps', type=int, default=150)
    ap.add_argument('--hidden', type=int, default=128); ap.add_argument('--window', type=int, default=64); ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--acc-weight', type=float, default=0.3)
    ap.add_argument('--res-weight', type=float, default=2.0); ap.add_argument('--infill-weight', type=float, default=2.0)
    ap.add_argument('--teacher', help='directory of refiner_teacher.py outputs; position target becomes UmeTrack landmarks')
    a = ap.parse_args()
    import torch
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rigs = {0: rd.load_fhrig(Path(a.profiles) / 'side0.fhrig'), 1: rd.load_fhrig(Path(a.profiles) / 'side1.fhrig')}
    fks = {s: rd.RigFK(rigs[s], device=device, dtype=torch.float64) for s in (0, 1)}
    lower = {s: torch.tensor(rigs[s]['lower'], device=device, dtype=torch.float32) for s in (0, 1)}
    upper = {s: torch.tensor(rigs[s]['upper'], device=device, dtype=torch.float32) for s in (0, 1)}
    train = load_split(a.data, a.train, rigs, a.teacher); dev = load_split(a.data, a.dev, rigs, a.teacher)
    cov = lambda ts: sum(int((t['teacher_w'].max(1) > 0).sum()) for t in ts)
    print(f'train tracks {len(train)} frames {sum(len(t["t"]) for t in train)} teacher frames {cov(train)}; dev tracks {len(dev)} frames {sum(len(t["t"]) for t in dev)} teacher frames {cov(dev)}', flush=True)
    model = rm.make_model(a.hidden).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * a.steps)
    rng = np.random.default_rng(0)
    gen = batches(train, a.window, a.batch, rng, {s: lower[s].cpu().numpy() for s in (0, 1)}, {s: upper[s].cpu().numpy() for s in (0, 1)})
    base = evaluate(model, dev, fks, lower, upper, device, np.random.default_rng(1))
    print('epoch 0 (identity):', json.dumps(base), flush=True)
    log = [dict(epoch=0, dev=base, params=nparams)]
    best = None
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.perf_counter(); tot = dict(pos=0.0, acc=0.0, n=0)
        for _ in range(a.steps):
            x, ta, tq, tt, wf, ww, mask, sides, teach, teach_w = next(gen)
            x = torch.tensor(x, device=device); sides_t = torch.tensor(sides, device=device)
            out_, _ = model(x)
            lo = torch.stack([lower[s] for s in sides]); up = torch.stack([upper[s] for s in sides])
            ang, q, tr = rm.apply_correction(x[..., 0:20], x[..., 20:24], x[..., 24:27], out_, lo[:, None], up[:, None])
            pts = fk_batch(fks, ang.double(), q.double(), tr.double(), sides_t)
            ml = finger_weights_to_landmarks(torch.tensor(mask.astype(np.float64), device=device, dtype=torch.float64))
            boost = 1.0 + a.infill_weight * ml                                # corrupted-finger frames count more
            if a.teacher:
                tgt = torch.tensor(teach, device=device, dtype=torch.float64)
                wl = torch.tensor(teach_w, device=device, dtype=torch.float64) * boost
            else:
                tgt = fk_batch(fks, torch.tensor(ta, device=device, dtype=torch.float64), torch.tensor(tq, device=device, dtype=torch.float64), torch.tensor(tt, device=device, dtype=torch.float64), sides_t)
                wl = finger_weights_to_landmarks(torch.tensor(wf, device=device, dtype=torch.float64)) * boost
            err = torch.linalg.norm(pts - tgt, dim=-1)                       # (B,T,21) mm
            lpos = (err * wl).sum() / wl.sum().clamp(min=1)
            acc = pts[:, 2:] - 2 * pts[:, 1:-1] + pts[:, :-2]
            lacc = torch.linalg.norm(acc, dim=-1).mean()
            # keep corrections small unless evidence justifies them (radians, metres*100)
            lres = out_[..., :20].abs().mean() + 100.0 * out_[..., 20:23].abs().mean() + out_[..., 23:26].abs().mean()
            loss = lpos + a.acc_weight * lacc + a.res_weight * lres
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot['pos'] += lpos.item(); tot['acc'] += lacc.item(); tot['n'] += 1
        m = evaluate(model, dev, fks, lower, upper, device, np.random.default_rng(1))
        rec = dict(epoch=ep, train_pos_mm=tot['pos'] / tot['n'], train_acc=tot['acc'] / tot['n'], seconds=time.perf_counter() - t0, dev=m)
        log.append(rec); print(json.dumps(rec), flush=True)
        torch.save(model.state_dict(), out / f'epoch-{ep:03d}.pt')
        key = 'terr' if a.teacher else 'err'
        score = (m['refiner'][key] or 0) + 0.5 * m['refiner']['jitter'] + 0.5 * (m['refiner']['tinfill' if a.teacher else 'infill'] or 0)
        if best is None or score < best:
            best = score; torch.save(model.state_dict(), out / 'best.pt'); (out / 'best.json').write_text(json.dumps(rec, indent=1))
        (out / 'log.json').write_text(json.dumps(log, indent=1))


if __name__ == '__main__':
    main()
