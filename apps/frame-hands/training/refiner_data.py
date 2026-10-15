"""Data for the learned causal pose refiner: rig profiles, forward kinematics, trace parsing, sequences.

Rig state comes from the native tracker's rig trace (tracker_rig_hook.h):
  frame <raw_ns> <hands>
  rig <source> <side> <status> [<cost> <rmse_px> qw qx qy qz tx ty tz a0..a19]   (fitted only)
Wrist is in the capture head frame, millimetres; angles in radians in the FHRIG joint order
(5 fingers x 4 joints, thumb first). Evidence per hand comes from the LIVE audit (views.txt):
per-view presence and the stereo residual of the two views' 21 landmarks.

Forward kinematics replicates native/hand_rig.h HandRig::landmarks exactly (verified by
check_fk.py against the native evaluator). Read-only on inputs.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import fh_common as fh

FINGER_LANDMARKS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]  # MediaPipe order
N_ANGLES = 20


def load_fhrig(path):
    """FHRIG 1 mm left <kind> <subject_side> <model_sha> <calib_sha> 20 21 17, then joints, landmarks+weights."""
    tokens = Path(path).read_text().split()
    if tokens[0] != 'FHRIG' or tokens[1] != '1' or tokens[2] != 'mm' or tokens[3] != 'left':
        raise ValueError('Unsupported rig header')
    kind, subject_side = tokens[4], int(tokens[5])
    nj, nl, nf = int(tokens[8]), int(tokens[9]), int(tokens[10])
    if (nj, nl, nf) != (20, 21, 17):
        raise ValueError('Unexpected rig sizes')
    vals = list(map(float, tokens[11:]))
    pos = 0
    axis = np.zeros((20, 3)); pivot = np.zeros((20, 3)); lower = np.zeros(20); upper = np.zeros(20)
    for j in range(20):
        axis[j] = vals[pos:pos + 3]; pivot[j] = vals[pos + 3:pos + 6]; lower[j] = vals[pos + 6]; upper[j] = vals[pos + 7]; pos += 8
    rest = np.zeros((21, 3)); weights = np.zeros((21, 17))
    for i in range(21):
        rest[i] = vals[pos:pos + 3]; pos += 3
        weights[i] = vals[pos:pos + 17]; pos += 17
    if pos != len(vals):
        raise ValueError('Trailing rig data')
    return dict(kind=kind, subject_side=subject_side, axis=axis, pivot=pivot, lower=lower, upper=upper, rest=rest, weights=weights)


# ------------------------------------------------------------------------------ FK (torch)

def _quat_mul(a, b):
    """Hamilton product, (..,4) wxyz."""
    aw, ax, ay, az = a.unbind(-1); bw, bx, by, bz = b.unbind(-1)
    return __import__('torch').stack((aw * bw - ax * bx - ay * by - az * bz,
                                      aw * bx + ax * bw + ay * bz - az * by,
                                      aw * by - ax * bz + ay * bw + az * bx,
                                      aw * bz + ax * by - ay * bx + az * bw), -1)


def _quat_rotate(q, v):
    """Rotate (..,3) vectors by (..,4) wxyz unit quaternions."""
    import torch
    qv = q[..., 1:]
    t = 2 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qv, t, dim=-1)


class RigFK:
    """Batched differentiable FK for one FHRIG profile. Angles (B,20) rad, wrist q (B,4) wxyz, t (B,3) mm -> (B,21,3) mm."""

    def __init__(self, rig, device='cpu', dtype=None):
        import torch
        self.torch = torch
        dtype = dtype or torch.float64
        self.axis = torch.as_tensor(rig['axis'], dtype=dtype, device=device)
        self.pivot = torch.as_tensor(rig['pivot'], dtype=dtype, device=device)
        self.rest = torch.as_tensor(rig['rest'], dtype=dtype, device=device)
        self.weights = torch.as_tensor(rig['weights'], dtype=dtype, device=device)
        self.lower = torch.as_tensor(rig['lower'], dtype=dtype, device=device)
        self.upper = torch.as_tensor(rig['upper'], dtype=dtype, device=device)
        self.subject_side = rig['subject_side']

    def frames(self, angles):
        """(B,20) -> 17 rigid frames: q (B,17,4), t (B,17,3). Frames 0,1 are identity."""
        torch = self.torch
        B = angles.shape[0]
        ident_q = torch.tensor([1.0, 0, 0, 0], dtype=angles.dtype, device=angles.device).expand(B, 4)
        zero_t = torch.zeros(B, 3, dtype=angles.dtype, device=angles.device)
        qs = [ident_q, ident_q]; ts = [zero_t, zero_t]
        for f in range(5):
            cq, ct = ident_q, zero_t
            for k in range(4):
                i = 4 * f + k
                half = angles[:, i:i + 1] * 0.5                      # axis is unit length
                s = torch.sin(half)
                q = torch.cat((torch.cos(half), self.axis[i] * s), -1)
                lt = self.pivot[i] - _quat_rotate(q, self.pivot[i].expand(B, 3))
                # cumulative.then(local): q = cq*q, t = cumulative.apply(local.t)
                nt = _quat_rotate(cq, lt) + ct
                nq = _quat_mul(cq, q)
                cq, ct = nq, nt
                if k >= 1:
                    qs.append(cq); ts.append(ct)
        return torch.stack(qs, 1), torch.stack(ts, 1)

    def landmarks(self, angles, wrist_q, wrist_t, side):
        torch = self.torch
        fq, ft = self.frames(angles)                                  # (B,17,4), (B,17,3)
        rest = self.rest[None, None]                                   # (1,1,21,3)
        pts = _quat_rotate(fq[:, :, None, :].expand(-1, -1, 21, -1), rest.expand(fq.shape[0], 17, 21, 3)) + ft[:, :, None, :]
        local = (pts * self.weights.T[None, :, :, None]).sum(1)      # (B,21,3)
        if side == 1:
            local = local * torch.tensor([-1.0, 1.0, 1.0], dtype=local.dtype, device=local.device)
        return _quat_rotate(wrist_q[:, None, :].expand(-1, 21, -1), local) + wrist_t[:, None, :]


# ------------------------------------------------------------------------------ traces

def parse_rig_trace(path):
    """-> list of (stamp_ns, [hand dict...]) in file order; hand: source, side, status, and for fitted: cost, rmse, q(4), t(3), angles(20)."""
    frames = []
    cur = None
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == 'frame':
            cur = (int(f[1]), []); frames.append(cur)
        elif f[0] == 'rig' and cur is not None:
            h = dict(source=int(f[1]), side=int(f[2]), status=f[3])
            if f[3] == 'fitted':
                v = np.array(list(map(float, f[4:])))
                h.update(cost=v[0], rmse=v[1], q=v[2:6], t=v[6:9], angles=v[9:29])
            cur[1].append(h)
    return frames


def hand_evidence(audit_set, source, cams):
    """Per-view presence and per-landmark stereo residual (m) for one hand id in one audit set."""
    views = [v for v in audit_set['views'] if v['id'] == source]
    presence = {v['camera']: v['presence'] for v in views}
    residual = np.full(21, np.nan)
    if len(views) >= 2 and all(v['camera'] in cams for v in views[:2]):
        a, b = views[0], views[1]
        ra, rb = cams[a['camera']].ray(a['pts']), cams[b['camera']].ray(b['pts'])
        for k in range(21):
            _, res = fh.triangulate([cams[a['camera']].origin, cams[b['camera']].origin], [ra[k], rb[k]], [a['presence'], b['presence']])
            residual[k] = res
    return presence, residual


def finger_evidence(presence, residual):
    """-> (10,) features: per finger [min view presence, residual mm (clipped, 50 if monocular)] ."""
    out = np.zeros(10)
    p = min(presence.values()) if presence else 0.0
    nviews = len(presence)
    for f, idx in enumerate(FINGER_LANDMARKS):
        r = residual[list(idx)]
        rmm = 50.0 if (nviews < 2 or np.isnan(r).all()) else float(np.nanmax(r) * 1000)
        out[2 * f] = p; out[2 * f + 1] = min(rmm, 50.0)
    return out


def build_sequences(trace_path, views_path, persist_dir, max_gap_frames=4):
    """Split each side's fitted hands into tracks (same source id, gaps <= max_gap filled by the previous frame
    and flagged). Returns list of dicts with arrays: t (s), angles (N,20), q (N,4), tr (N,3) mm, cost, rmse,
    evidence (N,10), observed (N,) bool, side, source."""
    frames = parse_rig_trace(trace_path)
    audit = fh.read_audit(views_path)
    cams = fh.load_calibration(Path(persist_dir))
    tracks = []
    current = {0: None, 1: None}
    last_index = {0: -1, 1: -1}
    for index, (stamp, hands) in enumerate(frames):
        for side in (0, 1):
            h = next((x for x in hands if x['side'] == side and x['status'] == 'fitted'), None)
            if h is None:
                continue
            a = audit.get(stamp)
            presence, residual = hand_evidence(a, h['source'], cams) if a else ({}, np.full(21, np.nan))
            row = dict(t=stamp / 1e9, angles=h['angles'], q=h['q'], tr=h['t'], cost=h['cost'], rmse=h['rmse'],
                       evidence=finger_evidence(presence, residual), observed=True, nviews=len(presence))
            tr = current[side]
            gap = index - last_index[side] - 1
            if tr is None or tr['source'] != h['source'] or gap > max_gap_frames:
                tr = dict(side=side, source=h['source'], rows=[])
                current[side] = tr; tracks.append(tr)
            else:
                for g in range(gap):      # fill short gaps with the last fit, flagged unobserved
                    filler = dict(tr['rows'][-1]); filler['observed'] = False
                    filler['t'] = tr['rows'][-1]['t'] + (row['t'] - tr['rows'][-1]['t']) * (g + 1) / (gap + 1)
                    filler['evidence'] = np.array([0.0, 50.0] * 5); filler['nviews'] = 0
                    tr['rows'].append(filler)
            tr['rows'].append(row); last_index[side] = index
    out = []
    for tr in tracks:
        rows = tr['rows']
        if len(rows) < 8:
            continue
        out.append(dict(side=tr['side'], source=tr['source'],
                        t=np.array([r['t'] for r in rows]), angles=np.stack([r['angles'] for r in rows]),
                        q=np.stack([r['q'] for r in rows]), tr=np.stack([r['tr'] for r in rows]),
                        cost=np.array([r['cost'] for r in rows]), rmse=np.array([r['rmse'] for r in rows]),
                        evidence=np.stack([r['evidence'] for r in rows]), observed=np.array([r['observed'] for r in rows]),
                        nviews=np.array([r['nviews'] for r in rows])))
    return out


def save_sequences(tracks, path):
    arrays = {}
    for i, tr in enumerate(tracks):
        for k, v in tr.items():
            if isinstance(v, np.ndarray):
                arrays[f'{i}/{k}'] = v
            else:
                arrays[f'{i}/{k}'] = np.array(v)
    np.savez_compressed(path, n=np.array(len(tracks)), **arrays)


def load_sequences(path):
    z = np.load(path)
    n = int(z['n'])
    out = []
    for i in range(n):
        tr = {k.split('/', 1)[1]: z[k] for k in z.files if k.startswith(f'{i}/')}
        tr['side'] = int(tr['side']); tr['source'] = int(tr['source'])
        out.append(tr)
    return out
