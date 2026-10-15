"""Causal pose refiner: a small GRU over per-frame rig state + evidence, predicting a correction.

Input features per frame (dim 70):
  angles (20) | wrist quaternion wxyz canonical (4) | wrist translation m (3) | d_angles from previous input (20)
  | d_translation m (3) | dt s (1) | finger evidence (10: per finger min presence, stereo residual mm/50)
  | fit cost/100 (1) | fit rmse px/10 (1) | observed flag (1) | side (1) | wrist angular velocity rotvec (3) | pad (2)
Output (26): angle residual (20), translation residual m (3), rotation residual rotvec (3) applied as
  angles' = clamp(angles + da), t' = t + dt_, q' = exp(dr) * q.
The GRU hidden state carries across frames; ncnn supports GRU with explicit hidden in/out.
"""
import math

import numpy as np

FEAT = 70
OUT = 26


def canonical(q):
    q = np.asarray(q, float)
    return -q if q[0] < 0 else q


def quat_mul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def rotvec_from_quat(q):
    q = canonical(q / np.linalg.norm(q))
    s = np.linalg.norm(q[1:])
    if s < 1e-9:
        return np.zeros(3)
    return q[1:] / s * (2 * math.atan2(s, q[0]))


def features(track, side):
    """track dict (angles, q, tr mm, t, evidence, cost, rmse, observed) -> (N, FEAT) float32."""
    n = len(track['t'])
    ang = track['angles']; q = np.array([canonical(x) for x in track['q']]); tr = track['tr'] / 1000.0
    f = np.zeros((n, FEAT), np.float32)
    f[:, 0:20] = ang; f[:, 20:24] = q; f[:, 24:27] = tr
    f[1:, 27:47] = np.diff(ang, axis=0); f[1:, 47:50] = np.diff(tr, axis=0)
    dt = np.diff(track['t']); f[1:, 50] = dt; f[0, 50] = 1 / 30
    ev = track['evidence'].copy(); ev[:, 1::2] /= 50.0; f[:, 51:61] = ev
    f[:, 61] = np.minimum(track['cost'], 300) / 100.0; f[:, 62] = np.minimum(track['rmse'], 30) / 10.0
    f[:, 63] = track['observed'].astype(np.float32); f[:, 64] = float(side)
    for i in range(1, n):
        f[i, 65:68] = rotvec_from_quat(quat_mul(q[i], quat_conj(q[i - 1])))
    return f


def make_model(hidden=128):
    import torch
    from torch import nn

    class Refiner(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(FEAT, hidden), nn.ReLU())
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, OUT))
            nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)   # start as identity

        def forward(self, x, h=None):
            z = self.inp(x)
            y, h2 = self.gru(z, h)
            return self.out(y), h2
    return Refiner()


def apply_correction(angles, q, tr_m, out, lower, upper):
    """Apply model output (torch tensors, (...,26)) to inputs; returns refined angles, q (wxyz), tr (m)."""
    import torch
    da, dtr, dr = out[..., :20], out[..., 20:23], out[..., 23:26]
    ang = torch.minimum(torch.maximum(angles + da, lower), upper)
    tr = tr_m + dtr
    theta = torch.linalg.norm(dr, dim=-1, keepdim=True)
    axis = dr / torch.clamp(theta, min=1e-9)
    dq = torch.cat((torch.cos(theta / 2), axis * torch.sin(theta / 2)), -1)
    w1, x1, y1, z1 = dq.unbind(-1); w2, x2, y2, z2 = q.unbind(-1)
    qn = torch.stack((w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2), -1)
    qn = qn / torch.linalg.norm(qn, dim=-1, keepdim=True)
    return ang, qn, tr
