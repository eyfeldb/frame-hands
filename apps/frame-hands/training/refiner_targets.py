"""Clean target trajectories for refiner training: non-causal smoothing with evidence gating.

For each track: frames are 'reliable' when observed, stereo (both views), per-finger residual small,
fit rmse reasonable and angles not pinned at joint limits. Per finger, unreliable spans are removed and
re-interpolated from reliable neighbours (bounded span); then a zero-phase Gaussian smoother (sigma in
frames) is applied to angles and translation, and a windowed average to the wrist rotation (via rotation
vectors relative to the window centre). Returns target angles/q/tr plus per-finger reliability weights.
These are self-derived estimates, not ground truth; they encode 'what a smooth, stereo-consistent
trajectory through the reliable frames looks like'.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from refiner_model import canonical, quat_mul, quat_conj, rotvec_from_quat

FINGER_JOINTS = [list(range(4 * f, 4 * f + 4)) for f in range(5)]


def _interp_spans(values, ok, max_span=8):
    """values (N,k); ok (N,) bool. Linear interpolate short unreliable spans; returns values, weight (N,)."""
    v = values.copy(); w = ok.astype(float)
    n = len(ok)
    i = 0
    while i < n:
        if ok[i]:
            i += 1; continue
        j = i
        while j < n and not ok[j]:
            j += 1
        span = j - i
        if i > 0 and j < n and span <= max_span:
            for k in range(i, j):
                a = (k - (i - 1)) / (j - (i - 1))
                v[k] = values[i - 1] * (1 - a) + values[j] * a
            w[i:j] = 0.5
        else:
            # edge or long span: hold the nearest reliable value, low weight
            src = i - 1 if i > 0 else (j if j < n else None)
            if src is not None:
                v[i:j] = values[src]
            w[i:j] = 0.1
        i = j
    return v, w


def build_targets(track, rig, sigma=1.5, residual_mm=12.0, rmse_px=8.0):
    n = len(track['t'])
    ang = track['angles']; tr = track['tr'] / 1000.0; q = np.array([canonical(x) for x in track['q']])
    ev = track['evidence']
    observed = track['observed']
    stereo = track['nviews'] >= 2
    fit_ok = observed & stereo & (track['rmse'] < rmse_px)
    at_limit = (np.abs(ang - rig['lower'][None]) < 1e-6) | (np.abs(ang - rig['upper'][None]) < 1e-6)
    tgt_ang = np.zeros_like(ang); w_finger = np.zeros((n, 5))
    for f, joints in enumerate(FINGER_JOINTS):
        ok = fit_ok & (ev[:, 2 * f + 1] < residual_mm) & (ev[:, 2 * f] > 0.7) & (at_limit[:, joints].sum(1) < 3)
        v, w = _interp_spans(ang[:, joints], ok)
        tgt_ang[:, joints] = gaussian_filter1d(v, sigma, axis=0, mode='nearest')
        w_finger[:, f] = w
    # wrist: reliable when fit ok and palm fingers (index..pinky) have small residuals
    ok_w = fit_ok & (ev[:, 3::2].min(1) < residual_mm * 1.5)
    v_tr, w_tr = _interp_spans(tr, ok_w)
    tgt_tr = gaussian_filter1d(v_tr, sigma, axis=0, mode='nearest')
    # rotation: smooth rotation vectors relative to a running reference
    rv = np.zeros((n, 3))
    for i in range(n):
        rv[i] = rotvec_from_quat(quat_mul(q[i], quat_conj(q[0])))
    v_rv, _ = _interp_spans(rv, ok_w)
    s_rv = gaussian_filter1d(v_rv, sigma, axis=0, mode='nearest')
    tgt_q = np.zeros((n, 4))
    for i in range(n):
        th = np.linalg.norm(s_rv[i]); ax = s_rv[i] / th if th > 1e-9 else np.array([1.0, 0, 0])
        dq = np.array([np.cos(th / 2), *(ax * np.sin(th / 2))])
        tgt_q[i] = canonical(quat_mul(dq, q[0]))
    tgt_ang = np.minimum(np.maximum(tgt_ang, rig['lower'][None]), rig['upper'][None])
    return dict(angles=tgt_ang, q=tgt_q, tr=tgt_tr, w_finger=w_finger, w_wrist=w_tr)
