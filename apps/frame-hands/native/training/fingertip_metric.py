"""Perception-level wobble metric for rig traces: fingertip motion, which is what a wearer sees (wrist position
noise is a few mm; wrist ORIENTATION noise times the 15 cm lever to the fingertips is the visible wobble).

usage: python fingertip_metric.py <rig.txt> <profiles-dir> [--filter none|oneeuro|kalman] [params] [--label x]

Per side, over consecutive fitted frames of the same track (<= 70 ms apart), after the optional causal filter:
  tip step   : mean over the 5 fingertips of the frame-to-frame displacement (mm), median / p90 / p99
  split      : how much of the raw tip step comes from wrist translation, wrist rotation, finger angles (medians)
  still rms  : RMS fingertip deviation (mm) in 1 s windows where the raw wrist moved < 30 mm/s, median of windows
  motion lag : in frames with raw wrist speed > 0.2 m/s, mean distance between filtered and raw fingertips (mm) and
               the equivalent delay (lag / speed, ms); 'none' gives 0 by definition
Filters (all causal, per track, reset on a new track):
  oneeuro  --fc-min <Hz> --beta <Hz per unit/s> --dcut <Hz>   (position in m, rotation rad, angles rad)
  kalman   --acc <m/s^2 process std> --meas <mm std>          constant-velocity Kalman on wrist position (and the
                                                              same per joint angle in rad with --acc-ang/--meas-ang),
                                                              orientation via oneeuro
"""
import argparse, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training'))
from refiner_data import RigFK, load_fhrig, parse_rig_trace  # noqa: E402

TIPS = [0, 1, 2, 3, 4]   # rig indices of the fingertips (thumb..pinky)


def qmul(p, q):
    w1, x1, y1, z1 = p; w2, x2, y2, z2 = q
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qinv(q): return np.array([q[0], -q[1], -q[2], -q[3]])


def qlog(q):
    q = q / np.linalg.norm(q)
    if q[0] < 0: q = -q
    s = np.linalg.norm(q[1:])
    if s < 1e-12: return np.zeros(3)
    return q[1:] / s * (2 * math.atan2(s, q[0]))


def qexp(v):
    n = np.linalg.norm(v)
    if n < 1e-12: return np.array([1.0, 0, 0, 0])
    return np.concatenate([[math.cos(n / 2)], v / n * math.sin(n / 2)])


def alpha(cutoff, dt): return 1 / (1 + 1 / (2 * math.pi * cutoff * dt))


class OneEuro:
    def __init__(s, fc_min, beta, dcut): s.fc, s.beta, s.dcut = fc_min, beta, dcut; s.x = None; s.dx = 0.0

    def __call__(s, x, dt):
        if s.x is None: s.x = x.copy(); return x
        d = (x - s.x) / dt
        s.dx = s.dx + (np.linalg.norm(d) - s.dx) * alpha(s.dcut, dt)
        a = alpha(s.fc + s.beta * s.dx, dt)
        s.x = s.x + (x - s.x) * a
        return s.x.copy()


class OneEuroRot:
    def __init__(s, fc_min, beta, dcut): s.fc, s.beta, s.dcut = fc_min, beta, dcut; s.q = None; s.dw = 0.0

    def __call__(s, q, dt):
        if s.q is None: s.q = q.copy(); return q
        rel = qlog(qmul(q, qinv(s.q)))
        s.dw = s.dw + (np.linalg.norm(rel) / dt - s.dw) * alpha(s.dcut, dt)
        a = alpha(s.fc + s.beta * s.dw, dt)
        s.q = qmul(qexp(rel * a), s.q); s.q /= np.linalg.norm(s.q)
        return s.q.copy()


class KalmanCV:
    """Independent constant-velocity Kalman per component."""
    def __init__(s, acc_std, meas_std, n): s.qa, s.r, s.n = acc_std, meas_std, n; s.x = None

    def __call__(s, z, dt):
        if s.x is None:
            s.x = np.stack([z, np.zeros_like(z)]); s.P = np.array([[s.r ** 2, 0], [0, (s.r / 0.033) ** 2]]); return z
        F = np.array([[1, dt], [0, 1]]); G = np.array([dt * dt / 2, dt])
        Q = np.outer(G, G) * s.qa ** 2
        x = F @ s.x; P = F @ s.P @ F.T + Q
        S = P[0, 0] + s.r ** 2; K = P[:, 0] / S
        y = z - x[0]
        s.x = x + np.outer(K, y); s.P = P - np.outer(K, P[0, :])
        return s.x[0].copy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('rig'); ap.add_argument('profiles'); ap.add_argument('--label', default='')
    ap.add_argument('--filter', default='none'); ap.add_argument('--fc-min', type=float, default=1.0); ap.add_argument('--beta', type=float, default=20.0)
    ap.add_argument('--dcut', type=float, default=1.0); ap.add_argument('--fc-min-rot', type=float, default=None); ap.add_argument('--beta-rot', type=float, default=None)
    ap.add_argument('--fc-min-ang', type=float, default=None); ap.add_argument('--beta-ang', type=float, default=None)
    ap.add_argument('--acc', type=float, default=2.0); ap.add_argument('--meas', type=float, default=4.0)
    ap.add_argument('--acc-ang', type=float, default=20.0); ap.add_argument('--meas-ang', type=float, default=0.05)
    ap.add_argument('--still-speed', type=float, default=0.03); ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    import torch
    frames = parse_rig_trace(a.rig)
    fks = {s: RigFK(load_fhrig(Path(a.profiles) / f'side{s}.fhrig')) for s in (0, 1)}
    out_rows = []
    for side in (0, 1):
        seq = []   # (t_s, source, q, t_mm, angles)
        for stamp, hands in frames:
            for h in hands:
                if h['status'] == 'fitted' and h['side'] == side: seq.append((stamp / 1e9, h['source'], h['q'], h['t'], h['angles']))
        if len(seq) < 10: continue
        # causal filtering per track
        filt = []; state = None; last = None
        for t, src, q, tr, ang in seq:
            if last is None or src != last[1] or t - last[0] > 0.15:
                if a.filter == 'oneeuro':
                    state = (OneEuro(a.fc_min, a.beta, a.dcut), OneEuroRot(a.fc_min_rot or a.fc_min, a.beta_rot if a.beta_rot is not None else a.beta * 0.15, a.dcut),
                             OneEuro(a.fc_min_ang or a.fc_min, a.beta_ang if a.beta_ang is not None else a.beta * 0.025, a.dcut))
                elif a.filter == 'kalman':
                    state = (KalmanCV(a.acc * 1000, a.meas, 3), OneEuroRot(a.fc_min_rot or a.fc_min, a.beta_rot if a.beta_rot is not None else a.beta * 0.15, a.dcut),
                             KalmanCV(a.acc_ang, a.meas_ang, 20))
                dt = 1 / 30
            else: dt = t - last[0]
            if a.filter == 'none': filt.append((q, tr, ang))
            else: filt.append((state[1](q, dt), state[0](tr, dt), state[2](ang, dt)))
            last = (t, src)
        fk = fks[side]
        def tips(qs, ts, angs):
            pts = fk.landmarks(torch.tensor(np.array(angs)), torch.tensor(np.array(qs)), torch.tensor(np.array(ts)), side).numpy()
            return pts[:, TIPS, :]
        raw_q = [s[2] for s in seq]; raw_t = [s[3] for s in seq]; raw_a = [s[4] for s in seq]
        raw_tips = tips(raw_q, raw_t, raw_a); fil_tips = tips([f[0] for f in filt], [f[1] for f in filt], [f[2] for f in filt])
        # decomposition of the raw tip step: translation-only, rotation-only, angles-only
        n = len(seq); consec = np.array([i > 0 and seq[i][1] == seq[i - 1][1] and seq[i][0] - seq[i - 1][0] <= 0.07 for i in range(n)])
        idx = np.where(consec)[0]
        rot_only = tips([raw_q[i] for i in idx], [raw_t[i - 1] for i in idx], [raw_a[i - 1] for i in idx])
        ang_only = tips([raw_q[i - 1] for i in idx], [raw_t[i - 1] for i in idx], [raw_a[i] for i in idx])
        prev = raw_tips[idx - 1]
        step_raw = np.linalg.norm(raw_tips[idx] - prev, axis=2).mean(1)
        step_fil = np.linalg.norm(fil_tips[idx] - fil_tips[idx - 1], axis=2).mean(1)
        step_tr = np.linalg.norm(np.array([raw_t[i] - raw_t[i - 1] for i in idx]), axis=1)
        step_rot = np.linalg.norm(rot_only - prev, axis=2).mean(1)
        step_ang = np.linalg.norm(ang_only - prev, axis=2).mean(1)
        # still windows (raw wrist speed) and motion lag
        times = np.array([s[0] for s in seq]); wrist = np.array(raw_t)
        speed = np.full(n, np.nan); speed[idx] = step_tr / 1000 / np.array([seq[i][0] - seq[i - 1][0] for i in idx])
        still_rms = []
        t0 = times[0]
        while t0 < times[-1]:
            m = (times >= t0) & (times < t0 + 1.0) & consec
            if m.sum() >= 20 and np.nanmean(speed[m]) < a.still_speed:
                dev = fil_tips[m] - fil_tips[m].mean(0, keepdims=True)
                still_rms.append(float(np.sqrt((dev ** 2).sum(2).mean())))
            t0 += 0.5
        moving = idx[speed[idx] > 0.2]
        lag = np.linalg.norm(fil_tips[moving] - raw_tips[moving], axis=2).mean(1) if len(moving) else np.array([np.nan])
        lag_ms = (lag / 1000) / speed[moving] * 1000 if len(moving) else np.array([np.nan])
        # roughness: second difference of the fingertips (mm per frame^2); smooth real motion is small, jitter is not
        idx2 = np.array([i for i in idx if i - 1 in set(idx)]) if len(idx) else np.array([], int)
        rough_fil = np.linalg.norm(fil_tips[idx2] - 2 * fil_tips[idx2 - 1] + fil_tips[idx2 - 2], axis=2).mean(1) if len(idx2) else np.array([np.nan])
        rough_raw = np.linalg.norm(raw_tips[idx2] - 2 * raw_tips[idx2 - 1] + raw_tips[idx2 - 2], axis=2).mean(1) if len(idx2) else np.array([np.nan])
        pct = lambda x, p: float(np.percentile(x, p)) if len(x) else float('nan')
        row = dict(side=side, n=len(idx), step_med=pct(step_fil, 50), step_p90=pct(step_fil, 90), step_p99=pct(step_fil, 99),
                   raw_med=pct(step_raw, 50), raw_p90=pct(step_raw, 90), tr=pct(step_tr, 50), rot=pct(step_rot, 50), ang=pct(step_ang, 50),
                   still=float(np.median(still_rms)) if still_rms else float('nan'), still_n=len(still_rms),
                   rough=pct(rough_fil, 50), rough_p90=pct(rough_fil, 90), rough_raw=pct(rough_raw, 50), rough_raw_p90=pct(rough_raw, 90),
                   lag=float(np.nanmean(lag)), lag_ms=float(np.nanmedian(lag_ms)), moving_n=len(moving))
        out_rows.append(row)
        if not a.quiet:
            print(f"{a.label or Path(a.rig).parent.name} side {side}: {len(idx)} steps")
            print(f"  tip step mm  median {row['step_med']:.2f}  p90 {row['step_p90']:.1f}  p99 {row['step_p99']:.0f}   (raw {row['raw_med']:.2f} / {row['raw_p90']:.1f})")
            print(f"  raw split medians: translation {row['tr']:.2f}  rotation {row['rot']:.2f}  angles {row['ang']:.2f} mm")
            print(f"  roughness (2nd diff) median {row['rough']:.2f} p90 {row['rough_p90']:.1f} mm  (raw {row['rough_raw']:.2f} / {row['rough_raw_p90']:.1f})")
            print(f"  still tip rms {row['still']:.2f} mm over {row['still_n']} windows;  motion lag {row['lag']:.1f} mm = {row['lag_ms']:.0f} ms ({row['moving_n']} moving frames)")
    if a.quiet:
        both = {k: np.nanmean([r[k] for r in out_rows]) for k in ('step_med', 'step_p90', 'step_p99', 'still', 'rough', 'rough_p90', 'lag', 'lag_ms')}
        print(f"{a.label:34s} step {both['step_med']:5.2f}/{both['step_p90']:5.1f}/{both['step_p99']:4.0f}  rough {both['rough']:5.2f}/{both['rough_p90']:5.1f}  still {both['still']:5.2f}  lag {both['lag']:5.1f}mm {both['lag_ms']:4.0f}ms")


if __name__ == '__main__': main()
