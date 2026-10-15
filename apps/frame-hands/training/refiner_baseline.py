"""Python port of the accepted processor filter (processor/baseline_motion.gd, mode 'predict') for
offline comparison. Operates on one hand track: positions in metres, wrist quaternion (w,x,y,z),
20 angles. Observations arrive at their capture times; sampling happens at render ticks.
This reproduces observe() and sample() from the GDScript; not bit-exact (float64 vs Godot float32).
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation as R

HOLD_AGE, HIDE_AGE, PREDICT_SECONDS, MAX_PREDICT_METERS, MAX_PREDICT_RADIANS = 0.100, 0.170, 0.025, 0.025, 0.12


def alpha(cutoff, dt):
    return 1.0 / (1.0 + 1.0 / (2 * math.pi * cutoff * dt))


def _rot(q_wxyz):
    return R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])


def rotation_vector(q):
    """q as scipy Rotation; canonical (w>=0) axis*angle."""
    v = q.as_rotvec()
    return v


def _slerp(a, b, t):
    from scipy.spatial.transform import Slerp
    return Slerp([0, 1], R.concatenate([a, b]))([t])[0]


def _limit(v, m):
    n = np.linalg.norm(v)
    return v * (m / n) if n > m else v


class BaselineMotion:
    def __init__(self):
        self.active = False; self.stamp = -1.0; self.capture_local = -1.0
        self.pending = False

    def reset(self, time, capture, p, q, a):
        self.active = True; self.stamp = time; self.capture_local = capture
        self.position = p.copy(); self.orientation = q; self.angles = a.copy()
        self.raw_position = p.copy(); self.raw_orientation = q; self.raw_angles = a.copy()
        self.rendered_position = p.copy(); self.rendered_orientation = q; self.rendered_angles = a.copy()
        self.velocity = np.zeros(3); self.angular_velocity = np.zeros(3); self.derivatives = np.zeros(20); self.acceleration = 0.0
        self.pending = False

    def observe(self, time, capture, now, p, q, a):
        if now - capture > HIDE_AGE:
            return False
        if self.active and time <= self.stamp:
            return False
        dt = time - self.stamp
        if not self.active or dt > 0.150 or now - self.capture_local > HIDE_AGE:
            self.reset(time, capture, p, q, a); return True
        if dt < 0.001:
            return False
        distance = np.linalg.norm(p - self.raw_position)
        rel = (q * self.raw_orientation.inv()).as_rotvec()
        if distance > 0.10 + 3.5 * dt or np.linalg.norm(rel) > 1.4 + 8.0 * dt:
            if self.pending and np.linalg.norm(p - self.pending_position) < 0.060 and np.linalg.norm((q * self.pending_orientation.inv()).as_rotvec()) < 0.35:
                self.reset(time, capture, p, q, a); return True
            self.pending = True; self.pending_position = p.copy(); self.pending_orientation = q
            return False
        self.pending = False
        v = (p - self.raw_position) / dt
        omega = rel / dt
        da = alpha(6.0, dt)
        self.acceleration = np.linalg.norm(v - self.velocity) / dt
        if v @ self.velocity < 0 or np.linalg.norm(v) < 0.03:
            self.velocity = v
        else:
            self.velocity = self.velocity + (v - self.velocity) * da
        if omega @ self.angular_velocity < 0:
            self.angular_velocity = omega
        else:
            self.angular_velocity = self.angular_velocity + (omega - self.angular_velocity) * da
        self.velocity = _limit(self.velocity, 2.0); self.angular_velocity = _limit(self.angular_velocity, 6.0)
        pa = alpha(4.0 + 15.0 * np.linalg.norm(self.velocity), dt)
        qa = alpha(4.0 + 1.5 * np.linalg.norm(self.angular_velocity), dt)
        self.position = self.position + (p - self.position) * pa
        self.orientation = _slerp(self.orientation, q, qa)
        for i in range(20):
            d = (a[i] - self.raw_angles[i]) / dt
            self.derivatives[i] = self.derivatives[i] + (d - self.derivatives[i]) * da
            self.angles[i] = self.angles[i] + (a[i] - self.angles[i]) * alpha(5.0 + 0.8 * abs(self.derivatives[i]), dt)
        self.raw_position = p.copy(); self.raw_orientation = q; self.raw_angles = a.copy()
        self.stamp = time; self.capture_local = capture
        return True

    def sample(self, now, delta):
        if not self.active:
            return None
        age = max(0.0, now - self.capture_local)
        if age >= HIDE_AGE:
            self.active = False; return None
        horizon = min(age, PREDICT_SECONDS) / (1.0 + self.acceleration / 40.0)
        target_p = self.position + _limit(self.velocity * horizon, MAX_PREDICT_METERS)
        rot = _limit(self.angular_velocity * horizon, MAX_PREDICT_RADIANS)
        target_q = (R.from_rotvec(rot) * self.orientation) if np.linalg.norm(rot) > 1e-6 else self.orientation
        blend = 1.0 - math.exp(-max(delta, 1e-5) / 0.012)
        self.rendered_position = self.rendered_position + (target_p - self.rendered_position) * blend
        self.rendered_orientation = _slerp(self.rendered_orientation, target_q, blend)
        self.rendered_angles = self.rendered_angles + (self.angles - self.rendered_angles) * blend
        return dict(position=self.rendered_position.copy(), orientation=self.rendered_orientation, angles=self.rendered_angles.copy(), age=age)


def run_baseline_on_track(t, q_wxyz, tr_mm, angles, latency=0.025):
    """Feed observations in order (capture at t, received at t+latency) and sample at each observation's
    receipt time; returns per-observation filtered (q wxyz, t mm, angles) aligned with inputs (NaN where hidden)."""
    m = BaselineMotion()
    out_q = np.full((len(t), 4), np.nan); out_t = np.full((len(t), 3), np.nan); out_a = np.full((len(t), 20), np.nan)
    last = None
    for i in range(len(t)):
        now = t[i] + latency
        q = _rot(q_wxyz[i])
        m.observe(t[i], t[i], now, tr_mm[i] / 1000.0, q, angles[i])
        delta = (now - last) if last is not None else 1 / 90
        last = now
        s = m.sample(now, delta)
        if s:
            qq = s['orientation'].as_quat()
            out_q[i] = [qq[3], qq[0], qq[1], qq[2]]; out_t[i] = s['position'] * 1000; out_a[i] = s['angles']
    return out_q, out_t, out_a
