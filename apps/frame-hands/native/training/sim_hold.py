"""Offline port of the hold v3 presentation stabiliser (processor/baseline_motion.gd) applied to a rig trace.

usage: python sim_hold.py <rig.txt> [--radius 0.025] [--tol 0.20] [--persist 5] [--per-second]

For each side: raw fit versus shown output, frame to frame: wrist step (mm) and the largest joint change (deg),
with the fraction of frames that moved (> 1 mm / > 1 deg), plus how far the shown pose lags the raw one
(wrist mm, joint deg; p95 and max). A stationary hand should give near-zero shown change; a moving hand should
show a lag that stays small. --per-second prints the moving fractions second by second (raw vs shown).
"""
import argparse, math, sys
from collections import defaultdict
import numpy as np

a = argparse.ArgumentParser(); a.add_argument('trace'); a.add_argument('--radius', type=float, default=0.025)
a.add_argument('--noise', type=float, default=0.005); a.add_argument('--jump', type=float, default=0.035)
a.add_argument('--catch', type=float, default=0.04); a.add_argument('--rest-speed', type=float, default=0.05); a.add_argument('--rest-time', type=float, default=0.12)
a.add_argument('--angle', type=float, default=0.10); a.add_argument('--noise-turn', type=float, default=0.02); a.add_argument('--rest-turn', type=float, default=0.3)
a.add_argument('--tol', type=float, default=0.20); a.add_argument('--joint-noise', type=float, default=0.05)
a.add_argument('--joint-catch', type=float, default=0.04)
a.add_argument('--rate', type=float, default=20.0, help='max shown joint speed rad/s (0 = none)')
a.add_argument('--angle-cutoff', type=float, default=4.0)
a.add_argument('--per-second', action='store_true'); a.add_argument('--no-hold', action='store_true')
A = a.parse_args()


def qmul(p, q):
    w1, x1, y1, z1 = p; w2, x2, y2, z2 = q
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qinv(q): return np.array([q[0], -q[1], -q[2], -q[3]])


def rotvec(q):
    q = q / np.linalg.norm(q)
    if q[0] < 0: q = -q
    ang = 2 * math.acos(min(1.0, q[0]))
    if ang < 1e-6: return np.zeros(3)
    return q[1:] / max(1e-12, math.sin(ang / 2)) * ang


def slerp(p, q, t):
    d = float(np.dot(p, q))
    if d < 0: q = -q; d = -d
    if d > 0.9995: r = p + t * (q - p); return r / np.linalg.norm(r)
    th = math.acos(d); return (math.sin((1 - t) * th) * p + math.sin(t * th) * q) / math.sin(th)


def alpha(cutoff, dt): return 1 / (1 + 1 / (2 * math.pi * cutoff * dt))


class Hand:
    def __init__(s): s.active = False

    def reset(s, sid, t, p, q, ang):
        s.active = True; s.sid = sid; s.stamp = t; s.position = p.copy(); s.orientation = q.copy(); s.angles = ang.copy()
        s.raw_p = p.copy(); s.raw_q = q.copy(); s.raw_a = ang.copy(); s.velocity = np.zeros(3); s.ang_vel = np.zeros(3)
        s.deriv = np.zeros(20); s.hold_p = p.copy(); s.hold_q = q.copy(); s.stab_p = p.copy(); s.stab_q = q.copy()
        s.moving = s.turning = s.catching = s.turn_catching = False; s.speed = s.aspeed = 0.0; s.rest = s.trest = 0.0
        s.jhold = ang.copy(); s.jcount = np.zeros(20, int); s.jfree = np.zeros(20, int)
        s.out = s.tout = 0; s.jump_pending = False; s.prev_step = np.zeros(3); s.prev_turn = np.zeros(3)
        s.prev_raw = ang.copy(); s.prev_delta = np.zeros(20)

    def stabilise(s, p, q, dt):
        # hold v4: two consecutive same-direction changes above the noise size release; jumps need confirmation
        blend = min(1.0, dt / 0.08)
        step = p - s.raw_p
        s.speed += (np.linalg.norm(step) / dt - s.speed) * blend
        d = np.linalg.norm(p - s.hold_p)
        consistent = np.linalg.norm(step) > A.noise and np.linalg.norm(s.prev_step) > A.noise and np.dot(step, s.prev_step) > 0
        if s.moving:
            if s.jump_pending: s.jump_pending = False; s.moving = False; s.catching = True
            elif np.linalg.norm(step) > A.jump: s.jump_pending = True
            else:
                s.hold_p = p.copy()
                s.rest = s.rest + dt if s.speed < A.rest_speed else 0.0
                if s.rest >= A.rest_time: s.moving = False; s.rest = 0.0; s.out = 0
        if not s.moving:
            s.out = s.out + 1 if d > A.radius else 0
            if not s.catching and ((consistent and d > 2 * A.noise) or s.out >= 2): s.catching = True
            if s.catching:
                s.hold_p = s.hold_p + (p - s.hold_p) * (1 - math.exp(-dt / A.catch))
                if np.linalg.norm(p - s.hold_p) < A.noise: s.catching = False; s.moving = True; s.rest = 0.0; s.hold_p = p.copy()
        s.prev_step = step
        tv = rotvec(qmul(q, qinv(s.raw_q)))
        s.aspeed += (np.linalg.norm(tv) / dt - s.aspeed) * blend
        ang = np.linalg.norm(rotvec(qmul(q, qinv(s.hold_q))))
        tcons = np.linalg.norm(tv) > A.noise_turn and np.linalg.norm(s.prev_turn) > A.noise_turn and np.dot(tv, s.prev_turn) > 0
        if s.turning:
            s.hold_q = q.copy()
            s.trest = s.trest + dt if s.aspeed < A.rest_turn else 0.0
            if s.trest >= A.rest_time: s.turning = False; s.trest = 0.0; s.tout = 0
        else:
            s.tout = s.tout + 1 if ang > A.angle else 0
            if not s.turn_catching and ((tcons and ang > 2 * A.noise_turn) or s.tout >= 2): s.turn_catching = True
            if s.turn_catching:
                s.hold_q = slerp(s.hold_q, q, 1 - math.exp(-dt / A.catch))
                if np.linalg.norm(rotvec(qmul(q, qinv(s.hold_q)))) < A.noise_turn: s.turn_catching = False; s.turning = True; s.trest = 0.0; s.hold_q = q.copy()
        s.prev_turn = tv
        return s.hold_p.copy(), s.hold_q.copy()

    def stabilise_angles(s, ang, dt):
        k = 1 - math.exp(-dt / A.joint_catch); limit = A.rate * dt if A.rate > 0 else float('inf')
        for i in range(20):
            diff = ang[i] - s.jhold[i]; step = max(-limit, min(limit, diff * k))
            delta = ang[i] - s.prev_raw[i]
            consistent = abs(delta) > A.joint_noise and abs(s.prev_delta[i]) > A.joint_noise and delta * s.prev_delta[i] > 0
            s.jcount[i] = s.jcount[i] + 1 if abs(diff) > A.tol else 0
            if s.jfree[i] > 0:
                s.jhold[i] += step
                if abs(ang[i] - s.jhold[i]) < A.joint_noise: s.jfree[i] = 0
            elif (consistent and abs(diff) > 2 * A.joint_noise) or s.jcount[i] >= 2:
                s.jfree[i] = 1; s.jhold[i] += step
            s.prev_delta[i] = delta
        s.prev_raw = ang.copy()
        return s.jhold.copy()

    def observe(s, sid, t, p, q, ang):
        dt = t - s.stamp if s.active else 1.0
        if s.active and sid != s.sid and not A.no_hold and dt <= 0.150 and np.linalg.norm(p - s.raw_p) < 0.08: s.sid = sid
        if not s.active or sid != s.sid or dt > 0.150: s.reset(sid, t, p, q, ang); return True
        if dt < 0.001: return False
        raw_p, raw_q = p, q
        if not A.no_hold:
            p, q = s.stabilise(p, q, dt); ang = s.stabilise_angles(ang, dt)
        v = (p - s.stab_p) / dt; omega = rotvec(qmul(q, qinv(s.stab_q))) / dt; da = alpha(6.0, dt)
        s.velocity = v if (np.dot(v, s.velocity) < 0 or np.linalg.norm(v) < 0.03) else s.velocity + (v - s.velocity) * da
        s.ang_vel = omega if np.dot(omega, s.ang_vel) < 0 else s.ang_vel + (omega - s.ang_vel) * da
        if np.linalg.norm(s.velocity) > 2: s.velocity *= 2 / np.linalg.norm(s.velocity)
        if np.linalg.norm(s.ang_vel) > 6: s.ang_vel *= 6 / np.linalg.norm(s.ang_vel)
        s.position = s.position + (p - s.position) * alpha(4 + 15 * np.linalg.norm(s.velocity), dt)
        s.orientation = slerp(s.orientation, q, alpha(4 + 1.5 * np.linalg.norm(s.ang_vel), dt))
        for i in range(20):
            s.deriv[i] += ((ang[i] - s.raw_a[i]) / dt - s.deriv[i]) * da
            s.angles[i] += (ang[i] - s.angles[i]) * alpha(A.angle_cutoff + 0.3 * abs(s.deriv[i]), dt)
        s.raw_p = raw_p; s.raw_q = raw_q; s.stab_p = p; s.stab_q = q; s.raw_a = ang; s.stamp = t
        return True


rows = defaultdict(list); stamp = None
for line in open(A.trace):
    f = line.split()
    if f[0] == 'frame': stamp = int(f[1]); continue
    if f[0] != 'rig' or f[3] != 'fitted': continue
    v = [float(x) for x in f[4:]]
    rows[int(f[2])].append((stamp / 1e9, int(f[1]), np.array(v[6:9]) / 1000, np.array(v[2:6]), np.array(v[9:29])))

for side in sorted(rows):
    h = Hand(); raw_steps = []; shown_steps = []; raw_j = []; shown_j = []; lag_p = []; lag_a = []; per = defaultdict(lambda: [0, 0, 0, 0, 0])
    prev = None; t0 = rows[side][0][0]
    for t, sid, p, q, ang in rows[side]:
        fresh = h.observe(sid, t, p, q, ang)
        if prev is not None and prev[0] == sid and t - prev[1] <= 0.07 and not fresh_reset_last:
            raw_steps.append(np.linalg.norm(p - prev[2]) * 1000); shown_steps.append(np.linalg.norm(h.position - prev[3]) * 1000)
            raw_j.append(np.degrees(np.abs(ang - prev[4]).max())); shown_j.append(np.degrees(np.abs(h.angles - prev[5]).max()))
            lag_p.append(np.linalg.norm(h.position - p) * 1000); lag_a.append(np.degrees(np.abs(h.angles - ang).max()))
            b = per[int(t - t0)]; b[0] += 1; b[1] += raw_steps[-1] > 1; b[2] += shown_steps[-1] > 1; b[3] += raw_j[-1] > 1; b[4] += shown_j[-1] > 1
        fresh_reset_last = (h.stamp == t and np.array_equal(h.position, p) and not A.no_hold and h.speed == 0.0 and not h.moving and h.jcount.sum() == 0 and prev is not None and (prev[0] != sid or t - prev[1] > 0.150))
        prev = (sid, t, p, h.position.copy(), ang, h.angles.copy())
    n = len(raw_steps)
    if not n: print(f'side {side}: no consecutive fits'); continue
    rs, ss, rj, sj = map(np.array, (raw_steps, shown_steps, raw_j, shown_j)); lp, la = np.array(lag_p), np.array(lag_a)
    print(f'side {side}: {n} consecutive fits')
    print(f'  wrist step mm   raw: moved>1mm {np.mean(rs > 1):.0%}  p50 {np.median(rs):.1f} p90 {np.percentile(rs, 90):.1f} max {rs.max():.0f}'
          f'  | shown: moved>1mm {np.mean(ss > 1):.0%}  p50 {np.median(ss):.2f} p90 {np.percentile(ss, 90):.1f} max {ss.max():.0f}')
    print(f'  joint step deg  raw: changed>1deg {np.mean(rj > 1):.0%}  p50 {np.median(rj):.1f} p90 {np.percentile(rj, 90):.0f} max {rj.max():.0f}'
          f'  | shown: changed>1deg {np.mean(sj > 1):.0%}  p50 {np.median(sj):.2f} p90 {np.percentile(sj, 90):.1f} max {sj.max():.0f}')
    print(f'  shown-vs-raw lag: wrist p50 {np.median(lp):.1f} p95 {np.percentile(lp, 95):.0f} max {lp.max():.0f} mm;  joints p50 {np.median(la):.0f} p95 {np.percentile(la, 95):.0f} max {la.max():.0f} deg')
    if A.per_second:
        print('  sec: frames raw-wrist/shown-wrist raw-joint/shown-joint (counts of frames that moved)')
        print('  ' + '  '.join(f'{k}:{v[0]} {v[1]}/{v[2]} {v[3]}/{v[4]}' for k, v in sorted(per.items())))
