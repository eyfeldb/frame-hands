"""Summarize a rig trace (FRAME_HANDS_RIG_TRACE): status counts, mono fits, wrist stability.

usage: python rig_trace_stats.py <rig.txt> [<rig.txt> ...]

Per file: hand-updates by status; fits split into stereo/mono (the "mono <source> <initialized>" marker
written by the mono-fit build); per-track frame-to-frame wrist translation change (mm) for stereo and mono
fits; wrist jump when a track switches between stereo and mono; pixel RMSE medians.
"""
import sys
from collections import Counter, defaultdict

import numpy as np


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float('nan')


def summarize(path):
    status = Counter(); kind = Counter(); rmse = defaultdict(list)
    last = {}            # source -> (stamp, wrist, kind)
    deltas = defaultdict(list); switches = []; sw_depth = []; sw_lateral = []; sw_ratio = []
    ang_steps = defaultdict(list); rot_steps = defaultdict(list)   # mean |joint angle change| deg, wrist rotation step deg
    last_pose = {}                                                  # source -> (stamp, angles, quat)
    run_len = defaultdict(int)                 # source -> frames in the current kind
    sw_dir = defaultdict(list)                 # 'stereo->mono' / 'mono->stereo' -> (jump, signed along-ray, run length before)
    stamp = None; pending = []   # rig lines of the current frame (mono marker follows its rig line)
    frames = 0

    def flush():
        for src, side, st, vals, mono in pending:
            status[st] += 1
            if st != 'fitted':
                last.pop(src, None); continue
            k = 'mono' if mono else 'stereo'
            kind[k] += 1; rmse[k].append(vals[1])
            wrist = np.array(vals[6:9]); angles = np.array(vals[9:29]); quat = np.array(vals[2:6])
            if src in last_pose and src in last and stamp - last_pose[src][0] <= 70_000_000 and last[src][2] == k:
                ang_steps[k].append(float(np.degrees(np.abs(angles - last_pose[src][1]).mean())))
                dq = abs(float(np.dot(quat, last_pose[src][2])))
                rot_steps[k].append(float(np.degrees(2 * np.arccos(min(1.0, dq)))))
            last_pose[src] = (stamp, angles, quat)
            if src in last and stamp - last[src][0] <= 70_000_000:
                d = float(np.linalg.norm(wrist - last[src][1]))
                if last[src][2] == k:
                    deltas[k].append(d)
                else:
                    switches.append(d)
                    # Decompose along the viewing direction (head origin ~ cameras) vs lateral.
                    prev = last[src][1]; ray = prev / max(np.linalg.norm(prev), 1e-6)
                    along = float(np.dot(wrist - prev, ray)); sw_depth.append(abs(along))
                    sw_lateral.append(float(np.sqrt(max(d * d - along * along, 0))))
                    mono_pt, stereo_pt = (wrist, prev) if k == 'mono' else (prev, wrist)
                    sw_ratio.append(float(np.linalg.norm(mono_pt) / max(np.linalg.norm(stereo_pt), 1e-6)))
                    sw_dir[f'{last[src][2]}->{k}'].append((d, along, run_len[src]))
            if src in last and last[src][2] == k:
                run_len[src] += 1
            else:
                run_len[src] = 1
            last[src] = (stamp, wrist, k)

    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] == 'frame':
                flush(); pending = []; stamp = int(p[1]); frames += 1
            elif p[0] == 'rig':
                src, side, st = int(p[1]), int(p[2]), p[3]
                vals = [float(x) for x in p[4:]] if st == 'fitted' else []
                pending.append([src, side, st, vals, False])
            elif p[0] == 'mono':
                src = int(p[1])
                for e in pending:
                    if e[0] == src:
                        e[4] = True
    flush()
    total = sum(status.values())
    print(f'== {path}')
    print(f'frames {frames}, hand updates {total}: ' + ', '.join(f'{k} {v} ({100 * v / max(total, 1):.1f}%)' for k, v in status.most_common()))
    print(f'fits: stereo {kind["stereo"]}, mono {kind["mono"]}')
    for k in ('stereo', 'mono'):
        if kind[k]:
            print(f'  {k}: rmse px median {pct(rmse[k], 50):.2f} 90% {pct(rmse[k], 90):.2f}; wrist step mm median {pct(deltas[k], 50):.1f} 90% {pct(deltas[k], 90):.1f} (n={len(deltas[k])}); '
                  f'joint angle step deg median {pct(ang_steps[k], 50):.2f} 90% {pct(ang_steps[k], 90):.2f}; wrist rotation step deg median {pct(rot_steps[k], 50):.2f} 90% {pct(rot_steps[k], 90):.2f}')
    if switches:
        print(f'  stereo<->mono switches {len(switches)}: wrist jump mm median {pct(switches, 50):.1f} 90% {pct(switches, 90):.1f}; '
              f'along-ray median {pct(sw_depth, 50):.1f} lateral median {pct(sw_lateral, 50):.1f}; '
              f'mono/stereo distance ratio 10% {pct(sw_ratio, 10):.2f} median {pct(sw_ratio, 50):.2f} 90% {pct(sw_ratio, 90):.2f}')
        for name, rows in sorted(sw_dir.items()):
            a = np.array(rows)
            short = a[a[:, 2] <= 3]; long_ = a[a[:, 2] > 3]
            print(f'    {name}: n={len(a)} jump median {pct(a[:, 0], 50):.1f}; signed along-ray (+ = farther) median {pct(a[:, 1], 50):+.1f} '
                  f'10% {pct(a[:, 1], 10):+.1f} 90% {pct(a[:, 1], 90):+.1f}; previous run frames median {pct(a[:, 2], 50):.0f}; '
                  f'runs<=3 frames: n={len(short)} jump median {pct(short[:, 0], 50) if len(short) else float("nan"):.1f}; '
                  f'runs>3: n={len(long_)} jump median {pct(long_[:, 0], 50) if len(long_) else float("nan"):.1f}')


if __name__ == '__main__':
    for p in sys.argv[1:]:
        summarize(p)
