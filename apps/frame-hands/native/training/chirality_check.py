"""Handedness sanity of fitted hands in a rig trace: forward-kinematics the fit (angles + wrist) to 21 points and
checks on which side of the palm plane the thumb lies. A left rig can only produce a left-shaped hand unless the
joints are contorted to match a right hand seen in the images (wrong side assignment), which is what the game
shows as hooked/claw fingers. Reports the fraction of frames whose thumb side disagrees with the hand's side,
per side, and the clustered seconds.
usage: python chirality_check.py <rig.txt> <profiles-dir> [--label x]
"""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training'))
from refiner_data import RigFK, load_fhrig, parse_rig_trace  # noqa: E402

MP_TO_RIG = [5, -1, 6, 7, 0, 8, 9, 10, 1, 11, 12, 13, 2, 14, 15, 16, 3, 17, 18, 19, 4]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('rig'); ap.add_argument('profiles'); ap.add_argument('--label', default='')
    a = ap.parse_args()
    import torch
    frames = parse_rig_trace(a.rig); t0 = frames[0][0]
    fks = {s: RigFK(load_fhrig(Path(a.profiles) / f'side{s}.fhrig')) for s in (0, 1)}
    for side in (0, 1):
        rows = [(stamp, h) for stamp, hands in frames for h in hands if h['status'] == 'fitted' and h['side'] == side]
        if not rows: continue
        pts = fks[side].landmarks(torch.tensor(np.stack([h['angles'] for _, h in rows])), torch.tensor(np.stack([h['q'] for _, h in rows])),
                                  torch.tensor(np.stack([h['t'] for _, h in rows])), side).numpy()
        mp = np.stack([pts[:, r] if r >= 0 else np.full((len(rows), 3), np.nan) for r in MP_TO_RIG], 1)   # (N,21,3) mediapipe order
        w, i, p, m, t = mp[:, 0], mp[:, 5], mp[:, 17], mp[:, 9], mp[:, 4]
        n = np.cross(i - w, p - w); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        c = (n * (t - m)).sum(1)                    # thumb tip side relative to the palm plane
        wrong = (c > 0) if side == 0 else (c < 0)
        secs = np.array([(s - t0) / 1e9 for s, _ in rows])
        bad = sorted(set(int(x) for x in secs[wrong]))
        print(f"{a.label or Path(a.rig).parent.name} side {side} ({'L' if side == 0 else 'R'}): {len(rows)} fits, thumb on the wrong side of the palm in {wrong.mean():.1%}"
              f" (|c| median {np.median(np.abs(c)) * 1000:.0f} mm); seconds affected: {len(bad)}" + (f" e.g. {bad[:12]}" if bad else ''))


if __name__ == '__main__': main()
