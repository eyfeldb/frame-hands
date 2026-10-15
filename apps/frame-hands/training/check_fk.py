"""Verify RigFK against the native evaluator (evaluate_hand_rig.cpp build) on random in-limit poses.

usage: python check_fk.py <profile.fhrig> <evaluate-rig.exe> <scratch-dir> [--side 0]
Input line format (verify_hand_rig.py): side qw qx qy qz tx ty tz a0..a19; output: side + 63 numbers.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import refiner_data as rd  # noqa: E402


def main():
    profile, exe, scratch = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    side = int(sys.argv[sys.argv.index('--side') + 1]) if '--side' in sys.argv else 0
    scratch.mkdir(parents=True, exist_ok=True)
    rig = rd.load_fhrig(profile)
    rng = np.random.default_rng(7)
    n = 256
    angles = rng.uniform(rig['lower'], rig['upper'], size=(n, 20))
    q = rng.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True); q[q[:, 0] < 0] *= -1
    t = rng.uniform(-600, 600, size=(n, 3))
    src = scratch / 'poses.txt'
    with src.open('w') as f:
        for i in range(n):
            f.write(' '.join([str(side)] + ['%.17g' % float(x) for x in (*q[i], *t[i], *angles[i])]) + '\n')
    out = scratch / 'native.txt'
    if out.exists():
        out.unlink()
    r = subprocess.run([str(exe), str(profile), str(src), str(out)], capture_output=True, text=True)
    if r.returncode:
        print('native evaluator failed:', r.stderr.strip()[:500]); sys.exit(1)
    native = np.loadtxt(out)[:, 1:].reshape(n, 21, 3)
    import torch
    fk = rd.RigFK(rig)
    with torch.no_grad():
        mine = fk.landmarks(torch.tensor(angles), torch.tensor(q), torch.tensor(t), side).numpy()
    err = np.abs(mine - native)
    print(f'profile {profile.name} side {side}: max abs diff {err.max():.3e} mm, mean {err.mean():.3e} mm over {n} poses')
    print('PASS' if err.max() < 1e-4 else 'FAIL')   # text round-trip of 17 significant digits; 1e-5 mm observed


if __name__ == '__main__':
    main()
