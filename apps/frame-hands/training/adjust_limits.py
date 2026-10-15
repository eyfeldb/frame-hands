"""Widen or change joint limits of a rig profile (FHRIG). Joints 0-3 thumb (CMC a/b, MCP, IP), 4-7 index, 8-11
middle, 12-15 ring, 16-19 pinky; limits in radians. Writes an 'experimental' profile.
usage: python adjust_limits.py <in.fhrig> <out.fhrig> --joint 1 -1.1 1.1 [--joint 3 -0.6 1.1 ...] [--note text]
"""
import argparse
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('out')
ap.add_argument('--joint', nargs=3, action='append', required=True, metavar=('INDEX', 'LOWER', 'UPPER'))
a = ap.parse_args()
rows = Path(a.src).read_text().splitlines(); header = rows[0].split()
joints = np.array([list(map(float, r.split())) for r in rows[1:21]]); rest = rows[21:]
for idx, lo, hi in a.joint:
    k = int(idx); lo = float(lo); hi = float(hi)
    if not 0 <= k < 20 or lo > hi or abs(lo) > 6.28 or abs(hi) > 6.28: raise SystemExit('bad limit')
    print(f'joint {k}: [{np.degrees(joints[k, 6]):.0f}, {np.degrees(joints[k, 7]):.0f}] -> [{np.degrees(lo):.0f}, {np.degrees(hi):.0f}] deg')
    joints[k, 6] = lo; joints[k, 7] = hi
header[4] = 'experimental'
Path(a.out).write_text(' '.join(header) + '\n' + '\n'.join(' '.join(format(v, '.17g') for v in row) for row in joints) + '\n' + '\n'.join(rest) + '\n')
print('wrote', a.out)
