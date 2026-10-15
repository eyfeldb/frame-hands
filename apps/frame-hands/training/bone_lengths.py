"""Is the rig profile the wearer's hand size? Compare bone lengths in the rig's rest pose (side0/side1.fhrig,
mm) with the median stereo-triangulated bone lengths of the LIVE audit of a recording (clean two-camera views,
small pair residual). Reports per-bone lengths, the ratio measured/rig and its spread, per side.
usage: python bone_lengths.py <session-dir> <profiles-dir> [--max-residual 0.008] [--min-presence 0.9]
"""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import fh_common as fh  # noqa: E402

MP_TO_RIG = [5, -1, 6, 7, 0, 8, 9, 10, 1, 11, 12, 13, 2, 14, 15, 16, 3, 17, 18, 19, 4]
BONES = [(0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20), (2, 3), (3, 4), (5, 9), (9, 13), (13, 17)]
NAMES = ['palm-index', 'index-1', 'index-2', 'index-3', 'palm-middle', 'middle-1', 'middle-2', 'middle-3', 'palm-ring', 'ring-1',
         'ring-2', 'ring-3', 'palm-pinky', 'pinky-1', 'pinky-2', 'pinky-3', 'thumb-2', 'thumb-3', 'knuckle-im', 'knuckle-mr', 'knuckle-rp']

def load_rig(path):
    tok = path.read_text().split()
    nj, nl = int(tok[8]), int(tok[9]); pos = 11
    pos += nj * 8
    rest = np.zeros((nl, 3))
    for i in range(nl):
        rest[i] = [float(tok[pos]), float(tok[pos + 1]), float(tok[pos + 2])]; pos += 3 + 17
    mp = np.full((21, 3), np.nan)
    for m, r in enumerate(MP_TO_RIG):
        if r >= 0: mp[m] = rest[r]
    return mp

def bone_lengths(pts):
    return np.array([np.linalg.norm(pts[a] - pts[b]) for a, b in BONES])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('session'); ap.add_argument('profiles')
    ap.add_argument('--max-residual', type=float, default=0.008); ap.add_argument('--min-presence', type=float, default=0.9)
    a = ap.parse_args()
    session = Path(a.session); profiles = Path(a.profiles)
    cams = fh.load_calibration(session / 'frame' / 'snapshot' / 'device' / 'persist')
    audit = fh.read_audit(session / 'frame' / 'live-views.txt')
    rig = {0: load_rig(profiles / 'side0.fhrig'), 1: load_rig(profiles / 'side1.fhrig')}
    measured = {0: [], 1: []}
    for stamp, a_ in audit.items():
        by_id = {}
        for v in a_['views']:
            if v['presence'] >= a.min_presence and v['camera'] in cams: by_id.setdefault(v['id'], []).append(v)
        for hid, vs in by_id.items():
            if len(vs) < 2 or hid not in a_['hands']: continue
            pts, res = fh.triangulate_hand(cams, [(cams[v['camera']], v['pts'], v['presence']) for v in vs[:2]])
            if res > a.max_residual: continue
            side = 1 if a_['hands'][hid][0] > 0.5 else 0
            measured[side].append(bone_lengths(pts) * 1000)
    for side in (0, 1):
        m = np.array(measured[side])
        if len(m) < 50: print(f'side {side}: only {len(m)} clean stereo hands'); continue
        r = bone_lengths(rig[side])
        med = np.median(m, 0); q1, q3 = np.percentile(m, 25, 0), np.percentile(m, 75, 0)
        ratio = med / r
        ok = np.isfinite(ratio)
        print(f'\nside {side} ({"left" if side == 0 else "right"}): {len(m)} clean stereo hands; measured/rig ratio median {np.nanmedian(ratio):.3f}, '
              f'IQR of per-bone ratios {np.nanpercentile(ratio, 25):.3f}..{np.nanpercentile(ratio, 75):.3f}')
        print('  bone          rig mm   measured median (IQR) mm   ratio')
        for i, name in enumerate(NAMES):
            if not ok[i]: continue
            print(f'  {name:12s} {r[i]:7.1f}   {med[i]:7.1f} ({q1[i]:5.1f}..{q3[i]:5.1f})   {ratio[i]:5.3f}')
        # whole-hand scale from palm bones only (least affected by finger flexion / landmark ambiguity)
        palm = [0, 4, 8, 12, 18, 19, 20]
        print(f'  palm-only scale (measured/rig): {np.median(ratio[palm]):.3f}; finger segments: {np.nanmedian(ratio[[1, 2, 5, 6, 9, 10, 13, 14]]):.3f}')

if __name__ == '__main__': main()
