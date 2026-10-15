"""Scale a side rig profile (FHRIG, mm) to the wearer's measured hand: a palm scale about the wrist (moves each
finger chain's base and the palm centre) and a per-finger chain scale about that base (shortens or lengthens the
pivots and landmarks of the finger). Axes, limits and skinning weights are unchanged. Writes an 'experimental'
profile whose calibration sha is the sha256 of the measurement JSON given.
usage: python scale_profile.py <side.fhrig> <out.fhrig> --measurement <json> --palm 0.95 --fingers 0.92 0.96 0.92 0.90 0.94
       (finger order: thumb index middle ring pinky; a single --fingers value applies to all)
"""
import argparse, hashlib
from pathlib import Path
import numpy as np

GROUPS = ((0, 6, 7), (1, 8, 9, 10), (2, 11, 12, 13), (3, 14, 15, 16), (4, 17, 18, 19))   # rig landmark indices per finger
PALM_CENTRE = 20

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('profile'); ap.add_argument('out'); ap.add_argument('--measurement', required=True)
    ap.add_argument('--palm', type=float, required=True); ap.add_argument('--fingers', type=float, nargs='+', required=True)
    ap.add_argument('--segments', type=float, nargs='*', default=[], help='per-segment scales, 15 values: thumb(3) index(3) middle(3) ring(3) pinky(3), '
                    'proximal..distal, applied along each chain after --fingers (1.0 = unchanged); the thumb has 2 measured segments: give 1.0 for its first')
    a = ap.parse_args()
    fingers = a.fingers if len(a.fingers) == 5 else [a.fingers[0]] * 5
    if not (0.7 <= a.palm <= 1.3) or any(not 0.7 <= f <= 1.3 for f in fingers): raise SystemExit('scale out of range')
    segments = a.segments if len(a.segments) == 15 else [1.0] * 15
    if any(not 0.5 <= s <= 1.5 for s in segments): raise SystemExit('segment scale out of range')
    rows = Path(a.profile).read_text().splitlines(); header = rows[0].split()
    if header[:4] != ['FHRIG', '1', 'mm', 'left'] or header[-3:] != ['20', '21', '17']: raise SystemExit('unsupported profile')
    joints = np.array([list(map(float, r.split())) for r in rows[1:21]]); landmarks = np.array([list(map(float, r.split())) for r in rows[21:42]])
    if joints.shape != (20, 8) or landmarks.shape != (21, 20): raise SystemExit('invalid rig dimensions')
    wrist = landmarks[5, :3].copy()
    for f, group in enumerate(GROUPS):
        base = joints[4 * f, 3:6].copy()
        new_base = wrist + (base - wrist) * a.palm
        for k in range(4):
            p = joints[4 * f + k, 3:6]
            joints[4 * f + k, 3:6] = new_base + (p - base) * fingers[f]
        for l in group:
            landmarks[l, :3] = new_base + (landmarks[l, :3] - base) * fingers[f]
        # per-segment: pivots P1 = joint 4f+1 (thumb: CMC landmark, fingers: MCP), P2 = joint 4f+2 (PIP / thumb MCP),
        # P3 = joint 4f+3 (DIP / thumb IP) and the tip landmark group[0]; segments P1->P2, P2->P3, P3->tip are
        # scaled in turn and everything distal shifts with them. Landmarks sitting on P2/P3 follow their pivot.
        P = [joints[4 * f + k, 3:6].copy() for k in (1, 2, 3)]
        tip = landmarks[group[0], :3].copy()
        sa, sb, sc = segments[3 * f:3 * f + 3]
        new2 = P[0] + (P[1] - P[0]) * sa
        new3 = new2 + (P[2] - P[1]) * sb
        new_tip = new3 + (tip - P[2]) * sc
        moves = {1: (P[1], new2), 2: (P[2], new3)}
        for k, (old, new) in moves.items():
            joints[4 * f + k + 1, 3:6] = new
            for l in group[1:]:
                if np.linalg.norm(landmarks[l, :3] - old) < 1e-6: landmarks[l, :3] = new
        landmarks[group[0], :3] = new_tip
    landmarks[PALM_CENTRE, :3] = wrist + (landmarks[PALM_CENTRE, :3] - wrist) * a.palm
    header[4] = 'experimental'; header[7] = hashlib.sha256(Path(a.measurement).read_bytes()).hexdigest()
    text = ' '.join(header) + '\n' + '\n'.join(' '.join(format(v, '.17g') for v in row) for row in list(joints) + list(landmarks)) + '\n'
    Path(a.out).write_text(text)
    print(f'wrote {a.out}: palm x{a.palm}, fingers {fingers}')

if __name__ == '__main__': main()
