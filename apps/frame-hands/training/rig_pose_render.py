"""Render the fitted rig skeleton of chosen frames as 3D stick figures from three fixed viewpoints (what the glove
shows, independent of the cameras), with the per-finger joint angles printed, to judge whether a pose such as a
vertical thumbs-up comes out as a natural fist with a straight thumb.
usage: python rig_pose_render.py <rig.txt> <profiles-dir> <out.png> --seconds 28 36 60 [--side 1]
"""
import argparse, math, sys
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).parent))
from refiner_data import RigFK, load_fhrig, parse_rig_trace  # noqa: E402

MP_TO_RIG = [5, -1, 6, 7, 0, 8, 9, 10, 1, 11, 12, 13, 2, 14, 15, 16, 3, 17, 18, 19, 4]
BONES = [(0, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
COL = {0: (200, 200, 200), 1: (0, 200, 255), 2: (0, 255, 0), 3: (255, 200, 0), 4: (255, 0, 255), 5: (0, 128, 255)}
FINGER = [0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]

def view(pts, R, size=320, scale=1.4):
    img = np.zeros((size, size, 3), np.uint8)
    p = (pts - pts[0]) @ R.T            # wrist at origin, rotate
    c = np.array([size / 2, size / 2]); k = size / (0.25 * scale)   # 25 cm box
    uv = np.stack([c[0] + p[:, 0] * k, c[1] - p[:, 1] * k], 1)
    for a, b in BONES:
        if np.all(np.isfinite(uv[a])) and np.all(np.isfinite(uv[b])):
            cv2.line(img, tuple(int(v) for v in uv[a]), tuple(int(v) for v in uv[b]), COL[FINGER[b]], 3)
    return img

def main():
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument('rig'); ap.add_argument('profiles'); ap.add_argument('out')
    ap.add_argument('--seconds', type=float, nargs='+', required=True); ap.add_argument('--side', type=int, default=1)
    a = ap.parse_args()
    frames = parse_rig_trace(a.rig); t0 = frames[0][0]
    fk = RigFK(load_fhrig(Path(a.profiles) / f'side{a.side}.fhrig'))
    rows = []
    # viewpoints in the head frame: front (looking along -z), side (from +x), top (from +y)
    front = np.eye(3); side = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float); top = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
    for sec in a.seconds:
        target = t0 + sec * 1e9
        stamp, hands = min(frames, key=lambda f: abs(f[0] - target))
        hs = [h for h in hands if h['status'] == 'fitted' and h['side'] == a.side]
        if not hs: continue
        h = hs[0]
        pts_rig = fk.landmarks(torch.tensor(h['angles'])[None], torch.tensor(h['q'])[None], torch.tensor(h['t'])[None], a.side).numpy()[0] / 1000.0
        mp = np.full((21, 3), np.nan)
        for m, r in enumerate(MP_TO_RIG):
            if r >= 0: mp[m] = pts_rig[r]
        tiles = [view(mp, front), view(mp, side), view(mp, top)]
        deg = np.degrees(h['angles'])
        text = np.zeros((320, 420, 3), np.uint8)
        cv2.putText(text, f'{sec:.0f}s side {a.side} rmse {h["rmse"]:.1f}px', (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for f in range(5):
            cv2.putText(text, f'{names[f]:6s} ' + ' '.join(f'{deg[4 * f + k]:6.0f}' for k in range(4)), (6, 50 + 24 * f), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL[f + 1], 1)
        cv2.putText(text, 'cols: base-a base-b PIP DIP (thumb: CMC-a CMC-b MCP IP)', (6, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        cv2.putText(text, 'views: front | side | top  (wrist at centre, 25 cm box)', (6, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        rows.append(np.hstack(tiles + [text]))
    cv2.imwrite(a.out, np.vstack(rows)); print('wrote', a.out)

if __name__ == '__main__': main()
