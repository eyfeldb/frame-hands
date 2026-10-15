"""Draw the fitted rig (forward kinematics of a rig trace, reprojected) over the recorded frames beside the live 2D
landmarks, zoomed on one hand, for chosen seconds. Green = live 2D model, blue = rig output reprojected. Where the
two disagree is where the rig fit cannot (or will not) follow the model.
usage: python rig_overlay.py <session-dir> <rig.txt> <profiles-dir> <out.png> --seconds 28 36 --side 1
"""
import argparse, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import fh_common as fh  # noqa: E402
from refiner_data import RigFK, load_fhrig, parse_rig_trace  # noqa: E402

MP_TO_RIG = [5, -1, 6, 7, 0, 8, 9, 10, 1, 11, 12, 13, 2, 14, 15, 16, 3, 17, 18, 19, 4]
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

def draw(img, pts, color):
    for a, b in BONES:
        if np.all(np.isfinite(pts[a])) and np.all(np.isfinite(pts[b])):
            cv2.line(img, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), color, 2)

def main():
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument('session'); ap.add_argument('rig'); ap.add_argument('profiles'); ap.add_argument('out')
    ap.add_argument('--seconds', type=float, nargs='+', required=True); ap.add_argument('--side', type=int, default=1); ap.add_argument('--zoom', type=int, default=360)
    a = ap.parse_args()
    session = Path(a.session)
    rec = fh.Recording(session / 'frame' / 'sets.bin'); stamps = rec.stamps(); t0 = stamps[0]
    cams = fh.load_calibration(session / 'frame' / 'snapshot' / 'device' / 'persist')
    audit = fh.read_audit(session / 'frame' / 'live-views.txt')
    frames = {s: hands for s, hands in parse_rig_trace(a.rig)}
    fks = {s: RigFK(load_fhrig(Path(a.profiles) / f'side{s}.fhrig')) for s in (0, 1)}
    rows = []
    for sec in a.seconds:
        target = t0 + sec * 1e9
        i = int(np.argmin([abs(s - target) for s in stamps])); s = stamps[i]
        imgs = rec.read(i)
        hands = [h for h in frames.get(s, []) if h['status'] == 'fitted' and h['side'] == a.side]
        tiles = []
        for cam_name in ('slam_left', 'slam_right'):
            img = imgs.get(cam_name)
            if img is None: continue
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
            cam = cams[cam_name]
            # live 2D views of this side at this set
            live = None
            au = audit.get(s)
            if au:
                for v in au['views']:
                    hid = v['id']
                    if v['camera'] == cam_name and hid in au['hands'] and (au['hands'][hid][0] > 0.5) == (a.side == 1):
                        live = np.asarray(v['pts']); draw(vis, live, (0, 255, 0))
            rigpts = None
            for h in hands:
                pts_rig = fks[a.side].landmarks(torch.tensor(h['angles'])[None], torch.tensor(h['q'])[None], torch.tensor(h['t'])[None], a.side).numpy()[0] / 1000.0
                mp = np.full((21, 3), np.nan)
                for m, r in enumerate(MP_TO_RIG):
                    if r >= 0: mp[m] = pts_rig[r]
                ok = np.all(np.isfinite(mp), 1)
                uv = np.full((21, 2), np.nan)
                uv[ok], _ = cam.project(mp[ok])
                rigpts = uv; draw(vis, uv, (255, 128, 0))
                err = np.nanmean(np.linalg.norm(uv - live, axis=1)) if live is not None else float('nan')
                cv2.putText(vis, f'{sec:.0f}s {cam_name} rig rmse {h["rmse"]:.1f}px  vs live {err:.1f}px', (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            ref = live if live is not None else rigpts
            if ref is not None:
                c = np.nanmean(ref, 0); half = max(60, float(np.nanmax(np.nanmax(ref, 0) - np.nanmin(ref, 0))) * 0.8)
                x0, y0 = int(c[0] - half), int(c[1] - half); x1, y1 = int(c[0] + half), int(c[1] + half)
                crop = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
                xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x1, vis.shape[1]), min(y1, vis.shape[0])
                crop[ya - y0:yb - y0, xa - x0:xb - x0] = vis[ya:yb, xa:xb]
                tiles.append(cv2.resize(crop, (a.zoom, a.zoom)))
            else:
                tiles.append(np.zeros((a.zoom, a.zoom, 3), np.uint8))
        rows.append(np.hstack(tiles))
    cv2.imwrite(a.out, np.vstack(rows)); print('wrote', a.out)

if __name__ == '__main__': main()
