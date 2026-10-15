"""Review hand-loss moments during close two-hand interaction in a recording.

Finds episodes where both hands were tracked close together (3D palm distance < --near) and then one hand
disappeared from the LIVE audit for >= --min-gap sets. For the worst episodes renders both cameras at the
last two-hand set, the first lost set and mid-loss, with the live landmarks (left hand green, right hand
red, mono views dashed), and prints a per-episode summary (which hand, which cameras still saw it).

usage: python review_interaction.py <session-dir> <out.png> [--near 0.14] [--min-gap 3] [--episodes 4]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fh_common as fh  # noqa: E402

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]


def draw(img, pts, color, thick):
    for a, b in BONES:
        cv2.line(img, tuple(np.round(pts[a]).astype(int)), tuple(np.round(pts[b]).astype(int)), color, thick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session'); ap.add_argument('out'); ap.add_argument('--near', type=float, default=0.14)
    ap.add_argument('--min-gap', type=int, default=3); ap.add_argument('--episodes', type=int, default=4)
    a = ap.parse_args()
    session = Path(a.session)
    rec = fh.Recording(session / 'frame' / 'sets.bin')
    audit = fh.read_audit(session / 'frame' / 'live-views.txt')
    cams = fh.load_calibration(session / 'frame' / 'snapshot' / 'device' / 'persist')
    stamps = rec.stamps()
    # per set: hands -> side, palm3d (if stereo), views per cam
    per = []
    for s in stamps:
        a_ = audit.get(s)
        if not a_:
            per.append(None); continue
        hands = {}
        for hid, (r, nv) in a_['hands'].items():
            vs = {v['camera']: v for v in a_['views'] if v['id'] == hid and v['presence'] >= 0.5}
            palm = None
            if len(vs) >= 2:
                vv = list(vs.values())
                pts, _ = fh.triangulate_hand(cams, [(cams[v['camera']], v['pts'], v['presence']) for v in vv[:2]])
                palm = pts[[0, 5, 9, 13, 17]].mean(0)
            hands[hid] = dict(side=int(r > 0.5), palm=palm, views=vs)
        per.append(hands)
    # episodes: close two-hand state followed by a set where a side is missing for >= min_gap sets
    episodes = []
    i = 0
    while i < len(per) - 1:
        h = per[i]
        if h:
            sides = {v['side']: v for v in h.values()}
            if len(sides) == 2 and all(v['palm'] is not None for v in sides.values()):
                d = np.linalg.norm(sides[0]['palm'] - sides[1]['palm'])
                if d < a.near:
                    nxt = per[i + 1]
                    nsides = {v['side'] for v in nxt.values()} if nxt else set()
                    missing = [sd for sd in (0, 1) if sd not in nsides]
                    if missing:
                        sd = missing[0]; j = i + 1
                        while j < len(per) and (not per[j] or sd not in {v['side'] for v in per[j].values()}):
                            j += 1
                        gap = j - i - 1
                        if gap >= a.min_gap:
                            episodes.append(dict(last_two=i, first_lost=i + 1, back=j, gap=gap, side=sd, dist=d))
                            i = j; continue
        i += 1
    episodes.sort(key=lambda e: -e['gap'])
    print(f'close-interaction loss episodes: {len(episodes)}; total lost sets {sum(e["gap"] for e in episodes)}')
    for e in episodes[:12]:
        t = (stamps[e['last_two']] - stamps[0]) / 1e9
        other = per[e['last_two']]
        print(f"  t={t:6.1f}s side {'R' if e['side'] else 'L'} lost for {e['gap']} sets ({e['gap'] / 30:.2f}s), palm distance {e['dist'] * 100:.1f} cm, "
              f"views at last two-hand set: " + ', '.join(f"{'R' if v['side'] else 'L'}:{'+'.join(c[5:6] for c in v['views'])}" for v in other.values()))
    # render
    rows = []
    for e in episodes[:a.episodes]:
        picks = [e['last_two'], e['first_lost'], min(e['back'] - 1, e['first_lost'] + max(1, e['gap'] // 2))]
        tiles = []
        for idx in picks:
            images = rec.read(idx)
            for cam in ('slam_left', 'slam_right'):
                img = cv2.cvtColor(images[cam], cv2.COLOR_GRAY2BGR)
                h = per[idx] or {}
                for hid, v in h.items():
                    if cam in v['views']:
                        draw(img, v['views'][cam]['pts'], (0, 255, 0) if v['side'] == 0 else (0, 0, 255), 2 if len(v['views']) >= 2 else 1)
                        cv2.putText(img, f"id{hid} {'R' if v['side'] else 'L'} p{v['views'][cam]['presence']:.2f} r{v['views'][cam]['right']:.2f}", tuple(np.round(v['views'][cam]['pts'][0]).astype(int) + np.array([5, 15])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(img, f"t={(stamps[idx] - stamps[0]) / 1e9:.2f}s {cam} {'lost ' + ('R' if e['side'] else 'L') if idx > e['last_two'] else 'both'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                tiles.append(cv2.resize(img, (528, 512)))
        rows.append(np.hstack(tiles))
    if rows:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(a.out, np.vstack(rows)); print('wrote', a.out)


if __name__ == '__main__':
    main()
