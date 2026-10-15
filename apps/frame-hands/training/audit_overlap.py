"""How much tracking is lost when the two hands are close to each other?

From the LIVE audit of a recording: per set, the two tracked hands' triangulated palm centres. Overlap
episodes = palm distance < --near metres (or either hand's 2D palm inside the other hand's 2D extent in a
camera). Reports per-view loss rate and single-hand/no-hand rates inside overlap episodes versus outside,
and the fraction of overlap sets where a hand is missing in a camera although its partner is seen.

usage: python audit_overlap.py <session-dir> [--near 0.15]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fh_common as fh  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session'); ap.add_argument('--near', type=float, default=0.15)
    a = ap.parse_args()
    session = Path(a.session)
    audit = fh.read_audit(session / 'frame' / 'live-views.txt')
    cams = fh.load_calibration(session / 'frame' / 'snapshot' / 'device' / 'persist')
    stamps = sorted(audit)
    # per set: hands with 3D palm (stereo) and per-camera 2D extents
    rows = []
    for s in stamps:
        a_ = audit[s]
        by_id = {}
        for v in a_['views']:
            if v['presence'] >= 0.5 and v['camera'] in cams:
                by_id.setdefault(v['id'], {})[v['camera']] = v
        hands = {}
        for hid, vs in by_id.items():
            if len(vs) >= 2:
                vv = list(vs.values())
                pts, res = fh.triangulate_hand(cams, [(cams[v['camera']], v['pts'], v['presence']) for v in vv[:2]])
                palm = pts[[0, 5, 9, 13, 17]].mean(0)
            else:
                palm = None
            hands[hid] = dict(views=vs, palm=palm, right=a_['hands'].get(hid, (0.5, 0))[0])
        rows.append((s, hands))
    # overlap by 2D: in any camera, a hand's palm centre lies within the other hand's 2D bounding box (expanded 20%)
    def overlap2d(h1, h2):
        for cam in cams:
            v1, v2 = h1['views'].get(cam), h2['views'].get(cam)
            if v1 is None or v2 is None:
                continue
            for a_, b_ in ((v1, v2), (v2, v1)):
                lo, hi = a_['pts'].min(0), a_['pts'].max(0); c = (lo + hi) / 2; ext = (hi - lo) * 0.6
                if np.all(np.abs(b_['pts'][9] - c) < ext):
                    return True
        return False
    stats = dict(sets=len(rows), overlap_sets=0, overlap_two_hands=0, overlap_one_hand=0, overlap_mono_views=0, nonoverlap_sets=0,
                 nonoverlap_two_hands=0, nonoverlap_mono_views=0, overlap_episodes=0, episode_lengths=[])
    prev_overlap = False; episode = 0; last_two_hands_ids = None
    for i, (s, hands) in enumerate(rows):
        ids = [h for h in hands if hands[h]['palm'] is not None or len(hands[h]['views']) >= 1]
        two = [h for h in hands if hands[h]['palm'] is not None]
        is_overlap = False
        if len(two) >= 2:
            h1, h2 = hands[two[0]], hands[two[1]]
            d = np.linalg.norm(h1['palm'] - h2['palm'])
            is_overlap = d < a.near or overlap2d(h1, h2)
        elif len(hands) >= 2:
            hs = list(hands.values())
            is_overlap = overlap2d(hs[0], hs[1])
        elif len(hands) == 1 and prev_overlap:
            is_overlap = True   # continuing an overlap episode after one hand dropped
        if is_overlap:
            stats['overlap_sets'] += 1
            if not prev_overlap:
                stats['overlap_episodes'] += 1; episode = 0
            episode += 1
            stats['overlap_two_hands'] += len(two) >= 2
            stats['overlap_one_hand'] += len(hands) == 1
            stats['overlap_mono_views'] += sum(1 for h in hands.values() if len(h['views']) == 1)
        else:
            if prev_overlap:
                stats['episode_lengths'].append(episode)
            stats['nonoverlap_sets'] += 1
            stats['nonoverlap_two_hands'] += len(two) >= 2
            stats['nonoverlap_mono_views'] += sum(1 for h in hands.values() if len(h['views']) == 1)
        prev_overlap = is_overlap
    report = dict(session=session.name, near_m=a.near, sets=stats['sets'], overlap_sets=stats['overlap_sets'], overlap_episodes=stats['overlap_episodes'],
                  episode_len_p50=float(np.median(stats['episode_lengths'])) if stats['episode_lengths'] else 0,
                  overlap_two_hand_rate=stats['overlap_two_hands'] / max(1, stats['overlap_sets']),
                  overlap_one_hand_rate=stats['overlap_one_hand'] / max(1, stats['overlap_sets']),
                  overlap_mono_views_per_set=stats['overlap_mono_views'] / max(1, stats['overlap_sets']),
                  nonoverlap_two_hand_rate=stats['nonoverlap_two_hands'] / max(1, stats['nonoverlap_sets']),
                  nonoverlap_mono_views_per_set=stats['nonoverlap_mono_views'] / max(1, stats['nonoverlap_sets']))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
