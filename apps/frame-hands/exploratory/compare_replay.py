"""Compare alternative poses with recorded source geometry on exact captures and IDs.

Metrics measure consistency/disagreement, not anatomical accuracy or user acceptance.
"""
import argparse
import json
from pathlib import Path
import struct
import numpy as np
from frame_data import capture_poses, read_audit, Ownership

# Comparable articulated segments; no invented thumb CMC or palm correspondence.
SEGMENTS = [(6, 7, 2, 3), (7, 0, 3, 4), (8, 9, 5, 6), (9, 10, 6, 7), (10, 1, 7, 8),
            (11, 12, 9, 10), (12, 13, 10, 11), (13, 2, 11, 12), (14, 15, 13, 14),
            (15, 16, 14, 15), (16, 3, 15, 16), (17, 18, 17, 18), (18, 19, 18, 19), (19, 4, 19, 20)]
COMMON = [(5, 0), (6, 2), (7, 3), (0, 4), (8, 5), (9, 6), (10, 7), (1, 8),
          (11, 9), (12, 10), (13, 11), (2, 12), (14, 13), (15, 14), (16, 15), (3, 16),
          (17, 17), (18, 18), (19, 19), (4, 20)]


def quantiles(v):
    return dict(zip(('p50', 'p95', 'max'), map(float, np.percentile(v, [50, 95, 100])))) if v else None


def source_hands(session):
    result = {}
    with (session/'relay-trace.bin').open('rb') as stream:
        while data := stream.read(3280):
            if len(data) != 3280 or data[:8] != b'FHRELAY1':
                raise ValueError('Invalid relay trace')
            raw = data[56:2712]
            if raw[:8] != b'FHHANDS1':
                continue
            stamp = struct.unpack_from('<Q', raw, 24)[0]
            if stamp in result:
                continue
            count = struct.unpack_from('<I', raw, 40)[0]
            if count > 2:
                raise ValueError('Too many source hands')
            hands = {}
            for i in range(count):
                ident, flags = struct.unpack_from('<II', raw, 64+272*i)
                points = np.array(struct.unpack_from('<63f', raw, 80+272*i)).reshape(21, 3)*1000
                if ident and np.isfinite(points).all():
                    hands[ident] = dict(side=flags&1, stereo=bool(flags&2), points=points)
            result[stamp] = hands
    return result


def analyze(session, predictions):
    baseline = source_hands(session)
    poses = capture_poses(session)
    audits = read_audit(session/'frame/live-views.txt')
    metrics = {key:[] for key in ('baseline_segment_change_mm', 'candidate_segment_change_mm',
                                 'baseline_landmark_step_mm', 'candidate_landmark_step_mm', 'pose_disagreement_mm')}
    stereo_metrics = {key:[] for key in metrics}
    previous = {}
    times_by_hands = {1:[], 2:[]}
    frames = hands = paired = baseline_present = alternate_present = 0
    spans = []
    current = None
    reassignments = 0
    with predictions.open() as f:
        for line in f:
            row = json.loads(line)
            frames += 1
            stamp = row['capture_mono_ns']
            raw_stamp = row['capture_raw_ns']
            seconds = row['seconds']
            source = baseline.get(stamp, {})
            visible_slots = {h['side'] for h in source.values()}
            baseline_present += len(visible_slots)
            alternate_present += len(row['hands'])
            if len(row['hands']) in times_by_hands:
                times_by_hands[len(row['hands'])].append(row['processing_ms'])
            # Only describes slot assignment; correctness requires visual labels.
            a = audits[raw_stamp]
            labels = [int(h['right'] > .5) for h in a['hands'].values()]
            duplicate = len(labels) >= 2 and len(set(labels)) < len(labels)
            if duplicate:
                if current is None:
                    current = dict(start=seconds, end=seconds, sets=0)
                current['end'] = seconds
                current['sets'] += 1
            elif current:
                spans.append(current)
                current = None
            cur = {}
            for h in row['hands']:
                hands += 1
                ident = h['source_id']
                if ident in a['hands']:
                    reassignments += int(h['side']=='R') != int(a['hands'][ident]['right']>.5)
                if ident not in source or stamp not in poses:
                    continue
                paired += 1
                raw = source[ident]['points']
                candidate = np.array(h['landmarks_head_mm'])
                xf = poses[stamp]
                old_world = raw@xf[:3, :3].T+xf[:3, 3]
                new_world = candidate@xf[:3, :3].T+xf[:3, 3]
                old_len = np.linalg.norm(raw[[b for _, _, a, b in SEGMENTS]]-raw[[a for _, _, a, b in SEGMENTS]], axis=1)
                new_len = np.linalg.norm(candidate[[b for a, b, _, _ in SEGMENTS]]-candidate[[a for a, b, _, _ in SEGMENTS]], axis=1)
                stereo_pair = source[ident]['stereo'] and len(h['views']) == 2
                disagreement = np.linalg.norm(candidate[[a for a,b in COMMON]]-raw[[b for a,b in COMMON]],axis=1).tolist()
                metrics['pose_disagreement_mm'].extend(disagreement)
                if stereo_pair:
                    stereo_metrics['pose_disagreement_mm'].extend(disagreement)
                last = previous.get(ident)
                if last and raw_stamp-last['stamp'] <= 100_000_000 and last['side'] == h['side']:
                    metrics['baseline_segment_change_mm'].extend(np.abs(old_len-last['old_len']).tolist())
                    metrics['candidate_segment_change_mm'].extend(np.abs(new_len-last['new_len']).tolist())
                    metrics['baseline_landmark_step_mm'].extend(np.linalg.norm(old_world[[b for a,b in COMMON]]-last['old_world'][[b for a,b in COMMON]],axis=1).tolist())
                    metrics['candidate_landmark_step_mm'].extend(np.linalg.norm(new_world[[a for a,b in COMMON]]-last['new_world'][[a for a,b in COMMON]],axis=1).tolist())
                    if stereo_pair and last['stereo']:
                        stereo_metrics['baseline_segment_change_mm'].extend(np.abs(old_len-last['old_len']).tolist())
                        stereo_metrics['candidate_segment_change_mm'].extend(np.abs(new_len-last['new_len']).tolist())
                        stereo_metrics['baseline_landmark_step_mm'].extend(np.linalg.norm(old_world[[b for a,b in COMMON]]-last['old_world'][[b for a,b in COMMON]],axis=1).tolist())
                        stereo_metrics['candidate_landmark_step_mm'].extend(np.linalg.norm(new_world[[a for a,b in COMMON]]-last['new_world'][[a for a,b in COMMON]],axis=1).tolist())
                cur[ident] = dict(stereo=stereo_pair, stamp=raw_stamp, side=h['side'], old_len=old_len, new_len=new_len, old_world=old_world, new_world=new_world)
            previous = cur
    if current:
        spans.append(current)
    return dict(frames=frames, candidate_hands=hands, paired_same_capture_and_id=paired,
                baseline_source_occupied_slots=baseline_present, candidate_occupied_slots=alternate_present,
                changed_assignment_vs_audit=reassignments, metrics={k:quantiles(v) for k,v in metrics.items()},
                stereo_only_metrics={k:quantiles(v) for k,v in stereo_metrics.items()},
                processing_ms_by_hands={str(k):quantiles(v) for k,v in times_by_hands.items()},
                duplicate_audit_spans=spans,
                interpretation='Consistency and occupancy only. Motion includes real gestures; fixed lengths are by construction. Counts do not establish identity accuracy. No ground truth.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', required=True, type=Path)
    p.add_argument('--predictions', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    result = analyze(args.session, args.predictions)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='duplicate_audit_spans'}, indent=2))
