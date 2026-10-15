"""Compare paired predictions by selected detector views, not local track IDs.

Track generation counters are private to each causal run and can diverge after
one different birth. Equal counters are not cross-run physical identities.
"""
import argparse, json
from pathlib import Path
import numpy as np


def hand_key(hand):
    return hand['side'], tuple(sorted((v['camera'], v['source_id']) for v in hand['source_views']))


def quantiles(values):
    return np.percentile(values, [50, 95, 100]).tolist() if values else None


def summarize(directory):
    directory = Path(directory)
    frames = matched = assignment = availability = 0
    errors, maxima, wrist, angles = [], [], [], []
    counts = [0, 0]
    with (directory / 'teacher.jsonl').open() as teacher, (directory / 'student.jsonl').open() as student, (directory / 'comparison-source-views.jsonl').open('x') as out:
        import itertools
        for left, right in itertools.zip_longest(teacher, student):
            if left is None or right is None:
                raise ValueError('Different replay lengths')
            ref, actual = json.loads(left), json.loads(right)
            if ref['capture_raw_ns'] != actual['capture_raw_ns']:
                raise ValueError('Unaligned captures')
            maps = [{hand_key(h): h for h in r['hands']} for r in (ref, actual)]
            assignment += maps[0].keys() != maps[1].keys()
            availability += {h['side'] for h in ref['hands']} != {h['side'] for h in actual['hands']}
            counts[0] += len(ref['hands']); counts[1] += len(actual['hands'])
            rows = []
            for key in maps[0].keys() & maps[1].keys():
                a, b = maps[0][key], maps[1][key]
                delta = np.asarray(a['vertices_world_mm']) - b['vertices_world_mm']
                error = float(np.linalg.norm(delta, axis=-1).mean())
                maximum = float(abs(delta).max())
                wrist_error = float(np.linalg.norm(np.asarray(a['wrist_world_mm'])[:3, 3] - np.asarray(b['wrist_world_mm'])[:3, 3]))
                angle_error = float(np.mean(np.abs(np.asarray(a['joint_angles']) - b['joint_angles'])))
                errors.append(error); maxima.append(maximum); wrist.append(wrist_error); angles.append(angle_error); matched += 1
                rows.append(dict(side=key[0], views=key[1], mean_vertex_distance_mm=error, wrist_distance_mm=wrist_error, mean_angle_difference_radians=angle_error))
            out.write(json.dumps(dict(stamp=ref['capture_raw_ns'], assignment_changed=maps[0].keys() != maps[1].keys(), matched=rows)) + '\n')
            frames += 1
    return dict(scope='Teacher agreement only. Matching requires same side and selected camera/detector views; independent track-generation IDs are not comparable.',
        frames=frames, teacher_hands=counts[0], student_hands=counts[1], assignment_changed_frames=assignment, side_availability_changed_frames=availability,
        matched_hands=matched, matched_mean_vertex_distance_mm_p50_p95_max=quantiles(errors), matched_max_coordinate_error_mm_p50_p95_max=quantiles(maxima),
        matched_wrist_distance_mm_p50_p95_max=quantiles(wrist), matched_mean_joint_angle_difference_radians_p50_p95_max=quantiles(angles))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    result = summarize(a.directory)
    (a.directory / 'summary-source-views.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
