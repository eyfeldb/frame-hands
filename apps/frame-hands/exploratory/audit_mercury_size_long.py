"""Audit declared calibration/freeze boundaries without rerunning image inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def stats(values):
    if not values:
        return None
    return dict(count=len(values), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), maximum=float(np.max(values)))


def audit(rows, protocol):
    x = np.float32(0)
    size = np.float32(.09)
    seen = [False, False]
    frozen = None
    counts = Counter()
    skips = Counter()
    metrics = {}
    previous = {}
    qualities = []
    boundary = protocol['freeze_seconds']
    for row in rows:
        phase = 'calibration' if row['seconds'] < boundary else 'evaluation'
        assert row['phase'] == phase
        counts[phase + '_frames'] += 1
        found = np.zeros((2, 2), bool)
        views = {}
        for hand in row['observations']:
            side = hand['side']
            views[side] = {}
            for view in hand['views']:
                if 'skipped' in view:
                    skips[phase + ': ' + view['skipped']] += 1
                    continue
                index = ('slam_left', 'slam_right').index(view['camera'])
                found[index, side] = view['model']['active']
                views[side][view['camera']] = view['model']
        mono = bool(np.any(found[0] != found[1]))
        counts[phase + '_singleton_frames'] += int(mono)
        counts[phase + '_no_stereo_frames'] += int(not np.any(found[0] & found[1]))
        pre = row['scale_pre']
        if phase == 'calibration':
            y = np.float32(1) if x > 100 else np.float32(np.float32(x / 100) ** 2)
            optimize = bool(x <= 100 and not mono)
            if any(found[:, side].any() and not seen[side] for side in (0, 1)):
                x = min(x, np.float32(50))
            np.testing.assert_allclose(pre['schedule_x'], x, atol=2e-6, rtol=0)
            np.testing.assert_allclose(pre['target_size'], size, atol=1e-8, rtol=0)
            np.testing.assert_allclose(pre['size_error'], y, atol=2e-8, rtol=0)
            assert pre['optimize_size'] == optimize
        else:
            if frozen is None:
                frozen = dict(size_m=float(size), schedule_x=float(x), stamp=row['stamp'])
            assert pre['optimize_size'] is False
            assert pre['target_size'] == frozen['size_m']
            assert pre['schedule_x'] == frozen['schedule_x']
        sizes = []
        by_side = {}
        for fit in row['fits']:
            side, branch = fit['side'], fit['branch']
            assert found[:, side].all(), 'Fit without paired active observations'
            counts[phase + '_' + branch + '_fits'] += 1
            counts[phase + '_' + branch + '_rejected'] += int(not fit['accepted'])
            by_side.setdefault(side, {})[branch] = fit
            if branch == 'calibrated' and fit['accepted'] and phase == 'calibration':
                if not mono:
                    total = np.float32(0)
                    for name in ('slam_left', 'slam_right'):
                        for c in views[side][name]['confidence_xy']:
                            total = np.float32(total + np.float32(c))
                    mean = np.float32(total / np.float32(42))
                    error = np.float32(fit['fit']['reprojection_error'])
                    quality = np.float32(mean * np.float32(1 / np.float32(error * np.float32(10) + np.float32(1))))
                    x = np.float32(x + quality)
                    qualities.append(float(quality))
                sizes.append(np.float32(fit['fit']['hand_size_m']))
                seen[side] = True
                counts['verified_scale_posts'] += 1
            if phase == 'evaluation':
                expected = .09 if branch == 'fixed' else frozen['size_m']
                np.testing.assert_allclose(fit['fit']['hand_size_m'], expected, atol=1e-8, rtol=0)
            key = phase + '_' + branch + '_' + str(side)
            m = metrics.setdefault(key, dict(size_mm=[], ncnn_disagreement_px=[], max_joint_step_mm=[], resets=Counter(), accepted=0))
            m['resets'][str(fit['reset_reason'])] += 1
            if fit['accepted']:
                m['accepted'] += 1
                m['size_mm'].append(fit['fit']['hand_size_m'] * 1000)
                m['ncnn_disagreement_px'].extend(v['median'] for v in fit['ncnn_disagreement_px'].values())
                old = previous.get(key)
                if old is not None and fit['reset_reason'] is None and 0 < row['stamp'] - old[0] <= 100000000:
                    m['max_joint_step_mm'].append(float(np.max(np.linalg.norm(np.asarray(fit['world_joints_mm']) - old[1], axis=1))))
                previous[key] = (row['stamp'], np.asarray(fit['world_joints_mm']))
        if sizes:
            size = np.float32(sum(sizes, np.float32(0)) / np.float32(len(sizes)))
        np.testing.assert_allclose(row['scale_post']['target_size'], size, atol=1e-8, rtol=0)
        np.testing.assert_allclose(row['scale_post']['schedule_x'], x, atol=2e-6, rtol=0)
    for m in metrics.values():
        for key in ('size_mm', 'ncnn_disagreement_px', 'max_joint_step_mm'):
            values = m[key]
            m[key] = stats(values)
            if values and key == 'size_mm':
                m[key]['minimum'] = float(min(values))
    return dict(counts=counts, geometry_rejections=skips, freeze=frozen,
                confidence_schedule_increments=stats(qualities), metrics=metrics,
                limitations='Current observations and sides are externally seeded. Joint steps and ncnn disagreement are proxies, not physical accuracy. No complete native calibration; no live changes.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    protocol = json.loads((a.result / 'protocol.json').read_text())
    path = a.result / 'frames.jsonl'
    with path.open() as source:
        result = audit((json.loads(line) for line in source), protocol)
    result['frames_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
