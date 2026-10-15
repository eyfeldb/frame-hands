"""Evaluate frozen <=100ms image-motion cue on actual mono episodes, offline only.

History is dropped on missing source, duplicate side, side change or missing HMD
context. Mono updates never refresh the stereo seed. Reacquisition is scored only
when it falls inside the same original100ms budget. No future data enters fitting.
"""
import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import shutil
import sys
import time
import cv2
import numpy as np
from probe_monocular_flow import flow_targets
from probe_monocular_bridge import BASE, NAMES, PALM_MP, RI, MI, transform, continue_rigid
from frame_data import Recording, read_audit, capture_poses, load_calibration
from palm_flow import PalmFlow
from audit_native_lengths import sha, stats


def select_cases(audit, poses, valid_stamps):
    history = {}
    cases = []
    reasons = Counter()
    for stamp, row in sorted(audit.items()):
        if stamp not in valid_stamps:
            history.clear()
            reasons['missing_head_frames'] += 1
            continue
        counts = Counter(int(h['right'] > .5) for h in row['hands'].values())
        history = {source: old for source, old in history.items() if source in row['hands']}
        for source, hand in row['hands'].items():
            side = int(hand['right'] > .5)
            views = [v for v in row['views'] if v['id'] == source and v['camera'] in NAMES]
            if counts[side] != 1 or len({v['camera'] for v in views}) != len(views):
                history.pop(source, None)
                reasons['ambiguous_source_rows'] += 1
                continue
            old = history.get(source)
            if old and (old['side'] != side or stamp-old['stamp'] > 100_000_000):
                history.pop(source)
                old = None
            pose = poses.get((stamp, source))
            stereo = hand['stereo'] >= 2 and len(views) == 2
            mono = hand['stereo'] < 2 and len(views) == 1
            if mono and old:
                cases.append(dict(kind='mono', raw_ns=stamp, prior_raw_ns=old['stamp'],
                                  source=source, side=side, view=views[0]['camera']))
                old['mono_count'] += 1
                old['last_view'] = views[0]['camera']
            elif mono:
                reasons['mono_without_eligible_seed'] += 1
            elif stereo and old and old['mono_count']:
                cases.append(dict(kind='reacquired', raw_ns=stamp, prior_raw_ns=old['stamp'],
                                  source=source, side=side, view=old['last_view']))
            if stereo and pose is not None and pose['side'] == side:
                history[source] = dict(stamp=stamp, side=side, mono_count=0)
            elif not mono:
                history.pop(source, None)
    return cases, dict(reasons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('session', 'native', 'upstream', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    if '143912Z' in str(a.session): raise ValueError('Latest holdout excluded')
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.upstream))
    cv2.setNumThreads(1)
    audit = read_audit(a.session/'frame/live-views.txt')
    heads = capture_poses(a.session)
    cameras = load_calibration(a.session/'frame/snapshot/device/persist')
    recording = Recording(a.session/'frame/sets.bin')
    first = recording.index[0][0]
    records = [json.loads(line) for line in (a.native/'palm-profile-viewer-v2/poses.jsonl').read_text().splitlines()]
    fitted = np.loadtxt(a.native/'rig-stage-v2/recorded-output.txt')
    if len(records) != len(fitted) or fitted.shape[1] != 104: raise ValueError('Mismatched seed data')
    poses = {}
    for row, fit in zip(records, fitted):
        mono = audit[row['raw_ns']]['mono_ns']
        if mono in heads:
            poses[(row['raw_ns'], row['source_id'])] = dict(side=row['side'], head=fit[41:].reshape(21, 3),
                    world=transform(fit[41:].reshape(21, 3), heads[mono]))
    selected, reasons = select_cases(audit, poses, {stamp for stamp, row in audit.items() if row['mono_ns'] in heads})

    @lru_cache(maxsize=24)
    def images(stamp):
        t = (stamp-first)/1e9
        frames = list(recording.frames(t-1e-7, t+1e-7))
        if len(frames) != 1 or frames[0][1] != stamp: raise ValueError('Exact image missing')
        return frames[0][2]

    rows = []
    for case in selected:
        stamp, prior_stamp, source, name = (case[k] for k in ('raw_ns', 'prior_raw_ns', 'source', 'view'))
        prior = poses[(prior_stamp, source)]
        seed = transform(prior['world'], np.linalg.inv(heads[audit[stamp]['mono_ns']]))
        old_view = next(v for v in audit[prior_stamp]['views'] if v['id'] == source and v['camera'] == name)
        current_views = {v['camera']: v for v in audit[stamp]['views'] if v['id'] == source}
        old_image, current_image = images(prior_stamp)[name][0], images(stamp)[name][0]
        start = time.perf_counter_ns()
        motion = PalmFlow(old_image, old_view['points'][PALM_MP], prior_stamp)
        detail = motion.advance(current_image, stamp)
        points, fit = seed.copy(), dict(accepted=False, reason='flow_rejected')
        if detail['supported']:
            targets = flow_targets(cameras[name], prior['head'], detail['affine'])
            points, fit = continue_rigid(cameras[name], seed, targets, (stamp-prior_stamp)/1e9, 'plane')
        elapsed = (time.perf_counter_ns()-start)/1000
        case.update(seconds=(stamp-first)/1e9, age_ms=(stamp-prior_stamp)/1e6,
                    flow=detail, fit=fit, host_us=elapsed, methods={})
        for mode, cloud in [('hold', seed), ('flow', points)]:
            values = {}
            for view_name, view in current_views.items():
                camera = cameras[view_name]
                values[view_name] = float(np.linalg.norm(camera.eye_to_window(camera.world_to_eye(cloud[RI]))-view['points'][MI], axis=1).mean())
            case['methods'][mode] = dict(head_mm=cloud.tolist(), observed_error_px=values[name],
                                        withheld_error_px=values.get(NAMES[1-NAMES.index(name)]))
        case['step_mm'] = float(np.linalg.norm(points-seed, axis=1).max())
        rows.append(case)
    with (a.output/'cases.jsonl').open('x') as f:
        for row in rows: f.write(json.dumps(row, allow_nan=False)+'\n')
    summary = {}
    for kind in ('mono', 'reacquired'):
        sub = [r for r in rows if r['kind'] == kind]
        summary[kind] = dict(cases=len(sub), admitted=sum(r['fit']['accepted'] for r in sub),
                            episodes=len({(r['prior_raw_ns'], r['source']) for r in sub}),
                            reasons=dict(Counter(r['flow']['reason'] for r in sub)),
                            step_mm=stats([r['step_mm'] for r in sub]), host_us=stats([r['host_us'] for r in sub]),
                            methods={m: {key: stats([r['methods'][m][key] for r in sub if r['methods'][m][key] is not None])
                                         for key in ('observed_error_px', 'withheld_error_px')} for m in ('hold', 'flow')})
    source = a.output/'source'; source.mkdir()
    files = [Path(__file__), BASE/'probe_monocular_flow.py', BASE/'probe_monocular_bridge.py',
             BASE.parent/'exploratory/palm_flow.py', BASE.parent/'exploratory/frame_data.py']
    for path in files: shutil.copy2(path, source/path.name)
    inputs = [a.session/'frame/sets.bin', a.session/'frame/live-views.txt', a.session/'relay-trace.bin', a.session/'sampler-trace.bin',
              a.session/'frame/snapshot/device/persist/device_config.json', a.session/'frame/snapshot/device/persist/xrservice.json',
              a.native/'palm-profile-viewer-v2/poses.jsonl', a.native/'rig-stage-v2/recorded-output.txt']
    report = dict(scope=__doc__, selection_rejections=reasons, results=summary,
                  source_sha256={p.name: sha(p) for p in files}, input_sha256={str(p): sha(p) for p in inputs},
                  versions=dict(python=sys.version, opencv=cv2.__version__, numpy=np.__version__),
                  limitations=['Mono frames have no independent withheld detector labels or physical ground truth.',
                               'Observed-camera detector error is agreement, not an independent accuracy measure.',
                               'Reacquisition scores are only endpoints within100ms; no later reference is used as contemporaneous truth.',
                               'Source continuity does not prove anatomical identity; old articulation is frozen.',
                               'No runtime/presenter change, native benchmark or wearer validation.'])
    (a.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
