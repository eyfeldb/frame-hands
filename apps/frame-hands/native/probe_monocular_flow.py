"""Fixed offline test of image-motion correction during <=100ms one-view gaps.

Reuses the rejected long-gap flow unchanged, only as a short position cue while
an observed-camera hand detection exists. Never extends hand presence/identity.
The other camera and current 3D pose are scoring-only. Not a runtime policy.
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
from probe_monocular_bridge import (BASE, NAMES, PALM_RIG, PALM_MP, RI, MI,
                                    transform, continue_rigid)
from frame_data import Recording, read_audit, capture_poses, load_calibration
from palm_flow import PalmFlow, PARAMETERS
from audit_native_lengths import sha, stats


def flow_targets(camera, prior_head, affine):
    """Map previous-camera projections; current head motion is already in the seed.

    Applying image displacement to current compensated projections would count
    headset motion twice. This function deliberately takes the PRIOR head pose.
    """
    affine = np.asarray(affine, dtype=float)
    if affine.shape != (2, 3) or not np.isfinite(affine).all():
        raise ValueError('Finite 2x3 affine required')
    pixels = camera.eye_to_window(camera.world_to_eye(prior_head[PALM_RIG]))
    return pixels @ affine[:, :2].T + affine[:, 2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('session', 'baseline', 'upstream', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if any('143912Z' in str(x) for x in (a.session, a.baseline)):
        raise ValueError('Latest accepted holdout excluded')
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.upstream))
    cv2.setNumThreads(1)
    recording = Recording(a.session/'frame/sets.bin')
    audit = read_audit(a.session/'frame/live-views.txt')
    heads = capture_poses(a.session)
    cameras = load_calibration(a.session/'frame/snapshot/device/persist')
    cases = [json.loads(x) for x in (a.baseline/'cases.jsonl').read_text().splitlines()]
    first = recording.index[0][0]

    @lru_cache(maxsize=32)
    def images(stamp):
        seconds = (stamp-first)/1e9
        values = list(recording.frames(seconds-1e-7, seconds+1e-7))
        if len(values) != 1 or values[0][1] != stamp:
            raise ValueError('Exact image missing')
        return values[0][2]

    output = []
    reasons = Counter()
    for index, case in enumerate(cases):
        stamp, prior_stamp = case['raw_ns'], case['prior_raw_ns']
        source, name = case['source'], case['view']
        age = (stamp-prior_stamp)/1e9
        if not 0 < age <= .100000001:
            raise ValueError('Outside frozen short-gap protocol')
        camera = cameras[name]
        current_views = [v for v in audit[stamp]['views'] if v['id'] == source and v['camera'] == name]
        prior_views = [v for v in audit[prior_stamp]['views'] if v['id'] == source and v['camera'] == name]
        if len(current_views) != 1 or len(prior_views) != 1:
            raise ValueError('Observed hand evidence missing')
        seed = np.array(case['methods']['hold']['head_mm'])
        # Recover previous-head coordinates from the world-held current seed.
        world = transform(seed, heads[audit[stamp]['mono_ns']])
        prior_head = transform(world, np.linalg.inv(heads[audit[prior_stamp]['mono_ns']]))
        old_image, current_image = images(prior_stamp)[name][0], images(stamp)[name][0]
        start = time.perf_counter_ns()
        flow = PalmFlow(old_image, prior_views[0]['points'][PALM_MP], prior_stamp)
        info = flow.advance(current_image, stamp)
        points, fit = seed.copy(), dict(accepted=False, reason='flow_rejected')
        if info['supported']:
            targets = flow_targets(camera, prior_head, info['affine'])
            points, fit = continue_rigid(camera, seed, targets, age, 'plane')
        elapsed_us = (time.perf_counter_ns()-start)/1000
        # Negative control uses exactly the same seed and a blank current image.
        blank = PalmFlow(old_image, prior_views[0]['points'][PALM_MP], prior_stamp)
        blank_info = blank.advance(np.zeros_like(current_image), stamp)
        reason = info['reason'] if not info['supported'] else ('admitted' if fit['accepted'] else 'geometry_rejected')
        reasons[reason] += 1
        # Only scoring below can see the withheld camera/current reference.
        hidden_name = NAMES[1-NAMES.index(name)]
        hidden = cameras[hidden_name]
        hidden_view = next(v for v in audit[stamp]['views'] if v['id'] == source and v['camera'] == hidden_name)
        error = np.linalg.norm(hidden.eye_to_window(hidden.world_to_eye(points[RI]))-hidden_view['points'][MI], axis=1)
        reference = np.array(case['reference_head_mm'])
        metric = dict(head_mm=points.tolist(), hidden_mean_px=float(error.mean()),
                      wrist_disagreement_mm=float(np.linalg.norm(points[5]-reference[5])),
                      tip_disagreement_mm=float(np.linalg.norm(points[:5]-reference[:5], axis=1).mean()),
                      accepted=bool(fit['accepted']), flow=info, fit=fit,
                      host_total_us=elapsed_us, blank_supported=bool(blank_info['supported']))
        case['methods']['flow'] = metric
        output.append(case)
        if index % 200 == 0:
            print(json.dumps(dict(done=index+1, total=len(cases), reasons=dict(reasons))), flush=True)
    with (a.output/'cases.jsonl').open('x') as f:
        for row in output:
            f.write(json.dumps(row, allow_nan=False)+'\n')
    metrics = [r['methods']['flow'] for r in output]
    delta = np.array([r['methods']['flow']['hidden_mean_px']-r['methods']['hold']['hidden_mean_px'] for r in output])
    subsets = {}
    for label, selected in [('all', output), ('admitted', [r for r in output if r['methods']['flow']['accepted']])]:
        subsets[label] = dict(count=len(selected), methods={
            mode: {key: stats([r['methods'][mode][key] for r in selected])
                   for key in ('hidden_mean_px', 'wrist_disagreement_mm', 'tip_disagreement_mm')}
            for mode in ('hold', 'velocity', 'plane', 'flow')})
    source = a.output/'source'
    source.mkdir()
    files = [Path(__file__), BASE/'probe_monocular_bridge.py', BASE.parent/'exploratory/palm_flow.py',
             BASE.parent/'exploratory/frame_data.py']
    for path in files:
        shutil.copy2(path, source/path.name)
    report = dict(scope=__doc__, cases=len(output), reasons=dict(reasons), subsets=subsets,
                  hidden_delta_px=stats(delta), better_over_5px=int((delta < -5).sum()),
                  worse_over_5px=int((delta > 5).sum()), blank_supported=sum(m['blank_supported'] for m in metrics),
                  host_total_us=stats([m['host_total_us'] for m in metrics]),
                  versions=dict(python=sys.version, opencv=cv2.__version__, numpy=np.__version__),
                  flow_parameters=PARAMETERS, source_sha256={x.name: sha(x) for x in files},
                  input_sha256={str(x): sha(x) for x in (a.baseline/'cases.jsonl', a.session/'frame/sets.bin', a.session/'frame/live-views.txt')},
                  limitations=['Artificial loss with both hands actually detected; not real occlusion recovery.',
                               'Unchanged flow can follow background. No presence, identity, finger or lifetime inference.',
                               'Whole previous pose moves rigidly at fixed depth; articulation/depth changes remain wrong.',
                               'Withheld detector observations and stereo reference are not physical ground truth.',
                               'Host CPU timings exclude camera/neural/render; no Frame performance or wearer validation.'])
    (a.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
