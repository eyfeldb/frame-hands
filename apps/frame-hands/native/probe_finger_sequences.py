"""Sequence audit of independent keypoints on pinned native failure windows.

Current-image hypotheses only. Causal persistence diagnostics never alter poses,
visibility, source assignment or runtime inputs. Uses exact native inference ROIs.
"""
import argparse
import json
import sys
import platform
from pathlib import Path
import numpy as np
import cv2
from audit_native_lengths import sha
from audit_crop_expansion import epi
from review_native_landmarks import MP_SEGMENTS

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'exploratory'))
from frame_data import Recording, read_audit, load_calibration
from mercury_probe import MercuryModels, MODEL_HASHES
from mercury_keypoints import StereoCrop, estimate
from probe_overlap_acquisition import affine

WINDOWS = [('phantom-finger', 6.0, 7.5), ('raised-finger', 37.4, 38.9),
           ('crop-transition', 39.6, 41.1)]


def extension(points):
    """Projected reach relative to palm length; not anatomical extension truth."""
    points = np.asarray(points, float)
    scale = np.linalg.norm(points[9] - points[0])
    if not np.isfinite(points).all() or scale < 1:
        return None
    return np.linalg.norm(points[[8, 12, 16, 20]] - points[[5, 9, 13, 17]], axis=1) / scale


def semantic_label(reach):
    if reach is None:
        return 0
    difference = reach[0] - reach[1]
    return 1 if difference > .25 else -1 if difference < -.25 else 0


def trace_crops(path):
    output = {}
    raw = None
    for line in path.read_text(encoding='utf-8').splitlines():
        fields = line.split()
        if fields[0] == 'set':
            raw = int(fields[1])
            if raw in output:
                raise ValueError('Duplicate crop set')
            output[raw] = []
        else:
            if fields[0] != 'crop' or len(fields) != 51 or raw is None:
                raise ValueError('Bad crop trace')
            numbers = np.array(fields[3:], float)
            if not np.isfinite(numbers).all():
                raise ValueError('Nonfinite crop trace')
            output[raw].append(dict(camera=fields[1], roi=numbers[:4], points=numbers[6:].reshape(21, 2)))
    return output


def panel(image, roi, points, title):
    matrix = affine(roi[:2], roi[2] * 1.3, roi[3], 224)
    inverse = cv2.invertAffineTransform(matrix)
    pixels = cv2.warpAffine(image, matrix, (224, 224), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    result = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    if points is not None:
        projected = np.c_[points, np.ones(21)] @ inverse.T
        for i, j in MP_SEGMENTS:
            cv2.line(result, tuple(np.rint(projected[i]).astype(int)), tuple(np.rint(projected[j]).astype(int)), (60, 230, 70), 1, cv2.LINE_AA)
        for label, i in zip('IMRL', (8, 12, 16, 20)):
            cv2.putText(result, label, tuple(np.rint(projected[i]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1)
    cv2.putText(result, title, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, .38, (80, 210, 255), 1, cv2.LINE_AA)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('session', 'trace', 'models', 'runtime', 'upstream', 'reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime.resolve()))
    sys.path.insert(0, str(args.upstream))
    traced = json.loads((args.trace / 'result.json').read_text(encoding='utf-8'))
    if not traced['outputs_match'] or sha(args.trace / 'crops.txt') != traced['trace_sha256']:
        raise ValueError('Unverified native trace')
    if sha(args.session / 'frame/sets.bin') != '34eb755f1f5285cdf0c94c6d45250ccb3739fe2409abbb958485ea13ef40d3d4':
        raise ValueError('Only pinned older recording permitted')
    if sha(args.trace / 'views.txt') != 'ed20b6bae52a5293b67b321bef7c628096522c819d161b0fd7ea9a28ea9f522c':
        raise ValueError('Native views changed')
    reference = json.loads((args.reference / 'result.json').read_text(encoding='utf-8'))
    sparse = {(r['case']['raw_ns'], r['case']['source'], v['camera']): v
              for r in reference['rows'] for v in r['views']}
    recording = Recording(args.session / 'frame/sets.bin')
    cameras = load_calibration(args.session / 'frame/snapshot/device/persist')
    audits = read_audit(args.trace / 'views.txt')
    crops = trace_crops(args.trace / 'crops.txt')
    model = MercuryModels(args.models)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    frame_counts = {}
    missing_pairs = []
    repeats = 0
    images = {}
    for window, start, end in WINDOWS:
        states = {}
        frame_counts[window] = 0
        window_rows = []
        for seconds, raw, frames in recording.frames(start, end):
            frame_counts[window] += 1
            audit = audits[raw]
            current = {}
            for source, hand in audit['hands'].items():
                # Reuse current tracker side only, not future fitted side or manual truth.
                side = int(hand['right'] >= .5)
                views = [v for name in ('slam_left', 'slam_right') for v in audit['views']
                         if v['id'] == source and v['camera'] == name]
                if len(views) != 2 or [v['camera'] for v in views] != ['slam_left', 'slam_right']:
                    missing_pairs.append(dict(window=window,raw_ns=raw,source=source,side=side,views=len(views)))
                    continue
                observations = []
                for view in views:
                    name = view['camera']
                    matches = [c for c in crops[raw] if c['camera'] == name and np.max(abs(c['points'] - view['points'])) < 1e-6]
                    if len(matches) != 1:
                        raise ValueError('Native crop cannot be uniquely joined')
                    roi = matches[0]['roi']
                    try:
                        crop = StereoCrop(cameras[name], roi[:2], roi[2], side)
                        result, _ = estimate(model, crop, frames[name][0])
                    except ValueError as error:
                        if str(error) not in ('Invalid fisheye inverse', 'Crop crosses camera forward hemisphere', 'Invalid stereographic radius'):
                            raise
                        result = dict(active=False, reason='Crop geometry rejected: ' + str(error))
                    old = sparse.get((raw, source, name))
                    if old is not None:
                        if result != old['result']:
                            raise ValueError('Sparse control changed (including side hypothesis)')
                        repeats += 1
                    observations.append(dict(camera=name, roi=roi.tolist(), native=view['points'].tolist(),
                                             native_presence=view['presence'], mercury=result))
                if not all('points_raw' in o['mercury'] for o in observations):
                    window_rows.append(dict(window=window, raw_ns=raw, seconds=seconds, source=source,
                                            side=side, usable=False, views=observations))
                    continue
                per_model = []
                key = (source, side)
                previous = states.get(key)
                for index, name in enumerate(('native', 'mercury')):
                    points = [np.array(o['native'] if name == 'native' else o['mercury']['points_raw']) for o in observations]
                    reaches = [extension(p) for p in points]
                    labels = [semantic_label(r) for r in reaches]
                    agreed = labels[0] if labels[0] == labels[1] else 0
                    # Count immediate predecessors only; no carry over absent hands, side changes or gaps.
                    old = previous['metrics'][index] if previous and 0 < raw - previous['raw_ns'] <= 100000000 else None
                    streak = old['streak'] + 1 if agreed and old and old['agreed'] == agreed else int(bool(agreed))
                    delta = None
                    if old and all(r is not None for r in reaches) and all(r is not None for r in old['reach']):
                        delta = float(np.max(np.abs(np.array(reaches) - old['reach'])))
                    error = epi(cameras, points)
                    per_model.append(dict(name=name, reach=[None if r is None else r.tolist() for r in reaches],
                                          labels=labels, agreed=agreed, streak=streak, persistent=agreed if streak >= 3 else 0,
                                          max_reach_step=delta, index_middle_epipolar_rms_degrees=float(np.sqrt(np.mean(error[5:13]**2)))))
                row = dict(window=window, raw_ns=raw, seconds=seconds, source=source, side=side,
                           usable=True, views=observations, metrics=per_model)
                current[key] = row
                window_rows.append(row)
                # Retain readable full comparisons at all frames for later selected review.
                canvas = np.zeros((480, 672, 3), np.uint8)
                cv2.putText(canvas, f'{window} {seconds:.3f}s side{side} source{source}', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1)
                for camera_index, o in enumerate(observations):
                    image = frames[o['camera']][0]
                    for column, (p, title) in enumerate(((None, o['camera']), (np.array(o['native']), 'native'), (np.array(o['mercury']['points_raw']), 'Mercury'))):
                        canvas[30+camera_index*224:254+camera_index*224, column*224:(column+1)*224] = panel(image, o['roi'], p, title)
                images[(raw, source)] = canvas
            states = current
        rows.extend(window_rows)
        print(window, len(window_rows), 'paired observations', flush=True)
    if repeats != len(sparse):
        raise ValueError(f'Incomplete sparse controls: {repeats} of {len(sparse)}')
    summaries = []
    selected = []
    for window, _, _ in WINDOWS:
        group = [r for r in rows if r['window'] == window and r['usable']]
        for side in (0, 1):
            seq = [r for r in group if r['side'] == side]
            if not seq:
                continue
            metrics = []
            for k in (0, 1):
                values = [r['metrics'][k] for r in seq]
                steps = [v['max_reach_step'] for v in values if v['max_reach_step'] is not None]
                metrics.append(dict(name=('native', 'mercury')[k], cross_view_label_conflicts=sum(v['labels'][0] * v['labels'][1] < 0 for v in values),
                                    persistent_index=sum(v['persistent'] == 1 for v in values), persistent_middle=sum(v['persistent'] == -1 for v in values),
                                    reach_step_p95=None if not steps else float(np.percentile(steps, 95)),
                                    max_reach_step=None if not steps else float(max(steps))))
            summaries.append(dict(window=window, side=side, pairs=len(seq), metrics=metrics))
            # Fixed ten evenly spaced frame reviews; no post-hoc cherry-picking by improvement.
            indices=np.unique(np.r_[np.rint(np.linspace(0, len(seq)-1, 10)).astype(int),[i for i,r in enumerate(seq) if (r['raw_ns'],r['source'],'slam_left') in sparse]]).astype(int)
            for i in indices:
                row=seq[i]; filename=f'{window}-side{side}-{i:03d}.png'
                if not cv2.imwrite(str(args.output/filename), images[(row['raw_ns'], row['source'])]):
                    raise IOError('Image write failed')
                selected.append(dict(file=filename, sha256=sha(args.output/filename), raw_ns=row['raw_ns'], source=row['source']))
            video=args.output/f'{window}-side{side}.mp4'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(672,480))
            if not writer.isOpened():
                raise IOError('Video encoder unavailable')
            for row in seq:writer.write(images[(row['raw_ns'],row['source'])])
            writer.release()
    (args.output/'rows.json').write_text(json.dumps(rows)+'\n',encoding='utf-8')
    report=dict(scope=__doc__,windows=WINDOWS,frames=frame_counts,missing_original_pairs=missing_pairs,low_presence_pairs=sum(r['usable'] and any(not v['mercury']['active'] for v in r['views']) for r in rows),summaries=summaries,sparse_controls_exact=repeats,selected=selected,
                rows_sha256=sha(args.output/'rows.json'),rejected_pairs=sum(not r['usable'] for r in rows),rejections=[dict(raw_ns=r['raw_ns'],source=r['source'],reasons=[o['mercury'].get('reason') for o in r['views'] if 'points_raw' not in o['mercury']]) for r in rows if not r['usable']],trace_result_sha256=sha(args.trace/'result.json'),inputs_sha256={str(p):sha(p) for p in (args.trace/'views.txt',args.session/'frame/snapshot/device/persist/device_config.json',args.session/'frame/snapshot/device/persist/xrservice.json')},recording_sha256='34eb755f1f5285cdf0c94c6d45250ccb3739fe2409abbb958485ea13ef40d3d4',
                sparse_reference_sha256=sha(args.reference/'result.json'),source_sha256=sha(Path(__file__)),models=MODEL_HASHES,
                runtime=dict(python=platform.python_version(),onnxruntime=model.version,numpy=np.__version__,opencv=cv2.__version__,system=platform.system()),
                limitations=['Selected failure windows, no anatomical labels or accuracy ground truth.',
                             'Projected reach ratio depends on viewpoint/foreshortening. Index/middle label means greater projected reach, not detected anatomical truth.',
                             'All finite keypoints are analyzed even below the model presence threshold; low-presence pairs are reported separately.',
                             'Three-frame persistence is an offline diagnostic, not a trusted pose selector or measured latency.',
                             'Current original tracker crops/side/availability are reused; independent acquisition and feedback are not tested.',
                             'Videos contain only paired observations; skipped/missing frames are omitted, so use raw timestamps for timing.',
                             'No Frame execution, live camera, input publication or accepted binary changes.'])
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summaries,indent=2),flush=True)


if __name__ == '__main__':
    main()
