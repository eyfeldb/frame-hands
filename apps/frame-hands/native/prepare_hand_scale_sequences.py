"""Freeze continuous windows for the exact-crop Lite/Full model diagnostic.
Existing tracker crops and source IDs remain fixed: this is not a tracker replay.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path
import numpy as np
from run_tracker_rig_check import sha
from probe_finger_sequences import trace_crops
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'exploratory'))
from frame_data import read_audit, Recording

WINDOWS = [('open', 6.0, 7.5), ('reacquisition', 25.0, 26.0),
           ('raised', 34.7, 35.7), ('label-conflict', 37.4, 38.9),
           ('crossing', 43.8, 45.3), ('back-hand', 48.4, 49.4)]
RECORDING_SHA = '34eb755f1f5285cdf0c94c6d45250ccb3739fe2409abbb958485ea13ef40d3d4'
VIEWS_SHA = 'ed20b6bae52a5293b67b321bef7c628096522c819d161b0fd7ea9a28ea9f522c'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('trace', 'session', 'reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    trace = json.loads((args.trace / 'result.json').read_text())
    if not trace['outputs_match'] or sha(args.trace / 'crops.txt') != trace['trace_sha256']:
        raise ValueError('Unverified native crop trace')
    if sha(args.session / 'frame/sets.bin') != RECORDING_SHA or sha(args.trace / 'views.txt') != VIEWS_SHA:
        raise ValueError('Recording or view audit changed')
    models = json.loads((args.reference / 'models.json').read_text())
    for model in models:
        if sha(args.reference / 'models' / (model['name'] + '.tflite')) != model['sha256']:
            raise ValueError('Model changed')
    recording = Recording(args.session / 'frame/sets.bin')
    audits = read_audit(args.trace / 'views.txt')
    crops = trace_crops(args.trace / 'crops.txt')
    origin = recording.index[0][0]
    requests, sets = [], []
    for window, start, end in WINDOWS:
        for raw, _, _ in recording.index:
            seconds = (raw - origin) / 1e9
            if not start <= seconds < end:
                continue
            audit = audits[raw]
            sets.append(dict(window=window, raw_ns=raw, sources=len(audit['hands']), views=len(audit['views'])))
            for source in audit['hands']:
                for camera in ('slam_left', 'slam_right'):
                    views = [v for v in audit['views'] if v['id'] == source and v['camera'] == camera]
                    if len(views) > 1:
                        raise ValueError('Duplicate source view')
                    for view in views:
                        matches = [c for c in crops[raw] if c['camera'] == camera and np.max(abs(c['points'] - view['points'])) < 1e-6]
                        if len(matches) != 1:
                            raise ValueError('Ambiguous crop join')
                        requests.append(dict(index=len(requests), case=f'{window}-{raw}-{source}',
                            window=window, raw_ns=raw, seconds=seconds, source=source, camera=camera,
                            roi=matches[0]['roi'].tolist(), native_points=view['points'].tolist(),
                            native_presence=view['presence'], native_right=view['right']))
    if not requests or len(requests) > 10000:
        raise ValueError('Unexpected request count')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'inputs').mkdir()
    shutil.copytree(args.reference / 'models', args.output / 'models')
    (args.output / 'models.json').write_text(json.dumps(models, indent=2) + '\n')
    (args.output / 'requests.json').write_text(json.dumps(dict(requests=requests, windows=WINDOWS, sets=sets,
        recording_sha256=RECORDING_SHA, views_sha256=VIEWS_SHA, trace_sha256=trace['trace_sha256'],
        preparer_sha256=sha(Path(__file__))), indent=2) + '\n')
    (args.output / 'requests.txt').write_text(''.join(f"{r['raw_ns']} {r['camera']} " + ' '.join(format(v, '.17g') for v in r['roi']) + '\n' for r in requests))
    print(json.dumps(dict(sets=len(sets), views=len(requests), windows=WINDOWS)))


if __name__ == '__main__':
    main()
