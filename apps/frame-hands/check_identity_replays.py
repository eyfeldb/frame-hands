"""Paired offline identity experiment. Uses authorized private recordings only."""
import json
import os
from pathlib import Path
import subprocess
import argparse
from audit_summary import summarize

BASE = Path('/home/steamos/devkit-game/FrameHands')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('recordings', nargs='*', default=['20261003T190304Z', '20261003T182412Z', '20261003T180304Z'])
parser.add_argument('--report', default='identity-latch-replays.json')
args = parser.parse_args()
results = {}
for recording in args.recordings:
    cases = {}
    pose_rows = {}
    for name, enabled in [('baseline', False), ('identity', True)]:
        env = {k: v for k, v in os.environ.items() if not k.startswith('FRAME_HANDS_')}
        env.update(FRAME_HANDS_ROTATION_SEARCH='1', FRAME_HANDS_FULL_RATE='1',
                   FRAME_HANDS_PALM_THRESHOLD='0.2', FRAME_HANDS_PAIR_CONTINUITY='1',
                   FRAME_HANDS_OPEN_SIDE='1', FRAME_HANDS_IDENTITY_LATCH=str(int(enabled)))
        folder = BASE/'private'/recording
        audit = folder/f'{name}-identity-audit.txt'
        poses = folder/f'{name}-identity-poses.txt'
        env['FRAME_HANDS_AUDIT'] = str(audit)
        with (folder/f'{name}-identity-replay.log').open('w') as log:
            subprocess.run([str(BASE/'upstream/build/ft-handreplay'), str(folder), '--threads', '3',
                            '--models', str(BASE/'upstream/models/ncnn'), '--slow', '0.001',
                            '--poses', str(poses)], env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=90)
        cases[name] = summarize(audit)
        pose_rows[name] = [line.split() for line in poses.read_text().splitlines() if line.strip()]
        print(recording, name, json.dumps(cases[name]), flush=True)
    a, b = pose_rows['baseline'], pose_rows['identity']
    assert len(a) == len(b)
    # Every measured/filtered coordinate and track/view identity must remain exact.
    identical = all(x[:2] == y[:2] and x[3:] == y[3:] for x, y in zip(a, b))
    assert identical, 'Identity-only experiment changed geometry or track membership'
    cases['pose_rows'] = len(a)
    cases['geometry_and_track_membership_identical'] = identical
    results[recording] = cases
    output = BASE/'evidence'/args.report
    output.write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results, indent=2), flush=True)
