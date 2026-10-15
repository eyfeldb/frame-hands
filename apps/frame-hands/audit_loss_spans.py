"""Report unique optical availability spans; no inference of physical visibility."""
import json
from pathlib import Path
import sys
from review_recording import read_audit

path = Path(sys.argv[1])
observations = sorted(read_audit(path).values(), key=lambda item: item['raw_ns'])
start = observations[0]['raw_ns']
runs = []
for index, item in enumerate(observations):
    sides = sorted(set('R' if hand['right'] > .5 else 'L' for hand in item['hands'].values()))
    state = ''.join(sides) or 'none'
    t = (item['raw_ns']-start)/1e9
    if not runs or runs[-1]['labels'] != state:
        runs.append(dict(labels=state, first_frame=index+1, last_frame=index+1,
                         start_seconds=t, last_seconds=t, count=0))
    runs[-1]['last_seconds'] = t
    runs[-1]['last_frame'] = index+1
    runs[-1]['count'] += 1
print(json.dumps(dict(observations=len(observations), spans=runs,
                     physical_visibility_verified=False), indent=2))
