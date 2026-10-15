"""Summarize numerical input traces without inferring physical ground truth."""
import argparse
import json
from pathlib import Path
from statistics import median

def spread(values):
    values=sorted(values)
    if not values:return None
    return dict(n=len(values),minimum=values[0],p10=values[int((len(values)-1)*.1)],median=median(values),maximum=values[-1])

def analyze(path):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]
    result=dict(source=str(path),physical_contact_verified=False,sides=[])
    for side in (0,1):
        geometry=[r for r in rows if r.get('pinch_geometry') and r['side']==side]
        signals=[r for r in rows if r.get('hand_input') and r['side']==side]
        eligible=[r for r in geometry if r['gesture_packet_fresh']]
        result['sides'].append(dict(side=side,observations=len(signals),
            model_gap_m=spread([r['model_tip_gap_m'] for r in eligible]),
            published_gap_m=spread([r['published_tip_gap_m'] for r in eligible]),
            model_below_begin=sum(r['model_tip_gap_m']<r['begin_threshold_m'] for r in eligible),
            published_below_begin=sum(r['published_tip_gap_m']<r['begin_threshold_m'] for r in eligible),
            valid_gesture_observations=sum(r['gesture_valid'] for r in signals),
            raw_down_observations=sum(bool(r['gesture_flags']&2) for r in signals),
            gated_down_observations=sum(r['pinch_down'] for r in signals),
            armed_observations=sum(r['pinch_armed'] for r in signals),
            observed_begin_counts=sorted({r['pinch_begins'] for r in signals}),
            observed_end_counts=sorted({r['pinch_ends'] for r in signals})))
    result['summaries']=[r for r in rows if r.get('summary') or r.get('routing_summary') or r.get('gaze_geometry')]
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('trace',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    text=json.dumps(analyze(a.trace),indent=2)+'\n'
    if a.output:a.output.write_text(text)
    print(text,end='')
